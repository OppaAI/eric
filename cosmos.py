"""
ERIC — Cosmos Reason 2 Interface
Vision + physical reasoning via vLLM
"""

import json
import base64
import logging
import requests

from config import (
    VLLM_URL, COSMOS_MODEL,
    CAMERA_WEBCAM, CAMERA_PANTILT,
    CAMERA_WIDTH, CAMERA_HEIGHT
)

log = logging.getLogger("eric.cosmos")

_BASE_SYSTEM_PROMPT = """
You are ERIC — Edge Robotics Innovation by Cosmos.
You are a search and rescue tracked ground robot.
The camera view is YOUR view — egocentric, first person.

Your hardware:
- Tracked robot chassis (~30cm wide), built for outdoor terrain
- NVIDIA Jetson Orin Nano Super 8GB
- Cosmos Reason 2 (2B W4A16) via vLLM — your vision and reasoning
- Two cameras: pan-tilt and webcam
- Total cost: ~$750 CAD, built by one person in Vancouver BC Canada
- Fully local edge AI — no cloud, no server

Your rules:
- You have NO arms — never engage in combat
- Avoid all obstacles and persons in your path
- Talk to people and robots to gather mission information
- If someone doesn't know anything, thank them and move on
- Reason carefully about the physical world from YOUR point of view

Terrain reasoning (egocentric — what is directly ahead of YOU):
- Pebbles/rough ground → slow down
- Smooth pavement → normal speed
- Obstacle in YOUR path → navigate around
- Clear path → proceed forward

Keep spoken responses under 3 sentences unless introducing yourself.
Speaking via TTS — be natural, not robotic.
"""

_system_prompt    = _BASE_SYSTEM_PROMPT
_mission_briefing = ""


def set_mission_briefing(briefing: str):
    global _system_prompt, _mission_briefing
    _mission_briefing = briefing.strip()
    _system_prompt = (
        _BASE_SYSTEM_PROMPT
        + "\n\n═══ MISSION BRIEFING ═══\n"
        + _mission_briefing
        + "\n═══════════════════════\n"
    )
    log.info(f"📋 Mission briefing loaded: {_mission_briefing[:80]}...")


def get_mission_briefing() -> str:
    return _mission_briefing


# ─── Camera ───────────────────────────────────────────────────────────────────
# Persistent capture objects — one per device index.
# Avoids repeated open/close overhead and keeps buffer state tuned.
import cv2 as _cv2
import time as _time
import threading as _threading

# ── Persistent background camera readers ─────────────────────────────────────
# The root cause of V4L2 select() timeouts on Jetson is that when Python is
# busy (Cosmos inference, TTS, mission logic) nobody reads from the camera.
# After ~1 second of no reads the kernel buffer fills and V4L2 stalls.
# Fix: one daemon thread per camera that reads continuously and stores the
# latest frame in a 1-slot queue. capture_frame() just grabs from the queue.

class _CameraReader:
    """
    Background thread that continuously reads from a V4L2 camera.

    Design: _run() calls cap.read() in a tight loop with NO blocking on the
    consumer side. The latest frame is stored in self._latest (protected by a
    lock) and a threading.Event is set each time a new frame arrives.
    get_frame() just waits on the event — it never touches cap.read() and
    never interferes with the reader loop.

    This guarantees the kernel V4L2 buffer is drained as fast as possible,
    preventing the select() timeout stalls that occur when the buffer fills up
    during long Cosmos inference calls.
    """

    RECONNECT_DELAY = 2.0   # seconds to wait before reconnecting after failure
    READ_TIMEOUT    = 3.0   # seconds get_frame() waits for a fresh frame

    def __init__(self, device: int, width: int = 640, height: int = 480):
        self.device   = device
        self.width    = width
        self.height   = height
        self._latest  = None               # most recent frame (numpy array)
        self._lock    = _threading.Lock()  # protects _latest
        self._event   = _threading.Event() # set whenever a new frame is stored
        self._stop    = _threading.Event()
        self._thread  = _threading.Thread(target=self._run, daemon=True, name=f"cam-{device}")
        self._thread.start()

    def _open(self):
        """Open camera with low-latency settings. Try GStreamer first, then V4L2."""
        # ── GStreamer path ────────────────────────────────────────────────────
        try:
            pipeline = (
                f"v4l2src device=/dev/video{self.device} io-mode=2 ! "
                "video/x-raw, width=640, height=480, framerate=30/1 ! "
                "videoconvert n-threads=2 ! "
                "video/x-raw, format=BGR ! "
                "appsink drop=1 max-buffers=1 sync=false"
            )
            cap = _cv2.VideoCapture(pipeline, _cv2.CAP_GSTREAMER)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    log.info(f"📷 Camera {self.device}: GStreamer reader started")
                    return cap
                cap.release()
        except Exception:
            pass

        # ── V4L2 path — try MJPG, fall back to YUYV ──────────────────────────
        # Set BUFFERSIZE=2 during open so the kernel has room to negotiate
        # format before we starve it. Reduce to 1 after warm-up.
        for fourcc_str in ("MJPG", "YUYV"):
            cap = _cv2.VideoCapture(self.device, _cv2.CAP_V4L2)
            if not cap.isOpened():
                log.warning(f"Camera {self.device}: V4L2 open failed")
                cap.release()
                continue

            cap.set(_cv2.CAP_PROP_FOURCC,      _cv2.VideoWriter_fourcc(*fourcc_str))
            cap.set(_cv2.CAP_PROP_FRAME_WIDTH,  self.width)
            cap.set(_cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(_cv2.CAP_PROP_FPS,          30)
            cap.set(_cv2.CAP_PROP_BUFFERSIZE,   2)   # warm-up buffer

            # Drain a few frames to let the kernel settle before going low-latency
            ok = False
            for _ in range(5):
                ret, _ = cap.read()
                if ret:
                    ok = True
                    break
                _time.sleep(0.05)

            if ok:
                cap.set(_cv2.CAP_PROP_BUFFERSIZE, 1)  # low-latency once warm
                log.info(f"📷 Camera {self.device}: V4L2 reader started ({fourcc_str})")
                return cap

            log.warning(f"Camera {self.device}: {fourcc_str} gave no frames — trying next format")
            cap.release()

        # ── Last resort: autodetect (let OpenCV negotiate everything) ─────────
        cap = _cv2.VideoCapture(self.device)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                log.info(f"📷 Camera {self.device}: auto-detect reader started")
                return cap
            cap.release()

        raise RuntimeError(f"Camera {self.device}: all open strategies failed")

    def _run(self):
        """Tight read loop — drains V4L2 buffer as fast as possible."""
        cap   = None
        fails = 0
        while not self._stop.is_set():
            if cap is None or not cap.isOpened():
                try:
                    if cap:
                        cap.release()
                    cap = self._open()
                    fails = 0
                except RuntimeError as e:
                    log.error(f"Camera {self.device}: {e} — retrying in {self.RECONNECT_DELAY * 3}s")
                    _time.sleep(self.RECONNECT_DELAY * 3)
                    continue
                except Exception as e:
                    log.error(f"Camera {self.device}: open failed ({e}) — retrying in {self.RECONNECT_DELAY}s")
                    _time.sleep(self.RECONNECT_DELAY)
                    continue

            ret, frame = cap.read()
            if not ret:
                fails += 1
                if fails >= 5:
                    log.warning(f"Camera {self.device}: {fails} consecutive read failures — reconnecting")
                    cap.release()
                    cap   = None
                    fails = 0
                    _time.sleep(self.RECONNECT_DELAY)
                continue
            fails = 0

            # Store latest frame and signal any waiting get_frame() call.
            # This never blocks — the consumer never touches cap.read().
            with self._lock:
                self._latest = frame
            self._event.set()   # wake get_frame() if it's waiting

        if cap:
            cap.release()

    def get_frame(self) -> "_cv2.Mat | None":
        """
        Return the most recent frame, waiting up to READ_TIMEOUT seconds for one.
        Clears the event so the next call waits for a genuinely new frame.
        """
        got = self._event.wait(timeout=self.READ_TIMEOUT)
        if not got:
            log.warning(f"Camera {self.device}: no frame available after {self.READ_TIMEOUT}s")
            return None
        self._event.clear()
        with self._lock:
            return self._latest

    def stop(self):
        self._stop.set()


# One reader per camera device, created on first use
_readers: dict[int, _CameraReader] = {}


def _get_reader(device: int) -> _CameraReader:
    if device not in _readers:
        reader = _CameraReader(device)
        _readers[device] = reader
        # Wait until the background thread delivers its first real frame
        # (up to 10s — covers the warm-up reads inside _open()).
        # This prevents the "no frame available" warning on the very first call.
        ready = reader._event.wait(timeout=10.0)
        if not ready:
            log.warning(f"Camera {device}: no frame within 10s of startup — proceeding anyway")
    return _readers[device]


def capture_frame(device: int = CAMERA_WEBCAM,
                  width: int = CAMERA_WIDTH,
                  height: int = CAMERA_HEIGHT) -> str | None:
    """Capture the latest frame from the persistent background reader."""
    try:
        frame = _get_reader(device).get_frame()
        if frame is None:
            return None

        # Resize if needed
        if width != CAMERA_WIDTH or height != CAMERA_HEIGHT:
            frame = _cv2.resize(frame, (width, height))

        # Cosmos pixel budget: 256k max
        h, w = frame.shape[:2]
        max_pixels = 256000
        if w * h > max_pixels:
            scale = (max_pixels / (w * h)) ** 0.5
            frame = _cv2.resize(frame, (int(w * scale), int(h * scale)))

        _, buf = _cv2.imencode(".jpg", frame, [_cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode("utf-8")
    except Exception as e:
        log.error(f"Camera {device} capture error: {e}")
        return None


def capture_frames_video(device: int = CAMERA_WEBCAM,
                         duration: float = 10.0,
                         fps_sample: float = 1.0) -> list[str]:
    """
    Capture video clip, return list of sampled base64 frames.
    Uses persistent cap with MJPEG + buffer=1 for low latency.
    duration: seconds to record
    fps_sample: frames per second to sample
    """
    try:
        reader = _get_reader(device)
        frames = []
        start  = _time.time()
        last   = 0.0
        interval = 1.0 / fps_sample

        while _time.time() - start < duration:
            frame = reader.get_frame()
            if frame is None:
                break
            elapsed = _time.time() - start
            if elapsed - last >= interval:
                frame = _cv2.resize(frame, (320, 240))
                _, buf = _cv2.imencode(".jpg", frame, [_cv2.IMWRITE_JPEG_QUALITY, 80])
                frames.append(base64.b64encode(buf).decode("utf-8"))
                last = elapsed

        log.info(f"📹 Captured {len(frames)} frames over {duration}s")
        return frames
    except Exception as e:
        log.error(f"Video capture error: {e}")
        return []


def capture_frame_raw(device: int = CAMERA_WEBCAM):
    """Capture raw RGB frame for Gradio display — from persistent background reader."""
    try:
        frame = _get_reader(device).get_frame()
        if frame is None:
            return None
        if device == CAMERA_WEBCAM:
            frame = _cv2.rotate(frame, _cv2.ROTATE_90_COUNTERCLOCKWISE)
        return _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
    except Exception:
        return None


# ─── Cosmos API ───────────────────────────────────────────────────────────────

def ask_cosmos(prompt: str, image_b64: str = None,
               frames: list[str] = None,
               max_tokens: int = 300, stream: bool = False):
    """
    Query Cosmos Reason 2 via vLLM.
    image_b64: single image
    frames: list of images for video reasoning
    stream=True → returns generator
    All calls are logged via logger.log_ai().
    """
    content = []

    # Multi-frame video
    if frames:
        for f in frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{f}"}
            })
    elif image_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
        })

    content.append({"type": "text", "text": prompt})

    payload = {
        "model":              COSMOS_MODEL,
        "messages": [
            {"role": "system", "content": _system_prompt},
            {"role": "user",   "content": content}
        ],
        "max_tokens":         max_tokens,
        "temperature":        0.7,
        "repetition_penalty": 1.15,
        "stream":             stream
    }

    try:
        if stream:
            return _stream_cosmos(payload)
        r = requests.post(VLLM_URL, json=payload, timeout=90)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
        log.info(f"🧠 Cosmos: {text[:120]}")
        # ── Log to eric_logger ───────────────────────────────────────────────
        try:
            from logger import log_ai as _log_ai
            label = "COSMOS_IMAGE" if (image_b64 or frames) else "COSMOS_TEXT"
            _log_ai(prompt[-400:], text, label=label)
        except Exception:
            pass
        return text
    except requests.exceptions.ConnectionError:
        msg = "Cannot connect to Cosmos. Is vLLM running? (bash launch/cosmos.sh)"
        log.error(msg)
        return msg
    except Exception as e:
        log.error(f"Cosmos error: {e}")
        try:
            from logger import log_exception as _log_exc
            _log_exc("ask_cosmos", e)
        except Exception:
            pass
        return f"Cosmos error: {e}"


def _stream_cosmos(payload: dict):
    try:
        with requests.post(VLLM_URL, json=payload, stream=True, timeout=90) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    line = line[6:]
                if line == "[DONE]":
                    break
                try:
                    delta = json.loads(line)["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
                except Exception:
                    continue
    except Exception as e:
        log.error(f"Stream error: {e}")
        yield ""


# ─── Scene Scan ───────────────────────────────────────────────────────────────

SCAN_PROMPT = """
The camera view is MY view as a ground robot — egocentric, first person.
I am on a search and rescue mission.

Analyze the scene. Think step by step:
1) What objects or people do I see and where are they relative to me?
2) Is anything in MY direct path?
3) What is the terrain like?
4) What is the safest next action?

Respond ONLY with valid JSON — no markdown, no extra text:
{
  "object": "person|robot|obstacle|vehicle|clear|unknown",
  "object_name": "specific name if identifiable, else null",
  "terrain": "pebbles|pavement|grass|clear",
  "distance": "close|medium|far",
  "in_my_path": true or false,
  "action": "stop|forward|slow|navigate_around",
  "speak": "what Eric says out loud right now, or null",
  "physical_reasoning": "1 sentence: what I see and why I chose this action"
}
"""

VIDEO_SCAN_PROMPT = """
These are frames from a 10-second video clip from my robot camera — egocentric view.
Analyze what happened over time:
1) What objects or people appeared? Did anything move toward or away from me?
2) What changed between the first and last frame?
3) What is the terrain and are there obstacles in my path?
4) What should I do next?

Respond ONLY with valid JSON:
{
  "object": "person|robot|obstacle|vehicle|clear|unknown",
  "object_name": "specific name if identifiable, else null",
  "terrain": "pebbles|pavement|grass|clear",
  "movement": "approaching|retreating|stationary|none",
  "in_my_path": true or false,
  "action": "stop|forward|slow|navigate_around",
  "speak": "what Eric says out loud right now, or null",
  "physical_reasoning": "1 sentence summarizing what changed and why I chose this action"
}
"""


def scan_scene(device: int = CAMERA_WEBCAM, use_video: bool = False,
               video_duration: float = 5.0) -> dict:
    """
    Scan scene with single frame or short video clip.
    use_video=True captures video_duration seconds at 1fps.
    """
    fallback = {"object": "unknown", "action": "forward",
                "terrain": "clear", "in_my_path": False,
                "speak": None, "object_name": None}

    if use_video:
        frames = capture_frames_video(device, duration=video_duration, fps_sample=1.0)
        if not frames:
            return fallback
        response = ask_cosmos(VIDEO_SCAN_PROMPT, frames=frames, max_tokens=200)
    else:
        # Try 640x480 first, auto-resizes if over pixel budget
        image = capture_frame(device, width=640, height=480)
        if not image:
            return fallback
        response = ask_cosmos(SCAN_PROMPT, image_b64=image, max_tokens=200)

    try:
        clean = response.replace("```json", "").replace("```", "").strip()
        start = clean.find("{")
        end   = clean.rfind("}") + 1
        if start >= 0 and end > start:
            clean = clean[start:end]
        return json.loads(clean)
    except Exception:
        log.warning(f"Could not parse scene JSON: {response[:200]}")
        return fallback


# ─── Digital Zoom Crop (wide-angle pan-tilt fix) ──────────────────────────────

def crop_zoom_region(frame_b64: str, x_ratio: float, y_ratio: float,
                     crop_frac: float = 0.4) -> str | None:
    """
    Crop and zoom into a region of interest in a base64 JPEG frame.
    Used to improve Cosmos identification of small objects seen by wide-angle cam.

    x_ratio: 0.0 (left) to 1.0 (right) — center of region of interest
    y_ratio: 0.0 (top)  to 1.0 (bottom) — center of region of interest
    crop_frac: fraction of frame to crop (0.4 = 40% of frame centred on target)

    Returns base64 JPEG of cropped+zoomed region.
    """
    try:
        import cv2
        import numpy as np

        data  = base64.b64decode(frame_b64)
        arr   = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return frame_b64  # return original if decode fails

        h, w = frame.shape[:2]

        # Compute crop box centred on x_ratio, y_ratio
        cw = int(w * crop_frac)
        ch = int(h * crop_frac)
        cx = int(w * x_ratio)
        cy = int(h * y_ratio)

        x1 = max(0, cx - cw // 2)
        y1 = max(0, cy - ch // 2)
        x2 = min(w, x1 + cw)
        y2 = min(h, y1 + ch)

        cropped = frame[y1:y2, x1:x2]
        # Resize back to original size — effective zoom
        zoomed  = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        _, buf = cv2.imencode(".jpg", zoomed, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return base64.b64encode(buf).decode("utf-8")

    except Exception as e:
        log.warning(f"crop_zoom_region error: {e}")
        return frame_b64  # return original on error


def capture_zoomed(x_ratio: float = 0.5, y_ratio: float = 0.5,
                   crop_frac: float = 0.4,
                   device: int = CAMERA_PANTILT) -> str | None:
    """
    Capture frame from pan-tilt camera and zoom into region of interest.
    Use this when Cosmos detects something in the pan-tilt wide-angle view
    but needs a closer look to identify it.

    Example:
        # After scan detects something at right side of frame
        zoomed = capture_zoomed(x_ratio=0.8, y_ratio=0.5, crop_frac=0.35)
        result = ask_cosmos("What is this object?", image_b64=zoomed)
    """
    frame = capture_frame(device, CAMERA_WIDTH, CAMERA_HEIGHT)
    if not frame:
        return None
    return crop_zoom_region(frame, x_ratio, y_ratio, crop_frac)


def multi_zoom_scan(device: int = CAMERA_PANTILT) -> list[str]:
    """
    Capture one wide frame then 4 zoomed crops (left, center-left,
    center-right, right). Returns list of 5 base64 frames.
    Gives Cosmos both context (wide) and detail (zoomed) in one call.
    Useful for small object identification with wide-angle cameras.
    """
    wide = capture_frame(device, CAMERA_WIDTH, CAMERA_HEIGHT)
    if not wide:
        return []

    frames = [wide]  # wide shot first for context

    # Zoom into 4 horizontal regions
    for x in [0.2, 0.4, 0.6, 0.8]:
        zoomed = crop_zoom_region(wide, x_ratio=x, y_ratio=0.5, crop_frac=0.45)
        if zoomed:
            frames.append(zoomed)

    log.info(f"📷 Multi-zoom scan: {len(frames)} frames (1 wide + 4 zoomed)")
    return frames
