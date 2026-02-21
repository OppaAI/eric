"""
ERIC — Cosmos Reason 2 Interface
Vision + physical reasoning via vLLM

Improvements:
- Dual camera scanning (webcam + pan-tilt)
- Video feed (10s at 640x480) for better detection
- Pan-tilt centering on detected objects/faces
- Autofocus control via v4l2
- Obstacle/wall detection prompt
- Mission complete detection
"""

import json
import base64
import logging
import subprocess
import time
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
- Two cameras: pan-tilt (looking around) and webcam (navigation)
- Total cost: ~$750 CAD, built by one person in Kelowna BC Canada
- Fully local edge AI — no cloud, no server

Your rules:
- You have NO arms — never engage in combat
- ALWAYS avoid walls, furniture, and obstacles — do not drive into them
- Avoid all obstacles and persons in your path
- Talk to people and robots to gather mission information
- If someone doesn't know anything, thank them and move on
- Reason carefully about the physical world from YOUR point of view

Terrain and obstacle reasoning (egocentric — what is directly ahead of YOU):
- Wall or large obstacle close ahead → STOP immediately, turn away
- Furniture, boxes, slippers, shoes → navigate around carefully
- Pebbles/rough ground → slow down
- Smooth pavement → normal speed
- Clear path → proceed forward

Keep spoken responses under 3 sentences unless introducing yourself.
Speaking via TTS — be natural, not robotic.
"""

_system_prompt    = _BASE_SYSTEM_PROMPT
_mission_briefing = ""

# Pan-tilt state (degrees from center)
_pan_angle  = 0
_tilt_angle = 0


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


# ─── Pan-Tilt Control ─────────────────────────────────────────────────────────

def pantilt(pan: int = 0, tilt: int = 0, speed: int = 50):
    """Send pan-tilt command to ESP32. pan/tilt in degrees from center."""
    global _pan_angle, _tilt_angle
    try:
        from motors import motors
        _pan_angle  = max(-90, min(90, pan))
        _tilt_angle = max(-45, min(45, tilt))
        motors._send_raw({"T": 133, "X": _pan_angle, "Y": _tilt_angle, "SPD": speed, "ACC": 10})
        log.info(f"🎥 Pan-tilt → X:{_pan_angle} Y:{_tilt_angle}")
    except Exception as e:
        log.error(f"Pan-tilt error: {e}")


def pantilt_center():
    """Return pan-tilt to center position."""
    pantilt(0, 0)


def pantilt_scan_left():
    pantilt(-45, 0)


def pantilt_scan_right():
    pantilt(45, 0)


def pantilt_center_on_object(frame_x_ratio: float, frame_y_ratio: float):
    """
    Center pan-tilt on detected object.
    frame_x_ratio: 0.0 (left) to 1.0 (right) — where object is in frame
    frame_y_ratio: 0.0 (top) to 1.0 (bottom)
    """
    # Convert frame position to pan/tilt offset
    pan_offset  = int((frame_x_ratio - 0.5) * 90)   # -45 to +45 degrees
    tilt_offset = int((frame_y_ratio - 0.5) * -45)  # -22 to +22 degrees
    new_pan  = max(-90, min(90, _pan_angle + pan_offset))
    new_tilt = max(-45, min(45, _tilt_angle + tilt_offset))
    pantilt(new_pan, new_tilt, speed=30)


# ─── Autofocus Control ────────────────────────────────────────────────────────

def autofocus_enable(device: int = CAMERA_WEBCAM):
    """Enable continuous autofocus on camera."""
    try:
        dev = f"/dev/video{device}"
        subprocess.run(
            ["v4l2-ctl", "-d", dev, "--set-ctrl=focus_automatic_continuous=1"],
            capture_output=True, timeout=2
        )
        log.info(f"🔍 Autofocus enabled on {dev}")
    except Exception as e:
        log.warning(f"Autofocus enable error: {e}")


def autofocus_trigger(device: int = CAMERA_WEBCAM):
    """
    Trigger autofocus and wait for it to settle.
    Disables continuous AF, triggers once, waits, re-enables.
    """
    try:
        dev = f"/dev/video{device}"
        subprocess.run(["v4l2-ctl", "-d", dev, "--set-ctrl=focus_automatic_continuous=0"], capture_output=True, timeout=2)
        time.sleep(0.3)
        subprocess.run(["v4l2-ctl", "-d", dev, "--set-ctrl=focus_automatic_continuous=1"], capture_output=True, timeout=2)
        time.sleep(1.5)  # Wait for autofocus to settle
        log.info(f"🔍 Autofocus triggered on {dev}")
    except Exception as e:
        log.warning(f"Autofocus trigger error: {e}")


# ─── Camera ───────────────────────────────────────────────────────────────────

# Persistent camera captures to avoid repeated open/close
_caps = {}

def _get_cap(device: int, width: int = 640, height: int = 480):
    """Get or create a persistent VideoCapture for a device."""
    import cv2
    if device not in _caps or not _caps[device].isOpened():
        cap = cv2.VideoCapture(device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer lag
        _caps[device] = cap
    return _caps[device]


def capture_frame(device: int = CAMERA_WEBCAM,
                  width: int = CAMERA_WIDTH,
                  height: int = CAMERA_HEIGHT) -> str | None:
    """Capture frame, return base64 JPEG."""
    try:
        import cv2
        cap = _get_cap(device, width, height)
        cap.grab()  # Flush stale buffer frame
        ret, frame = cap.read()
        if not ret:
            log.error(f"Camera {device}: frame capture failed")
            _caps.pop(device, None)
            return None

        # Rotate webcam if mounted sideways
        if device == CAMERA_WEBCAM:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Resize to fit Cosmos pixel budget (256k max)
        h, w = frame.shape[:2]
        max_pixels = 256000
        if w * h > max_pixels:
            scale = (max_pixels / (w * h)) ** 0.5
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode("utf-8")
    except Exception as e:
        log.error(f"Camera {device} error: {e}")
        _caps.pop(device, None)
        return None


def capture_frames_video(device: int = CAMERA_WEBCAM,
                         duration: float = 10.0,
                         fps_sample: float = 1.0) -> list[str]:
    """
    Capture video clip and return list of sampled base64 frames at 640x480.
    Each frame is kept at 640x480 but we limit total frames to stay under pixel budget.
    Max safe: 5 frames × ~50k pixels = ~250k total
    """
    try:
        import cv2
        cap = _get_cap(device, 640, 480)
        frames   = []
        start    = time.time()
        last     = 0.0
        interval = 1.0 / fps_sample

        while time.time() - start < duration:
            ret, frame = cap.read()
            if not ret:
                break
            elapsed = time.time() - start
            if elapsed - last >= interval:
                if device == CAMERA_WEBCAM:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                # Resize to 512x384 to keep pixel budget manageable across frames
                frame = cv2.resize(frame, (512, 384))
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frames.append(base64.b64encode(buf).decode("utf-8"))
                last = elapsed

        log.info(f"📹 Captured {len(frames)} frames over {duration:.1f}s from camera {device}")
        return frames
    except Exception as e:
        log.error(f"Video capture error: {e}")
        return []


def capture_frame_raw(device: int = CAMERA_WEBCAM):
    """Capture raw RGB frame for Gradio display."""
    try:
        import cv2
        cap = _get_cap(device, 640, 480)
        cap.grab()
        ret, frame = cap.read()
        if not ret:
            return None
        if device == CAMERA_WEBCAM:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
    """
    content = []

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
        return text
    except requests.exceptions.ConnectionError:
        msg = "Cannot connect to Cosmos. Is vLLM running?"
        log.error(msg)
        return msg
    except Exception as e:
        log.error(f"Cosmos error: {e}")
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


# ─── Scene Scan Prompts ───────────────────────────────────────────────────────

SCAN_PROMPT = """
The camera views are MY eyes as a ground robot — egocentric, first person.
I am on a search and rescue mission. I have TWO cameras: webcam (navigation) and pan-tilt (looking around).

Analyze BOTH views carefully. Think step by step:
1) What objects, people, or obstacles do I see in EITHER camera?
2) Is there a wall, furniture, or large obstacle CLOSE ahead in my navigation camera?
3) Are there small obstacles like shoes, slippers, or objects on the floor I might run over?
4) Is my mission target visible in either camera?
5) What is the terrain like?
6) What is the safest next action?

IMPORTANT: If a wall or large obstacle is close and filling the navigation camera frame → action MUST be "stop" or "navigate_around".

Respond ONLY with valid JSON — no markdown, no extra text:
{
  "object": "person|robot|obstacle|wall|vehicle|clear|unknown",
  "object_name": "specific name if identifiable, else null",
  "terrain": "pebbles|pavement|grass|clear",
  "distance": "close|medium|far",
  "in_my_path": true or false,
  "wall_ahead": true or false,
  "small_obstacle": true or false,
  "camera_with_target": "webcam|pantilt|none",
  "action": "stop|forward|slow|navigate_around",
  "speak": "what Eric says out loud right now, or null",
  "physical_reasoning": "1 sentence: what I see and why I chose this action",
  "mission_complete": false
}
"""

VIDEO_SCAN_PROMPT = """
These are frames from a 10-second video clip from my robot cameras — egocentric view.
I have TWO cameras providing these frames: webcam (navigation/floor) and pan-tilt (scanning environment).

Analyze what happened over time across ALL frames:
1) What objects or people appeared in ANY frame?
2) Is there a wall, furniture, or obstacle close ahead?
3) Are there small obstacles on the floor (shoes, slippers, cables)?
4) Did my mission target appear in any frame?
5) What changed between first and last frame?
6) What should I do next?

IMPORTANT: If wall or large obstacle is close → action MUST be "stop" or "navigate_around".

Respond ONLY with valid JSON:
{
  "object": "person|robot|obstacle|wall|vehicle|clear|unknown",
  "object_name": "specific name if identifiable, else null",
  "terrain": "pebbles|pavement|grass|clear",
  "movement": "approaching|retreating|stationary|none",
  "distance": "close|medium|far",
  "in_my_path": true or false,
  "wall_ahead": true or false,
  "small_obstacle": true or false,
  "camera_with_target": "webcam|pantilt|none",
  "action": "stop|forward|slow|navigate_around",
  "speak": "what Eric says out loud right now, or null",
  "physical_reasoning": "1 sentence summarizing what changed and why I chose this action",
  "mission_complete": false
}
"""

FACE_CENTER_PROMPT = """
I am looking at a person directly in front of me.
The pan-tilt camera is showing their face/body.

Estimate where the person's face is in the frame:
- x_ratio: 0.0=far left, 0.5=center, 1.0=far right
- y_ratio: 0.0=top, 0.5=center, 1.0=bottom

Respond ONLY with valid JSON:
{
  "face_visible": true or false,
  "x_ratio": 0.5,
  "y_ratio": 0.3
}
"""


def scan_scene_dual(use_video: bool = True, video_duration: float = 10.0) -> dict:
    """
    Scan with BOTH cameras simultaneously.
    Returns merged scan result.
    """
    fallback = {
        "object": "unknown", "action": "forward",
        "terrain": "clear", "in_my_path": False,
        "wall_ahead": False, "small_obstacle": False,
        "speak": None, "object_name": None,
        "camera_with_target": "none", "mission_complete": False,
        "distance": "far"
    }

    if use_video:
        # Capture video from both cameras
        log.info("📹 Capturing 10s video from both cameras...")
        frames_nav    = capture_frames_video(CAMERA_WEBCAM,  duration=video_duration, fps_sample=0.5)
        frames_pantilt = capture_frames_video(CAMERA_PANTILT, duration=video_duration, fps_sample=0.5)
        all_frames = frames_nav + frames_pantilt
        if not all_frames:
            return fallback
        response = ask_cosmos(VIDEO_SCAN_PROMPT, frames=all_frames, max_tokens=250)
    else:
        # Single frame from both cameras
        img_nav    = capture_frame(CAMERA_WEBCAM,  width=640, height=480)
        img_pantilt = capture_frame(CAMERA_PANTILT, width=640, height=480)
        frames = [f for f in [img_nav, img_pantilt] if f]
        if not frames:
            return fallback
        if len(frames) == 1:
            response = ask_cosmos(SCAN_PROMPT, image_b64=frames[0], max_tokens=250)
        else:
            response = ask_cosmos(SCAN_PROMPT, frames=frames, max_tokens=250)

    try:
        clean = response.replace("```json", "").replace("```", "").strip()
        start = clean.find("{")
        end   = clean.rfind("}") + 1
        if start >= 0 and end > start:
            clean = clean[start:end]
        result = json.loads(clean)
        # Ensure all keys exist
        for k, v in fallback.items():
            result.setdefault(k, v)
        return result
    except Exception:
        log.warning(f"Could not parse scene JSON: {response[:200]}")
        return fallback


def center_on_person() -> bool:
    """
    Use pan-tilt camera to find and center on a person's face.
    Returns True if face found and centered.
    """
    log.info("🎯 Centering pan-tilt on person...")
    image = capture_frame(CAMERA_PANTILT, width=640, height=480)
    if not image:
        return False

    response = ask_cosmos(FACE_CENTER_PROMPT, image_b64=image, max_tokens=80)
    try:
        clean = response.replace("```json", "").replace("```", "").strip()
        data  = json.loads(clean[clean.find("{"):clean.rfind("}")+1])
        if data.get("face_visible"):
            x = data.get("x_ratio", 0.5)
            y = data.get("y_ratio", 0.3)
            pantilt_center_on_object(x, y)
            time.sleep(0.8)
            autofocus_trigger(CAMERA_PANTILT)
            log.info(f"🎯 Centered on face at ({x:.2f}, {y:.2f})")
            return True
    except Exception as e:
        log.warning(f"Face centering error: {e}")
    return False


# Keep old scan_scene for backward compatibility
def scan_scene(device: int = CAMERA_WEBCAM, use_video: bool = False,
               video_duration: float = 5.0) -> dict:
    return scan_scene_dual(use_video=use_video, video_duration=video_duration)