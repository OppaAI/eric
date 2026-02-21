"""
E.R.I.C. — Cosmos Reason 2 Interface
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
You are E.R.I.C. — Edge Robotics Innovation by Cosmos.
You are a search and rescue tracked ground robot.
The camera view is YOUR view — egocentric, first person.

Your hardware:
- Tracked robot chassis (~30cm wide), built for outdoor terrain
- NVIDIA Jetson Orin Nano Super 8GB
- Cosmos Reason 2 (2B W4A16) via vLLM — your vision and reasoning
- Two cameras: pan-tilt and webcam
- Total cost: ~$750 CAD, built by one person in Kelowna BC Canada
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

def capture_frame(device: int = CAMERA_WEBCAM,
                  width: int = CAMERA_WIDTH,
                  height: int = CAMERA_HEIGHT) -> str | None:
    """Capture frame, return base64 JPEG. Tries full res, falls back if too large."""
    try:
        import cv2
        cap = cv2.VideoCapture(device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            log.error(f"Camera {device}: frame capture failed")
            return None

        # Check pixel count — Cosmos max is 256000
        h, w = frame.shape[:2]
        max_pixels = 256000
        if w * h > max_pixels:
            scale = (max_pixels / (w * h)) ** 0.5
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            log.info(f"Camera {device}: resized {w}x{h} → {int(w*scale)}x{int(h*scale)}")

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode("utf-8")
    except Exception as e:
        log.error(f"Camera {device} error: {e}")
        return None


def capture_frames_video(device: int = CAMERA_WEBCAM,
                         duration: float = 10.0,
                         fps_sample: float = 1.0) -> list[str]:
    """
    Capture video clip and return list of sampled base64 frames.
    duration: seconds to record
    fps_sample: frames per second to sample (1.0 = 1 frame/sec)
    """
    try:
        import cv2, time
        cap     = cv2.VideoCapture(device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        frames  = []
        start   = time.time()
        last    = 0.0
        interval = 1.0 / fps_sample

        while time.time() - start < duration:
            ret, frame = cap.read()
            if not ret:
                break
            elapsed = time.time() - start
            if elapsed - last >= interval:
                # Resize to fit Cosmos pixel budget across multiple frames
                # 10 frames × ~25k pixels each = ~250k total ≈ safe
                frame = cv2.resize(frame, (320, 240))
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frames.append(base64.b64encode(buf).decode("utf-8"))
                last = elapsed

        cap.release()
        log.info(f"📹 Captured {len(frames)} frames over {duration}s")
        return frames
    except Exception as e:
        log.error(f"Video capture error: {e}")
        return []


def capture_frame_raw(device: int = CAMERA_WEBCAM):
    """Capture raw RGB frame for Gradio display (640x480)."""
    try:
        import cv2
        cap = cv2.VideoCapture(device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        # Rotate cam2 (webcam) 90 degrees clockwise to fix sideways mount
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
        return text
    except requests.exceptions.ConnectionError:
        msg = "Cannot connect to Cosmos. Is vLLM running? (bash launch/cosmos.sh)"
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