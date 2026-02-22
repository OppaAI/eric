"""
ERIC — Cosmos Reason 2 Interface
Vision + physical reasoning via vLLM

Design:
- Navigation (moving): pan-tilt only, single frame, fast NAV_PROMPT
- Scanning (stopped):  dual camera, single frame, stable capture
- Stabilization:       pan-tilt settles before every capture
- LED:                 adaptive — on only when frame is dark
- Face/robot centering: pan-tilt only, settle before capture
"""

import json
import base64
import logging
import subprocess
import time
import requests
import numpy as np

from config import (
    VLLM_URL, COSMOS_MODEL,
    CAMERA_WEBCAM, CAMERA_PANTILT,
    CAMERA_WIDTH, CAMERA_HEIGHT
)

log = logging.getLogger("eric.cosmos")

# ─── Settle time after pan-tilt move before capture (seconds) ─────────────────
PANTILT_SETTLE  = 0.7   # wait after any pan-tilt command before capturing
LED_DARK_THRESH = 55    # mean brightness below this = dark, turn on LED

_BASE_SYSTEM_PROMPT = """
You are ERIC — Edge Robotics Innovation by Cosmos.
You are a search and rescue tracked ground robot.
The camera view is YOUR view — egocentric, first person.

Your hardware:
- Tracked robot chassis (~30cm wide), built for outdoor terrain
- NVIDIA Jetson Orin Nano Super 8GB
- Cosmos Reason 2 (2B W4A16) via vLLM — your vision and reasoning
- Two cameras: pan-tilt (wide angle, looking around + navigation) and webcam (close-up scanning)
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
    """Send pan-tilt command. pan/tilt in degrees from center."""
    global _pan_angle, _tilt_angle
    try:
        from motors import motors
        _pan_angle  = max(-90, min(90, pan))
        _tilt_angle = max(-45, min(45, tilt))
        motors.pantilt(_pan_angle, _tilt_angle, speed)
        log.info(f"🎥 Pan-tilt → X:{_pan_angle} Y:{_tilt_angle}")
    except Exception as e:
        log.error(f"Pan-tilt error: {e}")


def pantilt_center():
    """Return pan-tilt to center and wait for settle."""
    pantilt(0, 0)
    time.sleep(PANTILT_SETTLE)


def pantilt_move_wait(pan: int = 0, tilt: int = 0, speed: int = 50):
    """Move pan-tilt and wait for mechanical settle before capture."""
    pantilt(pan, tilt, speed)
    time.sleep(PANTILT_SETTLE)


def pantilt_center_on_target(frame_x_ratio: float, frame_y_ratio: float):
    """
    Center pan-tilt on detected object or face.
    frame_x_ratio: 0.0 (left) to 1.0 (right)
    frame_y_ratio: 0.0 (top) to 1.0 (bottom)
    """
    pan_offset  = int((frame_x_ratio - 0.5) * 80)
    tilt_offset = int((frame_y_ratio - 0.5) * -40)
    new_pan  = max(-90, min(90,  _pan_angle  + pan_offset))
    new_tilt = max(-45, min(45,  _tilt_angle + tilt_offset))
    pantilt_move_wait(new_pan, new_tilt, speed=30)


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
    """Trigger autofocus and wait for settle."""
    try:
        dev = f"/dev/video{device}"
        subprocess.run(["v4l2-ctl", "-d", dev, "--set-ctrl=focus_automatic_continuous=0"], capture_output=True, timeout=2)
        time.sleep(0.3)
        subprocess.run(["v4l2-ctl", "-d", dev, "--set-ctrl=focus_automatic_continuous=1"], capture_output=True, timeout=2)
        time.sleep(1.5)
        log.info(f"🔍 Autofocus triggered on {dev}")
    except Exception as e:
        log.warning(f"Autofocus trigger error: {e}")


# ─── LED Control ──────────────────────────────────────────────────────────────

def _led_on():
    try:
        from motors import motors
        motors.lights(base=0, head=255)
    except Exception as e:
        log.warning(f"LED on error: {e}")


def _led_off():
    try:
        from motors import motors
        motors.lights(base=0, head=0)
    except Exception as e:
        log.warning(f"LED off error: {e}")


def _frame_brightness(image_b64: str) -> float:
    """Return mean brightness (0-255) of a base64 JPEG frame."""
    try:
        import cv2
        data  = base64.b64decode(image_b64)
        arr   = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if frame is not None:
            return float(frame.mean())
    except Exception:
        pass
    return 128.0  # assume normal if we can't check


def _is_dark(image_b64: str) -> bool:
    return _frame_brightness(image_b64) < LED_DARK_THRESH


# ─── Camera ───────────────────────────────────────────────────────────────────

_caps = {}

def _get_cap(device: int, width: int = 640, height: int = 480):
    """Get or create persistent VideoCapture."""
    import cv2
    if device not in _caps or not _caps[device].isOpened():
        cap = cv2.VideoCapture(device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        _caps[device] = cap
    return _caps[device]


def _encode_frame(frame, device: int) -> str | None:
    """Rotate if webcam, resize to fit pixel budget, encode to base64 JPEG."""
    import cv2
    if device == CAMERA_WEBCAM:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    h, w = frame.shape[:2]
    max_pixels = 256000
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("utf-8")


def _grab_frame(device: int, width: int = 640, height: int = 480):
    """Raw grab — flush buffer and read one frame."""
    import cv2
    cap = _get_cap(device, width, height)
    cap.grab()  # flush stale buffer
    ret, frame = cap.read()
    if not ret:
        log.error(f"Camera {device}: read failed")
        _caps.pop(device, None)
        return None
    return frame


def capture_frame(device: int = CAMERA_WEBCAM,
                  width: int  = CAMERA_WIDTH,
                  height: int = CAMERA_HEIGHT,
                  adaptive_led: bool = False) -> str | None:
    """
    Capture a single stable frame.
    adaptive_led=True: check brightness, turn LED on if dark, recapture.
    Pan-tilt must already be settled before calling this.
    """
    try:
        frame = _grab_frame(device, width, height)
        if frame is None:
            return None

        b64 = _encode_frame(frame, device)
        if b64 is None:
            return None

        if adaptive_led and _is_dark(b64):
            log.info(f"🔦 Dark frame on cam {device} — LED on for recapture")
            _led_on()
            time.sleep(0.3)
            frame = _grab_frame(device, width, height)
            _led_off()
            if frame is not None:
                b64 = _encode_frame(frame, device)

        return b64
    except Exception as e:
        log.error(f"Camera {device} error: {e}")
        _caps.pop(device, None)
        return None


def capture_frame_raw(device: int = CAMERA_WEBCAM):
    """Capture raw RGB frame for Gradio display."""
    try:
        import cv2
        frame = _grab_frame(device, 480, 640)  # swap w/h since webcam is physically rotated 90°
        if frame is None:
            return None
        if device == CAMERA_WEBCAM:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


# ─── Dual Stable Capture (for scanning — robot stopped) ──────────────────────

def capture_dual_stable(adaptive_led: bool = True) -> list[str]:
    """
    Capture from both cameras while robot is stopped.
    Pan-tilt returns to center, settles, then both frames captured.
    Returns list of base64 frames (pantilt first, then webcam).
    """
    pantilt_center()  # includes settle wait
    frames = []

    f_pt = capture_frame(CAMERA_PANTILT, 640, 480, adaptive_led=adaptive_led)
    if f_pt:
        frames.append(f_pt)

    f_wc = capture_frame(CAMERA_WEBCAM, 640, 480, adaptive_led=adaptive_led)
    if f_wc:
        frames.append(f_wc)

    log.info(f"📷 Dual stable capture: {len(frames)} frames")
    return frames


# ─── Nav Capture (during movement — pan-tilt only, center, fast) ──────────────

def capture_nav_frame() -> str | None:
    """
    Fast single frame from pan-tilt only for navigation.
    Pan-tilt stays centered (0,0). No LED toggle.
    Robot may be moving — keep it fast.
    """
    return capture_frame(CAMERA_PANTILT, 640, 480, adaptive_led=False)


# ─── Cosmos API ───────────────────────────────────────────────────────────────

def ask_cosmos(prompt: str, image_b64: str = None,
               frames: list[str] = None,
               max_tokens: int = 300, stream: bool = False):
    """
    Query Cosmos Reason 2 via vLLM.
    image_b64: single image
    frames: list of images
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


# ─── Face / Robot Centering ───────────────────────────────────────────────────

FACE_CENTER_PROMPT = """
I am looking at a person OR robot directly in front of me.
Estimate where the face/head/camera of the person or robot is in the frame:
- x_ratio: 0.0=far left, 0.5=center, 1.0=far right
- y_ratio: 0.0=top, 0.5=center, 1.0=bottom

Respond ONLY with valid JSON:
{
  "face_visible": true or false,
  "target_type": "person|robot|unknown",
  "x_ratio": 0.5,
  "y_ratio": 0.3
}
"""


def center_on_person() -> bool:
    """
    Use pan-tilt camera to find and center on a person or robot.
    Pan-tilt settles before capture. Returns True if target found.
    """
    log.info("🎯 Centering pan-tilt on person/robot...")

    # Look slightly upward for face/head level
    pantilt_move_wait(0, -10)
    image = capture_frame(CAMERA_PANTILT, 640, 480, adaptive_led=True)
    if not image:
        pantilt_center()
        return False

    response = ask_cosmos(FACE_CENTER_PROMPT, image_b64=image, max_tokens=80)
    try:
        clean = response.replace("```json", "").replace("```", "").strip()
        data  = json.loads(clean[clean.find("{"):clean.rfind("}")+1])
        if data.get("face_visible"):
            x = data.get("x_ratio", 0.5)
            y = data.get("y_ratio", 0.3)
            pantilt_center_on_target(x, y)  # includes settle wait
            autofocus_trigger(CAMERA_PANTILT)
            log.info(f"🎯 Centered on {data.get('target_type','?')} at ({x:.2f}, {y:.2f})")
            return True
    except Exception as e:
        log.warning(f"Face centering error: {e}")

    pantilt_center()
    return False