"""
ERIC — Mission Logic

Camera strategy:
  Navigation (moving):  pan-tilt only, single frame, fast NAV_PROMPT
  Scanning  (stopped):  dual camera (pan-tilt + webcam), single stable frame each
  360° scan (stopped):  body rotates, pan-tilt tilts up/down, dual stable frames at each stop
  Face/robot centering: pan-tilt only, settle before capture

Stabilization rule:
  Every pantilt_move_wait() includes a settle delay.
  Captures only happen when robot is stopped or pan-tilt has settled.

LED:
  Adaptive — on only when captured frame is dark.
"""

import time
import threading
import logging
import json
import requests

from config import MOTOR_SPEED_SLOW, MOTOR_SPEED_NORMAL, MISSIONS_DIR, VLLM_URL, COSMOS_MODEL
from motors import motors
from cosmos import (
    ask_cosmos, set_mission_briefing,
    capture_frame,
    CAMERA_WEBCAM, CAMERA_PANTILT
)
from tts import speak

log = logging.getLogger("eric.mission")


class State:
    IDLE         = "idle"
    SEARCHING    = "searching"
    SCANNING_360 = "scanning_360"
    INTERACTING  = "interacting"
    AVOIDING     = "avoiding"
    COMPLETE     = "complete"


mission_state        = State.IDLE
mission_active       = False
conversation_history = []

_empty_scans       = 0
_avoid_attempts    = 0
_scans_since_360   = 0
EMPTY_SCAN_LIMIT   = 1   # trigger 360 after just 1 empty scan (was 2)
SCANS_BEFORE_360   = 2   # periodic 360 every 2 quick scans (was 4)
MAX_AVOID_ATTEMPTS = 3   # force 360 sooner (was 4)

_ui_callbacks = {"eric_says": None, "status": None, "log": None}


def register_ui_callbacks(**cbs):
    _ui_callbacks.update(cbs)


def _ui(key, text):
    cb = _ui_callbacks.get(key)
    if cb:
        try: cb(text)
        except Exception: pass


def eric_say(text):
    _ui("eric_says", text)
    speak(text)


def _cosmos_frames(frames, prompt, max_tokens=250, temp=0.3):
    from cosmos import _system_prompt as sys_prompt
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}}
        for f in frames
    ]
    content.append({"type": "text", "text": prompt})
    payload = {
        "model": COSMOS_MODEL,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": content}
        ],
        "max_tokens": max_tokens,
        "temperature": temp,
        "repetition_penalty": 1.15,
    }
    r = requests.post(VLLM_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def _parse_json(response, fallback, label="COSMOS"):
    try:
        clean = response.replace("```json", "").replace("```", "").strip()
        s = clean.find("{")
        e = clean.rfind("}") + 1
        if s >= 0 and e > s:
            result = json.loads(clean[s:e])

            # ── Normalize malformed fields Cosmos sometimes produces ────────
            # "object" may come back as a dict like {"person": [], "robot": ["rubblebucket"]}
            # Flatten it to the first non-empty category, or "clear".
            obj = result.get("object")
            if isinstance(obj, dict):
                priority = ["person", "robot", "slipper", "shoe", "obstacle", "wall", "clear"]
                flat = "unknown"
                for key in priority:
                    if obj.get(key):          # non-empty list → this category has detections
                        flat = key
                        # Also pull object_name from the first item in that list
                        items = obj[key]
                        if isinstance(items, list) and items and not result.get("object_name"):
                            result["object_name"] = str(items[0])
                        break
                    elif key in obj:          # key present but empty list → keep looking
                        flat = key           # use as fallback if nothing better
                result["object"] = flat

            # "object_name" may come back as a list — take first item
            name = result.get("object_name")
            if isinstance(name, list):
                result["object_name"] = name[0] if name else None

            # Any other field that should be a plain string but is a list/dict — stringify it
            _str_fields = ("terrain", "distance", "target_direction",
                           "clearest_direction", "action", "physical_reasoning")
            for field in _str_fields:
                val = result.get(field)
                if isinstance(val, (list, dict)):
                    result[field] = str(val)

            # Fill in missing keys from fallback
            for k, v in fallback.items():
                result.setdefault(k, v)

            # ── Print full result to terminal ──────────────────────────────
            print(f"\n{'─'*60}")
            print(f"🧠 {label}:")
            for k, v in result.items():
                icon = ""
                if k == "object"           and v not in ("clear", "unknown"): icon = "  ⚠️ "
                if k == "wall_ahead"       and v:                              icon = "  🚧 "
                if k == "obstacle_close"   and v:                              icon = "  🚧 "
                if k == "small_obstacle"   and v:                              icon = "  ⚠️ "
                if k == "target_visible"   and v:                              icon = "  🎯 "
                if k == "mission_complete" and v:                              icon = "  🏆 "
                if k == "speak"            and v:                              icon = "  🔊 "
                print(f"  {k:25s}: {v}{icon}")
            print(f"{'─'*60}\n")
            return result
    except Exception:
        log.warning(f"JSON parse failed: {response[:100]}")
        print(f"\n⚠️  RAW RESPONSE ({label}): {response[:400]}\n")
    return fallback


# ─── Mission File Loading ─────────────────────────────────────────────────────

def list_missions():
    if not MISSIONS_DIR.exists():
        return []
    return [f.stem for f in sorted(MISSIONS_DIR.glob("*.yaml"))]


def load_mission_file(name):
    try:
        import yaml
        path = MISSIONS_DIR / f"{name}.yaml"
        if not path.exists():
            return None
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        log.error(f"Failed to load mission {name}: {e}")
        return None


def get_briefing_from_file(name):
    data = load_mission_file(name)
    return data.get("briefing", "").strip() if data else None


# ─── Mission Control ──────────────────────────────────────────────────────────

def start_mission(briefing):
    global mission_active, mission_state, conversation_history
    global _empty_scans, _avoid_attempts, _scans_since_360

    if mission_active:
        return "Mission already active. Disengage first."
    if not briefing.strip():
        return "No mission briefing provided."

    conversation_history = []
    _empty_scans = _avoid_attempts = _scans_since_360 = 0
    set_mission_briefing(briefing)

    motors.pantilt(0, 0)
    time.sleep(0.5)

    ack = ask_cosmos(
        f"Mission briefing:\n\"{briefing}\"\n\n"
        "Acknowledge in 2-3 sentences. State your first action. Be concise.",
        max_tokens=150
    )
    eric_say(ack)

    mission_active = True
    mission_state  = State.SEARCHING
    _ui("status", "SEARCHING")
    motors.oled(0, "ERIC ACTIVE")
    motors.oled(1, "Searching...")
    motors.lights(base=128, head=255)

    threading.Thread(target=_mission_loop, daemon=True).start()
    return ack


def stop_mission():
    global mission_active, mission_state
    mission_active = False
    mission_state  = State.IDLE
    motors.stop()
    motors.lights(0, 0)
    motors.pantilt(0, 0)
    motors.oled(0, "ERIC STOPPED")
    motors.oled(1, "")
    eric_say("Mission disengaged. All systems halted.")
    _ui("status", "IDLE")


def resume_after_interaction():
    global mission_state, _empty_scans, _avoid_attempts, _scans_since_360
    if mission_active:
        _empty_scans = _avoid_attempts = _scans_since_360 = 0
        mission_state = State.SEARCHING
        motors.pantilt(0, 0)
        motors.forward(MOTOR_SPEED_SLOW)
        motors.oled(0, "ERIC ACTIVE")
        motors.oled(1, "Searching...")
        _ui("status", "SEARCHING")


# ─── Prompts ──────────────────────────────────────────────────────────────────

NAV_PROMPT = """
You are a tracked ground robot. These frames are from your forward camera while moving.
Study how the scene changes across frames.

RULES:
- Wall or large object filling lower half across multiple frames → wall_ahead = true
- Any object getting visibly closer/larger → obstacle_close = true
- Small ground hazard (cable, rug edge, step) → small_obstacle = true
- Any human or robot visible anywhere → person_visible = true
- ONLY set action=forward if path is clear for at least 1.5 meters
- When unsure: obstacle_close=true, action=stop

OUTPUT: A single JSON object. Every field is REQUIRED. Use ONLY the exact field names shown.
STRING fields must be a single word from the options listed — NOT a list, NOT a dict, NOT null.
BOOLEAN fields must be true or false.

Example output (copy this structure exactly, change values to match what you see):
{
  "wall_ahead": false,
  "obstacle_close": false,
  "small_obstacle": false,
  "person_visible": false,
  "action": "forward",
  "physical_reasoning": "Path is clear ahead for at least two meters."
}

Now analyze the frames and output ONLY the JSON object above. No markdown. No explanation. No extra fields.
"""

SCAN_360_PROMPT = """
You are a tracked ground robot. These images are from a full 360-degree scan — 4 body positions.
At each position the camera tilted up (far view) and down (floor view). You are completely stopped.

STEP 1 — SAFETY: Look at all floor-level frames. Which direction has the most open space?
STEP 2 — MISSION TARGET: People, robots, slippers, shoes — even partially visible counts. Set target_visible=true if 30%+ confident.
STEP 3 — SPEAK: If you see the target, set speak to an excited 1-sentence reaction. Otherwise null.

OUTPUT: A single JSON object. Every field is REQUIRED. Use ONLY the exact field names shown.
"object" must be one word: person, robot, slipper, shoe, obstacle, wall, clear, or unknown — NOT a list or dict.
"object_name" must be a short string or null — NOT a list or dict.
All other string fields: pick exactly one option from those shown.
Boolean fields: true or false only.

Example output (copy this structure exactly, change values to match what you see):
{
  "object": "clear",
  "object_name": null,
  "terrain": "tiles",
  "distance": "far",
  "in_my_path": false,
  "wall_ahead": false,
  "small_obstacle": false,
  "target_visible": false,
  "target_direction": "unknown",
  "clearest_direction": "front",
  "action": "forward",
  "speak": null,
  "physical_reasoning": "No target found. Hallway ahead is the clearest direction.",
  "mission_complete": false
}

Now analyze the images and output ONLY the JSON object above. No markdown. No explanation. No extra fields.
"""

QUICK_SCAN_PROMPT = """
You are a tracked ground robot. You are completely stopped. These frames are from your pan-tilt and webcam cameras.

OBSTACLE CHECK — lower half of every image:
- Anything filling/touching the bottom edge → wall_ahead = true
- Object within ~60cm ahead → obstacle_close = true AND in_my_path = true
- When in doubt: obstacle_close = true

MISSION TARGET CHECK:
- Person, robot, slipper, shoe — even partially visible → target_visible = true

OUTPUT: A single JSON object. Every field is REQUIRED. Use ONLY the exact field names shown.
"object" must be one word: person, robot, slipper, shoe, obstacle, wall, clear, or unknown — NOT a list or dict.
"object_name" must be a short string or null — NOT a list or dict.
All other string fields: pick exactly one option from those shown.
Boolean fields: true or false only.

Example output (copy this structure exactly, change values to match what you see):
{
  "object": "clear",
  "object_name": null,
  "terrain": "tiles",
  "distance": "far",
  "in_my_path": false,
  "wall_ahead": false,
  "obstacle_close": false,
  "small_obstacle": false,
  "target_visible": false,
  "target_direction": "front",
  "clearest_direction": "front",
  "action": "forward",
  "speak": null,
  "physical_reasoning": "Path is clear. No target visible.",
  "mission_complete": false
}

Now analyze the frames and output ONLY the JSON object above. No markdown. No explanation. No extra fields.
"""

_SCAN_FALLBACK = {
    "object": "unknown", "object_name": None, "terrain": "clear",
    "distance": "far", "in_my_path": False, "wall_ahead": False,
    "small_obstacle": False, "target_visible": False,
    "target_direction": "unknown", "clearest_direction": "front",
    "action": "stop", "speak": None,   # SAFE default — never forward on failure
    "physical_reasoning": "", "mission_complete": False
}

_NAV_FALLBACK = {
    "wall_ahead": False, "obstacle_close": False, "small_obstacle": False,
    "person_visible": False,
    "action": "stop", "physical_reasoning": ""  # SAFE default — stop on nav failure
}


# ─── Navigation Check (while moving) ─────────────────────────────────────────

# Nav clip settings — tune these
NAV_CLIP_DURATION = 10.0  # seconds of video per nav check
NAV_CLIP_FPS      = 2     # frames per second (10s x 2fps = 20 frames to Cosmos)
NAV_IMAGE_INTERVAL = 4.0  # seconds between nav image checks while moving

def _nav_check() -> dict:
    """
    Image-based nav check while moving.
    Single pan-tilt frame every NAV_IMAGE_INTERVAL seconds.
    Much faster than 10s video clip — allows more frequent obstacle checks.
    """
    _ui("log", "📷 Nav check...")
    motors.oled(1, "Nav check...")

    frame = capture_frame(CAMERA_PANTILT, 320, 240)
    if not frame:
        return dict(_NAV_FALLBACK)

    # Simplified nav prompt for single image
    NAV_IMAGE_PROMPT = """
You are a tracked ground robot moving forward. This is a single frame from your forward camera.

Check ONLY for immediate safety hazards:
- Wall or large object filling the lower 40% of frame → wall_ahead = true
- Any object within ~60cm directly ahead → obstacle_close = true
- Small ground obstacle (cables, edges, steps) → small_obstacle = true
- Person or robot visible anywhere → person_visible = true

OUTPUT: A single JSON object. Every field is REQUIRED. Use ONLY the exact field names shown.
STRING fields must be a single word from the options listed — NOT a list, NOT a dict, NOT null.
BOOLEAN fields must be true or false.

Example output (copy this structure exactly, change values to match what you see):
{
  "wall_ahead": false,
  "obstacle_close": false,
  "small_obstacle": false,
  "person_visible": false,
  "action": "forward",
  "physical_reasoning": "Path ahead is clear with no obstacles visible."
}

Now analyze the frame and output ONLY the JSON object above. No markdown. No explanation. No extra fields.
"""
    try:
        response = ask_cosmos(NAV_IMAGE_PROMPT, image_b64=frame, max_tokens=120)
        result = _parse_json(response, dict(_NAV_FALLBACK), label="NAV CHECK")

        if result.get("person_visible") and mission_active:
            motors.stop()
            _ui("log", "👤 Person spotted during nav — stopping")
            _ui("status", "PERSON SPOTTED")
            greeting = ask_cosmos(
                "You spotted someone ahead while navigating. "
                "Greet them and ask if they can help with your mission. 1-2 sentences.",
                max_tokens=60
            )
            eric_say(greeting)

        return result
    except Exception as e:
        log.error(f"Nav check error: {e}")
        return dict(_NAV_FALLBACK)


# ─── Quick Scan (stopped) ─────────────────────────────────────────────────────

def _quick_scan() -> dict:
    """
    Dual camera stable scan while stopped.
    Pan-tilt centers + settles, then both cameras captured.
    """
    motors.pantilt(0, 0)
    time.sleep(0.5)
    frames = []
    pt = capture_frame(CAMERA_PANTILT, 640, 480)
    if pt:
        frames.append(pt)
    wc = capture_frame(CAMERA_WEBCAM, 640, 480)
    if wc:
        frames.append(wc)
    if not frames:
        return dict(_SCAN_FALLBACK)
    try:
        print(f"\n📷 QUICK SCAN — {len(frames)} frames to Cosmos...")
        response = _cosmos_frames(frames, QUICK_SCAN_PROMPT, max_tokens=200, temp=0.3)
        return _parse_json(response, dict(_SCAN_FALLBACK), label="QUICK SCAN RESULT")
    except Exception as e:
        log.error(f"Quick scan error: {e}")
        return dict(_SCAN_FALLBACK)


# ─── 360° Scan (stopped) ──────────────────────────────────────────────────────

TURN_90_SEC      = 2.2   # seconds to turn 90° at MOTOR_SPEED_SLOW — tune if needed
BLUR_THRESHOLD   = 80.0  # Laplacian variance below this = blurry, retry
MAX_BLUR_RETRIES = 3


def _is_blurry(frame_b64: str) -> bool:
    """Return True if frame is too blurry to use (low Laplacian variance)."""
    try:
        import cv2
        import numpy as np
        import base64
        data  = base64.b64decode(frame_b64)
        arr   = np.frombuffer(data, np.uint8)
        img   = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False
        score = cv2.Laplacian(img, cv2.CV_64F).var()
        log.debug(f"Sharpness score: {score:.1f}")
        return score < BLUR_THRESHOLD
    except Exception:
        return False


def _capture_sharp(device: int, retries: int = MAX_BLUR_RETRIES) -> str | None:
    """Capture a frame, retrying if blurry. Returns best frame found."""
    best = None
    for attempt in range(retries):
        f = capture_frame(device, 640, 480)
        if f is None:
            break
        if not _is_blurry(f):
            return f   # sharp enough
        log.info(f"Blurry frame on cam {device} (attempt {attempt+1}) — waiting and retrying...")
        best = f       # keep as fallback
        time.sleep(0.5)
    return best  # return best we got even if still blurry


def _scan_360_smart() -> dict:
    """
    Full 360° scan using image-based scanning with early target identification.

    At each of 4 body positions (0°, 90°, 180°, 270°):
      1. Use _quick_scan() at each tilt position
      2. If target VISIBLE during scan → return immediately, don't continue
      3. If potential target — record and continue scan
      4. Collect all wide frames for final Cosmos 360 overview if needed

    This means Eric can identify a target mid-scan and stop early —
    much smarter than waiting to send 16 frames all at once.
    """
    global mission_state
    mission_state = State.SCANNING_360
    _ui("status", "360 SCANNING")
    motors.oled(0, "360 Scan")
    log.info("Starting smart 360 image scan")

    motors.stop()
    time.sleep(0.5)

    all_frames   = []   # collect for final overview if no target found
    best_spot    = None  # best potential target seen so far

    for pos in range(4):
        deg = pos * 90
        _ui("log", f"Scanning {deg}°...")
        motors.oled(1, f"Scan {deg}deg")

        # Tilt up to see mid/far range, then level
        for tilt, label in [(-20, "up"), (10, "level")]:
            motors.pantilt(0, tilt, 40)
            time.sleep(0.5)

            # Wide frame for overview collection
            f_pt = _capture_sharp(CAMERA_PANTILT)
            if f_pt:
                all_frames.append(f_pt)

            # Quick scan at this position
            result = _quick_scan()

            # ── Target found mid-scan — stop early ───────────────────────
            if result.get("target_visible"):
                log.info(f"🎯 Target VISIBLE at {deg}° tilt={label} — stopping scan early!")
                _ui("log", f"Target visible at {deg}° — stopping scan!")
                motors.oled(1, "TARGET FOUND!")
                return {
                    "object":             result.get("object", "person"),
                    "object_name":        result.get("object_name"),
                    "terrain":            result.get("terrain", "clear"),
                    "distance":           result.get("distance", "close"),
                    "in_my_path":         True,
                    "wall_ahead":         False,
                    "obstacle_close":     False,
                    "small_obstacle":     False,
                    "target_visible":     True,
                    "target_direction":   "front",
                    "clearest_direction": "front",
                    "action":             "stop",
                    "speak":              result.get("speak"),
                    "physical_reasoning": f"Target confirmed at {deg}° tilt={label}",
                    "mission_complete":   False
                }

            # ── Remember best non-empty result for re-visit ───────────────
            if result.get("object") not in ("clear", "unknown", None):
                if best_spot is None:
                    best_spot = (deg, result)
                    log.info(f"Potential target ({result.get('object')}) at {deg}° — continuing scan")

            # ── Hard obstacle detected ────────────────────────────────────
            if result.get("wall_ahead") or result.get("obstacle_close"):
                log.info(f"Obstacle at {deg}° during 360 scan")

        # Re-centre pan-tilt before body rotation
        motors.pantilt(0, 0)
        time.sleep(0.3)

        if pos < 3:
            motors.right(MOTOR_SPEED_SLOW)
            time.sleep(TURN_90_SEC)
            motors.stop()
            time.sleep(0.5)

    # ── Re-visit best potential target for a second look ─────────────────────
    if best_spot:
        deg, spot = best_spot
        _ui("log", f"Re-visiting best potential target at {deg}°...")
        steps_back = (4 - (deg // 90)) % 4
        if steps_back > 0:
            motors.right(MOTOR_SPEED_SLOW)
            time.sleep(TURN_90_SEC * steps_back)
            motors.stop()
            time.sleep(0.5)
        result = _quick_scan()
        if result.get("target_visible") or result.get("object") not in ("clear", "unknown"):
            return {
                "object":             result.get("object", "person"),
                "object_name":        result.get("object_name"),
                "terrain":            result.get("terrain", "clear"),
                "distance":           result.get("distance", "medium"),
                "in_my_path":         True,
                "wall_ahead":         False,
                "obstacle_close":     False,
                "small_obstacle":     False,
                "target_visible":     True,
                "target_direction":   "front",
                "clearest_direction": "front",
                "action":             "stop",
                "speak":              result.get("speak"),
                "physical_reasoning": "Target confirmed on second look",
                "mission_complete":   False
            }

    # ── No target found — send overview frames to Cosmos for direction ────────
    log.info(f"No target confirmed — sending {len(all_frames)} overview frames to Cosmos")
    _ui("log", f"360 done — {len(all_frames)} frames → Cosmos overview")
    motors.oled(1, "Analyzing...")

    if not all_frames:
        return dict(_SCAN_FALLBACK)

    try:
        response = _cosmos_frames(all_frames, SCAN_360_PROMPT, max_tokens=300, temp=0.2)
        return _parse_json(response, dict(_SCAN_FALLBACK), label="360° OVERVIEW")
    except Exception as e:
        log.error(f"360 overview Cosmos error: {e}")
        return dict(_SCAN_FALLBACK)


# ─── Direction Control ────────────────────────────────────────────────────────

def _face_direction(direction):
    if direction == "right":
        motors.right(MOTOR_SPEED_SLOW); time.sleep(1.5); motors.stop()
    elif direction == "back":
        motors.right(MOTOR_SPEED_SLOW); time.sleep(3.0); motors.stop()
    elif direction == "left":
        motors.left(MOTOR_SPEED_SLOW);  time.sleep(1.5); motors.stop()
    time.sleep(0.3)


# ─── Obstacle Avoidance ───────────────────────────────────────────────────────

def _avoid_obstacle(wall_ahead, small_obstacle) -> bool:
    """
    Obstacle avoidance. Returns True if 360 scan should be forced.
    Always re-scans after turning to confirm the new path is clear.
    """
    global _avoid_attempts, mission_state
    _avoid_attempts += 1
    mission_state = State.AVOIDING
    _ui("status", "AVOIDING")

    if wall_ahead:
        _ui("log", f"Wall — attempt {_avoid_attempts}")
        motors.oled(1, "Wall! Back up...")
        motors.stop(); time.sleep(0.3)

        # Back up more decisively
        motors.backward(MOTOR_SPEED_SLOW); time.sleep(1.5)
        motors.stop(); time.sleep(0.3)

        # Longer turn — alternate left/right, increase turn time each attempt
        turn_time = min(1.8 + (_avoid_attempts * 0.4), 3.5)
        if _avoid_attempts % 2 == 1:
            motors.right(MOTOR_SPEED_SLOW); time.sleep(turn_time)
        else:
            motors.left(MOTOR_SPEED_SLOW);  time.sleep(turn_time)
        motors.stop(); time.sleep(0.5)

        if _avoid_attempts >= MAX_AVOID_ATTEMPTS:
            _avoid_attempts = 0
            eric_say("Too many obstacles. Let me scan the full area.")
            return True  # trigger 360

        # Re-scan after turning — confirm new path is clear before moving
        _ui("log", "Re-scanning after avoidance...")
        rescan = _quick_scan()
        if rescan.get("wall_ahead") or rescan.get("obstacle_close"):
            _ui("log", "Still blocked after turn — trying again")
            return _avoid_obstacle(wall_ahead=True, small_obstacle=False)

    elif small_obstacle:
        _ui("log", "Small obstacle — stepping around")
        motors.oled(1, "Step around...")
        motors.stop(); time.sleep(0.2)
        motors.right(MOTOR_SPEED_SLOW); time.sleep(1.1)
        motors.stop(); time.sleep(0.2)
        motors.forward(MOTOR_SPEED_SLOW); time.sleep(1.2)
        motors.stop(); time.sleep(0.2)
        motors.left(MOTOR_SPEED_SLOW);   time.sleep(1.1)
        motors.stop(); time.sleep(0.4)

        # Quick check after step-around
        rescan = _quick_scan()
        if rescan.get("wall_ahead") or rescan.get("obstacle_close"):
            return _avoid_obstacle(wall_ahead=True, small_obstacle=False)

    return False


# ─── Mission Complete ─────────────────────────────────────────────────────────

def _handle_mission_complete(obj_name):
    global mission_active, mission_state
    log.info(f"MISSION COMPLETE — {obj_name}")
    mission_state = State.COMPLETE
    motors.stop()
    motors.oled(0, "MISSION DONE!")
    motors.oled(1, (obj_name or "Target")[:16])
    _ui("status", "MISSION COMPLETE")

    for _ in range(5):
        motors.lights(255, 255); time.sleep(0.25)
        motors.lights(0, 0);    time.sleep(0.25)
    motors.lights(128, 255)

    # Tilt up slightly to face the target at close range
    motors.pantilt(0, -10)
    time.sleep(0.5)

    announcement = ask_cosmos(
        f"You found: {obj_name or 'the target'}. Mission complete. "
        "Warm triumphant 2-3 sentence announcement.",
        max_tokens=120
    )
    eric_say(announcement)
    _ui("eric_says", announcement)
    _ui("log", f"COMPLETE: {announcement}")

    mission_active = False
    _ui("status", "MISSION COMPLETE")
    motors.oled(0, "TARGET FOUND!")
    motors.oled(1, "Mission done!")


# ─── Character Interaction ────────────────────────────────────────────────────

def handle_character_response(character, said):
    global conversation_history
    conversation_history.append({"character": character, "said": said, "time": time.time()})
    history = "\n".join(f"- {e['character']}: {e['said']}" for e in conversation_history[-5:])
    n = sum(1 for e in conversation_history if e["character"] == character)

    response = ask_cosmos(
        f"Talking to {character}. They said: \"{said}\"\n"
        f"Info gathered:\n{history}\nExchange #{n}.\n\n"
        "If off-topic or exchange 3+ with no new info: thank them, end with [MOVE_ON]\n"
        "Otherwise: respond and ask follow-up. 2 sentences max.",
        max_tokens=150
    )

    move_on = "[MOVE_ON]" in response
    clean   = response.replace("[MOVE_ON]", "").strip()
    eric_say(clean)
    _ui("log", f"[{character}]: {said}\n[Eric]: {clean}")
    if move_on:
        resume_after_interaction()
    return clean


# ─── Process Scan Result ──────────────────────────────────────────────────────

def _process_scan(scan, from_360=False):
    global mission_state, _empty_scans, _avoid_attempts, _scans_since_360

    obj            = scan.get("object", "unknown")
    obj_name       = scan.get("object_name")
    terrain        = scan.get("terrain", "clear")
    in_path        = scan.get("in_my_path", False)
    wall_ahead     = scan.get("wall_ahead", False)
    small_obs      = scan.get("small_obstacle", False)
    action         = scan.get("action", "forward")
    speak_tx       = scan.get("speak")
    reason         = scan.get("physical_reasoning", "")
    distance       = scan.get("distance", "far")
    complete       = scan.get("mission_complete", False)
    target_visible = scan.get("target_visible", False)
    target_dir     = scan.get("target_direction", "front")
    clear_dir      = scan.get("clearest_direction", "front")

    if reason:
        log.info(f"Cosmos: {reason}")
        _ui("log", f"Cosmos: {reason}")

    if complete:
        if speak_tx: eric_say(speak_tx)
        _handle_mission_complete(obj_name)
        return

    obstacle_close = scan.get("obstacle_close", False)

    # Treat unknown with in_path as obstacle — never blindly forward on uncertainty
    if obj == "unknown" and in_path:
        log.info("Unknown object in path — treating as obstacle")
        wall_ahead = True

    if wall_ahead or obstacle_close or (in_path and obj in ["wall", "obstacle"]):
        motors.stop()
        if speak_tx: eric_say(speak_tx)
        is_wall = wall_ahead or (obj == "wall")
        force_360 = _avoid_obstacle(wall_ahead=is_wall, small_obstacle=small_obs)
        if force_360:
            _scans_since_360 = SCANS_BEFORE_360
        else:
            motors.forward(MOTOR_SPEED_SLOW)
        mission_state = State.SEARCHING
        return

    if small_obs and not target_visible:
        _avoid_obstacle(wall_ahead=False, small_obstacle=True)
        motors.forward(MOTOR_SPEED_SLOW)

    if not wall_ahead and not obstacle_close and not small_obs:
        _avoid_attempts = 0

    if speak_tx:
        eric_say(speak_tx)

    if target_visible and from_360:
        _empty_scans = 0
        _ui("log", f"Target spotted {target_dir}!")
        motors.oled(1, f"Target {target_dir}!")
        _ui("status", "TARGET SPOTTED")
        _face_direction(target_dir)
        motors.forward(MOTOR_SPEED_SLOW)
        mission_state = State.SEARCHING
        return

    if in_path and obj in ["person", "robot"]:
        _empty_scans = 0
        motors.stop()
        mission_state = State.INTERACTING
        name = obj_name or obj
        motors.oled(0, name[:16])
        motors.oled(1, "Centering...")
        _ui("status", f"FOUND — {name}")

        # Tilt up slightly to face the person at close range
        motors.pantilt(0, -15)
        time.sleep(0.5)

        motors.oled(1, "Talking...")
        greeting = ask_cosmos(
            f"You see {name} ahead. Greet them and ask about your mission. 1-2 sentences.",
            max_tokens=80
        )
        eric_say(greeting)
        _ui("status", f"TALKING — {name}")
        return

    if obj in ["clear", "unknown"] and not target_visible:
        _empty_scans += 1
        _ui("log", f"Nothing found ({_empty_scans}/{EMPTY_SCAN_LIMIT})")

    if from_360 and clear_dir != "front":
        _face_direction(clear_dir)

    if action == "navigate_around":
        motors.left(MOTOR_SPEED_SLOW); time.sleep(0.8)
        motors.forward(MOTOR_SPEED_SLOW)
    elif action == "slow" or terrain == "carpet":
        motors.slow()
    elif action == "stop":
        motors.stop()
    elif action == "turn_right":
        motors.right(MOTOR_SPEED_SLOW); time.sleep(1.0); motors.stop()
        motors.forward(MOTOR_SPEED_SLOW)
    elif action == "turn_left":
        motors.left(MOTOR_SPEED_SLOW);  time.sleep(1.0); motors.stop()
        motors.forward(MOTOR_SPEED_SLOW)
    elif action == "turn_back":
        motors.right(MOTOR_SPEED_SLOW); time.sleep(3.0); motors.stop()
        motors.forward(MOTOR_SPEED_SLOW)
    else:
        motors.forward(MOTOR_SPEED_SLOW)

    mission_state = State.SEARCHING
    motors.oled(0, "ERIC ACTIVE")
    motors.oled(1, "Searching...")
    _ui("status", "SEARCHING")


# ─── Mission Loop ─────────────────────────────────────────────────────────────
# Nav clip is 10s — after each clip, do a quick stopped scan.
# After SCANS_BEFORE_360 quick scans with nothing found, do a full 360.
# The 10s clip itself IS the nav check — no separate interval needed.

_nav_clips_since_scan = 0
NAV_CLIPS_BETWEEN_SCANS = 2  # do a quick stopped scan every 2 nav clips (~20s of movement)


def _mission_loop():
    global mission_active, mission_state, _empty_scans, _scans_since_360
    global _avoid_attempts, _nav_clips_since_scan

    eric_say("Starting initial 360 degree scan of the area.")
    scan = _scan_360_smart()
    _process_scan(scan, from_360=True)

    if mission_active and mission_state == State.SEARCHING:
        motors.forward(MOTOR_SPEED_SLOW)

    _nav_clips_since_scan = 0

    while mission_active:
        try:
            if mission_state in (State.INTERACTING, State.COMPLETE):
                time.sleep(0.5)
                continue

            # ── Nav check while moving ────────────────────────────────────
            if _nav_clips_since_scan < NAV_CLIPS_BETWEEN_SCANS:
                _nav_clips_since_scan += 1
                nav = _nav_check()

                if nav.get("wall_ahead") or nav.get("obstacle_close"):
                    motors.stop()
                    _ui("log", f"Nav: obstacle — {nav.get('physical_reasoning','')}")
                    force_360 = _avoid_obstacle(
                        wall_ahead=nav.get("wall_ahead", False),
                        small_obstacle=nav.get("small_obstacle", False)
                    )
                    if force_360:
                        _scans_since_360 = SCANS_BEFORE_360
                        _nav_clips_since_scan = NAV_CLIPS_BETWEEN_SCANS  # force scan
                    else:
                        motors.forward(MOTOR_SPEED_SLOW)
                elif nav.get("action") == "slow":
                    motors.slow()
                elif nav.get("action") == "stop":
                    motors.stop()
                else:
                    motors.forward(MOTOR_SPEED_SLOW)
                continue

            # ── Stopped scan every NAV_CLIPS_BETWEEN_SCANS checks ─────────
            _nav_clips_since_scan = 0
            _scans_since_360 += 1
            motors.stop()
            time.sleep(0.3)

            do_360 = (_empty_scans >= EMPTY_SCAN_LIMIT or
                      _scans_since_360 >= SCANS_BEFORE_360)

            if do_360:
                if _empty_scans >= EMPTY_SCAN_LIMIT:
                    eric_say("Nothing found. Performing a full 360 scan.")
                else:
                    _ui("log", "Periodic 360 scan...")
                scan = _scan_360_smart()
                _scans_since_360 = _empty_scans = 0
                _process_scan(scan, from_360=True)
                if mission_active and mission_state == State.SEARCHING:
                    motors.forward(MOTOR_SPEED_SLOW)
            else:
                _ui("log", "Quick scan (stopped)...")
                motors.oled(1, "Scanning...")
                scan = _quick_scan()
                _process_scan(scan, from_360=False)

            time.sleep(0.3)

        except Exception as e:
            log.error(f"Mission loop error: {e}")
            time.sleep(1)

    motors.stop()
    mission_state = State.IDLE
    _ui("status", "IDLE")
    log.info("Mission loop ended")