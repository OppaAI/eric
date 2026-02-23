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
_last_good_scan      = None   # FIX B2: remember last valid scan so JSON failures don't freeze Eric
_target_keywords     = set()  # FIX B5: words from mission briefing used to detect target by name

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
        clean = response.replace("```json", "").replace("```python", "").replace("```", "").strip()

        # ── Reject pure string lists — ["r2d2"], ["wall_ahead", ...] ─────────
        # Cosmos nav check sometimes returns a Python list of strings instead of JSON.
        # These parse fine as JSON arrays but contain no useful data — discard them.
        arr_start = clean.find("[")
        obj_start = clean.find("{")
        if arr_start >= 0 and (obj_start < 0 or arr_start < obj_start):
            arr_end = clean.rfind("]") + 1
            if arr_end > arr_start:
                items = json.loads(clean[arr_start:arr_end])
                if isinstance(items, list) and items:
                    # If it's a list of dicts → merge them
                    if isinstance(items[0], dict):
                        result = _merge_array_items(items, fallback)
                        return _finalize_result(result, fallback, label)
                    # If it's a list of strings → Cosmos gave us garbage, treat as failure
                    log.warning(f"Cosmos returned string list {items[:3]} — ignoring")
                    raise ValueError("string list, not JSON object")

        # ── Normal single-object JSON ─────────────────────────────────────────
        s = clean.find("{")
        e = clean.rfind("}") + 1
        if s >= 0 and e > s:
            result = json.loads(clean[s:e])
            return _finalize_result(result, fallback, label)

    except Exception:
        log.warning(f"JSON parse failed: {response[:100]}")
        print(f"\n⚠️  RAW RESPONSE ({label}): {response[:400]}\n")
    # FIX B2: on parse failure, use last known-good scan rather than freezing on "stop"
    global _last_good_scan
    if _last_good_scan and label in ("QUICK SCAN RESULT", "NAV CHECK", "360° OVERVIEW"):
        log.info(f"JSON parse failed — using last good scan to avoid unnecessary stop")
        return dict(_last_good_scan)
    return fallback


# Object-name → category mapping for when Cosmos sets object="unknown"
# but object_name reveals what it actually is.
_NAME_TO_CATEGORY = {
    # obstacles / furniture
    "book": "obstacle", "box": "obstacle", "bag": "obstacle",
    "chair": "obstacle", "table": "obstacle", "desk": "obstacle",
    "bottle": "obstacle", "cup": "obstacle", "shoe": "shoe",
    "slipper": "slipper", "sandal": "slipper",
    # people
    "man": "person", "woman": "person", "person": "person",
    "human": "person", "child": "person", "kid": "person",
    # robots
    "droid": "robot", "robot": "robot", "r2": "robot", "bb8": "robot",
    # walls / structural
    "wall": "wall", "door": "wall", "fence": "wall",
}

_OBJ_PRIORITY = ["person", "robot", "slipper", "shoe", "obstacle", "wall", "clear", "unknown"]


def _infer_category(obj: str, name: str | None) -> str:
    """If obj is 'unknown' but name hints at a real category, return that category."""
    if obj not in ("unknown", "", None):
        return obj
    if not name:
        return obj or "unknown"
    name_lower = str(name).lower()
    for keyword, category in _NAME_TO_CATEGORY.items():
        if keyword in name_lower:
            return category
    return obj or "unknown"


def _merge_array_items(items: list, fallback: dict) -> dict:
    """Merge a list of per-frame result dicts into one combined result."""
    merged = dict(fallback)
    names = []
    for item in items:
        if not isinstance(item, dict):
            continue
        # Pick highest-priority object seen across frames
        item_obj = _infer_category(
            item.get("object", "unknown"),
            item.get("object_name")
        )
        merged_obj = merged.get("object", "unknown")
        if _OBJ_PRIORITY.index(item_obj) < _OBJ_PRIORITY.index(merged_obj):
            merged["object"] = item_obj
        # Collect names
        n = item.get("object_name")
        if n and str(n) not in names:
            names.append(str(n))
        # OR all boolean flags
        for flag in ("wall_ahead", "obstacle_close", "small_obstacle",
                     "target_visible", "in_my_path", "mission_complete"):
            if item.get(flag):
                merged[flag] = True
        # Take first non-empty string fields
        for field in ("terrain", "distance", "target_direction",
                      "clearest_direction", "action", "speak", "physical_reasoning"):
            if not merged.get(field) or merged[field] in (None, "", fallback.get(field)):
                val = item.get(field)
                if val and val not in (None, ""):
                    merged[field] = val
    merged["object_name"] = ", ".join(names) if names else None
    return merged


def _finalize_result(result: dict, fallback: dict, label: str) -> dict:
    """Normalize types, infer category from name, fill fallback, print."""
    # Flatten dict-type "object" field
    obj = result.get("object")
    if isinstance(obj, dict):
        priority = ["person", "robot", "slipper", "shoe", "obstacle", "wall", "clear"]
        flat = "unknown"
        for key in priority:
            if obj.get(key):
                flat = key
                items = obj[key]
                if isinstance(items, list) and items and not result.get("object_name"):
                    result["object_name"] = str(items[0])
                break
            elif key in obj:
                flat = key
        result["object"] = flat

    # Flatten list-type "object_name"
    name = result.get("object_name")
    if isinstance(name, list):
        result["object_name"] = ", ".join(str(x) for x in name if x) or None

    # Infer category from name when object is "unknown"
    result["object"] = _infer_category(result.get("object", "unknown"),
                                        result.get("object_name"))

    # FIX B5: Cosmos contradiction fix — if it named the mission target but set target_visible=False,
    # override it. Catches the "object: robot, object_name: r2d2, target_visible: False" bug.
    if _target_keywords and not result.get("target_visible"):
        obj_name_str  = str(result.get("object_name") or "").lower()
        obj_str       = str(result.get("object") or "").lower()
        combined      = obj_name_str + " " + obj_str
        matched       = [kw for kw in _target_keywords if kw in combined]
        if matched:
            log.info(f"🎯 Target keyword match '{matched}' in object_name — forcing target_visible=True")
            result["target_visible"] = True
            if result.get("target_direction", "unknown") == "unknown":
                result["target_direction"] = "front"
            if not result.get("action") or result["action"] == "stop":
                result["action"] = "forward"

    # FIX: infer in_my_path — Cosmos almost never sets this to True even when target is right ahead.
    # If target is visible, direction is front, and distance is close/nearby → it's in our path.
    if result.get("target_visible"):
        dist = str(result.get("distance", "")).lower()
        tdir = str(result.get("target_direction", "front")).lower()
        if dist in ("close", "nearby") or tdir == "front":
            result["in_my_path"] = True

    # Stringify any remaining list/dict in string fields
    for field in ("terrain", "distance", "target_direction",
                  "clearest_direction", "action", "physical_reasoning"):
        val = result.get(field)
        if isinstance(val, (list, dict)):
            result[field] = str(val)

    # Fill missing keys from fallback
    for k, v in fallback.items():
        result.setdefault(k, v)

    # FIX B2: remember this as the last known-good scan
    global _last_good_scan
    _last_good_scan = dict(result)

    # ── Print ──────────────────────────────────────────────────────────────
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
    global _empty_scans, _avoid_attempts, _scans_since_360, _target_keywords

    if mission_active:
        return "Mission already active. Disengage first."
    if not briefing.strip():
        return "No mission briefing provided."

    conversation_history = []
    _empty_scans = _avoid_attempts = _scans_since_360 = 0

    # FIX B5: extract target keywords from briefing so we can catch Cosmos contradictions
    # e.g. briefing "find r2d2" → _target_keywords = {"r2d2", "r2", "d2"}
    import re
    words = re.findall(r"[a-zA-Z0-9]+", briefing.lower())
    stopwords = {"find", "the", "a", "an", "go", "to", "and", "or", "in", "at",
                 "with", "for", "of", "is", "it", "locate", "search", "mission",
                 "your", "my", "deliver", "message", "room"}
    _target_keywords = {w for w in words if w not in stopwords and len(w) >= 2}
    log.info(f"🎯 Target keywords from briefing: {_target_keywords}")

    set_mission_briefing(briefing)

    motors.pantilt(0, 5)   # slight downward tilt — see ground objects at normal range
    motors.lights(0, 0)    # LEDs off — only turn on if scene is pitch black
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

    threading.Thread(target=_mission_loop, daemon=True).start()
    return ack


def stop_mission():
    global mission_active, mission_state
    mission_active = False
    mission_state  = State.IDLE
    motors.stop()
    motors.lights(0, 0)
    motors.pantilt(0, 5)
    motors.oled(0, "ERIC STOPPED")
    motors.oled(1, "")
    eric_say("Mission disengaged. All systems halted.")
    _ui("status", "IDLE")


def resume_after_interaction():
    global mission_state, _empty_scans, _avoid_attempts, _scans_since_360
    if mission_active:
        _empty_scans = _avoid_attempts = _scans_since_360 = 0
        mission_state = State.SEARCHING
        motors.pantilt(0, 5)   # ground-looking default
        motors.forward(MOTOR_SPEED_SLOW)
        motors.oled(0, "ERIC ACTIVE")
        motors.oled(1, "Searching...")
        _ui("status", "SEARCHING")


# ─── Prompts ──────────────────────────────────────────────────────────────────

NAV_PROMPT = """Analyze this camera frame and output ONLY the following JSON object. Do not output a list. Do not output markdown. Do not output any text before or after the JSON.

{"wall_ahead":false,"obstacle_close":false,"small_obstacle":false,"person_visible":false,"action":"forward","physical_reasoning":"Path clear."}

Replace the values based on what you see. Rules:
- wall_ahead: true only if a wall/large object fills the lower half of frame
- obstacle_close: true if any object is within 60cm directly ahead
- person_visible: true if any person or robot is visible
- action: must be exactly one of: forward, stop, slow"""

SCAN_360_PROMPT = """Output ONLY this JSON, no markdown, no extra fields:
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
  "physical_reasoning": "one sentence",
  "mission_complete": false
}
Rules: object must be one word: person/robot/slipper/shoe/obstacle/wall/clear/unknown. target_visible=true if mission target seen. target_direction and clearest_direction must be: front/left/right/back/unknown. mission_complete=true only when target is in_my_path AND distance is close/nearby."""

QUICK_SCAN_PROMPT = """Output ONLY this JSON, no markdown, no extra fields:
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
  "physical_reasoning": "one sentence",
  "mission_complete": false
}
Rules: object must be one word: person/robot/slipper/shoe/obstacle/wall/clear/unknown. target_visible=true if mission target seen — if object_name matches the target, target_visible MUST be true. distance must be: close/nearby/far. action must be: forward/stop/slow/turn_left/turn_right. mission_complete=true only when target is in_my_path AND distance is close/nearby."""

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
            _ui("log", "👤 Person spotted during nav — approaching before greeting")
            _ui("status", "PERSON SPOTTED")
            # FIX B3: drive toward person before greeting — they could be meters away
            motors.forward(MOTOR_SPEED_SLOW)
            time.sleep(2.5)
            motors.stop()
            time.sleep(0.4)
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

def _is_pitch_black(frame_b64: str, threshold: float = 20.0) -> bool:
    """Return True only if mean luminance is below threshold — genuinely pitch black."""
    try:
        import cv2
        import numpy as np
        import base64
        data  = base64.b64decode(frame_b64)
        arr   = np.frombuffer(data, np.uint8)
        img   = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False
        return float(img.mean()) < threshold
    except Exception:
        return False

def _quick_scan() -> dict:
    """
    Dual camera stable scan while stopped.
    Pan-tilt at ground-looking default (slight downward), then both cameras captured.
    LED only fires if frame is pitch black (luminance < 20).
    """
    motors.pantilt(0, 5)   # ground-looking default — sees objects on floor
    motors.lights(0, 0)    # start with lights off
    time.sleep(0.5)
    frames = []
    pt = capture_frame(CAMERA_PANTILT, 640, 480)
    if pt:
        # Only turn on LED if scene is pitch black
        if _is_pitch_black(pt):
            motors.lights(base=180, head=255)
            time.sleep(0.3)
            pt = capture_frame(CAMERA_PANTILT, 640, 480) or pt
            motors.lights(0, 0)
        frames.append(pt)

    time.sleep(0.2)   # small gap — prevents V4L2 select() timeout when grabbing both cams back-to-back

    wc = capture_frame(CAMERA_WEBCAM, 640, 480)
    if wc:
        if _is_pitch_black(wc):
            motors.lights(base=180, head=255)
            time.sleep(0.3)
            wc = capture_frame(CAMERA_WEBCAM, 640, 480) or wc
            motors.lights(0, 0)
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
    Full 360° scan: 8 body positions × 45° (finer than 4×90°, less overshoot).
    At each position: tilt down to ground level (5°) then up to mid-range (-15°).
    LED only on if pitch black.
    """
    global mission_state
    mission_state = State.SCANNING_360
    _ui("status", "360 SCANNING")
    motors.oled(0, "360 Scan")
    log.info("Starting smart 360 image scan (8×45°)")

    motors.stop()
    motors.lights(0, 0)
    time.sleep(0.5)

    all_frames   = []
    best_spot    = None

    TURN_45_SEC = TURN_90_SEC / 2   # half the 90° time

    for pos in range(8):
        deg = pos * 45
        _ui("log", f"Scanning {deg}°...")
        motors.oled(1, f"Scan {deg}deg")

        # Ground level first (see objects on floor), then mid-range
        for tilt, label in [(5, "ground"), (-15, "mid")]:
            motors.pantilt(0, tilt, 40)
            time.sleep(0.4)

            # Wide frame for overview collection — adaptive LED only if pitch black
            f_pt = _capture_sharp(CAMERA_PANTILT)
            if f_pt:
                if _is_pitch_black(f_pt):
                    motors.lights(base=180, head=255)
                    time.sleep(0.3)
                    f_pt = _capture_sharp(CAMERA_PANTILT) or f_pt
                    motors.lights(0, 0)
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

            if result.get("wall_ahead") or result.get("obstacle_close"):
                log.info(f"Obstacle at {deg}° during 360 scan")

        # Re-centre pan-tilt to ground default before body rotation
        motors.pantilt(0, 5)
        time.sleep(0.3)

        if pos < 7:
            motors.right(MOTOR_SPEED_SLOW)
            time.sleep(TURN_45_SEC)
            motors.stop()
            time.sleep(0.4)

    # ── Re-visit best potential target for a second look ─────────────────────
    if best_spot:
        deg, spot = best_spot
        _ui("log", f"Re-visiting best potential target at {deg}°...")
        steps_back = (8 - (deg // 45)) % 8
        if steps_back > 0:
            motors.right(MOTOR_SPEED_SLOW)
            time.sleep(TURN_45_SEC * steps_back)
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
    motors.lights(128, 255)   # brief celebratory lights, then off below

    # Tilt to ground-looking default to face the target at close range
    motors.pantilt(0, 5)
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



# ─── Approach Target ──────────────────────────────────────────────────────────

def _approach_target():
    """
    Drive toward the target in 2-second steps, scanning after each step.
    Stops when target is in_my_path at close/nearby distance, hits an obstacle,
    or loses sight of the target.
    Sets mission_state to INTERACTING when close enough.
    """
    global mission_state
    _ui("log", "Approaching target...")
    motors.oled(1, "Approaching...")

    for attempt in range(12):       # max ~24s total approach
        if not mission_active:
            break

        motors.forward(MOTOR_SPEED_SLOW)
        time.sleep(2.0)
        motors.stop()
        time.sleep(0.4)

        check = _quick_scan()
        dist  = str(check.get("distance", "far")).lower()

        if check.get("wall_ahead") or check.get("obstacle_close"):
            _ui("log", "Obstacle during approach — arrived or blocked")
            mission_state = State.INTERACTING
            return

        if check.get("in_my_path") or dist in ("close", "nearby"):
            _ui("log", f"Close to target after {attempt+1} steps — INTERACTING")
            mission_state = State.INTERACTING
            return

        if not check.get("target_visible", False):
            _ui("log", "Lost sight of target — resuming search")
            mission_state = State.SEARCHING
            motors.forward(MOTOR_SPEED_SLOW)
            return

    # Ran out of steps — treat as arrived (may just be Cosmos underestimating distance)
    _ui("log", "Approach complete — switching to INTERACTING")
    mission_state = State.INTERACTING


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
        _approach_target()
        return

    # Also trigger approach from a regular quick scan — target may be spotted while searching
    if target_visible and not from_360:
        _empty_scans = 0
        _ui("log", f"Target spotted during scan — approaching")
        _ui("status", "TARGET SPOTTED")
        if target_dir and target_dir not in ("front", "unknown"):
            _face_direction(target_dir)
        _approach_target()
        return

    if in_path and obj in ["person", "robot"]:
        _empty_scans = 0
        motors.stop()
        mission_state = State.INTERACTING
        name = obj_name or obj
        motors.oled(0, name[:16])
        motors.oled(1, "Centering...")
        _ui("status", f"FOUND — {name}")

        # Ground-looking tilt to see person/robot at close range
        motors.pantilt(0, 5)
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