"""
ERIC — Mission Logic

Camera strategy:
  Navigation (moving):  pan-tilt only, single frame, fast NAV_PROMPT
  Scanning  (stopped):  dual camera (pan-tilt + webcam), single stable frame each
  360° scan (stopped):  pan-tilt sweeps ±90° in 30° steps + ONE 180° chassis turn
                        (finer coverage, far less chassis movement than old 8×45° rotation)
  Face/robot centering: pan-tilt only, settle before capture

Stabilization rule:
  Every pantilt_move_wait() includes a settle delay.
  Captures only happen when robot is stopped or pan-tilt has settled.

LED:
  Adaptive — on only when captured frame is dark.

Sensor integration:
  _sensor_context() builds a text summary of LiDAR + OAK-D readings that is
  prepended to every Cosmos nav-check and scan prompt. This gives Cosmos real
  metric ground-truth distances so it reasons accurately rather than guessing
  from visual cues alone.

Nav2 integration:
  _move_forward() uses Nav2 send_goal() when available, falling back to direct
  motor control. Cosmos still decides WHERE to go — Nav2 handles HOW.

Async Cosmos:
  _cosmos_frames_async() submits Cosmos calls to a ThreadPoolExecutor so the
  mission loop can keep doing sensor checks while Cosmos is thinking.

Multi-step missions:
  Briefing is parsed by Cosmos into MissionStep objects at start.
  Each step has a target + action type (find_and_approach, deliver_message,
  speak_to, wait_for_response, photograph). Steps advance sequentially.
  Mission only ends after ALL steps are complete.

Eye-contact greeting:
  Persons are only greeted when Cosmos confirms they are close AND facing Eric.

Terrain speed control:
  TERRAIN_SPEED_MAP maps terrain strings to motor speeds. Impassable terrain
  (stairs, gaps, walls) triggers the full avoidance pipeline.

Logging:
  All AI calls, motor actions, and mission events are logged via logger.
"""

import time
import threading
import logging
import json
import math
import pathlib
import datetime
import dataclasses
import concurrent.futures
import requests

from typing import Optional

from config import MOTOR_SPEED_SLOW, MOTOR_SPEED_NORMAL, MOTOR_SPEED_FAST, MISSIONS_DIR, VLLM_URL, COSMOS_MODEL
from motors import motors
from cosmos import (
    ask_cosmos, set_mission_briefing, get_mission_briefing,
    capture_frame, capture_frames_video,
    CAMERA_WEBCAM, CAMERA_PANTILT
)
from tts import speak
from logger import (
    log_ai, log_action, log_mission_event,
    start_mission_log, end_mission_log, log_exception
)

log = logging.getLogger("eric.mission")

# ─── Async Cosmos executor ────────────────────────────────────────────────────
# Max 2 workers: one for nav checks, one for scan analysis.
# This lets the mission loop keep running sensor checks while Cosmos is thinking.
_cosmos_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="cosmos"
)


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

_empty_scans          = 0
_avoid_attempts       = 0
_scans_since_360      = 0
_target_spotted_count = 0   # consecutive scans that saw the target — resets on miss
EMPTY_SCAN_LIMIT      = 1   # trigger 360 after just 1 empty scan
SCANS_BEFORE_360      = 2   # periodic 360 every 2 quick scans
MAX_AVOID_ATTEMPTS    = 3   # force 360 after this many avoid failures
TARGET_CONFIRM_NEEDED = 1   # only needs 1 positive scan to approach

_ui_callbacks = {"eric_says": None, "status": None, "log": None}


# ─── Terrain Speed Map ────────────────────────────────────────────────────────
# None = impassable → triggers full avoidance pipeline + spoken warning
TERRAIN_SPEED_MAP: dict[str, float | None] = {
    # Fast — smooth flat surfaces
    "road":         MOTOR_SPEED_FAST,
    "floor":        MOTOR_SPEED_FAST,
    "tile":         MOTOR_SPEED_FAST,
    "tiles":        MOTOR_SPEED_FAST,
    "pavement":     MOTOR_SPEED_FAST,
    "concrete":     MOTOR_SPEED_FAST,
    "asphalt":      MOTOR_SPEED_FAST,
    "hardwood":     MOTOR_SPEED_FAST,
    "linoleum":     MOTOR_SPEED_FAST,
    "wood":         MOTOR_SPEED_FAST,
    "smooth":       MOTOR_SPEED_FAST,

    # Medium — outdoor traversable ground
    "grass":        MOTOR_SPEED_NORMAL,
    "lawn":         MOTOR_SPEED_NORMAL,
    "gravel":       MOTOR_SPEED_NORMAL,
    "dirt":         MOTOR_SPEED_NORMAL,
    "soil":         MOTOR_SPEED_NORMAL,
    "sand":         MOTOR_SPEED_NORMAL,
    "path":         MOTOR_SPEED_NORMAL,
    "clear":        MOTOR_SPEED_NORMAL,
    "flat":         MOTOR_SPEED_NORMAL,
    "ground":       MOTOR_SPEED_NORMAL,

    # Slow — rough, soft, or mildly risky
    "carpet":       MOTOR_SPEED_SLOW,
    "rug":          MOTOR_SPEED_SLOW,
    "mat":          MOTOR_SPEED_SLOW,
    "mud":          MOTOR_SPEED_SLOW,
    "wet":          MOTOR_SPEED_SLOW,
    "rocks":        MOTOR_SPEED_SLOW,
    "rocky":        MOTOR_SPEED_SLOW,
    "pebbles":      MOTOR_SPEED_SLOW,
    "slope":        MOTOR_SPEED_SLOW,   # shallow slope / ramp
    "ramp":         MOTOR_SPEED_SLOW,
    "step":         MOTOR_SPEED_SLOW,   # single small step / curb
    "curb":         MOTOR_SPEED_SLOW,
    "leaves":       MOTOR_SPEED_SLOW,
    "threshold":    MOTOR_SPEED_SLOW,
    "uneven":       MOTOR_SPEED_SLOW,
    "rough":        MOTOR_SPEED_SLOW,
    "bumpy":        MOTOR_SPEED_SLOW,

    # Impassable — stop and navigate around
    "stairs":       None,
    "staircase":    None,
    "steps":        None,
    "wall":         None,
    "fence":        None,
    "water":        None,
    "gap":          None,
    "cliff":        None,
    "ledge":        None,
    "deep_slope":   None,
    "steep":        None,
    "blockade":     None,
    "barrier":      None,
    "curbs":        None,   # plural = raised road barrier
}


def _speed_for_terrain(terrain: str) -> float | None:
    """
    Return target speed for a terrain string, or None if impassable.
    Fuzzy-matches Cosmos inventions like 'rough_grass' or 'wet tiles'.
    Falls back to MOTOR_SPEED_NORMAL for genuinely unknown terrain.
    """
    t = str(terrain).lower().strip() if terrain else "clear"
    if t in TERRAIN_SPEED_MAP:
        return TERRAIN_SPEED_MAP[t]
    # Partial keyword scan — longer keys first to avoid spurious short matches
    for key in sorted(TERRAIN_SPEED_MAP, key=len, reverse=True):
        if key in t:
            log.debug(f"Terrain '{t}' → fuzzy match '{key}'")
            return TERRAIN_SPEED_MAP[key]
    log.debug(f"Unknown terrain '{t}' — defaulting to NORMAL speed")
    return MOTOR_SPEED_NORMAL


# ─── Mission Step Engine ──────────────────────────────────────────────────────

@dataclasses.dataclass
class MissionStep:
    step_num:    int
    target:      str          # e.g. "Princess Leia", "R2-D2", "deer"
    action:      str          # see ACTION_TYPES below
    message:     str = ""     # text for deliver_message / speak_to
    photo_count: int = 1      # number of sharp photos to capture
    wait_sec:    int = 20     # seconds to wait for a response
    completed:   bool = False

# Valid action types:
#   find_and_approach  — get close, mark done (default)
#   deliver_message    — speak step.message to target, then advance
#   speak_to           — initiate conversation, wait wait_sec for reply
#   wait_for_response  — just wait wait_sec for target to say something
#   photograph         — save photo_count sharp close-range photos to disk

_mission_steps:     list[MissionStep] = []
_current_step_idx:  int = 0


def register_ui_callbacks(**cbs):
    _ui_callbacks.update(cbs)


def _ui(key, text):
    cb = _ui_callbacks.get(key)
    if cb:
        try: cb(text)
        except Exception: pass


def eric_say(text):
    _ui("eric_says", text)
    log_mission_event("eric_say", text[:120])
    # Truncate to 2 sentences max — long Cosmos responses block TTS for too long
    sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    short = ". ".join(sentences[:2])
    if short:
        short += "."
    speak(short or text)


# ─── Async Cosmos Wrapper ─────────────────────────────────────────────────────

def _cosmos_frames(frames, prompt, max_tokens=250, temp=0.3):
    """Synchronous Cosmos call with logging. Used directly or via async wrapper."""
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
    response = r.json()["choices"][0]["message"]["content"].strip()
    log_ai(prompt[-400:], response, label="COSMOS_FRAMES")
    return response


def _cosmos_frames_async(frames, prompt, max_tokens=250, temp=0.3) -> concurrent.futures.Future:
    """
    Submit Cosmos vision call to thread pool. Returns a Future immediately.
    Call future.result(timeout=60) when you actually need the answer.
    This lets the mission loop keep doing sensor checks while Cosmos is thinking.
    """
    return _cosmos_executor.submit(_cosmos_frames, frames, prompt, max_tokens, temp)


# ─── Mission Step Helpers ─────────────────────────────────────────────────────

def _parse_mission_steps(briefing: str) -> list[MissionStep]:
    """
    Ask Cosmos to parse the mission briefing into an ordered list of MissionStep objects.
    Falls back to a single find_and_approach step if parsing fails.
    """
    prompt = f"""You are parsing a robot mission briefing into structured, ordered steps.

BRIEFING:
\"\"\"{briefing}\"\"\"

Extract each discrete task as a step. Return ONLY a JSON array.

Valid action types:
  "find_and_approach"  — find the target and get within close range
  "deliver_message"    — speak a specific message to the target when close
  "speak_to"           — start a conversation with the target, wait for reply
  "wait_for_response"  — wait for the target to say something (use wait_sec)
  "photograph"         — take sharp close-range photos of the target (use photo_count)

JSON schema per step:
{{
  "step_num":    1,
  "target":      "Princess Leia",
  "action":      "deliver_message",
  "message":     "Help me Obi-Wan, you're my only hope",
  "photo_count": 1,
  "wait_sec":    20
}}

Example for multi-step mission:
[
  {{"step_num": 1, "target": "Princess Leia", "action": "deliver_message",
    "message": "Help me Obi-Wan, you're my only hope", "photo_count": 1, "wait_sec": 20}},
  {{"step_num": 2, "target": "R2-D2", "action": "speak_to",
    "message": "", "photo_count": 1, "wait_sec": 30}},
  {{"step_num": 3, "target": "deer", "action": "photograph",
    "message": "", "photo_count": 3, "wait_sec": 10}}
]

Return ONLY the JSON array. No markdown. No explanation. No extra text.
"""
    try:
        raw   = ask_cosmos(prompt, max_tokens=500)
        log_ai(prompt[-300:], raw, label="STEP_PARSE")
        clean = raw.replace("```json", "").replace("```", "").strip()
        s = clean.find("["); e = clean.rfind("]") + 1
        items = json.loads(clean[s:e])
        steps = []
        for i, it in enumerate(items):
            steps.append(MissionStep(
                step_num    = int(it.get("step_num",    i + 1)),
                target      = str(it.get("target",      "target")),
                action      = str(it.get("action",      "find_and_approach")),
                message     = str(it.get("message",     "")),
                photo_count = int(it.get("photo_count", 1)),
                wait_sec    = int(it.get("wait_sec",    20)),
            ))
        log.info(f"Parsed {len(steps)} mission steps: {[s.target for s in steps]}")
        return steps
    except Exception as e:
        log_exception("_parse_mission_steps", e)
        return [MissionStep(step_num=1, target="target", action="find_and_approach")]


def _current_step() -> Optional[MissionStep]:
    if _mission_steps and _current_step_idx < len(_mission_steps):
        return _mission_steps[_current_step_idx]
    return None


def _advance_step():
    """Mark the current step complete and move to the next, or end the mission."""
    global _current_step_idx
    step = _current_step()
    if step:
        step.completed = True
        log_mission_event(f"step_{step.step_num}_complete", f"{step.target} — {step.action}")

    _current_step_idx += 1

    if _current_step_idx >= len(_mission_steps):
        # All steps done
        last_target = step.target if step else "all targets"
        _handle_mission_complete(last_target)
    else:
        nxt = _current_step()
        msg = f"Step {_current_step_idx} complete. Now finding {nxt.target}."
        eric_say(msg)
        _ui("status", f"STEP {nxt.step_num}: {nxt.target.upper()}")
        _ui("log", msg)
        # Update Cosmos system prompt so it searches for the next target
        set_mission_briefing(
            f"CURRENT STEP {nxt.step_num} of {len(_mission_steps)}: "
            f"Find {nxt.target} and {nxt.action.replace('_', ' ')}.\n"
            f"Original mission: {get_mission_briefing()}"
        )
        # Resume searching
        global mission_state, _empty_scans, _avoid_attempts, _scans_since_360, _target_spotted_count
        _empty_scans = _avoid_attempts = _scans_since_360 = _target_spotted_count = 0
        mission_state = State.SEARCHING
        motors.forward(MOTOR_SPEED_SLOW)


def _execute_step_action(obj_name: str):
    """
    Called when Eric arrives at the current step's target.
    Executes the required action (speak, photograph, wait, etc.) then advances.
    """
    global mission_state
    step = _current_step()
    if not step:
        _handle_mission_complete(obj_name)
        return

    mission_state = State.INTERACTING
    motors.stop()
    log_mission_event("step_arrived", f"step={step.step_num} target={step.target} action={step.action}")
    log.info(f"Executing step {step.step_num}: {step.action} for {step.target}")

    if step.action == "find_and_approach":
        _advance_step()

    elif step.action == "deliver_message":
        msg = step.message or f"Message delivered to {step.target}."
        eric_say(msg)
        log_mission_event("message_delivered", f"to={step.target}: {msg}")
        motors.oled(0, "Delivering msg")
        motors.oled(1, step.target[:16])
        time.sleep(min(step.wait_sec, 10))
        _advance_step()

    elif step.action == "speak_to":
        greeting = ask_cosmos(
            f"You have found {step.target}. "
            + (f"Your mission: {step.message}. " if step.message else "")
            + "Greet them warmly and start the conversation. 2 sentences.",
            max_tokens=120
        )
        eric_say(greeting)
        log_mission_event("spoke_to", f"{step.target}: {greeting[:80]}")
        motors.oled(0, f"Talking to")
        motors.oled(1, step.target[:16])
        _ui("log", f"Waiting {step.wait_sec}s for {step.target} to respond...")
        time.sleep(step.wait_sec)
        _advance_step()

    elif step.action == "wait_for_response":
        eric_say(f"Waiting for {step.target} to respond.")
        motors.oled(0, "Waiting...")
        motors.oled(1, step.target[:16])
        _ui("log", f"Waiting up to {step.wait_sec}s for {step.target} to speak...")
        time.sleep(step.wait_sec)
        _advance_step()

    elif step.action == "photograph":
        eric_say(f"I will take {step.photo_count} photo{'s' if step.photo_count > 1 else ''} of {step.target}.")
        motors.oled(0, "Taking photos")
        motors.oled(1, step.target[:16])
        photos_taken = 0
        max_attempts = step.photo_count * 4

        for attempt in range(max_attempts):
            if photos_taken >= step.photo_count:
                break
            frame = capture_frame(CAMERA_PANTILT, 1280, 720)
            if frame and not _is_blurry(frame):
                import base64 as _b64
                ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
                fname = f"photo_{step.target.replace(' ', '_')}_{photos_taken + 1}_{ts}.jpg"
                out   = pathlib.Path("missions/photos") / fname
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(_b64.b64decode(frame))
                photos_taken += 1
                _ui("log", f"📸 Photo {photos_taken}/{step.photo_count} saved: {fname}")
                log_mission_event("photo_saved", fname)
                motors.oled(1, f"Photo {photos_taken}/{step.photo_count}")
                time.sleep(0.8)
            else:
                time.sleep(0.4)

        completion_msg = f"Captured {photos_taken} of {step.photo_count} photo(s) of {step.target}."
        eric_say(completion_msg)
        log_mission_event("photograph_done", completion_msg)
        _advance_step()

    else:
        log.warning(f"Unknown step action '{step.action}' — advancing")
        _advance_step()


def _parse_json(response, fallback, label="COSMOS"):
    try:
        clean = response.replace("```json", "").replace("```python", "").replace("```", "").strip()

        # ── Handle JSON array — Cosmos sometimes returns [{...}, {...}] ───────
        # Merge all items: pick the highest-priority object across all entries,
        # collect all object_names, and OR all boolean flags together.
        arr_start = clean.find("[")
        obj_start = clean.find("{")
        if arr_start >= 0 and (obj_start < 0 or arr_start < obj_start):
            arr_end = clean.rfind("]") + 1
            if arr_end > arr_start:
                items = json.loads(clean[arr_start:arr_end])
                if isinstance(items, list) and items:
                    result = _merge_array_items(items, fallback)
                    # skip to normalization below
                    return _finalize_result(result, fallback, label)

        # ── Normal single-object JSON ─────────────────────────────────────────
        s = clean.find("{")
        e = clean.rfind("}") + 1
        if s >= 0 and e > s:
            result = json.loads(clean[s:e])
            return _finalize_result(result, fallback, label)

    except Exception:
        log.warning(f"JSON parse failed: {response[:100]}")
        print(f"\n⚠️  RAW RESPONSE ({label}): {response[:400]}\n")
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
    # robots — broad coverage for Cosmos inventions
    "droid": "robot", "robot": "robot", "r2": "robot", "bb8": "robot",
    "toy_droid": "robot", "toy_robot": "robot", "toy droid": "robot",
    "mech": "robot", "android": "robot", "bot": "robot",
    # walls / structural
    "wall": "wall", "door": "wall", "fence": "wall",
}

# Non-standard object strings Cosmos invents that map to canonical categories.
# Applied in _finalize_result regardless of whether object is "unknown".
_OBJ_REMAP = {
    "toy_droid": "robot", "toy_robot": "robot", "toy droid": "robot",
    "toy robot": "robot", "droid": "robot", "android": "robot",
    "mech": "robot", "bot": "robot",
    "sandal": "slipper", "flip_flop": "slipper", "flip flop": "slipper",
    "sneaker": "shoe", "boot": "shoe",
    "human": "person", "man": "person", "woman": "person",
    "kid": "person", "child": "person",
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

    # ── Remap non-standard object strings Cosmos invents ─────────────────────
    raw_obj = str(result.get("object", "unknown")).lower().strip()
    if raw_obj in _OBJ_REMAP:
        log.info(f"Remapping object '{raw_obj}' → '{_OBJ_REMAP[raw_obj]}'")
        result["object"] = _OBJ_REMAP[raw_obj]
    elif "_" in raw_obj or " " in raw_obj:
        for key, val in _OBJ_REMAP.items():
            if key in raw_obj:
                log.info(f"Remapping object '{raw_obj}' → '{val}' (partial match '{key}')")
                result["object"] = val
                break

    # ── Normalize action to canonical set ────────────────────────────────────
    _VALID_ACTIONS = {"forward", "backward", "left", "right", "slow",
                      "stop", "navigate_around", "turn_left", "turn_right", "turn_back"}
    raw_action = str(result.get("action", "forward")).lower().strip()
    if raw_action not in _VALID_ACTIONS:
        _ACTION_MAP = {
            "move_forward": "forward", "go_forward": "forward", "continue": "forward",
            "move": "forward", "proceed": "forward", "advance": "forward",
            "go": "forward", "drive": "forward", "go_ahead": "forward",
            "turn": "turn_right", "avoid": "navigate_around", "reverse": "backward",
            "back_up": "backward", "back": "backward", "halt": "stop", "pause": "stop",
        }
        normalized = _ACTION_MAP.get(raw_action)
        if not normalized:
            normalized = "forward" if "forward" in raw_action else "stop"
        log.info(f"Normalized action '{raw_action}' → '{normalized}'")
        result["action"] = normalized

    # ── Consistency fix: if object IS the target, target_visible must be True ──
    _TARGET_OBJECTS = {"slipper", "shoe", "person", "robot"}
    if result.get("object") in _TARGET_OBJECTS and not result.get("target_visible"):
        log.info(f"Auto-correcting target_visible=True (object={result['object']})")
        result["target_visible"] = True

    # ── Consistency fix: action=stop requires an obstacle reason ─────────────
    if (result.get("action") == "stop"
            and not result.get("wall_ahead")
            and not result.get("obstacle_close")
            and not result.get("in_my_path")):
        log.info("Auto-correcting action: stop→forward (no obstacle present)")
        result["action"] = "forward"

    # Stringify any remaining list/dict in string fields
    for field in ("terrain", "distance", "target_direction",
                  "clearest_direction", "action", "physical_reasoning"):
        val = result.get(field)
        if isinstance(val, (list, dict)):
            result[field] = str(val)

    # Fill missing keys from fallback
    for k, v in fallback.items():
        result.setdefault(k, v)

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


# ─── Sensor Context ───────────────────────────────────────────────────────────

def _sensor_context() -> str:
    """
    Build a short sensor data summary to prepend to every Cosmos prompt.

    Pulls live readings from:
      - D500 LiDAR  (lidar.py) — front arc minimum distance
      - OAK-D Lite  (oakd.py)  — stereo depth at center-forward

    Returns a text block like:
      SENSOR DATA (ground truth — trust this over visual estimates):
      LiDAR front arc minimum distance: 0.42m
      OAK-D depth center-forward: 0.38m
      ⚠️  OAK-D: obstacle within caution range

    This context is prepended to nav-check and scan prompts so Cosmos
    gets real metric distances. Cosmos can then say "obstacle at 0.4m"
    rather than guessing "close" from a wide-angle camera image.

    Fails silently — if both sensors are unavailable, returns "".
    """
    lines = []

    # ── LiDAR ──────────────────────────────────────────────────────────────
    try:
        from lidar import get_status as lidar_status, lidar_available
        if lidar_available():
            ls = lidar_status()
            dist = ls.get("min_distance", 999)
            dist_str = f"{dist:.2f}m" if dist < 999 else "clear"
            lines.append(f"LiDAR front arc minimum distance: {dist_str}")
            if ls.get("obstacle_close"):
                lines.append("⚠️  LIDAR STOP ZONE: obstacle within 0.30m — do NOT move forward")
            elif ls.get("obstacle_near"):
                lines.append("⚠️  LiDAR caution: obstacle within 0.60m — slow or stop")
    except Exception:
        pass

    # ── OAK-D depth ────────────────────────────────────────────────────────
    try:
        from oakd import get_front_depth, oakd_available
        if oakd_available():
            d = get_front_depth()
            if d is not None:
                lines.append(f"OAK-D depth center-forward: {d:.2f}m")
                if d < 0.30:
                    lines.append("⚠️  OAK-D: obstacle VERY CLOSE (<0.30m) — stop immediately")
                elif d < 0.60:
                    lines.append("⚠️  OAK-D: obstacle within caution range (<0.60m) — slow down")
    except Exception:
        pass

    if not lines:
        return ""

    return (
        "SENSOR DATA (ground truth — trust this over visual estimates):\n"
        + "\n".join(lines)
        + "\n\n"
    )


# ─── Nav2 / Motor Movement Abstraction ───────────────────────────────────────

def _move_forward(duration_sec: float = 2.0, distance_m: float = 1.5):
    """
    Move Eric forward using Nav2 if available, else direct motor control.

    Nav2 path:
      - Compute a goal pose distance_m ahead of current pose in map frame
      - send_goal() hands off to Nav2's planner — it avoids obstacles on its own
      - Wait for Nav2 to reach goal or timeout

    Direct path:
      - motors.forward() for duration_sec, then stop

    This abstraction means mission.py never needs to care which mode is active.
    """
    from config import USE_NAV2
    if USE_NAV2:
        try:
            from nav2 import nav2_available, send_goal, get_pose, is_navigating
            if nav2_available():
                pose = get_pose()
                yaw  = pose["yaw"]
                tx   = pose["x"] + distance_m * math.cos(yaw)
                ty   = pose["y"] + distance_m * math.sin(yaw)
                send_goal(tx, ty, yaw)
                # Wait for Nav2 to finish or timeout
                deadline = time.time() + duration_sec + 8.0
                while is_navigating() and time.time() < deadline and mission_active:
                    time.sleep(0.2)
                return
        except Exception as e:
            log.warning(f"Nav2 move_forward failed ({e}) — falling back to direct")

    # Direct motor fallback
    motors.forward(MOTOR_SPEED_SLOW)
    time.sleep(duration_sec)
    motors.stop()


def _turn_nav2_or_direct(direction: str, duration_sec: float = 1.5):
    """
    Turn Eric using Nav2 (yaw goal) if available, else direct motor control.
    direction: "left" | "right" | "back"
    """
    from config import USE_NAV2
    if USE_NAV2:
        try:
            from nav2 import nav2_available, send_goal, get_pose, is_navigating
            if nav2_available():
                pose = get_pose()
                yaw_delta = {
                    "left":  math.pi / 2,
                    "right": -math.pi / 2,
                    "back":  math.pi,
                }.get(direction, 0.0)
                target_yaw = pose["yaw"] + yaw_delta
                # Stay in place — same x,y, just new heading
                send_goal(pose["x"], pose["y"], target_yaw)
                deadline = time.time() + duration_sec + 5.0
                while is_navigating() and time.time() < deadline and mission_active:
                    time.sleep(0.2)
                return
        except Exception as e:
            log.warning(f"Nav2 turn failed ({e}) — falling back to direct")

    # Direct motor fallback
    if direction == "left":
        motors.left(MOTOR_SPEED_SLOW);  time.sleep(duration_sec); motors.stop()
    elif direction == "right":
        motors.right(MOTOR_SPEED_SLOW); time.sleep(duration_sec); motors.stop()
    elif direction == "back":
        motors.right(MOTOR_SPEED_SLOW); time.sleep(duration_sec * 2); motors.stop()


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
    global _mission_steps, _current_step_idx

    if mission_active:
        return "Mission already active. Disengage first."
    if not briefing.strip():
        return "No mission briefing provided."

    conversation_history = []
    _empty_scans = _avoid_attempts = _scans_since_360 = _target_spotted_count = 0

    # ── Parse mission into steps ──────────────────────────────────────────────
    _mission_steps    = _parse_mission_steps(briefing)
    _current_step_idx = 0
    step_summaries    = [f"[{s.step_num}] {s.target} → {s.action}" for s in _mission_steps]

    # ── Start mission log ─────────────────────────────────────────────────────
    start_mission_log(briefing[:60], steps=step_summaries)
    log_mission_event("mission_start", briefing[:200])

    # ── Update Cosmos system prompt with first step ───────────────────────────
    first_step = _current_step()
    if first_step and first_step.target != "target":
        set_mission_briefing(
            f"CURRENT STEP 1 of {len(_mission_steps)}: "
            f"Find {first_step.target} and {first_step.action.replace('_', ' ')}.\n"
            f"Original mission: {briefing}"
        )
    else:
        set_mission_briefing(briefing)

    motors.pantilt(0, 5)   # slight downward tilt — see ground objects at normal range
    motors.lights(0, 0)    # LEDs off — only turn on if scene is pitch black
    time.sleep(0.5)

    step_info = f"I have {len(_mission_steps)} step{'s' if len(_mission_steps) > 1 else ''}: {', '.join(step_summaries)}." if len(_mission_steps) > 1 else ""
    ack = ask_cosmos(
        f"Mission briefing:\n\"{briefing}\"\n\n"
        + (f"Parsed steps: {step_info}\n\n" if step_info else "")
        + "Acknowledge in 2-3 sentences. State your first action. Be concise.",
        max_tokens=150
    )
    eric_say(ack)
    log_mission_event("mission_acknowledged", ack[:150])

    if len(_mission_steps) > 1:
        _ui("log", f"Multi-step mission: {' → '.join(s.target for s in _mission_steps)}")

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
    # Cancel any in-progress Nav2 goal
    try:
        from config import USE_NAV2
        if USE_NAV2:
            from nav2 import cancel_goal, nav2_available
            if nav2_available():
                cancel_goal()
    except Exception:
        pass
    motors.lights(0, 0)
    motors.pantilt(0, 5)
    motors.oled(0, "ERIC STOPPED")
    motors.oled(1, "")
    eric_say("Mission disengaged. All systems halted.")
    log_mission_event("mission_stopped", "operator abort")
    end_mission_log(completed=False)
    _ui("status", "IDLE")


def resume_after_interaction():
    global mission_state, _empty_scans, _avoid_attempts, _scans_since_360, _target_spotted_count
    if mission_active:
        _empty_scans = _avoid_attempts = _scans_since_360 = _target_spotted_count = 0
        mission_state = State.SEARCHING
        motors.pantilt(0, 5)   # ground-looking default
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

STEP 1 — OBSTACLE CHECK (lower half of every image):
- Anything filling/touching the bottom edge → wall_ahead = true
- Object within ~60cm directly ahead → obstacle_close = true AND in_my_path = true
- When in doubt → obstacle_close = true

STEP 2 — MISSION TARGET CHECK:
- If you set object to "slipper", "shoe", "person", or "robot" → you MUST set target_visible = true
- Any slipper, shoe, person, or robot visible anywhere in any frame → target_visible = true
- "far" distance does NOT mean target_visible = false — set it true even if far away
- target_visible = false ONLY if none of those objects appear anywhere

STEP 3 — ACTION:
- action = "stop" is ONLY valid when wall_ahead=true OR obstacle_close=true
- If path is clear and no obstacle: action = "forward"
- If target visible and no obstacle blocking: action = "forward"

CONSISTENCY RULE: If object is "slipper" or "shoe" or "person" or "robot", then target_visible MUST be true. These two fields must agree.

OUTPUT: A single JSON object. Every field is REQUIRED.
"object": one word from: person, robot, slipper, shoe, obstacle, wall, clear, unknown
"object_name": short string or null
Boolean fields: true or false only.

Example when target found:
{
  "object": "slipper",
  "object_name": "blue slipper",
  "terrain": "tiles",
  "distance": "far",
  "in_my_path": false,
  "wall_ahead": false,
  "obstacle_close": false,
  "small_obstacle": false,
  "target_visible": true,
  "target_direction": "front",
  "clearest_direction": "front",
  "action": "forward",
  "speak": "I can see a slipper ahead!",
  "physical_reasoning": "Slipper visible in frame — moving toward it.",
  "social_intent": "any detected social cues or null",
  "risk_assessment": "collision or hazard risk description or null"
  "mission_complete": false
}

Example when path is clear:
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
  "target_direction": "unknown",
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
    Single pan-tilt frame + live sensor data every NAV_IMAGE_INTERVAL seconds.
    LiDAR and OAK-D readings are prepended to the prompt as ground-truth context.
    Much faster than 10s video clip — allows more frequent obstacle checks.

    Eye-contact gate: persons are only greeted when Cosmos confirms they are
    CLOSE (within ~1.5m) AND facing Eric. Eliminates random greetings at people
    across the room who happen to be in frame.
    """
    _ui("log", "📷 Nav check...")
    motors.oled(1, "Nav check...")

    # ── Hardware safety check BEFORE asking Cosmos ──────────────────────────
    try:
        from lidar import obstacle_close as lidar_close, obstacle_near as lidar_near
        if lidar_close():
            log.info("Nav check: LiDAR STOP — returning obstacle result immediately")
            log_action("LIDAR_STOP", "obstacle within 0.30m stop zone")
            return {**_NAV_FALLBACK, "wall_ahead": True, "obstacle_close": True,
                    "action": "stop",
                    "physical_reasoning": "LiDAR: obstacle within 0.30m stop zone"}
        if lidar_near():
            log.info("Nav check: LiDAR near — slowing")
    except Exception:
        pass

    try:
        from oakd import get_front_depth, oakd_available
        if oakd_available():
            d = get_front_depth()
            if d is not None and d < 0.30:
                log.info(f"Nav check: OAK-D STOP at {d:.2f}m")
                log_action("OAKD_STOP", f"obstacle at {d:.2f}m")
                return {**_NAV_FALLBACK, "wall_ahead": True, "obstacle_close": True,
                        "action": "stop",
                        "physical_reasoning": f"OAK-D: obstacle at {d:.2f}m — within stop distance"}
    except Exception:
        pass

    frame = capture_frame(CAMERA_PANTILT, 320, 240)
    if not frame:
        return dict(_NAV_FALLBACK)

    sensor_ctx = _sensor_context()

    NAV_IMAGE_PROMPT = f"""{sensor_ctx}You are a tracked ground robot moving forward. This is a single frame from your forward camera.

Check ONLY for immediate safety hazards:
- Wall or large object filling the lower 40% of frame → wall_ahead = true
- Any object within ~60cm directly ahead → obstacle_close = true
- Small ground obstacle (cables, edges, steps) → small_obstacle = true
- Person or robot visible anywhere → person_visible = true

If sensor data above shows obstacle_close or LiDAR STOP ZONE, you MUST set wall_ahead=true and action=stop.

ACTION RULE: "action" must be EXACTLY one of these two words: "forward" or "stop"
- Use "stop" ONLY if wall_ahead=true OR obstacle_close=true
- Use "forward" in all other cases
- NEVER use "move_forward", "continue", "go", or any other value

OUTPUT: A single JSON object. Every field is REQUIRED. Use ONLY the exact field names shown.
BOOLEAN fields must be true or false.

Example output (copy this structure exactly, change values to match what you see):
{{
  "wall_ahead": false,
  "obstacle_close": false,
  "small_obstacle": false,
  "person_visible": false,
  "action": "forward",
  "physical_reasoning": "Path ahead is clear with no obstacles visible."
}}

Now analyze the frame and output ONLY the JSON object above. No markdown. No explanation. No extra fields.
"""
    try:
        payload = {
            "model": COSMOS_MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}},
                {"type": "text", "text": NAV_IMAGE_PROMPT.strip()}
            ]}],
            "max_tokens": 120,
            "temperature": 0.1,
            "repetition_penalty": 1.15,
        }
        r = requests.post(VLLM_URL, json=payload, timeout=30)
        r.raise_for_status()
        response = r.json()["choices"][0]["message"]["content"].strip()
        log_ai(NAV_IMAGE_PROMPT[-200:], response, label="NAV_CHECK")
        result = _parse_json(response, dict(_NAV_FALLBACK), label="NAV CHECK")

        # ── Eye-contact gate: only greet if person is close AND facing Eric ──
        if result.get("person_visible") and mission_active:
            motors.stop()
            _ui("log", "👤 Person spotted — checking proximity and eye contact...")

            ec_prompt = """Is there a person in this frame who is BOTH:
1. CLOSE to you (within approximately 1.5 meters), AND
2. FACING toward you (their face or body is oriented toward the camera)?

Answer ONLY with this JSON — no markdown, no extra text:
{"close_and_facing": true_or_false, "reasoning": "one sentence"}

Set close_and_facing=false if the person is far away, looking away, or has their back to you.
"""
            ec_frame = capture_frame(CAMERA_PANTILT, 320, 240)
            ec_result = {"close_and_facing": False}
            if ec_frame:
                try:
                    ec_payload = {
                        "model": COSMOS_MODEL,
                        "messages": [{"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ec_frame}"}},
                            {"type": "text", "text": ec_prompt}
                        ]}],
                        "max_tokens": 60,
                        "temperature": 0.1,
                    }
                    ec_r = requests.post(VLLM_URL, json=ec_payload, timeout=20)
                    ec_r.raise_for_status()
                    ec_raw = ec_r.json()["choices"][0]["message"]["content"].strip()
                    log_ai(ec_prompt, ec_raw, label="EYE_CONTACT")
                    ec_result = _parse_json(ec_raw, {"close_and_facing": False}, "EYE CONTACT")
                except Exception as e:
                    log_exception("eye_contact_check", e)

            if ec_result.get("close_and_facing"):
                _ui("log", f"👁️  Eye contact confirmed — greeting! ({ec_result.get('reasoning','')})")
                _ui("status", "PERSON SPOTTED")
                log_mission_event("person_greeted", ec_result.get("reasoning", ""))
                greeting = ask_cosmos(
                    "Someone is looking at you from close range. "
                    "Greet them warmly and ask if they can help with your mission. 1-2 sentences.",
                    max_tokens=60
                )
                eric_say(greeting)
            else:
                _ui("log", f"Person spotted but not close/facing ({ec_result.get('reasoning','')}) — continuing")
                motors.forward(MOTOR_SPEED_SLOW)

        return result
    except Exception as e:
        log_exception("_nav_check", e)
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
    Sensor context prepended to prompt — Cosmos gets LiDAR + OAK-D metric readings.
    LED only fires if frame is pitch black (luminance < 20).
    """
    motors.pantilt(0, 5)   # ground-looking default — sees objects on floor
    motors.lights(0, 0)    # start with lights off
    time.sleep(0.5)
    frames = []
    pt = capture_frame(CAMERA_PANTILT, 640, 480)
    if pt:
        if _is_pitch_black(pt):
            motors.lights(base=180, head=255)
            time.sleep(0.3)
            pt = capture_frame(CAMERA_PANTILT, 640, 480) or pt
            motors.lights(0, 0)
        frames.append(pt)
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

    sensor_ctx = _sensor_context()
    prompt = sensor_ctx + QUICK_SCAN_PROMPT if sensor_ctx else QUICK_SCAN_PROMPT

    try:
        print(f"\n📷 QUICK SCAN — {len(frames)} frames to Cosmos...")
        response = _cosmos_frames(frames, prompt, max_tokens=200, temp=0.3)
        result = _parse_json(response, dict(_SCAN_FALLBACK), label="QUICK SCAN RESULT")

        # ── Sensor override ──────────────────────────────────────────────────
        try:
            from lidar import obstacle_close as lidar_close
            if lidar_close():
                log.info("Quick scan: LiDAR override → wall_ahead=True")
                log_action("LIDAR_OVERRIDE", "quick scan")
                result["wall_ahead"]     = True
                result["obstacle_close"] = True
                result["action"]         = "stop"
        except Exception:
            pass

        try:
            from oakd import get_front_depth, oakd_available
            if oakd_available():
                d = get_front_depth()
                if d is not None and d < 0.30:
                    log.info(f"Quick scan: OAK-D override at {d:.2f}m → wall_ahead=True")
                    log_action("OAKD_OVERRIDE", f"quick scan at {d:.2f}m")
                    result["wall_ahead"]     = True
                    result["obstacle_close"] = True
                    result["action"]         = "stop"
        except Exception:
            pass

        return result
    except Exception as e:
        log_exception("_quick_scan", e)
        return dict(_SCAN_FALLBACK)


# ─── 360° Scan (stopped) ──────────────────────────────────────────────────────

TURN_90_SEC      = 2.2   # seconds to turn 90° at MOTOR_SPEED_SLOW — tune if needed
BLUR_THRESHOLD   = 80.0  # Laplacian variance below this = blurry, retry
MAX_BLUR_RETRIES = 3

# Video scan settings — used during 360° scan positions (Eric is stopped)
VIDEO_SCAN_DURATION = 3.0   # seconds per position — short enough to keep 360 moving
VIDEO_SCAN_FPS      = 2.0   # frames/sec → 6 frames per position


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


def _video_scan_at_position() -> dict:
    """
    Short video clip scan at a single 360° position.
    Eric is stopped — captures VIDEO_SCAN_DURATION seconds of footage.
    Cosmos reasons about temporal changes across frames (things getting closer,
    motion, etc.) which is more informative than a single frame.
    Falls back to _quick_scan() if video capture fails.
    """
    _ui("log", "🎬 Video scan...")
    motors.lights(0, 0)

    frames = capture_frames_video(CAMERA_PANTILT,
                                  duration=VIDEO_SCAN_DURATION,
                                  fps_sample=VIDEO_SCAN_FPS)
    if not frames:
        log.warning("Video capture returned no frames — falling back to quick scan")
        return _quick_scan()

    # Also grab one webcam frame for extra context
    wc = capture_frame(CAMERA_WEBCAM, 320, 240)
    if wc:
        frames.append(wc)

    # LED check on last frame
    if frames and _is_pitch_black(frames[-1]):
        motors.lights(base=180, head=255)
        time.sleep(0.3)
        frames = capture_frames_video(CAMERA_PANTILT,
                                      duration=VIDEO_SCAN_DURATION,
                                      fps_sample=VIDEO_SCAN_FPS) or frames
        motors.lights(0, 0)

    sensor_ctx = _sensor_context()
    prompt = sensor_ctx + QUICK_SCAN_PROMPT if sensor_ctx else QUICK_SCAN_PROMPT

    try:
        print(f"\n🎬 VIDEO SCAN — {len(frames)} frames to Cosmos...")
        response = _cosmos_frames(frames, prompt, max_tokens=200, temp=0.3)
        result = _parse_json(response, dict(_SCAN_FALLBACK), label="VIDEO SCAN RESULT")

        # Sensor overrides — same as _quick_scan
        try:
            from lidar import obstacle_close as lidar_close
            if lidar_close():
                result["wall_ahead"] = True
                result["obstacle_close"] = True
                result["action"] = "stop"
        except Exception:
            pass

        try:
            from oakd import get_front_depth, oakd_available
            if oakd_available():
                d = get_front_depth()
                if d is not None and d < 0.30:
                    result["wall_ahead"] = True
                    result["obstacle_close"] = True
                    result["action"] = "stop"
        except Exception:
            pass

        return result
    except Exception as e:
        log.error(f"Video scan error: {e} — falling back to quick scan")
        return _quick_scan()


def _scan_360_pantilt() -> dict:
    """
    Full 360° scan: pan-tilt sweeps from -90° to +90° in 30° steps (7 positions)
    while chassis stays still, then ONE 180° chassis turn, then sweeps again.

    Coverage:
      Phase 1 — chassis 0°:   pan = -90, -60, -30, 0, +30, +60, +90  (7 stops)
      Phase 2 — chassis 180°: chassis turns 180°
      Phase 3 — chassis 180°: pan = -90, -60, -30, 0, +30, +60, +90  (7 stops)
    = 14 positions × 2 tilts = 28 frames maximum; chassis only rotates once.

    At each pan position: tilt to ground level (TILT_LOW) then mid-range (TILT_MID).
    If target found mid-scan, chassis is turned to face it and scan returns immediately.
    One final Cosmos overview pass is run over all collected frames if no target found.
    """
    global mission_state
    mission_state = State.SCANNING_360
    _ui("status", "360 SCANNING")
    motors.oled(0, "360 Scan")
    motors.stop()
    time.sleep(0.3)
    log.info("Starting pan-tilt 360 scan (7×30° steps + 180° chassis turn)")
    log_mission_event("scan_360_start", "pan-tilt sweep 7×30° + 180° chassis")

    all_frames: list[str] = []

    # Pan positions: 7 stops covering -90° to +90° in 30° increments
    PAN_STEPS  = [-90, -60, -30, 0, 30, 60, 90]
    TILT_LOW   =  10   # ground-looking (positive = down on UGV Beast pan-tilt)
    TILT_MID   = -10   # mid-range / horizon
    PAN_SETTLE = 0.35  # seconds after pantilt() before capture — motor settle time

    def _pan_to_chassis_turn_sec(pan: int) -> float:
        """Estimate chassis turn duration to face a target spotted at pan angle pan."""
        # At ±90° the robot needs a full 90° chassis turn to face forward.
        # We already know TURN_90_SEC for 90° at MOTOR_SPEED_SLOW.
        return abs(pan) / 90.0 * TURN_90_SEC

    def _sweep(phase_label: str) -> dict | None:
        """
        Sweep all PAN_STEPS. Returns target result dict if found, else None.
        Appends captured frames to all_frames.
        """
        for pan in PAN_STEPS:
            if not mission_active:
                return None
            _ui("log", f"{phase_label}: pan {pan:+d}°")
            motors.oled(1, f"Pan {pan:+d}d")

            for tilt, tilt_label in [(TILT_LOW, "ground"), (TILT_MID, "mid")]:
                motors.pantilt(pan, tilt, speed=60)
                time.sleep(PAN_SETTLE)

                # ── Adaptive LED ─────────────────────────────────────────────
                frame = _capture_sharp(CAMERA_PANTILT)
                if not frame:
                    continue
                if _is_pitch_black(frame):
                    motors.lights(base=180, head=255)
                    time.sleep(0.2)
                    frame = _capture_sharp(CAMERA_PANTILT) or frame
                    motors.lights(0, 0)
                all_frames.append(frame)

                # ── Quick video scan at this position ────────────────────────
                result = _video_scan_at_position()

                if result.get("target_visible"):
                    log.info(f"🎯 Target at {phase_label} pan={pan:+d}° tilt={tilt_label}")
                    _ui("log", f"Target visible at pan {pan:+d}° — turning chassis to face it!")
                    log_mission_event("target_found_mid_scan", f"pan={pan} tilt={tilt_label}")
                    motors.oled(1, "TARGET FOUND!")
                    motors.stop()
                    time.sleep(0.2)

                    # Turn chassis to face the target (pan angle → chassis turn)
                    if pan < -15:
                        turn_sec = _pan_to_chassis_turn_sec(pan)
                        _ui("log", f"Turning left {turn_sec:.1f}s to face target at pan {pan}°")
                        motors.left(MOTOR_SPEED_SLOW)
                        time.sleep(turn_sec)
                        motors.stop()
                    elif pan > 15:
                        turn_sec = _pan_to_chassis_turn_sec(pan)
                        _ui("log", f"Turning right {turn_sec:.1f}s to face target at pan {pan}°")
                        motors.right(MOTOR_SPEED_SLOW)
                        time.sleep(turn_sec)
                        motors.stop()

                    # Re-centre pan after chassis has turned to face target
                    motors.pantilt(0, 5)
                    time.sleep(0.3)

                    return {
                        **result,
                        "target_visible":     True,
                        "target_direction":   "front",
                        "in_my_path":         True,
                        "action":             "forward",
                        "physical_reasoning": f"Target found at pan={pan}° during pantilt scan; chassis turned to face it.",
                        "mission_complete":   False,
                    }
        return None

    # ── Phase 1: Forward-facing 180° arc ─────────────────────────────────────
    found = _sweep("Front arc")
    if found:
        return found

    # ── Phase 2: Single 180° chassis turn ────────────────────────────────────
    _ui("log", "Turning chassis 180° for rear sweep...")
    motors.oled(1, "Turning 180...")
    motors.pantilt(0, 5)   # centre pan-tilt before chassis turn
    time.sleep(0.3)
    motors.right(MOTOR_SPEED_SLOW)
    time.sleep(TURN_90_SEC * 2.0)   # 180° ≈ 2× the calibrated 90° time
    motors.stop()
    time.sleep(0.5)
    log_action("CHASSIS_180", "rear sweep phase")

    # ── Phase 3: Rear (now forward-facing) 180° arc ───────────────────────────
    found = _sweep("Rear arc")
    if found:
        return found

    # ── No target found — Cosmos overview of all collected frames ─────────────
    motors.pantilt(0, 5)
    _ui("log", f"360 scan done — {len(all_frames)} frames → Cosmos overview")
    motors.oled(1, "Analyzing...")
    log_mission_event("scan_360_complete", f"{len(all_frames)} frames, no target found")

    if not all_frames:
        return dict(_SCAN_FALLBACK)

    sensor_ctx = _sensor_context()
    prompt_360 = sensor_ctx + SCAN_360_PROMPT if sensor_ctx else SCAN_360_PROMPT

    try:
        # Use async future so we can do sensor checks while Cosmos is thinking
        future = _cosmos_frames_async(all_frames, prompt_360, max_tokens=300, temp=0.2)
        response = future.result(timeout=90)
        log_ai(prompt_360[-300:], response, label="360_OVERVIEW")
        return _parse_json(response, dict(_SCAN_FALLBACK), label="360° PAN-TILT OVERVIEW")
    except Exception as e:
        log_exception("_scan_360_pantilt_overview", e)
        return dict(_SCAN_FALLBACK)


def _scan_360_smart() -> dict:
    """
    Legacy chassis-rotation 360° scan (8×45° body turns).
    Kept as fallback if pan-tilt hardware is unavailable.
    Prefer _scan_360_pantilt() for normal operation.
    """
    global mission_state
    mission_state = State.SCANNING_360
    _ui("status", "360 SCANNING (chassis)")
    motors.oled(0, "360 Scan")
    log.info("Starting legacy chassis 360 scan (8×45°)")

    motors.stop()
    motors.lights(0, 0)
    time.sleep(0.5)

    all_frames   = []
    best_spot    = None

    TURN_45_SEC = TURN_90_SEC / 2

    for pos in range(8):
        deg = pos * 45
        _ui("log", f"Scanning {deg}°...")
        motors.oled(1, f"Scan {deg}deg")

        for tilt, label in [(5, "ground"), (-15, "mid")]:
            motors.pantilt(0, tilt, 40)
            time.sleep(0.4)

            f_pt = _capture_sharp(CAMERA_PANTILT)
            if f_pt:
                if _is_pitch_black(f_pt):
                    motors.lights(base=180, head=255)
                    time.sleep(0.3)
                    f_pt = _capture_sharp(CAMERA_PANTILT) or f_pt
                    motors.lights(0, 0)
                all_frames.append(f_pt)

            result = _video_scan_at_position()

            if result.get("target_visible"):
                log.info(f"🎯 Target VISIBLE at {deg}° tilt={label} — stopping scan early!")
                _ui("log", f"Target visible at {deg}° — stopping scan!")
                motors.oled(1, "TARGET FOUND!")
                motors.stop()
                time.sleep(0.2)
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
                    "action":             "forward",
                    "speak":              result.get("speak"),
                    "physical_reasoning": f"Target confirmed at {deg}° tilt={label}",
                    "mission_complete":   False
                }

            if result.get("object") not in ("clear", "unknown", None):
                if best_spot is None:
                    best_spot = (deg, result)

            if result.get("wall_ahead") or result.get("obstacle_close"):
                log.info(f"Obstacle at {deg}° during 360 scan")

        motors.pantilt(0, 5)
        time.sleep(0.3)

        if pos < 7:
            motors.right(MOTOR_SPEED_SLOW)
            time.sleep(TURN_45_SEC)
            motors.stop()
            time.sleep(0.4)

    if best_spot:
        deg, spot = best_spot
        _ui("log", f"Re-visiting best potential target at {deg}°...")
        steps_back = (8 - (deg // 45)) % 8
        if steps_back > 0:
            motors.right(MOTOR_SPEED_SLOW)
            time.sleep(TURN_45_SEC * steps_back)
            motors.stop()
            time.sleep(0.5)
        result = _video_scan_at_position()
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
                "action":             "forward",
                "speak":              result.get("speak"),
                "physical_reasoning": "Target confirmed on second look",
                "mission_complete":   False
            }

    log.info(f"No target confirmed — sending {len(all_frames)} overview frames to Cosmos")
    _ui("log", f"360 done — {len(all_frames)} frames → Cosmos overview")
    motors.oled(1, "Analyzing...")

    if not all_frames:
        return dict(_SCAN_FALLBACK)

    sensor_ctx = _sensor_context()
    prompt_360 = sensor_ctx + SCAN_360_PROMPT if sensor_ctx else SCAN_360_PROMPT

    try:
        response = _cosmos_frames(all_frames, prompt_360, max_tokens=300, temp=0.2)
        return _parse_json(response, dict(_SCAN_FALLBACK), label="360° OVERVIEW")
    except Exception as e:
        log_exception("_scan_360_smart", e)
        return dict(_SCAN_FALLBACK)


def _best_360_scan() -> dict:
    """
    Use pan-tilt 360 scan by default. Falls back to chassis rotation if pan-tilt fails.
    """
    try:
        return _scan_360_pantilt()
    except Exception as e:
        log_exception("_scan_360_pantilt", e)
        log.warning("Pan-tilt scan failed — falling back to chassis rotation")
        return _scan_360_smart()


# ─── Direction Control ────────────────────────────────────────────────────────

def _face_direction(direction: str):
    """Turn robot to face a direction. Handles all strings Cosmos might return."""
    d = str(direction).lower().strip()
    if d in ("right", "right_side"):
        motors.right(MOTOR_SPEED_SLOW); time.sleep(1.5); motors.stop()
    elif d in ("left", "left_side"):
        motors.left(MOTOR_SPEED_SLOW);  time.sleep(1.5); motors.stop()
    elif d in ("back", "behind", "backward", "rear"):
        motors.right(MOTOR_SPEED_SLOW); time.sleep(3.0); motors.stop()
    elif d in ("side",):
        motors.right(MOTOR_SPEED_SLOW); time.sleep(0.8); motors.stop()
    elif d in ("down", "below", "front", "ahead", "forward", "unknown", ""):
        pass  # already facing it or already arrived — no turn needed
    time.sleep(0.3)


# ─── Obstacle Avoidance ───────────────────────────────────────────────────────
# Full avoidance logic (back up → LiDAR arc scan → Cosmos reason → turn → verify)
# lives in avoidance.py. This is just a thin shim so existing call sites work.

def _avoid_obstacle(wall_ahead: bool, small_obstacle: bool) -> bool:
    """
    Delegate to avoidance.py's smart pipeline:
      1. Instant backup
      2. LiDAR full-arc scan → pick clearest direction
      3. Cosmos Reason 2 (camera + sensor data) → refine direction + turn_sec
      4. Execute turn → verify path clear → retry or force 360

    Returns True if a full 360° scan should be forced.
    """
    try:
        from avoidance import avoid_obstacle
        return avoid_obstacle(wall_ahead=wall_ahead, small_obstacle=small_obstacle)
    except ImportError:
        log.error("avoidance.py not found — using legacy inline avoidance")
        return _avoid_obstacle_legacy(wall_ahead, small_obstacle)


def _avoid_obstacle_legacy(wall_ahead: bool, small_obstacle: bool) -> bool:
    """
    Fallback if avoidance.py is missing. Mirrors original behaviour.
    """
    global _avoid_attempts, mission_state
    _avoid_attempts += 1
    mission_state = State.AVOIDING
    _ui("status", "AVOIDING")

    if wall_ahead:
        _ui("log", f"Wall — attempt {_avoid_attempts}")
        motors.oled(1, "Wall! Back up...")
        motors.stop(); time.sleep(0.3)
        motors.backward(MOTOR_SPEED_SLOW); time.sleep(1.5)
        motors.stop(); time.sleep(0.3)
        turn_sec = min(1.8 + (_avoid_attempts * 0.4), 3.5)
        direction = "right" if _avoid_attempts % 2 == 1 else "left"
        _turn_nav2_or_direct(direction, turn_sec)
        motors.stop(); time.sleep(0.5)
        if _avoid_attempts >= MAX_AVOID_ATTEMPTS:
            _avoid_attempts = 0
            eric_say("Too many obstacles. Let me scan the full area.")
            return True
        rescan = _quick_scan()
        if rescan.get("wall_ahead") or rescan.get("obstacle_close"):
            return _avoid_obstacle_legacy(wall_ahead=True, small_obstacle=False)
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
        rescan = _quick_scan()
        if rescan.get("wall_ahead") or rescan.get("obstacle_close"):
            return _avoid_obstacle_legacy(wall_ahead=True, small_obstacle=False)
    return False


# ─── Mission Complete ─────────────────────────────────────────────────────────

def _handle_mission_complete(obj_name):
    global mission_active, mission_state
    log.info(f"MISSION COMPLETE — {obj_name}")
    log_mission_event("mission_complete", obj_name or "target")
    mission_state = State.COMPLETE
    motors.stop()
    # Cancel any active Nav2 goal
    try:
        from config import USE_NAV2
        if USE_NAV2:
            from nav2 import cancel_goal, nav2_available
            if nav2_available():
                cancel_goal()
    except Exception:
        pass
    motors.oled(0, "MISSION DONE!")
    motors.oled(1, (obj_name or "Target")[:16])
    _ui("status", "MISSION COMPLETE")

    for _ in range(5):
        motors.lights(255, 255); time.sleep(0.25)
        motors.lights(0, 0);    time.sleep(0.25)
    motors.lights(128, 255)

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
    log_mission_event("announcement", announcement[:150])

    end_mission_log(completed=True)

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

def _approach_target():
    """
    Drive toward target in 2-second steps, scanning after each.
    Uses _move_forward() so Nav2 is used when available.
    Stops when: close enough, obstacle hit, person seen, or 3 consecutive invisible scans.
    On arrival, calls _execute_step_action() so multi-step missions advance properly.
    """
    global mission_state
    _ui("log", "Approaching target...")
    motors.oled(1, "Approaching...")
    _ui("status", "APPROACHING")
    log_mission_event("approach_start", "driving toward target")
    invisible_count = 0

    _NEAR_DISTANCES = {"close", "near", "nearby", "very_close", "very close", "right there"}
    _TARGET_OBJECTS = {"slipper", "shoe", "person", "robot"}

    for attempt in range(12):       # max ~24s total approach
        if not mission_active:
            break
        _move_forward(duration_sec=2.0, distance_m=0.5)
        motors.stop()
        time.sleep(0.4)
        check = _quick_scan()
        dist  = str(check.get("distance", "far")).lower()
        obj   = check.get("object", "unknown")
        tdir  = str(check.get("target_direction", "front")).lower().strip()

        # ── Obstacle → avoid and continue approach ───────────────────────────
        if check.get("wall_ahead") or check.get("obstacle_close"):
            _ui("log", "Obstacle during approach — avoiding")
            log_action("AVOID_DURING_APPROACH", f"obj={obj}")
            force_360 = _avoid_obstacle(
                wall_ahead=check.get("wall_ahead", False),
                small_obstacle=check.get("small_obstacle", False)
            )
            if force_360:
                mission_state = State.SEARCHING
                return
            continue

        if check.get("mission_complete"):
            _execute_step_action(check.get("object_name"))
            return

        # ── Close enough — execute the step action ───────────────────────────
        if check.get("target_visible") and obj in _TARGET_OBJECTS and dist in _NEAR_DISTANCES:
            _ui("log", f"Target confirmed close ({dist}) — executing step action")
            log_mission_event("target_reached", f"obj={obj} dist={dist}")
            _execute_step_action(check.get("object_name") or obj)
            return

        if dist in _NEAR_DISTANCES or check.get("in_my_path"):
            _ui("log", f"Close to target ({dist}) — executing step action")
            _execute_step_action(check.get("object_name") or obj)
            return

        # ── Target dropped below camera → already on top of it ───────────────
        if tdir in ("down", "below"):
            _ui("log", "Target below camera — arrived!")
            motors.pantilt(0, 20)
            time.sleep(0.3)
            _execute_step_action(check.get("object_name") or obj)
            return

        # ── Lateral steering ─────────────────────────────────────────────────
        if tdir in ("left", "left_side"):
            _ui("log", "Target drifted left — correcting")
            motors.left(MOTOR_SPEED_SLOW); time.sleep(0.4); motors.stop()
        elif tdir in ("right", "right_side"):
            _ui("log", "Target drifted right — correcting")
            motors.right(MOTOR_SPEED_SLOW); time.sleep(0.4); motors.stop()

        # ── Pan-tilt vertical tracking ────────────────────────────────────────
        tilt_map = {"far": 5, "medium": 10, "mid": 10,
                    "close": 15, "near": 18, "nearby": 18, "very_close": 22}
        motors.pantilt(0, tilt_map.get(dist, 5), 60)

        # ── Person nearby → eye-contact gate before greeting ─────────────────
        if obj == "person" and (dist in _NEAR_DISTANCES or check.get("in_my_path")):
            name = check.get("object_name") or "person"
            _ui("log", "Person nearby during approach — checking eye contact...")
            motors.oled(1, "Person!")

            # Eye-contact check
            ec_frame = capture_frame(CAMERA_PANTILT, 320, 240)
            greet    = True
            if ec_frame:
                try:
                    ec_payload = {
                        "model": COSMOS_MODEL,
                        "messages": [{"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ec_frame}"}},
                            {"type": "text", "text":
                                '{"close_and_facing": true_or_false, "reasoning": "one sentence"} '
                                '— Is the person within 1.5m AND facing/looking toward you?'}
                        ]}],
                        "max_tokens": 50,
                        "temperature": 0.1,
                    }
                    ec_r = requests.post(VLLM_URL, json=ec_payload, timeout=15)
                    ec_r.raise_for_status()
                    ec_raw = ec_r.json()["choices"][0]["message"]["content"].strip()
                    log_ai("eye_contact_approach", ec_raw, label="EYE_CONTACT")
                    ec    = _parse_json(ec_raw, {"close_and_facing": True}, "EYE CONTACT APPROACH")
                    greet = ec.get("close_and_facing", True)
                    if not greet:
                        _ui("log", f"Person not facing Eric — skipping greeting ({ec.get('reasoning','')})")
                except Exception as e:
                    log_exception("eye_contact_approach", e)

            if greet:
                _ui("status", f"FOUND — {name}")
                greeting = ask_cosmos(
                    f"You see {name} nearby. Greet them and ask if they can help with your mission. 1-2 sentences.",
                    max_tokens=80
                )
                eric_say(greeting)
                log_mission_event("person_greeted_approach", name)
                mission_state = State.INTERACTING
                return

        if not check.get("target_visible", False):
            invisible_count += 1
            _ui("log", f"Target not visible ({invisible_count}/3)")
            if invisible_count >= 3:
                _ui("log", "Lost target — resuming search")
                log_mission_event("target_lost", "resuming search after 3 invisible scans")
                mission_state = State.SEARCHING
                motors.forward(MOTOR_SPEED_SLOW)
                return
        else:
            invisible_count = 0

    _ui("log", "Approach complete — INTERACTING")
    mission_state = State.INTERACTING


def _process_scan(scan, from_360=False):
    global mission_state, _empty_scans, _avoid_attempts, _scans_since_360
    global _target_spotted_count

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
        
    # ── Social intent + risk assessment (from Egocentric recipe) ─────────
    social = scan.get("social_intent")
    risk   = scan.get("risk_assessment")
    if social:
        _ui("log", f"👤 Social: {social}")
    if risk:
        _ui("log", f"⚠️  Risk: {risk}")

    if complete:
        if speak_tx: eric_say(speak_tx)
        _execute_step_action(obj_name)
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
        log_action("AVOID", f"wall={wall_ahead} obstacle_close={obstacle_close} obj={obj}")
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
        try:
            from avoidance import reset_avoid_counter
            reset_avoid_counter()
        except ImportError:
            pass

    if speak_tx:
        eric_say(speak_tx)

    # ── Target persistence: Cosmos often flip-flops target_visible ───────────
    if target_visible:
        _target_spotted_count += 1
    else:
        if _target_spotted_count > 0:
            _target_spotted_count -= 1
            if _target_spotted_count > 0:
                log.info("Target_visible=False but keeping target lock (Cosmos flip-flop guard)")
                target_visible = True

    if target_visible:
        _empty_scans = 0
        _target_spotted_count = 0
        direction = str(target_dir).lower().strip() if target_dir else "front"

        if direction in ("down", "below"):
            _ui("log", "Target is directly below — already arrived!")
            motors.stop()
            motors.pantilt(0, 20)
            time.sleep(0.5)
            _execute_step_action(obj_name)
            return

        if from_360:
            _ui("log", f"Target spotted at {direction} — approaching!")
            motors.oled(1, f"Target {direction}!")
        else:
            _ui("log", "Target spotted during scan — approaching")
        _ui("status", "TARGET SPOTTED")
        log_mission_event("target_spotted", f"direction={direction} obj={obj} name={obj_name}")
        if direction not in ("front", "ahead", "unknown", ""):
            _face_direction(direction)
        _approach_target()
        return

    # Person or robot visible → only stop if near, keep moving if far
    _NEAR_DISTANCES = {"close", "near", "nearby", "very_close", "very close", "right there"}
    if obj in ["person", "robot"] and not target_visible:
        dist_str = str(distance).lower()
        if dist_str in _NEAR_DISTANCES or in_path:
            _empty_scans = 0
            motors.stop()
            mission_state = State.INTERACTING
            name = obj_name or obj
            motors.oled(0, name[:16])
            motors.oled(1, "Talking...")
            _ui("status", f"FOUND — {name}")
            motors.pantilt(0, 5)
            time.sleep(0.5)

            # ── Eye-contact gate before greeting ─────────────────────────────
            ec_frame = capture_frame(CAMERA_PANTILT, 320, 240)
            should_greet = True
            if ec_frame:
                try:
                    ec_payload = {
                        "model": COSMOS_MODEL,
                        "messages": [{"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ec_frame}"}},
                            {"type": "text", "text":
                                '{"close_and_facing": true_or_false, "reasoning": "one sentence"} '
                                '— Is the person within 1.5m AND facing/looking toward you?'}
                        ]}],
                        "max_tokens": 50,
                        "temperature": 0.1,
                    }
                    ec_r = requests.post(VLLM_URL, json=ec_payload, timeout=15)
                    ec_r.raise_for_status()
                    ec_raw = ec_r.json()["choices"][0]["message"]["content"].strip()
                    log_ai("eye_contact_scan", ec_raw, label="EYE_CONTACT")
                    ec = _parse_json(ec_raw, {"close_and_facing": True}, "EYE CONTACT SCAN")
                    should_greet = ec.get("close_and_facing", True)
                    if not should_greet:
                        _ui("log", f"Person near but not facing Eric — not greeting ({ec.get('reasoning','')})")
                        mission_state = State.SEARCHING
                        motors.forward(MOTOR_SPEED_SLOW)
                        return
                except Exception as e:
                    log_exception("eye_contact_scan", e)

            if should_greet:
                greeting = ask_cosmos(
                    f"You see {name} {'ahead' if in_path else 'nearby'} ({dist_str} away). "
                    "Greet them and ask about your mission. 1-2 sentences.",
                    max_tokens=80
                )
                eric_say(greeting)
                log_mission_event("person_greeted_scan", name)
                _ui("status", f"TALKING — {name}")
            return
        else:
            _ui("log", f"Person ({obj_name or 'unknown'}) visible but {dist_str} — continuing")

    if obj in ["clear", "unknown"] and not target_visible:
        _empty_scans += 1
        _ui("log", f"Nothing found ({_empty_scans}/{EMPTY_SCAN_LIMIT})")

    if from_360 and clear_dir != "front":
        _face_direction(clear_dir)

    # ── Terrain-aware speed control ───────────────────────────────────────────
    terrain_speed = _speed_for_terrain(terrain)

    if terrain_speed is None:
        # Impassable terrain — treat exactly like a wall
        _ui("log", f"🚧 Impassable terrain '{terrain}' — avoiding")
        log_action("IMPASSABLE_TERRAIN", terrain)
        motors.stop()
        eric_say(f"I see {terrain} ahead. I cannot cross that. Finding another way.")
        force_360 = _avoid_obstacle(wall_ahead=True, small_obstacle=False)
        if force_360:
            _scans_since_360 = SCANS_BEFORE_360
        else:
            motors.forward(MOTOR_SPEED_SLOW)
        return

    if terrain and terrain not in ("clear", "unknown", ""):
        speed_label = ("FAST" if terrain_speed == MOTOR_SPEED_FAST
                       else "SLOW" if terrain_speed == MOTOR_SPEED_SLOW
                       else "NORMAL")
        _ui("log", f"Terrain '{terrain}' → {speed_label} speed")
        log_action("TERRAIN_SPEED", f"{terrain} → {speed_label}")

    if action == "navigate_around":
        _turn_nav2_or_direct("left", 0.8)
        motors.forward(terrain_speed)
    elif action == "stop":
        motors.stop()
    elif action == "turn_right":
        _turn_nav2_or_direct("right", 1.0)
        motors.forward(terrain_speed)
    elif action == "turn_left":
        _turn_nav2_or_direct("left", 1.0)
        motors.forward(terrain_speed)
    elif action == "turn_back":
        _turn_nav2_or_direct("back", 1.5)
        motors.forward(terrain_speed)
    else:
        motors.forward(terrain_speed)

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
    log_mission_event("initial_360_scan", "beginning")
    scan = _best_360_scan()
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
                    log_action("NAV_OBSTACLE", nav.get("physical_reasoning", ""))
                    force_360 = _avoid_obstacle(
                        wall_ahead=nav.get("wall_ahead", False),
                        small_obstacle=nav.get("small_obstacle", False)
                    )
                    if force_360:
                        _scans_since_360 = SCANS_BEFORE_360
                        _nav_clips_since_scan = NAV_CLIPS_BETWEEN_SCANS
                    else:
                        motors.forward(MOTOR_SPEED_SLOW)
                elif nav.get("action") == "slow":
                    motors.slow()
                    log_action("NAV_SLOW", "nav check requested slow")
                elif nav.get("action") == "stop":
                    motors.stop()
                    log_action("NAV_STOP", "nav check requested stop")
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
                log_mission_event("360_scan_triggered",
                                  f"empty={_empty_scans} scans_since={_scans_since_360}")
                scan = _best_360_scan()
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
            log_exception("_mission_loop", e)
            time.sleep(1)

    motors.stop()
    mission_state = State.IDLE
    _ui("status", "IDLE")
    log.info("Mission loop ended")
