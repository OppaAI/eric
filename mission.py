"""
ERIC — Mission Logic

Camera strategy:
  Navigation (moving):  Layer 1 (LiDAR/OAK-D) handles safety automatically.
                        Layer 2 (YOLO on OAK-D Myriad X) detects people/animals.
                        No Cosmos called while moving — Eric moves continuously.
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

def _safe_to_fwd() -> bool:
    """Guard before every motors.forward() — checks LiDAR obstacle state."""
    try:
        from lidar import safe_to_forward
        return safe_to_forward()
    except Exception:
        return True  # lidar not loaded — allow forward
from motors import motors
from cosmos import (
    ask_cosmos, ask_cosmos_plain, set_mission_briefing, get_mission_briefing,
    capture_frame, capture_frames_video,
    start_frame_buffer, get_buffered_frames,
    CAMERA_WEBCAM, CAMERA_PANTILT
)
from tts import speak
from logger import (
    log_ai, log_action, log_mission_event,
    start_mission_log, end_mission_log, log_exception
)
from alarm import sound_alarm, stop_alarm, AlarmType

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


# ─── Mission State Container ──────────────────────────────────────────────────
# Consolidates every mutable module-level global into one typed dataclass.
#
# Benefits vs 20+ scattered globals:
#   • Thread-safety  — attribute access is atomic; no partial-update windows
#   • Testability    — reset() gives a clean slate without a module reload
#   • Debuggability  — repr() dumps all state in one log line
#   • Readability    — _ms.mission_active is explicit, not a mystery global
#
# External callers (GUI, etc.) import _ms directly:
#     from mission import _ms
#     if _ms.mission_active: ...
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class MissionState:
    """Single source of truth for all mutable mission state."""

    # ── Core control ──────────────────────────────────────────────────────────
    mission_active:       bool  = False
    mission_state:        str   = State.IDLE
    conversation_history: list  = dataclasses.field(default_factory=list)

    # ── Search / avoidance counters ───────────────────────────────────────────
    empty_scans:          int   = 0
    avoid_attempts:       int   = 0
    scans_since_360:      int   = 0
    target_spotted_count: int   = 0
    nav_clips_since_scan: int   = 0

    # ── Mission step engine ───────────────────────────────────────────────────
    mission_steps:        list  = dataclasses.field(default_factory=list)
    current_step_idx:     int   = 0

    # ── YAML mission metadata ─────────────────────────────────────────────────
    mission_alarm_type:    str   = AlarmType.HAZARD
    mission_target_objects: list = dataclasses.field(default_factory=list)
    mission_flags:          dict = dataclasses.field(default_factory=dict)
    mission_find_count:     int  = 0
    mission_hazard_log:     list = dataclasses.field(default_factory=list)

    # ── Async nav check ───────────────────────────────────────────────────────
    pending_nav:      object = None   # concurrent.futures.Future | None
    last_nav_result:  dict   = dataclasses.field(default_factory=dict)

    # ── YOLO Layer 2 detection ────────────────────────────────────────────────
    yolo_person_detected:    bool   = False
    yolo_detect_label:       object = None
    yolo_detect_distance:    object = None
    yolo_detect_bearing:     object = None
    yolo_detect_bearing_deg: object = None
    yolo_detect_time:        float  = 0.0

    # ── TTS head movement ─────────────────────────────────────────────────────
    head_talking:            bool   = False

    def reset_counters(self):
        """Reset search/avoidance counters — call when starting a new search phase."""
        self.empty_scans          = 0
        self.avoid_attempts       = 0
        self.scans_since_360      = 0
        self.target_spotted_count = 0
        self.nav_clips_since_scan = 0

    def reset_for_new_mission(self):
        """Full reset — call at mission start."""
        self.conversation_history    = []
        self.mission_find_count      = 0
        self.mission_hazard_log      = []
        self.pending_nav             = None
        self.last_nav_result         = {}
        self.yolo_person_detected    = False
        self.yolo_detect_label       = None
        self.yolo_detect_distance    = None
        self.yolo_detect_bearing     = None
        self.yolo_detect_bearing_deg = None
        self.yolo_detect_time        = 0.0
        self.reset_counters()

    def __repr__(self) -> str:
        return (
            f"MissionState(active={self.mission_active}, state={self.mission_state}, "
            f"step={self.current_step_idx}/{len(self.mission_steps)}, "
            f"empty={self.empty_scans}, avoid={self.avoid_attempts}, "
            f"spotted={self.target_spotted_count})"
        )


# ── Module-level singleton — the only mutable state in this module ────────────
_ms = MissionState()

# ── YOLO callback lock (replaces old _yolo_lock module global) ────────────────
_yolo_lock = threading.Lock()

# ── UI callback registry (infrastructure, not mission state) ──────────────────
_ui_callbacks: dict = {"eric_says": None, "status": None, "log": None}

# ── Backward-compat module-level accessors ────────────────────────────────────
# gui.py imports: mission_active, mission_state, conversation_history
# These are thin functions — gui.py must call them to get live state.
# The bare-name imports in gui.py line 31 are replaced by _ms references below.

def get_mission_active() -> bool:
    return _ms.mission_active

def get_mission_state() -> str:
    return _ms.mission_state

def get_conversation_history() -> list:
    return _ms.conversation_history

# ── Tuning constants (never mutated at runtime) ──────────────────────────────
EMPTY_SCAN_LIMIT        = 5    # trigger 360 after 5 consecutive empty scans
SCANS_BEFORE_360        = 10   # periodic 360 every 10 quick scans
MAX_AVOID_ATTEMPTS      = 3    # force 360 after this many avoid failures
TARGET_CONFIRM_NEEDED   = 1    # only needs 1 positive scan to approach
DETECTION_CONFIDENCE_MIN = 0.0   # Cosmos does not emit confidence scores — always 0.0
                                  # below this, sweep detections are treated as hallucinations and skipped


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
    target:      str          # e.g. "person", "robot", "cat"
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


def register_ui_callbacks(**cbs):
    _ui_callbacks.update(cbs)


def _ui(key, text):
    """Deliver a UI event. Never raises — a broken callback must not crash the mission."""
    cb = _ui_callbacks.get(key)
    if cb:
        try:
            cb(text)
        except Exception as _exc:
            log.warning(f"UI callback '{key}' raised: {_exc}")


def _head_talk_thread(tilt: int):
    """
    Background thread — occasional natural head micro-movements while Eric speaks.
    Pattern: hold at centre (random duration) -> move to random small angle -> return to centre.
    Pan +-5 degrees, tilt offset +-3 degrees. Feels organic, not mechanical.
    Stops when _head_talking flag is cleared.
    """
    import random
    try:
        while _ms.head_talking:
            # Hold at centre — random pause, sometimes long sometimes short
            centre_hold = random.uniform(2.0, 6.0)
            t0 = time.time()
            while _ms.head_talking and (time.time() - t0) < centre_hold:
                time.sleep(0.1)
            if not _ms.head_talking:
                break

            # Small random position — pan +-5, slight tilt offset +-3
            rand_pan  = random.choice([-5, -4, -3, -2, 2, 3, 4, 5])
            rand_tilt = tilt + random.choice([-3, -2, 0, 0, 2, 3])
            motors.pantilt(rand_pan, rand_tilt, 30)

            # Hold briefly at that angle
            move_hold = random.uniform(0.8, 2.5)
            t0 = time.time()
            while _ms.head_talking and (time.time() - t0) < move_hold:
                time.sleep(0.1)
            if not _ms.head_talking:
                break

            # Return to centre
            motors.pantilt(0, tilt, 30)

    except Exception:
        pass
    finally:
        try:
            motors.pantilt(0, tilt, 30)   # return to centre
        except Exception:
            pass


def eric_say(text):
    if not text:
        return
    # Don't speak or display raw JSON — Cosmos sometimes leaks it into speak field
    text_stripped = str(text).strip()
    if text_stripped.startswith("{") or text_stripped.startswith("["):
        log.warning(f"eric_say received JSON instead of plain text — suppressed: {text_stripped[:80]}")
        return
    _ui("eric_says", text_stripped)
    log_mission_event("eric_say", text_stripped[:120])

    # Start head movement thread while speaking — only if mission flag is set
    _head_move = _ms.mission_flags.get("head_talk", False)
    if _head_move:
        try:
            _current_tilt = getattr(_ms, "last_confirm_tilt", 10)
            _ms.head_talking = True
            _ht = threading.Thread(target=_head_talk_thread, args=(_current_tilt,), daemon=True)
            _ht.start()
        except Exception:
            pass

    speak(text_stripped)  # speak full text — TTS handles all sentences

    # Stop head movement — only if it was started
    if _head_move:
        try:
            from tts import wait_speak_stop
            wait_speak_stop()
        except Exception:
            pass
        try:
            _ms.head_talking = False
        except Exception:
            pass


# ─── Async Cosmos Wrapper ─────────────────────────────────────────────────────

def _cosmos_frames(frames, prompt, max_tokens=250, temp=0.3):
    """Synchronous Cosmos call with logging. Used directly or via async wrapper."""
    from cosmos import _system_prompt as sys_prompt

    # ── Token budget guard — model max_model_len=2048 ─────────────────────────
    # Each image costs ~256 tokens. System prompt + mission briefing can be large.
    # Estimate: 4 chars ~ 1 token. Reserve max_tokens for output.
    # Budget: 2048 - max_tokens - (num_frames * 256) - 50 (safety margin)
    _IMAGE_TOKENS   = 256  # vLLM vision token cost per image
    _CHAR_PER_TOKEN = 4
    _token_budget   = 2048 - max_tokens - (len(frames) * _IMAGE_TOKENS) - 50
    _char_budget    = max(_token_budget, 200) * _CHAR_PER_TOKEN

    # Truncate system prompt (keep tail — mission briefing is appended at end)
    _sys = sys_prompt or ""
    _sys_char_limit = int(_char_budget * 0.4)
    if len(_sys) > _sys_char_limit:
        _sys = _sys[-_sys_char_limit:]
        log.debug(f"_cosmos_frames: system prompt truncated to {_sys_char_limit} chars")

    # Remaining budget for user prompt
    _prompt_char_limit = max(_char_budget - len(_sys), 200)
    _prompt = prompt if len(prompt) <= _prompt_char_limit else prompt[-_prompt_char_limit:]
    if _prompt != prompt:
        log.debug(f"_cosmos_frames: user prompt truncated to {_prompt_char_limit} chars")

    img_content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}}
        for f in frames
    ]
    img_content.append({"type": "text", "text": _prompt})
    payload = {
        "model": COSMOS_MODEL,
        "messages": [
            {"role": "system", "content": _sys},
            {"role": "user",   "content": img_content}
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
    # ── Simple mission: skip Cosmos entirely — no KV cache bleed risk ────────
    # If the briefing has no explicit step markers, it's a simple find mission.
    # Build a single step directly from target_objects — no Cosmos call needed.
    _step_markers = ["step 1:", "step 2:", "step1.", "step2.", "deliver_message",
                     "find_and_approach", "speak_to", "step_num"]
    _is_multistep = any(m in briefing.lower() for m in _step_markers)
    if not _is_multistep:
        _tgt = (_ms.mission_target_objects[0]
                if _ms.mission_target_objects else "target")
        log.info(f"Simple mission — building single step: find_and_approach {_tgt!r}")
        return [MissionStep(step_num=1, target=_tgt, action="find_and_approach")]

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
  "target":      "person",
  "action":      "deliver_message",
  "message":     "Package delivered.",
  "photo_count": 1,
  "wait_sec":    20
}}

Example for multi-step mission:
[
  {{"step_num": 1, "target": "person", "action": "deliver_message",
    "message": "Package delivered.", "photo_count": 1, "wait_sec": 20}},
  {{"step_num": 2, "target": "robot", "action": "speak_to",
    "message": "", "photo_count": 1, "wait_sec": 30}},
  {{"step_num": 3, "target": "cat", "action": "photograph",
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
    if _ms.mission_steps and _ms.current_step_idx < len(_ms.mission_steps):
        return _ms.mission_steps[_ms.current_step_idx]
    return None


def _advance_step():
    """Mark the current step complete and move to the next, or end the mission."""
    step = _current_step()
    if step:
        step.completed = True
        log_mission_event(f"step_{step.step_num}_complete", f"{step.target} — {step.action}")

    _ms.current_step_idx += 1

    if _ms.current_step_idx >= len(_ms.mission_steps):
        # All steps done
        last_target = step.target if step else "all targets"
        _handle_mission_complete(last_target)
    else:
        nxt = _current_step()
        msg = f"Step {_ms.current_step_idx} complete. Now finding {nxt.target}."
        eric_say(msg)
        _ui("status", f"STEP {nxt.step_num}: {nxt.target.upper()}")
        _ui("log", msg)
        # Update Cosmos system prompt so it searches for the next target
        set_mission_briefing(
            f"CURRENT STEP {nxt.step_num} of {len(_ms.mission_steps)}: "
            f"Find {nxt.target} and {nxt.action.replace('_', ' ')}.\n"
            f"Original mission: {get_mission_briefing()}"
        )
        # Resume searching
        _ms.reset_counters()
        try:
            from avoidance import reset_avoid_counter
            reset_avoid_counter()
        except ImportError as _exc:
            log.debug(f"avoidance module not loaded: {_exc}")
        _ms.mission_state = State.SEARCHING
        if _safe_to_fwd():
            motors.forward(MOTOR_SPEED_SLOW)


def _execute_step_action(obj_name: str):
    """
    Called when Eric arrives at the current step's target.
    Executes the required action (speak, photograph, wait, etc.) then advances.
    """
    step = _current_step()
    if not step:
        _handle_mission_complete(obj_name)
        return

    _ms.mission_state = State.INTERACTING
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
        greeting = ask_cosmos_plain(
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

    except Exception as _exc:  # optional component
        log.debug(f"optional component error: {_exc}")
        log.debug(f"JSON parse failed (label={label}): {response[:80]}")
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

    # ── Step 0: remap aliased field names Cosmos (2B) frequently hallucinates ─
    # The model invents slight variations of canonical names. Catch them all here
    # before any downstream logic sees them. "canonical" wins if both exist.
    _FIELD_ALIASES: dict[str, str] = {
        # speak
        "speaker":             "speak",
        "speech":              "speak",
        "say":                 "speak",
        "spoken":              "speak",
        "tts":                 "speak",
        "announcement":        "speak",
        "narration":           "speak",
        "response":            "speak",
        # target_visible
        "target_visibility":   "target_visible",
        "targetvisible":       "target_visible",
        "target_found":        "target_visible",
        "found":               "target_visible",
        "detected":            "target_visible",
        # physical_reasoning
        "reasoning":           "physical_reasoning",
        "reason":              "physical_reasoning",
        "explanation":         "physical_reasoning",
        "analysis":            "physical_reasoning",
        "observation":         "physical_reasoning",
        "notes":               "physical_reasoning",
        "summary":             "physical_reasoning",
        # object_name
        "name":                "object_name",
        "label":               "object_name",
        "object_label":        "object_name",
        # action
        "movement":            "action",
        "next_action":         "action",
        "recommended_action":  "action",
        # clearest_direction
        "clear_direction":     "clearest_direction",
        "best_direction":      "clearest_direction",
        "open_direction":      "clearest_direction",
        # target_direction
        "direction":           "target_direction",
        "target_location":     "target_direction",
        "target_side":         "target_direction",
    }
    for alias, canonical in _FIELD_ALIASES.items():
        if alias in result:
            if canonical not in result:
                log.info(f"Field alias: '{alias}' → '{canonical}'")
                result[canonical] = result.pop(alias)
            else:
                result.pop(alias)   # canonical already present — drop the duplicate

    # ── Step 0b: strip unknown fields so they don't pollute the debug print ───
    _VALID_FIELDS = {
        "object", "object_name", "terrain", "distance", "in_my_path",
        "wall_ahead", "obstacle_close", "small_obstacle", "void_ahead",
        "target_visible", "target_direction", "clearest_direction",
        "action", "speak", "physical_reasoning", "mission_complete",
        # nav-check only
        "person_visible",
        # optional / extended
        "severity", "social_intent", "risk_assessment",
    }
    stray = [k for k in list(result) if k not in _VALID_FIELDS]
    if stray:
        log.info(f"Dropping unknown fields from Cosmos output: {stray}")
        for k in stray:
            result.pop(k)

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

    # ── Consistency fix: if object matches mission target, target_visible must be True ──
    # Cosmos sometimes sees the target but second-guesses target_visible=False.
    # If the object field matches any keyword in mission_target_objects, force True.
    _obj_val  = str(result.get("object", "")).lower()
    _name_val = str(result.get("object_name", "") or "").lower()
    _targets  = [t.lower() for t in (_ms.mission_target_objects or [])]
    if _obj_val not in ("", "unknown", "clear") and not result.get("target_visible"):
        # Check if object or object_name matches any target keyword
        _matched = any(
            (kw in _obj_val or kw in _name_val or _obj_val in kw)
            for kw in _targets
        ) if _targets else False
        if _matched:
            log.info(f"Auto-correcting target_visible=True (object={_obj_val} matched targets={_targets})")
            result["target_visible"] = True

    # ── Note: stop→forward auto-correction removed.
    # Cosmos saying stop with no explicit obstacle flag is valid —
    # it may have seen something the sensor fields don't capture.
    # Hardware sensor overrides in _quick_scan/_nav_check handle false stops.

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
        if k == "detection_confidence":  # hidden from display
            continue
    #if False and isinstance(v, float):
     #       icon = f"  {'✅' if v >= DETECTION_CONFIDENCE_MIN else '❌ LOW'}"
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
      - D500 LiDAR  (lidar.py) — front arc min distance + void/return-count check
      - OAK-D Lite  (oakd.py)  — stereo depth at center-forward + floor-drop check

    Void detection is included: if either sensor detects a floor disappearing
    (too few LiDAR returns, or OAK-D shows floor depth jumping) this context
    will contain a hard VOID WARNING so Cosmos knows to stop.

    Fails silently — if both sensors are unavailable, returns "".
    """
    lines = []

    # ── LiDAR obstacle + void ──────────────────────────────────────────────
    try:
        from lidar import get_status as lidar_status, lidar_available, lidar_void_ahead
        if lidar_available():
            ls = lidar_status()
            dist = ls.get("min_distance", 999)
            dist_str = f"{dist:.2f}m" if dist < 999 else "clear"
            lines.append(f"LiDAR front arc minimum distance: {dist_str}")
            if ls.get("obstacle_close"):
                lines.append("⚠️  LIDAR STOP ZONE: obstacle within 0.30m — do NOT move forward")
            elif ls.get("obstacle_near"):
                lines.append("⚠️  LiDAR caution: obstacle within 0.60m — slow or stop")

            # LiDAR void check removed — D500 is horizontal and unreliable for drops.
            # OAK-D stereo depth handles floor-drop detection below.
    except Exception as _exc:  # lidar
        log.debug(f"lidar unavailable: {_exc}")

    # ── OAK-D obstacle + floor-drop ───────────────────────────────────────
    try:
        from oakd import get_front_depth, oakd_available, get_floor_drop
        if oakd_available():
            d = get_front_depth()
            if d is not None:
                lines.append(f"OAK-D depth center-forward: {d:.2f}m")
                if d < 0.30:
                    lines.append("⚠️  OAK-D: obstacle VERY CLOSE (<0.30m) — stop immediately")
                elif d < 0.60:
                    lines.append("⚠️  OAK-D: obstacle within caution range (<0.60m) — slow down")

            # Floor-drop check disabled — false positives on flat floors
            drop = {"void_detected": False}  # get_floor_drop()
            if drop["void_detected"] and drop["confidence"] == "high":
                edge = drop["floor_edge_m"]
                mid  = drop["floor_mid_m"]
                edge_str = f"{edge:.1f}m" if edge is not None else "none"
                mid_str  = f"{mid:.1f}m"  if mid  is not None else "none"
                lines.append(
                    f"🕳️  OAK-D FLOOR DROP WARNING (HIGH confidence): "
                    f"floor at mid={mid_str} but drops to {edge_str} at edge — "
                    f"{drop['reason']} — STOP, do NOT move forward"
                )
    except Exception as _exc:  # oakd/yolo
        log.debug(f"oakd/yolo unavailable: {_exc}")

    # ── YOLO Layer 2 — live person/animal positions (ground truth) ────────
    # These are hardware detections from the OAK-D Myriad X VPU — not visual
    # estimates. Trust these distances and bearings for spatial reasoning.
    try:
        from oakd import oakd_available, yolo_available
        if oakd_available() and yolo_available():
            import time as _t
            _now = _t.monotonic()
            try:
                from oakd import _last_yolo_positions, _yolo_lock
                with _yolo_lock:
                    positions = dict(_last_yolo_positions)
            except Exception as _exc:  # oakd/yolo
                log.debug(f"oakd/yolo error: {_exc}")
                positions = {}
            fresh = {
                label: pos for label, pos in positions.items()
                if (_now - pos.get("timestamp", 0)) < 4.0   # only last 4 seconds
            }
            if fresh:
                for label, pos in fresh.items():
                    age = _now - pos.get("timestamp", _now)
                    lines.append(
                        f"YOLO L2 detection: {label} at {pos['dist_m']:.1f}m "
                        f"bearing={pos['bearing']} ({pos['bearing_deg']:+.0f}°) "
                        f"confidence={pos['confidence']:.0%} [{age:.1f}s ago]"
                    )
                lines.append(
                    "↑ These are hardware depth measurements from the OAK-D VPU — "
                    "more accurate than visual distance estimates."
                )
    except Exception as _exc:  # optional component
        log.debug(f"optional component unavailable: {_exc}")

    # ── Nav2 pose + navigation state ──────────────────────────────────────
    try:
        from config import USE_NAV2
        if USE_NAV2:
            from nav2 import nav2_available, get_pose, is_navigating, cancel_goal
            if nav2_available():
                pose = get_pose()
                nav_str = "NAVIGATING to goal" if is_navigating() else "stationary"
                lines.append(
                    f"Nav2 pose: x={pose['x']:.2f}m y={pose['y']:.2f}m "
                    f"yaw={math.degrees(pose['yaw']):.0f}° — {nav_str}"
                )
    except Exception as _exc:  # nav2
        log.debug(f"nav2 unavailable: {_exc}")

    if not lines:
        return ""

    return (
        "SENSOR DATA (ground truth — trust these over visual estimates):\n"
        + "\n".join(lines)
        + "\n\n"
    )


# ─── Nav2 / Motor Movement Abstraction ───────────────────────────────────────

def _void_check() -> dict:
    """
    Central void/drop safety gate — disabled.
    OAK-D void detection causes false positives on low-texture floors at 15cm mount.
    LiDAR handles all obstacle safety.
    """
    return {"void": False, "confidence": "low",
            "reason": "void check disabled", "source": "none"}


def _move_forward(duration_sec: float = 2.0, distance_m: float = 1.5):
    """
    Move Eric forward using Nav2 if available, else direct motor control.
    ALWAYS runs _void_check() first — will not move if a floor drop is detected.
    """
    # ── Void gate — never move forward into a hole ────────────────────────
    void = _void_check()
    if void["void"]:
        motors.stop()
        log.warning(f"🕳️  _move_forward BLOCKED by void ({void['source']}): {void['reason']}")
        log_action("VOID_BLOCK_MOVE", f"{void['source']}: {void['reason']}")
        _ui("log", f"🕳️  VOID DETECTED — stopping! ({void['source']}: {void['reason']})")
        _ui("status", "VOID DETECTED — STOPPED")
        motors.oled(0, "VOID AHEAD!")
        motors.oled(1, "STOP")
        eric_say("I detect a drop or hole ahead. Stopping for safety.")
        return

    from config import USE_NAV2
    if USE_NAV2:
        try:
            from nav2 import nav2_available, send_goal, get_pose, is_navigating
            if nav2_available():
                pose = get_pose()
                yaw  = pose["yaw"]
                tx   = pose["x"] + distance_m * math.cos(yaw)
                ty   = pose["y"] + distance_m * math.sin(yaw)
                _nav2_start = time.time()
                send_goal(tx, ty, yaw)
                # Wait for Nav2 to finish or timeout
                deadline = time.time() + duration_sec + 8.0
                while is_navigating() and time.time() < deadline and _ms.mission_active:
                    time.sleep(0.2)
                _nav2_elapsed = time.time() - _nav2_start
                # If Nav2 returned in <0.8s it almost certainly rejected the goal
                # (status 6 = costmap blocked by person/obstacle). Fall through to
                # direct motors so Eric actually moves toward the target.
                if _nav2_elapsed >= 0.8:
                    return   # Nav2 navigated successfully
                log.warning(
                    f"Nav2 instant-return ({_nav2_elapsed:.2f}s) — likely status 6, "                    f"falling back to direct motors"
                )
        except Exception as e:
            log.warning(f"Nav2 move_forward failed ({e}) — falling back to direct")

    # Direct motor fallback
    if _safe_to_fwd():
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
                while is_navigating() and time.time() < deadline and _ms.mission_active:
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


def get_briefing_from_file(name: str) -> str | None:
    data = load_mission_file(name)
    return data.get("briefing", "").strip() if data else None


def get_mission_metadata(name: str) -> dict:
    """Return full YAML dict for a named mission (for GUI display / start_mission)."""
    return load_mission_file(name) or {}


# ─── Mission Control ──────────────────────────────────────────────────────────

def start_mission(briefing: str, mission_name: str = ""):
    """
    Start a mission from a briefing string (and optional YAML mission name).
    If mission_name is provided, alarm_type and target_objects are loaded from
    the YAML file. Free-text briefings default to no alarm, generic scan.
    """

    if _ms.mission_active:
        return "Mission already active. Disengage first."
    if not briefing.strip():
        return "No mission briefing provided."

    _ms.reset_for_new_mission()

    # Always reset Cosmos context at mission start — belt-and-suspenders.
    # stop_mission() does this too, but if the previous mission ended
    # unexpectedly (exception, crash) this ensures a clean slate regardless.
    from cosmos import reset_mission_context
    reset_mission_context()

    try:
        from avoidance import reset_avoid_counter
        reset_avoid_counter()
    except ImportError as _exc:
        log.debug(f"avoidance module not loaded: {_exc}")

    # ── Load YAML metadata if mission_name given ──────────────────────────────
    _ms.mission_alarm_type     = AlarmType.HAZARD
    _ms.mission_target_objects = []
    _ms.mission_flags          = {}

    if mission_name:
        yaml_data = load_mission_file(mission_name)
        if yaml_data:
            raw_alarm = yaml_data.get("alarm_type", AlarmType.HAZARD)
            # Normalize "none" string → AlarmType.NONE so sound_alarm handles it correctly
            if str(raw_alarm).lower() in ("none", "null", ""):
                raw_alarm = AlarmType.NONE
            _ms.mission_alarm_type     = raw_alarm
            _ms.mission_target_objects = yaml_data.get("target_objects", [])
            _ms.mission_flags          = {k: v for k, v in yaml_data.items()
                                       if k not in ("briefing", "name", "alarm_type",
                                                    "target_objects")}
            log.info(f"Mission YAML loaded: alarm={_ms.mission_alarm_type} "
                     f"targets={_ms.mission_target_objects} flags={list(_ms.mission_flags)}")

    # ── Parse mission into steps ──────────────────────────────────────────────
    _ms.mission_steps    = _parse_mission_steps(briefing)
    _ms.current_step_idx = 0
    step_summaries    = [f"[{s.step_num}] {s.target} → {s.action}" for s in _ms.mission_steps]

    # ── Start mission log ─────────────────────────────────────────────────────
    start_mission_log(briefing[:60], steps=step_summaries)
    log_mission_event("mission_start", briefing[:200])

    # ── Update Cosmos system prompt with first step ───────────────────────────
    first_step = _current_step()
    if first_step and first_step.target != "target":
        set_mission_briefing(
            f"CURRENT STEP 1 of {len(_ms.mission_steps)}: "
            f"Find {first_step.target} and {first_step.action.replace('_', ' ')}.\n"
            f"Original mission: {briefing}"
        )
    else:
        set_mission_briefing(briefing)

    motors.pantilt(0, -5)   # slight downward tilt — see ground objects at normal range
    motors.lights(0, 0)    # LEDs off — only turn on if scene is pitch black
    time.sleep(0.5)

    # GAP 3 FIX: KV cache warm-up — force vLLM to prefill and cache the full
    # system prompt (including mission briefing) before the real ack call.
    # Cost: one tiny request (~1s). Benefit: every subsequent Cosmos call this
    # mission pays only the delta (scene snapshot), not the full system prompt.
    # Cuts TTFT on Orin Nano W4A16 2B from ~1.5s to ~300ms per call.
    try:
        from cosmos import _system_prompt as _sys_p
        _warmup_payload = {
            "model": COSMOS_MODEL,
            "messages": [
                {"role": "system", "content": _sys_p},
                {"role": "user",   "content": [{"type": "text", "text": "ready"}]},
            ],
            "max_tokens": 1,
            "temperature": 0.0,
        }
        requests.post(VLLM_URL, json=_warmup_payload, timeout=15)
        log.info("\u2705 vLLM KV cache warmed for this mission's system prompt")
    except Exception as _wu_exc:
        log.debug(f"KV cache warm-up skipped ({_wu_exc}) — non-fatal")

    step_info = f"I have {len(_ms.mission_steps)} step{'s' if len(_ms.mission_steps) > 1 else ''}: {', '.join(step_summaries)}." if len(_ms.mission_steps) > 1 else ""
    ack = ask_cosmos_plain(
        f"Mission briefing:\n\"{briefing}\"\n\n"
        + (f"Parsed steps: {step_info}\n\n" if step_info else "")
        + "Acknowledge in 2-3 sentences. State your first action. Be concise.",
        max_tokens=150
    )
    eric_say(ack)
    log_mission_event("mission_acknowledged", ack[:150])

    if len(_ms.mission_steps) > 1:
        _ui("log", f"Multi-step mission: {' → '.join(s.target for s in _ms.mission_steps)}")

    _ms.mission_active = True
    _ms.mission_state  = State.SEARCHING
    _ui("status", "SEARCHING")
    motors.oled(0, "ERIC ACTIVE")
    motors.oled(1, "Searching...")

    threading.Thread(target=_mission_loop, daemon=True).start()
    return ack


def stop_mission():
    """
    Cleanly stop the current mission and fully reset all runtime state.

    Fixes applied here:
      0. _ms.reset_for_new_mission() + YAML field clear — wipes conversation
         history, step engine, YOLO state, counters, and all YAML-loaded fields
         so nothing bleeds into the next mission.
      1. Cancel pending_nav future — prevents stale nav result bleeding into
         the next mission's first loop iteration.
      2. stop_alarm() — ensures siren/alert from a find doesn't carry into
         next mission startup.
      3. Recycle _cosmos_executor — cancels all in-flight Cosmos threads and
         creates a fresh pool. Root cause of Jetson overload after 3 missions:
         zombie futures accumulate, burning GPU/CPU while new ones queue up.
      4. Flush vLLM KV cache — releases GPU memory held by the previous
         mission's context. Biggest contributor to slowdown by mission 3.
      5. reset_mission_context() — wipes _system_prompt back to base so the
         next mission's acknowledgement call has zero memory of this mission.
      6. gc.collect() — releases numpy arrays, base64 frames, and dicts still
         referenced by completed futures.
    """
    global _cosmos_executor

    _ms.mission_active = False
    _ms.mission_state  = State.IDLE
    motors.stop()

    # 0. Full _ms wipe — clears conversation_history, step engine, YOLO state,
    #    counters, and YAML-loaded fields so nothing bleeds into the next mission.
    #    reset_for_new_mission() handles most fields; YAML fields need explicit clear.
    _ms.reset_for_new_mission()
    _ms.mission_steps          = []
    _ms.current_step_idx       = 0
    _ms.mission_target_objects = []
    _ms.mission_flags          = {}
    _ms.mission_alarm_type     = AlarmType.HAZARD
    log.info("stop_mission: _ms fully reset — no state bleeds into next mission")

    # 1. Cancel pending nav future
    if _ms.pending_nav is not None:
        _ms.pending_nav.cancel()
        _ms.pending_nav = None
        log.info("stop_mission: pending nav future cancelled")

    # 2. Stop any running alarm
    try:
        stop_alarm()
    except Exception as _exc:
        log.debug(f"stop_alarm error: {_exc}")

    # 3. Recycle executor — kills all zombie Cosmos threads
    try:
        _cosmos_executor.shutdown(wait=False, cancel_futures=True)
    except TypeError:
        _cosmos_executor.shutdown(wait=False)   # Python < 3.9
    _cosmos_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="cosmos"
    )
    log.info("stop_mission: cosmos executor recycled — zombie threads cleared")

    # 4. Flush vLLM KV cache
    try:
        import requests as _req
        vllm_base = VLLM_URL.rstrip("/").rsplit("/", 1)[0]
        _req.post(f"{vllm_base}/reset_prefix_cache", timeout=5)
        log.info("stop_mission: vLLM KV cache flushed")
    except Exception as _exc:
        log.debug(f"vLLM cache flush skipped ({_exc})")

    # 5. Reset Cosmos system prompt — no cross-mission memory
    from cosmos import reset_mission_context
    reset_mission_context()

    # 6. Python GC
    import gc; gc.collect()
    log.info("stop_mission: GC collected")

    # Cancel any in-progress Nav2 goal
    try:
        from config import USE_NAV2
        if USE_NAV2:
            from nav2 import cancel_goal, nav2_available
            if nav2_available():
                cancel_goal()
    except Exception as _exc:
        log.debug(f"nav2 unavailable: {_exc}")

    motors.lights(0, 0)
    motors.pantilt(0, -5)
    motors.oled(0, "ERIC STOPPED")
    motors.oled(1, "")
    eric_say("Mission disengaged. All systems halted.")
    log_mission_event("mission_stopped", "operator abort")
    end_mission_log(completed=False)
    _ui("status", "IDLE")


def resume_after_interaction():
    if _ms.mission_active:
        _ms.reset_counters()
        try:
            from avoidance import reset_avoid_counter
            reset_avoid_counter()
        except ImportError as _exc:
            log.debug(f"avoidance module not loaded: {_exc}")
        _ms.mission_state = State.SEARCHING
        motors.pantilt(0, -5)   # ground-looking default
        if _safe_to_fwd():
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

VOID / DROP DETECTION — CRITICAL:
- Floor texture suddenly disappears ahead → void_ahead = true
- You can see a stair edge, step edge, or ledge in the lower half of the frame → void_ahead = true
- The floor ends and there is open space or a lower level visible → void_ahead = true
- If sensor data includes a VOID WARNING → void_ahead = true, action = stop
- NEVER set action=forward when void_ahead=true

OUTPUT: A single JSON object. Every field is REQUIRED. Use ONLY the exact field names shown.
STRING fields must be a single word from the options listed — NOT a list, NOT a dict, NOT null.
BOOLEAN fields must be true or false.

Example output (copy this structure exactly, change values to match what you see):
{
  "wall_ahead": false,
  "obstacle_close": false,
  "small_obstacle": false,
  "void_ahead": false,
  "person_visible": false,
  "action": "forward",
  "physical_reasoning": "Path is clear ahead for at least two meters."
}

Now analyze the frames and output ONLY the JSON object above. No markdown. No explanation. No extra fields.
"""

SCAN_360_PROMPT = """
You are a tracked ground robot. These images are from a full 360-degree scan — pan-tilt sweep.
You are completely stopped.

STEP 1 — VOID/DROP CHECK (HIGHEST PRIORITY):
- Stair edges, step edges, ledge lips, holes, gaps → void_ahead = true
- Floor texture ends and open space / lower level is visible → void_ahead = true
- Floor suddenly much further away or missing → void_ahead = true
- If sensor data includes VOID WARNING → void_ahead = true
- NEVER set clearest_direction toward a void

STEP 2 — OBSTACLE SAFETY: Which direction has the most open space?

STEP 3 — MISSION TARGET: People, robots, slippers, shoes — even partially visible counts.
Set target_visible=true if 50%+ confident you can see a specific distinguishing feature
in this frame (not general knowledge about what the target looks like).
CRITICAL: If the mission overlay says you are looking for a person and you can see ANY
person in the frame — set target_visible=true and object='person'. Do not suppress a
visible person just because you cannot verify every appearance detail from this distance.
The robot will do a hardware+webcam confirmation pass anyway — do NOT hold back a detection.

STEP 4 — SPEAK: One excited sentence if target found, otherwise null.

OUTPUT: A single JSON object. Use ONLY these exact field names — no others:
{
  "object": "clear",
  "object_name": null,
  "terrain": "tiles",
  "distance": "far",
  "in_my_path": false,
  "wall_ahead": false,
  "obstacle_close": false,
  "small_obstacle": false,
  "void_ahead": false,
  "target_visible": false,
  "target_direction": "unknown",
  "clearest_direction": "front",
  "action": "forward",
  "speak": null,
  "physical_reasoning": "No target found. Hallway ahead is the clearest direction.",
  "mission_complete": false
}

"object" must be ONE word: person | robot | slipper | shoe | obstacle | wall | clear | unknown | pokemon | figure | animal
"speak" = speech output. NOT "speaker". NOT "speech". NOT "tts".
"physical_reasoning" = reasoning. NOT "reasoning". NOT "explanation".
"target_visible" = detection flag. NOT "target_visibility". NOT "target_found".
No markdown. No explanation. Output ONLY the JSON object.
"""

VIDEO_SWEEP_360_PROMPT = """
You are a tracked ground robot. These frames are from a continuous 360° panoramic sweep
captured while my chassis rotated — ordered left to right, covering the full environment.
I am an observation/exploration robot — I am NOT looking for one specific target.

Study the full sweep and report:
1. HAZARDS — stairs, drops, voids, obstacles, blocked paths (highest priority)
2. ENVIRONMENT — terrain types, room layout, notable features or changes
3. OBSERVATIONS — people, animals, objects of interest
4. BEST DIRECTION — clearest safe path to continue

"speak": One vivid sentence narrating what you see — imagine describing it to an audience.
"physical_reasoning": Summarize the environment and why you chose that direction.

Output ONLY this JSON — no markdown, no extra fields:
{
  "object": "clear",
  "object_name": null,
  "terrain": "tiles",
  "distance": "far",
  "in_my_path": false,
  "wall_ahead": false,
  "obstacle_close": false,
  "small_obstacle": false,
  "void_ahead": false,
  "target_visible": false,
  "target_direction": "unknown",
  "clearest_direction": "front",
  "action": "forward",
  "speak": null,
  "physical_reasoning": "Open area ahead, proceeding forward.",
  "mission_complete": false
}

"object": person|robot|animal|obstacle|wall|clear|unknown
"action": forward|stop|turn_left|turn_right
Output ONLY the JSON object. No explanation before or after.
"""


QUICK_SCAN_PROMPT = """
You are a tracked ground robot. You are stopped. Analyze the frames and sensor data.

RULES (in priority order):
1. VOID/DROP — stair edge, hole, floor ending → void_ahead=true, action=stop
2. OBSTACLE — object <60cm ahead OR filling lower frame → obstacle_close=true, action=stop
3. TARGET — slipper/shoe/person/robot visible ANYWHERE, even partially, even at an angle,
   even at the edge of frame → target_visible=true.
   CRITICAL: If the mission overlay above says you are looking for a specific person and you
   can see ANY person in the frame — set target_visible=true and object='person'.
   Do NOT set target_visible=false when you can clearly see a person standing in the room.
4. Otherwise → action=forward

"speak": one SHORT plain sentence if target found, otherwise null. No backticks. No lists.
"physical_reasoning": one plain sentence describing what you see. No backticks.

Output ONLY this JSON — no markdown, no extra fields:
{"object":"clear","object_name":null,"terrain":"tiles","distance":"far","in_my_path":false,"wall_ahead":false,"obstacle_close":false,"small_obstacle":false,"void_ahead":false,"target_visible":false,"target_direction":"unknown","clearest_direction":"front","action":"forward","speak":null,"physical_reasoning":"Path clear.","mission_complete":false}

"object": person|robot|slipper|shoe|obstacle|wall|clear|unknown
"distance": near|mid|far
"target_direction": front|left|right|unknown
"clearest_direction": front|left|right
"action": forward|stop|turn_left|turn_right
Output ONLY the JSON object. No explanation before or after.
"""

_SCAN_FALLBACK = {
    "object": "unknown", "object_name": None, "terrain": "clear",
    "distance": "far", "in_my_path": False, "wall_ahead": False,
    "small_obstacle": False, "void_ahead": False, "target_visible": False,
        "target_direction": "unknown", "clearest_direction": "front",
    "action": "stop", "speak": None,   # SAFE default — never forward on failure
    "physical_reasoning": "", "mission_complete": False
}

_NAV_FALLBACK = {
    "wall_ahead": False, "obstacle_close": False, "small_obstacle": False,
    "void_ahead": False, "person_visible": False,
    "action": "stop", "physical_reasoning": ""  # SAFE default — stop on nav failure
}


# ─── Navigation Check (while moving) ─────────────────────────────────────────

# Nav clip settings — tune these
NAV_CLIP_DURATION = 10.0  # seconds of video per nav check
NAV_CLIP_FPS      = 2     # frames per second (10s x 2fps = 20 frames to Cosmos)
NAV_IMAGE_INTERVAL = 6.0  # seconds between nav image checks while moving

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
    except Exception as _exc:  # lidar
        log.debug(f"lidar unavailable: {_exc}")

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
    except Exception as _exc:  # oakd/yolo
        log.debug(f"oakd/yolo unavailable: {_exc}")

    # ── Hardware safety: void/drop check ──────────────────────────────────
    void = _void_check()
    if void["void"]:
        motors.stop()
        log.warning(f"🕳️  Nav check: void detected ({void['source']}): {void['reason']}")
        log_action("VOID_BLOCK_NAV", f"{void['source']}: {void['reason']}")
        _ui("log", f"🕳️  VOID/DROP detected — stopping! ({void['source']})")
        _ui("status", "VOID DETECTED — STOPPED")
        motors.oled(0, "VOID AHEAD!")
        motors.oled(1, "STOP")
        return {**_NAV_FALLBACK, "wall_ahead": True, "obstacle_close": True,
                "action": "stop",
                "physical_reasoning": f"Void/drop detected by {void['source']}: {void['reason']}"}

    frame = capture_frame(CAMERA_PANTILT, 320, 240)
    if not frame:
        return dict(_NAV_FALLBACK)

    sensor_ctx = _sensor_context()

    NAV_IMAGE_PROMPT = f"""{sensor_ctx}You are a tracked ground robot moving forward. This is a single frame from your forward camera.

Check for immediate safety hazards in this order:

VOID / DROP (HIGHEST PRIORITY — look at lower third of frame):
- Stair edge, step lip, ledge, hole, gap, or floor texture abruptly ending → void_ahead = true
- Open air or a lower level visible where the floor should be → void_ahead = true
- If sensor data above includes a VOID or FLOOR DROP WARNING → void_ahead = true
- When void_ahead = true: wall_ahead = true, action = stop, NEVER forward

OBSTACLES:
- Wall or large object filling the lower 40% of frame → wall_ahead = true
- Any object within ~60cm directly ahead → obstacle_close = true
- Small ground obstacle (cables, edges) → small_obstacle = true

PEOPLE:
- Person or robot visible anywhere → person_visible = true

If sensor data above shows LIDAR STOP ZONE or OAK-D OBSTACLE, set wall_ahead=true and action=stop.

ACTION RULE: "action" must be EXACTLY one of: "forward" or "stop"
- "stop" if wall_ahead=true OR obstacle_close=true OR void_ahead=true
- "forward" in all other cases

OUTPUT: A single JSON object. Every field is REQUIRED.
{{
  "wall_ahead": false,
  "obstacle_close": false,
  "small_obstacle": false,
  "void_ahead": false,
  "person_visible": false,
  "action": "forward",
  "physical_reasoning": "Path ahead is clear with no obstacles or drops visible."
}}

No markdown. No explanation. No extra fields.
"""
    try:
        payload = {
            "model": COSMOS_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}},
                    {"type": "text", "text": NAV_IMAGE_PROMPT.strip()}
                ]}
            ],
            "max_tokens": 120,
            "temperature": 0.7,
            "top_p": 0.8,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
        }
        r = requests.post(VLLM_URL, json=payload, timeout=30)
        r.raise_for_status()
        response = r.json()["choices"][0]["message"]["content"].strip()
        log_ai(NAV_IMAGE_PROMPT[-200:], response, label="NAV_CHECK")
        result = _parse_json(response, dict(_NAV_FALLBACK), label="NAV CHECK")

        # ── Physical reasoning text gate: catch wall detections Cosmos missed in JSON ──
        reasoning_text = (result.get("physical_reasoning") or "").lower()
        wall_keywords = ["blocks my way", "cannot proceed", "no clear path",
                         "path is blocked", "blocking my path", "wall blocks",
                         "wall ahead", "directly blocking"]
        if any(kw in reasoning_text for kw in wall_keywords) and not result.get("wall_ahead"):
            log.warning(f"⚠️  NAV_CHECK: Cosmos text says wall but JSON says False — overriding: {reasoning_text[:80]}")
            result["wall_ahead"] = True
            result["action"] = "stop"

        # ── Void gate: Cosmos sees a drop — stop immediately ──────────────
        if result.get("void_ahead"):
            motors.stop()
            log.warning(f"🕳️  NAV_CHECK: Cosmos sees void ahead — stopping")
            log_action("VOID_COSMOS_NAV", result.get("physical_reasoning", ""))
            _ui("log", f"🕳️  Cosmos: void/drop detected — stopping!")
            _ui("status", "VOID DETECTED — STOPPED")
            motors.oled(0, "VOID AHEAD!")
            motors.oled(1, "STOP")
            return {**result, "wall_ahead": True, "obstacle_close": True, "action": "stop"}

        # ── Eye-contact gate: only greet if person is close AND facing Eric ──
        if result.get("person_visible") and _ms.mission_active:
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
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ec_frame}"}},
                                {"type": "text", "text": ec_prompt}
                            ]}
                        ],
                        "max_tokens": 60,
                        "temperature": 0.7,
                        "top_p": 0.8,
                        "presence_penalty": 1.5,
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
                greeting = ask_cosmos_plain(
                    "Someone is looking at you from close range. "
                    "Greet them warmly and ask if they can help with your mission. 1-2 sentences.",
                    max_tokens=60
                )
                eric_say(greeting)
            else:
                _ui("log", f"Person spotted but not close/facing ({ec_result.get('reasoning','')}) — continuing")
                if _safe_to_fwd():
                    motors.forward(MOTOR_SPEED_SLOW)

        return result
    except Exception as e:
        log_exception("_nav_check", e)
        return dict(_NAV_FALLBACK)


def _nav_check_async() -> dict:
    """
    Fire-and-forget video nav check — never blocks Eric.

    Strategy (Option 1 + 2 combined):
      - Snapshot the last 6 frames from the rolling buffer instantly (no wait).
      - Fire a Cosmos call async using those frames for temporal context.
      - Return the PREVIOUS completed result immediately so the mission loop
        can act on it right now.
      - Next cycle: collect the result that just finished, fire another call.

    Hardware safety gates (LiDAR, OAK-D, void) still run synchronously first —
    they are fast (no Cosmos) and must never be delayed.

    Falls back to _nav_check() (single frame, synchronous) if buffer is empty.
    """

    # ── Hardware safety always first — no Cosmos delay ────────────────────
    try:
        from lidar import obstacle_close as lidar_close
        if lidar_close():
            log_action("LIDAR_STOP", "obstacle within 0.30m")
            result = {**_NAV_FALLBACK, "wall_ahead": True, "obstacle_close": True,
                      "action": "stop",
                      "physical_reasoning": "LiDAR: obstacle within 0.30m stop zone"}
            _ms.last_nav_result = result
            return result
    except Exception as _exc:  # lidar
        log.debug(f"lidar unavailable: {_exc}")

    try:
        from oakd import get_front_depth, oakd_available
        if oakd_available():
            d = get_front_depth()
            if d is not None and d < 0.30:
                log_action("OAKD_STOP", f"obstacle at {d:.2f}m")
                result = {**_NAV_FALLBACK, "wall_ahead": True, "obstacle_close": True,
                          "action": "stop",
                          "physical_reasoning": f"OAK-D: obstacle at {d:.2f}m"}
                _ms.last_nav_result = result
                return result
    except Exception as _exc:  # oakd/yolo
        log.debug(f"oakd/yolo unavailable: {_exc}")

    void = _void_check()
    if void["void"]:
        motors.stop()
        log_action("VOID_BLOCK_NAV", f"{void['source']}: {void['reason']}")
        _ui("log", f"🕳️  VOID detected — stopping! ({void['source']})")
        result = {**_NAV_FALLBACK, "wall_ahead": True, "obstacle_close": True,
                  "action": "stop",
                  "physical_reasoning": f"Void: {void['reason']}"}
        _ms.last_nav_result = result
        return result

    # ── Collect previous Cosmos result if ready ───────────────────────────
    if _ms.pending_nav is not None and _ms.pending_nav.done():
        try:
            raw = _ms.pending_nav.result(timeout=0)
            parsed = _parse_json(raw, dict(_NAV_FALLBACK), label="NAV_ASYNC")
            log_ai("nav_check_async", raw, label="NAV_ASYNC")

            # Void gate on Cosmos result
            if parsed.get("void_ahead"):
                motors.stop()
                log_action("VOID_COSMOS_NAV_ASYNC", parsed.get("physical_reasoning", ""))
                _ui("log", "🕳️  Cosmos (async): void ahead — stopping!")
                parsed.update({"wall_ahead": True, "obstacle_close": True, "action": "stop"})

            # Eye-contact gate — person visible in buffered frames
            if parsed.get("person_visible") and _ms.mission_active:
                motors.stop()
                _ui("log", "👤 Person in video frames — checking proximity...")
                ec_frame = capture_frame(CAMERA_PANTILT, 320, 240)
                if ec_frame:
                    ec_prompt = (
                        "Is there a person BOTH close (within ~1.5m) AND facing you?\n"
                        "JSON only: {\"close_and_facing\": true_or_false, \"reasoning\": \"one sentence\"}"
                    )
                    try:
                        ec_payload = {
                            "model": COSMOS_MODEL,
                            "messages": [
                                {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": [
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ec_frame}"}},
                                    {"type": "text", "text": ec_prompt}
                                ]}
                            ],
                            "max_tokens": 60, "temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5,
                        }
                        ec_r = requests.post(VLLM_URL, json=ec_payload, timeout=20)
                        ec_r.raise_for_status()
                        ec_result = _parse_json(
                            ec_r.json()["choices"][0]["message"]["content"].strip(),
                            {"close_and_facing": False}, "EYE_CONTACT"
                        )
                        if ec_result.get("close_and_facing"):
                            _ui("log", f"👁️  Eye contact confirmed — greeting!")
                            log_mission_event("person_greeted", ec_result.get("reasoning", ""))
                            greeting = ask_cosmos_plain(
                                "Someone is close and looking at you. Greet them and ask if they can help with your mission. 1-2 sentences.",
                                max_tokens=60
                            )
                            eric_say(greeting)
                        else:
                            _ui("log", f"Person not close/facing — continuing")
                            if _safe_to_fwd():
                                motors.forward(MOTOR_SPEED_SLOW)
                    except Exception as e:
                        log_exception("eye_contact_async", e)
                        if _safe_to_fwd():
                            motors.forward(MOTOR_SPEED_SLOW)

            _ms.last_nav_result = parsed
        except Exception as e:
            log_exception("_nav_check_async collect", e)
        _ms.pending_nav = None

    # ── Fire next Cosmos call async — does not block ──────────────────────
    if _ms.pending_nav is None:
        frames = get_buffered_frames(CAMERA_PANTILT, n=6)
        if not frames:
            # Buffer cold — fall back to synchronous single-frame check
            _ui("log", "📷 Nav check (sync fallback)...")
            return _nav_check()

        sensor_ctx = _sensor_context()
        NAV_VIDEO_PROMPT = f"""{sensor_ctx}You are a tracked ground robot moving forward. These are the last few seconds of footage from your forward camera — analyse them for changes over time.

VOID / DROP (HIGHEST PRIORITY — look at lower third of each frame):
- Stair edge, hole, gap, or floor abruptly ending → void_ahead = true
- If sensor data shows VOID or FLOOR DROP WARNING → void_ahead = true
- When void_ahead = true: wall_ahead = true, action = stop

OBSTACLES:
- Wall or large object ahead → wall_ahead = true
- Object within ~60cm → obstacle_close = true
- Small ground obstacle → small_obstacle = true

PEOPLE: person or robot visible anywhere → person_visible = true

ACTION: "stop" if wall_ahead OR obstacle_close OR void_ahead. "forward" otherwise.

OUTPUT — single JSON, no markdown:
{{
  "wall_ahead": false,
  "obstacle_close": false,
  "small_obstacle": false,
  "void_ahead": false,
  "person_visible": false,
  "action": "forward",
  "physical_reasoning": "one sentence"
}}"""

        def _call():
            try:
                content = [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}}
                    for f in frames
                ]
                content.append({"type": "text", "text": NAV_VIDEO_PROMPT.strip()})
                payload = {
                    "model": COSMOS_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": content}
                    ],
                    "max_tokens": 120, "temperature": 0.7, "top_p": 0.8,
                    "presence_penalty": 1.5, "repetition_penalty": 1.0,
                }
                r = requests.post(VLLM_URL, json=payload, timeout=30)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                log_exception("nav_async_call", e)
                return ""

        _ms.pending_nav = _cosmos_executor.submit(_call)
        _ui("log", f"📹 Async video nav ({len(frames)} frames) fired — acting on last result")

    # ── Return last known result immediately — never waits ────────────────
    if _ms.last_nav_result:
        return _ms.last_nav_result

    # Very first call — no result yet, fall back to fast single-frame check
    _ui("log", "📷 Nav check (first call, sync)...")
    result = _nav_check()
    _ms.last_nav_result = result
    return result


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
    except Exception as _exc:  # optional component
        log.debug(f"optional component error: {_exc}")
        return False

def _approach_scan() -> dict:
    """
    Lightweight Cosmos scan used ONLY during _approach_target().

    Differences from _quick_scan():
      1. NO webcam confirmation pass — target already confirmed at scan time.
         The second Cosmos call was causing "approach confirms → webcam rejects"
         false negatives on a 2B model viewing a tilted/partial target.
      2. Prompt explicitly lowers threshold: "even a partial view counts".
         The robot is already moving toward a confirmed sighting — it does not
         need 60% certainty to keep going.
      3. Always single short clip (1.5 s) — keeps approach loop latency low.

    Hardware void/obstacle checks still run first (same as _quick_scan).
    """
    # ── Hardware void pre-check ───────────────────────────────────────────────
    hw_void = _void_check()
    if hw_void["void"]:
        log.warning(f"🕳️  _approach_scan pre-check: void ({hw_void['source']}): {hw_void['reason']}")
        log_action("VOID_PRECHECK_APPROACH", hw_void["reason"])
        return {
            **_SCAN_FALLBACK,
            "wall_ahead": True, "void_ahead": True, "action": "stop",
            "physical_reasoning": f"Hardware void pre-check: {hw_void['reason']}"
        }

    motors.pantilt(0, -5)
    motors.lights(0, 0)
    time.sleep(0.25)

    clip_frames = capture_frames_video(CAMERA_PANTILT, duration=1.5, fps_sample=2.0)
    if not clip_frames:
        f = _capture_sharp(CAMERA_PANTILT)
        clip_frames = [f] if f else []

    if not clip_frames:
        return dict(_SCAN_FALLBACK)

    if _is_pitch_black(clip_frames[-1]):
        motors.lights(base=180, head=255)
        time.sleep(0.3)
        extra = _capture_sharp(CAMERA_PANTILT)
        if extra:
            clip_frames.append(extra)
        motors.lights(0, 0)

    sensor_ctx = _sensor_context()
    mission_ov = _get_mission_scan_overlay()

    # Approach-specific prompt: lower threshold, remind Cosmos we are closing in
    approach_note = (
        "APPROACH MODE: You are driving toward a target you have already spotted. "
        "Even a partial, angled, or edge view of the target counts as visible. "
        "Set target_visible=true if there is ANY reasonable chance the target is present.\n\n"
    )
    prompt = mission_ov + approach_note + (sensor_ctx + QUICK_SCAN_PROMPT
                                           if sensor_ctx else QUICK_SCAN_PROMPT)

    try:
        print(f"\n🎯 APPROACH SCAN — {len(clip_frames)} frames (no webcam confirm)...")
        response = _cosmos_frames(clip_frames, prompt, max_tokens=200, temp=0.2)
        result   = _parse_json(response, dict(_SCAN_FALLBACK), label="APPROACH SCAN")

        # ── Sensor hard-gates (same as _quick_scan) ───────────────────────────
        try:
            from lidar import obstacle_close as lidar_close
            if lidar_close():
                result["wall_ahead"]     = True
                result["obstacle_close"] = True
                result["action"]         = "stop"
        except Exception as _exc:  # lidar
            log.debug(f"lidar unavailable: {_exc}")

        try:
            from oakd import get_front_depth, oakd_available
            if oakd_available():
                d = get_front_depth()
                if d is not None and d < 0.30:
                    result["wall_ahead"]     = True
                    result["obstacle_close"] = True
                    result["action"]         = "stop"
        except Exception as _exc:  # oakd/yolo
            log.debug(f"oakd/yolo unavailable: {_exc}")

        return result
    except Exception as e:
        log_exception("_approach_scan", e)
        return dict(_SCAN_FALLBACK)


def _quick_scan() -> dict:
    """
    Industrial-standard stopped scan.

    Strategy:
      1. Hardware void pre-check (LiDAR + OAK-D) — no Cosmos, instant.
      2. Pan-tilt at TILT_SCAN (5°) — wide angle, the primary search sensor.
         Short 2-second video clip (3 frames) — beats single frame for accuracy.
         Cosmos asked to find target and assess path.
      3. If candidate found → webcam confirmation zoom shot before committing.
         Webcam (longer focal length) only fires here — resource-efficient.
      4. Sensor overrides applied last (LiDAR/OAK-D hard-gate Cosmos result).

    Tilt at 5° shows ground 1-3m ahead — the correct industrial search angle.
    30° steep-down is only for void detection, not target search.
    """
    # ── 1. Hardware void pre-check ────────────────────────────────────────────
    hw_void = _void_check()
    if hw_void["void"]:
        log.warning(f"🕳️  _quick_scan pre-check: void ({hw_void['source']}): {hw_void['reason']}")
        log_action("VOID_PRECHECK", hw_void["reason"])
        return {
            **_SCAN_FALLBACK,
            "wall_ahead": True, "void_ahead": True, "action": "stop",
            "physical_reasoning": f"Hardware void pre-check: {hw_void['reason']}"
        }

    # ── 2. Pan-tilt wide-angle scan at 10° (upward tilt to see standing people) ─
    # -5° looks at floor ~1m ahead at 40cm robot height — misses faces at 2m.
    # +10° looks at torso/face height for a standing person at 1.5-3m range.
    motors.pantilt(0, 10)
    motors.lights(0, 0)
    time.sleep(0.3)

    # Short video clip: 2s at 1.5fps = 3 frames
    # Motion context > single frame; same Cosmos token cost as one 720p frame
    clip_frames = capture_frames_video(CAMERA_PANTILT, duration=2.0, fps_sample=1.5)
    if not clip_frames:
        f = _capture_sharp(CAMERA_PANTILT)
        clip_frames = [f] if f else []

    if not clip_frames:
        return dict(_SCAN_FALLBACK)

    # Adaptive LED
    if _is_pitch_black(clip_frames[-1]):
        motors.lights(base=180, head=255)
        time.sleep(0.3)
        extra = _capture_sharp(CAMERA_PANTILT)
        if extra:
            clip_frames.append(extra)
        motors.lights(0, 0)

    sensor_ctx = _sensor_context()
    mission_ov = _get_mission_scan_overlay()
    prompt = mission_ov + (sensor_ctx + QUICK_SCAN_PROMPT
                           if sensor_ctx else QUICK_SCAN_PROMPT)

    try:
        print(f"\n📷 QUICK SCAN — {len(clip_frames)} frames (pan-tilt 10°)...")
        response = _cosmos_frames(clip_frames, prompt, max_tokens=300, temp=0.3)
        result = _parse_json(response, dict(_SCAN_FALLBACK), label="QUICK SCAN")

        # ── 3a. Person seen but not flagged as target — cross-check with LiDAR ──
        # The 2B model frequently outputs object='person', target_visible=False
        # because it copies the default JSON example rather than following the
        # mission overlay instructions. If:
        #   • Cosmos says object='person' (it does see someone), AND
        #   • This is a find-and-greet mission (AlarmType.NONE), AND
        #   • LiDAR confirms something is physically present at <2m
        # → treat the person as the target. The visual description confirmation
        # in _confirm_and_photograph_target() will still gate the final greeting.
        if (not result.get("target_visible")
                and result.get("object") == "person"
                and (
                    _ms.mission_alarm_type == AlarmType.NONE
                    or str(_ms.mission_alarm_type).lower() in ("none", "null", "")
                )):
            try:
                from lidar import get_status as _lidar_s, lidar_available
                if lidar_available():
                    _ls = _lidar_s()
                    _front_m = _ls.get("front_arc_min_m", 999)
                    if _front_m < 2.0:
                        log.info(
                            f"QUICK_SCAN: object=person + LiDAR={_front_m:.2f}m "
                            f"→ promoting target_visible=True (2B model failed overlay)"
                        )
                        result["target_visible"] = True
                        result["object_name"]    = result.get("object_name") or "person"
            except Exception as _lxc:
                log.debug(f"lidar cross-check: {_lxc}")

        # ── 3. Candidate found — webcam confirmation ───────────────────────────
        # Wide angle detected something → zoom in with webcam to confirm.
        # Only now do we pay the cost of a second camera frame + Cosmos call.
        if result.get("target_visible"):
            _ui("log", "🔍 Candidate found — webcam confirmation...")
            wc = capture_frame(CAMERA_WEBCAM, 640, 480)
            if wc:
                if _is_pitch_black(wc):
                    motors.lights(base=180, head=255)
                    time.sleep(0.2)
                    wc = capture_frame(CAMERA_WEBCAM, 640, 480) or wc
                    motors.lights(0, 0)
                confirm_frames = clip_frames + [wc]
                confirm_prompt = (
                    mission_ov +
                    "CONFIRMATION: You flagged a possible target. "
                    "The last image is a zoom shot from the close-up webcam. "
                    "Confirm: is the target actually present? "
                    "Set target_visible=true only if 60%+ confident.\n\n"
                ) + (sensor_ctx + QUICK_SCAN_PROMPT if sensor_ctx else QUICK_SCAN_PROMPT)
                try:
                    confirm_resp = _cosmos_frames(confirm_frames, confirm_prompt,
                                                  max_tokens=200, temp=0.1)
                    confirmed = _parse_json(confirm_resp, dict(_SCAN_FALLBACK),
                                            label="QUICK SCAN CONFIRM")
                    if confirmed.get("target_visible"):
                        _ui("log", "✅ Target confirmed by webcam")
                        result = confirmed
                    else:
                        _ui("log", "❌ Webcam: false positive — continuing search")
                        result["target_visible"] = False
                except Exception as e:
                    log_exception("quick_scan_confirm", e)
                    # Keep original result if confirmation call fails

        # ── 4. Physical reasoning text gate (quick_scan) ────────────────────
        qs_reasoning = (result.get("physical_reasoning") or "").lower()
        # Strip backtick noise before keyword check — model wraps garbage in backticks
        qs_reasoning = qs_reasoning.replace("`", "").strip()
        qs_wall_keywords = ["blocks my way", "cannot proceed", "no clear path",
                             "path is blocked", "blocking my path", "wall blocks",
                             "wall ahead", "directly blocking"]
        if any(kw in qs_reasoning for kw in qs_wall_keywords) and not result.get("wall_ahead"):
            log.warning(f"⚠️  QUICK_SCAN: Cosmos text says wall but JSON says False — overriding: {qs_reasoning[:80]}")
            result["wall_ahead"] = True
            result["action"] = "stop"

        # ── 5. Sensor hard-gates ─────────────────────────────────────────────
        try:
            from lidar import obstacle_close as lidar_close
            if lidar_close():
                log_action("LIDAR_OVERRIDE", "quick scan")
                result["wall_ahead"] = True
                result["obstacle_close"] = True
                result["action"] = "stop"
        except Exception as _exc:  # lidar
            log.debug(f"lidar unavailable: {_exc}")

        try:
            from oakd import get_front_depth, oakd_available
            if oakd_available():
                d = get_front_depth()
                if d is not None and d < 0.30:
                    log_action("OAKD_OVERRIDE", f"quick scan {d:.2f}m")
                    result["wall_ahead"] = True
                    result["obstacle_close"] = True
                    result["action"] = "stop"
        except Exception as _exc:  # oakd/yolo
            log.debug(f"oakd/yolo unavailable: {_exc}")

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
    except Exception as _exc:  # optional component
        log.debug(f"optional component error: {_exc}")
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
        time.sleep(0.8)  # raised from 0.5 — allow chassis vibration to fully damp
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
        except Exception as _exc:  # lidar
            log.debug(f"lidar unavailable: {_exc}")

        try:
            from oakd import get_front_depth, oakd_available
            if oakd_available():
                d = get_front_depth()
                if d is not None and d < 0.30:
                    result["wall_ahead"] = True
                    result["obstacle_close"] = True
                    result["action"] = "stop"
        except Exception as _exc:  # oakd/yolo
            log.debug(f"oakd/yolo unavailable: {_exc}")

        return result
    except Exception as e:
        log.error(f"Video scan error: {e} — falling back to quick scan")
        return _quick_scan()


def _scan_360_pantilt() -> dict:
    """
    360° pan-tilt sweep — async inference pipeline.

    Design:
      - Fixed tilt TILT_GROUND (-15°) throughout — ground-level focus, no tilt changes
        during sweep. Tilt only moves for webcam confirmation after a candidate found.
      - One sharp 320×240 frame per pan position — no video clips, no waiting for
        a clip to finish recording.
      - Cosmos called ASYNC immediately after each capture. Pan moves to the next
        position while inference runs — inference and movement fully overlap.
      - After all positions captured, collect all async results.
      - First position returning target_visible=True triggers confirmation:
          1. Pan-tilt back to the detected angle
          2. Webcam zoom shot (640×480) for confirmation
          3. Single confirmation Cosmos call (synchronous — decision point)
      - No final overview pass — redundant given per-position inference.

    Timing (Jetson Orin Nano, Cosmos 2B):
      Capture:  ~0.05s per frame
      Pan move: ~0.3s settle
      Inference: ~4-6s (overlaps with next pan move)
      Total for 7-position phase: ~3s capture + ~5s final wait = ~8s
      vs old design: 7 × (1.5s clip + 0.5s tilt + ~5s inference) = ~49s
    """
    _ms.mission_state = State.SCANNING_360
    _ui("status", "360 SCANNING")
    motors.oled(0, "360 Scan")
    motors.stop()
    time.sleep(0.5)  # raised from 0.2 — chassis damping before 360 sweep capture
    log.info("Starting pan-tilt 360 scan — async inference pipeline")
    log_mission_event("scan_360_start", "async sweep 7×30° + 180° chassis")

    # ── Constants ─────────────────────────────────────────────────────────────
    PAN_STEPS    = [-90, -60, -30, 0, 30, 60, 90]
    TILT_GROUND  = -15  # used for chassis turn alignment only
    TILT_SEARCH  = 10   # face/torso height for standing person at 1.5-3m
    PAN_SETTLE   = 0.40 # seconds after pantilt() before capture — raised from 0.25 for servo + frame settle

    # Webcam is zip-tied to the pan-tilt head, offset slightly to the LEFT.
    # When pan-tilt is at angle X, webcam center is at X + WEBCAM_PAN_OFFSET.
    # Compensate by panning WEBCAM_PAN_OFFSET degrees RIGHT during confirmation
    # so the webcam centers on the candidate rather than seeing it at the edge.
    # ── MEASURE THIS PHYSICALLY ──────────────────────────────────────────────
    # Place an object at 1m directly ahead (0°). Pan until webcam centers it.
    # That angle is your offset. Positive = webcam is left of pan-tilt center.
    # Example: if webcam centers at pan=+8°, set WEBCAM_PAN_OFFSET = 8
    WEBCAM_PAN_OFFSET = 8   # degrees — adjust after physical measurement

    def _pan_to_chassis_turn_sec(pan: int) -> float:
        return abs(pan) / 90.0 * TURN_90_SEC

    def _confirm_candidate(pan: int, scan_frame: str,
                           sensor_ctx: str, mission_ov: str) -> dict | None:
        """
        Candidate found at pan angle. Tilt pan-tilt to exact detected angle,
        capture webcam zoom shot, run synchronous confirmation call.
        Returns confirmed result dict or None if false positive.
        """
        _ui("log", f"🔍 Candidate at pan {pan:+d}° — aiming & webcam confirmation...")
        log_mission_event("candidate_found", f"pan={pan}")

        # Pan-tilt to detected angle, compensated for webcam physical offset.
        # Webcam is zip-tied to pan-tilt head but offset left — pan right by
        # WEBCAM_PAN_OFFSET degrees so webcam centers on the candidate.
        webcam_pan = max(-90, min(90, pan + WEBCAM_PAN_OFFSET))
        motors.pantilt(webcam_pan, TILT_SEARCH, speed=80)
        time.sleep(PAN_SETTLE + 0.1)

        wc_frame = capture_frame(CAMERA_WEBCAM, 640, 480)
        if wc_frame and _is_pitch_black(wc_frame):
            motors.lights(base=180, head=255)
            time.sleep(0.2)
            wc_frame = capture_frame(CAMERA_WEBCAM, 640, 480) or wc_frame
            motors.lights(0, 0)

        confirm_frames = [scan_frame]
        if wc_frame:
            confirm_frames.append(wc_frame)

        confirm_prompt = (
            mission_ov +
            "CONFIRMATION: Wide-angle pan-tilt camera flagged a possible target at this bearing. "
            "The first image is the wide-angle frame that flagged the candidate. "
            "The second image is from a narrow focal length webcam mounted slightly LEFT "
            "of the pan-tilt on the same servo head — it sees a narrower, more detailed "
            "view of the same scene. The target may appear toward the right side of the "
            "webcam frame due to the physical offset. "
            "Use BOTH images together to confirm or deny — the target may be clearer "
            "in one than the other. "
            "Set target_visible=true ONLY if you can identify a specific visual feature "
            "you actually see in the frame (shape, colour, marking) — not general knowledge. "
            "on what is visible. Do not inflate — a score above 0.55 must be justified "
            "by observable evidence in the frame. "
            "Be conservative — a false positive wastes mission time.\n\n"
        ) + (sensor_ctx + SCAN_360_PROMPT if sensor_ctx else SCAN_360_PROMPT)

        try:
            resp = _cosmos_frames(confirm_frames, confirm_prompt,
                                  max_tokens=150, temp=0.1)
            result = _parse_json(resp, dict(_SCAN_FALLBACK),
                                 label=f"CONFIRM pan={pan}")
        except Exception as e:
            log_exception(f"confirm pan={pan}", e)
            return None

        if not result.get("target_visible"):
            _ui("log", f"❌ False positive at pan {pan:+d}° — continuing")
            log_mission_event("false_positive", f"pan={pan}")
            return None

        # ── Confidence gate on confirmation result ────────────────────────
        conf = float(result.get("detection_confidence", 0.0))
        if conf < DETECTION_CONFIDENCE_MIN:
            _ui("log",
                f"❌ Confirmation rejected at pan {pan:+d}° — "
                f"detection_confidence {conf:.2f} < {DETECTION_CONFIDENCE_MIN}")
            log_mission_event("false_positive", f"pan={pan} conf={conf:.2f}")
            return None

        # Confirmed — turn chassis to face target
        log.info(f"🎯 Target CONFIRMED at pan={pan:+d}°")
        _ui("log", f"✅ Target confirmed at pan {pan:+d}° — turning chassis")
        log_mission_event("target_confirmed", f"pan={pan}")
        motors.oled(1, "TARGET FOUND!")
        motors.stop()
        time.sleep(0.15)

        if pan < -15:
            motors.left(MOTOR_SPEED_SLOW)
            time.sleep(_pan_to_chassis_turn_sec(pan))
            motors.stop()
        elif pan > 15:
            motors.right(MOTOR_SPEED_SLOW)
            time.sleep(_pan_to_chassis_turn_sec(pan))
            motors.stop()

        motors.pantilt(0, TILT_SEARCH)
        time.sleep(0.2)

        return {
            **result,
            "target_visible":     True,
            "target_direction":   "front",
            "in_my_path":         True,
            "action":             "forward",
            "physical_reasoning": (
                f"Target confirmed by wide-angle + webcam at pan={pan}°; "
                "chassis turned to face."
            ),
            "mission_complete": False,
        }

    def _sweep_async(phase_label: str) -> dict | None:
        """
        Capture all positions first, submit Cosmos async for each,
        then collect results. Inference and panning fully overlap.
        """
        # ── Stage 1: pan through all positions, capture one frame each ────────
        # Submit async inference immediately after each capture.
        # Pan-tilt moves to next position while Cosmos thinks about previous one.
        sensor_ctx  = _sensor_context()
        mission_ov  = _get_mission_scan_overlay()
        prompt      = mission_ov + (sensor_ctx + SCAN_360_PROMPT
                                    if sensor_ctx else SCAN_360_PROMPT)

        captures: list[tuple[int, str]] = []   # (pan_angle, frame_b64)
        futures:  list[tuple[int, str, object]] = []  # (pan, frame, future)

        for pan in PAN_STEPS:
            if not _ms.mission_active:
                return None

            _ui("log", f"{phase_label}: pan {pan:+d}°")
            motors.oled(1, f"Pan {pan:+d}d")

            # Move pan-tilt to position — use TILT_SEARCH to see standing people
            motors.pantilt(pan, TILT_SEARCH, speed=70)
            time.sleep(PAN_SETTLE)

            # Single sharp frame — no video clip
            frame = _capture_sharp(CAMERA_PANTILT)
            if frame is None:
                log.warning(f"No frame at pan={pan}° — skipping")
                continue

            # Adaptive LED if dark
            if _is_pitch_black(frame):
                motors.lights(base=180, head=255)
                time.sleep(0.15)
                frame = _capture_sharp(CAMERA_PANTILT) or frame
                motors.lights(0, 0)

            captures.append((pan, frame))

            # Submit async Cosmos immediately — runs while we pan to next position
            future = _cosmos_frames_async([frame], prompt, max_tokens=150, temp=0.2)
            futures.append((pan, frame, future))

            log.debug(f"Async Cosmos submitted for pan={pan}°")

        if not futures:
            return None

        # ── Stage 2: collect results in order, confirm first candidate ────────
        _ui("log", f"{phase_label}: collecting {len(futures)} inference results...")
        for pan, frame, future in futures:
            if not _ms.mission_active:
                return None
            try:
                # Wait for this position's result (most will already be done)
                response = future.result(timeout=30)
                result   = _parse_json(response, dict(_SCAN_FALLBACK),
                                       label=f"SWEEP pan={pan}")
            except Exception as e:
                log_exception(f"sweep collect pan={pan}", e)
                continue

            if not result.get("target_visible"):
                log.debug(f"pan={pan}°: clear")
                continue

            # Candidate — confirm with webcam
            # _confirm_candidate() blocks for up to ~10s (two Cosmos calls).
            # GAP 4 FIX: check YOLO flag immediately after it returns.
            # If Layer 2 detected something during the confirmation window,
            # surface it now — it is fresher and more reliable than continuing
            # to collect potentially stale sweep results.
            confirmed = _confirm_candidate(pan, frame, sensor_ctx, mission_ov)

            with _yolo_lock:
                yolo_fired = _ms.yolo_person_detected
            if yolo_fired:
                _ui("log", "⚡ YOLO fired during 360 confirmation — "
                           "handing off to YOLO handler immediately")
                log_mission_event("yolo_preempts_360", f"post confirm pan={pan}")
                # Return a sentinel so _best_360_scan / _process_scan skips
                # further sweep processing — the mission loop will call
                # _handle_yolo_detection() on the very next iteration.
                return None   # YOLO flag set, mission loop handles it

            if confirmed:
                return confirmed
            # False positive — continue collecting remaining results

        return None

    # ── Phase 1: Forward 180° arc ─────────────────────────────────────────────
    found = _sweep_async("Front arc")
    if found:
        return found

    if not _ms.mission_active:
        return dict(_SCAN_FALLBACK)

    # ── Phase 2: Chassis 180° turn ────────────────────────────────────────────
    _ui("log", "Turning 180° for rear sweep...")
    motors.oled(1, "Turning 180...")
    motors.pantilt(0, TILT_SEARCH)
    time.sleep(0.2)
    motors.right(MOTOR_SPEED_SLOW)
    time.sleep(TURN_90_SEC * 2.0)
    motors.stop()
    time.sleep(0.4)
    log_action("CHASSIS_180", "rear sweep")

    # ── Phase 3: Rear 180° arc ────────────────────────────────────────────────
    found = _sweep_async("Rear arc")
    if found:
        return found

    # ── No target found ───────────────────────────────────────────────────────
    motors.pantilt(0, TILT_SEARCH)
    _ui("log", "360 scan complete — no target found")
    motors.oled(1, "No target")
    log_mission_event("scan_360_complete", "no target found")
    return dict(_SCAN_FALLBACK)


def _scan_360_smart() -> dict:
    """
    Legacy chassis-rotation 360° scan (8×45° body turns).
    Kept as fallback if pan-tilt hardware is unavailable.
    Prefer _scan_360_pantilt() for normal operation.
    """
    _ms.mission_state = State.SCANNING_360
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

        motors.pantilt(0, -5)
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


def _scan_360_video_sweep() -> dict:
    """
    Observation-mode 360° scan — continuous video during chassis rotation.

    Design (for nature/inspection/exploration missions):
      - NO per-position stops or stabilization waits
      - Start video capture → rotate chassis 360° → stop capture
      - Send full panoramic video clip to Cosmos as ONE inference call
      - Cosmos sees the world as a flowing panorama — best for temporal/spatial reasoning
      - No early-exit (observation missions don't have a single target to stop for)
      - Falls back to _scan_360_pantilt() if video capture fails

    Timing:
      Chassis 360°: TURN_90_SEC × 4 ≈ 8.8s
      Video fps: 2 → ~17 frames covering the full rotation
      Inference: ~6-8s
      Total: ~17s  (vs ~8s for target hunt — acceptable for exploration)

    Use when: scan_strategy = "video_sweep" in mission YAML
    Best for: nature explorer, safety inspection, patrol, environmental survey
    """
    _ms.mission_state = State.SCANNING_360
    _ui("status", "360 VIDEO SWEEP")
    motors.oled(0, "360 Sweep")
    motors.stop()
    time.sleep(0.3)
    log.info("Starting observation 360° video sweep — continuous rotation")
    log_mission_event("scan_360_video_start", "continuous chassis rotation + video")

    TILT_LEVEL = -10   # slight downward tilt — ground-level focus
    motors.pantilt(0, TILT_LEVEL)
    time.sleep(0.3)

    # Adaptive LED — pre-check brightness before rotation starts
    test_frame = capture_frame(CAMERA_PANTILT, 320, 240)
    if test_frame and _is_pitch_black(test_frame):
        motors.lights(base=150, head=200)
        log.info("Low light — LEDs on for video sweep")

    # ── Capture video WHILE rotating ─────────────────────────────────────────
    # capture_frames_video() runs in the calling thread — chassis rotation
    # runs as a timed motor command that we let expire naturally.
    # We start the motor, then capture in a tight loop for the rotation duration.
    turn_duration = TURN_90_SEC * 4.0   # full 360°

    _ui("log", "🎬 Recording 360° panoramic video sweep...")
    motors.oled(1, "Recording...")

    motors.right(MOTOR_SPEED_SLOW)
    frames = capture_frames_video(CAMERA_PANTILT,
                                  duration=turn_duration,
                                  fps_sample=2.0)
    motors.stop()
    motors.lights(0, 0)
    time.sleep(0.3)
    motors.pantilt(0, TILT_LEVEL)

    log_mission_event("scan_360_video_captured", f"{len(frames)} frames")

    if not frames:
        log.warning("Video sweep captured no frames — falling back to pan-tilt scan")
        return _scan_360_pantilt()

    # ── Single Cosmos call with full panoramic video ──────────────────────────
    sensor_ctx  = _sensor_context()
    mission_ov  = _get_mission_scan_overlay()
    prompt      = (mission_ov
                   + (sensor_ctx + "\n\n" if sensor_ctx else "")
                   + VIDEO_SWEEP_360_PROMPT)

    _ui("log", f"🧠 Cosmos panoramic analysis ({len(frames)} frames)...")
    motors.oled(1, "Analyzing...")

    try:
        response = _cosmos_frames(frames, prompt, max_tokens=300, temp=0.2)
        result   = _parse_json(response, dict(_SCAN_FALLBACK), label="VIDEO SWEEP 360")

        reasoning = result.get("physical_reasoning", "")
        if reasoning:
            _ui("log", f"👁️  Eric observes: {reasoning}")

        log_mission_event("scan_360_video_complete",
                          f"target={result.get('target_visible')} "
                          f"dir={result.get('clearest_direction')}")
        return result

    except Exception as e:
        log_exception("_scan_360_video_sweep", e)
        return dict(_SCAN_FALLBACK)


def _circumnavigate_obstacle() -> bool:
    """
    Attempt to peek around a blocking obstacle (e.g. a cardboard box) by
    side-stepping left then right, doing a quick scan from each position.

    Only runs when YAML flag  circumnavigate_on_empty: true  is set.
    Completely inert for all other missions — the flag is False by default.

    SLAM-safe: all movement goes through _move_forward() and
    _turn_nav2_or_direct() which already support Nav2 goals.
    When SLAM is added those calls automatically use the costmap —
    no changes needed here.

    Strategy (tuned for a ~30cm wide box at ~1–2m distance):
      1. Side-step LEFT  → quick scan  → found? return True
      2. Return to centre
      3. Side-step RIGHT → quick scan  → found? return True
      4. Return to centre
      5. If still nothing, return False so the normal 360 runs as fallback

    Tune SIDE_STEP_SEC and SIDE_DIST_M in the YAML or here for your room.

    Returns True if the target was found during circumnavigation so the
    caller can call _process_scan() on the result instead of a 360.
    """
    if not _ms.mission_flags.get("circumnavigate_on_empty", False):
        return False

    # ── Tunable constants ─────────────────────────────────────────────────────
    SIDE_STEP_SEC  = float(_ms.mission_flags.get("circum_step_sec",  1.8))
    SIDE_DIST_M    = float(_ms.mission_flags.get("circum_dist_m",    0.4))
    FORWARD_SEC    = float(_ms.mission_flags.get("circum_forward_sec", 0.0))

    log.info("🔄 Circumnavigation: trying to peek around obstacle")
    log_mission_event("circumnavigate_start", f"step_sec={SIDE_STEP_SEC} dist_m={SIDE_DIST_M}")
    _ui("log", "🔄 Obstacle blocking — peeking around it...")
    _ui("status", "CIRCUMNAVIGATING")
    motors.oled(0, "Peek around")

    # Optional short forward nudge before side-stepping (closes distance to box)
    if FORWARD_SEC > 0:
        _move_forward(duration_sec=FORWARD_SEC, distance_m=SIDE_DIST_M * 0.5)
        motors.stop()
        time.sleep(0.3)

    for side, turn_fwd, turn_back in [
        ("left",  "left",  "right"),
        ("right", "right", "left"),
    ]:
        if not _ms.mission_active:
            break

        motors.oled(1, f"Peek {side}")
        _ui("log", f"   Stepping {side}...")

        # Step sideways
        _turn_nav2_or_direct(turn_fwd, SIDE_STEP_SEC)
        motors.stop()
        time.sleep(0.4)   # settle before capture

        # Quick scan from new position
        _ui("log", f"   Scanning from {side} position...")
        scan = _quick_scan()

        if scan.get("target_visible"):
            log.info(f"🎯 Circumnavigate: target found from {side} position!")
            log_mission_event("circumnavigate_found", f"side={side}")
            _ui("log", f"✅ Found from {side}! Returning to centre then approaching")
            # Return to centre so approach starts from a known heading
            _turn_nav2_or_direct(turn_back, SIDE_STEP_SEC)
            motors.stop()
            time.sleep(0.3)
            _process_scan(scan, from_360=True)
            return True

        _ui("log", f"   Nothing from {side} — returning to centre")
        # Return to centre before trying the other side
        _turn_nav2_or_direct(turn_back, SIDE_STEP_SEC)
        motors.stop()
        time.sleep(0.3)

    log_mission_event("circumnavigate_failed", "target not found from either side")
    _ui("log", "🔄 Circumnavigation: no target found — falling back to 360 scan")
    return False


def _best_360_scan() -> dict:
    """
    Route to the correct 360° scan strategy based on mission YAML.

    scan_strategy values (set in mission YAML):
      "target_hunt"   — async frame-per-position with early-exit (DEFAULT)
                        Best for: search & rescue, find missions, security
      "video_sweep"   — continuous rotation + single panoramic video inference
                        Best for: nature explorer, inspection, patrol, survey

    Falls back to chassis rotation (_scan_360_smart) if pan-tilt hardware fails.
    """
    strategy = str(_ms.mission_flags.get("scan_strategy", "target_hunt")).lower().strip()

    if strategy == "video_sweep":
        log.info("360 scan strategy: VIDEO_SWEEP (observation mode)")
        try:
            return _scan_360_video_sweep()
        except Exception as e:
            log_exception("_scan_360_video_sweep", e)
            log.warning("Video sweep failed — falling back to pan-tilt scan")
            return _scan_360_pantilt()
    else:
        # "target_hunt" or any unrecognised value → fast async early-exit scan
        if strategy not in ("target_hunt",):
            log.warning(f"Unknown scan_strategy '{strategy}' — using target_hunt")
        log.info("360 scan strategy: TARGET_HUNT (async early-exit)")
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
# ─── Mission Complete ─────────────────────────────────────────────────────────

def _capture_final_photo(obj_name: str, ts_str: str, alarm_type: str) -> list[str]:
    """
    Capture sharp, centred final photos of the confirmed target from both cameras.

    Pipeline:
      1. Stop motors + centre pan-tilt → settle 0.6 s (chassis + servo damp).
      2. Adaptive LED — light up if frame is dark.
      3. Pan-tilt camera (primary): up to PHOTO_MAX_BLUR_RETRIES attempts.
         Each attempt: capture → Laplacian blur check → Cosmos centre check.
         Cosmos is asked whether the target is in the centre third of the frame.
         If off-centre, nudge pan by PHOTO_PAN_NUDGE_DEG degrees and retry.
         Fall back to the sharpest frame seen if loop never converges.
      4. Webcam (secondary): same blur-retry loop AND same Cosmos centre check.
         Pan stays at the accepted angle from step 3 so both cameras see the same scene.
         If the webcam sees the target off-centre, pan nudges independently for the
         webcam pass — the webcam has a slightly different physical offset on the head.
      5. Save both files:
           <alarm>_<n>_<obj>_<ts>_pantilt.jpg
           <alarm>_<n>_<obj>_<ts>_webcam.jpg
      6. Returns list of saved filenames for logging.

    Never raises — all errors are caught and logged.
    """
    import base64 as _b64

    PHOTO_MAX_BLUR_RETRIES = 4     # attempts per camera before giving up on sharpness
    PHOTO_SETTLE_SEC       = 0.6   # chassis + servo damping before first capture
    PHOTO_RETRY_SETTLE_SEC = 0.5   # wait between retries
    PHOTO_PAN_NUDGE_DEG    = 12    # degrees to nudge pan if target is off-centre

    out_dir  = pathlib.Path("missions/photos")
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_obj = obj_name.replace(" ", "_")[:20]
    saved    = []

    # ── 1. Stop + centre pan-tilt ─────────────────────────────────────────────
    motors.stop()
    # Use tilt stored by eye-contact/confirm pass — same angle that just worked.
    # Falls back to 10 (face height) if not set.
    _photo_tilt = getattr(_ms, "last_confirm_tilt", 10)
    time.sleep(PHOTO_SETTLE_SEC)

    # ── 2. Adaptive LED ───────────────────────────────────────────────────────
    test_frame = capture_frame(CAMERA_PANTILT, 320, 240)
    led_on = test_frame and _is_pitch_black(test_frame)
    if led_on:
        motors.lights(base=180, head=255)
        time.sleep(0.3)

    # ── 3. Pan-tilt camera — sharp + centred ─────────────────────────────────
    _ui("log", "📸 Capturing final photo (pan-tilt)...")
    pt_frame  = None
    pt_best   = None    # sharpest frame seen regardless of centering (fallback)
    pan_angle = 0       # current pan offset from centre, adjusted by nudges

    for attempt in range(PHOTO_MAX_BLUR_RETRIES):
        motors.pantilt(pan_angle, _photo_tilt)
        time.sleep(PHOTO_RETRY_SETTLE_SEC if attempt > 0 else 0.0)

        f = capture_frame(CAMERA_PANTILT, 640, 480)
        if not f:
            break

        # Blur check — reject if Laplacian variance is below threshold
        if _is_blurry(f):
            _ui("log", f"📸 Pan-tilt attempt {attempt + 1}: blurry — retrying")
            if pt_best is None:
                pt_best = f        # keep first frame as absolute last resort
            time.sleep(PHOTO_RETRY_SETTLE_SEC)
            continue

        pt_best = f    # not blurry — update best

        # Cosmos centre check: is the target in the middle third of the frame?
        centre_prompt = (
            f"This is a photo of the confirmed target: {obj_name}. "
            "Is the main target approximately centred — within the middle horizontal "
            "third of the frame (not clipped at the far left or far right edge)? "
            'Reply ONLY with this JSON: {"centred": true_or_false, "offset": "left|right|centre"}'
        )
        try:
            cr = requests.post(VLLM_URL, json={
                "model": COSMOS_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}},
                        {"type": "text", "text": centre_prompt},
                    ]}
                ],
                "max_tokens": 40,
                "temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5,
            }, timeout=20)
            cr.raise_for_status()
            craw    = cr.json()["choices"][0]["message"]["content"].strip()
            cresult = _parse_json(craw, {"centred": True, "offset": "centre"}, "PHOTO_CENTRE")
            log_ai("photo_centre_check", craw, label="PHOTO_CENTRE")

            if cresult.get("centred", True):
                pt_frame = f
                _ui("log", f"📸 Pan-tilt: sharp + centred (attempt {attempt + 1})")
                break
            else:
                offset = str(cresult.get("offset", "centre")).lower()
                _ui("log", f"📸 Pan-tilt attempt {attempt + 1}: target {offset} — nudging pan")
                if offset == "left":
                    pan_angle = max(-60, pan_angle - PHOTO_PAN_NUDGE_DEG)
                elif offset == "right":
                    pan_angle = min(60,  pan_angle + PHOTO_PAN_NUDGE_DEG)

        except Exception as e:
            log_exception("photo_centre_cosmos", e)
            pt_frame = f    # Cosmos failed — accept the sharp frame as-is
            break

    # Centre loop never converged — use sharpest frame seen
    if pt_frame is None:
        pt_frame = pt_best
        if pt_frame:
            _ui("log", "📸 Pan-tilt: using sharpest frame (centre check did not converge)")

    if pt_frame:
        try:
            fname = out_dir / (
                f"{alarm_type}_{_ms.mission_find_count:02d}_"
                f"{safe_obj}_{ts_str}_pantilt.jpg"
            )
            fname.write_bytes(_b64.b64decode(pt_frame))
            _ui("log", f"📸 Saved pan-tilt: {fname.name}")
            log_mission_event("photo_saved_pantilt", fname.name)
            saved.append(fname.name)
        except Exception as e:
            log_exception("photo_save_pantilt", e)

    # ── 4. Webcam — sharp + centred ──────────────────────────────────────────
    # Start at the accepted pan angle from the pan-tilt pass, then nudge
    # independently — the webcam sits slightly offset on the same servo head
    # so it may need a small additional correction of its own.
    _ui("log", "📸 Capturing final photo (webcam)...")
    wc_pan_angle = pan_angle   # inherit accepted pan, may shift further below
    motors.pantilt(wc_pan_angle, _photo_tilt)
    time.sleep(0.3)

    wc_frame = None
    wc_best  = None

    for attempt in range(PHOTO_MAX_BLUR_RETRIES):
        motors.pantilt(wc_pan_angle, -20)  # tilt down — webcam is close to ground, person is above
        time.sleep(PHOTO_RETRY_SETTLE_SEC if attempt > 0 else 0.0)

        f = capture_frame(CAMERA_WEBCAM, 640, 480)
        if not f:
            break

        # Blur check
        if _is_blurry(f):
            _ui("log", f"📸 Webcam attempt {attempt + 1}: blurry — retrying")
            if wc_best is None:
                wc_best = f
            time.sleep(PHOTO_RETRY_SETTLE_SEC)
            continue

        wc_best = f   # not blurry — update best

        # Cosmos centre check (same prompt as pan-tilt pass)
        centre_prompt = (
            f"This is a photo of the confirmed target: {obj_name}. "
            "Is the main target approximately centred — within the middle horizontal "
            "third of the frame (not clipped at the far left or far right edge)? "
            'Reply ONLY with this JSON: {"centred": true_or_false, "offset": "left|right|centre"}'
        )
        try:
            cr = requests.post(VLLM_URL, json={
                "model": COSMOS_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}},
                        {"type": "text", "text": centre_prompt},
                    ]}
                ],
                "max_tokens": 40,
                "temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5,
            }, timeout=20)
            cr.raise_for_status()
            craw    = cr.json()["choices"][0]["message"]["content"].strip()
            cresult = _parse_json(craw, {"centred": True, "offset": "centre"}, "PHOTO_CENTRE_WC")
            log_ai("photo_centre_check_webcam", craw, label="PHOTO_CENTRE_WC")

            if cresult.get("centred", True):
                wc_frame = f
                _ui("log", f"📸 Webcam: sharp + centred (attempt {attempt + 1})")
                break
            else:
                offset = str(cresult.get("offset", "centre")).lower()
                _ui("log", f"📸 Webcam attempt {attempt + 1}: target {offset} — nudging pan")
                if offset == "left":
                    wc_pan_angle = max(-60, wc_pan_angle - PHOTO_PAN_NUDGE_DEG)
                elif offset == "right":
                    wc_pan_angle = min(60,  wc_pan_angle + PHOTO_PAN_NUDGE_DEG)

        except Exception as e:
            log_exception("photo_centre_cosmos_webcam", e)
            wc_frame = f   # Cosmos failed — accept the sharp frame as-is
            break

    if wc_frame is None:
        wc_frame = wc_best
        if wc_frame:
            _ui("log", "📸 Webcam: using sharpest frame (centre check did not converge)")

    if wc_frame:
        try:
            fname = out_dir / (
                f"{alarm_type}_{_ms.mission_find_count:02d}_"
                f"{safe_obj}_{ts_str}_webcam.jpg"
            )
            fname.write_bytes(_b64.b64decode(wc_frame))
            _ui("log", f"📸 Saved webcam:   {fname.name}")
            log_mission_event("photo_saved_webcam", fname.name)
            saved.append(fname.name)
        except Exception as e:
            log_exception("photo_save_webcam", e)

    # ── 5. LEDs off + re-centre pan ───────────────────────────────────────────
    if led_on:
        motors.lights(0, 0)
    motors.pantilt(0, -15)    # re-centre pan — both cameras may have nudged it

    return saved


def _trigger_mission_alarm(obj_name: str, location_hint: str = "",
                           severity: str = "WARNING", frame_b64: str = None):
    """
    Fire the mission-specific alarm, log the find, optionally save a photo.
    Called whenever a hazard/target/suspicious object is confirmed.

    obj_name:      what was found ("unattended bag", "gas canister", "injured person")
    location_hint: Cosmos physical_reasoning text, used as location description
    severity:      CRITICAL / WARNING / ADVISORY (hazard patrol) or just context
    frame_b64:     if provided and YAML photo_on_find=True, saves a timestamped photo
    """

    _ms.mission_find_count += 1
    ts = datetime.datetime.now().strftime("%H:%M:%S")

    # ── Build spoken announcement ─────────────────────────────────────────────
    alarm_type = _ms.mission_alarm_type

    if alarm_type == AlarmType.SIREN:
        msg = (f"EMERGENCY! I have found {obj_name}! "
               f"Location: {location_hint or 'current position'}. "
               f"Requesting immediate rescue assistance!")
    elif alarm_type == AlarmType.SUSPICIOUS:
        msg = (f"SECURITY ALERT! I have identified a suspicious object: {obj_name}. "
               f"Location: {location_hint or 'current position'}. "
               f"Alerting security personnel immediately. Do NOT approach.")
    elif alarm_type == AlarmType.NATURE:
        # Respect announce_location flag — don't announce location for wildlife (may startle)
        if _ms.mission_flags.get("announce_location", True):
            loc_str = f" {location_hint or 'Right here in front of me'}."
        else:
            loc_str = ""
        msg = (f"I've discovered something wonderful — {obj_name}!{loc_str} "
               f"Let me get a closer look!")
    elif alarm_type == AlarmType.NONE:
        # Narrative/story missions — quiet find, no alarm, just a friendly note
        msg = f"I have located {obj_name}."
    else:  # HAZARD / default
        msg = (f"{severity}! I have found a hazard: {obj_name}. "
               f"Location: {location_hint or 'current position'}. "
               f"This requires attention.")

    # ── Log the find ──────────────────────────────────────────────────────────
    entry = {
        "find_num":  _ms.mission_find_count,
        "time":      ts,
        "obj":       obj_name,
        "severity":  severity,
        "location":  location_hint,
        "alarm":     alarm_type,
    }
    _ms.mission_hazard_log.append(entry)
    log_mission_event("target_alarm", f"[{severity}] {obj_name} @ {location_hint}")
    _ui("log", f"🚨 [{alarm_type.upper()}] #{_ms.mission_find_count}: {obj_name} — {severity}")

    # ── Save photos if configured ─────────────────────────────────────────────
    # Captures fresh frames from BOTH cameras with blur-check + Cosmos centre-check
    # rather than saving the scan frame that triggered the alarm.
    if _ms.mission_flags.get("photo_on_find"):
        saved_photos = _capture_final_photo(
            obj_name  = obj_name,
            ts_str    = ts.replace(":", ""),
            alarm_type = alarm_type,
        )
        if not saved_photos:
            _ui("log", "⚠️  photo_on_find: no photos saved (capture failed)")

    # ── OLED display ──────────────────────────────────────────────────────────
    oled_label = "FOUND!" if alarm_type == AlarmType.NONE else f"{alarm_type.upper()}!"
    motors.oled(0, oled_label)
    motors.oled(1, obj_name[:16])

    # ── Sound alarm (non-blocking — TTS + LED + tone run in background) ───────
    sound_alarm(alarm_type, detail=msg)

    # ── Announce via TTS immediately (alarm.sound_alarm handles this) ─────────
    _ui("status", f"🚨 {alarm_type.upper()}: {obj_name}")

    # ── Security sweep: back away after alerting ──────────────────────────────
    if alarm_type == AlarmType.SUSPICIOUS and _ms.mission_flags.get("back_away_on_find"):
        _ui("log", "Security protocol: backing away from suspicious object")
        log_action("BACK_AWAY_SUSPICIOUS", obj_name)
        motors.backward(MOTOR_SPEED_SLOW)
        time.sleep(3.0)
        motors.stop()
        time.sleep(0.5)
        # Turn 180° to face away
        motors.right(MOTOR_SPEED_SLOW)
        time.sleep(TURN_90_SEC * 2.0)
        motors.stop()

    # ── Search and rescue: stay with target ───────────────────────────────────
    if alarm_type == AlarmType.SIREN and _ms.mission_flags.get("stay_with_target"):
        _ui("log", "SAR protocol: staying with casualty — repeating location broadcast")
        log_action("SAR_STAY", obj_name)
        # Repeat broadcast every 15s for 60s while rescuers come
        for i in range(4):
            time.sleep(15)
            if not _ms.mission_active:
                break
            from tts import speak
            speak(f"Still with casualty: {obj_name}. {location_hint}. "
                  f"Awaiting rescue team. This is broadcast {i + 2}.")


def _mission_report() -> str:
    """
    Build a plain-text summary report of everything found this mission.
    Called at mission end or on request.
    """
    if not _ms.mission_hazard_log:
        return "Mission complete. No hazards or targets were found during this patrol."

    lines = [f"MISSION REPORT — {len(_ms.mission_hazard_log)} finding(s):"]
    for e in _ms.mission_hazard_log:
        lines.append(
            f"  #{e['find_num']:02d} [{e['time']}] [{e['severity']}] "
            f"{e['obj']} — {e['location'] or 'location unrecorded'}"
        )

    # Severity summary for hazard/security missions
    criticals = sum(1 for e in _ms.mission_hazard_log if e["severity"] == "CRITICAL")
    warnings  = sum(1 for e in _ms.mission_hazard_log if e["severity"] == "WARNING")
    if criticals or warnings:
        lines.append(f"\nSUMMARY: {criticals} CRITICAL, {warnings} WARNING items require action.")

    return "\n".join(lines)


def _get_mission_scan_overlay() -> str:
    """
    Returns mission-specific instructions to prepend to Cosmos scan prompts.
    This customises what Cosmos looks for based on the active mission type.
    Includes character hints and current stage goal from YAML if available.
    """
    if not _ms.mission_target_objects and not _ms.mission_alarm_type:
        return ""

    alarm = _ms.mission_alarm_type
    targets = _ms.mission_target_objects

    if alarm == AlarmType.SIREN:
        if targets:
            # Specific targets defined — hunt exactly those, not generic "injured person"
            target_list = ", ".join(targets)
            base = (
                f"SEARCH AND RESCUE MODE: You are searching for {target_list}.\n"
                f"If you see {target_list} — set target_visible=true immediately "
                f"and set speak to an urgent rescue announcement.\n"
                "Do NOT wait for a person — your target may be an object, animal, or figure.\n"
                "Also watch for: smoke, fire, structural collapse, flooding.\n\n"
            )
        else:
            # No specific targets — default SAR person-hunting behaviour
            target_list = "injured or unconscious people"
            base = (
                "SEARCH AND RESCUE MODE: You are searching for injured or unconscious people.\n"
                "If you see ANY person who appears injured, unconscious, on the floor, "
                "or in distress — set target_visible=true, set object='person', "
                "set severity='CRITICAL', and set speak to an urgent rescue announcement.\n"
                "Also watch for: smoke, fire, structural collapse, flooding.\n\n"
            )

    elif alarm == AlarmType.SUSPICIOUS:
        target_list = ", ".join(targets) if targets else "unattended bags or suspicious objects"
        base = (
            f"SECURITY SWEEP MODE: You are scanning for {target_list}.\n"
            "Suspicious objects include: unattended bags, packages with wires, "
            "unusual canisters or containers, objects with timers or electronics attached.\n"
            "If you see anything suspicious — set target_visible=true, "
            "set object='obstacle', set severity='CRITICAL'.\n"
            "DO NOT move closer. Report exact location and description.\n\n"
        )

    elif alarm == AlarmType.HAZARD:
        target_list = ", ".join(targets) if targets else "hazards and dangerous conditions"
        base = (
            f"HAZARD PATROL MODE: You are scanning for {target_list}.\n"
            "Classify each hazard: CRITICAL (immediate danger) / WARNING (needs attention) "
            "/ ADVISORY (monitor).\n"
            "Look carefully for: wet floors, exposed wires, gas canisters, fire, smoke, "
            "blocked exits, chemical spills, structural damage.\n"
            "Set target_visible=true for any hazard found. Report location precisely.\n\n"
        )

    elif alarm == AlarmType.NATURE:
        target_list = ", ".join(targets) if targets else "wildlife and interesting plants"
        base = (
            f"NATURE EXPLORE MODE: You are documenting {target_list}.\n"
            "Describe everything you see with scientific curiosity and poetic appreciation.\n"
            "Wildlife: note species, behaviour, colours, movement.\n"
            "Plants/flowers: describe colours, structure, seasonal state.\n"
            "Scenic features: lighting, textures, composition.\n"
            "Set target_visible=true for any wildlife, flower, or interesting scene.\n"
            "Use vivid descriptive language in the speak field.\n\n"
        )

    elif alarm == AlarmType.NONE:
        # Narrative/story missions — guide Cosmos to find story characters
        target_list = ", ".join(targets) if targets else "mission targets"
        # Pull owner/creator description from character hints if available
        chars = _ms.mission_flags.get("characters", [])
        owner_desc = ""
        for ch in (chars if isinstance(chars, list) else []):
            if isinstance(ch, dict) and any(
                kw in ch.get("name", "").lower() for kw in ["owner", "creator"]
            ):
                owner_desc = ch.get("hint", "").split("\n")[0].strip()
                break
        desc_line = (
            f"You are specifically looking for: {owner_desc}\n"
            if owner_desc else
            f"You are looking for: {target_list}\n"
        )
        base = (
            f"NARRATIVE MISSION MODE: {desc_line}"
            "When you see the target person — set target_visible=true, "
            "set object='person', set object_name='creator'.\n"
            "If you see a person who does NOT match: set target_visible=false, "
            "set object='person', set object_name=null.\n"
            "No alarm will sound — this is a find-and-greet mission.\n\n"
        )

    else:
        base = ""

    # ── Append character hints and stage goal from YAML ───────────────────────
    # These give Cosmos the knowledge to roleplay characters correctly and
    # know what sub-goal it's working toward at each stage of the mission.
    return base + _get_stage_context() + _get_character_context()


def _get_character_context() -> str:
    """
    Build a character hint block from the YAML 'characters' list.
    Injected into Cosmos prompts so Eric knows how to interact with each character.
    """
    characters = _ms.mission_flags.get("characters", [])
    if not characters:
        return ""
    lines = ["CHARACTER GUIDE (how to interact with each person you meet):"]
    for c in characters:
        name = c.get("name", "Unknown")
        hint = c.get("hint", "")
        if hint:
            lines.append(f"  {name}: {hint}")
    lines.append("")
    return "\n".join(lines) + "\n"


def _get_stage_context() -> str:
    """
    Return the current mission stage goal from the YAML 'mission_stages' list.
    Gives Cosmos a focused sub-goal that matches the current step index.
    """
    stages = _ms.mission_flags.get("mission_stages", [])
    if not stages or _ms.current_step_idx >= len(stages):
        return ""
    stage = stages[_ms.current_step_idx]
    goal  = stage.get("goal", "") if isinstance(stage, dict) else str(stage)
    if not goal:
        return ""
    return f"CURRENT STAGE GOAL: {goal}\n\n"


def _handle_mission_complete(obj_name):
    log.info(f"MISSION COMPLETE — {obj_name}")
    log_mission_event("mission_complete", obj_name or "target")
    _ms.mission_state = State.COMPLETE
    motors.stop()
    stop_alarm()   # cancel any running alarm pattern
    try:
        from config import USE_NAV2
        if USE_NAV2:
            from nav2 import cancel_goal, nav2_available
            if nav2_available():
                cancel_goal()
    except Exception as _exc:  # nav2
        log.debug(f"nav2 unavailable: {_exc}")
    motors.oled(0, "MISSION DONE!")
    motors.oled(1, (obj_name or "Target")[:16])
    _ui("status", "MISSION COMPLETE")

    for _ in range(5):
        motors.lights(255, 255); time.sleep(0.25)
        motors.lights(0, 0);    time.sleep(0.25)
    motors.lights(128, 255)

    motors.pantilt(0, -5)
    time.sleep(0.5)

    # Build report for patrol/security missions
    report = _mission_report()
    if _ms.mission_hazard_log:
        _ui("log", f"📋 MISSION REPORT:\n{report}")

    announcement = ask_cosmos(
        f"You found: {obj_name or 'the target'}. Mission complete. "
        + (f"You found {len(_ms.mission_hazard_log)} item(s) during the mission. " if _ms.mission_hazard_log else "")
        + "Warm triumphant 2-3 sentence announcement.",
        max_tokens=120
    )
    eric_say(announcement)
    _ui("eric_says", announcement)
    _ui("log", f"COMPLETE: {announcement}")
    log_mission_event("announcement", announcement[:150])

    end_mission_log(completed=True)

    _ms.mission_active = False

    # Same cleanup as stop_mission() — recycle executor, flush cache, GC
    # so the Jetson is clean before the operator starts the next mission.
    global _cosmos_executor
    try:
        _cosmos_executor.shutdown(wait=False, cancel_futures=True)
    except TypeError:
        _cosmos_executor.shutdown(wait=False)
    _cosmos_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="cosmos"
    )
    try:
        import requests as _req
        vllm_base = VLLM_URL.rstrip("/").rsplit("/", 1)[0]
        _req.post(f"{vllm_base}/reset_prefix_cache", timeout=5)
    except Exception:
        pass
    from cosmos import reset_mission_context
    reset_mission_context()
    import gc; gc.collect()
    log.info("mission_complete: executor recycled, cache flushed, GC collected")

    _ui("status", "MISSION COMPLETE")
    motors.oled(0, "TARGET FOUND!")
    motors.oled(1, "Mission done!")


# ─── Character Interaction ────────────────────────────────────────────────────

def handle_character_response(character, said):
    # Store what the character said
    _ms.conversation_history.append({"speaker": character, "said": said, "time": time.time()})
    n = sum(1 for e in _ms.conversation_history if e.get("speaker") == character)

    # Build full dialogue history including Eric's own prior responses
    history_lines = []
    for e in _ms.conversation_history[-8:]:
        history_lines.append(f"{e.get('speaker', '?')}: {e['said']}")
    history = "\n".join(history_lines)

    # Include per-character hint from YAML if available
    char_hint = ""
    for c in _ms.mission_flags.get("characters", []):
        if c.get("name", "").lower() == character.lower():
            char_hint = f"Character note: {c.get('hint', '')}\n" if c.get("hint") else ""
            break

    response = ask_cosmos_plain(
        f"You are ERIC, a tracked ground robot on a mission. You are having a conversation.\n"
        f"You are speaking to: {character}\n"
        f"{char_hint}"
        f"Conversation so far:\n{history}\n\n"
        f"{character} just said: \"{said}\"\n\n"
        "Respond as ERIC — directly, specifically, in your own voice. "
        "Do NOT repeat phrases you have already used. "
        "Do NOT use filler like 'gotcha', 'sure thing', 'cool beans', 'awesome', 'appreciate the'. "
        "If this exchange has no new information after 3+ turns: wrap up warmly, end with [MOVE_ON]. "
        "Plain spoken words only. No JSON.",
        max_tokens=600
    )

    move_on = "[MOVE_ON]" in response
    clean   = response.replace("[MOVE_ON]", "").strip()

    # Strip JSON blobs — Cosmos sometimes returns JSON even when asked for plain text
    if clean.startswith("{") or clean.startswith("["):
        import re as _re
        # Try to extract plain text outside the JSON object
        plain = _re.sub(r"\{.*?\}", "", clean, flags=_re.DOTALL).strip()
        if plain:
            clean = plain
        else:
            # Pure JSON — retry with harder plain text constraint
            clean = ask_cosmos(
                f"Respond to {character} who said: \"{said}\". "
                "Plain spoken words only — absolutely no JSON, no curly brackets, "
                "no formatting. 2 sentences max.",
                max_tokens=100
            ).replace("[MOVE_ON]", "").strip()

    # Store Eric's response in history so future turns know what was already said
    _ms.conversation_history.append({"speaker": "Eric", "said": clean, "time": time.time()})

    eric_say(clean)  # character interactions — Eric is stopped, longer reply ok
    _ui("log", f"[{character}]: {said}\n[Eric]: {clean}")
    if move_on:
        resume_after_interaction()
    return clean


# ─── Process Scan Result ──────────────────────────────────────────────────────

def _confirm_and_photograph_target():
    """
    Called when ERIC arrives at the target (person/creator).

    Steps:
      1. Tilt LOW (-5°) — full body view for description confirmation
      2. Tilt SWEEP (-15° → 0° → +15° → +25°) — search for face, ask Cosmos
         at each tilt angle if this matches description + can see face
      3. Lock tilt at best angle where face is visible
      4. Eye contact gate — wait in silence for direct gaze
      5. Greet
      6. Call existing _capture_final_photo() for dual-cam sharp photos
      7. Execute step action (mission complete)
    """
    import datetime

    motors.stop()
    _ms.mission_state = State.INTERACTING
    ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Branch: alarm missions (SAR/siren) go straight to alarm on arrival ──
    _is_alarm_mission = (
        _ms.mission_alarm_type not in (AlarmType.NONE, None)
        and str(_ms.mission_alarm_type).lower() not in ("none", "null", "")
    )
    if _is_alarm_mission:
        _ui("log", "🚨 Arrived at target — triggering alarm")
        motors.pantilt(0, 10, 50)   # face level
        time.sleep(0.3)
        f = capture_frame(CAMERA_PANTILT, 640, 480)
        reason = "Target confirmed at close range"
        if f:
            try:
                r = requests.post(VLLM_URL, json={
                    "model": COSMOS_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}},
                            {"type": "text", "text":
                                "Describe the person's position and condition in one sentence. "
                                "Are they conscious? On the floor? Standing? Injured?"}
                        ]}],
                    "max_tokens": 60, "temperature": 0.3,
                }, timeout=15)
                r.raise_for_status()
                reason = r.json()["choices"][0]["message"]["content"].strip()
            except Exception as _ae:
                log_exception("alarm_confirm", _ae)
        _trigger_mission_alarm(
            _ms.mission_target_objects[0] if _ms.mission_target_objects else "victim",
            location_hint = reason,
            severity      = "CRITICAL",
        )
        return

    # ── Step 1: Full body confirm at low tilt (greet/narrative missions) ──
    motors.pantilt(0, -5, 60)
    time.sleep(0.5)
    _ui("log", "👁️  Arrived — confirming target description...")
    motors.oled(1, "Confirming...")

    step_obj = _current_step()
    chars = _ms.mission_flags.get("characters", [])
    target_desc = ""
    for ch in (chars if isinstance(chars, list) else []):
        if isinstance(ch, dict) and "owner" in ch.get("name", "").lower():
            target_desc = ch.get("hint", "")
            break
    if not target_desc:
        # Build description from target_objects if no character hint
        _tgts = _ms.mission_target_objects
        target_desc = ", ".join(_tgts) if _tgts else "the target person"

    confirm_frame = capture_frame(CAMERA_PANTILT, 640, 480)
    description_confirmed = True
    if confirm_frame:
        try:
            r = requests.post(VLLM_URL, json={
                "model": COSMOS_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{confirm_frame}"}},
                    {"type": "text", "text":
                        f"Target description: {target_desc}\n"
                        "Does the person in this image match ALL criteria?\n"
                        'Answer ONLY: {"confirmed": true_or_false, "reasoning": "one sentence"}'}
                ]}],
                "max_tokens": 80, "temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5,
            }, timeout=20)
            r.raise_for_status()
            res = _parse_json(r.json()["choices"][0]["message"]["content"].strip(),
                              {"confirmed": True}, "TARGET_CONFIRM")
            description_confirmed = res.get("confirmed", True)
            _ui("log", f"{'✅' if description_confirmed else '❌'} Description: {res.get('reasoning','')}")
            log_mission_event("target_confirm", f"confirmed={description_confirmed}")
        except Exception as e:
            log_exception("target_confirm", e)

    if not description_confirmed:
        _ui("log", "⚠️  Description mismatch — asking stranger for help")
        motors.oled(1, "Asking...")

        # Lift tilt to face level to talk to them naturally
        motors.pantilt(0, 15, 50)
        time.sleep(0.5)

        _tgt_desc = ", ".join(_ms.mission_target_objects) if _ms.mission_target_objects else "a specific person"
        ask = ask_cosmos(
            f"You have stopped in front of someone who does not match your target's description. "
            f"Ask them warmly if they have seen the person you are looking for: {_tgt_desc}. "
            "Keep it to 2 sentences. Plain spoken words only — no JSON.",
            max_tokens=80
        )
        eric_say(ask)
        log_mission_event("asked_stranger", ask[:100])

        # Wait briefly for a response (TTS plays, person might reply)
        time.sleep(4.0)

        # Thank them and move on
        thanks = ask_cosmos(
            "Thank the person briefly and let them know you will keep searching. "
            "One sentence. Plain spoken words only — no JSON.",
            max_tokens=40
        )
        eric_say(thanks)

        # Resume search
        motors.pantilt(0, -5, 60)
        _ms.mission_state = State.SEARCHING
        _ms.target_spotted_count = 0
        if _safe_to_fwd():
            motors.forward(MOTOR_SPEED_SLOW)
        return

    # ── Step 2: Tilt sweep to find face ──────────────────────────────────
    # Sweep from low to high, asking Cosmos at each angle if face is visible
    _ui("log", "👁️  Sweeping tilt to find face...")
    motors.oled(1, "Finding face...")
    TILT_ANGLES = [0, 10, 20, 30, 40, 50]  # sweep upward — robot is low, face is high
    best_tilt = 30   # fallback — better default for standing person
    face_found = False

    for tilt in TILT_ANGLES:
        motors.pantilt(0, tilt, 50)
        time.sleep(0.5)
        f = capture_frame(CAMERA_PANTILT, 320, 240)
        if not f:
            continue
        try:
            r = requests.post(VLLM_URL, json={
                "model": COSMOS_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}},
                    {"type": "text", "text":
                        "Can you clearly see the person's FACE (eyes, nose, mouth visible) "
                        "in this image?\n"
                        'Answer ONLY: {"face_visible": true_or_false}'}
                ]}],
                "max_tokens": 20, "temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5,
            }, timeout=10)
            r.raise_for_status()
            res = _parse_json(r.json()["choices"][0]["message"]["content"].strip(),
                              {"face_visible": False}, "FACE_SWEEP")
            if res.get("face_visible"):
                best_tilt = tilt
                face_found = True
                _ui("log", f"👁️  Face found at tilt={tilt}°")
                break
        except Exception as e:
            log_exception("face_sweep", e)

    if not face_found:
        _ui("log", f"Face sweep done — using tilt={best_tilt}°")

    # Lock tilt at face level
    motors.pantilt(0, best_tilt, 40)
    time.sleep(0.5)

    # ── Step 3: Eye contact gate ──────────────────────────────────────────
    _ui("log", "👁️  Waiting for eye contact (silent)...")
    motors.oled(1, "Eye contact...")
    eye_confirmed = False

    for attempt in range(8):
        if not _ms.mission_active:
            break
        ec_frame = capture_frame(CAMERA_PANTILT, 320, 240)
        if not ec_frame:
            time.sleep(3)
            continue
        try:
            r = requests.post(VLLM_URL, json={
                "model": COSMOS_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ec_frame}"}},
                    {"type": "text", "text":
                        "Is the person looking DIRECTLY at the camera lens — "
                        "held deliberate eye contact, not glancing, not looking away?\n"
                        'Answer ONLY: {"eye_contact": true_or_false, "reasoning": "one sentence"}'}
                ]}],
                "max_tokens": 60, "temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5,
            }, timeout=15)
            r.raise_for_status()
            ec = _parse_json(r.json()["choices"][0]["message"]["content"].strip(),
                             {"eye_contact": False}, "EYE_CONTACT")
            if ec.get("eye_contact"):
                _ui("log", f"👁️  Eye contact: {ec.get('reasoning','')}")
                log_mission_event("eye_contact_confirmed", ec.get("reasoning",""))
                eye_confirmed = True
                break
            else:
                _ui("log", f"No eye contact yet ({attempt+1}/8) — {ec.get('reasoning','')}")
        except Exception as e:
            log_exception("eye_contact_gate", e)
        time.sleep(3)

    if not eye_confirmed:
        _ui("log", "Eye contact timeout — proceeding")
        log_mission_event("eye_contact_timeout", "proceeding")

    # ── Step 4: Photo FIRST — before greeting so it's never skipped ─────
    _ui("log", "📸 Taking final photos...")
    motors.oled(1, "Smile!")
    _ms.last_confirm_tilt = best_tilt
    motors.pantilt(0, best_tilt, 40)
    time.sleep(0.3)
    saved = _capture_final_photo(
        obj_name  = "creator",
        ts_str    = ts_str,
        alarm_type = "find"
    )
    if saved:
        _ui("log", f"📸 Photos saved: {', '.join(saved)}")

    # ── Step 5: Greet ─────────────────────────────────────────────────────
    if not _ms.mission_active:
        return
    greeting = ask_cosmos_plain(
        "Greet your creator. Two sentences maximum. "
        "Speak directly to him — not about him. "
        "Start with his name or a direct address. "
        "Example: 'OppaAI. I found you.' or 'Hello. I have been looking for you.' "
        "No thinking. No planning. No JSON. Just the greeting.",
        max_tokens=60,
        temperature=0.5
    )
    eric_say(greeting)
    log_mission_event("creator_greeted", greeting[:150])
    time.sleep(1.0)

    # Return to low tilt
    motors.pantilt(0, -5, 60)

    # ── Step 6: Wait for operator dismissal ──────────────────────────────
    # For narrative missions (alarm_type=none) with wait_for_dismiss=true in YAML,
    # stay in place after greeting until the operator ends the mission manually.
    # This lets the moment land rather than Eric immediately shutting down.
    _wait_for_dismiss = _ms.mission_flags.get("wait_for_dismiss", False)
    if _wait_for_dismiss:
        _ui("log", "Waiting for operator to end mission (wait_for_dismiss=true)")
        motors.oled(0, "Mission done")
        motors.oled(1, "Press STOP")
        motors.lights(128, 255)
        while _ms.mission_active:
            time.sleep(1.0)
        return

    # ── Step 6: Complete ──────────────────────────────────────────────────
    log_mission_event("mission_complete", "creator found, confirmed, greeted, photographed")
    _ms.mission_steps     = []   # clear any bleed-through steps from KV cache parse
    _ms.current_step_idx  = 0
    _ms.mission_active    = False
    return


def _approach_target():
    """
    Industrial-standard approach pipeline.

    Design (Layer 3 — closes on confirmed target):
    ────────────────────────────────────────────────
    1. Hardware-first distance gate (OAK-D depth + YOLO position memory):
       - If OAK-D reports < ARRIVE_DIST_M  → arrived, execute step action.
       - If YOLO has a fresh reading < ARRIVE_DIST_M  → arrived.
       These fire before any Cosmos call — zero extra latency.

    2. Move in 1.5-second clips (shortened from 2 s to stay reactive).
       After each clip, re-check hardware distance first.

    3. Cosmos scan only every APPROACH_SCAN_INTERVAL moves (not every step).
       This means Cosmos is called ~every 4.5 s not ~every 10 s.
       The scan uses _approach_scan() — identical to _quick_scan() but:
         a) No webcam confirmation pass (target already confirmed at scan time).
         b) Lower confidence threshold hint in prompt ("even partial view").

    4. Flip-flop guard (shared with _process_scan):
       Uses the global _ms.target_spotted_count so persistence carries across
       the scan→approach boundary. A single "invisible" scan does NOT abort.
       APPROACH_INVISIBLE_LIMIT consecutive bad frames → lost target.

    5. Lateral steering via YOLO bearing (hardware) first, Cosmos second.
       Proportional turn: 0.2 s at ±5°, up to 0.7 s at ±45°.

    6. Pan-tilt vertical tracking — tilts down as Cosmos says "near/close".

    7. Eye-contact gate for person targets (unchanged).

    On arrival: calls _execute_step_action() so multi-step missions advance.
    On lost target: resets to SEARCHING and resumes forward motion.
    """

    _ui("log", "Approaching target (hardware-first)...")
    motors.oled(1, "Approaching...")
    _ui("status", "APPROACHING")
    log_mission_event("approach_start", "hardware-first approach pipeline")

    _NEAR_DISTANCES     = {"close", "near", "nearby", "very_close", "very close",
                           "right there"}
    _TARGET_OBJECTS     = {"slipper", "shoe", "person", "robot"}
    # Raise arrival distance for person targets — 0.65m is too close, Eric hits them
    _person_mission = any(
        kw in str(_ms.mission_target_objects).lower()
        for kw in ("person", "man", "woman", "human")
    )
    _default_arrive = 0.40 if _person_mission else 0.30
    ARRIVE_DIST_M = float(_ms.mission_flags.get("approach_distance", _default_arrive))
    APPROACH_MOVE_SEC   = 1.5    # shorter clips → more hardware checks per meter
    APPROACH_SCAN_EVERY = 3      # Cosmos scan every N move clips (was every 1)
    APPROACH_MAX_MOVES  = 25     # max total move clips (~37.5 s)
    APPROACH_INVISIBLE_LIMIT = 5 # consecutive Cosmos misses before "lost target"

    invisible_count   = 0
    moves_since_scan  = 0
    nav2_fail_count   = 0   # consecutive Nav2 failures — disable after limit
    NAV2_FAIL_LIMIT   = 1   # first Nav2 status-6 → immediately use direct motors
                            # Nav2 status 6 = can't plan path (person on costmap)
                            # Direct motors handle this correctly

    # ── Seed flip-flop counter so the first bad frame doesn't abort us ────────
    # We enter _approach_target because _process_scan just confirmed the target.
    # Make sure _ms.target_spotted_count reflects that so the guard has headroom.
    if _ms.target_spotted_count < 2:
        _ms.target_spotted_count = 2
        log.debug("approach: seeded _ms.target_spotted_count=2 from confirmed entry")

    for attempt in range(APPROACH_MAX_MOVES):
        if not _ms.mission_active:
            break

        # ── Hardware distance gate BEFORE moving ─────────────────────────────
        # Check LiDAR first — always reliable, no depth failures
        try:
            from lidar import min_front_distance
            _lidar_front = min_front_distance()
            if _lidar_front is not None and _lidar_front < ARRIVE_DIST_M:
                _ui("log", f"✅ LiDAR arrival: {_lidar_front:.2f}m — arrived")
                log_mission_event("hw_arrival_lidar", f"{_lidar_front:.2f}m")
                motors.stop()
                _confirm_and_photograph_target()
                return
        except Exception as _exc:
            log.debug(f"lidar unavailable: {_exc}")

        # Check OAK-D depth
        try:
            from oakd import get_front_depth, oakd_available
            if oakd_available():
                hw_dist = get_front_depth()
                if hw_dist is not None and hw_dist < ARRIVE_DIST_M:
                    _ui("log", f"✅ OAK-D arrival: {hw_dist:.2f}m — arrived")
                    log_mission_event("hw_arrival_oakd", f"{hw_dist:.2f}m")
                    motors.stop()
                    _confirm_and_photograph_target()
                    return
        except Exception as _exc:  # oakd/yolo
            log.debug(f"oakd/yolo unavailable: {_exc}")

        # Check YOLO position memory (hardware depth — more accurate than Cosmos text)
        try:
            from oakd import get_last_yolo_position, yolo_available, oakd_available
            if oakd_available() and yolo_available():
                step = _current_step()
                # Map mission step target to YOLO class labels
                _YOLO_LABEL_MAP = {
                    "person": ["person"], "robot": ["person"],
                    "slipper": [], "shoe": [],   # YOLO doesn't detect footwear
                }
                target_key = (step.target.lower() if step else "").split()[0]
                yolo_labels = _YOLO_LABEL_MAP.get(target_key, ["person"])
                for lbl in yolo_labels:
                    ypos = get_last_yolo_position(lbl)
                    if ypos:
                        age = time.monotonic() - ypos.get("timestamp", 0)
                        if age < 3.0 and ypos["dist_m"] < ARRIVE_DIST_M:
                            _ui("log", f"✅ YOLO arrival: {lbl} at {ypos['dist_m']:.2f}m "
                                       f"({age:.1f}s ago) — executing step action")
                            log_mission_event("hw_arrival_yolo", f"{lbl} {ypos['dist_m']:.2f}m")
                            motors.stop()
                            motors.pantilt(0, -15)
                            time.sleep(0.3)
                            _execute_step_action(lbl)
                            return
        except Exception as _exc:  # oakd/yolo
            log.debug(f"oakd/yolo unavailable: {_exc}")

        # ── YOLO bearing-based steering (hardware, before moving) ─────────────
        try:
            from oakd import get_last_yolo_position, yolo_available, oakd_available
            if oakd_available() and yolo_available():
                ypos = get_last_yolo_position("person")
                if ypos and (time.monotonic() - ypos.get("timestamp", 0)) < 3.0:
                    bearing_deg = ypos.get("bearing_deg", 0.0)
                    if abs(bearing_deg) > 8.0:
                        turn_sec = max(0.15, min(0.7, abs(bearing_deg) / 45.0 * 0.7))
                        direction = "left" if bearing_deg < 0 else "right"
                        _ui("log", f"YOLO steer: {direction} {bearing_deg:+.0f}° ({turn_sec:.2f}s)")
                        _turn_nav2_or_direct(direction, turn_sec)
        except Exception as _exc:  # oakd/yolo
            log.debug(f"oakd/yolo unavailable: {_exc}")

        # ── Move one clip ─────────────────────────────────────────────────────
        # Briefly disable LIDAR safety so the target itself (a small toy/object
        # on the floor) does not stop Eric before it can get close enough for
        # visual confirmation. Safety is ALWAYS re-enabled after the move.
        _lidar_suppressed = False
        try:
            from lidar import min_front_distance, set_safety_active
            _front = min_front_distance()
            # Only suppress LiDAR for very small objects (toys, slippers <30cm away)
            # Never suppress for person-sized targets — use hardware arrival instead
            _is_person_target = any(
                kw in str(getattr(_ms, "mission_target_objects", [])).lower()
                for kw in ("person", "man", "woman", "human")
            )
            if 0.15 < _front < 0.35 and not _is_person_target:
                # Small object close ahead — suppress LIDAR for this clip only
                set_safety_active(False)
                _lidar_suppressed = True
                log.info(f"Approach: LIDAR safety suppressed for small-target clip "
                         f"(front={_front:.2f}m)")
        except Exception:
            pass
        try:
            from config import USE_NAV2
            if USE_NAV2 and nav2_fail_count >= NAV2_FAIL_LIMIT:
                # Nav2 keeps failing (person blocking costmap) — direct motors
                if _safe_to_fwd():
                    motors.forward(MOTOR_SPEED_SLOW)
                time.sleep(APPROACH_MOVE_SEC)
                motors.stop()
            else:
                _move_forward(duration_sec=APPROACH_MOVE_SEC, distance_m=0.4)
                # If Nav2 finished almost instantly it likely failed (status 6)
                # Normal navigation takes >1s; <0.5s = instant rejection
                try:
                    from nav2 import nav2_available, is_navigating
                    if USE_NAV2 and nav2_available() and not is_navigating():
                        # Not navigating right after _move_forward = Nav2 rejected/failed
                        nav2_fail_count += 1
                        log.info(f"Nav2 fail count: {nav2_fail_count}/{NAV2_FAIL_LIMIT}")
                except Exception:
                    pass
        finally:
            if _lidar_suppressed:
                try:
                    from lidar import set_safety_active
                    set_safety_active(True)
                    log.info("Approach: LIDAR safety restored")
                except Exception:
                    pass
        motors.stop()
        time.sleep(0.25)
        moves_since_scan += 1

        # ── Hardware distance gate AFTER moving ───────────────────────────────
        try:
            from oakd import get_front_depth, oakd_available
            if oakd_available():
                hw_dist = get_front_depth()
                if hw_dist is not None:
                    if hw_dist < ARRIVE_DIST_M:
                        _ui("log", f"✅ OAK-D post-move arrival: {hw_dist:.2f}m")
                        log_mission_event("hw_arrival_oakd_post", f"{hw_dist:.2f}m")
                        motors.stop()
                        _confirm_and_photograph_target()
                        return
                    elif hw_dist < 0.50:
                        # Very close — avoid crushing the target with another move
                        _ui("log", f"OAK-D: {hw_dist:.2f}m — very close, scanning now")
                        moves_since_scan = APPROACH_SCAN_EVERY  # force scan now
        except Exception as _exc:  # oakd/yolo
            log.debug(f"oakd/yolo unavailable: {_exc}")

        # ── Cosmos approach scan (every APPROACH_SCAN_EVERY moves) ───────────
        if moves_since_scan < APPROACH_SCAN_EVERY:
            continue   # skip Cosmos this clip — rely on hardware only

        moves_since_scan = 0
        check = _approach_scan()   # no webcam confirmation, lower threshold

        dist    = str(check.get("distance", "far")).lower()
        obj     = check.get("object", "unknown")
        tdir    = str(check.get("target_direction", "front")).lower().strip()
        in_path = check.get("in_my_path", False)

        # ── Same target keyword override as _process_scan ───────────────────
        if not check.get("target_visible"):
            _tgt_kws = ["creator", "owner"] + [t.lower() for t in (_ms.mission_target_objects or [])]
            _rsn = str(check.get("physical_reasoning", "")).lower()
            _onm = str(check.get("object_name") or "").lower()
            if any(kw in _onm or kw in _rsn for kw in _tgt_kws):
                log.info("Approach scan: target keyword in reasoning — forcing target_visible=True")
                check["target_visible"] = True
                check["object_name"] = check.get("object_name") or "creator"

        # ── Obstacle → avoid ─────────────────────────────────────────────────
        if check.get("wall_ahead") or check.get("obstacle_close"):
            _ui("log", f"Obstacle during approach — avoiding (obj={obj})")
            log_action("AVOID_DURING_APPROACH", f"obj={obj}")
            from avoidance import avoid_obstacle
            force_360 = avoid_obstacle(
                wall_ahead=check.get("wall_ahead", False),
                small_obstacle=check.get("small_obstacle", False)
            )
            if force_360:
                _ms.mission_state = State.SEARCHING
                return
            # After avoidance, reseed counter so we don't immediately declare lost
            if _ms.target_spotted_count < 2:
                _ms.target_spotted_count = 2
            continue

        if check.get("mission_complete"):
            _execute_step_action(check.get("object_name"))
            return

        # ── Cosmos distance confirmation — only if LiDAR agrees ─────────────
        if check.get("target_visible") and obj in _TARGET_OBJECTS and dist in _NEAR_DISTANCES:
            # Verify with LiDAR before acting — Cosmos "near" is often wrong
            _lidar_ok = False
            try:
                from lidar import min_front_distance
                _lf = min_front_distance()
                if _lf is not None and _lf < ARRIVE_DIST_M:
                    _lidar_ok = True
            except Exception:
                _lidar_ok = True  # no lidar — trust Cosmos
            if _lidar_ok:
                _ui("log", f"Cosmos+LiDAR: target close ({dist}) — arrived")
                log_mission_event("cosmos_arrival", f"obj={obj} dist={dist}")
                motors.stop()
                _confirm_and_photograph_target()
                return
            else:
                _ui("log", f"Cosmos says near but LiDAR disagrees — keep approaching")

        if dist in _NEAR_DISTANCES or in_path:
            _ui("log", f"Cosmos: close/in-path ({dist}) — executing step action")
            _execute_step_action(check.get("object_name") or obj)
            return

        # ── Target directly below camera → already on top of it ──────────────
        if tdir in ("down", "below"):
            _ui("log", "Target below camera — arrived!")
            motors.pantilt(0, -20)
            time.sleep(0.3)
            _execute_step_action(check.get("object_name") or obj)
            return

        # ── Cosmos lateral steering (only if YOLO bearing not fresh) ─────────
        try:
            from oakd import get_last_yolo_position, yolo_available, oakd_available
            yolo_fresh = (oakd_available() and yolo_available() and
                          get_last_yolo_position("person") is not None and
                          (time.monotonic() - (get_last_yolo_position("person") or {}).get("timestamp", 0)) < 2.0)
        except Exception as _exc:  # oakd/yolo
            log.debug(f"oakd/yolo error: {_exc}")
            yolo_fresh = False

        if not yolo_fresh:
            if tdir in ("left", "left_side"):
                _ui("log", "Cosmos: target drifted left — correcting")
                motors.left(MOTOR_SPEED_SLOW); time.sleep(0.35); motors.stop()
            elif tdir in ("right", "right_side"):
                _ui("log", "Cosmos: target drifted right — correcting")
                motors.right(MOTOR_SPEED_SLOW); time.sleep(0.35); motors.stop()

        # ── Pan-tilt vertical tracking ────────────────────────────────────────
        # Stay low (-5°) during approach — face-lift happens at confirmation
        motors.pantilt(0, -5, 60)

        # ── Person nearby → eye-contact gate before greeting ─────────────────
        if obj == "person" and (dist in _NEAR_DISTANCES or in_path):
            name = check.get("object_name") or "person"
            _ui("log", "Person nearby during approach — checking eye contact...")
            motors.oled(1, "Person!")
            ec_frame = capture_frame(CAMERA_PANTILT, 320, 240)
            greet = True
            if ec_frame:
                try:
                    ec_payload = {
                        "model": COSMOS_MODEL,
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ec_frame}"}},
                            {"type": "text", "text":
                                '{"close_and_facing": true_or_false, "reasoning": "one sentence"} '
                                '— Is the person within 1.5m AND facing/looking toward you?'}
                        ]}],
                        "max_tokens": 50,
                        "temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5,
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
                _ms.mission_state = State.INTERACTING
                return

        # ── Flip-flop persistence guard ───────────────────────────────────────
        target_visible = check.get("target_visible", False)
        if target_visible:
            _ms.target_spotted_count += 1
            invisible_count = 0
        else:
            if _ms.target_spotted_count > 0:
                _ms.target_spotted_count -= 1
            if _ms.target_spotted_count > 0:
                log.info(f"Approach flip-flop: forcing visible "
                         f"(lock remaining={_ms.target_spotted_count})")
                invisible_count = 0   # don't count this against us
            else:
                invisible_count += 1
                _ui("log", f"Target not visible in approach "
                           f"({invisible_count}/{APPROACH_INVISIBLE_LIMIT})")
                if invisible_count >= APPROACH_INVISIBLE_LIMIT:
                    _ui("log", "Lost target after 5 consecutive bad scans — resuming search")
                    log_mission_event("target_lost",
                                      "5 consecutive invisible Cosmos scans in approach")
                    _ms.mission_state = State.SEARCHING
                    if _safe_to_fwd():
                        motors.forward(MOTOR_SPEED_SLOW)
                    return

    # ── Approach timeout — treat as arrived rather than give up ──────────────
    _ui("log", "Approach timeout — assuming arrived, executing step action")
    log_mission_event("approach_timeout", f"{APPROACH_MAX_MOVES} moves")
    motors.stop()
    _confirm_and_photograph_target()


def _process_scan(scan, from_360=False):

    obj            = scan.get("object", "unknown")
    obj_name       = scan.get("object_name")
    terrain        = scan.get("terrain", "clear")
    in_path        = scan.get("in_my_path", False)
    wall_ahead     = scan.get("wall_ahead", False)
    small_obs      = scan.get("small_obstacle", False)
    void_ahead_f   = scan.get("void_ahead", False)
    action         = scan.get("action", "forward")
    speak_tx       = scan.get("speak")
    reason         = scan.get("physical_reasoning", "")
    distance       = scan.get("distance", "far")
    complete       = scan.get("mission_complete", False)
    target_visible = scan.get("target_visible", False)
    target_dir     = scan.get("target_direction", "front")
    clear_dir      = scan.get("clearest_direction", "front")

    # ── object_name / reasoning override: if Cosmos identified the target, trust it ──
    # Cosmos often puts "Creator stands facing away..." in physical_reasoning
    # but leaves object_name=None and target_visible=False. Check both.
    if not target_visible:
        target_keywords = ["creator", "owner"] + [
            t.lower() for t in (_ms.mission_target_objects or [])
        ]
        reason_lower = str(reason).lower()
        obj_name_lower = str(obj_name or "").lower()
        if any(kw in obj_name_lower for kw in target_keywords):
            log.info(f"_process_scan: object_name='{obj_name}' matches target — forcing target_visible=True")
            target_visible = True
        elif any(kw in reason_lower for kw in target_keywords):
            log.info(f"_process_scan: reasoning mentions target keyword — forcing target_visible=True")
            target_visible = True

    # ── Confidence gate — disabled (always 0.0 from Cosmos) ──────────────
    det_conf = float(scan.get("detection_confidence", 0.0))
    if target_visible and det_conf < DETECTION_CONFIDENCE_MIN:
        pass  # threshold is 0.0 — never suppresses

    if reason:
        log.info(f"Cosmos: {reason}")
        # Strip JSON from UI — only show plain-language reasoning
        reason_display = reason.strip()
        if reason_display.startswith("{") or reason_display.startswith("["):
            reason_display = "(scan complete)"
        _ui("log", f"🧠 {reason_display}")

    # ── Social intent + risk assessment ──────────────────────────────────
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

    # ── Hardware void check — always run regardless of Cosmos result ──────
    hw_void = _void_check()
    if hw_void["void"] or void_ahead_f:
        motors.stop()
        source = hw_void["source"] if hw_void["void"] else "Cosmos vision"
        reason_str = hw_void["reason"] if hw_void["void"] else reason
        log.warning(f"🕳️  VOID in _process_scan ({source}): {reason_str}")
        log_action("VOID_BLOCK_SCAN", f"{source}: {reason_str}")
        _ui("log", f"🕳️  VOID/DROP ahead ({source}) — backing away")
        _ui("status", "VOID — STOPPED")
        motors.oled(0, "VOID AHEAD!")
        motors.oled(1, "Back up!")
        if speak_tx:
            eric_say(speak_tx)
        else:
            eric_say("I detect a drop or void ahead. Backing away for safety.")
        # Back up and turn away from the void
        motors.backward(MOTOR_SPEED_SLOW)
        time.sleep(1.5)
        motors.stop()
        time.sleep(0.3)
        # Turn away — prefer the clearest direction reported by Cosmos
        away = clear_dir if clear_dir not in ("front", "unknown", "") else "right"
        if away in ("left", "left_side"):
            motors.left(MOTOR_SPEED_SLOW); time.sleep(1.8); motors.stop()
        else:
            motors.right(MOTOR_SPEED_SLOW); time.sleep(1.8); motors.stop()
        log_mission_event("void_avoided", f"backed + turned {away}")
        _ms.mission_state = State.SEARCHING
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
        from avoidance import avoid_obstacle
        force_360 = avoid_obstacle(wall_ahead=is_wall, small_obstacle=small_obs)
        if force_360:
            _ms.scans_since_360 = SCANS_BEFORE_360
        else:
            if _safe_to_fwd():
                motors.forward(MOTOR_SPEED_SLOW)
        _ms.mission_state = State.SEARCHING
        return

    if small_obs and not target_visible:
        from avoidance import avoid_obstacle
        avoid_obstacle(wall_ahead=False, small_obstacle=True)
        if _safe_to_fwd():
            motors.forward(MOTOR_SPEED_SLOW)

    if not wall_ahead and not obstacle_close and not small_obs:
        _ms.avoid_attempts = 0
        try:
            from avoidance import reset_avoid_counter
            reset_avoid_counter()
        except ImportError as _exc:
            log.debug(f"avoidance module not loaded: {_exc}")

    if speak_tx and target_visible:
        eric_say(speak_tx)

    # ── Target persistence: Cosmos often flip-flops target_visible ───────────
    if target_visible:
        _ms.target_spotted_count += 1
    else:
        if _ms.target_spotted_count > 0:
            _ms.target_spotted_count -= 1
            if _ms.target_spotted_count > 0:
                log.info("Target_visible=False but keeping target lock (Cosmos flip-flop guard)")
                target_visible = True

    if target_visible:
        _ms.empty_scans = 0
        _ms.target_spotted_count = 0
        direction = str(target_dir).lower().strip() if target_dir else "front"

        # ── Mission-specific alarm: hazard/rescue/security/nature find ────────
        # Trigger alarm for special missions when target matches a mission target.
        # For SAR and security, always alarm on any confirmed target_visible.
        # For hazard patrol / nature, alarm when it matches target_objects list.
        _should_alarm = False
        _alarm_severity = "WARNING"
        # AlarmType.NONE = narrative/find mission — never trigger alarm
        if _ms.mission_alarm_type == AlarmType.NONE:
            _should_alarm = False
        elif _ms.mission_alarm_type in (AlarmType.SIREN, AlarmType.SUSPICIOUS):
            t_lower = str(obj_name or obj).lower()
            _should_alarm = any(kw.lower() in t_lower
                                for kw in (_ms.mission_target_objects or [obj]))
            # Also fire if reasoning describes person as down/unconscious
            if not _should_alarm and obj == "person":
                _sar_kws = ["lying", "unconscious", "on the floor", "on the ground",
                            "fallen", "collapsed", "motionless", "injured", "down"]
                _should_alarm = any(kw in reason.lower() for kw in _sar_kws)
            _alarm_severity = "CRITICAL"
        elif _ms.mission_alarm_type in (AlarmType.HAZARD, AlarmType.NATURE):
            t_lower = str(obj_name or obj).lower()
            _should_alarm = any(kw.lower() in t_lower
                                for kw in (_ms.mission_target_objects or [obj]))
            # Also fire if reasoning describes person as down/unconscious
            if not _should_alarm and obj == "person":
                _sar_kws = ["lying", "unconscious", "on the floor", "on the ground",
                            "fallen", "collapsed", "motionless", "injured", "down"]
                _should_alarm = any(kw in reason.lower() for kw in _sar_kws)
            _alarm_severity = scan.get("severity", "CRITICAL")

        if _should_alarm:
            # Approach first if not already close — then alarm on arrival
            if _ms.mission_flags.get("approach_on_detect", True):
                _ui("log", "SAR: target found — approaching before alarm")
                log_mission_event("target_spotted", f"direction={direction} obj={obj} name={obj_name}")
                if direction not in ("front", "ahead", "unknown", ""):
                    _face_direction(direction)
                _approach_target()   # _confirm_and_photograph_target fires on arrival
            else:
                motors.stop()
                _trigger_mission_alarm(
                    obj_name or obj,
                    location_hint = reason,
                    severity      = _alarm_severity,
                )
            return

        if direction in ("down", "below"):
            _ui("log", "Target is directly below — already arrived!")
            motors.stop()
            motors.pantilt(0, -20)
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

        # ── Greet immediately on spot — before approach ───────────────────
        # For narrative missions (AlarmType.NONE), greet as soon as target confirmed.
        # Photo happens after approach when closer.
        _is_narrative_greet = (
            _ms.mission_alarm_type == AlarmType.NONE
            or str(_ms.mission_alarm_type).lower() in ("none", "null", "")
        )
        if _is_narrative_greet:
            motors.stop()
            motors.pantilt(0, 20, 50)   # look up toward face
            time.sleep(0.3)
            _greeting = ask_cosmos_plain(
                "You just spotted your target. Greet them now in one sentence. "
                "Speak directly to them. No thinking. No JSON.",
                max_tokens=40,
                temperature=0.5
            )
            if not _greeting or _greeting.strip().startswith("{"):
                _greeting = "I found you. I am on my way."
            eric_say(_greeting)
            log_mission_event("target_greeted_on_spot", _greeting[:100])

        _approach_target()
        return

    # Person or robot visible → only stop if near, keep moving if far
    _NEAR_DISTANCES = {"close", "near", "nearby", "very_close", "very close", "right there"}
    if obj in ["person", "robot"] and not target_visible:
        dist_str = str(distance).lower()

        # ── For find-and-greet missions (AlarmType.NONE): approach ANY visible person.
        # _confirm_and_photograph_target() will check the description (glasses, purple
        # sweater, etc.) and either greet or ask for directions if they don't match.
        # This bypasses the "near only" gate that causes Eric to keep walking past you.
        _is_narrative = (
            _ms.mission_alarm_type == AlarmType.NONE
            or str(_ms.mission_alarm_type).lower() in ("none", "null", "")
        )
        if _is_narrative:
            _ms.empty_scans = 0
            _ui("log", f"Find-and-greet mission: person visible — approaching to confirm identity")
            log_mission_event("person_approach_narrative", f"dist={dist_str} obj={obj}")
            _ms.target_spotted_count = max(_ms.target_spotted_count, 1)
            _person_dir = str(target_dir).lower().strip() if target_dir else "front"
            if _person_dir not in ("front", "ahead", "unknown", ""):
                _face_direction(_person_dir)
            _approach_target()
            return

        if dist_str in _NEAR_DISTANCES or in_path:
            _ms.empty_scans = 0
            motors.stop()
            _ms.mission_state = State.INTERACTING
            name = obj_name or obj
            motors.oled(0, name[:16])
            motors.oled(1, "Talking...")
            _ui("status", f"FOUND — {name}")
            motors.pantilt(0, -5)
            time.sleep(0.5)

            # ── Eye-contact gate before greeting ─────────────────────────────
            ec_frame = capture_frame(CAMERA_PANTILT, 320, 240)
            should_greet = True
            if ec_frame:
                try:
                    ec_payload = {
                        "model": COSMOS_MODEL,
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ec_frame}"}},
                            {"type": "text", "text":
                                '{"close_and_facing": true_or_false, "reasoning": "one sentence"} '
                                '— Is the person within 1.5m AND facing/looking toward you?'}
                        ]}],
                        "max_tokens": 50,
                        "temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5,
                    }
                    ec_r = requests.post(VLLM_URL, json=ec_payload, timeout=15)
                    ec_r.raise_for_status()
                    ec_raw = ec_r.json()["choices"][0]["message"]["content"].strip()
                    log_ai("eye_contact_scan", ec_raw, label="EYE_CONTACT")
                    ec = _parse_json(ec_raw, {"close_and_facing": True}, "EYE CONTACT SCAN")
                    should_greet = ec.get("close_and_facing", True)
                    if not should_greet:
                        _ui("log", f"Person near but not facing Eric — not greeting ({ec.get('reasoning','')})")
                        _ms.mission_state = State.SEARCHING
                        if _safe_to_fwd():
                            motors.forward(MOTOR_SPEED_SLOW)
                        return
                except Exception as e:
                    log_exception("eye_contact_scan", e)

            if should_greet:
                # For narrative missions — don't just greet and freeze,
                # hand off to approach so _confirm_and_photograph_target()
                # can do the description check and proper greeting
                _is_nar = (
                    _ms.mission_alarm_type == AlarmType.NONE
                    or str(_ms.mission_alarm_type).lower() in ("none", "null", "")
                )
                if _is_nar:
                    _ui("log", "Narrative mission: handing near person to approach pipeline")
                    _ms.mission_state = State.SEARCHING
                    _ms.target_spotted_count = max(_ms.target_spotted_count, 1)
                    _approach_target()
                    return
                greeting = ask_cosmos_plain(
                    f"You see {name} {'ahead' if in_path else 'nearby'} ({dist_str} away). "
                    "Greet them briefly and ask if they can help with your mission. 1-2 sentences.",
                    max_tokens=80,
                    temperature=0.7
                )
                if not greeting or greeting.strip().startswith("{"):
                    greeting = "Hello there. I am ERIC. Can you help me with my mission?"
                eric_say(greeting)
                log_mission_event("person_greeted_scan", name)
                _ui("status", f"TALKING — {name}")
                # Resume searching after greeting — don't freeze
                time.sleep(1.0)
                _ms.mission_state = State.SEARCHING
                if _safe_to_fwd():
                    motors.forward(MOTOR_SPEED_SLOW)
            return
        else:
            _ui("log", f"Person ({obj_name or 'unknown'}) visible but {dist_str} — continuing")

    if obj in ["clear", "unknown"] and not target_visible:
        _ms.empty_scans += 1
        _ui("log", f"Nothing found ({_ms.empty_scans}/{EMPTY_SCAN_LIMIT})")

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
        from avoidance import avoid_obstacle
        force_360 = avoid_obstacle(wall_ahead=True, small_obstacle=False)
        if force_360:
            _ms.scans_since_360 = SCANS_BEFORE_360
        else:
            if _safe_to_fwd():
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
        if _safe_to_fwd():
            motors.forward(terrain_speed)
    elif action == "stop":
        motors.stop()
    elif action == "turn_right":
        _turn_nav2_or_direct("right", 1.0)
        if _safe_to_fwd():
            motors.forward(terrain_speed)
    elif action == "turn_left":
        _turn_nav2_or_direct("left", 1.0)
        if _safe_to_fwd():
            motors.forward(terrain_speed)
    elif action == "turn_back":
        _turn_nav2_or_direct("back", 1.5)
        if _safe_to_fwd():
            motors.forward(terrain_speed)
    else:
        if _safe_to_fwd():
            motors.forward(terrain_speed)

    _ms.mission_state = State.SEARCHING
    motors.oled(0, "ERIC ACTIVE")
    motors.oled(1, "Searching...")
    _ui("status", "SEARCHING")


# ─── Mission Loop ─────────────────────────────────────────────────────────────
# Layer 1 (LiDAR + OAK-D) handles obstacle/void safety automatically.
# Layer 2 (YOLO on OAK-D Myriad X) handles person/animal detection via callback.
# Eric moves continuously — no Cosmos called while moving.
# Stopped scans (_quick_scan, _best_360_scan) happen on a timer.

# ── Mission loop constants (immutable) ───────────────────────────────────────
NAV_CLIPS_BETWEEN_SCANS = 5   # stopped scan every 5 movement intervals
NAV_MOVE_INTERVAL       = 6.0  # seconds per movement clip
# YOLO detection state and _yolo_lock are in _ms / module scope above


YOLO_POSITION_STALE_S  = 3.0    # seconds — detection data older than this is stale


# ─── Callback from oakd.py ────────────────────────────────────────────────────

def _on_yolo_detection(label: str, distance_m: float,
                       bearing: str, bearing_deg: float):
    """
    Layer 2 YOLO callback — fired from oakd.py reader thread.
    Runs in background thread — only sets flags, never blocks.
    Mission loop picks these up on next iteration.

    Signature change: now accepts bearing_deg for proportional steering.

    If mission is INTERACTING: stores detection but does not set
    _ms.yolo_person_detected — the mission loop will pick it up on
    the next iteration when the state returns to SEARCHING.
    Log entry is always written so the operator sees it in the GUI.
    """

    if not _ms.mission_active:
        return

    # Always log — even during interaction, operator should see this
    _ui("log", f"👁️  YOLO: {label} at {distance_m:.1f}m "
               f"({bearing} / {bearing_deg:+.0f}°)")
    log_action("YOLO_DETECT", f"{label} {distance_m:.1f}m "
                               f"{bearing} {bearing_deg:+.0f}°")

    if _ms.mission_state in (State.COMPLETE,):
        return  # Mission done — ignore

    # Store position even when INTERACTING (recovered on next SEARCHING tick)
    with _yolo_lock:
        _ms.yolo_detect_label       = label
        _ms.yolo_detect_distance    = distance_m
        _ms.yolo_detect_bearing     = bearing
        _ms.yolo_detect_bearing_deg = bearing_deg
        _ms.yolo_detect_time        = time.monotonic()

        if _ms.mission_state not in (State.INTERACTING,):
            _ms.yolo_person_detected = True
        # If INTERACTING: flag stays False — picked up on next SEARCHING loop


# ─── Register / unregister ────────────────────────────────────────────────────

def _register_yolo_callback():
    """Register Layer 2 YOLO callback with oakd.py."""
    try:
        from oakd import set_yolo_callback, set_yolo_active
        set_yolo_callback(_on_yolo_detection)
        set_yolo_active(True)
        log.info("✅ Layer 2 YOLO callback registered")
    except Exception as e:
        log.warning(f"YOLO callback registration failed ({e}) — Layer 2 disabled")


def _unregister_yolo_callback():
    """Pause YOLO detections when mission stops."""
    try:
        from oakd import set_yolo_callback, set_yolo_active, clear_yolo_motor_stop
        set_yolo_active(False)
        set_yolo_callback(None)
        clear_yolo_motor_stop()
    except Exception as _exc:  # oakd/yolo
        log.debug(f"oakd/yolo unavailable: {_exc}")


# ─── Handle detection in mission loop ────────────────────────────────────────

def _handle_yolo_detection() -> bool:
    """
    Handle a pending YOLO detection from the mission loop.
    Returns True if a detection was handled (loop should skip rest of iteration).

    Fixes applied vs previous version:
    ─────────────────────────────────
    • Proportional steering: turn duration scales with |bearing_deg|, not fixed 0.3s.
    • Layer 2 motor guard: checks oakd.yolo_motor_stop_issued() before any
      if _safe_to_fwd():
          motors.forward() to avoid overriding a Layer 2 stop with a move command.
    • Stale data guard: if stored detection is older than YOLO_POSITION_STALE_S,
      falls back to oakd.get_last_yolo_position() for fresh spatial data.
    • Mission-aware: approach_on_detect flag from YAML still controls behaviour.
    """

    with _yolo_lock:
        if not _ms.yolo_person_detected:
            return False
        label       = _ms.yolo_detect_label
        dist_m      = _ms.yolo_detect_distance
        bearing     = _ms.yolo_detect_bearing
        bearing_deg = _ms.yolo_detect_bearing_deg
        detect_age  = time.monotonic() - _ms.yolo_detect_time
        _ms.yolo_person_detected = False   # clear flag

    # ── Stale detection guard ─────────────────────────────────────────────────
    # If the stored detection is old (Cosmos was running), get fresh position.
    if detect_age > YOLO_POSITION_STALE_S:
        try:
            from oakd import get_last_yolo_position
            fresh = get_last_yolo_position(label)
            if fresh:
                dist_m      = fresh["dist_m"]
                bearing     = fresh["bearing"]
                bearing_deg = fresh["bearing_deg"]
                _ui("log", f"YOLO: refreshed position from OAK-D memory "
                           f"({detect_age:.1f}s old → {dist_m:.1f}m {bearing})")
            else:
                _ui("log", f"YOLO: detection {detect_age:.1f}s old, no fresh data — skipping")
                return True  # consumed the flag, don't act on stale data
        except Exception as _exc:  # oakd/yolo
            log.debug(f"oakd/yolo unavailable: {_exc}")

    # ── Clear the Layer 2 motor guard ────────────────────────────────────────
    # Layer 2 already called motors.stop() for this detection.
    # We take over mission logic from here. Clear the flag so mission loop
    # knows it can issue motor commands again once it has handled this event.
    try:
        from oakd import yolo_motor_stop_issued, clear_yolo_motor_stop
        if yolo_motor_stop_issued():
            clear_yolo_motor_stop()
    except Exception as _exc:  # oakd/yolo
        log.debug(f"oakd/yolo unavailable: {_exc}")

    # ── Mission flag ──────────────────────────────────────────────────────────
    approach    = _ms.mission_flags.get("approach_on_detect", True)
    detect_dist = float(_ms.mission_flags.get("detect_distance", 2.0))

    if dist_m > detect_dist:
        # ── Far detection: slow + steer proportionally toward target ──────────
        # Turn duration: 0.2s at bearing ±5°, up to 0.8s at bearing ±45°
        turn_sec = max(0.2, min(0.8, abs(bearing_deg) / 45.0 * 0.8))
        motors.slow()
        _ui("log", f"YOLO: {label} at {dist_m:.1f}m ({bearing} / "
                   f"{bearing_deg:+.0f}°) — steering {turn_sec:.2f}s")

        if bearing_deg < -5.0:     # target is to the left
            _turn_nav2_or_direct("left", turn_sec)
        elif bearing_deg > 5.0:    # target is to the right
            _turn_nav2_or_direct("right", turn_sec)
        # If within ±5° — already centred, just slow forward

        # Safety: don't issue forward if Layer 2 issued a stop since we read the flag
        try:
            from oakd import yolo_motor_stop_issued
            if not yolo_motor_stop_issued():
                if _safe_to_fwd():
                    motors.forward(MOTOR_SPEED_SLOW)
        except Exception as _exc:  # oakd/yolo
            log.debug(f"oakd/yolo error: {_exc}")
            if _safe_to_fwd():
                motors.forward(MOTOR_SPEED_SLOW)
        return True

    # ── Close enough — stop and execute step action or greet ─────────────────
    motors.stop()
    motors.oled(0, label[:16])
    motors.oled(1, f"{dist_m:.1f}m {bearing}")
    _ui("status", f"YOLO FOUND — {label.upper()}")
    log_mission_event("yolo_found", f"{label} {dist_m:.1f}m {bearing} {bearing_deg:+.0f}°")

    # Build sensor context snapshot for Cosmos — include YOLO ground truth
    sensor_ctx = _sensor_context()
    step = _current_step()
    step_info = f"Current mission step: find {step.target} and {step.action}." if step else ""

    # ── Check if YOLO label matches the current mission step target ────────────
    # If it does, YOLO hardware has confirmed the target at close range — no
    # need for a Cosmos vision call. Execute the step action directly.
    # "person" YOLO label matches any human-target step
    _YOLO_TARGET_STEPS = {"person", "robot"}
    step_is_person_target = (step is not None and
                             any(kw in step.target.lower()
                                 for kw in ("person", "man", "woman", "human",
                                            "droid", "robot")))

    if approach and step_is_person_target and label in _YOLO_TARGET_STEPS:
        _ui("log", f"✅ YOLO hardware confirms step target '{step.target}' "
                   f"at {dist_m:.2f}m — executing step action directly")
        log_mission_event("yolo_step_complete",
                          f"{label} at {dist_m:.2f}m matches step target={step.target}")

        # GAP 1 FIX: greeting submitted async — mission_state = INTERACTING
        # immediately so the main loop gates on it while Cosmos thinks.
        # _execute_step_action() is called from the async thread once the
        # greeting TTS completes so the interaction sequence is preserved.
        _ms.mission_state = State.INTERACTING
        motors.pantilt(0, -15)

        _greeting_prompt = (
            f"Your YOLO sensors just confirmed {label} at {dist_m:.2f}m to your {bearing}. "
            f"{step_info} "
            "Greet them warmly in ONE sentence."
        )
        _step_label = label   # capture for closure

        def _greet_and_execute():
            try:
                greeting = ask_cosmos(_greeting_prompt, max_tokens=60)
                eric_say(greeting)
            except Exception as _ge:
                log.warning(f"YOLO greeting failed ({_ge}) — skipping greeting")
            time.sleep(0.3)
            _execute_step_action(_step_label)

        _cosmos_executor.submit(_greet_and_execute)
        return True

    # ── Not a step-target match — normal greeting / observation ───────────────
    # GAP 1 FIX: set INTERACTING first so the main loop parks immediately,
    # then submit greeting to the executor — never blocks the hot path.
    _ms.mission_state = State.INTERACTING

    if approach:
        _ui("log", f"YOLO: {label} confirmed at {dist_m:.1f}m — greeting async")
        _greet_prompt = (
            f"{sensor_ctx}"
            f"HARDWARE DETECTION — Layer 2 YOLO (OAK-D Myriad X VPU) has confirmed:\n"
            f"  Target: {label}\n"
            f"  Distance: {dist_m:.2f}m (stereo depth — ground truth)\n"
            f"  Bearing: {bearing} ({bearing_deg:+.0f}° off-centre)\n"
            f"  {step_info}\n\n"
            "You have stopped and turned to face them. "
            "Greet them naturally, mention that your sensors detected them, "
            "and ask if they can help with your mission. 1-2 sentences."
        )

        def _greet_approach():
            try:
                greeting = ask_cosmos(_greet_prompt, max_tokens=100)
                eric_say(greeting)
            except Exception as _ge:
                log.warning(f"YOLO approach greeting failed ({_ge})")

        _cosmos_executor.submit(_greet_approach)
    else:
        _ui("log", f"YOLO: {label} at {dist_m:.1f}m — reporting async (approach_on_detect=false)")
        _report_prompt = (
            f"{sensor_ctx}"
            f"HARDWARE DETECTION — Layer 2 YOLO confirmed {label} at {dist_m:.2f}m "
            f"to your {bearing} ({bearing_deg:+.0f}°). {step_info} "
            "Report what you observe. Stay in position. 1-2 sentences."
        )

        def _report_observe():
            try:
                report = ask_cosmos(_report_prompt, max_tokens=80)
                eric_say(report)
            except Exception as _re:
                log.warning(f"YOLO observe report failed ({_re})")

        _cosmos_executor.submit(_report_observe)

    return True

def _mission_loop():

    # ── Register YOLO Layer 2 callback ────────────────────────────────────────
    _register_yolo_callback()

    # ── Initial sweep SKIPPED — start moving immediately ────
    # Was: 180° chassis sweep + full 360 scan before moving (~60s total).
    # Now: single quick scan then move — much faster startup.
    _ui("log", "🔍 Quick initial scan...")
    _ui("status", "SCANNING")
    motors.oled(0, "ERIC ACTIVE")
    motors.oled(1, "Quick scan...")
    motors.stop()
    motors.pantilt(0, -5)
    time.sleep(0.3)

    quick = _quick_scan()
    _process_scan(quick, from_360=False)


    if _ms.mission_active and _ms.mission_state == State.SEARCHING:
        if _safe_to_fwd():
            motors.forward(MOTOR_SPEED_SLOW)

    _ms.nav_clips_since_scan = 0

    while _ms.mission_active:
        try:
            if _ms.mission_state in (State.INTERACTING, State.COMPLETE):
                time.sleep(0.5)
                continue

            # ── Layer 2: handle pending YOLO detection ────────────────────────
            # Check YOLO flag set by _on_yolo_detection() callback.
            # Layer 1 already stopped/slowed motors — we just handle mission logic.
            if _handle_yolo_detection():
                continue

            # ── Move continuously — poll every 200ms for YOLO detections ──────
            # Industrial standard: move until a reason to stop, not on a timer.
            # YOLO detects birds/people mid-move, handled within 200ms not 5s.
            # Layer 1 (LiDAR/OAK-D) still fires motors.stop() instantly from
            # its own thread — the poll handles mission-level reactions only.
            if _ms.nav_clips_since_scan < NAV_CLIPS_BETWEEN_SCANS:
                _ms.nav_clips_since_scan += 1
                if _safe_to_fwd():
                    motors.forward(MOTOR_SPEED_SLOW)

                move_deadline       = time.monotonic() + NAV_MOVE_INTERVAL
                yolo_broke          = False
                _last_async_nav_t   = 0.0   # GAP 2: track last async nav fire time
                while time.monotonic() < move_deadline and _ms.mission_active:
                    time.sleep(0.1)  # 100ms poll — faster stop response

                    # ── IMMEDIATE STOP if mission deactivated ─────────────────
                    if not _ms.mission_active:
                        motors.stop()
                        break

                    # YOLO found something mid-move — stop and handle it now
                    with _yolo_lock:
                        yolo_pending = _ms.yolo_person_detected
                    if yolo_pending:
                        motors.stop()
                        yolo_broke = True
                        break

                    # Mission paused mid-move (interaction started)
                    if _ms.mission_state in (State.INTERACTING, State.COMPLETE):
                        motors.stop()
                        yolo_broke = True
                        break

                    # ── Poll LiDAR directly inside move loop ──────────────────
                    # The LiDAR thread already calls motors.stop() but we also
                    # need to break out of THIS loop and flush the stale async
                    # nav result that says "action: forward" — otherwise the
                    # next _nav_check_async() call returns the old cached result
                    # and immediately re-starts forward motion into the wall.
                    try:
                        from lidar import obstacle_close as _lidar_close_poll
                        if _lidar_close_poll():
                            motors.stop()
                            _ms.last_nav_result = {}  # flush stale "forward" result
                            _ms.nav_clips_since_scan = NAV_CLIPS_BETWEEN_SCANS  # force stopped scan before resuming
                            yolo_broke = True
                            break
                    except Exception:
                        pass

                    # GAP 2 FIX: fire async Cosmos nav check while moving.
                    # _nav_check_async() returns instantly (fire-and-forget +
                    # last_nav_result). Hardware gates (LiDAR/OAK-D) always run
                    # first inside it — Cosmos vision runs in parallel.
                    # Only acts on result if Cosmos signals a hard stop.
                    now_nav = time.monotonic()
                    if now_nav - _last_async_nav_t >= NAV_IMAGE_INTERVAL:
                        _last_async_nav_t = now_nav
                        try:
                            nav_r = _nav_check_async()
                            if nav_r.get("action") == "stop" or nav_r.get("wall_ahead"):
                                motors.stop()
                                _ui("log", f"🛑 Async nav check: stop — "
                                           f"{nav_r.get('physical_reasoning', '')[:60]}")
                                yolo_broke = True   # reuse flag — skip post-move LiDAR check
                                break
                        except Exception as _nav_exc:
                            log.debug(f"async nav check error: {_nav_exc}")

                if not _ms.mission_active:
                    motors.stop()
                    break

                if yolo_broke:
                    continue

                # After moving, check if Layer 1 stopped for an obstacle
                try:
                    from lidar import obstacle_close as lidar_close
                    if lidar_close():
                        _ui("log", "LiDAR: obstacle — avoiding")
                        log_action("LAYER1_OBSTACLE", "lidar triggered avoidance")
                        from avoidance import avoid_obstacle
                        force_360 = avoid_obstacle(wall_ahead=True, small_obstacle=False)
                        if force_360:
                            _ms.scans_since_360 = SCANS_BEFORE_360
                            _ms.nav_clips_since_scan = NAV_CLIPS_BETWEEN_SCANS
                        else:
                            # Force a quick scan before resuming — don't blindly forward
                            # after avoidance when the path ahead may still be unclear
                            _ms.nav_clips_since_scan = NAV_CLIPS_BETWEEN_SCANS
                except Exception as _exc:  # lidar
                    log.debug(f"lidar unavailable: {_exc}")
                continue

            # ── Stopped scan every NAV_CLIPS_BETWEEN_SCANS movement intervals ─
            _ms.nav_clips_since_scan = 0
            _ms.scans_since_360 += 1
            motors.stop()
            time.sleep(0.5)  # raised from 0.3 — chassis damping before capture

            do_360 = (_ms.empty_scans >= EMPTY_SCAN_LIMIT or
                      _ms.scans_since_360 >= SCANS_BEFORE_360)

            if do_360:
                # ── Try circumnavigation first (only if YAML flag set) ─────────
                # Peeks around the blocking obstacle before committing to a full
                # 360. If it finds the target, _process_scan() is called inside
                # _circumnavigate_obstacle() and we skip the 360 entirely.
                if _ms.empty_scans >= EMPTY_SCAN_LIMIT:
                    if _circumnavigate_obstacle():
                        _ms.scans_since_360 = _ms.empty_scans = 0
                        if _ms.mission_active and _ms.mission_state == State.SEARCHING:
                            if not _void_check()["void"]:
                                if _safe_to_fwd():
                                    motors.forward(MOTOR_SPEED_SLOW)
                        continue   # back to top of loop — target handled
                    eric_say("Nothing found. Performing a full 360 scan.")
                else:
                    _ui("log", "Periodic 360 scan...")
                log_mission_event("360_scan_triggered",
                                  f"empty={_ms.empty_scans} scans_since={_ms.scans_since_360}")
                scan = _best_360_scan()
                _ms.scans_since_360 = _ms.empty_scans = 0
                _process_scan(scan, from_360=True)
                try:
                    from avoidance import reset_avoid_counter
                    reset_avoid_counter()
                except ImportError as _exc:
                    log.debug(f"avoidance module not loaded: {_exc}")
                # Only resume forward if mission still active and not blocked
                if _ms.mission_active and _ms.mission_state == State.SEARCHING:
                    if not _void_check()["void"]:
                        if _safe_to_fwd():
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
    _unregister_yolo_callback()
    _ms.mission_state = State.IDLE
    _ui("status", "IDLE")
    log.info("Mission loop ended")