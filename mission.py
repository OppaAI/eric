"""
ERIC — Mission Logic

Camera strategy:
  Navigation (moving):  pan-tilt image every NAV_IMAGE_INTERVAL seconds
  Scanning  (stopped):  scan_and_identify() — wide spot + webcam close-up ID
  360° scan (stopped):  body rotates in 4 stops, scan_and_identify() at each
                        → confirms target mid-scan and stops early if found
  Face centering:       pan-tilt + body align so target is in both cameras

Stabilization rule:
  Every pantilt_move_wait() includes a settle delay.
  Captures only happen when robot is stopped or pan-tilt has settled.

LED:
  Adaptive — on only when captured frame is dark.

Nav2 (optional):
  If USE_NAV2=true and ROS2 Nav2 is running, Eric sends goal poses.
  Falls back transparently to direct motor control if Nav2 unavailable.

LiDAR (optional):
  If USE_LIDAR=true, D500 runs as independent safety layer.
  Stops Eric if obstacle < LIDAR_STOP_DIST regardless of Cosmos state.
"""

import time
import threading
import logging
import json
import requests

from config import (
    MOTOR_SPEED_SLOW, MOTOR_SPEED_NORMAL, MISSIONS_DIR,
    VLLM_URL, COSMOS_MODEL, USE_NAV2
)
from motors import motors
from cosmos import (
    ask_cosmos, set_mission_briefing,
    capture_frame, capture_dual_stable, capture_nav_frame,
    center_on_person, scan_and_identify,
    pantilt, pantilt_center, pantilt_move_wait,
    autofocus_trigger, autofocus_enable,
    CAMERA_WEBCAM, CAMERA_PANTILT
)
from tts import speak

log = logging.getLogger("eric.mission")


# ─── State ────────────────────────────────────────────────────────────────────

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

EMPTY_SCAN_LIMIT   = 1   # trigger 360 after 1 empty scan
SCANS_BEFORE_360   = 2   # periodic 360 every 2 nav checks
MAX_AVOID_ATTEMPTS = 3   # force 360 after this many failed avoids

NAV_IMAGE_INTERVAL = 4.0   # seconds between nav image checks while moving
TURN_90_SEC        = 2.2   # seconds to rotate 90° at MOTOR_SPEED_SLOW — tune if needed
BLUR_THRESHOLD     = 80.0
MAX_BLUR_RETRIES   = 3

_ui_callbacks = {"eric_says": None, "status": None, "log": None}


# ─── UI Callbacks ─────────────────────────────────────────────────────────────

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


# ─── Cosmos Multi-Frame Helper ────────────────────────────────────────────────

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
            for k, v in fallback.items():
                result.setdefault(k, v)
            print(f"\n{'─'*60}")
            print(f"🧠 {label}:")
            for k, v in result.items():
                icon = ""
                if k == "wall_ahead"       and v: icon = "  🚧"
                if k == "obstacle_close"   and v: icon = "  🚧"
                if k == "small_obstacle"   and v: icon = "  ⚠️"
                if k == "target_visible"   and v: icon = "  🎯"
                if k == "confirmed_target" and v: icon = "  🎯"
                if k == "mission_complete" and v: icon = "  🏆"
                if k == "speak"            and v: icon = "  🔊"
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

    try:
        autofocus_enable(CAMERA_WEBCAM)
        autofocus_enable(CAMERA_PANTILT)
    except Exception:
        pass

    pantilt_center()

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
    _cancel_nav2()
    motors.stop()
    motors.lights(0, 0)
    pantilt_center()
    motors.oled(0, "ERIC STOPPED")
    motors.oled(1, "")
    eric_say("Mission disengaged. All systems halted.")
    _ui("status", "IDLE")


def resume_after_interaction():
    global mission_state, _empty_scans, _avoid_attempts, _scans_since_360
    if mission_active:
        _empty_scans = _avoid_attempts = _scans_since_360 = 0
        mission_state = State.SEARCHING
        pantilt_center()
        _move_forward()
        motors.oled(0, "ERIC ACTIVE")
        motors.oled(1, "Searching...")
        _ui("status", "SEARCHING")


# ─── Nav2 / Motor Helpers ─────────────────────────────────────────────────────

def _move_forward():
    """Move forward — via Nav2 if available, else direct motor."""
    if USE_NAV2:
        try:
            from nav2 import nav2_available, navigate_to_person
            if nav2_available():
                navigate_to_person("front")
                return
        except Exception:
            pass
    motors.forward(MOTOR_SPEED_SLOW)


def _navigate_direction(direction: str):
    """
    Navigate toward a direction.
    Uses Nav2 goal if available, else direct motor turn + forward.
    direction: "front" | "left" | "right" | "back"
    """
    if USE_NAV2:
        try:
            from nav2 import nav2_available, navigate_to_person
            if nav2_available():
                navigate_to_person(direction)
                return
        except Exception:
            pass
    # Direct motor fallback
    _face_direction(direction)
    motors.forward(MOTOR_SPEED_SLOW)


def _cancel_nav2():
    """Cancel any active Nav2 goal."""
    if USE_NAV2:
        try:
            from nav2 import cancel_goal, nav2_available
            if nav2_available():
                cancel_goal()
        except Exception:
            pass


# ─── Blur Check ───────────────────────────────────────────────────────────────

def _is_blurry(frame_b64: str) -> bool:
    try:
        import cv2
        import numpy as np
        import base64
        data  = base64.b64decode(frame_b64)
        arr   = np.frombuffer(data, np.uint8)
        img   = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False
        return cv2.Laplacian(img, cv2.CV_64F).var() < BLUR_THRESHOLD
    except Exception:
        return False


def _capture_sharp_local(device: int) -> str | None:
    best = None
    for attempt in range(MAX_BLUR_RETRIES):
        f = capture_frame(device, 640, 480, adaptive_led=True)
        if f is None:
            break
        if not _is_blurry(f):
            return f
        best = f
        time.sleep(0.5)
    return best


# ─── Prompts ──────────────────────────────────────────────────────────────────

NAV_IMAGE_PROMPT = """
You are a tracked ground robot moving forward. This is a single frame from your forward camera.

Check ONLY for immediate safety hazards:
- Wall or large object filling the lower 40% of frame → wall_ahead = true
- Any object within ~60cm directly ahead → obstacle_close = true
- Small ground obstacle (cables, steps, rug edges) → small_obstacle = true
- Person or robot visible anywhere → person_visible = true

Respond ONLY with valid JSON:
{
  "wall_ahead": false,
  "obstacle_close": false,
  "small_obstacle": false,
  "person_visible": false,
  "action": "forward|slow|stop|turn_left|turn_right",
  "physical_reasoning": "one sentence"
}
"""

SCAN_360_OVERVIEW_PROMPT = """
These images are from a full 360-degree scan — 4 body positions (0°, 90°, 180°, 270°).
At each position I captured wide-angle frames tilted up and level. I am completely stopped.
No target was confirmed during the scan. Help me decide what to do next.

STEP 1 — OBSTACLES: Which direction has the most clear open space? → clearest_direction

STEP 2 — ANYTHING INTERESTING: Any partial view of a person, robot, or figure in any frame?

STEP 3 — DIRECTION: Where should I go to continue searching?

Respond ONLY with valid JSON:
{
  "object": "person|robot|obstacle|wall|clear|unknown",
  "object_name": null,
  "terrain": "carpet|tiles|wood|clear",
  "distance": "close|medium|far",
  "in_my_path": false,
  "wall_ahead": false,
  "obstacle_close": false,
  "small_obstacle": false,
  "target_visible": false,
  "target_direction": "front|right|back|left|unknown",
  "clearest_direction": "front|right|back|left",
  "action": "forward|slow|turn_right|turn_left|turn_back|navigate_around|stop",
  "speak": null,
  "physical_reasoning": "1 sentence",
  "mission_complete": false
}
"""

_SCAN_FALLBACK = {
    "object": "unknown", "object_name": None, "terrain": "clear",
    "distance": "far", "in_my_path": False, "wall_ahead": False,
    "obstacle_close": False, "small_obstacle": False,
    "target_visible": False, "target_direction": "unknown",
    "clearest_direction": "front",
    "action": "stop", "speak": None,
    "physical_reasoning": "", "mission_complete": False
}

_NAV_FALLBACK = {
    "wall_ahead": False, "obstacle_close": False, "small_obstacle": False,
    "person_visible": False,
    "action": "stop", "physical_reasoning": ""
}


# ─── Navigation Check (image-based, while moving) ─────────────────────────────

def _nav_check() -> dict:
    """
    Single image nav check while robot is moving.
    Much faster than video clip — allows more frequent obstacle detection.
    Person spotted → stop and greet.
    """
    _ui("log", "📷 Nav check...")
    motors.oled(1, "Nav check...")

    frame = capture_nav_frame()
    if not frame:
        return dict(_NAV_FALLBACK)

    try:
        response = ask_cosmos(NAV_IMAGE_PROMPT, image_b64=frame, max_tokens=120)
        result   = _parse_json(response, dict(_NAV_FALLBACK), label="NAV CHECK")

        if result.get("person_visible") and mission_active:
            motors.stop()
            _cancel_nav2()
            _ui("log", "👤 Person spotted during nav — stopping")
            _ui("status", "PERSON SPOTTED")
            greeting = ask_cosmos(
                "You just spotted someone ahead while navigating. "
                "Greet them warmly and ask if they can help with your mission. 1-2 sentences.",
                max_tokens=60
            )
            eric_say(greeting)

        return result
    except Exception as e:
        log.error(f"Nav check error: {e}")
        return dict(_NAV_FALLBACK)


# ─── Smart 360° Scan ──────────────────────────────────────────────────────────

def _scan_360() -> dict:
    """
    Full 360° body rotation scan using image-based dual-camera identification.

    At each of 4 body positions (0°, 90°, 180°, 270°):
      1. scan_and_identify() — wide-angle spots object, webcam confirms identity
      2. If target CONFIRMED mid-scan → return immediately (early exit)
      3. If potential but unconfirmed → remember best candidate
      4. Collect overview frames throughout

    After full rotation:
      - If best candidate found → revisit and try one more ID
      - If still nothing → send overview frames to Cosmos for direction guidance

    This is image-based (not video) — faster, lighter on VRAM, smarter.
    """
    global mission_state
    mission_state = State.SCANNING_360
    _ui("status", "360 SCANNING")
    motors.oled(0, "360 Scan")
    log.info("Starting smart 360 image scan")

    motors.stop()
    _cancel_nav2()
    time.sleep(0.5)

    all_frames = []
    best_spot  = None   # (body_steps_from_start, result) of best unconfirmed potential

    for pos in range(4):
        deg = pos * 90
        _ui("log", f"Scanning {deg}°...")
        motors.oled(1, f"Scan {deg}deg")

        for tilt, label in [(-20, "up"), (10, "level")]:
            pantilt_move_wait(0, tilt, speed=40)

            # Collect overview frame
            f_wide = _capture_sharp_local(CAMERA_PANTILT)
            if f_wide:
                all_frames.append(f_wide)

            # Smart dual-camera identification at this position
            result = scan_and_identify(adaptive_led=True)

            # ── TARGET CONFIRMED — stop scan early ────────────────────────
            if result.get("confirmed_target"):
                log.info(f"🎯 Target CONFIRMED at {deg}° tilt={label} — early exit!")
                _ui("log", f"Target confirmed at {deg}° — stopping scan!")
                motors.oled(1, "TARGET FOUND!")
                pantilt_center()
                return _confirmed_to_scan_result(result, deg)

            # ── Potential target — remember best ─────────────────────────
            if (result.get("potential_target") and
                    result.get("confidence") in ("medium", "high")):
                if best_spot is None or result.get("confidence") == "high":
                    best_spot = (pos, result)
                    log.info(f"Potential target at {deg}° ({result.get('confidence')}) — continuing")

            # ── Hard obstacle ─────────────────────────────────────────────
            if result.get("wall_ahead") or result.get("obstacle_close"):
                log.info(f"Obstacle detected at {deg}° during 360 scan")

        pantilt_center()

        # Rotate body 90° (skip on last position)
        if pos < 3:
            motors.right(MOTOR_SPEED_SLOW)
            time.sleep(TURN_90_SEC)
            motors.stop()
            time.sleep(0.5)

    # ── Revisit best unconfirmed candidate ────────────────────────────────────
    if best_spot:
        orig_pos, _ = best_spot
        _ui("log", f"Revisiting best potential at {orig_pos * 90}°...")
        log.info(f"Revisiting potential target at pos {orig_pos}")

        # Rotate back — we're currently at pos 3, need to get to orig_pos
        steps_back = (4 - orig_pos) % 4
        if steps_back > 0:
            motors.right(MOTOR_SPEED_SLOW)
            time.sleep(TURN_90_SEC * steps_back)
            motors.stop()
            time.sleep(0.5)

        result = scan_and_identify(adaptive_led=True)
        if result.get("confirmed_target"):
            log.info("🎯 Target confirmed on second look!")
            pantilt_center()
            return _confirmed_to_scan_result(result, orig_pos * 90)

    # ── No target — send overview to Cosmos for direction ─────────────────────
    log.info(f"No target — sending {len(all_frames)} overview frames to Cosmos")
    _ui("log", f"360 done — {len(all_frames)} frames → Cosmos overview")
    motors.oled(1, "Analyzing...")

    if not all_frames:
        return dict(_SCAN_FALLBACK)

    try:
        response = _cosmos_frames(all_frames, SCAN_360_OVERVIEW_PROMPT, max_tokens=250, temp=0.2)
        return _parse_json(response, dict(_SCAN_FALLBACK), label="360° OVERVIEW")
    except Exception as e:
        log.error(f"360 Cosmos overview error: {e}")
        return dict(_SCAN_FALLBACK)


def _confirmed_to_scan_result(result: dict, deg: int) -> dict:
    """Convert scan_and_identify() confirmed result to _process_scan() format."""
    return {
        "object":             result.get("object_type", "person"),
        "object_name":        result.get("object_name"),
        "terrain":            "clear",
        "distance":           "close",
        "in_my_path":         True,
        "wall_ahead":         False,
        "obstacle_close":     False,
        "small_obstacle":     False,
        "target_visible":     True,
        "target_direction":   "front",
        "clearest_direction": "front",
        "action":             "stop",
        "speak":              result.get("speak"),
        "physical_reasoning": f"Target confirmed by webcam ID at {deg}°",
        "mission_complete":   False
    }


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
    Cancels any Nav2 goal before manoeuvring.
    """
    global _avoid_attempts, mission_state
    _avoid_attempts += 1
    mission_state = State.AVOIDING
    _ui("status", "AVOIDING")
    _cancel_nav2()

    if wall_ahead:
        _ui("log", f"Wall — attempt {_avoid_attempts}")
        motors.oled(1, "Wall! Back up...")
        motors.stop(); time.sleep(0.3)
        motors.backward(MOTOR_SPEED_SLOW); time.sleep(1.5)
        motors.stop(); time.sleep(0.3)

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

        # Re-scan after turning
        _ui("log", "Re-scanning after avoidance...")
        frame = capture_nav_frame()
        if frame:
            r = ask_cosmos(NAV_IMAGE_PROMPT, image_b64=frame, max_tokens=100)
            rescan = _parse_json(r, dict(_NAV_FALLBACK), label="POST-AVOID CHECK")
            if rescan.get("wall_ahead") or rescan.get("obstacle_close"):
                _ui("log", "Still blocked — trying again")
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

    return False


# ─── Mission Complete ─────────────────────────────────────────────────────────

def _handle_mission_complete(obj_name):
    global mission_active, mission_state
    log.info(f"MISSION COMPLETE — {obj_name}")
    mission_state = State.COMPLETE
    _cancel_nav2()
    motors.stop()
    motors.oled(0, "MISSION DONE!")
    motors.oled(1, (obj_name or "Target")[:16])
    _ui("status", "MISSION COMPLETE")

    for _ in range(5):
        motors.lights(255, 255); time.sleep(0.25)
        motors.lights(0, 0);    time.sleep(0.25)
    motors.lights(128, 255)

    pantilt_move_wait(0, -10)
    autofocus_trigger(CAMERA_PANTILT)
    center_on_person()

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
    obstacle_close = scan.get("obstacle_close", False)
    small_obs      = scan.get("small_obstacle", False)
    action         = scan.get("action", "forward")
    speak_tx       = scan.get("speak")
    reason         = scan.get("physical_reasoning", "")
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

    # Unknown in path → treat as obstacle
    if obj == "unknown" and in_path:
        log.info("Unknown object in path — treating as obstacle")
        wall_ahead = True

    if wall_ahead or obstacle_close or (in_path and obj in ["wall", "obstacle"]):
        motors.stop()
        _cancel_nav2()
        if speak_tx: eric_say(speak_tx)
        is_wall   = wall_ahead or (obj == "wall")
        force_360 = _avoid_obstacle(wall_ahead=is_wall, small_obstacle=small_obs)
        if force_360:
            _scans_since_360 = SCANS_BEFORE_360
        else:
            _move_forward()
        mission_state = State.SEARCHING
        return

    if small_obs and not target_visible:
        _avoid_obstacle(wall_ahead=False, small_obstacle=True)
        _move_forward()

    if not wall_ahead and not obstacle_close and not small_obs:
        _avoid_attempts = 0

    if speak_tx:
        eric_say(speak_tx)

    if target_visible and from_360:
        _empty_scans = 0
        _ui("log", f"Target spotted {target_dir}!")
        motors.oled(1, f"Target {target_dir}!")
        _ui("status", "TARGET SPOTTED")
        _navigate_direction(target_dir)
        mission_state = State.SEARCHING
        return

    if in_path and obj in ["person", "robot"]:
        _empty_scans = 0
        motors.stop()
        _cancel_nav2()
        mission_state = State.INTERACTING
        name = obj_name or obj
        motors.oled(0, name[:16])
        motors.oled(1, "Centering...")
        _ui("status", f"FOUND — {name}")

        if not center_on_person():
            pantilt_move_wait(0, -15)
        autofocus_trigger(CAMERA_PANTILT)
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
        _navigate_direction(clear_dir)
    elif action == "navigate_around":
        motors.left(MOTOR_SPEED_SLOW); time.sleep(0.8)
        _move_forward()
    elif action in ("slow",) or terrain == "carpet":
        motors.slow()
    elif action == "stop":
        motors.stop()
    elif action == "turn_right":
        motors.right(MOTOR_SPEED_SLOW); time.sleep(1.0); motors.stop()
        _move_forward()
    elif action == "turn_left":
        motors.left(MOTOR_SPEED_SLOW);  time.sleep(1.0); motors.stop()
        _move_forward()
    elif action == "turn_back":
        motors.right(MOTOR_SPEED_SLOW); time.sleep(3.0); motors.stop()
        _move_forward()
    else:
        _move_forward()

    mission_state = State.SEARCHING
    motors.oled(0, "ERIC ACTIVE")
    motors.oled(1, "Searching...")
    _ui("status", "SEARCHING")


# ─── Mission Loop ─────────────────────────────────────────────────────────────

_nav_checks_since_scan = 0
NAV_CHECKS_BETWEEN_SCANS = 3  # stopped scan every 3 nav checks (~12s of movement)


def _mission_loop():
    global mission_active, mission_state, _empty_scans, _scans_since_360
    global _avoid_attempts, _nav_checks_since_scan

    eric_say("Starting initial 360 degree scan of the area.")
    scan = _scan_360()
    _process_scan(scan, from_360=True)

    if mission_active and mission_state == State.SEARCHING:
        _move_forward()

    _nav_checks_since_scan = 0

    while mission_active:
        try:
            if mission_state in (State.INTERACTING, State.COMPLETE):
                time.sleep(0.5)
                continue

            # ── Nav image check while moving ──────────────────────────────
            if _nav_checks_since_scan < NAV_CHECKS_BETWEEN_SCANS:
                _nav_checks_since_scan += 1
                time.sleep(NAV_IMAGE_INTERVAL)  # move for interval, then check

                if not mission_active or mission_state == State.INTERACTING:
                    continue

                nav = _nav_check()

                if nav.get("wall_ahead") or nav.get("obstacle_close"):
                    motors.stop()
                    _cancel_nav2()
                    _ui("log", f"Nav: obstacle — {nav.get('physical_reasoning','')}")
                    force_360 = _avoid_obstacle(
                        wall_ahead=nav.get("wall_ahead", False),
                        small_obstacle=nav.get("small_obstacle", False)
                    )
                    if force_360:
                        _scans_since_360 = SCANS_BEFORE_360
                        _nav_checks_since_scan = NAV_CHECKS_BETWEEN_SCANS
                    else:
                        _move_forward()
                elif nav.get("action") == "slow":
                    motors.slow()
                elif nav.get("action") == "stop":
                    motors.stop()
                    _cancel_nav2()
                else:
                    if not (USE_NAV2 and _nav2_active()):
                        motors.forward(MOTOR_SPEED_SLOW)
                continue

            # ── Stopped scan ──────────────────────────────────────────────
            _nav_checks_since_scan = 0
            _scans_since_360 += 1
            motors.stop()
            _cancel_nav2()
            time.sleep(0.3)

            do_360 = (_empty_scans >= EMPTY_SCAN_LIMIT or
                      _scans_since_360 >= SCANS_BEFORE_360)

            if do_360:
                if _empty_scans >= EMPTY_SCAN_LIMIT:
                    eric_say("Nothing found. Performing a full 360 scan.")
                else:
                    _ui("log", "Periodic 360 scan...")
                scan = _scan_360()
                _scans_since_360 = _empty_scans = 0
                _process_scan(scan, from_360=True)
                if mission_active and mission_state == State.SEARCHING:
                    _move_forward()
            else:
                _ui("log", "Quick stopped scan...")
                motors.oled(1, "Scanning...")
                # Use scan_and_identify for stopped quick scan too
                result = scan_and_identify(adaptive_led=True)
                if result.get("confirmed_target"):
                    scan = _confirmed_to_scan_result(result, 0)
                else:
                    # Build minimal scan dict for _process_scan
                    scan = {
                        **dict(_SCAN_FALLBACK),
                        "wall_ahead":     result.get("wall_ahead", False),
                        "obstacle_close": result.get("obstacle_close", False),
                        "target_visible": result.get("potential_target", False),
                        "physical_reasoning": result.get("description", ""),
                    }
                _process_scan(scan, from_360=False)

            time.sleep(0.3)

        except Exception as e:
            log.error(f"Mission loop error: {e}")
            time.sleep(1)

    motors.stop()
    _cancel_nav2()
    mission_state = State.IDLE
    _ui("status", "IDLE")
    log.info("Mission loop ended")


def _nav2_active() -> bool:
    """True if Nav2 is currently executing a goal."""
    try:
        from nav2 import is_navigating
        return is_navigating()
    except Exception:
        return False