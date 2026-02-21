"""
ERIC — Mission Logic
360-degree body rotation scan, dual camera, wall avoidance, mission complete
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
    capture_frame, center_on_person, pantilt, pantilt_center,
    autofocus_trigger, autofocus_enable, CAMERA_WEBCAM, CAMERA_PANTILT
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
EMPTY_SCAN_LIMIT   = 2
SCANS_BEFORE_360   = 4
MAX_AVOID_ATTEMPTS = 4

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


def _parse_json(response, fallback):
    try:
        clean = response.replace("```json", "").replace("```", "").strip()
        s = clean.find("{")
        e = clean.rfind("}") + 1
        if s >= 0 and e > s:
            result = json.loads(clean[s:e])
            for k, v in fallback.items():
                result.setdefault(k, v)
            return result
    except Exception:
        log.warning(f"JSON parse failed: {response[:100]}")
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
        motors.forward(MOTOR_SPEED_SLOW)
        motors.oled(0, "ERIC ACTIVE")
        motors.oled(1, "Searching...")
        _ui("status", "SEARCHING")


# ─── 360° Scan ────────────────────────────────────────────────────────────────

SCAN_360_PROMPT = """
These images are from a full 360-degree scan of my surroundings.
I captured at 0, 90, 180, 270 degrees body rotation.
At each position: pan-tilt tilted up (far/mid) and down (floor/near).
Both webcam and pan-tilt camera used at each stop. Total: up to 16 images.

Analyze ALL images carefully for my mission target:
- Slippers are large soft footwear on the floor — look carefully at floor-level frames
- Report target_visible=true even if partially visible or uncertain
- Report which direction the target is from my current facing

Respond ONLY with valid JSON, no markdown:
{
  "object": "person|robot|slipper|shoe|obstacle|wall|clear|unknown",
  "object_name": "description or null",
  "terrain": "carpet|tiles|clear",
  "distance": "close|medium|far",
  "in_my_path": false,
  "wall_ahead": false,
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

QUICK_SCAN_PROMPT = """
Live camera frames from my webcam and pan-tilt camera while moving.
Fast obstacle and target check only.

Respond ONLY with valid JSON:
{
  "object": "person|robot|slipper|shoe|obstacle|wall|clear|unknown",
  "object_name": null,
  "terrain": "carpet|tiles|clear",
  "distance": "close|medium|far",
  "in_my_path": false,
  "wall_ahead": false,
  "small_obstacle": false,
  "target_visible": false,
  "target_direction": "front",
  "action": "forward|slow|navigate_around|stop",
  "speak": null,
  "physical_reasoning": "1 sentence",
  "mission_complete": false
}
"""

_SCAN_FALLBACK = {
    "object": "unknown", "object_name": None, "terrain": "clear",
    "distance": "far", "in_my_path": False, "wall_ahead": False,
    "small_obstacle": False, "target_visible": False,
    "target_direction": "unknown", "clearest_direction": "front",
    "action": "forward", "speak": None,
    "physical_reasoning": "", "mission_complete": False
}


def _scan_360():
    global mission_state
    mission_state = State.SCANNING_360
    _ui("status", "360 SCANNING")
    motors.oled(0, "360 Scan")
    log.info("Starting 360 scan")

    motors.stop()
    time.sleep(0.4)
    all_frames = []

    for pos in range(4):
        deg = pos * 90
        _ui("log", f"Scanning {deg}...")
        motors.oled(1, f"Scan {deg}deg")

        for tilt, label in [(-20, "up"), (15, "floor")]:
            pantilt(0, tilt, speed=40)
            time.sleep(0.8)
            f1 = capture_frame(CAMERA_WEBCAM,  640, 480)
            f2 = capture_frame(CAMERA_PANTILT, 640, 480)
            if f1: all_frames.append(f1)
            if f2: all_frames.append(f2)

        pantilt_center()
        time.sleep(0.3)

        if pos < 3:
            motors.right(MOTOR_SPEED_SLOW)
            time.sleep(1.5)   # tune for true 90deg on your floor
            motors.stop()
            time.sleep(0.4)

    log.info(f"360 scan done — {len(all_frames)} frames -> Cosmos")
    _ui("log", f"360 done — {len(all_frames)} frames -> Cosmos")
    motors.oled(1, "Analyzing...")

    try:
        response = _cosmos_frames(all_frames, SCAN_360_PROMPT, max_tokens=300, temp=0.2)
        log.info(f"360 result: {response[:200]}")
        return _parse_json(response, dict(_SCAN_FALLBACK))
    except Exception as e:
        log.error(f"360 Cosmos error: {e}")
        return dict(_SCAN_FALLBACK)


def _quick_scan():
    pantilt(0, 12, speed=80)
    time.sleep(0.3)
    f1 = capture_frame(CAMERA_WEBCAM,  640, 480)
    f2 = capture_frame(CAMERA_PANTILT, 640, 480)
    pantilt_center()

    frames = [f for f in [f1, f2] if f]
    if not frames:
        return dict(_SCAN_FALLBACK)

    try:
        response = _cosmos_frames(frames, QUICK_SCAN_PROMPT, max_tokens=200, temp=0.3)
        return _parse_json(response, dict(_SCAN_FALLBACK))
    except Exception as e:
        log.error(f"Quick scan error: {e}")
        return dict(_SCAN_FALLBACK)


def _face_direction(direction):
    if direction == "right":
        motors.right(MOTOR_SPEED_SLOW); time.sleep(1.5); motors.stop()
    elif direction == "back":
        motors.right(MOTOR_SPEED_SLOW); time.sleep(3.0); motors.stop()
    elif direction == "left":
        motors.left(MOTOR_SPEED_SLOW);  time.sleep(1.5); motors.stop()
    time.sleep(0.3)


# ─── Obstacle Avoidance ───────────────────────────────────────────────────────

def _avoid_obstacle(wall_ahead, small_obstacle):
    global _avoid_attempts, mission_state
    _avoid_attempts += 1
    mission_state = State.AVOIDING
    _ui("status", "AVOIDING")

    if wall_ahead:
        _ui("log", f"Wall — attempt {_avoid_attempts}")
        motors.oled(1, "Wall! Back up...")
        motors.stop(); time.sleep(0.3)
        motors.backward(MOTOR_SPEED_SLOW); time.sleep(1.2)
        motors.stop(); time.sleep(0.2)
        if _avoid_attempts % 2 == 1:
            motors.right(MOTOR_SPEED_SLOW); time.sleep(1.5)
        else:
            motors.left(MOTOR_SPEED_SLOW);  time.sleep(1.5)
        motors.stop(); time.sleep(0.3)
        if _avoid_attempts >= MAX_AVOID_ATTEMPTS:
            _avoid_attempts = 0
            eric_say("I keep hitting obstacles. Let me do a full scan.")
            return True  # trigger 360

    elif small_obstacle:
        _ui("log", "Small obstacle — stepping around")
        motors.oled(1, "Step around...")
        motors.stop(); time.sleep(0.2)
        motors.right(MOTOR_SPEED_SLOW); time.sleep(0.9)
        motors.stop(); time.sleep(0.2)
        motors.forward(MOTOR_SPEED_SLOW); time.sleep(1.0)
        motors.stop(); time.sleep(0.2)
        motors.left(MOTOR_SPEED_SLOW);  time.sleep(0.9)
        motors.stop()

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

    pantilt(0, 10); time.sleep(0.5)
    autofocus_trigger(CAMERA_PANTILT); time.sleep(1.5)
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


# ─── Mission Loop ─────────────────────────────────────────────────────────────

def _mission_loop():
    global mission_active, mission_state, _empty_scans, _scans_since_360, _avoid_attempts

    eric_say("Starting initial 360 degree scan of the area.")
    scan = _scan_360()
    _process_scan(scan, from_360=True)

    if mission_active and mission_state == State.SEARCHING:
        motors.forward(MOTOR_SPEED_SLOW)

    while mission_active:
        try:
            if mission_state in (State.INTERACTING, State.COMPLETE):
                time.sleep(0.5)
                continue

            _scans_since_360 += 1
            do_360 = _empty_scans >= EMPTY_SCAN_LIMIT or _scans_since_360 >= SCANS_BEFORE_360

            if do_360:
                motors.stop(); time.sleep(0.3)
                if _empty_scans >= EMPTY_SCAN_LIMIT:
                    eric_say("Nothing found. Performing a full 360 scan.")
                else:
                    _ui("log", "Periodic 360 scan...")
                scan = _scan_360()
                _scans_since_360 = _empty_scans = 0
                _process_scan(scan, from_360=True)
                if mission_active and mission_state == State.SEARCHING:
                    motors.forward(MOTOR_SPEED_SLOW)
            else:
                _ui("log", "Quick scan...")
                motors.oled(1, "Scanning...")
                scan = _quick_scan()
                _process_scan(scan, from_360=False)

            time.sleep(0.5)

        except Exception as e:
            log.error(f"Mission loop error: {e}")
            time.sleep(1)

    motors.stop()
    mission_state = State.IDLE
    _ui("status", "IDLE")
    log.info("Mission loop ended")


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

    if wall_ahead or (in_path and obj == "wall"):
        motors.stop()
        if speak_tx: eric_say(speak_tx)
        force_360 = _avoid_obstacle(wall_ahead=True, small_obstacle=False)
        if force_360:
            _scans_since_360 = SCANS_BEFORE_360
        else:
            motors.forward(MOTOR_SPEED_SLOW)
        mission_state = State.SEARCHING
        return

    if small_obs and not target_visible:
        _avoid_obstacle(wall_ahead=False, small_obstacle=True)
        motors.forward(MOTOR_SPEED_SLOW)

    if not wall_ahead and not small_obs:
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

        if not center_on_person():
            pantilt(0, -15); time.sleep(0.5)
        autofocus_trigger(CAMERA_PANTILT); time.sleep(1.0)

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