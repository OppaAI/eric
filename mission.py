"""
ERIC — Mission Logic
Loads missions from YAML files, runs autonomous search and rescue loop

Improvements:
- Dual camera scanning (webcam + pan-tilt)
- 10s video feed at 640x480 for better detection
- Wall and obstacle avoidance
- Small obstacle detection (slippers, shoes, cables)
- Pan-tilt centering on detected faces
- Autofocus on target when found
- Mission complete detection and announcement
"""

import time
import threading
import logging
from pathlib import Path

from config import SCAN_INTERVAL, MOTOR_SPEED_SLOW, MOTOR_SPEED_NORMAL, MISSIONS_DIR
from motors import motors
from cosmos import (
    ask_cosmos, scan_scene_dual, set_mission_briefing, get_mission_briefing,
    center_on_person, pantilt, pantilt_center, autofocus_trigger,
    CAMERA_WEBCAM, CAMERA_PANTILT
)
from tts import speak

log = logging.getLogger("eric.mission")


class State:
    IDLE         = "idle"
    SEARCHING    = "searching"
    INTERACTING  = "interacting"
    AVOIDING     = "avoiding"
    COMPLETE     = "complete"
    LOST         = "lost"


# ─── Mission State ────────────────────────────────────────────────────────────
mission_state        = State.IDLE
mission_active       = False
conversation_history = []

_empty_scans      = 0
_search_phase     = 0
_avoid_attempts   = 0
EMPTY_SCAN_LIMIT  = 3        # fewer scans before search pattern (video takes longer)
MAX_AVOID_ATTEMPTS = 4       # max consecutive avoidance moves before stopping to reassess

_ui_callbacks = {"eric_says": None, "status": None, "log": None}


def register_ui_callbacks(**cbs):
    _ui_callbacks.update(cbs)


def _ui(key: str, text: str):
    cb = _ui_callbacks.get(key)
    if cb:
        try:
            cb(text)
        except Exception:
            pass


def eric_say(text: str):
    _ui("eric_says", text)
    speak(text)


# ─── Mission File Loading ─────────────────────────────────────────────────────

def list_missions() -> list[str]:
    if not MISSIONS_DIR.exists():
        return []
    return [f.stem for f in sorted(MISSIONS_DIR.glob("*.yaml"))]


def load_mission_file(name: str) -> dict | None:
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


# ─── Mission Control ──────────────────────────────────────────────────────────

def start_mission(briefing: str) -> str:
    global mission_active, mission_state, conversation_history
    global _empty_scans, _search_phase, _avoid_attempts

    if mission_active:
        return "⚠️ Mission already active. Disengage first."
    if not briefing.strip():
        return "⚠️ No mission briefing provided."

    conversation_history = []
    _empty_scans         = 0
    _search_phase        = 0
    _avoid_attempts      = 0
    set_mission_briefing(briefing)

    # Enable autofocus on both cameras at start
    try:
        from cosmos import autofocus_enable
        autofocus_enable(CAMERA_WEBCAM)
        autofocus_enable(CAMERA_PANTILT)
    except Exception:
        pass

    # Center pan-tilt
    pantilt_center()

    ack = ask_cosmos(
        f"You just received this mission briefing:\n\"{briefing}\"\n\n"
        "Acknowledge in 2-3 sentences. State your immediate first action. "
        "Be concise and mission-focused.",
        max_tokens=150
    )
    eric_say(ack)

    mission_active = True
    mission_state  = State.SEARCHING
    _ui("status", f"🟢 {State.SEARCHING.upper()}")
    motors.oled(0, "ERIC ACTIVE")
    motors.oled(1, "Searching...")
    motors.lights(base=128, head=255)  # lights on for better camera visibility

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
    _ui("status", "🔴 IDLE")


def resume_after_interaction():
    global mission_state, _empty_scans, _search_phase, _avoid_attempts
    if mission_active:
        _empty_scans    = 0
        _search_phase   = 0
        _avoid_attempts = 0
        mission_state   = State.SEARCHING
        pantilt_center()
        motors.forward()
        motors.oled(0, "ERIC ACTIVE")
        motors.oled(1, "Searching...")
        _ui("status", f"🟢 {State.SEARCHING.upper()}")


# ─── Wall / Obstacle Avoidance ────────────────────────────────────────────────

def _avoid_obstacle(wall_ahead: bool, small_obstacle: bool):
    """
    Execute avoidance maneuver based on obstacle type.
    wall_ahead: large wall or furniture blocking path
    small_obstacle: small item on floor (slippers, shoes)
    """
    global _avoid_attempts, mission_state

    _avoid_attempts += 1
    mission_state = State.AVOIDING
    _ui("status", "⚠️ AVOIDING OBSTACLE")

    if wall_ahead:
        log.info(f"🧱 Wall detected — avoidance attempt {_avoid_attempts}")
        _ui("log", f"🧱 Wall ahead — avoiding (attempt {_avoid_attempts})")
        motors.oled(1, "Wall! Turning...")

        motors.stop()
        time.sleep(0.3)

        # Back up first
        motors.backward(MOTOR_SPEED_SLOW)
        time.sleep(1.0)
        motors.stop()
        time.sleep(0.2)

        # Alternate left/right turns to find clear path
        if _avoid_attempts % 2 == 1:
            motors.right(MOTOR_SPEED_SLOW)
            time.sleep(1.5)
        else:
            motors.left(MOTOR_SPEED_SLOW)
            time.sleep(1.5)
        motors.stop()
        time.sleep(0.3)

        if _avoid_attempts >= MAX_AVOID_ATTEMPTS:
            # Stuck — ask Cosmos what to do
            _avoid_attempts = 0
            eric_say("I appear to be stuck. Reassessing the environment.")
            _ui("log", "🔄 Too many avoidance attempts — reassessing")

    elif small_obstacle:
        log.info("👟 Small obstacle on floor — carefully navigating around")
        _ui("log", "👟 Small obstacle — navigating around")
        motors.oled(1, "Obstacle!")

        motors.stop()
        time.sleep(0.2)
        motors.right(MOTOR_SPEED_SLOW)
        time.sleep(0.8)
        motors.stop()
        time.sleep(0.2)
        motors.forward(MOTOR_SPEED_SLOW)
        time.sleep(0.8)
        motors.stop()
        time.sleep(0.2)
        motors.left(MOTOR_SPEED_SLOW)
        time.sleep(0.8)
        motors.stop()


# ─── Mission Complete ─────────────────────────────────────────────────────────

def _handle_mission_complete(obj_name: str):
    """Called when Cosmos determines the mission target has been found."""
    global mission_active, mission_state

    log.info(f"🎯 MISSION COMPLETE — Found: {obj_name}")
    mission_state = State.COMPLETE

    motors.stop()
    motors.oled(0, "MISSION DONE!")
    motors.oled(1, obj_name[:16] if obj_name else "Target found!")
    _ui("status", "🎯 MISSION COMPLETE")

    # Flash lights to celebrate
    for _ in range(3):
        motors.lights(255, 255)
        time.sleep(0.3)
        motors.lights(0, 0)
        time.sleep(0.3)
    motors.lights(128, 255)

    # Center pan-tilt on target and trigger autofocus
    log.info("🎯 Centering camera on target...")
    pantilt(0, -10)   # Tilt slightly down for ground-level targets
    time.sleep(0.5)
    autofocus_trigger(CAMERA_PANTILT)
    time.sleep(1.0)

    # Try to center on face if it's a person
    center_on_person()

    announcement = ask_cosmos(
        f"You have found your mission target: {obj_name or 'the target'}! "
        "The mission is complete. "
        "Make a triumphant but warm announcement in 2-3 sentences. "
        "Describe that you found them and that you're signaling for help.",
        max_tokens=120
    )
    eric_say(announcement)
    _ui("eric_says", announcement)
    _ui("log", f"🎯 MISSION COMPLETE: {announcement}")

    mission_active = False
    _ui("status", "🎯 MISSION COMPLETE — Awaiting next orders")
    motors.oled(0, "TARGET FOUND!")
    motors.oled(1, "Mission done!")


# ─── Search Pattern ───────────────────────────────────────────────────────────

def _execute_search_pattern():
    """Systematic search when nothing found."""
    global _search_phase, _empty_scans

    patterns = [
        ("Scanning right...",   lambda: (motors.stop(), pantilt(45, 0), time.sleep(2), pantilt_center())),
        ("Moving forward...",   lambda: (motors.forward(), time.sleep(2.0), motors.stop())),
        ("Scanning left...",    lambda: (motors.stop(), pantilt(-45, 0), time.sleep(2), pantilt_center())),
        ("Turning right...",    lambda: (motors.right(), time.sleep(1.2), motors.stop())),
        ("Moving forward...",   lambda: (motors.forward(), time.sleep(2.0), motors.stop())),
        ("Turning left...",     lambda: (motors.left(), time.sleep(2.4), motors.stop())),
        ("Moving forward...",   lambda: (motors.forward(), time.sleep(2.0), motors.stop())),
    ]

    phase         = _search_phase % len(patterns)
    label, action = patterns[phase]

    log.info(f"🔍 Search pattern phase {phase}: {label}")
    _ui("log",  f"🔍 {label}")
    motors.oled(1, label[:16])

    try:
        action()
    except Exception as e:
        log.error(f"Search pattern error: {e}")
        motors.stop()

    _search_phase += 1
    _empty_scans   = 0

    if _search_phase % len(patterns) == 0:
        _ask_cosmos_what_to_do()


def _ask_cosmos_what_to_do():
    response = ask_cosmos(
        "I have searched my current area systematically and cannot find my target. "
        "Based on my mission briefing, what should I do next? "
        "Should I continue forward, backtrack, or try a different direction? "
        "Respond in 2 sentences — be decisive.",
        max_tokens=100
    )
    eric_say(response)
    _ui("log", f"🧠 Eric decides: {response}")


# ─── Character Interaction ────────────────────────────────────────────────────

def handle_character_response(character: str, said: str) -> str:
    global conversation_history

    conversation_history.append({
        "character": character,
        "said":      said,
        "time":      time.time()
    })

    history_text = "\n".join(
        f"- {e['character']} told me: {e['said']}"
        for e in conversation_history[-5:]
    )

    exchanges = sum(1 for e in conversation_history if e["character"] == character)

    response = ask_cosmos(
        f"I am talking to {character}. They just said: \"{said}\"\n\n"
        f"Information gathered so far:\n{history_text}\n\n"
        f"This is exchange #{exchanges} with {character}.\n\n"
        "Evaluate:\n"
        "1) Is this relevant to my mission?\n"
        "2) Have I gotten all useful info from them?\n"
        "3) Are they going off-topic or being overly chatty?\n\n"
        "If off-topic OR exchange #3+ with no new mission info:\n"
        "  → Politely apologize, thank them, say you must continue. End with: [MOVE_ON]\n\n"
        "If useful info:\n"
        "  → Respond naturally, ask focused follow-up if needed.\n\n"
        "2 sentences max. Be warm but decisive.",
        max_tokens=150
    )

    should_move_on = "[MOVE_ON]" in response
    clean_response = response.replace("[MOVE_ON]", "").strip()

    eric_say(clean_response)
    _ui("log", f"[{character}]: {said}\n[Eric]: {clean_response}")

    if should_move_on:
        log.info(f"💨 Eric politely left {character}")
        _ui("log", f"💨 Eric moved on from {character}")
        resume_after_interaction()

    return clean_response


# ─── Mission Loop ─────────────────────────────────────────────────────────────

def _mission_loop():
    global mission_active, mission_state

    # Start moving immediately
    motors.forward(MOTOR_SPEED_SLOW)

    while mission_active:
        try:
            if mission_state in (State.INTERACTING, State.COMPLETE):
                time.sleep(0.5)
                continue

            # Dual camera scan with 10s video
            _ui("log", "👁️ Scanning with both cameras...")
            motors.oled(1, "Scanning...")
            scan = scan_scene_dual(use_video=True, video_duration=10.0)
            _process_scan(scan)

            # Small delay between scans (video already takes ~10s)
            time.sleep(1.0)

        except Exception as e:
            log.error(f"Mission loop error: {e}")
            time.sleep(1)

    motors.stop()
    mission_state = State.IDLE
    _ui("status", "🔴 IDLE")
    log.info("Mission loop ended")


def _process_scan(scan: dict):
    global mission_state, _empty_scans, _avoid_attempts

    obj         = scan.get("object", "unknown")
    obj_name    = scan.get("object_name")
    terrain     = scan.get("terrain", "clear")
    in_path     = scan.get("in_my_path", False)
    wall_ahead  = scan.get("wall_ahead", False)
    small_obs   = scan.get("small_obstacle", False)
    action      = scan.get("action", "forward")
    speak_tx    = scan.get("speak")
    reason      = scan.get("physical_reasoning", "")
    distance    = scan.get("distance", "far")
    complete    = scan.get("mission_complete", False)

    if reason:
        log.info(f"💭 {reason}")
        _ui("log", f"💭 {reason}")

    # ── Mission complete? ──────────────────────────────────────────────────────
    if complete:
        _handle_mission_complete(obj_name)
        return

    # ── Wall / obstacle avoidance (highest priority) ───────────────────────────
    if wall_ahead or (in_path and obj == "wall"):
        if speak_tx:
            eric_say(speak_tx)
        _avoid_obstacle(wall_ahead=True, small_obstacle=False)
        mission_state = State.SEARCHING
        return

    if small_obs:
        _avoid_obstacle(wall_ahead=False, small_obstacle=True)
        # Continue scanning after avoidance

    # Reset avoidance counter when path is clear
    if not wall_ahead and not small_obs:
        _avoid_attempts = 0

    if speak_tx:
        eric_say(speak_tx)

    # ── Person or robot found ──────────────────────────────────────────────────
    if in_path and obj in ["person", "robot"]:
        _empty_scans  = 0
        _search_phase = 0
        motors.stop()
        mission_state = State.INTERACTING
        display_name  = obj_name or obj

        motors.oled(0, display_name[:16])
        motors.oled(1, "Centering...")
        _ui("status", f"👤 FOUND — {display_name}")
        _ui("log", f"👤 Found: {display_name} ({distance})")

        # Center pan-tilt on face and autofocus
        log.info(f"🎯 Found {display_name} — centering camera...")
        centered = center_on_person()
        if not centered:
            # Fall back to fixed tilt toward face height
            pantilt(0, -15)
            time.sleep(0.5)
        autofocus_trigger(CAMERA_PANTILT)
        time.sleep(1.0)

        motors.oled(1, "Talking...")
        greeting = ask_cosmos(
            f"You see {display_name} directly ahead. "
            "Greet them and ask if they have information relevant to your mission. "
            "1-2 sentences only.",
            max_tokens=80
        )
        eric_say(greeting)
        _ui("status", f"💬 TALKING — {display_name}")
        return

    # ── Nothing found ──────────────────────────────────────────────────────────
    if obj in ["clear", "unknown"]:
        _empty_scans += 1
        log.info(f"🔍 Empty scan {_empty_scans}/{EMPTY_SCAN_LIMIT}")
        _ui("log",  f"🔍 Nothing found ({_empty_scans}/{EMPTY_SCAN_LIMIT})")

        if _empty_scans >= EMPTY_SCAN_LIMIT:
            mission_state = State.LOST
            _ui("status", "🔍 SEARCHING — executing search pattern")
            motors.oled(1, "Searching...")

            if _empty_scans == EMPTY_SCAN_LIMIT:
                eric_say(
                    "I can't find my target in this area. "
                    "Executing systematic search pattern."
                )

            _execute_search_pattern()
            return

    # ── Normal navigation ──────────────────────────────────────────────────────
    if action == "navigate_around":
        motors.left(MOTOR_SPEED_SLOW)
        time.sleep(0.8)
        motors.forward()
    elif action == "slow" or terrain == "pebbles":
        motors.slow()
    elif action == "stop":
        motors.stop()
    else:
        motors.forward()

    mission_state = State.SEARCHING
    motors.oled(0, "ERIC ACTIVE")
    motors.oled(1, "Searching...")
    _ui("status", f"🟢 {State.SEARCHING.upper()}")