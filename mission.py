"""
E.R.I.C. — Mission Logic
Loads missions from YAML files, runs autonomous search and rescue loop
"""

import time
import threading
import logging
from pathlib import Path

from config import SCAN_INTERVAL, MOTOR_SPEED_SLOW, MOTOR_SPEED_NORMAL, MISSIONS_DIR
from motors import motors
from cosmos import ask_cosmos, scan_scene, set_mission_briefing, get_mission_briefing
from tts import speak

log = logging.getLogger("eric.mission")


class State:
    IDLE         = "idle"
    SEARCHING    = "searching"    # moving forward, scanning
    INTERACTING  = "interacting"  # stopped, talking to character
    LOST         = "lost"         # nothing found, executing search pattern
    COMPLETE     = "complete"


# ─── Mission State ────────────────────────────────────────────────────────────
mission_state        = State.IDLE
mission_active       = False
conversation_history = []

# Search pattern tracking
_empty_scans      = 0          # consecutive scans with nothing found
_search_phase     = 0          # which phase of search pattern we're in
EMPTY_SCAN_LIMIT  = 4          # scans before triggering search pattern

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
    global _empty_scans, _search_phase

    if mission_active:
        return "⚠️ Mission already active. Disengage first."
    if not briefing.strip():
        return "⚠️ No mission briefing provided."

    conversation_history = []
    _empty_scans         = 0
    _search_phase        = 0
    set_mission_briefing(briefing)

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

    threading.Thread(target=_mission_loop, daemon=True).start()
    return ack


def stop_mission():
    global mission_active, mission_state
    mission_active = False
    mission_state  = State.IDLE
    motors.stop()
    motors.oled(0, "ERIC STOPPED")
    motors.oled(1, "")
    eric_say("Mission disengaged. All systems halted.")
    _ui("status", "🔴 IDLE")


def resume_after_interaction():
    global mission_state, _empty_scans, _search_phase
    if mission_active:
        # Reset search counters — we found someone, restart search fresh
        _empty_scans  = 0
        _search_phase = 0
        mission_state = State.SEARCHING
        motors.forward()
        motors.oled(0, "ERIC ACTIVE")
        motors.oled(1, "Searching...")
        _ui("status", f"🟢 {State.SEARCHING.upper()}")


# ─── Search Pattern ───────────────────────────────────────────────────────────

def _execute_search_pattern():
    """
    Systematic search when nothing found after EMPTY_SCAN_LIMIT scans.
    Cycles through: turn right → forward → turn left → forward → turn back → forward
    Each phase gives a new vantage point before Cosmos scans again.
    """
    global _search_phase, _empty_scans

    patterns = [
        ("Scanning right...",    lambda: (motors.right(), time.sleep(1.2), motors.stop())),
        ("Scanning forward...",  lambda: (motors.forward(), time.sleep(1.5), motors.stop())),
        ("Scanning left...",     lambda: (motors.left(), time.sleep(1.2), motors.stop())),
        ("Scanning forward...",  lambda: (motors.forward(), time.sleep(1.5), motors.stop())),
        ("Turning back...",      lambda: (motors.right(), time.sleep(2.4), motors.stop())),
        ("Scanning forward...",  lambda: (motors.forward(), time.sleep(2.0), motors.stop())),
    ]

    phase        = _search_phase % len(patterns)
    label, action = patterns[phase]

    log.info(f"🔍 Search pattern phase {phase}: {label}")
    _ui("log", f"🔍 {label}")
    motors.oled(1, label[:16])
    action()

    _search_phase += 1
    _empty_scans   = 0  # reset after each search move — give Cosmos fresh chance

    # After full cycle (6 phases), ask Cosmos what to do
    if _search_phase % len(patterns) == 0:
        _ask_cosmos_what_to_do()


def _ask_cosmos_what_to_do():
    """After exhausting search pattern, ask Cosmos for reasoning on next step."""
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

    while mission_active:
        try:
            if mission_state == State.INTERACTING:
                time.sleep(0.5)
                continue

            scan = scan_scene()
            _process_scan(scan)
            time.sleep(SCAN_INTERVAL)

        except Exception as e:
            log.error(f"Mission loop error: {e}")
            time.sleep(1)

    motors.stop()
    mission_state = State.IDLE
    _ui("status", "🔴 IDLE")
    log.info("Mission loop ended")


def _process_scan(scan: dict):
    global mission_state, _empty_scans

    obj      = scan.get("object", "unknown")
    obj_name = scan.get("object_name")
    terrain  = scan.get("terrain", "clear")
    in_path  = scan.get("in_my_path", False)
    action   = scan.get("action", "forward")
    speak_tx = scan.get("speak")
    reason   = scan.get("physical_reasoning", "")

    if reason:
        log.info(f"💭 {reason}")
        _ui("log", f"💭 {reason}")

    if speak_tx:
        eric_say(speak_tx)

    # Terrain adjustment
    if terrain == "pebbles":
        motors.slow()
    elif terrain in ["pavement", "clear"] and action == "forward":
        motors.forward()

    # Person or robot in path — stop and interact
    if in_path and obj in ["person", "robot"]:
        _empty_scans  = 0   # found something — reset counter
        _search_phase = 0
        motors.stop()
        mission_state = State.INTERACTING
        display_name  = obj_name or obj

        motors.oled(0, display_name[:16])
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

    # Nothing useful found in this scan
    if obj in ["clear", "unknown"]:
        _empty_scans += 1
        log.info(f"🔍 Empty scan {_empty_scans}/{EMPTY_SCAN_LIMIT}")
        _ui("log",  f"🔍 Nothing found ({_empty_scans}/{EMPTY_SCAN_LIMIT})")

        if _empty_scans >= EMPTY_SCAN_LIMIT:
            # Trigger search pattern
            mission_state = State.LOST
            _ui("status", "🔍 SEARCHING — executing search pattern")
            motors.oled(1, "Searching...")

            if _empty_scans == EMPTY_SCAN_LIMIT:
                # First time hitting limit — announce it
                eric_say(
                    "I can't find my target in this area. "
                    "Executing systematic search pattern."
                )

            _execute_search_pattern()
            return

    # Normal navigation
    if action == "navigate_around":
        motors.left(MOTOR_SPEED_SLOW)
        time.sleep(0.8)
        motors.forward()
    elif action == "slow":
        motors.slow()
    elif action == "stop":
        motors.stop()
    else:
        motors.forward()

    mission_state = State.SEARCHING
    motors.oled(0, "ERIC ACTIVE")
    motors.oled(1, "Searching...")
    _ui("status", f"🟢 {State.SEARCHING.upper()}")
