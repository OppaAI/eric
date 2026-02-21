"""
E.R.I.C. — Mission Logic
Loads missions from YAML files, runs autonomous search and rescue loop
"""

import time
import threading
import logging
from pathlib import Path

from config import SCAN_INTERVAL, MOTOR_SPEED_SLOW, MISSIONS_DIR
from motors import motors
from cosmos import ask_cosmos, scan_scene, set_mission_briefing, get_mission_briefing
from tts import speak

log = logging.getLogger("eric.mission")


class State:
    IDLE         = "idle"
    SEARCHING    = "searching"
    INTERACTING  = "interacting"
    COMPLETE     = "complete"


# ─── Mission State ────────────────────────────────────────────────────────────
mission_state        = State.IDLE
mission_active       = False
conversation_history = []   # what characters told Eric during the mission

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
    """Update UI and speak simultaneously."""
    _ui("eric_says", text)
    speak(text)


# ─── Mission File Loading ─────────────────────────────────────────────────────

def list_missions() -> list[str]:
    """Return list of available mission names (yaml filenames without extension)."""
    if not MISSIONS_DIR.exists():
        return []
    return [f.stem for f in sorted(MISSIONS_DIR.glob("*.yaml"))]


def load_mission_file(name: str) -> dict | None:
    """Load a mission YAML file by stem name. Returns dict or None."""
    try:
        import yaml
        path = MISSIONS_DIR / f"{name}.yaml"
        if not path.exists():
            log.warning(f"Mission file not found: {path}")
            return None
        with open(path) as f:
            data = yaml.safe_load(f)
        log.info(f"📂 Loaded mission: {data.get('name', name)}")
        return data
    except Exception as e:
        log.error(f"Failed to load mission {name}: {e}")
        return None


def get_briefing_from_file(name: str) -> str | None:
    """Convenience: load mission file and return the briefing text."""
    data = load_mission_file(name)
    if data:
        return data.get("briefing", "").strip()
    return None


# ─── Mission Control ──────────────────────────────────────────────────────────

def start_mission(briefing: str) -> str:
    """
    Start mission with a briefing string.
    Briefing can be typed manually or loaded from a YAML file.
    """
    global mission_active, mission_state, conversation_history

    if mission_active:
        return "⚠️ Mission already active. Disengage first."

    if not briefing.strip():
        return "⚠️ No mission briefing provided."

    conversation_history = []
    set_mission_briefing(briefing)

    # Eric acknowledges and states first action
    ack = ask_cosmos(
        f"You just received this mission briefing:\n\"{briefing}\"\n\n"
        "Acknowledge it in 2-3 sentences. State your immediate first action. "
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
    """Stop mission immediately."""
    global mission_active, mission_state
    mission_active = False
    mission_state  = State.IDLE
    motors.stop()
    motors.oled(0, "ERIC STOPPED")
    motors.oled(1, "")
    eric_say("Mission disengaged. All systems halted.")
    _ui("status", "🔴 IDLE")


def resume_after_interaction():
    """Resume movement after a character interaction."""
    global mission_state
    if mission_active:
        mission_state = State.SEARCHING
        motors.forward()
        motors.oled(0, "ERIC ACTIVE")
        motors.oled(1, "Searching...")
        _ui("status", f"🟢 {State.SEARCHING.upper()}")


# ─── Character Interaction ────────────────────────────────────────────────────

def handle_character_response(character: str, said: str) -> str:
    """
    User typed as a character. Eric reasons about it and responds.
    If off-topic, Eric politely takes his leave and resumes mission.
    """
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

    # Count how many exchanges with this character
    exchanges = sum(1 for e in conversation_history if e["character"] == character)

    response = ask_cosmos(
        f"I am talking to {character}. They just said: \"{said}\"\n\n"
        f"Information gathered so far:\n{history_text}\n\n"
        f"This is exchange #{exchanges} with {character}.\n\n"
        "Evaluate this response:\n"
        "1) Is what they said relevant to my mission?\n"
        "2) Have I already gotten all useful information from them?\n"
        "3) Are they going off-topic or being overly chatty?\n\n"
        "If they are off-topic OR this is exchange #3 or more with no new mission info:\n"
        "  → Politely apologize, thank them, say you must continue your mission, say goodbye.\n"
        "  → End your response with exactly: [MOVE_ON]\n\n"
        "If they have useful mission information:\n"
        "  → Respond naturally, ask a focused follow-up if needed.\n\n"
        "Respond in 2 sentences as Eric. Be warm but decisive.",
        max_tokens=150
    )

    # Check if Eric decided to move on
    should_move_on = "[MOVE_ON]" in response
    clean_response = response.replace("[MOVE_ON]", "").strip()

    eric_say(clean_response)
    _ui("log", f"[{character}]: {said}\n[Eric]: {clean_response}")

    if should_move_on:
        log.info(f"💨 Eric politely left conversation with {character}")
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
    global mission_state

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
        motors.stop()
        mission_state = State.INTERACTING
        display_name  = obj_name or obj

        motors.oled(0, display_name[:16])
        motors.oled(1, "Talking...")

        greeting = ask_cosmos(
            f"You see {display_name} directly ahead. "
            "Greet them briefly and ask if they have any information relevant to your mission. "
            "1-2 sentences only.",
            max_tokens=80
        )
        eric_say(greeting)
        _ui("status", f"💬 TALKING — {display_name}")
        return

    # Navigation
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
    _ui("status", f"🟢 {State.SEARCHING.upper()}")
