"""
E.R.I.C. — Mission Logic
Search and rescue mission state machine
"""

import time
import asyncio
import threading
import logging

from config import SCAN_INTERVAL, MOTOR_SPEED_SLOW
from motors import motors
from cosmos import ask_cosmos, scan_scene, set_mission_briefing
from tts import speak, speak_streaming

log = logging.getLogger("eric.mission")


class State:
    IDLE       = "idle"
    BRIEFING   = "briefing"    # waiting for mission briefing
    SEARCHING  = "searching"   # moving, scanning for targets
    INTERACTING = "interacting" # stopped, talking to character
    COMPLETE   = "complete"


# ─── Mission State ────────────────────────────────────────────────────────────

mission_state    = State.IDLE
mission_active   = False
mission_briefing = ""
conversation_history = []  # Eric remembers what he was told during mission

# Callbacks to update Gradio UI
_ui_callbacks = {
    "eric_says":    None,   # fn(text)
    "status":       None,   # fn(text)
    "log":          None,   # fn(text)
}


def register_ui_callbacks(**callbacks):
    _ui_callbacks.update(callbacks)


def _ui_update(key: str, text: str):
    cb = _ui_callbacks.get(key)
    if cb:
        try:
            cb(text)
        except Exception:
            pass


def eric_say(text: str):
    """Speak and update UI simultaneously."""
    _ui_update("eric_says", text)
    speak(text)


# ─── Character Interaction ────────────────────────────────────────────────────

def handle_character_response(character_name: str, user_input: str) -> str:
    """
    User typed as the character. Eric responds based on mission context
    and conversation history. Returns Eric's response.
    """
    global conversation_history

    # Add to history
    conversation_history.append({
        "character": character_name,
        "said": user_input,
        "time": time.time()
    })

    history_text = "\n".join([
        f"- {e['character']} told me: {e['said']}"
        for e in conversation_history[-5:]  # last 5 interactions
    ])

    prompt = f"""
I am currently talking to {character_name}.
They just said: "{user_input}"

Information gathered so far in this mission:
{history_text if history_text else "Nothing yet."}

Based on the mission briefing and what I've been told:
1) What does this information mean for my mission?
2) What do I say back to {character_name}?
3) What is my next action after this conversation?

Respond in 2-3 sentences as Eric speaking naturally. 
Be decisive. If this person has useful info, acknowledge it clearly.
If not, thank them politely and indicate you'll move on.
"""
    response = ask_cosmos(prompt, max_tokens=200)
    eric_say(response)
    _ui_update("log", f"[{character_name}]: {user_input}\n[Eric]: {response}")
    return response


# ─── Mission Loop ─────────────────────────────────────────────────────────────

def start_mission(briefing: str):
    """Start mission with a briefing prompt."""
    global mission_active, mission_state, mission_briefing, conversation_history

    if mission_active:
        return "Mission already active."

    mission_briefing     = briefing
    conversation_history = []
    set_mission_briefing(briefing)

    # Eric acknowledges the briefing and plans
    plan_prompt = f"""
You just received this mission briefing:
"{briefing}"

Acknowledge the briefing in 2-3 sentences.
Then state your immediate first action.
Be concise, mission-focused.
"""
    response = ask_cosmos(plan_prompt, max_tokens=150)
    eric_say(response)

    mission_active = True
    mission_state  = State.SEARCHING
    _ui_update("status", f"🟢 MISSION ACTIVE — {State.SEARCHING}")

    threading.Thread(target=_mission_loop, daemon=True).start()
    return response


def stop_mission():
    """Stop mission immediately."""
    global mission_active, mission_state
    mission_active = False
    mission_state  = State.IDLE
    motors.stop()
    motors.oled(0, "ERIC STOPPED")
    motors.oled(1, "")
    eric_say("Mission disengaged. All systems halted.")
    _ui_update("status", "🔴 IDLE")


def _mission_loop():
    """Autonomous scan-decide-act loop."""
    global mission_active, mission_state

    motors.oled(0, "ERIC ACTIVE")
    motors.oled(1, "Searching...")

    while mission_active:
        try:
            scan = scan_scene()
            _process_scan(scan)
            time.sleep(SCAN_INTERVAL)
        except Exception as e:
            log.error(f"Mission loop error: {e}")
            time.sleep(1)

    motors.stop()
    mission_state = State.IDLE
    _ui_update("status", "🔴 IDLE")
    log.info("Mission loop ended")


def _process_scan(scan: dict):
    """Process scene scan and act."""
    global mission_state

    obj       = scan.get("object", "unknown")
    obj_name  = scan.get("object_name")
    terrain   = scan.get("terrain", "clear")
    in_path   = scan.get("in_my_path", False)
    action    = scan.get("action", "forward")
    speak_txt = scan.get("speak")
    reasoning = scan.get("physical_reasoning", "")

    if reasoning:
        log.info(f"💭 {reasoning}")
        _ui_update("log", f"💭 {reasoning}")

    if speak_txt:
        eric_say(speak_txt)

    # Terrain
    if terrain == "pebbles":
        motors.slow()
    elif terrain in ["pavement", "clear"] and action == "forward":
        motors.forward()

    # Something in path — stop and wait for user input
    if in_path and obj in ["person", "robot"] and obj_name:
        motors.stop()
        mission_state = State.INTERACTING
        display_name  = obj_name or obj
        motors.oled(0, display_name[:16])
        motors.oled(1, "Talking...")

        greeting = ask_cosmos(
            f"You stopped because you see {display_name} directly in your path. "
            f"Greet them and ask if they have any information relevant to your mission. "
            f"Keep it to 1-2 sentences.",
            max_tokens=100
        )
        eric_say(greeting)
        _ui_update("status", f"💬 INTERACTING — {display_name}")
        # Mission loop pauses here — user inputs character response via UI
        # handle_character_response() is called from UI
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
    motors.oled(0, "ERIC ACTIVE")
    motors.oled(1, "Searching...")
    _ui_update("status", f"🟢 {State.SEARCHING.upper()}")


def resume_after_interaction():
    """Resume mission movement after a character interaction."""
    global mission_state
    if mission_active:
        mission_state = State.SEARCHING
        motors.forward()
        _ui_update("status", f"🟢 {State.SEARCHING.upper()}")
