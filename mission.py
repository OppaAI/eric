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
    capture_frame, capture_dual_stable, capture_nav_frame, capture_nav_clip,
    center_on_person,
    pantilt, pantilt_center, pantilt_move_wait,
    autofocus_trigger, autofocus_enable,
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


def _parse_json(response, fallback, label="COSMOS"):
    try:
        clean = response.replace("```json", "").replace("```", "").strip()
        s = clean.find("{")
        e = clean.rfind("}") + 1
        if s >= 0 and e > s:
            result = json.loads(clean[s:e])
            for k, v in fallback.items():
                result.setdefault(k, v)
            # ── Print full result to terminal ──────────────────────────────
            print(f"\n{'─'*60}")
            print(f"🧠 {label}:")
            for k, v in result.items():
                # Highlight important fields
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


# ─── Prompts ──────────────────────────────────────────────────────────────────

NAV_PROMPT = """
These are frames from a 10-second video clip of my pan-tilt camera while I am moving.
Analyze the MOTION and what is approaching — egocentric, first person view.

Look for:
- Obstacles or walls getting closer over the frames
- People or robots appearing in the scene
- Terrain changes (carpet, tiles, steps)
- Any object moving into my path

Respond ONLY with valid JSON:
{
  "wall_ahead": false,
  "obstacle_close": false,
  "small_obstacle": false,
  "person_visible": false,
  "action": "forward|slow|stop|turn_left|turn_right",
  "physical_reasoning": "one sentence describing what you saw across the clip"
}
"""

SCAN_360_PROMPT = """
These images are from a full 360-degree scan of my surroundings.
I captured at 0, 90, 180, 270 degrees body rotation.
At each position: pan-tilt tilted up (far/mid) and down (floor/near).
Both pan-tilt and webcam cameras used at each stop (pan-tilt first, then webcam).
Total: up to 16 images.

Analyze ALL images carefully for my mission target:
- Slippers are large soft footwear on the floor — look carefully at floor-level frames
- Robots have cameras, lights, chassis — check face/camera area for centering
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
I am stopped. These are stable frames from my pan-tilt (wide angle) and webcam cameras.
Obstacle and mission target check.

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

_NAV_FALLBACK = {
    "wall_ahead": False, "obstacle_close": False, "small_obstacle": False,
    "action": "forward", "physical_reasoning": ""
}


# ─── Navigation Check (while moving) ─────────────────────────────────────────

# Nav clip settings — tune these
NAV_CLIP_DURATION = 10.0  # seconds of video per nav check
NAV_CLIP_FPS      = 2     # frames per second (10s x 2fps = 20 frames to Cosmos)


def _nav_check() -> dict:
    """
    Video nav check using pan-tilt camera.
    Captures a 10s clip while robot moves, sends all frames to Cosmos.
    Cosmos reasons about motion, approaching obstacles, and people over time.
    If person spotted, stop and greet before resuming.
    """
    _ui("log", f"🎬 Nav clip ({NAV_CLIP_DURATION}s)...")
    motors.oled(1, "Nav scan...")

    frames = capture_nav_clip(
        duration_sec=NAV_CLIP_DURATION,
        fps=NAV_CLIP_FPS,
        width=640,
        height=480
    )

    if not frames:
        log.warning("Nav clip: no frames captured — fallback")
        # Fallback to single frame
        frame = capture_nav_frame()
        if not frame:
            return dict(_NAV_FALLBACK)
        frames = [frame]

    try:
        print(f"\n🚗 NAV CLIP CHECK — {len(frames)} frames to Cosmos...")
        response = _cosmos_frames(frames, NAV_PROMPT, max_tokens=150, temp=0.2)
        result = _parse_json(response, dict(_NAV_FALLBACK), label="NAV CLIP RESULT")

        # Greet person/robot if spotted while moving
        if result.get("person_visible") and mission_active:
            motors.stop()
            _ui("log", "👤 Person spotted — greeting!")
            _ui("status", "GREETING")
            greeting = ask_cosmos(
                "You just spotted someone ahead while navigating. "
                "Give a warm friendly greeting and ask if they can help with your mission. "
                "1-2 sentences only.",
                max_tokens=60
            )
            eric_say(greeting)
            time.sleep(1.0)

        return result
    except Exception as e:
        log.error(f"Nav clip check error: {e}")
        return dict(_NAV_FALLBACK)


# ─── Quick Scan (stopped) ─────────────────────────────────────────────────────

def _quick_scan() -> dict:
    """
    Dual camera stable scan while stopped.
    Pan-tilt centers + settles, then both cameras captured.
    """
    frames = capture_dual_stable(adaptive_led=True)
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
        f = capture_frame(device, 640, 480, adaptive_led=True)
        if f is None:
            break
        if not _is_blurry(f):
            return f   # sharp enough
        log.info(f"Blurry frame on cam {device} (attempt {attempt+1}) — waiting and retrying...")
        best = f       # keep as fallback
        time.sleep(0.5)
    return best  # return best we got even if still blurry


def _scan_360() -> dict:
    """
    Full 360° body rotation scan.
    At each of 4 positions (0°, 90°, 180°, 270°):
      - tilt up (far/mid) and down (floor/near)
      - capture pan-tilt + webcam, retry if blurry
    """
    global mission_state
    mission_state = State.SCANNING_360
    _ui("status", "360 SCANNING")
    motors.oled(0, "360 Scan")
    log.info("Starting 360 scan")

    motors.stop()
    time.sleep(0.5)
    all_frames = []

    for pos in range(4):
        deg = pos * 90
        _ui("log", f"Scanning {deg}°...")
        motors.oled(1, f"Scan {deg}deg")

        # Tilt up (far/mid-range) then down (floor/near)
        for tilt, label in [(-20, "up"), (15, "floor")]:
            pantilt_move_wait(0, tilt, speed=40)  # includes settle wait

            # Pan-tilt frame — retry if blurry
            f_pt = _capture_sharp(CAMERA_PANTILT)
            if f_pt:
                all_frames.append(f_pt)
                log.info(f"  {deg}° tilt={label}: pan-tilt ✓ (sharp)")

            # Webcam frame — retry if blurry
            f_wc = _capture_sharp(CAMERA_WEBCAM)
            if f_wc:
                all_frames.append(f_wc)
                log.info(f"  {deg}° tilt={label}: webcam ✓ (sharp)")

        # Return to center before rotating body
        pantilt_center()

        if pos < 3:
            motors.right(MOTOR_SPEED_SLOW)
            time.sleep(TURN_90_SEC)   # tuned for true 90° — adjust TURN_90_SEC if off
            motors.stop()
            time.sleep(0.5)           # settle after body rotation

    log.info(f"360 scan done — {len(all_frames)} frames → Cosmos")
    _ui("log", f"360 done — {len(all_frames)} frames → Cosmos")
    motors.oled(1, "Analyzing...")

    try:
        print(f"\n🔄 360° SCAN — sending {len(all_frames)} frames to Cosmos...")
        response = _cosmos_frames(all_frames, SCAN_360_PROMPT, max_tokens=300, temp=0.2)
        return _parse_json(response, dict(_SCAN_FALLBACK), label="360° SCAN RESULT")
    except Exception as e:
        log.error(f"360 Cosmos error: {e}")
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
    """Returns True if 360 scan should be forced."""
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
        motors.left(MOTOR_SPEED_SLOW);   time.sleep(0.9)
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

    # Look slightly up, settle, then center on person or robot
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

        # Center on person or robot using pan-tilt (settled capture)
        if not center_on_person():
            pantilt_move_wait(0, -15)  # fallback: tilt up slightly
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
    scan = _scan_360()
    _process_scan(scan, from_360=True)

    if mission_active and mission_state == State.SEARCHING:
        motors.forward(MOTOR_SPEED_SLOW)

    _nav_clips_since_scan = 0

    while mission_active:
        try:
            if mission_state in (State.INTERACTING, State.COMPLETE):
                time.sleep(0.5)
                continue

            # ── Video nav check while moving (10s clip → Cosmos) ─────────────
            if _nav_clips_since_scan < NAV_CLIPS_BETWEEN_SCANS:
                _nav_clips_since_scan += 1
                nav = _nav_check()  # captures 10s video clip, robot keeps moving

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

            # ── Stopped scan every NAV_CLIPS_BETWEEN_SCANS clips ─────────────
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
                scan = _scan_360()
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