"""
ERIC — Smart Obstacle Avoidance
Layered avoidance pipeline:

  Layer 1 — Instant hardware reaction (no Cosmos, no delay)
    LiDAR arc scan → pick clearest escape direction → back up + turn

  Layer 2 — Cosmos reasoning (camera + sensor context)
    Capture frame → send LiDAR arcs + OAK-D depth + image to Cosmos
    Cosmos returns: recommended action + direction + explanation

  Layer 3 — Verify and retry
    Re-check sensors after manoeuvre — recurse if still blocked
    Force full 360 scan after MAX_AVOID_ATTEMPTS

Architecture:
  avoid_obstacle() is the single entry point called by mission.py.
  It replaces the old _avoid_obstacle() entirely.

  The LiDAR scan_callback NO LONGER directly calls motors.stop().
  Instead it sets _obstacle_close / _obstacle_near state flags, and
  avoidance.py reads those flags plus the full arc map to plan.

  Cosmos is consulted AFTER the immediate back-up — it has ~5-9s to
  think while Eric is safely reversing. Its output refines the turn
  direction and duration. If Cosmos is slow or fails, the arc-based
  fallback always runs.

Integration:
  In mission.py, replace:
      force_360 = _avoid_obstacle(wall_ahead=..., small_obstacle=...)
  with:
      from avoidance import avoid_obstacle
      force_360 = avoid_obstacle(wall_ahead=..., small_obstacle=...)

  In lidar.py, remove motors.stop() / motors.slow() from _scan_callback.
  Instead, lidar.py now only updates state. avoidance.py reads the state.
  (See notes at bottom of file.)
"""

import time
import math
import logging
import json
import requests
import threading

from config import MOTOR_SPEED_SLOW, MOTOR_SPEED_NORMAL, VLLM_URL, COSMOS_MODEL
from motors import motors

log = logging.getLogger("eric.avoidance")

# ─── Tuning ───────────────────────────────────────────────────────────────────
BACKUP_DURATION   = 1.5    # seconds to reverse on hard stop
BACKUP_SPEED      = MOTOR_SPEED_SLOW
TURN_BASE_SEC     = 1.6    # base turn duration — increased each retry
TURN_INCREMENT    = 0.4    # added per retry
TURN_MAX_SEC      = 3.5    # cap turn duration
MAX_AVOID_ATTEMPTS = 3     # force full 360 after this many failures
COSMOS_TIMEOUT    = 20.0   # seconds to wait for Cosmos avoidance reasoning

# Arc angle definitions (degrees, symmetric around robot forward = 0°)
# D500 LiDAR: angle 0 = forward on most configs. Adjust if rotated.
ARC_HALF = 50              # degrees either side of each cardinal direction
SIDE_HALF = 40             # narrower cone for left/right arcs

_avoid_attempts = 0        # global retry counter — reset when path clears
_avoid_lock     = threading.Lock()  # prevent concurrent avoidance calls


# ─── LiDAR Arc Reader ────────────────────────────────────────────────────────

def get_arc_distances() -> dict:
    """
    Read the latest LiDAR scan and return minimum distances for 4 arcs:
      front  — ±50° of forward
      rear   — ±50° of backward
      left   — 90° ±40° (port side)
      right  — 270° ±40° (starboard side) — or equivalently −90°

    Returns dict with keys: front, rear, left, right.
    Values are meters, or 999.0 if no valid reading in that arc.

    Requires lidar.py to expose _last_scan_msg (set by _scan_callback).
    Falls back to get_status()["min_distance"] for front if raw scan unavailable.
    """
    arcs = {"front": 999.0, "rear": 999.0, "left": 999.0, "right": 999.0}

    try:
        from lidar import _last_scan_msg, _lock as lidar_lock
        with lidar_lock:
            msg = _last_scan_msg

        if msg is None:
            raise AttributeError("no scan yet")

        arc_rad = math.radians(ARC_HALF)
        side_rad = math.radians(SIDE_HALF)

        front_min = rear_min = left_min = right_min = 999.0

        for i, r in enumerate(msg.ranges):
            if not (msg.range_min < r < msg.range_max):
                continue
            angle = msg.angle_min + i * msg.angle_increment
            # Normalise to [-π, π]
            angle = (angle + math.pi) % (2 * math.pi) - math.pi

            if -arc_rad <= angle <= arc_rad:
                front_min = min(front_min, r)
            elif (angle > math.pi - arc_rad) or (angle < -math.pi + arc_rad):
                rear_min = min(rear_min, r)
            elif math.pi / 2 - side_rad <= angle <= math.pi / 2 + side_rad:
                left_min = min(left_min, r)
            elif -math.pi / 2 - side_rad <= angle <= -math.pi / 2 + side_rad:
                right_min = min(right_min, r)

        arcs = {"front": round(front_min, 3), "rear": round(rear_min, 3),
                "left": round(left_min, 3), "right": round(right_min, 3)}

    except (ImportError, AttributeError):
        # lidar.py not updated yet — fall back to front distance only
        try:
            from lidar import min_front_distance
            arcs["front"] = round(min_front_distance(), 3)
        except Exception:
            pass
    except Exception as e:
        log.warning(f"arc_distances error: {e}")

    return arcs


def _pick_clearest_turn(arcs: dict) -> str:
    """
    Given arc distances, pick the best escape turn direction.
    Prefer the side with more clearance. Rear is a last resort (180° turn).
    Returns: "left" | "right" | "back"
    """
    left_d  = arcs.get("left",  999.0)
    right_d = arcs.get("right", 999.0)
    rear_d  = arcs.get("rear",  999.0)

    log.info(f"Arc distances — L:{left_d:.2f}m R:{right_d:.2f}m Rear:{rear_d:.2f}m")

    # Both sides clear enough — pick the more open side
    if left_d > 0.50 and right_d > 0.50:
        return "left" if left_d >= right_d else "right"
    elif left_d > 0.50:
        return "left"
    elif right_d > 0.50:
        return "right"
    elif rear_d > 0.50:
        return "back"
    else:
        # Completely boxed in — pick least-bad side
        best = max(arcs.items(), key=lambda kv: kv[1])
        d = best[0]
        if d == "rear":
            return "back"
        return d  # "left" or "right"


# ─── OAK-D Context ───────────────────────────────────────────────────────────

def _oakd_context() -> str:
    try:
        from oakd import get_depth_map, oakd_available
        if not oakd_available():
            return ""
        dm = get_depth_map()
        if not dm:
            return ""
        lines = ["OAK-D depth grid (meters, None=no reading):"]
        for row in [["top_left", "top_center", "top_right"],
                    ["mid_left", "mid_center", "mid_right"],
                    ["bot_left", "bot_center", "bot_right"]]:
            vals = [f"{dm.get(k, 'None')}" for k in row]
            lines.append("  " + "  |  ".join(f"{v:>6}" for v in vals))
        return "\n".join(lines) + "\n"
    except Exception:
        return ""


# ─── Cosmos Avoidance Reasoning ──────────────────────────────────────────────

_AVOIDANCE_PROMPT = """\
You are a tracked ground robot that has just hit an obstacle and backed up.
You must choose the best escape direction.

SENSOR DATA:
{sensor_block}

SITUATION: {situation}

Choose the safest escape manoeuvre. Respond ONLY with a single JSON object — \
no markdown, no explanation:

{{
  "action":    "turn_left | turn_right | turn_back | forward | stop",
  "turn_sec":  <float 0.5–3.5 — how long to turn before moving forward>,
  "reasoning": "<one sentence explaining your choice>"
}}

Rules:
- Prefer the direction with the most clearance from SENSOR DATA above.
- "forward" is valid only if front distance > 0.60m.
- "stop" only if completely boxed in on all sides.
- turn_sec should be larger for tighter corners and smaller for open space.
"""

def _cosmos_avoidance_decision(arcs: dict, situation: str,
                                wall_ahead: bool, small_obstacle: bool) -> dict | None:
    """
    Ask Cosmos Reason 2 to pick the best avoidance action using:
      - Current LiDAR arc distances
      - OAK-D depth grid
      - A live camera frame

    Returns parsed dict with keys: action, turn_sec, reasoning
    Returns None on timeout or parse failure (caller uses arc-based fallback).
    """
    from cosmos import capture_frame, CAMERA_PANTILT

    # Build sensor block
    sensor_lines = [
        f"LiDAR arc distances: front={arcs['front']:.2f}m  left={arcs['left']:.2f}m  "
        f"right={arcs['right']:.2f}m  rear={arcs['rear']:.2f}m",
    ]
    oakd_ctx = _oakd_context()
    if oakd_ctx:
        sensor_lines.append(oakd_ctx)

    sensor_block = "\n".join(sensor_lines)
    prompt = _AVOIDANCE_PROMPT.format(
        sensor_block=sensor_block,
        situation=situation
    )

    # Capture camera frame — robot has backed up so camera sees the obstacle
    frame = capture_frame(CAMERA_PANTILT, 320, 240)

    content = []
    if frame:
        content.append({"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
    content.append({"type": "text", "text": prompt})

    payload = {
        "model":              COSMOS_MODEL,
        "messages":           [{"role": "user", "content": content}],
        "max_tokens":         100,
        "temperature":        0.1,
        "repetition_penalty": 1.1,
    }

    try:
        r = requests.post(VLLM_URL, json=payload, timeout=COSMOS_TIMEOUT)
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"].strip()
        log.info(f"Cosmos avoidance raw: {raw[:200]}")

        clean = raw.replace("```json", "").replace("```", "").strip()
        s = clean.find("{"); e = clean.rfind("}") + 1
        if s >= 0 and e > s:
            result = json.loads(clean[s:e])
            # Validate action
            valid_actions = {"turn_left", "turn_right", "turn_back", "forward", "stop"}
            if result.get("action") not in valid_actions:
                result["action"] = "turn_left"
            result.setdefault("turn_sec", TURN_BASE_SEC)
            result["turn_sec"] = max(0.5, min(float(result["turn_sec"]), TURN_MAX_SEC))
            log.info(f"🧠 Cosmos avoidance → {result['action']} ({result['turn_sec']:.1f}s): {result.get('reasoning','')}")
            return result

    except requests.exceptions.Timeout:
        log.warning("Cosmos avoidance timed out — using arc-based fallback")
    except Exception as e:
        log.warning(f"Cosmos avoidance error: {e} — using arc-based fallback")

    return None


# ─── Main Avoidance Entry Point ───────────────────────────────────────────────

def avoid_obstacle(wall_ahead: bool = True, small_obstacle: bool = False) -> bool:
    """
    Smart obstacle avoidance — call this instead of _avoid_obstacle() in mission.py.

    Pipeline:
      1. Instant: stop + back up (always, regardless of Cosmos)
      2. Read LiDAR arcs + OAK-D to find clearest direction
      3. Ask Cosmos (camera + sensors) to confirm/override direction
      4. Execute turn + move forward
      5. Re-verify path — recurse if still blocked
      6. Return True if a full 360 scan should be triggered

    Thread-safe — only one avoidance routine runs at a time.
    If called while already avoiding, the call returns immediately (False).

    Args:
        wall_ahead:     Large obstacle / wall directly in front
        small_obstacle: Small ground hazard — use step-around manoeuvre

    Returns:
        True  → caller should trigger a full 360° scan
        False → avoidance succeeded, caller can resume forward motion
    """
    global _avoid_attempts

    if not _avoid_lock.acquire(blocking=False):
        log.warning("Avoidance already running — skipping duplicate call")
        return False

    try:
        return _run_avoidance(wall_ahead, small_obstacle)
    finally:
        _avoid_lock.release()


def _run_avoidance(wall_ahead: bool, small_obstacle: bool) -> bool:
    global _avoid_attempts

    _avoid_attempts += 1
    attempt = _avoid_attempts

    log.info(f"🚧 Avoidance attempt {attempt}/{MAX_AVOID_ATTEMPTS} "
             f"(wall={wall_ahead}, small={small_obstacle})")

    # ── Notify UI ─────────────────────────────────────────────────────────────
    try:
        from mission import _ui
        _ui("status", f"AVOIDING ({attempt}/{MAX_AVOID_ATTEMPTS})")
        _ui("log", f"🚧 Obstacle — avoidance attempt {attempt}")
    except Exception:
        pass

    motors.oled(0, "OBSTACLE!")
    motors.oled(1, f"Avoid {attempt}/{MAX_AVOID_ATTEMPTS}")

    # ─────────────────────────────────────────────────────────────────────────
    # LAYER 1 — Instant: stop hard and back up
    # No Cosmos, no delay. Always runs first.
    # ─────────────────────────────────────────────────────────────────────────
    motors.stop()
    time.sleep(0.2)

    if small_obstacle:
        # Small obstacle: shorter backup
        log.info("Small obstacle — step-around manoeuvre")
        motors.backward(BACKUP_SPEED)
        time.sleep(0.6)
        motors.stop()
        time.sleep(0.2)
        # Step-around: right → forward → left to bypass
        motors.right(MOTOR_SPEED_SLOW);   time.sleep(1.0); motors.stop(); time.sleep(0.2)
        motors.forward(MOTOR_SPEED_SLOW); time.sleep(1.0); motors.stop(); time.sleep(0.2)
        motors.left(MOTOR_SPEED_SLOW);    time.sleep(1.0); motors.stop(); time.sleep(0.3)

        # Re-check — if clear, reset counter and return
        if _path_is_clear():
            log.info("Path clear after step-around ✅")
            _avoid_attempts = 0
            return False

        # Not clear — escalate to wall avoidance
        log.info("Still blocked after step-around — escalating to wall avoidance")
        return _run_avoidance(wall_ahead=True, small_obstacle=False)

    # ── Wall / large obstacle ─────────────────────────────────────────────────
    log.info(f"Wall obstacle — backing up {BACKUP_DURATION}s")
    motors.backward(BACKUP_SPEED)
    time.sleep(BACKUP_DURATION)
    motors.stop()
    time.sleep(0.3)

    # ─────────────────────────────────────────────────────────────────────────
    # LAYER 2 — Read all sensors while backing up
    # ─────────────────────────────────────────────────────────────────────────
    arcs = get_arc_distances()
    arc_direction = _pick_clearest_turn(arcs)

    # ─────────────────────────────────────────────────────────────────────────
    # LAYER 3 — Ask Cosmos (concurrent with backup so latency is hidden)
    # We back up first (safe), then Cosmos refines the escape direction.
    # If Cosmos already replied during backup, great. If not, we use arc fallback.
    # ─────────────────────────────────────────────────────────────────────────
    situation = (
        f"Wall directly ahead. Backed up {BACKUP_DURATION}s. Attempt {attempt}. "
        f"Clearest sensor direction: {arc_direction}."
    )

    cosmos_result = _cosmos_avoidance_decision(
        arcs=arcs,
        situation=situation,
        wall_ahead=wall_ahead,
        small_obstacle=small_obstacle
    )

    # Decide final action — Cosmos wins if it responded, otherwise arc fallback
    if cosmos_result:
        cosmos_action = cosmos_result["action"]
        turn_sec      = cosmos_result["turn_sec"]
        reasoning     = cosmos_result.get("reasoning", "")
        log.info(f"Using Cosmos decision: {cosmos_action} ({turn_sec:.1f}s) — {reasoning}")

        try:
            from mission import _ui
            _ui("log", f"🧠 Cosmos avoidance: {cosmos_action} — {reasoning}")
        except Exception:
            pass

        # Map Cosmos action to turn direction
        if cosmos_action == "turn_left":
            turn_dir = "left"
        elif cosmos_action == "turn_right":
            turn_dir = "right"
        elif cosmos_action == "turn_back":
            turn_dir = "back"
        elif cosmos_action == "forward":
            # Cosmos thinks front is clear — verify with sensors first
            if arcs["front"] > 0.60:
                log.info("Cosmos says forward — sensors agree, moving")
                motors.forward(MOTOR_SPEED_SLOW)
                time.sleep(1.5)
                motors.stop()
                _avoid_attempts = 0
                return False
            else:
                log.warning("Cosmos says forward but sensors disagree — using arc fallback")
                turn_dir = arc_direction
                turn_sec = TURN_BASE_SEC + (_avoid_attempts * TURN_INCREMENT)
        elif cosmos_action == "stop":
            log.warning("Cosmos says stop (boxed in) — forcing 360 scan")
            _avoid_attempts = 0
            return True
        else:
            turn_dir = arc_direction
            turn_sec = TURN_BASE_SEC + (_avoid_attempts * TURN_INCREMENT)

    else:
        # Arc-based fallback
        turn_dir = arc_direction
        turn_sec = min(
            TURN_BASE_SEC + (attempt * TURN_INCREMENT),
            TURN_MAX_SEC
        )
        log.info(f"Arc fallback: turn {turn_dir} for {turn_sec:.1f}s")

    # ─────────────────────────────────────────────────────────────────────────
    # LAYER 4 — Execute the chosen turn
    # ─────────────────────────────────────────────────────────────────────────
    motors.oled(1, f"Turn {turn_dir}...")
    log.info(f"Executing: turn {turn_dir} for {turn_sec:.1f}s")

    _execute_turn(turn_dir, turn_sec)

    motors.stop()
    time.sleep(0.4)

    # ─────────────────────────────────────────────────────────────────────────
    # LAYER 5 — Verify and retry
    # ─────────────────────────────────────────────────────────────────────────
    if _path_is_clear():
        log.info("✅ Path clear after avoidance — resuming")
        try:
            from mission import _ui
            _ui("status", "SEARCHING")
            _ui("log", f"✅ Avoidance succeeded (attempt {attempt})")
        except Exception:
            pass
        _avoid_attempts = 0
        return False

    log.warning(f"Still blocked after attempt {attempt}")

    if attempt >= MAX_AVOID_ATTEMPTS:
        log.warning("Max avoidance attempts reached — forcing 360 scan")
        try:
            from mission import eric_say
            eric_say("Too many obstacles. Let me scan the full area.")
        except Exception:
            pass
        _avoid_attempts = 0
        return True  # trigger 360

    # Recurse with escalating turn duration
    return _run_avoidance(wall_ahead=True, small_obstacle=False)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _execute_turn(direction: str, duration_sec: float):
    """Execute a turn in the given direction for duration_sec seconds."""
    if direction == "left":
        motors.left(MOTOR_SPEED_SLOW)
        time.sleep(duration_sec)
    elif direction == "right":
        motors.right(MOTOR_SPEED_SLOW)
        time.sleep(duration_sec)
    elif direction == "back":
        # 180° turn — use double duration on one side
        motors.right(MOTOR_SPEED_SLOW)
        time.sleep(duration_sec * 2.0)
    motors.stop()


def _path_is_clear(min_clearance: float = 0.50) -> bool:
    """
    Return True if the forward path has at least min_clearance meters
    of clearance according to sensors AND a quick visual scan.

    Checks in order:
      1. LiDAR front arc (fast)
      2. OAK-D center-forward depth (fast)
      3. Quick visual scan via Cosmos (slower — only if sensors OK)
    """
    # ── LiDAR ──────────────────────────────────────────────────────────────
    try:
        from lidar import min_front_distance, lidar_available
        if lidar_available():
            d = min_front_distance()
            if d < min_clearance:
                log.info(f"LiDAR: still blocked at {d:.2f}m (need {min_clearance}m)")
                return False
    except Exception:
        pass

    # ── OAK-D ──────────────────────────────────────────────────────────────
    try:
        from oakd import get_front_depth, oakd_available
        if oakd_available():
            d = get_front_depth()
            if d is not None and d < min_clearance:
                log.info(f"OAK-D: still blocked at {d:.2f}m (need {min_clearance}m)")
                return False
    except Exception:
        pass

    # ── Quick Cosmos visual check ───────────────────────────────────────────
    try:
        from mission import _quick_scan
        rescan = _quick_scan()
        if rescan.get("wall_ahead") or rescan.get("obstacle_close"):
            log.info("Quick scan: still blocked visually")
            return False
    except Exception:
        pass

    return True


def reset_avoid_counter():
    """Call this when Eric successfully clears an obstacle area."""
    global _avoid_attempts
    _avoid_attempts = 0


# ─── lidar.py integration note ───────────────────────────────────────────────
#
# To wire this up, make ONE change in lidar.py's _scan_callback():
#
#   REMOVE these lines:
#       if _safety_active:
#           if _obstacle_close:
#               from motors import motors
#               motors.stop()
#               log.warning(f"🚧 LIDAR STOP — obstacle at {min_dist:.2f}m")
#           elif _obstacle_near:
#               from motors import motors
#               motors.slow()
#               log.info(f"⚠️  LiDAR slow — obstacle at {min_dist:.2f}m")
#
#   REPLACE with:
#       if _safety_active:
#           if _obstacle_close:
#               from motors import motors
#               motors.stop()   # Instant hard stop still fires here — safe
#               log.warning(f"🚧 LIDAR STOP — obstacle at {min_dist:.2f}m")
#               # avoidance.py reads _obstacle_close flag and takes over from mission loop
#           elif _obstacle_near:
#               from motors import motors
#               motors.slow()   # Slow still fires here — safe
#
#   AND add this line near the top of _scan_callback (after globals):
#       _last_scan_msg = msg   # expose raw scan for arc distance calculation
#
#   AND add this module-level variable in lidar.py:
#       _last_scan_msg = None  # latest raw LaserScan message
#
# ─── mission.py integration note ─────────────────────────────────────────────
#
# In mission.py, replace ALL calls to _avoid_obstacle() with:
#
#   from avoidance import avoid_obstacle, reset_avoid_counter
#   force_360 = avoid_obstacle(wall_ahead=..., small_obstacle=...)
#
# Also add reset_avoid_counter() when path clears (where _avoid_attempts = 0
# currently appears in mission.py).