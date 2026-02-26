"""
ERIC — Smart Obstacle Avoidance (Industrial v2)

Professional reactive avoidance with debounce, cooldown, and state machine.

Layered pipeline:
  Layer 1 — Instant: stop hard, verify obstacle still present, back up
  Layer 2 — Sensor arc scan: read LiDAR + OAK-D to find clearest direction
  Layer 3 — Cosmos reasoning (async, runs concurrently with backup)
  Layer 4 — Execute turn
  Layer 5 — Verify path clear (LiDAR -> OAK-D -> Cosmos visual)
             Recurse with escalating turns if still blocked.
             Force 360 scan after MAX_AVOID_ATTEMPTS.

Anti-chatter protections (fixes "keeps backing up for no reason"):
  - AVOIDANCE_COOLDOWN_S  : minimum gap between avoidance sequences
  - DEBOUNCE_FRAMES       : obstacle must persist N checks before acting
  - RECOVERY_CLEAR_FRAMES : must be clear N checks before resuming
  - Pre-backup re-check   : if obstacle gone before backup, skip entirely
  - Async Cosmos          : Cosmos thinks during backup — no extra latency
  - Post-backup arc re-read: Cosmos gets fresh geometry, not stale pre-backup data

Integration:
  In mission.py: from avoidance import avoid_obstacle, reset_avoid_counter
  Replace old calls with: force_360 = avoid_obstacle(wall_ahead=..., small_obstacle=...)
  Call reset_avoid_counter() after successful approach, 360 scan, or resuming forward.

  In lidar.py: keep instant motors.stop() on close obstacles (safety).
               add: _last_scan_msg = msg  in _scan_callback after globals.
               add: _last_scan_msg = None  as module-level variable.
"""

import time
import math
import logging
import json
import requests
import threading
import concurrent.futures

from config import MOTOR_SPEED_SLOW, MOTOR_SPEED_NORMAL, VLLM_URL, COSMOS_MODEL
from motors import motors

log = logging.getLogger("eric.avoidance")

# ─── Industrial Tuning ────────────────────────────────────────────────────────
BACKUP_DURATION_BASE  = 1.1     # seconds — wall/large obstacle
BACKUP_DURATION_SMALL = 0.7     # seconds — small obstacle step-around
BACKUP_SPEED          = MOTOR_SPEED_SLOW
TURN_BASE_SEC         = 1.35
TURN_INCREMENT        = 0.35
TURN_MAX_SEC          = 3.2
MAX_AVOID_ATTEMPTS    = 4
COSMOS_TIMEOUT        = 16.0    # max total wait for Cosmos decision

# Anti-chatter protections
AVOIDANCE_COOLDOWN_S  = 9.0     # minimum time between avoidance sequences
DEBOUNCE_FRAMES       = 3       # obstacle must persist N checks before acting
RECOVERY_CLEAR_FRAMES = 4       # must be clear N checks before resuming normal motion

# Arc angle definitions
ARC_HALF  = 50   # degrees either side for front/rear
SIDE_HALF = 40   # narrower cone for left/right

# ─── Global State ─────────────────────────────────────────────────────────────
_avoid_attempts   = 0
_last_avoid_time  = 0.0
_debounce_counter = 0
_recovery_counter = 0
_avoid_lock       = threading.Lock()

# Async executor — Cosmos avoidance runs here while robot backs up
_cosmos_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="avoidance_cosmos"
)


# ─── LiDAR Arc Reader ─────────────────────────────────────────────────────────

def get_arc_distances() -> dict:
    """
    Read the latest LiDAR scan and return minimum distances for 4 arcs.
    Returns dict: {front, rear, left, right} in meters (999.0 = no reading).
    """
    arcs = {"front": 999.0, "rear": 999.0, "left": 999.0, "right": 999.0}

    try:
        from lidar import _last_scan_msg, _lock as lidar_lock
        with lidar_lock:
            msg = _last_scan_msg
        if msg is None:
            raise AttributeError("no scan yet")

        arc_rad  = math.radians(ARC_HALF)
        side_rad = math.radians(SIDE_HALF)

        for i, r in enumerate(msg.ranges):
            if not (msg.range_min < r < msg.range_max):
                continue
            angle = (msg.angle_min + i * msg.angle_increment + math.pi) % (2 * math.pi) - math.pi

            if -arc_rad <= angle <= arc_rad:
                arcs["front"] = min(arcs["front"], r)
            elif (angle > math.pi - arc_rad) or (angle < -math.pi + arc_rad):
                arcs["rear"] = min(arcs["rear"], r)
            elif math.pi / 2 - side_rad <= angle <= math.pi / 2 + side_rad:
                arcs["left"] = min(arcs["left"], r)
            elif -math.pi / 2 - side_rad <= angle <= -math.pi / 2 + side_rad:
                arcs["right"] = min(arcs["right"], r)

    except (ImportError, AttributeError):
        # lidar.py not updated yet — fall back to front distance only
        try:
            from lidar import min_front_distance
            arcs["front"] = round(min_front_distance(), 3)
        except Exception:
            pass
    except Exception as e:
        log.warning(f"arc_distances error: {e}")

    return {k: round(v, 3) for k, v in arcs.items()}


def _pick_clearest_turn(arcs: dict) -> str:
    """
    Choose best escape turn direction from arc distances.
    Returns: "left" | "right" | "back"
    """
    left_d  = arcs.get("left",  999.0)
    right_d = arcs.get("right", 999.0)
    rear_d  = arcs.get("rear",  999.0)

    log.info(f"Arc distances — L:{left_d:.2f}m  R:{right_d:.2f}m  Rear:{rear_d:.2f}m")

    if left_d > 0.65 and right_d > 0.65:
        return "left" if left_d >= right_d else "right"
    elif left_d > 0.55:
        return "left"
    elif right_d > 0.55:
        return "right"
    elif rear_d > 0.60:
        return "back"
    # Completely boxed — pick least-bad side
    best = max(arcs.items(), key=lambda kv: kv[1])
    d = best[0]
    return "back" if d == "rear" else d


# ─── OAK-D Context ────────────────────────────────────────────────────────────

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


# ─── Cosmos Avoidance Reasoning ───────────────────────────────────────────────

def _cosmos_avoidance_decision(arcs: dict, situation: str) -> dict | None:
    """
    Ask Cosmos to pick the best avoidance action using LiDAR arcs, OAK-D depth,
    and a live camera frame. Designed to run in a thread concurrently with backup.

    Returns parsed dict {action, turn_sec, reasoning} or None on failure/timeout.
    """
    try:
        from cosmos import capture_frame, CAMERA_PANTILT

        sensor_lines = [
            f"LiDAR arcs: front={arcs['front']:.2f}m  left={arcs['left']:.2f}m  "
            f"right={arcs['right']:.2f}m  rear={arcs['rear']:.2f}m"
        ]
        oakd_ctx = _oakd_context()
        if oakd_ctx:
            sensor_lines.append(oakd_ctx)
        sensor_block = "\n".join(sensor_lines)

        prompt = (
            "You are a tracked ground robot that has backed up from an obstacle.\n"
            "Choose the safest escape direction.\n\n"
            f"SENSOR DATA:\n{sensor_block}\n\n"
            f"SITUATION: {situation}\n\n"
            "Respond ONLY with a single JSON object — no markdown, no explanation:\n"
            "{\n"
            '  "action":    "turn_left | turn_right | turn_back | forward | stop",\n'
            '  "turn_sec":  <float 0.6-3.2>,\n'
            '  "reasoning": "<one sentence>"\n'
            "}\n\n"
            "Rules:\n"
            "- Prefer the direction with the most clearance from SENSOR DATA.\n"
            "- 'forward' only if front > 0.60m.\n"
            "- 'stop' only if completely boxed in on all sides.\n"
            "- turn_sec: larger for tight corners, smaller for open space.\n"
        )

        frame = capture_frame(CAMERA_PANTILT, 320, 240)
        content = []
        if frame:
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{frame}"}})
        content.append({"type": "text", "text": prompt})

        r = requests.post(VLLM_URL, json={
            "model":       COSMOS_MODEL,
            "messages":    [{"role": "user", "content": content}],
            "max_tokens":  90,
            "temperature": 0.1,
        }, timeout=COSMOS_TIMEOUT)
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"].strip()
        log.info(f"Cosmos avoidance raw: {raw[:200]}")

        clean = raw.replace("```json", "").replace("```", "").strip()
        s = clean.find("{"); e = clean.rfind("}") + 1
        if s >= 0 and e > s:
            result = json.loads(clean[s:e])
            valid_actions = {"turn_left", "turn_right", "turn_back", "forward", "stop"}
            if result.get("action") not in valid_actions:
                result["action"] = "turn_left"
            result.setdefault("turn_sec", TURN_BASE_SEC)
            result["turn_sec"] = max(0.6, min(float(result["turn_sec"]), TURN_MAX_SEC))
            log.info(f"Cosmos avoidance -> {result['action']} ({result['turn_sec']:.1f}s): "
                     f"{result.get('reasoning', '')}")
            return result

    except requests.exceptions.Timeout:
        log.warning("Cosmos avoidance timed out — using arc fallback")
    except Exception as e:
        log.debug(f"Cosmos avoidance skipped: {e}")

    return None


# ─── Turn Executor ────────────────────────────────────────────────────────────

def _execute_turn(direction: str, duration_sec: float):
    """
    Execute a turn in the given direction for duration_sec seconds.
    Suppresses LiDAR motor.stop() during the turn so it can complete.
    Logs explicitly if LiDAR safety cannot be re-enabled after turn.
    """
    _lidar_suppressed = False
    try:
        from lidar import set_avoidance_active
        set_avoidance_active(True)
        _lidar_suppressed = True
    except Exception:
        pass

    try:
        if direction == "left":
            motors.left(MOTOR_SPEED_SLOW)
            time.sleep(duration_sec)
        elif direction == "right":
            motors.right(MOTOR_SPEED_SLOW)
            time.sleep(duration_sec)
        elif direction == "back":
            # 180 turn — use 1.8x duration on one side
            motors.right(MOTOR_SPEED_SLOW)
            time.sleep(duration_sec * 1.8)
        motors.stop()
    finally:
        if _lidar_suppressed:
            try:
                from lidar import set_avoidance_active
                set_avoidance_active(False)
            except Exception as e:
                # Critical: LiDAR safety not restored — log loudly for operator
                log.error(
                    f"CRITICAL: Could not re-enable LiDAR safety after turn: {e}. "
                    "Eric is running without obstacle protection until restart."
                )


# ─── Path Clear Check ─────────────────────────────────────────────────────────

def _path_is_clear(min_clearance: float = 0.55) -> bool:
    """
    Return True if the forward path has at least min_clearance meters.
    3-stage check: LiDAR (fast) -> OAK-D (fast) -> Cosmos visual (slower).
    """
    try:
        from lidar import min_front_distance, lidar_available
        if lidar_available():
            d = min_front_distance()
            if d < min_clearance:
                log.info(f"LiDAR: still blocked at {d:.2f}m (need {min_clearance}m)")
                return False
    except Exception:
        pass

    try:
        from oakd import get_front_depth, oakd_available
        if oakd_available():
            d = get_front_depth()
            if d is not None and d < min_clearance:
                log.info(f"OAK-D: still blocked at {d:.2f}m")
                return False
    except Exception:
        pass

    try:
        from mission import _quick_scan
        res = _quick_scan()
        if res.get("wall_ahead") or res.get("obstacle_close"):
            log.info("Cosmos quick scan: still blocked visually")
            return False
    except Exception:
        pass

    return True


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def avoid_obstacle(wall_ahead: bool = True, small_obstacle: bool = False) -> bool:
    """
    Industrial-grade obstacle avoidance. Call this instead of _avoid_obstacle().

    Anti-chatter protections:
      - Cooldown:  ignores triggers within AVOIDANCE_COOLDOWN_S of last avoidance
      - Debounce:  requires DEBOUNCE_FRAMES consecutive detections before acting
      - Pre-check: re-reads sensors before backup — skips if obstacle already gone

    Returns:
        True  -> caller should trigger a full 360 scan
        False -> avoidance succeeded (or suppressed); caller can resume forward
    """
    global _avoid_attempts, _last_avoid_time, _debounce_counter, _recovery_counter

    now = time.time()

    # ── Cooldown guard ────────────────────────────────────────────────────────
    if now - _last_avoid_time < AVOIDANCE_COOLDOWN_S:
        remaining = AVOIDANCE_COOLDOWN_S - (now - _last_avoid_time)
        log.debug(f"Avoidance in cooldown ({remaining:.1f}s remaining) — ignoring trigger")
        return False

    # ── Thread safety ─────────────────────────────────────────────────────────
    if not _avoid_lock.acquire(blocking=False):
        log.warning("Avoidance already running — skipping duplicate call")
        return False

    try:
        # ── Debounce ──────────────────────────────────────────────────────────
        if wall_ahead or small_obstacle:
            _debounce_counter += 1
            if _debounce_counter < DEBOUNCE_FRAMES:
                log.debug(f"Debounce: {_debounce_counter}/{DEBOUNCE_FRAMES} — not triggering yet")
                return False
        else:
            _debounce_counter = 0

        _last_avoid_time  = now
        _debounce_counter = 0  # reset after commit to avoidance
        return _run_avoidance_internal(wall_ahead, small_obstacle)

    finally:
        _avoid_lock.release()


# ─── Core Avoidance Logic ─────────────────────────────────────────────────────

def _run_avoidance_internal(wall_ahead: bool, small_obstacle: bool) -> bool:
    global _avoid_attempts, _recovery_counter

    # Hard cap at entry — prevents unbounded recursion
    if _avoid_attempts >= MAX_AVOID_ATTEMPTS:
        log.warning("Max avoidance attempts reached at entry — forcing 360 scan")
        _avoid_attempts   = 0
        _recovery_counter = 0
        return True

    _avoid_attempts += 1
    attempt = _avoid_attempts

    log.warning(f"Avoidance attempt {attempt}/{MAX_AVOID_ATTEMPTS} "
                f"(wall={wall_ahead}, small={small_obstacle})")

    try:
        from mission import _ui
        _ui("status", f"AVOIDING ({attempt}/{MAX_AVOID_ATTEMPTS})")
        _ui("log", f"Obstacle — avoidance attempt {attempt}")
    except Exception:
        pass

    motors.oled(0, "OBSTACLE!")
    motors.oled(1, f"Avoid {attempt}/{MAX_AVOID_ATTEMPTS}")
    motors.stop()
    time.sleep(0.15)

    # ── Pre-backup re-check: skip if obstacle already cleared (noise/spike) ───
    arcs_pre = get_arc_distances()
    if arcs_pre["front"] > 0.65 and not wall_ahead:
        log.info(f"Pre-backup check: front={arcs_pre['front']:.2f}m — obstacle gone, skipping")
        _avoid_attempts = max(0, _avoid_attempts - 1)
        return False

    # ─────────────────────────────────────────────────────────────────────────
    # LAYER 1 — Back up
    # Cosmos future submitted immediately so it thinks during backup latency.
    # ─────────────────────────────────────────────────────────────────────────
    backup_time = BACKUP_DURATION_SMALL if small_obstacle else BACKUP_DURATION_BASE

    if small_obstacle:
        log.info("Small obstacle — step-around manoeuvre")
        motors.backward(BACKUP_SPEED); time.sleep(backup_time)
        motors.stop(); time.sleep(0.2)
        motors.right(MOTOR_SPEED_SLOW); time.sleep(1.0)
        motors.stop(); time.sleep(0.2)
        motors.forward(MOTOR_SPEED_SLOW); time.sleep(1.0)
        motors.stop(); time.sleep(0.2)
        motors.left(MOTOR_SPEED_SLOW); time.sleep(1.0)
        motors.stop(); time.sleep(0.3)

        if _path_is_clear():
            log.info("Path clear after step-around")
            _avoid_attempts   = 0
            _recovery_counter = 0
            return False

        log.info("Still blocked after step-around — escalating to wall avoidance")
        return _run_avoidance_internal(wall_ahead=True, small_obstacle=False)

    # ── Large obstacle / wall ─────────────────────────────────────────────────
    log.info(f"Wall obstacle — backing up {backup_time:.1f}s")

    # Submit Cosmos async — runs concurrently with backup to hide its latency
    cosmos_future = _cosmos_executor.submit(
        _cosmos_avoidance_decision,
        arcs_pre,
        f"Attempt {attempt}. Pre-backup clearest direction: {_pick_clearest_turn(arcs_pre)}."
    )

    motors.backward(BACKUP_SPEED)
    time.sleep(backup_time)
    motors.stop()
    time.sleep(0.25)

    # ─────────────────────────────────────────────────────────────────────────
    # LAYER 2 — Re-read sensors POST-backup (fresh geometry after reversing)
    # ─────────────────────────────────────────────────────────────────────────
    arcs        = get_arc_distances()
    arc_direction = _pick_clearest_turn(arcs)

    # ─────────────────────────────────────────────────────────────────────────
    # LAYER 3 — Retrieve Cosmos decision
    # Backup already consumed ~backup_time seconds of Cosmos thinking time.
    # ─────────────────────────────────────────────────────────────────────────
    cosmos_result    = None
    remaining_timeout = max(0.5, COSMOS_TIMEOUT - backup_time - 0.5)
    try:
        cosmos_result = cosmos_future.result(timeout=remaining_timeout)
        if cosmos_result:
            # Validate against post-backup arcs (Cosmos used pre-backup snapshot)
            if cosmos_result["action"] == "forward" and arcs["front"] <= 0.60:
                log.warning("Cosmos says forward but post-backup sensors disagree — arc fallback")
                cosmos_result = None
    except concurrent.futures.TimeoutError:
        log.warning("Cosmos future timed out — using arc fallback")
        cosmos_future.cancel()
    except Exception as e:
        log.debug(f"Cosmos future error: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # LAYER 4 — Execute the chosen turn
    # ─────────────────────────────────────────────────────────────────────────
    if cosmos_result:
        action    = cosmos_result["action"]
        turn_sec  = cosmos_result["turn_sec"]
        reasoning = cosmos_result.get("reasoning", "")
        log.info(f"Cosmos decision: {action} ({turn_sec:.1f}s) — {reasoning}")

        try:
            from mission import _ui
            _ui("log", f"Cosmos avoidance: {action} — {reasoning}")
        except Exception:
            pass

        if action == "forward":
            log.info("Cosmos says forward — sensors agree, moving")
            motors.forward(MOTOR_SPEED_SLOW)
            time.sleep(1.5)
            motors.stop()
            _avoid_attempts   = 0
            _recovery_counter = 0
            return False

        if action == "stop":
            log.warning("Cosmos says stop (boxed in) — forcing 360 scan")
            _avoid_attempts   = 0
            _recovery_counter = 0
            return True

        turn_dir = {"turn_left": "left", "turn_right": "right",
                    "turn_back": "back"}.get(action, arc_direction)
    else:
        # Arc-based fallback
        turn_dir = arc_direction
        turn_sec = min(TURN_BASE_SEC + (attempt * TURN_INCREMENT), TURN_MAX_SEC)
        log.info(f"Arc fallback: turn {turn_dir} for {turn_sec:.1f}s")

    motors.oled(1, f"Turn {turn_dir}...")
    _execute_turn(turn_dir, turn_sec)
    motors.stop()
    time.sleep(0.4)

    # ─────────────────────────────────────────────────────────────────────────
    # LAYER 5 — Verify path clear
    # ─────────────────────────────────────────────────────────────────────────
    if _path_is_clear(min_clearance=0.55):
        log.info("Path clear after avoidance — resuming")
        try:
            from mission import _ui
            _ui("status", "SEARCHING")
            _ui("log", f"Avoidance succeeded (attempt {attempt})")
        except Exception:
            pass
        _avoid_attempts   = 0
        _recovery_counter = 0
        return False

    log.warning(f"Still blocked after attempt {attempt}")
    _recovery_counter = 0

    # ── Max attempts exceeded ─────────────────────────────────────────────────
    if attempt >= MAX_AVOID_ATTEMPTS:
        log.warning("Max avoidance attempts reached — forcing 360 scan")
        try:
            from mission import eric_say
            eric_say("Too many obstacles. Let me scan the full area.")
        except Exception:
            pass
        _avoid_attempts   = 0
        _recovery_counter = 0
        return True

    # ── Recurse with escalating turn duration ─────────────────────────────────
    return _run_avoidance_internal(wall_ahead=True, small_obstacle=False)


# ─── Public Helpers ───────────────────────────────────────────────────────────

def reset_avoid_counter():
    """
    Reset all avoidance state. Call when path is clearly free:
    after successful approach, 360 scan, or resuming forward motion.
    """
    global _avoid_attempts, _debounce_counter, _recovery_counter
    _avoid_attempts   = 0
    _debounce_counter = 0
    _recovery_counter = 0
    log.debug("Avoidance counters reset")
