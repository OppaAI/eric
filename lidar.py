"""
ERIC — LiDAR Safety Monitor  (Layer 1) + SLAM Integration
Waveshare UGV Beast D500 LiDAR via ROS2 LaserScan topic.

ROS2 node sharing:
  Uses ros_core.get_node() — no longer chases _node imports from nav2/odom.
  The node-chaining import pattern (from nav2 import _node; from odom import
  _node) was fragile: if either module hadn't initialised yet the chain fell
  through to creating a duplicate node, and both modules could end up with
  their subscriptions on different executors.

Layer 1 guarantee:
  Every scan callback fires motors.stop() / motors.slow() BEFORE returning.
  Completely independent of Cosmos, Nav2, and mission logic.

Staleness watchdog:
  If no scan arrives for STALE_TIMEOUT_S seconds, _lidar_ok is cleared and
  lidar_available() returns False. Background thread monitors continuously.

Motor UART heartbeat:
  Checks every MOTOR_HB_INTERVAL_S that motors._ser is open.
  On failure: clears _motor_link_ok so lidar_available() returns False.

Architecture:
  D500 LiDAR → /scan → _scan_callback() (ros_core spin thread)
                              │
              ┌───────────────┼────────────────────┐
              ▼               ▼                    ▼
     obstacle check     void check (kept    update _last_scan_time
     (every scan)       for reference,      (staleness watchdog)
                        not called in
                        scan callback)
              │
     motors.stop/slow
"""

import logging
import math
import threading
import time

log = logging.getLogger("eric.lidar")

# ── Safety distances (meters) ──────────────────────────────────────────────────
STOP_DIST        = 0.30
SLOW_DIST        = 0.60
FRONT_ARC_DEG    = 60
CHASSIS_BLIND_M  = 0.12   # ignore returns closer than this — antenna / chassis self-detection

# ── Motor heartbeat watchdog ───────────────────────────────────────────────────
MOTOR_HB_INTERVAL_S = 2.0
_motor_link_ok      = True
_motor_hb_thread    = None

# ── Staleness watchdog ─────────────────────────────────────────────────────────
STALE_TIMEOUT_S      = 2.0
STALE_CHECK_INTERVAL = 0.5

# ── Module state ───────────────────────────────────────────────────────────────
_lidar_ok         = False
_lidar_stale      = False
_obstacle_close   = False
_obstacle_near    = False
_void_active      = False
_min_distance     = 999.0
_safety_active    = True
_avoidance_active = False

_LOG_RATE_S         = 10.0
_last_stop_log_time = 0.0
_last_slow_log_time = 0.0
_last_stop_dist     = 999.0
_last_slow_dist     = 999.0

_node           = None
_sub            = None
_watchdog_thread = None

_last_scan_msg  = None
_last_scan_time = 0.0

_lock = threading.Lock()


# ─── Public API ───────────────────────────────────────────────────────────────

def lidar_available() -> bool:
    with _lock:
        return _lidar_ok and not _lidar_stale and _motor_link_ok


def obstacle_close() -> bool:
    with _lock:
        return _obstacle_close and _safety_active and not _lidar_stale


def obstacle_near() -> bool:
    with _lock:
        return _obstacle_near and _safety_active and not _lidar_stale


def min_front_distance() -> float:
    with _lock:
        return _min_distance


def safe_to_forward() -> bool:
    """
    True if it is safe to call motors.forward().
    Returns False if LiDAR offline — cannot confirm path is clear.
    Call before EVERY motors.forward() in mission.py.
    """
    with _lock:
        if not _lidar_ok or _lidar_stale or not _motor_link_ok:
            return False
        if not _safety_active:
            return True
        return not _obstacle_close and not _obstacle_near


def set_safety_active(active: bool):
    global _safety_active
    _safety_active = active
    log.info(f"LiDAR safety: {'ENABLED' if active else 'DISABLED'}")


def set_avoidance_active(active: bool):
    """
    Call True before executing a turn in avoidance.py, False after.
    LiDAR still detects but will not call motors.stop() during the turn.
    """
    global _avoidance_active
    _avoidance_active = active
    log.debug(f"LiDAR avoidance: {'ACTIVE — stop suppressed' if active else 'INACTIVE'}")


def get_last_scan():
    with _lock:
        return _last_scan_msg


# ─── Void / floor-drop detection (reference — not called in scan callback) ────

def lidar_void_ahead(
    min_return_ratio: float = 0.06,
    front_arc_deg:    int   = 30,
) -> dict:
    """
    Detect floor voids from D500 scan.
    NOTE: D500 is a horizontal 2D scanner — void detection is kept here for
    reference but is NOT called from _scan_callback. OAK-D stereo depth handles
    floor-drop detection. Calling this on a horizontal scanner produces false
    positives on every wide doorway or open room.
    """
    with _lock:
        msg = _last_scan_msg
    if msg is None:
        return {"void_detected": False, "confidence": "low",
                "return_ratio": 1.0, "mean_distance": None,
                "reason": "no scan data"}

    n         = len(msg.ranges)
    angle_inc = msg.angle_increment
    angle_min = msg.angle_min
    arc_rad   = math.radians(front_arc_deg)

    total_in_arc = 0
    valid_ranges = []
    for i, r in enumerate(msg.ranges):
        angle = angle_min + i * angle_inc
        if -arc_rad <= angle <= arc_rad:
            total_in_arc += 1
            if msg.range_min < r < msg.range_max:
                valid_ranges.append(r)

    if total_in_arc == 0:
        return {"void_detected": False, "confidence": "low",
                "return_ratio": 1.0, "mean_distance": None,
                "reason": "no arc indices found"}

    return_ratio = len(valid_ranges) / total_in_arc
    mean_dist    = float(sum(valid_ranges) / len(valid_ranges)) if valid_ranges else None

    void_detected = False
    confidence    = "low"
    reason        = "normal floor returns"

    if return_ratio < 0.05:
        void_detected, confidence = True, "high"
        reason = (f"front arc {return_ratio:.0%} valid returns — floor void or drop")
    elif return_ratio < min_return_ratio:
        void_detected, confidence = True, "medium"
        reason = (f"front arc sparse {return_ratio:.0%} — likely stairs or gap")
    elif mean_dist is not None and mean_dist > 3.5 and return_ratio < 0.4:
        void_detected, confidence = True, "medium"
        reason = (f"sparse far returns ({mean_dist:.1f}m, {return_ratio:.0%}) — stairwell")

    return {
        "void_detected": void_detected,
        "confidence":    confidence,
        "return_ratio":  round(return_ratio, 3),
        "mean_distance": round(mean_dist, 2) if mean_dist is not None else None,
        "reason":        reason,
    }


# ─── Front depth — 3-patch sampling ──────────────────────────────────────────

def get_front_depth_lidar() -> float | None:
    """
    Minimum range in front arc using three angle sub-samples.
    One textureless patch can't silence the whole front check.
    """
    with _lock:
        msg = _last_scan_msg
    if msg is None:
        return None

    angle_inc    = msg.angle_increment
    angle_min    = msg.angle_min
    sub_arcs_deg = [-15.0, 0.0, 15.0]
    half_width   = math.radians(8.0)
    samples      = []

    for centre_deg in sub_arcs_deg:
        centre_rad = math.radians(centre_deg)
        arc_ranges = [
            r for i, r in enumerate(msg.ranges)
            if abs((angle_min + i * angle_inc) - centre_rad) <= half_width
            and CHASSIS_BLIND_M < r < msg.range_max
        ]
        if arc_ranges:
            samples.append(min(arc_ranges))

    return min(samples) if samples else None


# ─── Initialisation ───────────────────────────────────────────────────────────

def init_lidar() -> bool:
    """
    Subscribe to /scan topic from D500 LiDAR using the shared ros_core node.
    Returns True if ROS2 available and subscription created.
    Also starts staleness watchdog and motor heartbeat threads.
    """
    global _lidar_ok, _lidar_stale, _sub

    try:
        from ros_core import get_node, ensure_spinning
        from sensor_msgs.msg import LaserScan
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

        node = get_node()
        if node is None:
            raise RuntimeError("ROS2 not available")

        # D500 publishes /scan with BEST_EFFORT reliability.
        # Default RELIABLE QoS causes zero messages to be received.
        scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        _sub = node.create_subscription(LaserScan, "/scan", _scan_callback, scan_qos)

        ensure_spinning()   # no-op if already spinning from odom/nav2

        _lidar_ok    = True
        _lidar_stale = False
        log.info("✅ LiDAR: D500 safety monitor active")
        _start_staleness_watchdog()
        _start_motor_heartbeat()
        return True

    except ImportError:
        log.warning("⚠️  ROS2 not found — LiDAR safety monitor disabled")
        _lidar_ok = False
        return False
    except Exception as e:
        log.warning(f"⚠️  LiDAR init failed ({e}) — safety monitor disabled")
        _lidar_ok = False
        return False


# ─── Staleness watchdog ───────────────────────────────────────────────────────

def _start_staleness_watchdog():
    global _watchdog_thread
    if _watchdog_thread and _watchdog_thread.is_alive():
        return
    _watchdog_thread = threading.Thread(
        target=_staleness_loop, daemon=True, name="lidar-watchdog"
    )
    _watchdog_thread.start()
    log.info("👁  LiDAR staleness watchdog started")


def _staleness_loop():
    global _lidar_stale
    was_stale = False
    while True:
        time.sleep(STALE_CHECK_INTERVAL)
        with _lock:
            last = _last_scan_time
            ok   = _lidar_ok
        if not ok:
            continue
        age      = time.monotonic() - last
        is_stale = age > STALE_TIMEOUT_S
        if is_stale and not was_stale:
            log.warning(f"⚠️  LiDAR STALE — no scan for {age:.1f}s")
            with _lock:
                _lidar_stale = True
            was_stale = True
        elif not is_stale and was_stale:
            log.info("✅ LiDAR scan data resumed")
            with _lock:
                _lidar_stale = False
            was_stale = False


# ─── Scan callback ────────────────────────────────────────────────────────────

def _scan_callback(msg):
    """
    Process LaserScan from D500 (~10Hz).
    All safety reactions happen here — zero latency, no queuing.
    Void check intentionally disabled — OAK-D handles floor-drop detection.
    """
    global _obstacle_close, _obstacle_near, _min_distance
    global _last_scan_msg, _last_scan_time, _void_active

    try:
        now       = time.monotonic()
        angle_inc = msg.angle_increment
        angle_min = msg.angle_min
        arc_rad   = math.radians(FRONT_ARC_DEG)

        with _lock:
            _last_scan_msg  = msg
            _last_scan_time = now

        # Front-arc obstacle detection
        # CHASSIS_BLIND_M filters self-returns from antenna / chassis body
        front_distances = [
            r for i, r in enumerate(msg.ranges)
            if -arc_rad <= (angle_min + i * angle_inc) <= arc_rad
            and CHASSIS_BLIND_M < r < msg.range_max
        ]

        if not front_distances:
            with _lock:
                _obstacle_close = False
                _obstacle_near  = False
                _min_distance   = 999.0
        else:
            min_dist = min(front_distances)
            is_close = min_dist < STOP_DIST
            is_near  = min_dist < SLOW_DIST

            with _lock:
                _min_distance   = min_dist
                _obstacle_close = is_close
                _obstacle_near  = is_near
                sa = _safety_active
                av = _avoidance_active

            if sa and not av:
                if is_close:
                    _motors_stop("LIDAR STOP", f"obstacle at {min_dist:.2f}m")
                elif is_near:
                    _motors_slow("LiDAR slow", f"obstacle at {min_dist:.2f}m")
            elif sa and av and is_close:
                log.debug(f"LiDAR: {min_dist:.2f}m — suppressed during avoidance")

        with _lock:
            _void_active = False   # D500 doesn't do void detection — OAK-D does

    except Exception as e:
        log.error(f"LiDAR scan callback error: {e}")


# ─── Motor helpers ────────────────────────────────────────────────────────────

def _motors_stop(tag: str, reason: str):
    global _last_stop_log_time, _last_stop_dist
    try:
        from motors import motors
        motors.stop()
        now = time.monotonic()
        try:
            dist = float(reason.split("at ")[-1].rstrip("m"))
        except Exception:
            dist = 0.0
        if now - _last_stop_log_time >= _LOG_RATE_S or abs(dist - _last_stop_dist) > 0.10:
            log.warning(f"🚧 {tag} — {reason}")
            _last_stop_log_time = now
            _last_stop_dist     = dist
    except Exception:
        pass


def _motors_slow(tag: str, reason: str):
    global _last_slow_log_time, _last_slow_dist
    try:
        from motors import motors
        motors.slow()
        now = time.monotonic()
        try:
            dist = float(reason.split("at ")[-1].rstrip("m"))
        except Exception:
            dist = 0.0
        if now - _last_slow_log_time >= _LOG_RATE_S or abs(dist - _last_slow_dist) > 0.10:
            log.debug(f"⚠️  {tag} — {reason}")
            _last_slow_log_time = now
            _last_slow_dist     = dist
    except Exception:
        pass


# ─── Motor UART heartbeat watchdog ────────────────────────────────────────────

def _start_motor_heartbeat():
    global _motor_hb_thread
    if _motor_hb_thread and _motor_hb_thread.is_alive():
        return
    _motor_hb_thread = threading.Thread(
        target=_motor_hb_loop, daemon=True, name="motor-heartbeat"
    )
    _motor_hb_thread.start()
    log.info("💓 Motor UART heartbeat watchdog started")


def _motor_hb_loop():
    global _motor_link_ok
    was_ok = True
    while True:
        time.sleep(MOTOR_HB_INTERVAL_S)
        try:
            from motors import motors as _m
            ok = _m._ser is not None and _m._ser.is_open
        except Exception:
            ok = True   # motors not loaded yet — assume ok
        if not ok and was_ok:
            log.error("💔 Motor UART link LOST — lidar safety paused")
            with _lock:
                _motor_link_ok = False
            was_ok = False
        elif ok and not was_ok:
            log.info("💓 Motor UART link restored — lidar safety re-enabled")
            with _lock:
                _motor_link_ok = True
            was_ok = True


# ─── Status ───────────────────────────────────────────────────────────────────

def get_status() -> dict:
    with _lock:
        return {
            "available":      _lidar_ok and not _lidar_stale and _motor_link_ok,
            "stale":          _lidar_stale,
            "motor_link_ok":  _motor_link_ok,
            "safety_active":  _safety_active,
            "obstacle_close": _obstacle_close,
            "obstacle_near":  _obstacle_near,
            "void_active":    _void_active,
            "min_distance":   round(_min_distance, 2),
        }


def lidar_status_html() -> str:
    s = get_status()
    if not _lidar_ok:
        return """
        <div style="background:#1a1a1a;border:1px solid #444;border-radius:8px;
                    padding:10px;font-family:monospace;color:#666">
            📡 LiDAR: not connected
        </div>"""
    if s["stale"]:
        return """
        <div style="background:#1a1a1a;border:2px solid #cc0000;border-radius:8px;
                    padding:10px;font-family:monospace;">
            <div style="color:#cc0000;font-weight:bold">📡 LiDAR D500 — ⚠️ STALE</div>
            <div style="color:#888;font-size:0.85em;margin-top:4px">
                /scan stopped — safety disabled until restored
            </div>
        </div>"""
    if s["void_active"]:
        color, label = "#cc0000", "🕳️  VOID / DROP DETECTED"
    elif s["obstacle_close"]:
        color, label = "#cc0000", "🚧 OBSTACLE CLOSE"
    elif s["obstacle_near"]:
        color, label = "#ff6600", "⚠️  OBSTACLE NEAR"
    else:
        color, label = "#76b900", "✅ CLEAR"
    dist     = s["min_distance"]
    dist_str = f"{dist:.2f}m" if dist < 999 else "—"
    return f"""
    <div style="background:#1a1a1a;border:1px solid {color};border-radius:8px;
                padding:10px;font-family:monospace;">
        <div style="color:{color};font-weight:bold">📡 LiDAR D500 — {label}</div>
        <div style="color:#aaa;font-size:0.85em;margin-top:4px">
            Front: <span style="color:#fff">{dist_str}</span>
            &nbsp;|&nbsp; Stop at: {STOP_DIST}m
            &nbsp;|&nbsp; Slow at: {SLOW_DIST}m
        </div>
    </div>"""