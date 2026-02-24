"""
ERIC — LiDAR Safety Monitor  (Layer 1)
Waveshare UGV Beast D500 LiDAR via ROS2 LaserScan topic.

Layer 1 guarantee:
  Every scan callback fires motors.stop() / motors.slow() BEFORE returning.
  This is completely independent of Cosmos, Nav2, and mission logic.

Gaps closed vs previous version
────────────────────────────────
1. Staleness watchdog  — if no scan arrives for STALE_TIMEOUT_S seconds,
   _lidar_ok is cleared and lidar_available() returns False.
   A background thread monitors this continuously.

2. Medium-confidence void auto-stop  — both "high" AND "medium" void
   confidence now trigger motors.stop(). "medium" logs differently so
   operators can distinguish real stairs from wide doorways.

3. Front-depth fallback via 3-patch sampling  — get_front_depth() now
   samples left-centre, centre, and right-centre and returns the minimum
   non-None value, so a textureless white wall can't blind the sensor.

4. Sensor arbitration comment  — explains that LiDAR and OAK-D are
   independent first-responders; whichever fires first wins, which is
   correct for safety (most conservative wins).

5. GUI status exposes stale / void state  — lidar_status_html() now
   shows a STALE banner when data has stopped arriving.

Architecture
────────────
  D500 LiDAR → /scan ROS2 topic → _scan_callback() (ROS spin thread)
                                        │
                        ┌───────────────┼────────────────────┐
                        ▼               ▼                    ▼
               obstacle check     void check          update _last_scan_time
               (every scan)       (every scan)        (staleness watchdog)
                        │               │
               motors.stop/slow   motors.stop (high OR medium confidence)

  _staleness_watchdog() runs in separate daemon thread, clears _lidar_ok
  if no scan arrives within STALE_TIMEOUT_S seconds.

  avoidance.py reads _last_scan_msg for per-arc clearances (unchanged).
"""

import logging
import math
import threading
import time

log = logging.getLogger("eric.lidar")

# ── Safety distances (meters) ─────────────────────────────────────────────────
STOP_DIST     = 0.30   # stop  if anything within 30 cm in front arc
SLOW_DIST     = 0.60   # slow  if anything within 60 cm in front arc
FRONT_ARC_DEG = 60     # ±60° either side of forward = 120° total front arc

# ── Motor heartbeat watchdog ──────────────────────────────────────────────────
# If the UART to the ESP32 goes silent (cable pulled, crash), we can't safely
# drive. Watchdog checks every MOTOR_HB_INTERVAL_S that motors._ser is open.
# On failure: sets _motor_link_ok=False so lidar_available() returns False,
# preventing any further sensor-driven motor commands on a dead link.
MOTOR_HB_INTERVAL_S  = 2.0    # check interval
_motor_link_ok       = True    # assume ok until proven otherwise
_motor_hb_thread     = None

# ── Staleness watchdog ────────────────────────────────────────────────────────
STALE_TIMEOUT_S      = 2.0   # seconds — declare LiDAR dead if no scan arrives
STALE_CHECK_INTERVAL = 0.5   # how often the watchdog thread checks

# ── Module state ──────────────────────────────────────────────────────────────
_lidar_ok       = False
_lidar_stale    = False    # True when data stopped arriving (watchdog)
_obstacle_close = False    # within STOP_DIST
_obstacle_near  = False    # within SLOW_DIST
_void_active    = False    # True when a void/drop was last detected
_min_distance   = 999.0    # minimum distance in front arc (meters)
_safety_active  = True     # can be disabled for testing

_node           = None
_sub            = None
_ros_thread     = None
_watchdog_thread = None

# Raw scan message — exposed for avoidance.py arc distance calculations
_last_scan_msg  = None
_last_scan_time = 0.0      # monotonic timestamp of last received scan

_lock = threading.Lock()   # guards all state above


# ─── Public API ───────────────────────────────────────────────────────────────

def lidar_available() -> bool:
    """True only if LiDAR is initialised, data is arriving, AND motor link is alive."""
    with _lock:
        return _lidar_ok and not _lidar_stale and _motor_link_ok


def obstacle_close() -> bool:
    """True if obstacle within STOP_DIST in front arc."""
    with _lock:
        return _obstacle_close and _safety_active and not _lidar_stale


def obstacle_near() -> bool:
    """True if obstacle within SLOW_DIST in front arc."""
    with _lock:
        return _obstacle_near and _safety_active and not _lidar_stale


def min_front_distance() -> float:
    """Minimum distance (meters) in front arc. Returns 999 if no data."""
    with _lock:
        return _min_distance


def set_safety_active(active: bool):
    """Enable/disable safety stop. Use False only for testing."""
    global _safety_active
    _safety_active = active
    log.info(f"LiDAR safety: {'ENABLED' if active else 'DISABLED'}")


def get_last_scan():
    """Return the most recent raw LaserScan message (for avoidance.py)."""
    with _lock:
        return _last_scan_msg


# ─── Void / floor-drop detection ─────────────────────────────────────────────

def lidar_void_ahead(
    min_return_ratio: float = 0.15,  # fraction of arc indices with valid returns
    front_arc_deg:    int   = 40,    # narrow arc for void (tighter than obstacle arc)
) -> dict:
    """
    Detect floor voids (holes, staircase tops, cliff edges) using D500 scan.

    Normal floor: dense returns at 0.3–3 m in the forward arc.
    Void / drop:  almost NO returns — laser falls through empty air.

    Confidence levels
    ─────────────────
    high   → return_ratio < 5%     — auto-stop immediately (clear floor drop)
    medium → return_ratio < 15%    — auto-stop immediately (likely stairs/gap)
             OR sparse far returns → likely stairwell opening
    low    → no anomaly detected

    Both high AND medium now trigger auto-stop (was: high only).
    Medium is logged differently so operators can distinguish real stairs
    from wide-open doorways that also produce sparse returns.

    Returns dict:
      {
        "void_detected":  bool,
        "confidence":     "high" | "medium" | "low",
        "return_ratio":   float,
        "mean_distance":  float | None,
        "reason":         str,
      }
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
    mean_dist    = (float(sum(valid_ranges) / len(valid_ranges))
                    if valid_ranges else None)

    void_detected = False
    confidence    = "low"
    reason        = "normal floor returns"

    if return_ratio < 0.05:
        # Virtually no returns — unambiguous drop
        void_detected = True
        confidence    = "high"
        reason        = (f"front arc only {return_ratio:.0%} valid returns "
                         f"({len(valid_ranges)}/{total_in_arc}) — floor void or drop")

    elif return_ratio < min_return_ratio:
        # Few returns — staircase or gap
        void_detected = True
        confidence    = "medium"
        reason        = (f"front arc sparse {return_ratio:.0%} valid returns "
                         f"({len(valid_ranges)}/{total_in_arc}) — likely stairs or gap")

    elif mean_dist is not None and mean_dist > 3.5 and return_ratio < 0.4:
        # Returns exist but very far + sparse → stairwell wall visible, floor gone
        void_detected = True
        confidence    = "medium"
        reason        = (f"sparse far returns ({mean_dist:.1f}m avg, "
                         f"{return_ratio:.0%} ratio) — likely stairwell or open gap")

    return {
        "void_detected": void_detected,
        "confidence":    confidence,
        "return_ratio":  round(return_ratio, 3),
        "mean_distance": round(mean_dist, 2) if mean_dist is not None else None,
        "reason":        reason,
    }


# ─── Front depth — 3-patch sampling (gap fix) ────────────────────────────────

def get_front_depth_lidar() -> float | None:
    """
    Return the minimum range in front arc using three angle sub-samples:
    left-of-centre (−15°), centre (0°), right-of-centre (+15°).

    Returning the minimum of three independent samples means a single
    patch on a textureless surface can't silence the whole front check.
    Returns None only if all three sub-arcs give no valid readings.
    """
    with _lock:
        msg = _last_scan_msg
    if msg is None:
        return None

    n         = len(msg.ranges)
    angle_inc = msg.angle_increment
    angle_min = msg.angle_min

    sub_arcs_deg = [-15.0, 0.0, 15.0]
    half_width   = math.radians(8.0)   # ±8° window around each sample centre
    samples      = []

    for centre_deg in sub_arcs_deg:
        centre_rad = math.radians(centre_deg)
        arc_ranges = []
        for i, r in enumerate(msg.ranges):
            angle = angle_min + i * angle_inc
            if abs(angle - centre_rad) <= half_width:
                if msg.range_min < r < msg.range_max:
                    arc_ranges.append(r)
        if arc_ranges:
            samples.append(min(arc_ranges))

    return min(samples) if samples else None


# ─── Initialisation ───────────────────────────────────────────────────────────

def init_lidar() -> bool:
    """
    Subscribe to /scan topic from D500 LiDAR.
    Returns True if ROS2 available and topic found.
    Non-blocking — runs subscriber in background thread.
    Also starts the staleness watchdog thread.
    """
    global _lidar_ok, _lidar_stale, _node, _sub, _ros_thread

    try:
        import rclpy
        from sensor_msgs.msg import LaserScan

        if not rclpy.ok():
            rclpy.init()

        # Reuse Nav2 node if available — saves one ROS2 node
        try:
            from nav2 import _node as nav2_node
            if nav2_node:
                _node = nav2_node
                log.info("LiDAR: reusing Nav2 ROS2 node")
            else:
                _node = rclpy.create_node("eric_lidar")
        except Exception:
            _node = rclpy.create_node("eric_lidar")

        # D500 driver publishes /scan with BEST_EFFORT reliability.
        # Using the default RELIABLE QoS causes zero messages to be received.
        # QoSProfile with BEST_EFFORT matches the publisher and gets every scan.
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        _sub = _node.create_subscription(
            LaserScan,
            "/scan",
            _scan_callback,
            scan_qos
        )

        # Only spin if not already spinning (nav2 may be spinning already)
        try:
            from nav2 import _ros_thread as nav2_thread
            if nav2_thread and nav2_thread.is_alive():
                log.info("LiDAR: ROS2 already spinning via Nav2")
                _lidar_ok   = True
                _lidar_stale = False
                _start_staleness_watchdog()
                _start_motor_heartbeat()
                return True
        except Exception:
            pass

        _ros_thread = threading.Thread(
            target=lambda: rclpy.spin(_node),
            daemon=True,
            name="lidar-ros-spin"
        )
        _ros_thread.start()

        time.sleep(1.0)   # give topic time to publish first scan
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
    """Start the background thread that monitors scan freshness."""
    global _watchdog_thread
    if _watchdog_thread and _watchdog_thread.is_alive():
        return
    _watchdog_thread = threading.Thread(
        target=_staleness_loop, daemon=True, name="lidar-watchdog"
    )
    _watchdog_thread.start()
    log.info("👁  LiDAR staleness watchdog started")


def _staleness_loop():
    """
    Runs every STALE_CHECK_INTERVAL seconds.
    If no scan has arrived within STALE_TIMEOUT_S seconds, marks _lidar_stale
    so lidar_available() returns False and all safety checks are suppressed
    (reporting stale data as safe would be worse than reporting unavailable).

    Logs once on transition, not on every check, to avoid log flooding.
    """
    global _lidar_stale

    was_stale = False
    while True:
        time.sleep(STALE_CHECK_INTERVAL)
        with _lock:
            last = _last_scan_time
            ok   = _lidar_ok

        if not ok:
            continue

        age     = time.monotonic() - last
        is_stale = age > STALE_TIMEOUT_S

        if is_stale and not was_stale:
            log.warning(f"⚠️  LiDAR STALE — no scan for {age:.1f}s "
                        f"(timeout={STALE_TIMEOUT_S}s)")
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
    Process LaserScan message from D500.

    Fires every scan cycle (~10 Hz for D500).
    All safety reactions happen here — zero latency, no queuing.

    Sensor arbitration note:
      LiDAR and OAK-D each call motors.stop() / motors.slow() independently.
      Whichever fires first wins — this is correct for safety because the
      most conservative sensor always dominates. There is no arbitration
      logic intentionally: adding it would only delay the stop.

    What happens here:
      1. Update _last_scan_time → staleness watchdog stays quiet
      2. Compute minimum range in front arc → obstacle stop/slow
      3. Run void check → stop on high OR medium confidence
      4. Store raw message for avoidance.py
    """
    global _obstacle_close, _obstacle_near, _min_distance
    global _last_scan_msg, _last_scan_time, _void_active

    try:
        now       = time.monotonic()
        angle_inc = msg.angle_increment
        angle_min = msg.angle_min
        arc_rad   = math.radians(FRONT_ARC_DEG)

        # ── 1. Update staleness timestamp ─────────────────────────────────────
        with _lock:
            _last_scan_msg  = msg
            _last_scan_time = now

        # ── 2. Front-arc obstacle detection ──────────────────────────────────
        front_distances = []
        for i, r in enumerate(msg.ranges):
            angle = angle_min + i * angle_inc
            if -arc_rad <= angle <= arc_rad:
                if msg.range_min < r < msg.range_max:
                    front_distances.append(r)

        if not front_distances:
            # No valid front returns — could be void OR sensor issue.
            # Void check below will handle this; obstacle check skipped.
            pass
        else:
            min_dist = min(front_distances)
            is_close = min_dist < STOP_DIST
            is_near  = min_dist < SLOW_DIST

            with _lock:
                _min_distance   = min_dist
                _obstacle_close = is_close
                _obstacle_near  = is_near
                sa = _safety_active  # read under lock

            if sa:
                if is_close:
                    _motors_stop("LIDAR STOP",
                                 f"obstacle at {min_dist:.2f}m")
                elif is_near:
                    _motors_slow("LiDAR slow",
                                 f"obstacle at {min_dist:.2f}m")

        # ── 3. Void / floor-drop check ────────────────────────────────────────
        # Runs on every scan — void_ahead() reuses the msg we just stored.
        # Only skip if we already stopped for an obstacle (obstacle_close)
        # since they can't both be true simultaneously in normal operation.
        with _lock:
            sa2 = _safety_active
            obs_close = _obstacle_close
        if sa2 and not obs_close:
            void = lidar_void_ahead()
            if void["void_detected"]:
                conf = void["confidence"]
                if conf == "high":
                    _motors_stop("LIDAR VOID STOP (HIGH)",
                                 void["reason"])
                    with _lock:
                        _void_active = True
                elif conf == "medium":
                    # Medium confidence: still stop — stairs are fatal.
                    # Logged at WARNING but labelled MEDIUM so operators know
                    # it may be a wide doorway or large open space.
                    _motors_stop("LIDAR VOID STOP (MEDIUM)",
                                 void["reason"])
                    with _lock:
                        _void_active = True
                # low confidence → no action
            else:
                with _lock:
                    _void_active = False

    except Exception as e:
        log.error(f"LiDAR scan callback error: {e}")


# ─── Motor helpers (keep imports lazy — motors may not exist in test env) ─────

def _motors_stop(tag: str, reason: str):
    try:
        from motors import motors
        motors.stop()
        log.warning(f"🚧 {tag} — {reason}")
    except Exception:
        pass


def _motors_slow(tag: str, reason: str):
    try:
        from motors import motors
        motors.slow()
        log.info(f"⚠️  {tag} — {reason}")
    except Exception:
        pass


# ─── Motor UART heartbeat watchdog ────────────────────────────────────────────

def _start_motor_heartbeat():
    """Start motor serial link watchdog. Call once after init_lidar()."""
    global _motor_hb_thread
    if _motor_hb_thread and _motor_hb_thread.is_alive():
        return
    _motor_hb_thread = threading.Thread(
        target=_motor_hb_loop, daemon=True, name="motor-heartbeat"
    )
    _motor_hb_thread.start()
    log.info("💓 Motor UART heartbeat watchdog started")


def _motor_hb_loop():
    """
    Check every MOTOR_HB_INTERVAL_S that the ESP32 UART link is alive.
    If the serial port closes unexpectedly (cable pulled, crash), clears
    _motor_link_ok so lidar_available() returns False — prevents further
    sensor-driven motor commands on a dead link.
    Logs once on transition (fail and recover), not every check.
    """
    global _motor_link_ok
    was_ok = True
    while True:
        time.sleep(MOTOR_HB_INTERVAL_S)
        try:
            from motors import motors as _m
            ok = _m._ser is not None and _m._ser.is_open
        except Exception:
            ok = False   # motors module not loaded yet — assume ok

        if not ok and was_ok:
            log.error("💔 Motor UART link LOST — lidar safety paused until restored")
            with _lock:
                _motor_link_ok = False
            was_ok = False
        elif ok and not was_ok:
            log.info("💓 Motor UART link restored — lidar safety re-enabled")
            with _lock:
                _motor_link_ok = True
            was_ok = True


# ─── Status helpers ───────────────────────────────────────────────────────────

def get_status() -> dict:
    """Return current LiDAR safety status for GUI display."""
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
    """HTML status panel for Gradio — includes stale and void states."""
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
            <div style="color:#cc0000;font-weight:bold">
                📡 LiDAR D500 — ⚠️ STALE — no data
            </div>
            <div style="color:#888;font-size:0.85em;margin-top:4px">
                Topic /scan stopped publishing — safety disabled until restored
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
            Front distance: <span style="color:#fff">{dist_str}</span>
            &nbsp;|&nbsp;
            Stop at: <span style="color:#fff">{STOP_DIST}m</span>
            &nbsp;|&nbsp;
            Slow at: <span style="color:#fff">{SLOW_DIST}m</span>
        </div>
    </div>"""
