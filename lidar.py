"""
ERIC — LiDAR Safety Monitor
Waveshare UGV Beast D500 LiDAR via ROS2 LaserScan topic.
Runs as independent safety layer — stops Eric if obstacle too close,
regardless of what Cosmos is doing.

Architecture:
  D500 LiDAR → /scan topic → lidar.py safety monitor
                                    ↓
                          if obstacle < STOP_DIST → motors.stop()
                          if obstacle < SLOW_DIST → motors.slow()

This is INDEPENDENT of Cosmos reasoning — pure reactive safety.
Cosmos still handles mission logic and longer-range decisions.
"""

import logging
import threading
import time

log = logging.getLogger("eric.lidar")

# Safety distances (meters)
STOP_DIST       = 0.30   # stop if anything within 30cm in front arc
SLOW_DIST       = 0.60   # slow if anything within 60cm in front arc
FRONT_ARC_DEG   = 60     # degrees either side of forward = 120° total front arc

_lidar_ok       = False
_obstacle_close = False   # within STOP_DIST
_obstacle_near  = False   # within SLOW_DIST
_min_distance   = 999.0   # minimum distance in front arc
_safety_active  = True    # can be disabled for testing
_node           = None
_sub            = None
_ros_thread     = None

# Raw scan message — exposed for avoidance.py arc distance calculations
# avoidance.py reads this to get per-direction clearances (front/left/right/rear)
_last_scan_msg  = None

# Lock for thread-safe distance reads
_lock = threading.Lock()


def lidar_available() -> bool:
    return _lidar_ok


def obstacle_close() -> bool:
    """True if obstacle within STOP_DIST in front arc."""
    with _lock:
        return _obstacle_close and _safety_active


def obstacle_near() -> bool:
    """True if obstacle within SLOW_DIST in front arc."""
    with _lock:
        return _obstacle_near and _safety_active


def min_front_distance() -> float:
    """Minimum distance (meters) in front arc. Returns 999 if no data."""
    with _lock:
        return _min_distance


def set_safety_active(active: bool):
    """Enable/disable safety stop. Use False only for testing."""
    global _safety_active
    _safety_active = active
    log.info(f"LiDAR safety: {'ENABLED' if active else 'DISABLED'}")


def lidar_void_ahead(
    min_return_ratio: float = 0.15,   # if front arc has fewer valid returns than this → void
    front_arc_deg: int = 40,          # narrow arc for void check (tighter than obstacle arc)
) -> dict:
    """
    Detect floor voids (holes, staircase tops, cliff edges, balcony gaps) using
    the D500 LiDAR scan.

    Key insight: a normal floor fills the front arc with dense, consistent returns
    at 0.3–3m. A void (hole, stair drop, gap) produces almost NO returns in that
    arc because the laser goes straight down through empty air and the return
    either misses the sensor or is beyond the max range.

    This is different from obstacle detection (too many close returns) —
    void detection looks for TOO FEW returns in the forward arc.

    Returns dict:
      {
        "void_detected":  bool,
        "confidence":     "high" | "medium" | "low",
        "return_ratio":   float,   # fraction of arc indices that gave valid returns
        "mean_distance":  float,   # mean of valid returns (m), or None
        "reason":         str,
      }
    """
    with _lock:
        msg = _last_scan_msg

    if msg is None:
        return {"void_detected": False, "confidence": "low",
                "return_ratio": 1.0, "mean_distance": None,
                "reason": "no scan data"}

    import math

    n          = len(msg.ranges)
    angle_inc  = msg.angle_increment
    angle_min  = msg.angle_min
    arc_rad    = math.radians(front_arc_deg)

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

    if return_ratio < min_return_ratio:
        void_detected = True
        confidence = "high" if return_ratio < 0.05 else "medium"
        reason = (f"front arc has only {return_ratio:.0%} valid returns "
                  f"({len(valid_ranges)}/{total_in_arc}) — floor void or drop")
    elif mean_dist is not None and mean_dist > 3.5 and return_ratio < 0.4:
        # Returns exist but very far + sparse → stairwell wall visible, floor gone
        void_detected = True
        confidence = "medium"
        reason = (f"sparse far returns ({mean_dist:.1f}m avg, {return_ratio:.0%} ratio) "
                  "— likely stairwell or open gap")

    return {
        "void_detected": void_detected,
        "confidence":    confidence,
        "return_ratio":  round(return_ratio, 3),
        "mean_distance": round(mean_dist, 2) if mean_dist is not None else None,
        "reason":        reason,
    }


def init_lidar() -> bool:
    """
    Subscribe to /scan topic from D500 LiDAR.
    Returns True if ROS2 available and topic found.
    Non-blocking — runs subscriber in background thread.
    """
    global _lidar_ok, _node, _sub, _ros_thread

    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import LaserScan

        # Reuse existing ROS2 init if nav2 already started it
        if not rclpy.ok():
            rclpy.init()

        # Try to reuse node from nav2.py to save resources
        try:
            from nav2 import _node as nav2_node
            if nav2_node:
                _node = nav2_node
                log.info("LiDAR: reusing Nav2 ROS2 node")
            else:
                _node = rclpy.create_node("eric_lidar")
        except Exception:
            _node = rclpy.create_node("eric_lidar")

        _sub = _node.create_subscription(
            LaserScan,
            "/scan",
            _scan_callback,
            10  # QoS depth
        )

        # Only spin if not already spinning (nav2 may already be spinning the node)
        try:
            from nav2 import _ros_thread as nav2_thread
            if nav2_thread and nav2_thread.is_alive():
                log.info("LiDAR: ROS2 already spinning via Nav2")
                _lidar_ok = True
                return True
        except Exception:
            pass

        _ros_thread = threading.Thread(
            target=lambda: rclpy.spin(_node),
            daemon=True
        )
        _ros_thread.start()

        # Give it a moment to confirm topic
        time.sleep(1.0)
        _lidar_ok = True
        log.info("✅ LiDAR: D500 safety monitor active")
        return True

    except ImportError:
        log.warning("⚠️  ROS2 not found — LiDAR safety monitor disabled")
        _lidar_ok = False
        return False
    except Exception as e:
        log.warning(f"⚠️  LiDAR init failed ({e}) — safety monitor disabled")
        _lidar_ok = False
        return False


def _scan_callback(msg):
    """
    Process LaserScan message.
    D500 outputs 360° scan. We care about front arc only.
    angle_min is usually -π, angle_max is +π, angle_increment is small.
    Index 0 = directly behind on most LiDARs — check your D500 config.

    The raw scan message is stored in _last_scan_msg so avoidance.py can
    read per-direction arc distances (front/left/right/rear) for smart
    manoeuvring decisions. The instant stop/slow still fires here for safety —
    avoidance.py takes over the full manoeuvre from mission.py.
    """
    global _obstacle_close, _obstacle_near, _min_distance, _last_scan_msg

    import math

    try:
        n         = len(msg.ranges)
        angle_inc = msg.angle_increment
        angle_min = msg.angle_min

        # Store raw message for avoidance.py arc distance calculations
        with _lock:
            _last_scan_msg = msg

        # Find indices covering front arc (±FRONT_ARC_DEG degrees)
        arc_rad   = math.radians(FRONT_ARC_DEG)
        front_min_angle = -arc_rad
        front_max_angle = +arc_rad

        front_distances = []
        for i, r in enumerate(msg.ranges):
            angle = angle_min + i * angle_inc
            if front_min_angle <= angle <= front_max_angle:
                if msg.range_min < r < msg.range_max:  # valid reading
                    front_distances.append(r)

        if not front_distances:
            return

        min_dist = min(front_distances)

        with _lock:
            _min_distance   = min_dist
            _obstacle_close = min_dist < STOP_DIST
            _obstacle_near  = min_dist < SLOW_DIST

        # ── Instant safety reaction — independent of Cosmos and avoidance.py ──
        # motors.stop() / motors.slow() fire here immediately (no latency).
        # Full manoeuvring (backup + turn + retry) is handled by avoidance.py
        # which is called from mission.py when it detects _obstacle_close.
        if _safety_active:
            if _obstacle_close:
                from motors import motors
                motors.stop()
                log.warning(f"🚧 LIDAR STOP — obstacle at {min_dist:.2f}m")
                # avoidance.py reads _obstacle_close and runs the full manoeuvre
            elif _obstacle_near:
                from motors import motors
                motors.slow()
                log.info(f"⚠️  LiDAR slow — obstacle at {min_dist:.2f}m")

    except Exception as e:
        log.error(f"LiDAR scan callback error: {e}")


def get_status() -> dict:
    """Return current LiDAR safety status for GUI display."""
    with _lock:
        return {
            "available":      _lidar_ok,
            "safety_active":  _safety_active,
            "obstacle_close": _obstacle_close,
            "obstacle_near":  _obstacle_near,
            "min_distance":   round(_min_distance, 2)
        }


def lidar_status_html() -> str:
    """HTML status display for Gradio."""
    s = get_status()

    if not s["available"]:
        return """
        <div style="background:#1a1a1a;border:1px solid #444;border-radius:8px;
                    padding:10px;font-family:monospace;color:#666">
            📡 LiDAR: not connected
        </div>"""

    color = "#cc0000" if s["obstacle_close"] else \
            "#ff6600" if s["obstacle_near"]  else "#76b900"
    label = "🚧 OBSTACLE CLOSE" if s["obstacle_close"] else \
            "⚠️  OBSTACLE NEAR"  if s["obstacle_near"]  else "✅ CLEAR"
    dist  = s["min_distance"]
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
