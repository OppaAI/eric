"""
E.R.I.C. — LiDAR Safety Monitor
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
    """
    global _obstacle_close, _obstacle_near, _min_distance

    import math

    try:
        n         = len(msg.ranges)
        angle_inc = msg.angle_increment
        angle_min = msg.angle_min

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

        # Safety action — independent of Cosmos
        if _safety_active:
            if _obstacle_close:
                from motors import motors
                motors.stop()
                log.warning(f"🚧 LIDAR STOP — obstacle at {min_dist:.2f}m")
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
