"""
ERIC — Wheel Odometry Publisher
Reads encoder feedback from ESP32 over UART and publishes /odom ROS2 topic.
Required by SLAM Toolbox for localisation between LiDAR scans.

ESP32 UART feedback format (Waveshare UGV Beast):
  {"T":1001,"L":speed_l,"R":speed_r}   — wheel speeds in m/s
  {"T":1003,...,"ax":...,"ay":...}      — IMU data (yaw unreliable, ignored)

  NOTE: Run `cat /dev/ttyTHS1` while driving to confirm exact format.
        If your firmware sends different keys, update _parse_feedback() below.
        Fields L/R may also be named "lv"/"rv" or "left"/"right" depending
        on firmware version — adjust KEY_LEFT / KEY_RIGHT constants below.

Architecture:
  ESP32 UART → _uart_reader_loop() → dead-reckoning integration
                                           ↓
                              ROS2 /odom topic (nav_msgs/Odometry)
                                           ↓
                              SLAM Toolbox (map building)
                              Nav2 (path planning)

Dead-reckoning:
  Tracked robot — differential drive kinematics.
  x, y, theta integrated from left/right wheel speeds + time delta.
  No IMU yaw fusion — ESP32 yaw is too noisy to help.
  SLAM Toolbox scan-matching corrects accumulated drift every scan cycle.

Usage:
  init_odom()           → start UART reader + ROS2 publisher
  odom_available()      → True if receiving feedback and publishing
  get_pose()            → {x, y, theta} current dead-reckoning pose
  reset_pose()          → zero pose at mission start
"""

import json
import logging
import math
import threading
import time

log = logging.getLogger("eric.odom")

# ── Robot geometry (Waveshare UGV Beast) ──────────────────────────────────────
# Measure these on your actual robot if SLAM drift is excessive.
WHEEL_BASE_M     = 0.30    # distance between left and right tracks (meters)
                           # UGV Beast: approx 30cm — measure track centre-to-centre

# ── ESP32 UART feedback keys ──────────────────────────────────────────────────
# Adjust these if your firmware uses different field names.
# Run: cat /dev/ttyTHS1  to see raw output and confirm keys.
FEEDBACK_TYPE    = 1001    # T value for speed feedback packet
KEY_LEFT         = "L"     # left wheel speed key
KEY_RIGHT        = "R"     # right wheel speed key

# ── Module state ──────────────────────────────────────────────────────────────
_odom_ok         = False
_x               = 0.0
_y               = 0.0
_theta           = 0.0     # heading in radians
_vx              = 0.0     # current linear velocity
_vtheta          = 0.0     # current angular velocity
_last_time       = 0.0
_lock            = threading.Lock()

_node            = None
_odom_pub        = None
_tf_broadcaster  = None
_ros_thread      = None
_uart_thread     = None


# ─── Public API ───────────────────────────────────────────────────────────────

def odom_available() -> bool:
    """True if odometry is running and publishing."""
    return _odom_ok


def get_pose() -> dict:
    """Return current dead-reckoning pose as {x, y, theta}."""
    with _lock:
        return {"x": round(_x, 3), "y": round(_y, 3), "theta": round(_theta, 4)}


def reset_pose():
    """Zero the odometry pose. Call at mission start for a clean origin."""
    global _x, _y, _theta, _vx, _vtheta
    with _lock:
        _x      = 0.0
        _y      = 0.0
        _theta  = 0.0
        _vx     = 0.0
        _vtheta = 0.0
    log.info("🔄 Odometry pose reset to origin")


# ─── Initialisation ───────────────────────────────────────────────────────────

def init_odom() -> bool:
    """
    Start UART reader and ROS2 odometry publisher.
    Returns True if ROS2 available and UART readable.
    Safe to call once at startup — non-blocking.
    """
    global _odom_ok, _node, _odom_pub, _tf_broadcaster, _ros_thread
    global _uart_thread, _last_time

    try:
        import rclpy
        from nav_msgs.msg import Odometry
        from geometry_msgs.msg import TransformStamped
        from tf2_ros import TransformBroadcaster

        if not rclpy.ok():
            rclpy.init()

        _node = rclpy.create_node("eric_odom")

        _odom_pub       = _node.create_publisher(Odometry, "/odom", 10)
        _tf_broadcaster = TransformBroadcaster(_node)

        # Start ROS2 executor in background thread
        import rclpy.executors
        _executor = rclpy.executors.SingleThreadedExecutor()
        _executor.add_node(_node)
        _ros_thread = threading.Thread(
            target=_executor.spin,
            daemon=True,
            name="odom-ros-spin"
        )
        _ros_thread.start()

        _last_time = time.monotonic()

        # Start UART reader thread
        _uart_thread = threading.Thread(
            target=_uart_reader_loop,
            daemon=True,
            name="odom-uart-reader"
        )
        _uart_thread.start()

        _odom_ok = True
        log.info("✅ Odometry: UART reader + /odom publisher active")
        return True

    except ImportError:
        log.warning("⚠️  ROS2 not found — odometry disabled (SLAM unavailable)")
        _odom_ok = False
        return False
    except Exception as e:
        log.warning(f"⚠️  Odom init failed ({e}) — SLAM localisation degraded")
        _odom_ok = False
        return False


# ─── UART reader ──────────────────────────────────────────────────────────────

def _uart_reader_loop():
    """
    Read JSON feedback from ESP32 over UART.
    Parses wheel speed packets and integrates pose.
    Publishes /odom and broadcasts map→odom TF on every update.

    If UART is unavailable, falls back to reading from motors._current_left/right
    which are updated by every motors._send() call — this gives approximate
    odometry without needing ESP32 feedback, at the cost of no slip detection.
    """
    log.info("Odom UART reader started")

    while True:
        try:
            from motors import motors as _m
            if _m._ser and _m._ser.is_open:
                _read_from_uart(_m._ser)
            else:
                # Fallback: integrate from commanded speeds
                _integrate_from_commanded()
                time.sleep(0.05)   # 20Hz when using fallback
            # Always publish at steady rate — SLAM needs continuous /odom
            with _lock:
                x, y, theta, vx, vtheta = _x, _y, _theta, _vx, _vtheta
            _publish_odom(x, y, theta, vx, vtheta)
        except Exception as e:
            log.debug(f"Odom UART loop error: {e}")
            time.sleep(0.1)


def _read_from_uart(ser):
    """Read one line from UART only if data is waiting — never blocks."""
    try:
        if ser.in_waiting == 0:
            time.sleep(0.02)   # 50Hz polling when idle
            return
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            return
        data = json.loads(line)
        if data.get("T") == FEEDBACK_TYPE:
            speed_l = float(data.get(KEY_LEFT, 0.0))
            speed_r = float(data.get(KEY_RIGHT, 0.0))
            _integrate_speeds(speed_l, speed_r)
    except (json.JSONDecodeError, KeyError, ValueError):
        pass   # non-JSON lines (debug output etc) — ignore silently
    except Exception as e:
        log.debug(f"UART read error: {e}")


def _integrate_from_commanded():
    """
    Fallback odometry using last commanded motor speeds.
    Less accurate than encoder feedback but better than nothing.
    motors._send() updates _current_left/_current_right on every command.
    Note: UGV Beast uses negative = forward convention, so negate here.
    """
    try:
        from motors import motors as _m
        # Negate: negative motor value = forward on UGV Beast
        speed_l = -_m._current_left
        speed_r = -_m._current_right
        _integrate_speeds(speed_l, speed_r)
    except Exception:
        pass


def _integrate_speeds(speed_l: float, speed_r: float):
    """
    Differential drive dead-reckoning integration.
    Updates x, y, theta from left/right wheel speeds and elapsed time.
    Publishes updated /odom message and broadcasts base_link TF.

    Kinematics:
      v     = (speed_r + speed_l) / 2          linear velocity
      omega = (speed_r - speed_l) / WHEEL_BASE  angular velocity
      theta += omega * dt
      x     += v * cos(theta) * dt
      y     += v * sin(theta) * dt
    """
    global _x, _y, _theta, _vx, _vtheta, _last_time

    now = time.monotonic()
    with _lock:
        dt         = now - _last_time
        _last_time = now

    if dt <= 0 or dt > 1.0:
        # Skip if time delta is invalid or too large (startup / pause)
        return

    v     = (speed_r + speed_l) / 2.0
    omega = (speed_r - speed_l) / WHEEL_BASE_M

    with _lock:
        _theta  += omega * dt
        _x      += v * math.cos(_theta) * dt
        _y      += v * math.sin(_theta) * dt
        _vx      = v
        _vtheta  = omega
        x, y, theta, vx, vtheta = _x, _y, _theta, _vx, _vtheta

    _publish_odom(x, y, theta, vx, vtheta)


# ─── ROS2 publishing ──────────────────────────────────────────────────────────

def _publish_odom(x: float, y: float, theta: float,
                  vx: float, vtheta: float):
    """
    Publish nav_msgs/Odometry on /odom and broadcast odom→base_link TF.
    Both are required by SLAM Toolbox and Nav2.
    """
    if not _odom_ok or _node is None or _odom_pub is None:
        return

    try:
        import rclpy.time
        from nav_msgs.msg import Odometry
        from geometry_msgs.msg import TransformStamped, Quaternion
        from std_msgs.msg import Header

        now_msg = _node.get_clock().now().to_msg()

        # Yaw → quaternion (z, w only for 2D)
        qz = math.sin(theta / 2.0)
        qw = math.cos(theta / 2.0)

        # ── Odometry message ──────────────────────────────────────────────────
        odom              = Odometry()
        odom.header.stamp = now_msg
        odom.header.frame_id          = "odom"
        odom.child_frame_id           = "base_link"

        odom.pose.pose.position.x     = x
        odom.pose.pose.position.y     = y
        odom.pose.pose.position.z     = 0.0
        odom.pose.pose.orientation.x  = 0.0
        odom.pose.pose.orientation.y  = 0.0
        odom.pose.pose.orientation.z  = qz
        odom.pose.pose.orientation.w  = qw

        odom.twist.twist.linear.x     = vx
        odom.twist.twist.angular.z    = vtheta

        # Covariance — moderate uncertainty since we're using dead-reckoning
        # Diagonal elements: x, y, z, roll, pitch, yaw
        odom.pose.covariance[0]  = 0.05   # x
        odom.pose.covariance[7]  = 0.05   # y
        odom.pose.covariance[35] = 0.1    # yaw — higher uncertainty
        odom.twist.covariance[0]  = 0.05
        odom.twist.covariance[35] = 0.1

        _odom_pub.publish(odom)

        # ── TF: odom → base_link ──────────────────────────────────────────────
        if _tf_broadcaster:
            t                             = TransformStamped()
            t.header.stamp                = now_msg
            t.header.frame_id             = "odom"
            t.child_frame_id              = "base_link"
            t.transform.translation.x     = x
            t.transform.translation.y     = y
            t.transform.translation.z     = 0.0
            t.transform.rotation.x        = 0.0
            t.transform.rotation.y        = 0.0
            t.transform.rotation.z        = qz
            t.transform.rotation.w        = qw
            _tf_broadcaster.sendTransform(t)

    except Exception as e:
        log.debug(f"Odom publish error: {e}")


# ─── Status ───────────────────────────────────────────────────────────────────

def get_status() -> dict:
    pose = get_pose()
    return {
        "available": _odom_ok,
        "x":         pose["x"],
        "y":         pose["y"],
        "theta_deg": round(math.degrees(pose["theta"]), 1),
    }