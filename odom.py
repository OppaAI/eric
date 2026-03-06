"""
ERIC — Wheel Odometry Publisher
Reads encoder feedback from ESP32 over UART and publishes /odom ROS2 topic.
Required by SLAM Toolbox for localisation between LiDAR scans.

ESP32 UART feedback format (Waveshare UGV Beast):
  {"T":1001,"L":speed_l,"R":speed_r}   — wheel speeds in m/s

  NOTE: Run `cat /dev/ttyTHS1` while driving to confirm exact format.
        Fields L/R may also be named "lv"/"rv" depending on firmware.
        Adjust KEY_LEFT / KEY_RIGHT constants below if needed.

UART sharing:
  This module no longer opens /dev/ttyTHS1 directly. Instead it registers
  a subscriber queue with motors.subscribe_uart(1001, queue), so all
  modules share the single port owned by motors.py. Battery packets,
  odom packets, and future IMU packets are routed without byte theft.

ROS2 node sharing:
  Uses ros_core.get_node() — the single shared 'eric_robot' node.
  No longer creates its own node or executor. ensure_spinning() is called
  once after subscribing, which is a no-op if already spinning.

Dead-reckoning:
  Tracked robot — differential drive kinematics.
  x, y, theta integrated from left/right wheel speeds + time delta.
  No IMU yaw fusion — ESP32 yaw is too noisy.
  SLAM Toolbox scan-matching corrects accumulated drift every scan cycle.

Odometry sign convention:
  motors._send() stores raw commanded values — negative = forward on UGV Beast.
  ODOM_SIGN = -1 negates before integrating so positive = forward in odom frame.
  If SLAM map drifts backwards on straight runs, flip ODOM_SIGN to +1.

Usage:
  init_odom()           → start UART subscriber + ROS2 publisher
  odom_available()      → True if receiving feedback and publishing
  get_pose()            → {x, y, theta} current dead-reckoning pose
  reset_pose()          → zero pose at mission start
"""

import json
import logging
import math
import queue
import threading
import time

log = logging.getLogger("eric.odom")

# ── Robot geometry (Waveshare UGV Beast) ──────────────────────────────────────
WHEEL_BASE_M  = 0.30   # distance between left and right tracks (metres)
                       # Measure track centre-to-centre on your robot.
                       # Excessive SLAM drift on turns = wrong WHEEL_BASE_M.

# ── ESP32 UART feedback keys ──────────────────────────────────────────────────
FEEDBACK_TYPE = 1001   # T value for speed feedback packet
KEY_LEFT      = "L"    # left wheel speed key  — change to "lv" if needed
KEY_RIGHT     = "R"    # right wheel speed key — change to "rv" if needed

# ── Sign convention ───────────────────────────────────────────────────────────
# motors._send() stores raw commanded values where negative = forward.
# Negate here so the odom frame uses positive = forward convention.
# If SLAM builds a mirrored or reversed map on straight runs, flip to +1.
ODOM_SIGN = -1

# ── Module state ──────────────────────────────────────────────────────────────
_odom_ok        = False
_x              = 0.0
_y              = 0.0
_theta          = 0.0
_vx             = 0.0
_vtheta         = 0.0
_last_time      = 0.0
_lock           = threading.Lock()

_odom_pub       = None
_tf_broadcaster = None
_uart_thread    = None
_uart_queue     = queue.Queue(maxsize=50)   # packets routed from motors.py


# ─── Public API ───────────────────────────────────────────────────────────────

def odom_available() -> bool:
    return _odom_ok


def get_pose() -> dict:
    """Return current dead-reckoning pose as {x, y, theta}."""
    with _lock:
        return {"x": round(_x, 3), "y": round(_y, 3), "theta": round(_theta, 4)}


def reset_pose():
    """Zero the odometry pose. Call at mission start for a clean origin."""
    global _x, _y, _theta, _vx, _vtheta
    with _lock:
        _x = _y = _theta = _vx = _vtheta = 0.0
    log.info("🔄 Odometry pose reset to origin")


# ─── Initialisation ───────────────────────────────────────────────────────────

def init_odom() -> bool:
    """
    Start UART subscriber and ROS2 odometry publisher.
    Must be called before init_slam() and init_nav2().
    Returns True if ROS2 available and UART subscriber registered.
    Non-blocking — all work happens in background threads.
    """
    global _odom_ok, _odom_pub, _tf_broadcaster, _uart_thread, _last_time

    # ── Subscribe to UART packets via motors router ────────────────────────────
    try:
        from motors import motors as _m
        _m.subscribe_uart(FEEDBACK_TYPE, _uart_queue)
        log.info(f"Odom: subscribed to UART T={FEEDBACK_TYPE} via motors router")
    except Exception as e:
        log.warning(f"⚠️  Odom: could not subscribe to UART router ({e}) "
                    "— falling back to commanded-speed integration")

    # ── ROS2 publisher ─────────────────────────────────────────────────────────
    try:
        from ros_core import get_node, ensure_spinning
        from nav_msgs.msg import Odometry
        from tf2_ros import TransformBroadcaster

        node = get_node()
        if node is None:
            raise RuntimeError("ROS2 not available")

        _odom_pub       = node.create_publisher(Odometry, "/odom", 10)
        _tf_broadcaster = TransformBroadcaster(node)
        ensure_spinning()

        _last_time = time.monotonic()

        # Start UART consumer thread
        _uart_thread = threading.Thread(
            target=_uart_consumer_loop,
            daemon=True,
            name="odom-uart-consumer"
        )
        _uart_thread.start()

        _odom_ok = True
        log.info("✅ Odometry: /odom publisher active")
        return True

    except ImportError:
        log.warning("⚠️  ROS2 not found — odometry disabled (SLAM unavailable)")
        _odom_ok = False
        return False
    except Exception as e:
        log.warning(f"⚠️  Odom init failed ({e}) — SLAM localisation degraded")
        _odom_ok = False
        return False


# ─── UART consumer ────────────────────────────────────────────────────────────

def _uart_consumer_loop():
    """
    Drain the UART queue populated by motors.py router.
    On queue timeout: fall back to integrating from commanded speeds.
    Publishes /odom at steady rate regardless of source.
    """
    log.info("Odom UART consumer started")
    fallback_interval = 0.05   # 20 Hz fallback when no UART packets

    while True:
        try:
            # Block up to fallback_interval — then use commanded speeds
            data = _uart_queue.get(timeout=fallback_interval)
            speed_l = float(data.get(KEY_LEFT,  0.0)) * ODOM_SIGN
            speed_r = float(data.get(KEY_RIGHT, 0.0)) * ODOM_SIGN
            _integrate_speeds(speed_l, speed_r)
        except queue.Empty:
            # No UART feedback — integrate from last commanded speeds
            _integrate_from_commanded()
        except Exception as e:
            log.debug(f"Odom consumer error: {e}")
            time.sleep(0.1)


def _integrate_from_commanded():
    """
    Fallback: integrate using last commanded motor speeds.
    Less accurate than encoder feedback (no slip detection) but better
    than no odometry at all. ODOM_SIGN applied here too.
    """
    try:
        from motors import motors as _m
        speed_l = _m._current_left  * ODOM_SIGN
        speed_r = _m._current_right * ODOM_SIGN
        _integrate_speeds(speed_l, speed_r)
    except Exception:
        pass


def _integrate_speeds(speed_l: float, speed_r: float):
    """
    Differential drive dead-reckoning integration.
    Updates x, y, theta from left/right wheel speeds and elapsed time.
    Publishes /odom and broadcasts odom→base_link TF on every call.

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
        return   # invalid delta — startup gap or long pause

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
    Both required by SLAM Toolbox and Nav2.
    """
    if not _odom_ok or _odom_pub is None:
        return

    try:
        from ros_core import get_node
        from nav_msgs.msg import Odometry
        from geometry_msgs.msg import TransformStamped

        node    = get_node()
        now_msg = node.get_clock().now().to_msg()
        qz      = math.sin(theta / 2.0)
        qw      = math.cos(theta / 2.0)

        # ── Odometry message ──────────────────────────────────────────────────
        odom                          = Odometry()
        odom.header.stamp             = now_msg
        odom.header.frame_id          = "odom"
        odom.child_frame_id           = "base_link"
        odom.pose.pose.position.x     = x
        odom.pose.pose.position.y     = y
        odom.pose.pose.position.z     = 0.0
        odom.pose.pose.orientation.z  = qz
        odom.pose.pose.orientation.w  = qw
        odom.twist.twist.linear.x     = vx
        odom.twist.twist.angular.z    = vtheta

        # Covariance — moderate uncertainty for dead-reckoning
        odom.pose.covariance[0]   = 0.05   # x
        odom.pose.covariance[7]   = 0.05   # y
        odom.pose.covariance[35]  = 0.1    # yaw — higher uncertainty
        odom.twist.covariance[0]  = 0.05
        odom.twist.covariance[35] = 0.1

        _odom_pub.publish(odom)

        # ── TF: odom → base_link ──────────────────────────────────────────────
        if _tf_broadcaster:
            t                         = TransformStamped()
            t.header.stamp            = now_msg
            t.header.frame_id         = "odom"
            t.child_frame_id          = "base_link"
            t.transform.translation.x = x
            t.transform.translation.y = y
            t.transform.translation.z = 0.0
            t.transform.rotation.z    = qz
            t.transform.rotation.w    = qw
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

# ─── ROS2 OdomNode wrapper ────────────────────────────────────────────────────
# odom.py already publishes /odom and TF via init_odom().
# This wrapper adds a lifecycle-style start/stop API consistent
# with the other nodes, and adds /odom/pose status topic.
#
# Topics added:
#   Publish    /odom/pose_simple   std_msgs/String  — JSON {x, y, theta_deg}
#   Subscribe  /odom/reset         std_msgs/Empty   — reset pose to origin
#
# The existing /odom and TF broadcast from init_odom() are UNCHANGED.
#
# Usage (from main.py):
#   from odom import init_odom, start_odom_node
#   init_odom()        # existing — starts UART subscriber + /odom publisher
#   start_odom_node()  # new — starts pose/reset topic interfaces

import threading as _odom_node_threading

_odom_node_inst   = None
_odom_node_thread = None
_odom_node_lock   = _odom_node_threading.Lock()


class OdomNode:
    """ROS2 node lifecycle wrapper for odom.py. Adds pose/reset topics."""

    def __init__(self):
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String, Empty

        rclpy.init(args=None)
        self._node = Node("eric_odom_node")

        # Publish simplified pose — useful for Telegram/GUI display
        self._pose_pub = self._node.create_publisher(String, "/odom/pose_simple", 10)

        # Subscribe /odom/reset — zero pose from any node
        self._reset_sub = self._node.create_subscription(
            Empty, "/odom/reset", self._on_reset, 10
        )

        # Publish at 5 Hz
        self._timer = self._node.create_timer(0.2, self._publish_pose)

        log.info("OdomNode: ROS2 pose/reset topics active")

    def _on_reset(self, msg):
        reset_pose()
        log.info("OdomNode: pose reset via /odom/reset topic")

    def _publish_pose(self):
        from std_msgs.msg import String
        import json as _j
        pose = get_pose()
        pose["theta_deg"] = round(math.degrees(pose["theta"]), 1)
        self._pose_pub.publish(String(data=_j.dumps(pose)))

    def spin(self):
        import rclpy
        try:
            rclpy.spin(self._node)
        except Exception as e:
            log.debug(f"OdomNode spin ended: {e}")
        finally:
            self._node.destroy_node()
            try:
                rclpy.shutdown()
            except Exception:
                pass

    def destroy(self):
        try:
            self._node.destroy_node()
        except Exception:
            pass


def start_odom_node() -> bool:
    """Launch OdomNode in a background daemon thread."""
    global _odom_node_inst, _odom_node_thread
    with _odom_node_lock:
        if _odom_node_inst is not None:
            return True
        try:
            _odom_node_inst = OdomNode()
            _odom_node_thread = _odom_node_threading.Thread(
                target=_odom_node_inst.spin,
                daemon=True,
                name="odom-node-spin"
            )
            _odom_node_thread.start()
            log.info("OdomNode: spinning in background thread")
            return True
        except Exception as e:
            log.error(f"OdomNode: failed to start — {e}")
            _odom_node_inst = None
            return False


def stop_odom_node():
    global _odom_node_inst
    with _odom_node_lock:
        if _odom_node_inst:
            _odom_node_inst.destroy()
            _odom_node_inst = None
