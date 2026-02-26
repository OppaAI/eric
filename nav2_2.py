"""
ERIC — ROS2 Nav2 Integration  (SLAM version)
Sends navigation goals to Nav2 stack with live SLAM map support.
Falls back gracefully if ROS2 not running.

SLAM changes (vs nav2.py):
  - init_nav2() waits for /map topic before declaring Nav2 ready
  - send_goal() uses map frame from SLAM Toolbox (not dead-reckoning)
  - navigate_to_person() uses map-aware goals with costmap clearance check
  - costmap_clear() helper — clears Nav2 costmap when SLAM map updates
  - map_ready() — True once SLAM has built enough map to plan paths
  - Startup sequence: init_odom() → init_slam() → init_nav2()

Architecture:
  odom.py    → /odom          ──┐
  lidar_2.py → /scan          ──┼──→ SLAM Toolbox → /map
  oakd_2.py  → /oakd/depth    ──┘         ↓
                                     Nav2 costmap
                                          ↓
                               nav2_2.py send_goal()
                                          ↓
                                    motors (via Nav2 cmd_vel)

Usage:
  nav2_available()          → check if ROS2/Nav2 is running
  map_ready()               → True once SLAM map is usable for planning
  send_goal(x, y, yaw)     → navigate to map coordinates
  cancel_goal()            → abort navigation
  get_pose()               → current robot pose on map
  is_navigating()          → True while Nav2 is executing a goal
  costmap_clear()          → clear costmap after SLAM map update
"""

import logging
import math
import threading
import time

log = logging.getLogger("eric.nav2")

_nav2_ok        = False   # True once ROS2 + Nav2 confirmed running
_map_ok         = False   # True once /map has been received from SLAM
_navigating     = False
_current_goal   = None
_pose           = {"x": 0.0, "y": 0.0, "yaw": 0.0}

# ROS2 objects (lazy-initialized)
_node           = None
_executor       = None
_nav_client     = None
_tf_buffer      = None
_ros_thread     = None
_map_sub        = None
_shutdown_event = threading.Event()

# Map readiness — need minimum coverage before planning is reliable
MAP_MIN_CELLS   = 100    # minimum occupied cells before we trust the map


def nav2_available() -> bool:
    return _nav2_ok


def map_ready() -> bool:
    """
    True when SLAM has built enough map for Nav2 path planning.
    mission.py should check this before sending goals — planning on an
    empty map produces poor paths.
    """
    return _nav2_ok and _map_ok


def is_navigating() -> bool:
    return _navigating


def get_pose() -> dict:
    """Return current robot pose as {x, y, yaw} in map frame."""
    _refresh_pose()
    return dict(_pose)


def _spin_thread_fn():
    try:
        _executor.spin()
    except Exception as e:
        log.debug(f"ROS2 spin exited: {e}")
    finally:
        _shutdown_event.set()


def init_nav2() -> bool:
    """
    Initialize ROS2 node and Nav2 action client.
    Also subscribes to /map to track when SLAM has built usable coverage.

    Call AFTER init_odom() and init_slam() — Nav2 needs both /odom and /map
    to be publishing before it can accept goals.

    Returns True if Nav2 action server found within timeout.
    """
    global _nav2_ok, _map_ok, _node, _executor, _nav_client
    global _tf_buffer, _ros_thread, _map_sub

    try:
        import rclpy
        import rclpy.executors
        from rclpy.node import Node
        from nav2_msgs.action import NavigateToPose
        from rclpy.action import ActionClient
        from tf2_ros import Buffer, TransformListener
        from nav_msgs.msg import OccupancyGrid

        if not rclpy.ok():
            rclpy.init()

        # Reuse odom node if available — all modules share one node
        try:
            from odom import _node as odom_node
            if odom_node:
                _node = odom_node
                log.info("Nav2: reusing odom ROS2 node")
            else:
                raise Exception("odom node not ready")
        except Exception:
            _node = rclpy.create_node("eric_nav2")

        _executor = rclpy.executors.SingleThreadedExecutor()
        _executor.add_node(_node)

        _tf_buffer  = Buffer()
        _           = TransformListener(_tf_buffer, _node)
        _nav_client = ActionClient(_node, NavigateToPose, "navigate_to_pose")

        # Subscribe to /map — track when SLAM has usable coverage
        _map_sub = _node.create_subscription(
            OccupancyGrid,
            "/map",
            _map_callback,
            10
        )

        # Start spin thread
        _ros_thread = threading.Thread(
            target=_spin_thread_fn,
            daemon=True,
            name="rclpy-spin"
        )
        _ros_thread.start()

        # Wait for Nav2 action server
        log.info("⏳ Waiting for Nav2 action server...")
        if _nav_client.wait_for_server(timeout_sec=5.0):
            _nav2_ok = True
            log.info("✅ Nav2 connected — autonomous navigation enabled")
        else:
            log.warning("⚠️  Nav2 action server not found — falling back to direct motor control")
            _nav2_ok = False

        return _nav2_ok

    except ImportError:
        log.warning("⚠️  ROS2 not found — falling back to direct motor control")
        _nav2_ok = False
        return False
    except Exception as e:
        log.warning(f"⚠️  Nav2 init failed ({e}) — falling back to direct motor control")
        _nav2_ok = False
        return False


def _map_callback(msg):
    """
    Called when SLAM Toolbox publishes a new /map.
    Counts known cells (0=free, 100=occupied) to determine if map is
    large enough for reliable path planning.
    """
    global _map_ok
    try:
        known = sum(1 for c in msg.data if c >= 0)   # -1 = unknown
        if known >= MAP_MIN_CELLS and not _map_ok:
            _map_ok = True
            w = round(msg.info.width  * msg.info.resolution, 1)
            h = round(msg.info.height * msg.info.resolution, 1)
            log.info(f"🗺️  Map ready for Nav2 planning — {w}m × {h}m ({known} known cells)")
    except Exception as e:
        log.debug(f"map_callback error: {e}")


def send_goal(x: float, y: float, yaw: float = 0.0,
              on_complete=None, on_fail=None) -> bool:
    """
    Send a navigation goal to Nav2.
    x, y: target position in map frame (meters, from SLAM origin)
    yaw: target heading (radians, 0 = forward)
    on_complete: optional callback when goal reached
    on_fail: optional callback if goal fails

    Checks map_ready() before sending — won't plan on empty SLAM map.
    Returns True if goal was accepted.
    """
    global _navigating, _current_goal

    if not _nav2_ok:
        log.warning("Nav2 not available — cannot send goal")
        return False

    if not _map_ok:
        log.warning("SLAM map not ready — Nav2 goal deferred (map building)")
        return False

    try:
        from nav2_msgs.action import NavigateToPose
        from geometry_msgs.msg import PoseStamped
        from std_msgs.msg import Header

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp    = _node.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.position.z = 0.0
        goal_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

        _navigating = True
        log.info(f"🧭 Nav2 goal → ({x:.2f}, {y:.2f}, yaw={math.degrees(yaw):.0f}°)")

        future = _nav_client.send_goal_async(goal_msg)
        future.add_done_callback(
            lambda f: _on_goal_response(f, on_complete, on_fail)
        )
        return True

    except Exception as e:
        log.error(f"send_goal error: {e}")
        _navigating = False
        return False


def _on_goal_response(future, on_complete, on_fail):
    global _navigating, _current_goal
    try:
        goal_handle = future.result()
    except Exception as e:
        log.error(f"Nav2 goal response error: {e}")
        _navigating = False
        if on_fail:
            on_fail()
        return

    if not goal_handle.accepted:
        log.warning("Nav2 goal rejected")
        _navigating = False
        if on_fail:
            on_fail()
        return

    _current_goal = goal_handle
    log.info("Nav2 goal accepted — navigating...")

    result_future = goal_handle.get_result_async()
    result_future.add_done_callback(
        lambda f: _on_goal_result(f, on_complete, on_fail)
    )


def _on_goal_result(future, on_complete, on_fail):
    global _navigating, _current_goal
    _navigating   = False
    _current_goal = None

    try:
        status = future.result().status
        from action_msgs.msg import GoalStatus
        if status == GoalStatus.STATUS_SUCCEEDED:
            log.info("✅ Nav2 goal reached")
            if on_complete:
                on_complete()
        else:
            log.warning(f"Nav2 goal failed — status {status}")
            if on_fail:
                on_fail()
    except Exception as e:
        log.error(f"Nav2 result error: {e}")
        if on_fail:
            on_fail()


def cancel_goal():
    """Cancel any active Nav2 navigation goal."""
    global _navigating, _current_goal
    if _current_goal:
        try:
            _current_goal.cancel_goal_async()
            log.info("Nav2 goal cancelled")
        except Exception as e:
            log.debug(f"cancel_goal error: {e}")
    _navigating   = False
    _current_goal = None


def costmap_clear():
    """
    Clear Nav2 costmaps — call after SLAM map significantly updates
    so Nav2 replans with fresh obstacle data rather than stale costmap.
    """
    try:
        import subprocess
        for costmap in ("global_costmap", "local_costmap"):
            subprocess.run(
                ["ros2", "service", "call",
                 f"/nav2/{costmap}/clear_entirely_costmap",
                 "nav2_msgs/srv/ClearEntireCostmap", "{}"],
                capture_output=True, timeout=3.0
            )
        log.debug("Nav2 costmaps cleared")
    except Exception as e:
        log.debug(f"costmap_clear error: {e}")


def _refresh_pose():
    """Update _pose from TF map→base_link transform (provided by SLAM Toolbox)."""
    global _pose
    if not _nav2_ok or not _tf_buffer:
        return
    try:
        import rclpy.time
        t = _tf_buffer.lookup_transform(
            "map", "base_link", rclpy.time.Time()
        )
        _pose["x"] = t.transform.translation.x
        _pose["y"] = t.transform.translation.y
        q = t.transform.rotation
        _pose["yaw"] = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
    except Exception:
        pass


def navigate_to_person(direction: str = "front") -> bool:
    """
    High-level: navigate toward a detected person using SLAM map pose.
    Uses map-frame coordinates from SLAM (not dead-reckoning offsets).

    Waits for map_ready() before attempting — won't send goal on empty map.
    direction: "front" | "left" | "right" | "behind"
    """
    if not _nav2_ok:
        return False

    if not _map_ok:
        log.info("navigate_to_person: map not ready yet — using direct approach")
        return False

    pose = get_pose()
    yaw  = pose["yaw"]

    # Offset distances tuned for SAR approach — stop 1m from person
    dir_offsets = {
        "front":  (1.0,  0.0,  0.0),
        "left":   (0.5,  0.7,  math.pi / 2),
        "right":  (0.5, -0.7, -math.pi / 2),
        "behind": (-1.0, 0.0,  math.pi),
    }
    dx, dy, dyaw = dir_offsets.get(direction, (1.0, 0.0, 0.0))

    target_x   = pose["x"] + dx * math.cos(yaw) - dy * math.sin(yaw)
    target_y   = pose["y"] + dx * math.sin(yaw) + dy * math.cos(yaw)
    target_yaw = yaw + dyaw

    log.info(f"🧭 Navigating toward person ({direction}) → "
             f"({target_x:.2f}, {target_y:.2f})")
    return send_goal(target_x, target_y, target_yaw)


def shutdown():
    """
    Clean shutdown of ROS2 node.
    Call from main.py via atexit / signal handlers.
    """
    global _nav2_ok, _map_ok, _node, _executor

    log.info("Nav2 shutting down...")
    _nav2_ok = False
    _map_ok  = False

    cancel_goal()

    if _executor is not None:
        try:
            _executor.shutdown(timeout_sec=2.0)
        except Exception as e:
            log.debug(f"executor.shutdown error: {e}")

    if _ros_thread is not None and _ros_thread.is_alive():
        _ros_thread.join(timeout=3.0)
        if _ros_thread.is_alive():
            log.warning("ROS2 spin thread did not exit cleanly within 3 s")

    try:
        import rclpy
        if _node is not None:
            _node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    except Exception as e:
        log.debug(f"rclpy.shutdown error: {e}")
    finally:
        _node     = None
        _executor = None

    log.info("Nav2 shutdown complete")


def get_status() -> dict:
    pose = get_pose()
    return {
        "available":   _nav2_ok,
        "map_ready":   _map_ok,
        "navigating":  _navigating,
        "x":           round(pose["x"], 2),
        "y":           round(pose["y"], 2),
        "yaw_deg":     round(math.degrees(pose["yaw"]), 1),
    }
