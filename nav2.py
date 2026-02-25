"""
ERIC — ROS2 Nav2 Integration
Sends navigation goals to Nav2 stack.
Falls back gracefully if ROS2 not running.

Architecture:
  Cosmos decides WHERE to go (mission reasoning)
  Nav2 decides HOW to get there safely (path planning + obstacle avoidance)
  LiDAR (D500) + OAK-D Lite feed Nav2 costmap continuously

Usage:
  nav2_available()          → check if ROS2/Nav2 is running
  send_goal(x, y, yaw)     → navigate to map coordinates
  cancel_goal()            → abort navigation
  get_pose()               → current robot pose on map
  is_navigating()          → True while Nav2 is executing a goal
"""

import logging
import threading
import time

log = logging.getLogger("eric.nav2")

_nav2_ok        = False   # True once ROS2 + Nav2 confirmed running
_navigating     = False
_current_goal   = None
_pose           = {"x": 0.0, "y": 0.0, "yaw": 0.0}

# ROS2 objects (lazy-initialized)
_node           = None
_executor       = None   # SingleThreadedExecutor — safer for clean shutdown
_nav_client     = None
_tf_buffer      = None
_ros_thread     = None
_shutdown_event = threading.Event()   # signals spin thread to stop


def nav2_available() -> bool:
    return _nav2_ok


def is_navigating() -> bool:
    return _navigating


def get_pose() -> dict:
    """Return current robot pose as {x, y, yaw} in map frame."""
    _refresh_pose()
    return dict(_pose)


def _spin_thread_fn():
    """
    Background spin thread.

    Uses SingleThreadedExecutor.spin() instead of rclpy.spin() so we can
    call executor.shutdown() cleanly from the main thread before Python tears
    down daemon threads.  Without this, the C++ layer inside rclpy raises an
    exception during pthread_exit and std::terminate() fires.
    """
    try:
        _executor.spin()
    except Exception as e:
        # Normal during shutdown — executor.shutdown() wakes spin() with an
        # internal exception.  Log at debug so it doesn't pollute the console.
        log.debug(f"ROS2 spin exited: {e}")
    finally:
        _shutdown_event.set()


def init_nav2() -> bool:
    """
    Initialize ROS2 node and Nav2 action client.
    Returns True if successful, False if ROS2 not available.
    Called once at startup — non-blocking, runs ROS spin in background thread.
    """
    global _nav2_ok, _node, _executor, _nav_client, _tf_buffer, _ros_thread

    try:
        import rclpy
        import rclpy.executors
        from rclpy.node import Node
        from nav2_msgs.action import NavigateToPose
        from rclpy.action import ActionClient
        from tf2_ros import Buffer, TransformListener

        if not rclpy.ok():
            rclpy.init()

        _node      = rclpy.create_node("eric_nav2")
        _executor  = rclpy.executors.SingleThreadedExecutor()
        _executor.add_node(_node)

        _tf_buffer = Buffer()
        _          = TransformListener(_tf_buffer, _node)
        _nav_client = ActionClient(_node, NavigateToPose, "navigate_to_pose")

        # Start the spin thread — daemon=True so it doesn't block process exit,
        # but shutdown() will cleanly stop it before the interpreter tears it down.
        _ros_thread = threading.Thread(
            target=_spin_thread_fn,
            daemon=True,
            name="rclpy-spin"
        )
        _ros_thread.start()

        # Wait up to 5 s for Nav2 action server
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


def send_goal(x: float, y: float, yaw: float = 0.0,
              on_complete=None, on_fail=None) -> bool:
    """
    Send a navigation goal to Nav2.
    x, y: target position in map frame (meters)
    yaw: target heading (radians, 0 = forward)
    on_complete: optional callback when goal reached
    on_fail: optional callback if goal fails
    Returns True if goal was accepted.
    """
    global _navigating, _current_goal

    if not _nav2_ok:
        log.warning("Nav2 not available — cannot send goal")
        return False

    try:
        import math
        from nav2_msgs.action import NavigateToPose
        from geometry_msgs.msg import PoseStamped
        from std_msgs.msg import Header

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header = Header()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = _node.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.position.z = 0.0

        # Convert yaw to quaternion
        goal_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

        _navigating = True
        log.info(f"🧭 Nav2 goal → ({x:.2f}, {y:.2f}, yaw={yaw:.2f})")

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
            log.debug(f"cancel_goal error (may be harmless): {e}")
    _navigating   = False
    _current_goal = None


def _refresh_pose():
    """Update _pose from TF map→base_link transform."""
    global _pose
    if not _nav2_ok or not _tf_buffer:
        return
    try:
        import math
        import rclpy.time  # explicit import — avoids NameError in module scope
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
        pass   # TF not ready yet — use last known pose


def navigate_to_person(direction: str = "front") -> bool:
    """
    High-level: navigate toward a person detected by Cosmos.
    direction: "front" | "left" | "right" | "behind"
    Uses relative goal offset from current pose.
    """
    if not _nav2_ok:
        return False

    import math
    pose = get_pose()
    yaw  = pose["yaw"]

    dir_offsets = {
        "front":  (1.5,  0.0,  0.0),
        "left":   (0.5,  0.8,  math.pi / 2),
        "right":  (0.5, -0.8, -math.pi / 2),
        "behind": (-1.0, 0.0,  math.pi),
    }
    dx, dy, dyaw = dir_offsets.get(direction, (1.5, 0.0, 0.0))

    target_x   = pose["x"] + dx * math.cos(yaw) - dy * math.sin(yaw)
    target_y   = pose["y"] + dx * math.sin(yaw) + dy * math.cos(yaw)
    target_yaw = yaw + dyaw

    log.info(f"Navigating toward person ({direction}) → ({target_x:.2f}, {target_y:.2f})")
    return send_goal(target_x, target_y, target_yaw)


def shutdown():
    """
    Clean shutdown of ROS2 node.

    Call this from main.py via atexit / signal handlers BEFORE the interpreter
    tears down daemon threads.  Proper ordering prevents std::terminate().

    Correct call order:
      1. cancel_goal()          — tell Nav2 to abort
      2. executor.shutdown()    — wakes spin() cleanly, thread exits normally
      3. node.destroy_node()    — releases ROS2 resources
      4. rclpy.shutdown()       — tears down the context
    """
    global _nav2_ok, _node, _executor

    log.info("Nav2 shutting down...")
    _nav2_ok = False

    # 1. Cancel any active navigation goal
    cancel_goal()

    # 2. Stop the executor — this wakes _spin_thread_fn() cleanly
    if _executor is not None:
        try:
            _executor.shutdown(timeout_sec=2.0)
        except Exception as e:
            log.debug(f"executor.shutdown error: {e}")

    # 3. Wait for spin thread to actually exit (up to 3 s)
    if _ros_thread is not None and _ros_thread.is_alive():
        _ros_thread.join(timeout=3.0)
        if _ros_thread.is_alive():
            log.warning("ROS2 spin thread did not exit cleanly within 3 s")

    # 4. Destroy node and shutdown rclpy context
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