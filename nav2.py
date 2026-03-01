"""
ERIC — ROS2 Nav2 Integration  (SLAM version)
Sends navigation goals to Nav2 with live SLAM map support.

Fixes vs previous version:
──────────────────────────
1. ROS2 node sharing — uses ros_core.get_node() instead of creating its own
   node and executor. No longer registers a node in multiple executors.

2. Startup race fixed — init_nav2() now waits for slam_available() to return
   True before starting its Nav2 action server timeout countdown.
   Previously Nav2's 5s timeout expired before SLAM Toolbox had started
   (typically 8-12s on Jetson), so Nav2 always reported unavailable.

3. MAP_MIN_CELLS counts occupied cells only — was counting all known cells
   (free=0 counts as known). 100 free cells can appear in the first 1-2 scans
   before any walls are mapped. Now counts c > 0 (occupied) only, with a
   lower threshold of 50 — meaning real wall cells must be present.

4. costmap_clear() service path fixed — was "/nav2/{costmap}/..." but the
   actual Nav2 service name has no /nav2/ prefix. Silent failures every time
   SLAM updated. Corrected to "/{costmap}/clear_entirely_costmap".

Architecture:
  odom.py    → /odom     ──┐
  lidar.py → /scan     ──┼──→ SLAM Toolbox → /map
  oakd.py  → /oakd/depth┘          ↓
                                Nav2 costmap
                                     ↓
                          nav2.py send_goal()
                                     ↓
                               motors (cmd_vel)

Usage:
  nav2_available()         → True once ROS2 + Nav2 running
  map_ready()              → True once SLAM map has enough occupied cells
  send_goal(x, y, yaw)    → navigate to map coordinates
  cancel_goal()           → abort navigation
  get_pose()              → current robot pose on map
  is_navigating()         → True while goal is executing
  costmap_clear()         → clear costmap after SLAM map update
"""

import logging
import math
import threading
import time

log = logging.getLogger("eric.nav2")

_nav2_ok      = False
_map_ok       = False
_navigating   = False
_current_goal = None
_pose         = {"x": 0.0, "y": 0.0, "yaw": 0.0}

_nav_client   = None
_tf_buffer    = None
_map_sub      = None

# Occupied-cell threshold for declaring map ready for path planning.
# Counts cells with value > 0 (occupied) — NOT free cells (value=0).
# 50 occupied cells means real walls are present, not just scanned empty space.
MAP_MIN_OCCUPIED_CELLS = 50

# How long to wait for SLAM to be available before giving up on Nav2 init.
SLAM_WAIT_TIMEOUT = 60.0   # seconds — SLAM Toolbox takes 8-15s on Jetson


def nav2_available() -> bool:
    return _nav2_ok


def map_ready() -> bool:
    """
    True when SLAM has mapped enough occupied cells for reliable path planning.
    Check this before sending goals — planning on an empty map produces bad paths.
    """
    return _nav2_ok and _map_ok


def is_navigating() -> bool:
    return _navigating


def get_pose() -> dict:
    """Return current robot pose as {x, y, yaw} in map frame."""
    _refresh_pose()
    return dict(_pose)


def init_nav2() -> bool:
    """
    Initialise Nav2 action client using the shared ros_core node.
    Waits for slam_available() before starting — Nav2 action server needs
    the SLAM map to be publishing before it can plan paths.

    Call after init_odom(), init_lidar(), init_slam() in main.py.
    Returns True if Nav2 action server found within timeout.
    """
    global _nav2_ok, _map_ok, _nav_client, _tf_buffer, _map_sub

    try:
        from ros_core import get_node, ensure_spinning
        from nav2_msgs.action import NavigateToPose
        from rclpy.action import ActionClient
        from tf2_ros import Buffer, TransformListener
        from nav_msgs.msg import OccupancyGrid

        node = get_node()
        if node is None:
            raise RuntimeError("ROS2 not available")

        _tf_buffer  = Buffer()
        _           = TransformListener(_tf_buffer, node)
        _nav_client = ActionClient(node, NavigateToPose, "navigate_to_pose")

        # Subscribe to /map for occupied-cell tracking
        _map_sub = node.create_subscription(
            OccupancyGrid, "/map", _map_callback, 10
        )

        ensure_spinning()

        # ── Wait for SLAM to be available before timing out on Nav2 ───────────
        # SLAM Toolbox typically takes 8-15s to start on the Jetson.
        # If we start the Nav2 timeout countdown immediately it always expires
        # before SLAM is ready, so Nav2 reports unavailable on every boot.
        log.info("⏳ Waiting for SLAM map before connecting Nav2...")
        slam_deadline = time.monotonic() + SLAM_WAIT_TIMEOUT
        while time.monotonic() < slam_deadline:
            try:
                from slam import slam_available
                if slam_available():
                    log.info("🗺️  SLAM ready — connecting Nav2 action server...")
                    break
            except ImportError:
                break   # slam.py not loaded — skip wait
            time.sleep(1.0)
        else:
            log.warning("⚠️  SLAM did not become available within "
                        f"{SLAM_WAIT_TIMEOUT}s — continuing Nav2 init anyway")

        # ── Wait for Nav2 action server (retry loop) ──────────────────────────
        # Nav2 lifecycle activation on Jetson takes 45-70s.
        # Retry every 5s for up to 90s instead of one blocking wait.
        log.info("⏳ Waiting for Nav2 action server...")
        nav2_deadline = time.monotonic() + 90.0
        _nav2_ok = False
        while time.monotonic() < nav2_deadline:
            if _nav_client.wait_for_server(timeout_sec=5.0):
                _nav2_ok = True
                log.info("✅ Nav2 connected — autonomous navigation enabled")
                break
            elapsed = int(time.monotonic() - (nav2_deadline - 90.0))
            log.info(f"⏳ Nav2 not ready yet — retrying... ({elapsed}s elapsed)")
        if not _nav2_ok:
            log.warning("⚠️  Nav2 action server not found — "
                        "falling back to direct motor control")

        return _nav2_ok

    except ImportError:
        log.warning("⚠️  ROS2 not found — Nav2 disabled")
        _nav2_ok = False
        return False
    except Exception as e:
        log.warning(f"⚠️  Nav2 init failed ({e}) — direct motor control only")
        _nav2_ok = False
        return False


def _map_callback(msg):
    """
    Called on each /map update from SLAM Toolbox.
    Counts occupied cells (value > 0) to decide if map is ready for planning.

    Previous version counted ALL known cells (c >= 0), which includes free
    space (value=0). 100 free cells appear in the first 1-2 scans before any
    walls are mapped — declaring map_ready() that early caused Nav2 to plan
    paths through unmapped walls.
    """
    global _map_ok
    try:
        occupied = sum(1 for c in msg.data if c > 0)
        if occupied >= MAP_MIN_OCCUPIED_CELLS and not _map_ok:
            _map_ok = True
            w = round(msg.info.width  * msg.info.resolution, 1)
            h = round(msg.info.height * msg.info.resolution, 1)
            log.info(f"🗺️  Map ready for Nav2 — {w}m × {h}m "
                     f"({occupied} occupied cells)")
    except Exception as e:
        log.debug(f"map_callback error: {e}")


def send_goal(x: float, y: float, yaw: float = 0.0,
              on_complete=None, on_fail=None) -> bool:
    """
    Send a navigation goal to Nav2.
    x, y: target in map frame (metres from SLAM origin).
    yaw:  target heading (radians, 0 = forward).
    Checks map_ready() — won't plan on an empty SLAM map.
    Returns True if goal was accepted.
    """
    global _navigating, _current_goal

    if not _nav2_ok:
        log.warning("Nav2 not available")
        return False
    if not _map_ok:
        log.warning("SLAM map not ready — goal deferred")
        return False

    try:
        from nav2_msgs.action import NavigateToPose
        from geometry_msgs.msg import PoseStamped
        from ros_core import get_node

        node     = get_node()
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id    = "map"
        goal_msg.pose.header.stamp       = node.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x    = float(x)
        goal_msg.pose.pose.position.y    = float(y)
        goal_msg.pose.pose.position.z    = 0.0
        goal_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

        _navigating = True
        log.info(f"🧭 Nav2 goal → ({x:.2f}, {y:.2f}, "
                 f"yaw={math.degrees(yaw):.0f}°)")

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
    Clear Nav2 costmaps after SLAM map updates significantly.
    Fixed: was "/nav2/{costmap}/..." — correct path has no /nav2/ prefix.
    Verify with: ros2 service list | grep clear
    """
    try:
        import subprocess
        for costmap in ("global_costmap", "local_costmap"):
            subprocess.run(
                ["ros2", "service", "call",
                 f"/{costmap}/clear_entirely_costmap",
                 "nav2_msgs/srv/ClearEntireCostmap", "{}"],
                capture_output=True, timeout=3.0
            )
        log.debug("Nav2 costmaps cleared")
    except Exception as e:
        log.debug(f"costmap_clear error: {e}")


def _refresh_pose():
    """Update _pose from TF map→base_link (provided by SLAM Toolbox)."""
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
    Navigate toward a detected person using SLAM map pose.
    Waits for map_ready() — will not send a goal on an empty map.
    direction: "front" | "left" | "right" | "behind"
    """
    if not _nav2_ok:
        return False
    if not _map_ok:
        log.info("navigate_to_person: map not ready — using direct approach")
        return False

    pose = get_pose()
    yaw  = pose["yaw"]
    dir_offsets = {
        "front":  ( 1.0,  0.0,  0.0),
        "left":   ( 0.5,  0.7,  math.pi / 2),
        "right":  ( 0.5, -0.7, -math.pi / 2),
        "behind": (-1.0,  0.0,  math.pi),
    }
    dx, dy, dyaw = dir_offsets.get(direction, (1.0, 0.0, 0.0))
    target_x   = pose["x"] + dx * math.cos(yaw) - dy * math.sin(yaw)
    target_y   = pose["y"] + dx * math.sin(yaw) + dy * math.cos(yaw)
    target_yaw = yaw + dyaw
    log.info(f"🧭 Navigating toward person ({direction}) → "
             f"({target_x:.2f}, {target_y:.2f})")
    return send_goal(target_x, target_y, target_yaw)


def shutdown():
    """Clean Nav2 shutdown — call from main.py atexit."""
    global _nav2_ok, _map_ok
    log.info("Nav2 shutting down...")
    _nav2_ok = False
    _map_ok  = False
    cancel_goal()
    # ros_core.shutdown() handles node/executor/rclpy teardown
    log.info("Nav2 shutdown complete")


def get_status() -> dict:
    pose = get_pose()
    return {
        "available":  _nav2_ok,
        "map_ready":  _map_ok,
        "navigating": _navigating,
        "x":          round(pose["x"], 2),
        "y":          round(pose["y"], 2),
        "yaw_deg":    round(math.degrees(pose["yaw"]), 1),
    }