"""
ERIC — Shared ROS2 Core
Single node + single executor shared across all ERIC modules.

Problem solved:
  odom.py, slam.py, lidar.py, and nav2.py each tried to share a node
  by importing each other's _node private, AND each created their own
  SingleThreadedExecutor and called executor.spin() in their own thread.
  rclpy explicitly does not support one node registered in multiple executors —
  the result is non-deterministic missed callbacks, especially at 10-20 Hz.

Solution:
  This module owns the ONE node and ONE MultiThreadedExecutor.
  All other modules call get_node() to create publishers/subscribers/clients.
  The spin thread is started once — subsequent calls to ensure_spinning() are no-ops.

Static transforms published on startup:
  base_link → base_lidar_link  (0, 0, 0.2m) — D500 LiDAR mount height
  base_link → base_footprint   (0, 0, 0)    — required by Nav2/SLAM default params
  These replace the manual `ros2 run tf2_ros static_transform_publisher` calls.

Nav2 launch:
  launch_nav2() starts the full Nav2 stack as a subprocess using nav2_bringup.
  Called from main.py after init_slam() confirms the map is building.

Usage:
  from ros_core import get_node, ensure_spinning, ros_ok, launch_nav2

  node = get_node()                    # get the shared Node (creates if needed)
  ensure_spinning()                    # start spin thread if not already running
  sub  = node.create_subscription(...) # normal rclpy API
  pub  = node.create_publisher(...)    # normal rclpy API
"""

import logging
import subprocess
import threading
import time
from pathlib import Path

log = logging.getLogger("eric.ros_core")

_node        = None
_executor    = None
_spin_thread = None
_tf_pub      = None   # StaticTransformBroadcaster
_lock        = threading.Lock()
_init_done   = False

_nav2_process = None  # subprocess.Popen for Nav2 bringup

# ── Static transform config ───────────────────────────────────────────────────
# Add or edit entries here — no need to run manual static_transform_publisher.
# Format: (x, y, z, roll, pitch, yaw, parent_frame, child_frame)
STATIC_TRANSFORMS = [
    # D500 LiDAR is mounted 20cm above base_link, centred, no rotation
    (0.0, 0.0, 0.2, 0.0, 0.0, 0.0, "base_link", "base_lidar_link"),
    # Nav2 + SLAM Toolbox default params expect base_footprint — map to base_link
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "base_link", "base_footprint"),
    # OAK-D camera is mounted 15cm above ground, centred, no rotation
    (0.0, 0.0, 0.15, 0.0, 0.0, 0.0, "base_link", "oakd_link"),
]

# ── Nav2 params file ──────────────────────────────────────────────────────────
NAV2_PARAMS_PATH = Path("/tmp/eric_nav2_params.yaml")
NAV2_PARAMS = """\
bt_navigator:
  ros__parameters:
    global_frame: map
    robot_base_frame: base_link
    odom_topic: /odom
    bt_loop_duration: 10
    default_server_timeout: 20
    default_nav_to_pose_bt_xml: "/opt/ros/humble/share/nav2_bt_navigator/behavior_trees/navigate_to_pose_w_replanning_and_recovery.xml"
    default_nav_through_poses_bt_xml: "/opt/ros/humble/share/nav2_bt_navigator/behavior_trees/navigate_through_poses_w_replanning_and_recovery.xml"
    navigators: ["navigate_to_pose", "navigate_through_poses"]
    navigate_to_pose:
      plugin: "nav2_bt_navigator/NavigateToPoseNavigator"
    navigate_through_poses:
      plugin: "nav2_bt_navigator/NavigateThroughPosesNavigator"

controller_server:
  ros__parameters:
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    failure_tolerance: 0.3
    progress_checker_plugins: ["progress_checker"]
    goal_checker_plugins: ["general_goal_checker"]
    controller_plugins: ["FollowPath"]
    progress_checker:
      plugin: "nav2_controller::SimpleProgressChecker"
      required_movement_radius: 0.5
      movement_time_allowance: 10.0
    general_goal_checker:
      stateful: True
      plugin: "nav2_controller::SimpleGoalChecker"
      xy_goal_tolerance: 0.25
      yaw_goal_tolerance: 0.25
    FollowPath:
      plugin: "nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController"
      desired_linear_vel: 0.3
      lookahead_dist: 0.6
      min_lookahead_dist: 0.3
      max_lookahead_dist: 0.9
      lookahead_time: 1.5
      rotate_to_heading_angular_vel: 1.8
      transform_tolerance: 0.1
      use_velocity_scaled_lookahead_dist: false
      min_approach_linear_velocity: 0.05
      approach_velocity_scaling_dist: 1.0
      use_collision_detection: true
      max_allowed_time_to_collision_up_to_carrot: 1.0
      use_regulated_linear_velocity_scaling: true
      use_fixed_curvature_lookahead: false
      curvature_feedback_gain: 3.5
      regulated_linear_scaling_min_radius: 0.9
      regulated_linear_scaling_min_speed: 0.25
      use_rotate_to_heading: true
      allow_reversing: false
      rotate_to_heading_min_angle: 0.785
      max_angular_accel: 3.2
      max_robot_pose_search_dist: 10.0

planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner/NavfnPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: true

smoother_server:
  ros__parameters:
    smoother_plugins: ["simple_smoother"]
    simple_smoother:
      plugin: "nav2_smoother::SimpleSmoother"
      tolerance: 1.0e-10
      max_its: 1000
      do_refinement: True

behavior_server:
  ros__parameters:
    costmap_topic: local_costmap/costmap_raw
    footprint_topic: local_costmap/published_footprint
    cycle_frequency: 10.0
    behavior_plugins: ["spin", "backup", "drive_on_heading", "assisted_teleop", "wait"]
    spin:
      plugin: "nav2_behaviors::Spin"
    backup:
      plugin: "nav2_behaviors::BackUp"
    drive_on_heading:
      plugin: "nav2_behaviors::DriveOnHeading"
    wait:
      plugin: "nav2_behaviors::Wait"
    assisted_teleop:
      plugin: "nav2_behaviors::AssistedTeleop"
    global_frame: odom
    robot_base_frame: base_link
    transform_tolerance: 0.1
    simulate_ahead_time: 2.0
    max_rotational_vel: 1.0
    min_rotational_vel: 0.4
    rotational_acc_lim: 3.2

waypoint_follower:
  ros__parameters:
    loop_rate: 20
    stop_on_failure: false
    action_server_result_timeout: 900.0
    waypoint_task_executor_plugin: "wait_at_waypoint"
    wait_at_waypoint:
      plugin: "nav2_waypoint_follower::WaitAtWaypoint"
      enabled: True
      waypoint_pause_duration: 200

velocity_smoother:
  ros__parameters:
    smoothing_frequency: 20.0
    scale_velocities: False
    feedback: "OPEN_LOOP"
    max_velocity: [0.5, 0.0, 2.0]
    min_velocity: [-0.5, 0.0, -2.0]
    max_accel: [2.5, 0.0, 3.2]
    max_decel: [-2.5, 0.0, -3.2]
    odom_topic: "odom"
    odom_duration: 0.1
    deadband_velocity: [0.0, 0.0, 0.0]
    velocity_timeout: 1.0

collision_monitor:
  ros__parameters:
    base_frame_id: "base_link"
    odom_frame_id: "odom"
    cmd_vel_in_topic: "cmd_vel_smoothed"
    cmd_vel_out_topic: "cmd_vel"
    state_topic: "collision_monitor_state"
    transform_tolerance: 0.2
    source_timeout: 1.0
    base_shift_correction: True
    stop_pub_timeout: 2.0
    polygons: ["FootprintApproach"]
    FootprintApproach:
      type: "polygon"
      action_type: "approach"
      footprint_topic: "/local_costmap/published_footprint"
      time_before_collision: 1.2
      simulation_time_step: 0.1
      min_points: 6
      visualize: False
      enabled: True
    observation_sources: ["scan"]
    scan:
      type: "scan"
      topic: "/scan"

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: odom
      robot_base_frame: base_link
      rolling_window: true
      width: 3
      height: 3
      resolution: 0.05
      footprint: "[[0.115, 0.10], [0.115, -0.10], [-0.115, -0.10], [-0.115, 0.10]]"
      plugins: ["obstacle_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.35
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        publish_voxel_map: True
        origin_z: 0.0
        z_resolution: 0.05
        z_voxels: 16
        max_obstacle_height: 2.0
        mark_threshold: 0
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
      always_send_full_costmap: True

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 1.0
      global_frame: map
      robot_base_frame: base_link
      footprint: "[[0.115, 0.10], [0.115, -0.10], [-0.115, -0.10], [-0.115, 0.10]]"
      resolution: 0.05
      track_unknown_space: true
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_topic: /map
        map_subscribe_transient_local: True
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.35
      always_send_full_costmap: True

map_server:
  ros__parameters:
    yaml_filename: ""

map_saver:
  ros__parameters:
    save_map_timeout: 5.0
    free_thresh_default: 0.25
    occupied_thresh_default: 0.65
    map_subscribe_transient_local: True

amcl:
  ros__parameters:
    use_sim_time: False

lifecycle_manager_navigation:
  ros__parameters:
    use_sim_time: false
    autostart: true
    bond_timeout: 4.0
    attempt_respawn_reconnection: true
    bond_respawn_max_duration: 10.0
    node_names:
      - controller_server
      - smoother_server
      - planner_server
      - behavior_server
      - bt_navigator
      - waypoint_follower
      - velocity_smoother
"""


def ros_ok() -> bool:
    """True if rclpy is initialised and the shared node exists."""
    try:
        import rclpy
        return rclpy.ok() and _node is not None
    except ImportError:
        return False


def get_node():
    """
    Return the shared ROS2 node, initialising rclpy and creating the node
    on first call. Publishes static transforms on first call.
    Thread-safe — safe to call from any module at import time.
    Returns None if ROS2 is not available.
    """
    global _node, _executor, _init_done

    if _node is not None:
        return _node

    with _lock:
        if _node is not None:          # double-checked locking
            return _node

        try:
            import rclpy
            import rclpy.executors

            if not rclpy.ok():
                rclpy.init()

            _node     = rclpy.create_node("eric_robot")
            _executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
            _executor.add_node(_node)
            _init_done = True
            log.info("✅ ROS2 core: shared node 'eric_robot' created")

            # Publish static transforms immediately after node creation
            _publish_static_transforms()

            return _node

        except ImportError:
            log.warning("⚠️  ROS2 not available — all ROS2 modules will be disabled")
            return None
        except Exception as e:
            log.warning(f"⚠️  ROS2 core init failed: {e}")
            return None


def _publish_static_transforms():
    """
    Publish all static transforms defined in STATIC_TRANSFORMS.
    Called once on node creation — no manual static_transform_publisher needed.

    Transforms published:
      base_link → base_lidar_link  : D500 LiDAR at 20cm height
      base_link → base_footprint   : required by Nav2/SLAM default params
    """
    global _tf_pub
    try:
        import math
        from geometry_msgs.msg import TransformStamped
        from tf2_ros import StaticTransformBroadcaster

        _tf_pub = StaticTransformBroadcaster(_node)
        transforms = []

        for x, y, z, roll, pitch, yaw, parent, child in STATIC_TRANSFORMS:
            t = TransformStamped()
            t.header.stamp            = _node.get_clock().now().to_msg()
            t.header.frame_id         = parent
            t.child_frame_id          = child
            t.transform.translation.x = x
            t.transform.translation.y = y
            t.transform.translation.z = z
            # Convert RPY to quaternion
            cy = math.cos(yaw   * 0.5)
            sy = math.sin(yaw   * 0.5)
            cp = math.cos(pitch * 0.5)
            sp = math.sin(pitch * 0.5)
            cr = math.cos(roll  * 0.5)
            sr = math.sin(roll  * 0.5)
            t.transform.rotation.w = cr * cp * cy + sr * sp * sy
            t.transform.rotation.x = sr * cp * cy - cr * sp * sy
            t.transform.rotation.y = cr * sp * cy + sr * cp * sy
            t.transform.rotation.z = cr * cp * sy - sr * sp * cy
            transforms.append(t)
            log.info(f"📐 Static TF: {parent} → {child} "
                     f"(xyz={x},{y},{z} rpy={roll},{pitch},{yaw})")

        _tf_pub.sendTransform(transforms)
        log.info(f"✅ ROS2 core: {len(transforms)} static transforms published")

    except Exception as e:
        log.warning(f"⚠️  Static transform publish failed: {e}")


def ensure_spinning():
    """
    Start the shared executor spin thread if not already running.
    Safe to call multiple times — only one thread is ever started.
    Call this once after all subscriptions/publishers have been created.
    """
    global _spin_thread

    if _spin_thread is not None and _spin_thread.is_alive():
        return   # already spinning

    if _executor is None:
        get_node()   # ensure node + executor exist

    if _executor is None:
        return   # ROS2 unavailable

    with _lock:
        if _spin_thread is not None and _spin_thread.is_alive():
            return

        _spin_thread = threading.Thread(
            target=_spin_fn,
            daemon=True,
            name="ros-core-spin"
        )
        _spin_thread.start()
        log.info("✅ ROS2 core: executor spin thread started")


def _spin_fn():
    try:
        _executor.spin()
    except Exception as e:
        log.debug(f"ROS2 executor spin exited: {e}")


def launch_nav2() -> subprocess.Popen | None:
    """
    Launch the full Nav2 navigation stack as a subprocess.
    Writes nav2 params to NAV2_PARAMS_PATH and launches nav2_bringup.
    Returns the Popen process, or None on failure.

    Call from main.py after init_slam() — Nav2 needs the SLAM map topic
    to be publishing before its costmap layers can configure themselves.

    The nav2.py action client (init_nav2) will connect to the
    navigate_to_pose action server that this brings up.
    """
    global _nav2_process

    try:
        NAV2_PARAMS_PATH.write_text(NAV2_PARAMS)
        cmd = [
            "ros2", "launch", "nav2_bringup", "navigation_launch.py",
            f"params_file:={NAV2_PARAMS_PATH}",
            "use_sim_time:=false",
        ]
        _nav2_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        log.info(f"🧭 Nav2 stack launched (PID {_nav2_process.pid})")
        return _nav2_process
    except Exception as e:
        log.error(f"❌ Nav2 launch failed: {e}")
        return None


def nav2_process_ok() -> bool:
    """True if the Nav2 subprocess is still running."""
    return _nav2_process is not None and _nav2_process.poll() is None


def shutdown():
    """
    Clean ROS2 shutdown — call once from main.py atexit / signal handler.
    Stops Nav2 subprocess, executor, destroys node, shuts down rclpy.
    """
    global _node, _executor, _spin_thread, _init_done, _nav2_process

    log.info("ROS2 core: shutting down...")

    # Stop Nav2 subprocess first
    if _nav2_process is not None and _nav2_process.poll() is None:
        try:
            _nav2_process.terminate()
            _nav2_process.wait(timeout=5.0)
            log.info("Nav2 subprocess stopped")
        except Exception as e:
            log.debug(f"Nav2 subprocess stop error: {e}")
            try:
                _nav2_process.kill()
            except Exception:
                pass
    _nav2_process = None

    if _executor is not None:
        try:
            _executor.shutdown(timeout_sec=2.0)
        except Exception as e:
            log.debug(f"executor.shutdown: {e}")

    if _spin_thread is not None and _spin_thread.is_alive():
        _spin_thread.join(timeout=3.0)
        if _spin_thread.is_alive():
            log.warning("ROS2 spin thread did not exit within 3s")

    try:
        import rclpy
        if _node is not None:
            _node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    except Exception as e:
        log.debug(f"rclpy.shutdown: {e}")
    finally:
        _node      = None
        _executor  = None
        _init_done = False

    log.info("ROS2 core: shutdown complete")