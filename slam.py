"""
ERIC — Online SLAM Manager
Manages SLAM Toolbox online async mapping during missions.
Builds a live 2D occupancy map from D500 LiDAR as ERIC explores.

Architecture:
  D500 /scan  ──┐
  /odom        ──┼──→ SLAM Toolbox (online_async) → /map topic
  /tf           ──┘                                      ↓
                                                   Nav2 costmap
                                                   (path planning)

SLAM Toolbox must be installed:
  sudo apt install ros-humble-slam-toolbox

Usage:
  init_slam()              → launch SLAM Toolbox subprocess + monitor
  slam_available()         → True when map is being built
  save_map(mission_id)     → save .pgm + .yaml to missions/ folder
  reset_slam()             → clear map between missions
  get_status()             → map size, pose, status for GUI

Maps are saved to:
  missions/YYYY-MM-DD_HHMMSS_<mission_id>/map.pgm
  missions/YYYY-MM-DD_HHMMSS_<mission_id>/map.yaml

Each mission gets its own folder — full history preserved for SAR review.
"""

import logging
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

log = logging.getLogger("eric.slam")

# ── Config ────────────────────────────────────────────────────────────────────
MAPS_DIR             = Path("missions")          # base directory for saved maps
SLAM_LAUNCH_TIMEOUT  = 15.0                      # seconds to wait for SLAM ready
SLAM_CHECK_INTERVAL  = 2.0                       # how often to check map topic
MAP_TOPIC            = "/map"                    # SLAM Toolbox publishes here
SLAM_NODE_NAME       = "async_slam_toolbox_node" # ROS2 node name to check

# ── SLAM Toolbox params written at runtime ────────────────────────────────────
# These are written to a temp file and passed to slam_toolbox at launch.
# Tuned for D500 LiDAR + UGV Beast geometry.
SLAM_PARAMS = """
slam_toolbox:
  ros__parameters:
    # Solver
    solver_plugin: solver_plugins::CeresSolver
    ceres_linear_solver: SPARSE_NORMAL_CHOLESKY
    ceres_preconditioner: SCHUR_JACOBI
    ceres_trust_strategy: LEVENBERG_MARQUARDT

    # Online async mode — builds map in real time, no pre-built map needed
    mode: mapping

    # Topics
    odom_frame:   odom
    map_frame:    map
    base_frame:   base_link
    scan_topic:   /scan
    use_scan_matching: true
    use_scan_barycenter: true

    # Map resolution and size
    resolution: 0.05             # 5cm per cell — good for indoor SAR
    max_laser_range: 12.0        # D500 max range
    minimum_travel_distance: 0.2 # only update map after moving 20cm
    minimum_travel_heading: 0.1  # or turning 0.1 rad (~6°)

    # Loop closure — finds when robot returns to known area
    loop_search_maximum_distance: 3.0
    do_loop_closing: true
    loop_match_minimum_chain_size: 10
    loop_match_maximum_variance_covariance: 3.0
    loop_match_minimum_response_coarse: 0.35
    loop_match_minimum_response_fine: 0.45

    # Scan matcher
    correlation_search_space_dimension: 0.5
    correlation_search_space_resolution: 0.01
    correlation_search_space_smear_deviation: 0.1
    fine_search_space_dimension: 0.03
    fine_search_space_resolution: 0.003
    coarse_search_space_dimension: 0.5
    coarse_search_space_resolution: 0.03
    coarse_angle_search_space_dimension: 0.349
    coarse_angle_search_space_resolution: 0.0349

    # Serial mode off — we use async
    tf_buffer_duration: 30.0
    stack_size_to_use: 40000000
    enable_interactive_mode: false
"""

# ── Module state ──────────────────────────────────────────────────────────────
_slam_ok          = False
_slam_process     = None     # subprocess handle
_map_received     = False    # True once first /map message arrives
_map_width        = 0        # cells
_map_height       = 0        # cells
_map_resolution   = 0.05     # m/cell
_lock             = threading.Lock()
_monitor_thread   = None
_map_sub          = None
_node             = None


# ─── Public API ───────────────────────────────────────────────────────────────

def slam_available() -> bool:
    """True when SLAM is running and map data is being received."""
    return _slam_ok and _map_received


def init_slam() -> bool:
    """
    Launch SLAM Toolbox in online async mode and subscribe to /map.
    Returns True if SLAM started successfully.
    Call at mission start — after init_odom() and init_lidar().
    """
    global _slam_ok, _slam_process, _map_received, _node, _map_sub

    _map_received = False

    # Write SLAM params to temp file
    params_path = Path("/tmp/eric_slam_params.yaml")
    params_path.write_text(SLAM_PARAMS)

    # Launch slam_toolbox as a subprocess
    try:
        cmd = [
            "ros2", "launch", "slam_toolbox", "online_async_launch.py",
            f"params_file:={params_path}",
            "use_sim_time:=false",
        ]
        _slam_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        log.info(f"🗺️  SLAM Toolbox launched (PID {_slam_process.pid})")
    except FileNotFoundError:
        log.error("❌ slam_toolbox not found — install: sudo apt install ros-humble-slam-toolbox")
        _slam_ok = False
        return False
    except Exception as e:
        log.error(f"❌ SLAM launch failed: {e}")
        _slam_ok = False
        return False

    # Subscribe to /map to confirm SLAM is producing data
    try:
        import rclpy
        from nav_msgs.msg import OccupancyGrid

        try:
            from odom import _node as odom_node
            if odom_node:
                _node = odom_node
            else:
                raise Exception("odom node not ready")
        except Exception:
            _node = rclpy.create_node("eric_slam_monitor")

        _map_sub = _node.create_subscription(
            OccupancyGrid,
            MAP_TOPIC,
            _map_callback,
            10
        )
    except Exception as e:
        log.warning(f"⚠️  Could not subscribe to /map: {e}")

    # Start monitor thread — waits for first map, logs SLAM health
    _monitor_thread = threading.Thread(
        target=_slam_monitor_loop,
        daemon=True,
        name="slam-monitor"
    )
    _monitor_thread.start()

    # Wait for SLAM to come up
    deadline = time.monotonic() + SLAM_LAUNCH_TIMEOUT
    while time.monotonic() < deadline:
        if _is_slam_node_running():
            _slam_ok = True
            log.info("✅ SLAM Toolbox online — map building active")
            return True
        time.sleep(1.0)

    log.warning("⚠️  SLAM Toolbox did not start within timeout — map building unavailable")
    _slam_ok = False
    return False


def save_map(mission_id: str = "") -> str | None:
    """
    Save current map to missions/ directory.
    Returns the path to the saved map folder, or None on failure.
    Call at mission end or on operator request.

    Saves:
      missions/YYYY-MM-DD_HHMMSS_<mission_id>/map.pgm
      missions/YYYY-MM-DD_HHMMSS_<mission_id>/map.yaml
    """
    if not _slam_ok:
        log.warning("save_map: SLAM not running — nothing to save")
        return None

    timestamp  = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    folder     = MAPS_DIR / f"{timestamp}_{mission_id}".rstrip("_")
    folder.mkdir(parents=True, exist_ok=True)
    map_prefix = str(folder / "map")

    try:
        result = subprocess.run(
            ["ros2", "service", "call",
             "/slam_toolbox/save_map",
             "slam_toolbox/srv/SaveMap",
             f"{{name: {{data: '{map_prefix}'}}}}"],
            capture_output=True, text=True, timeout=10.0
        )
        if result.returncode == 0:
            log.info(f"🗺️  Map saved → {folder}/map.pgm + map.yaml")
            return str(folder)
        else:
            log.error(f"save_map service call failed: {result.stderr}")
            return None
    except subprocess.TimeoutExpired:
        log.error("save_map: service call timed out")
        return None
    except Exception as e:
        log.error(f"save_map error: {e}")
        return None


def reset_slam():
    """
    Clear the current map and restart SLAM from scratch.
    Call between missions if robot stays in same process.
    """
    global _map_received, _map_width, _map_height

    try:
        subprocess.run(
            ["ros2", "service", "call",
             "/slam_toolbox/clear",
             "slam_toolbox/srv/Clear", "{}"],
            capture_output=True, timeout=5.0
        )
        with _lock:
            _map_received = False
            _map_width    = 0
            _map_height   = 0
        log.info("🔄 SLAM map cleared — fresh map on next mission")
    except Exception as e:
        log.warning(f"reset_slam error: {e}")


def shutdown_slam():
    """Stop SLAM Toolbox process. Call at clean shutdown."""
    global _slam_ok, _slam_process
    _slam_ok = False
    if _slam_process and _slam_process.poll() is None:
        try:
            _slam_process.terminate()
            _slam_process.wait(timeout=5.0)
            log.info("SLAM Toolbox stopped")
        except Exception as e:
            log.warning(f"SLAM shutdown error: {e}")
            _slam_process.kill()
    _slam_process = None


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _map_callback(msg):
    """Called when /map publishes — confirms SLAM is producing data."""
    global _map_received, _map_width, _map_height, _map_resolution
    with _lock:
        _map_received   = True
        _map_width      = msg.info.width
        _map_height     = msg.info.height
        _map_resolution = msg.info.resolution


def _is_slam_node_running() -> bool:
    """Check if slam_toolbox ROS2 node is alive."""
    try:
        result = subprocess.run(
            ["ros2", "node", "list"],
            capture_output=True, text=True, timeout=3.0
        )
        return SLAM_NODE_NAME in result.stdout
    except Exception:
        return False


def _slam_monitor_loop():
    """
    Background thread — monitors SLAM health.
    Logs when map first arrives. Detects if SLAM process dies unexpectedly.
    """
    global _slam_ok, _map_received

    map_logged = False
    while True:
        time.sleep(SLAM_CHECK_INTERVAL)

        if not _slam_ok:
            break

        # Log when first map arrives
        if _map_received and not map_logged:
            with _lock:
                w, h, r = _map_width, _map_height, _map_resolution
            log.info(f"🗺️  First map received — {w}×{h} cells @ {r}m/cell "
                     f"({w*r:.1f}m × {h*r:.1f}m coverage)")
            map_logged = True

        # Check if SLAM process died
        if _slam_process and _slam_process.poll() is not None:
            log.error("❌ SLAM Toolbox process died unexpectedly — map building stopped")
            _slam_ok = False
            break


# ─── Status ───────────────────────────────────────────────────────────────────

def get_status() -> dict:
    with _lock:
        return {
            "available":     _slam_ok,
            "map_received":  _map_received,
            "map_width_m":   round(_map_width  * _map_resolution, 1),
            "map_height_m":  round(_map_height * _map_resolution, 1),
            "map_cells":     _map_width * _map_height,
        }


def slam_status_html() -> str:
    """HTML status panel for Gradio GUI."""
    s = get_status()

    if not s["available"]:
        return """
        <div style="background:#1a1a1a;border:1px solid #444;border-radius:8px;
                    padding:10px;font-family:monospace;color:#666">
            🗺️  SLAM: not running
        </div>"""

    if not s["map_received"]:
        color, label = "#ff6600", "⏳ waiting for first scan..."
    else:
        color, label = "#76b900", f"✅ mapping — {s['map_width_m']}m × {s['map_height_m']}m"

    return f"""
    <div style="background:#1a1a1a;border:1px solid {color};border-radius:8px;
                padding:10px;font-family:monospace;">
        <div style="color:{color};font-weight:bold">🗺️  SLAM Toolbox — {label}</div>
        <div style="color:#aaa;font-size:0.85em;margin-top:4px">
            Cells mapped: <span style="color:#fff">{s['map_cells']:,}</span>
        </div>
    </div>"""
