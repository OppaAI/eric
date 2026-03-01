"""
ERIC — Online SLAM Manager
Manages SLAM Toolbox online async mapping during missions.
Builds a live 2D occupancy map from D500 LiDAR as ERIC explores.

Architecture:
  D500 /scan  ──┐
  /odom        ──┼──→ SLAM Toolbox (online_async) → /map topic
  /tf           ──┘                                      ↓
                                                   Nav2 costmap

ROS2 node sharing:
  Uses ros_core.get_node() — no longer creates its own node.
  /map subscription shares the single 'eric_robot' node with
  odom, lidar, and nav2.

SLAM Toolbox must be installed:
  sudo apt install ros-humble-slam-toolbox

Usage:
  init_slam()              → launch SLAM Toolbox subprocess + monitor
  slam_available()         → True when map is being built
  save_map(mission_id)     → save .pgm + .yaml to missions/ folder
  reset_slam()             → clear map between missions
  get_status()             → map size, status dict for GUI

Maps saved to:
  missions/YYYY-MM-DD_HHMMSS_<mission_id>/map.pgm + map.yaml
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
MAPS_DIR             = Path("missions")
SLAM_LAUNCH_TIMEOUT  = 40.0    # seconds to wait for SLAM node to appear (Jetson needs 20-35s)
SLAM_CHECK_INTERVAL  = 2.0     # health monitor poll interval
MAP_TOPIC            = "/map"
SLAM_NODE_NAME       = "slam_toolbox"

# ── SLAM Toolbox params ───────────────────────────────────────────────────────
# Tuned for D500 LiDAR + UGV Beast.
# minimum_travel_distance reduced to 0.05m (was 0.20m) — the robot must be
# able to update the map while turning in a doorway without moving 20cm
# linearly first. At 0.20m, SLAM misses walls discovered during slow turns.
SLAM_PARAMS = """
slam_toolbox:
  ros__parameters:
    solver_plugin: solver_plugins::CeresSolver
    ceres_linear_solver: SPARSE_NORMAL_CHOLESKY
    ceres_preconditioner: SCHUR_JACOBI
    ceres_trust_strategy: LEVENBERG_MARQUARDT

    mode: mapping

    odom_frame:   odom
    map_frame:    map
    base_frame:   base_link
    scan_topic:   /scan
    use_scan_matching: true
    use_scan_barycenter: true

    resolution: 0.05             # 5cm per cell — good for indoor SAR
    max_laser_range: 12.0        # D500 max range

    # Reduced from 0.20m — allows map updates during slow turns and
    # doorway traversal without needing a full 20cm linear move first.
    minimum_travel_distance: 0.05
    minimum_travel_heading: 0.05  # ~3° — was 0.1 rad (~6°)

    loop_search_maximum_distance: 3.0
    do_loop_closing: true
    loop_match_minimum_chain_size: 10
    loop_match_maximum_variance_covariance: 3.0
    loop_match_minimum_response_coarse: 0.35
    loop_match_minimum_response_fine: 0.45

    correlation_search_space_dimension: 0.5
    correlation_search_space_resolution: 0.01
    correlation_search_space_smear_deviation: 0.1
    fine_search_space_dimension: 0.03
    fine_search_space_resolution: 0.003
    coarse_search_space_dimension: 0.5
    coarse_search_space_resolution: 0.03
    coarse_angle_search_space_dimension: 0.349
    coarse_angle_search_space_resolution: 0.0349

    tf_buffer_duration: 30.0
    stack_size_to_use: 40000000
    enable_interactive_mode: false
"""

# ── Module state ──────────────────────────────────────────────────────────────
_slam_ok        = False
_slam_process   = None
_map_received   = False
_map_width      = 0
_map_height     = 0
_map_resolution = 0.05
_lock           = threading.Lock()
_monitor_thread = None
_map_sub        = None


# ─── Public API ───────────────────────────────────────────────────────────────

def slam_available() -> bool:
    """True when SLAM is running and map data is being received."""
    return _slam_ok and _map_received


def init_slam() -> bool:
    """
    Launch SLAM Toolbox in online async mode and subscribe to /map.
    Call after init_odom() — SLAM needs /odom to be publishing.
    Returns True if SLAM Toolbox node appeared within SLAM_LAUNCH_TIMEOUT.
    """
    global _slam_ok, _slam_process, _map_received, _map_sub

    _map_received = False

    # Write SLAM params to temp file
    params_path = Path("/tmp/eric_slam_params.yaml")
    params_path.write_text(SLAM_PARAMS)

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
        log.error("❌ slam_toolbox not found — install: "
                  "sudo apt install ros-humble-slam-toolbox")
        _slam_ok = False
        return False
    except Exception as e:
        log.error(f"❌ SLAM launch failed: {e}")
        _slam_ok = False
        return False

    # Subscribe to /map using shared ros_core node
    try:
        from ros_core import get_node, ensure_spinning
        from nav_msgs.msg import OccupancyGrid

        node = get_node()
        if node is not None:
            _map_sub = node.create_subscription(
                OccupancyGrid, MAP_TOPIC, _map_callback, 10
            )
            ensure_spinning()
        else:
            log.warning("⚠️  ROS2 unavailable — /map subscription skipped")
    except Exception as e:
        log.warning(f"⚠️  Could not subscribe to /map: {e}")

    # Start health monitor thread
    _monitor_thread = threading.Thread(
        target=_slam_monitor_loop,
        daemon=True,
        name="slam-monitor"
    )
    _monitor_thread.start()

    # Wait for SLAM Toolbox node to appear
    deadline = time.monotonic() + SLAM_LAUNCH_TIMEOUT
    while time.monotonic() < deadline:
        if _is_slam_node_running():
            _slam_ok = True
            log.info("✅ SLAM Toolbox online — map building active")
            return True
        time.sleep(1.0)

    log.warning("⚠️  SLAM Toolbox did not start within timeout")
    _slam_ok = False
    return False


def save_map(mission_id: str = "") -> str | None:
    """
    Save current map to missions/ directory.
    Returns the path to the saved folder, or None on failure.
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
    """Clear the current map and restart SLAM from scratch."""
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
        log.info("🔄 SLAM map cleared")
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


# ─── Internal ─────────────────────────────────────────────────────────────────

def _map_callback(msg):
    """Called when SLAM Toolbox publishes a new /map."""
    global _map_received, _map_width, _map_height, _map_resolution
    with _lock:
        _map_received   = True
        _map_width      = msg.info.width
        _map_height     = msg.info.height
        _map_resolution = msg.info.resolution


def _is_slam_node_running() -> bool:
    try:
        result = subprocess.run(
            ["ros2", "node", "list"],
            capture_output=True, text=True, timeout=3.0
        )
        return SLAM_NODE_NAME in result.stdout
    except Exception:
        return False


def _slam_monitor_loop():
    """Monitor SLAM health — logs first map, detects process death."""
    global _slam_ok
    map_logged = False
    while True:
        time.sleep(SLAM_CHECK_INTERVAL)
        if not _slam_ok:
            break
        if _map_received and not map_logged:
            with _lock:
                w, h, r = _map_width, _map_height, _map_resolution
            log.info(f"🗺️  First map received — {w}×{h} cells @ {r}m/cell "
                     f"({w*r:.1f}m × {h*r:.1f}m coverage)")
            map_logged = True
        if _slam_process and _slam_process.poll() is not None:
            log.error("❌ SLAM Toolbox process died — map building stopped")
            _slam_ok = False
            break


# ─── Status ───────────────────────────────────────────────────────────────────

def get_status() -> dict:
    with _lock:
        return {
            "available":    _slam_ok,
            "map_received": _map_received,
            "map_width_m":  round(_map_width  * _map_resolution, 1),
            "map_height_m": round(_map_height * _map_resolution, 1),
            "map_cells":    _map_width * _map_height,
        }


def slam_status_html() -> str:
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