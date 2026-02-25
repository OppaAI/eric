"""
ERIC — Edge Robotics Innovation by Cosmos
================================================
NVIDIA Cosmos Cookoff 2026
Stack:
  - Cosmos Reason 2 (vLLM)       : vision + physical reasoning
  - Piper via RealtimeTTS        : streaming TTS, CPU only, zero VRAM
  - Waveshare UGV Beast          : tracked robot via serial UART → ESP32
  - Gradio                       : dual camera GUI + mission control
  - ROS2 Nav2 (optional)         : autonomous path planning
  - D500 LiDAR (optional)        : reactive obstacle safety layer
  - OAK-D Lite (optional)        : stereo depth perception
Hardware:
  - Jetson Orin Nano Super 8GB
  - Waveshare UGV Beast (tracked, D500 LiDAR, OAK-D Lite)
  - ~$750 CAD total cost
  - Vancouver BC Canada
Usage:
  uv run main.py
  # Then open http://JETSON_IP:7860
Enable Nav2 + LiDAR + OAK-D:
  Set USE_NAV2=true, USE_LIDAR=true, USE_OAKD=true in .env
  Then: ros2 launch ugv_tools navigation.launch.py
"""
import atexit
import logging
import signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("eric")


# ── Graceful shutdown ─────────────────────────────────────────────────────────
# MUST be registered before init_nav2() is called so the executor is stopped
# cleanly before Python tears down daemon threads.  Without this the C++ layer
# inside rclpy fires std::terminate() when pthread_exit is called mid-flight.

def _graceful_shutdown(signum=None, frame=None):
    """
    Called on SIGINT, SIGTERM, or normal process exit via atexit.
    Stops Nav2 executor first, then lets Python continue its normal teardown.
    """
    try:
        from nav2 import shutdown as nav2_shutdown
        nav2_shutdown()
    except Exception as e:
        log.debug(f"Nav2 shutdown skipped ({e})")


atexit.register(_graceful_shutdown)
signal.signal(signal.SIGTERM, _graceful_shutdown)
# Leave SIGINT as default (KeyboardInterrupt) so Ctrl-C still works normally;
# atexit will fire nav2_shutdown() on exit anyway.


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("🤖 ERIC starting — Edge Robotics Innovation by Cosmos")

    from config import USE_NAV2, USE_LIDAR, USE_OAKD

    # ── Optional: ROS2 Nav2 ───────────────────────────────────────────────────
    if USE_NAV2:
        log.info("🗺️  Nav2 enabled — initializing ROS2...")
        from nav2 import init_nav2, nav2_available
        init_nav2()
        if nav2_available():
            log.info("✅ Nav2 ready — full autonomous navigation enabled")
        else:
            log.warning("⚠️  Nav2 unavailable — using direct motor control")

    # ── Optional: D500 LiDAR safety monitor ──────────────────────────────────
    if USE_LIDAR:
        log.info("📡 LiDAR enabled — initializing D500 safety monitor...")
        from lidar import init_lidar, lidar_available
        init_lidar()
        if lidar_available():
            log.info("✅ LiDAR safety monitor active")
        else:
            log.warning("⚠️  LiDAR unavailable — no hardware safety layer")

    # ── Optional: OAK-D Lite depth camera ────────────────────────────────────
    if USE_OAKD:
        log.info("📷 OAK-D Lite enabled — initializing stereo depth...")
        from oakd import init_oakd, oakd_available
        init_oakd()
        if oakd_available():
            log.info("✅ OAK-D depth perception active")
        else:
            log.warning("⚠️  OAK-D unavailable — no depth perception")

    # ── Cosmos connectivity test ──────────────────────────────────────────────
    from cosmos import ask_cosmos
    test = ask_cosmos("Say exactly: ERIC online and ready.", max_tokens=20)
    log.info(f"Cosmos test: {test}")

    # ── Launch Gradio GUI (blocking) ──────────────────────────────────────────
    try:
        from gui import launch
    except ImportError as e:
        import traceback
        traceback.print_exc()
        raise SystemExit(f"Failed to import gui.launch: {e}")

    launch()


if __name__ == "__main__":
    main()