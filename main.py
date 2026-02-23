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

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("eric")


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
    from gui import launch
    launch()


if __name__ == "__main__":
    main()