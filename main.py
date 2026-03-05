"""
ERIC — Edge Robotics Innovation by Cosmos
================================================
NVIDIA Cosmos Cookoff 2026
Stack:
  - Cosmos Reason 2 (vLLM)       : vision + physical reasoning
  - Piper via RealtimeTTS        : streaming TTS, CPU only, zero VRAM
  - Waveshare UGV Beast          : tracked robot via serial UART → ESP32
  - Gradio                       : dual camera GUI + mission control
  - ROS2 Nav2 + SLAM Toolbox     : autonomous path planning + online mapping
  - D500 LiDAR                   : /scan for SLAM + reactive safety layer
  - OAK-D Lite                   : stereo depth + YOLO person detection
Hardware:
  - Jetson Orin Nano Super 8GB
  - Waveshare UGV Beast (tracked, D500 LiDAR, OAK-D Lite)
  - ~$750 CAD total cost
  - Vancouver BC Canada
Usage:
  source /opt/ros/humble/setup.bash
  uv run main.py
  # Then open http://JETSON_IP:7860
Enable Nav2 + LiDAR + OAK-D:
  Set USE_NAV2=true, USE_LIDAR=true, USE_OAKD=true in .env

ROS2 init order (enforced below):
  motors -> odom -> lidar -> slam -> oakd -> nav2
  motors owns UART; odom needs motors router; lidar publishes /scan;
  slam needs /odom + /scan; nav2 waits internally for slam_available().
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
    Shutdown order is the reverse of init order.
    """
    try:
        from nav2 import shutdown as nav2_shutdown
        nav2_shutdown()
    except Exception as e:
        log.debug(f"Nav2 shutdown skipped ({e})")
    try:
        from slam import shutdown_slam
        shutdown_slam()
    except Exception as e:
        log.debug(f"SLAM shutdown skipped ({e})")
    try:
        from ros_core import shutdown as ros_shutdown
        ros_shutdown()
    except Exception as e:
        log.debug(f"ROS2 core shutdown skipped ({e})")


atexit.register(_graceful_shutdown)
signal.signal(signal.SIGTERM, _graceful_shutdown)
# Leave SIGINT as default (KeyboardInterrupt) so Ctrl-C still works normally;
# atexit will fire _graceful_shutdown() on exit anyway.


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("ERIC starting — Edge Robotics Innovation by Cosmos")

    from config import USE_NAV2, USE_LIDAR, USE_OAKD

    # motors.py is always imported — it owns the UART port and must come first
    # so that odom.py can subscribe to its router.
    from motors import motors  # noqa: F401 — side effect: opens serial port

    # ── ROS2 stack — init order matters ──────────────────────────────────────
    # odom, lidar, slam, nav2 all share the single ros_core node.
    # Each module gracefully no-ops if ROS2 is not available.

    if USE_NAV2:
        log.info("Nav2 + SLAM enabled — initializing ROS2 stack...")

        # 1. Odometry — subscribes to UART T=1001, publishes /odom + TF
        from odom import init_odom, odom_available
        init_odom()
        if odom_available():
            log.info("Odometry: /odom publisher active")
        else:
            log.warning("Odometry unavailable — SLAM localisation degraded")

        # 2. LiDAR — publishes /scan (required for SLAM map building)
        #    Always init when Nav2 is on — SLAM cannot map without /scan.
        log.info("LiDAR initializing for SLAM /scan topic...")
        from lidar import init_lidar, lidar_available
        init_lidar()
        if lidar_available():
            log.info("LiDAR: /scan publishing — safety monitor active")
        else:
            log.warning("LiDAR unavailable — SLAM map building disabled")

        # 3. SLAM Toolbox — online async mapping from /scan + /odom
        from slam import init_slam, slam_available, _slam_ok as slam_node_ok
        init_slam()
        if slam_node_ok:
            log.info("SLAM Toolbox: node running — map building started")
        else:
            log.warning("SLAM unavailable — Nav2 will attempt direct motor control")

        # 4. OAK-D — stereo depth + /oakd/depth publisher for Nav2 costmap.
        #    Init before Nav2 so the depth topic exists when Nav2 configures
        #    its costmap layers.
        if USE_OAKD:
            log.info("OAK-D Lite initializing (SLAM mode — /oakd/depth -> Nav2 costmap)...")
            from oakd import (init_oakd, oakd_available,
                                set_yolo_callback, set_yolo_active, yolo_available)
            init_oakd()
            if oakd_available():
                log.info("OAK-D depth perception active — publishing /oakd/depth")
                _register_yolo_callback(set_yolo_callback, set_yolo_active, yolo_available)
            else:
                log.warning("OAK-D unavailable — no depth perception or /oakd/depth")

        # 5. Nav2 — launch the full Nav2 stack, then connect action client
        from ros_core import launch_nav2
        launch_nav2()
        from nav2 import init_nav2, nav2_available
        init_nav2()
        if nav2_available():
            log.info("Nav2 ready — full autonomous navigation + SLAM enabled")
        else:
            log.warning("Nav2 unavailable — using direct motor control")

    else:
        # Non-ROS mode — LiDAR and OAK-D still provide reactive safety
        if USE_LIDAR:
            log.info("LiDAR enabled (safety only, no ROS2)...")
            from lidar import init_lidar, lidar_available
            init_lidar()
            if lidar_available():
                log.info("LiDAR safety monitor active")
            else:
                log.warning("LiDAR unavailable — no hardware safety layer")

        if USE_OAKD:
            log.info("OAK-D Lite enabled (depth safety only, no SLAM)...")
            from oakd import (init_oakd, oakd_available,
                                set_yolo_callback, set_yolo_active, yolo_available)
            init_oakd()
            if oakd_available():
                log.info("OAK-D depth perception active")
                _register_yolo_callback(set_yolo_callback, set_yolo_active, yolo_available)
            else:
                log.warning("OAK-D unavailable — no depth perception")

    # ── Voice pipeline init ──────────────────────────────────────────────────
    from config import ASR_ENABLED
    if ASR_ENABLED:
        from voice import init_voice, start_voice_pipeline

        if init_voice():
            log.info("Voice: models loaded — starting pipeline...")

            def _on_utterance(text: str, is_wake: bool):
                """
                Route transcribed utterances to mission system.
                is_wake=True  → first activation (wake word heard)
                is_wake=False → active session utterance
                """
                if is_wake:
                    log.info(f"Voice: wake word activated — {text!r}")
                    return
                # Pass utterance to mission as if typed in GUI
                # During active mission: treat as character response
                # Outside mission: treat as new mission briefing command
                try:
                    # Check email commands first
                    from config import EMAIL_ENABLED
                    if EMAIL_ENABLED:
                        from email_handler import handle_voice_email_command
                        email_response = handle_voice_email_command(text)
                        if email_response:
                            from tts import speak
                            speak(email_response)
                            return

                    from mission import get_mission_active, handle_character_response, start_mission
                    if get_mission_active():
                        log.info(f"Voice: routing to character comms → {text!r}")
                        handle_character_response("Operator", text)
                    else:
                        log.info(f"Voice: routing as mission command → {text!r}")
                        start_mission(text)
                except Exception as e:
                    log.error(f"Voice: utterance routing error — {e}")

            def _on_state_change(state: str):
                log.debug(f"Voice state: {state}")
                # Forward to GUI status if available
                try:
                    from gui import set_voice_state
                    set_voice_state(state)
                except Exception:
                    pass

            start_voice_pipeline(
                on_utterance=_on_utterance,
                on_state_change=_on_state_change,
            )
        else:
            log.warning("Voice: init failed — voice input unavailable")

    # ── Email handler init ───────────────────────────────────────────────────
    from config import EMAIL_ENABLED
    if EMAIL_ENABLED:
        from email_handler import init_email, start_email_timer
        if init_email():
            start_email_timer()
            log.info("Email: handler ready — checking every 30 min")
        else:
            log.warning("Email: init failed — check ERIC_EMAIL_PASSWORD in .env")

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


def _register_yolo_callback(set_yolo_callback, set_yolo_active, yolo_available):
    """Register the YOLO detection callback and activate Layer 2 detection."""
    def _on_yolo_detection(label: str, dist_m: float,
                           bearing: str, bearing_deg: float):
        log.info(
            f"YOLO: {label} at {dist_m:.1f}m ({bearing}, {bearing_deg:+.1f} deg)"
        )
        # Forward to mission.py when ready:
        # from mission import on_detection
        # on_detection(label, dist_m, bearing, bearing_deg)

    set_yolo_callback(_on_yolo_detection)
    set_yolo_active(True)

    if yolo_available():
        log.info("OAK-D YOLO Layer 2 active — person/animal detection on Myriad X")
    else:
        log.warning("OAK-D YOLO unavailable — check blob file in models/")


if __name__ == "__main__":
    main()
