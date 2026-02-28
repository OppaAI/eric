"""
Run odom.py standalone for testing.
Usage: uv run run_odom.py
"""
import time
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

from odom import init_odom, get_status

if init_odom():
    print("✅ Odometry running — publishing /odom")
    try:
        while True:
            s = get_status()
            print(f"\r  x={s['x']:6.3f}  y={s['y']:6.3f}  θ={s['theta_deg']:6.1f}°  ", end="", flush=True)
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nStopped")
else:
    print("❌ Odometry failed to start — check ROS2 and UART")
