"""
ERIC — Configuration
All settings loaded from environment / .env file
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ─── Cosmos (vLLM) ────────────────────────────────────────────────────────────
VLLM_URL     = os.getenv("VLLM_URL", "http://localhost:8000/v1/chat/completions")
COSMOS_MODEL = os.getenv("COSMOS_MODEL", "embedl/Cosmos-Reason2-2B-W4A16")

# ─── Serial (Waveshare ESP32 via Jetson UART) ─────────────────────────────────
SERIAL_PORT        = os.getenv("SERIAL_PORT", "/dev/ttyTHS1")
SERIAL_BAUD        = 115200
MOTOR_SPEED_SLOW   = 0.15   # m/s
MOTOR_SPEED_NORMAL = 0.30   # m/s
MOTOR_SPEED_FAST   = 0.50   # m/s

# ─── Camera ───────────────────────────────────────────────────────────────────
CAMERA_WEBCAM  = int(os.getenv("CAMERA_WEBCAM",  "2"))
CAMERA_PANTILT = int(os.getenv("CAMERA_PANTILT", "0"))
CAMERA_WIDTH   = 640
CAMERA_HEIGHT  = 480
SCAN_INTERVAL  = 3.0   # seconds between Cosmos scans during mission

# ─── TTS (Piper) ──────────────────────────────────────────────────────────────
PIPER_BINARY = os.getenv("PIPER_BINARY", str(Path.home() / "piper/piper"))
PIPER_MODEL  = os.getenv("PIPER_MODEL",  str(Path.home() / "piper/voices/en_US-danny-low.onnx"))

# ─── Missions ─────────────────────────────────────────────────────────────────
MISSIONS_DIR = Path(__file__).parent / "missions"

# ─── Gradio ───────────────────────────────────────────────────────────────────
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))
GRADIO_HOST = os.getenv("GRADIO_HOST", "0.0.0.0")

# ─── ROS2 / Nav2 / LiDAR ──────────────────────────────────────────────────────
# Set USE_NAV2=true in .env to enable autonomous navigation via Nav2
# If false or ROS2 not running, Eric falls back to direct motor control
USE_NAV2          = os.getenv("USE_NAV2",    "false").lower() == "true"
USE_LIDAR         = os.getenv("USE_LIDAR",   "false").lower() == "true"
LIDAR_STOP_DIST   = float(os.getenv("LIDAR_STOP_DIST", "0.30"))  # meters
LIDAR_SLOW_DIST   = float(os.getenv("LIDAR_SLOW_DIST", "0.60"))  # meters
