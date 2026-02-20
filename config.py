"""
E.R.I.C. — Configuration
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path.home() / "AGi/ERIC/.env")

# ─── Telegram ─────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("ERIC_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

# ─── Cosmos (vLLM) ────────────────────────────────────────────────────────────
VLLM_URL     = "http://localhost:8000/v1/chat/completions"
COSMOS_MODEL = "embedl/Cosmos-Reason2-2B-W4A16"

# ─── Serial (Waveshare ESP32 via Jetson UART) ─────────────────────────────────
SERIAL_PORT = os.getenv("SERIAL_PORT", "/dev/ttyTHS1")
SERIAL_BAUD = 115200

MOTOR_SPEED_SLOW   = 0.15
MOTOR_SPEED_NORMAL = 0.30
MOTOR_SPEED_FAST   = 0.50

# ─── Camera ───────────────────────────────────────────────────────────────────
CAMERA_WEBCAM    = int(os.getenv("CAMERA_WEBCAM", "0"))
CAMERA_PANTILT   = int(os.getenv("CAMERA_PANTILT", "1"))
CAMERA_WIDTH     = 320
CAMERA_HEIGHT    = 240
SCAN_INTERVAL    = 3.0  # seconds between Cosmos scans during mission

# ─── TTS (Piper) ──────────────────────────────────────────────────────────────
PIPER_BINARY = str(Path.home() / "piper/piper")
PIPER_MODEL  = str(Path.home() / "piper/voices/en_US-lessac-medium.onnx")

# ─── Gradio ───────────────────────────────────────────────────────────────────
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))
GRADIO_HOST = os.getenv("GRADIO_HOST", "0.0.0.0")
