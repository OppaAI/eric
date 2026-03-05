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
MOTOR_SPEED_SLOW   = float(os.getenv("MOTOR_SPEED_SLOW",   "0.22"))
MOTOR_SPEED_NORMAL = float(os.getenv("MOTOR_SPEED_NORMAL", "0.30"))
MOTOR_SPEED_FAST   = float(os.getenv("MOTOR_SPEED_FAST",   "0.50"))

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

# ─── ASR / Voice Pipeline (faster-whisper + silero-vad + ECAPA-TDNN) ────────────
ASR_MODEL              = os.getenv("ASR_MODEL",    "distil-small.en")  # distil-small.en | distil-medium.en | base | turbo
ASR_DEVICE             = os.getenv("ASR_DEVICE",   "cpu")              # cpu only — no VRAM conflict with Cosmos
ASR_LANGUAGE           = os.getenv("ASR_LANGUAGE", "en")               # en | fr | None (auto-detect)
ASR_SAMPLE_RATE        = int(os.getenv("ASR_SAMPLE_RATE", "16000"))
ASR_ENABLED            = os.getenv("ASR_ENABLED",  "true").lower() == "true"

# Wake word — comma-separated list, any match activates session
ASR_WAKE_WORDS         = [w.strip() for w in os.getenv("ASR_WAKE_WORDS", "hey eric,hi eric,eric").split(",")]
ASR_SESSION_TIMEOUT_SEC = int(os.getenv("ASR_SESSION_TIMEOUT_SEC", "150"))  # 2.5 min

# Speaker verification via ECAPA-TDNN (SpeechBrain)
# Set ASR_VERIFY_SPEAKER=true to require voice match before activating session
ASR_VERIFY_SPEAKER     = os.getenv("ASR_VERIFY_SPEAKER",  "false").lower() == "true"
ASR_SPEAKER_EMBEDDING  = os.getenv("ASR_SPEAKER_EMBEDDING", str(Path.home() / ".eric/speaker_embedding.pt"))
ASR_VERIFY_THRESHOLD   = float(os.getenv("ASR_VERIFY_THRESHOLD", "0.75"))  # 0-1, higher = stricter

# ─── ROS2 / Nav2 / LiDAR / OAK-D ─────────────────────────────────────────────
# Set USE_NAV2=true in .env to enable autonomous navigation via Nav2
# If false or ROS2 not running, Eric falls back to direct motor control
USE_NAV2          = os.getenv("USE_NAV2",    "false").lower() == "true"
USE_LIDAR         = os.getenv("USE_LIDAR",   "false").lower() == "true"
USE_OAKD          = os.getenv("USE_OAKD",    "false").lower() == "true"
LIDAR_STOP_DIST   = float(os.getenv("LIDAR_STOP_DIST", "0.30"))  # meters
LIDAR_SLOW_DIST   = float(os.getenv("LIDAR_SLOW_DIST", "0.60"))  # meters
