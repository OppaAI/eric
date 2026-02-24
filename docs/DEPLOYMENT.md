# ERIC — Deployment Guide

← [Back to README](../README.md)

---

## Requirements

**Hardware:**
- Jetson Orin Nano Super 8GB
- Waveshare UGV Beast (tracked robot with ESP32)
- YDLIDAR D500 *(optional — for LiDAR safety layer)*
- OAK-D Lite *(optional — for depth perception)*
- USB Webcam

**Software:**
- JetPack 6.2.2 (Ubuntu 22.04 · CUDA 12.6)
- Python 3.10+
- `uv` package manager
- Docker (for vLLM Cosmos container)
- ROS2 Humble *(optional — for LiDAR + Nav2)*

---

## Step 1 — Clone and Install

```bash
git clone https://github.com/OppaAi/eric
cd eric
uv sync
```

> `uv sync` reads `pyproject.toml` and installs all Python dependencies into a local virtual environment. No `pip install` needed.

---

## Step 2 — Configure Environment

```bash
cp .env.example .env
nano .env
```

```bash
# ── Serial (Waveshare UGV Beast ESP32) ───────────────────────
SERIAL_PORT=/dev/ttyTHS1
# check: ls /dev/tty* before and after USB plug

# ── Cosmos (vLLM) ────────────────────────────────────────────
VLLM_URL=http://localhost:8000/v1/chat/completions
COSMOS_MODEL=embedl/Cosmos-Reason2-2B-W4A16

# ── TTS (Piper) ───────────────────────────────────────────────
PIPER_BINARY=/home/YOUR_USER/piper/piper
PIPER_MODEL=/home/YOUR_USER/piper/voices/en_US-danny-low.onnx

# ── Cameras ───────────────────────────────────────────────────
# Find your camera indices:
# python3 -c "import cv2; [print(i, cv2.VideoCapture(i).read()[0]) for i in range(6)]"
CAMERA_WEBCAM=2
CAMERA_PANTILT=0

# ── Motor speeds ──────────────────────────────────────────────
MOTOR_SPEED_SLOW=0.22
MOTOR_SPEED_NORMAL=0.30
MOTOR_SPEED_FAST=0.50

# ── Safety thresholds ─────────────────────────────────────────
LIDAR_STOP_DIST=0.30    # metres — hard stop
LIDAR_SLOW_DIST=0.60    # metres — slow down

# ── Optional modules (set true only after ROS2 launch is running) ──
USE_NAV2=false
USE_LIDAR=false
USE_OAKD=false

# ── Gradio UI ─────────────────────────────────────────────────
GRADIO_PORT=7860
GRADIO_HOST=0.0.0.0
```

---

## Step 3 — Install Piper TTS

```bash
mkdir -p ~/piper && cd ~/piper
wget https://github.com/rhasspy/piper/releases/latest/download/piper_linux_aarch64.tar.gz
tar -xzf piper_linux_aarch64.tar.gz

mkdir -p voices && cd voices
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/danny/low/en_US-danny-low.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/danny/low/en_US-danny-low.onnx.json

# Test
echo "ERIC is online." | ~/piper/piper -m ~/piper/voices/en_US-danny-low.onnx --output_file /tmp/test.wav
aplay /tmp/test.wav
```

---

## Step 4 — Launch Cosmos vLLM

```bash
bash launch/cosmos.sh
```

Wait for startup (takes ~3 minutes on first load):

```bash
docker logs -f vllm-server
# Wait for: INFO:     Application startup complete.
```

Verify Cosmos is responding:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "embedl/Cosmos-Reason2-2B-W4A16",
    "messages": [{"role": "user", "content": "Say: ERIC online"}],
    "max_tokens": 20
  }' | python3 -m json.tool
```

---

## Step 5 — (Optional) Start ROS2 Nav2 + LiDAR

Only needed if `USE_LIDAR=true` or `USE_NAV2=true`:

```bash
# Terminal 1 — LiDAR
ros2 launch ugv_tools lidar.launch.py

# Terminal 2 — Nav2 + SLAM
ros2 launch ugv_tools navigation.launch.py

# Verify LiDAR is publishing
ros2 topic hz /scan           # should show ~10 Hz
ros2 topic echo /scan --once  # should show ranges array
```

> If ROS2 is not running and `USE_LIDAR=false`, Eric falls back to camera-only obstacle detection via Cosmos. All other features work normally.

---

## Step 6 — Start ERIC

```bash
uv run main.py
```

Expected output:

```
INFO eric: ERIC starting — Edge Robotics Innovation by Cosmos
INFO eric.lidar: LiDAR safety monitor active        ← if USE_LIDAR=true
INFO eric.oakd:  OAK-D depth perception active      ← if USE_OAKD=true
INFO eric: Cosmos test: ERIC online and ready.
INFO eric.gui:   Gradio UI launching on :7860
```

---

## Step 7 — Open the GUI

```
http://JETSON_IP:7860
```

Three columns:
- **Left:** Live pan-tilt camera + webcam + LiDAR arc distances + OAK-D depth grid
- **Centre:** Mission dropdown + ENGAGE/STOP + Eric speech + character comms + system log
- **Right:** Module status lights + motor telemetry + manual override (▲▼◀▶ + spin)

---

## Step 8 — Run a Mission

1. Select a mission from the dropdown (click ↺ to refresh if you added new YAMLs)
2. Read or edit the briefing
3. Press **ENGAGE**
4. When Eric stops at a character — type the character's name and what they say in CHARACTER COMMS, then click TRANSMIT
5. Press **STOP** at any time to halt all motors immediately

---

## Dependencies

```bash
# Python (managed by uv)
uv sync

# ROS2 Humble (if USE_LIDAR or USE_NAV2)
sudo apt install ros-humble-nav2-bringup ros-humble-rplidar-ros ros-humble-slam-toolbox \
                 python3-colcon-common-extensions

# System audio (for gTTS fallback and alarm tones)
sudo apt install python3-pygame portaudio19-dev

# OAK-D (if USE_OAKD)
pip install depthai
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | \
  sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

---

## Troubleshooting

| Problem | Check |
|---|---|
| `SERIAL_PORT not found` | `ls /dev/ttyTHS1` — check cable and: `sudo chmod 666 /dev/ttyTHS1` |
| Camera index wrong | `python3 -c "import cv2; [print(i, cv2.VideoCapture(i).read()[0]) for i in range(6)]"` |
| Cosmos timeout | `docker ps` — vLLM must be running. Check `docker logs vllm-server` |
| Piper TTS silent | Check `PIPER_BINARY` and `PIPER_MODEL` paths in `.env`. Run `aplay /tmp/test.wav` |
| LiDAR not publishing | `ros2 topic hz /scan` — check USB and relaunch `lidar.launch.py` |
| OAK-D not detected | `lsusb \| grep Luxonis` — try a different USB3 port |
| Eric spins but doesn't move | Try `MOTOR_SPEED_NORMAL=0.40` in `.env` |
| Mission YAML not showing | Click ↺ refresh. Check: `python3 -c "import yaml; yaml.safe_load(open('missions/your.yaml'))"` |
