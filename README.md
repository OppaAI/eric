# E.R.I.C. — Edge Robotics Innovation by Cosmos

**NVIDIA Cosmos Cookoff 2026 Entry**

E.R.I.C. is a search and rescue ground robot powered by NVIDIA Cosmos Reason 2, running fully at the edge on a $750 CAD Jetson Orin Nano Super 8GB. No cloud. No server. Just a tracked robot reasoning about the physical world in real time.

## Demo

**Mission: Find Princess Leia**

Eric navigates a backyard Star Wars Lego scene, talks to characters, gathers information, and uses Cosmos Reason 2 to plan and reason at every step.

## Hardware

| Component | Details |
|---|---|
| SBC | Jetson Orin Nano Super 8GB |
| Robot chassis | Waveshare UGV Rover (tracked) |
| Camera 1 | Webcam (navigation) |
| Camera 2 | Pan-tilt camera |
| TTS | Piper (CPU, zero VRAM) |
| Total cost | ~$750 CAD |
| Location | Vancouver BC, Canada |

## Stack

- **Cosmos Reason 2 (2B W4A16)** via vLLM — vision + physical reasoning
- **Piper** via RealtimeTTS — streaming TTS, CPU only
- **Gradio** — dual camera live feed GUI + mission control
- **Waveshare ESP32** — motor control via serial UART

## Project Structure

```
eric/
├── main.py          # Entry point
├── config.py        # All configuration
├── cosmos.py        # Cosmos Reason 2 interface + camera
├── motors.py        # Waveshare serial motor control
├── tts.py           # Piper/gTTS text-to-speech
├── mission.py       # Mission state machine + logic
├── gui.py           # Gradio dual-camera UI
├── launch/
│   └── cosmos.sh    # vLLM Docker launch script
├── .env.example     # Environment template
└── pyproject.toml   # uv dependencies
```

## Quick Start

```bash
git clone https://github.com/OppaAI/eric
cd eric

# Install dependencies
uv sync

# Configure
cp .env.example .env
nano .env  # add your Telegram bot token

# Start Cosmos (vLLM)
bash launch/cosmos.sh

# Wait ~3 minutes for Cosmos to load, then:
uv run main.py

# Open GUI
http://JETSON_IP:7860
```

## GUI

```
┌─────────────────┬──────────────────────────┐
│  Pan-tilt feed  │  Mission Status          │
│                 │  Eric Says (top)         │
├─────────────────┤  Mission Briefing input  │
│  Webcam feed    │  [ENGAGE] [DISENGAGE]    │
│  (navigation)   │                          │
│                 │  Character Interaction   │
│                 │  (type as Lego figures)  │
└─────────────────┴──────────────────────────┘
```

## Mission Flow

1. Enter mission briefing (who to find, who might know, who to avoid)
2. Press ENGAGE — Eric acknowledges and starts moving
3. Eric scans scene every 3 seconds with Cosmos
4. When Eric stops at a character — type as that character in the UI
5. Eric reasons about the reply and decides next action
6. Mission continues until rescue target is located

## vLLM Performance (Jetson Orin Nano 8GB)

- Model: `embedl/Cosmos-Reason2-2B-W4A16`
- TPS: ~16-17 tokens/second
- GPU utilization: 0.75
- Startup time: ~3 minutes
- Vision inference: ~5-9 seconds per frame

## Built by

Solo developer, Vancouver BC Canada  
Built for the NVIDIA Cosmos Cookoff 2026
