# ERIC — Edge Robotics Innovation by Cosmos

**NVIDIA Cosmos Cookoff 2026 Entry**

ERIC is a search and rescue ground robot powered by NVIDIA Cosmos Reason 2, running fully at the edge on a ~$750 CAD Jetson Orin Nano Super 8GB. No cloud. No server. Just a tracked robot reasoning about the physical world in real time.

## Demo

**Mission: Find Princess Leia**

Eric navigates a backyard Star Wars Lego scene, talks to characters, gathers information, and uses Cosmos Reason 2 to plan and reason at every step — entirely on-device.

## Hardware

| Component | Details |
|---|---|
| SBC | Jetson Orin Nano Super 8GB |
| Robot | Waveshare UGV Rover (tracked) |
| Camera 1 | Webcam (navigation) |
| Camera 2 | Pan-tilt camera |
| TTS | Piper (CPU, zero VRAM) |
| Total | ~$750 CAD |
| Location | Vancover BC, Canada |

## Stack

| Component | Role |
|---|---|
| Cosmos Reason 2 (2B W4A16) via vLLM | Vision + physical reasoning |
| Piper via RealtimeTTS | Streaming TTS, CPU only |
| Gradio | Dual camera live feed + mission control UI |
| Waveshare ESP32 serial UART | Motor control |

## Project Structure

```
eric/
├── main.py               # Entry point
├── config.py             # All configuration (env vars)
├── cosmos.py             # Cosmos Reason 2 API + camera
├── motors.py             # Waveshare serial motor control + OLED
├── tts.py                # Piper / gTTS text-to-speech
├── mission.py            # Mission state machine + YAML loader
├── gui.py                # Gradio dual-camera UI
├── missions/             # Mission briefing files
│   ├── star_wars.yaml    # Find Princess Leia
│   ├── search_rescue.yaml # Lost hiker rescue
│   └── office_mystery.yaml # Missing USB drive
├── launch/
│   └── cosmos.sh         # vLLM Docker launch script
├── env.example           # Environment config template
└── pyproject.toml        # uv dependencies
```

## Quick Start

```bash
git clone https://github.com/OppaAi/eric
cd eric

# Install dependencies
uv sync

# Configure
cp env.example .env
nano .env   # set SERIAL_PORT, camera indices

# 1. Start Cosmos (takes ~3 minutes to load)
bash launch/cosmos.sh
docker logs -f vllm-server   # wait for "Application startup complete"

# 2. Start Eric
uv run main.py

# 3. Open the GUI
http://JETSON_IP:7860
```

## Missions

Missions are defined as YAML files in the `missions/` folder. Select one from the dropdown in the GUI, or type a briefing directly. Eric's entire reasoning adapts to whatever mission you give him.

**Example missions included:**

| File | Mission |
|---|---|
| `star_wars.yaml` | Find Princess Leia — navigate a Star Wars Lego scene |
| `search_rescue.yaml` | Find a missing hiker in backyard terrain |
| `office_mystery.yaml` | Locate a missing USB drive in an office |

**Mission YAML format:**
```yaml
name: "My Mission"
description: "Short description"

briefing: |
  Full mission briefing text.
  Eric reads this and uses it for all reasoning during the mission.
  Include: who to find, who might know, who/what to avoid.

characters:
  - name: "Character Name"
    hint: "How they behave"
```

**Add your own mission** — drop any `.yaml` file in `missions/` and it appears in the GUI dropdown instantly.

## GUI

```
┌─────────────────┬──────────────────────────────────┐
│  Pan-tilt feed  │  📋 Mission (dropdown + text box)│
│                 │  [🚀 ENGAGE]  [🛑 DISENGAGE]     │
├─────────────────│  Status bar                      │
│  Webcam feed    │  🔊 Eric Says                    │
│  (navigation)   │  💬 Character Interaction        │
│                 │  🕹️ Manual Controls              │
│                 │  📜 Mission Log                  │
└─────────────────┴──────────────────────────────────┘
```

**How to run a mission:**
1. Select a mission from the dropdown (or type your own briefing)
2. Press **ENGAGE** — Eric acknowledges and starts moving autonomously
3. Eric scans the scene every 3 seconds with Cosmos vision
4. When Eric stops at a character — type as that character in the interaction box
5. Eric reasons about the reply and decides next action based on mission briefing
6. Press **DISENGAGE** anytime to stop

## Cosmos Performance (Jetson Orin Nano 8GB)

| Metric | Value |
|---|---|
| Model | `embedl/Cosmos-Reason2-2B-W4A16` |
| TPS | ~16–17 tokens/second |
| Vision inference | ~5–9 seconds per frame |
| GPU utilization | 0.75 |
| Startup time | ~3 minutes |
| VRAM for model | ~2.3 GB |

## Dependencies

```bash
uv add python-telegram-bot requests python-dotenv \
        opencv-python-headless gtts pygame \
        RealtimeTTS pyserial pyyaml gradio
```

## Built by

Solo developer — Kelowna BC, Canada  
Built for the NVIDIA Cosmos Cookoff 2026  
https://github.com/OppaAi/eric
