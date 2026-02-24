# ERIC — Architecture

← [Back to README](../README.md)

---

## System Overview

```mermaid
flowchart TD
    COSMOS["🟢 NVIDIA COSMOS REASON 2\nMission Brain · Navigation · Avoidance · Conversations\nembedl/Cosmos-Reason2-2B-W4A16 via vLLM"]
    style COSMOS fill:#76b900,color:#000,stroke:#4a7a00,stroke-width:3px

    subgraph SENSORS["Sensor Inputs"]
        CAM2["Pan-tilt camera\nNavigation + 360° sweep"]
        CAM1["Webcam\nClose-up scanning"]
        LIDAR["D500 LiDAR\n360° arc map F/L/R/Rear"]
        OAKD["OAK-D Lite\n3×3 depth grid + floor-drop"]
    end

    subgraph AVOIDANCE["3-Layer Smart Avoidance"]
        AV1["Layer 1 — Instant\nmotors.stop() + backward\nNo Cosmos · No delay"]
        AV2["Layer 2 — Arc Scan\nLiDAR + OAK-D\npick_clearest_turn()"]
        AV3["🟢 Layer 3 — Cosmos\nCamera + all sensor data\nescape direction + turn_sec"]
        style AV3 fill:#76b900,color:#000,stroke:#4a7a00,stroke-width:2px
        AV1 --> AV2 --> AV3
    end

    subgraph SAFETY["Independent Safety — runs always"]
        LIDAR_MON["LiDAR monitor\n< 0.30m → hard stop\n< 0.60m → slow"]
        VOID["_void_check()\nOAK-D floor-drop\n+ LiDAR return sparsity\ngates every forward move"]
    end

    subgraph ALARM["Alarm System — alarm.py"]
        SIREN["🚨 SIREN\nRed strobe + rising tone"]
        HAZARD["⚠️ HAZARD\nAmber pulse + beep"]
        SUSP["🔴 SUSPICIOUS\nRed strobe + staccato"]
        NATURE["🌿 NATURE\nGreen pulse · no tone"]
    end

    MOTORS["ESP32 Motors — Waveshare UGV Beast\nSerial UART 115200 · JSON protocol"]
    LOGGER["logger.py\nActivity buffer · AI JSONL · Mission JSONL"]

    CAM1 & CAM2 --> COSMOS
    LIDAR --> LIDAR_MON & AV2
    OAKD --> VOID & AV2
    CAM2 --> AV3

    COSMOS -->|"nav decision + direction"| MOTORS
    COSMOS -->|"triggers avoidance"| AVOIDANCE
    COSMOS -->|"target confirmed"| ALARM
    AV3 -->|"escape turn"| MOTORS
    LIDAR_MON -->|"hard stop / slow"| MOTORS
    VOID -->|"void: stop + announce"| MOTORS
    ALARM -->|"TTS + LED + audio tone"| MOTORS
    COSMOS --> LOGGER
    MOTORS --> LOGGER
```

---

## Smart Avoidance — Cosmos as Escape Director

```mermaid
flowchart TD
    TRIGGER["Obstacle Detected\nLiDAR < 0.30m or Cosmos wall_ahead"]

    L1["LAYER 1 — Instant Hardware\nmotors.stop() + backward 1.5s\nNo Cosmos · No delay · < 100ms"]
    L2["LAYER 2 — Sensor Arc Scan\nLiDAR: front / left / right / rear arcs\nOAK-D: 3×3 depth grid\npick_clearest_turn() → best direction"]

    COSMOS_AV["🟢 LAYER 3 — COSMOS REASON 2\nINPUT: camera frame + LiDAR arcs + OAK-D grid\nOUTPUT: turn_left | turn_right | turn_back + exact turn_sec\nphysical_reasoning: 'Left arc 0.92m vs 0.18m front'"]
    style COSMOS_AV fill:#76b900,color:#000,stroke:#4a7a00,stroke-width:3px

    TIMEOUT{"Cosmos replied\nwithin 20s?"}
    TURN_C["Execute Cosmos direction\nfor Cosmos turn_sec"]
    TURN_A["Execute arc-based direction\n(escalating turn duration)"]
    VERIFY["Verify path clear\nLiDAR + OAK-D + quick visual scan"]
    CLEAR{"Path clear?"}
    RESUME["Resume forward motion"]
    RETRY["Retry — longer turn · attempt N+1"]
    FORCE360["Force full 360° scan\nMAX_AVOID_ATTEMPTS reached"]

    TRIGGER --> L1 --> L2 --> COSMOS_AV --> TIMEOUT
    TIMEOUT -->|"Yes"| TURN_C --> VERIFY
    TIMEOUT -->|"No"| TURN_A --> VERIFY
    VERIFY --> CLEAR
    CLEAR -->|"Yes"| RESUME
    CLEAR -->|"No, attempt < MAX"| RETRY --> L1
    CLEAR -->|"No, attempt = MAX"| FORCE360
```

---

## Mission State Machine

```mermaid
stateDiagram-v2
    [*] --> Idle : System start
    Idle --> Initialising : ENGAGE pressed
    Initialising --> Scanning : 🟢 Cosmos parses steps + initial 360 scan
    Scanning --> Reasoning : 🟢 Cosmos receives frames + sensor context
    Reasoning --> Moving : 🟢 Cosmos decides direction
    Reasoning --> Interacting : 🟢 Target spotted + eye-contact gate passed
    Moving --> VoidCheck : Before every forward move
    VoidCheck --> Moving : All clear
    VoidCheck --> Stopped : Void detected — back away
    Moving --> Scanning : Scan interval elapsed
    Moving --> Avoiding : LiDAR obstacle < 0.30m
    Avoiding --> Avoiding : Still blocked — 🟢 Cosmos picks new escape
    Avoiding --> Scanning : Path clear — resume
    Avoiding --> Scanning : Max attempts — force 360
    Interacting --> WaitingForInput : Eric speaks to character
    WaitingForInput --> Reasoning : Operator types character reply
    Reasoning --> AlarmFired : 🟢 Mission target confirmed
    AlarmFired --> Advancing : Next mission step
    Advancing --> Scanning : More steps remain
    Advancing --> MissionComplete : All steps done
    MissionComplete --> Idle : DISENGAGE
    Idle --> [*]
```

---

## Project Structure

```
eric/
├── main.py           # Entry point — init Nav2, LiDAR, OAK-D, Cosmos, GUI
├── config.py         # All config via .env
├── cosmos.py         # Cosmos API, camera capture, digital zoom, briefing
├── motors.py         # Waveshare serial: motors + OLED + LED + pan-tilt
├── tts.py            # Piper streaming TTS (CPU, zero VRAM) + gTTS fallback
├── mission.py        # Mission engine: state machine + steps + scans
├── alarm.py          # Multi-modal alert: TTS + LED strobe + pygame tones
├── logger.py         # Structured logging: activity buffer + AI JSONL
├── avoidance.py      # 3-layer smart avoidance — Cosmos as escape director
├── nav2.py           # ROS2 Nav2 integration (graceful fallback)
├── lidar.py          # D500 LiDAR: obstacle monitor + void detection
├── oakd.py           # OAK-D Lite: stereo depth + floor-drop detection
├── gui.py            # Gradio cockpit UI
├── missions/         # YAML mission files
├── logs/             # Auto-created — activity, AI, mission JSONL
├── missions/photos/  # Auto-created — timestamped find photos
└── launch/
    └── cosmos.sh     # vLLM Docker launch script
```

---

## Key Systems

### Void / Drop Detection (3 Layers)

| Layer | What it checks | Void signal |
|---|---|---|
| OAK-D `get_floor_drop()` | Depth at bottom strip of frame (y=0.85) vs mid-frame | Jump > 1.2m or < 5% valid returns |
| LiDAR `lidar_void_ahead()` | Valid return count in front 40° arc | < 15% return ratio = floor gone |
| 🟢 Cosmos `void_ahead` | Visual — lower third of every frame | Stair edge, floor texture ends, open air |

> **Why LiDAR silence = danger:** The D500 scans horizontally at ~20cm height. At a staircase top the laser beam falls through open air — zero returns. Old code treated `999m = clear`. Now, sparse returns = void.

### Terrain-Based Speed Control

Cosmos reports terrain type in every scan result. Eric maps it automatically:

| Tier | Examples | Speed |
|---|---|---|
| Fast | road, tile, floor, concrete, hardwood | `MOTOR_SPEED_FAST` |
| Normal | grass, gravel, dirt, path | `MOTOR_SPEED_NORMAL` |
| Slow | carpet, mud, rocks, slope, wet | `MOTOR_SPEED_SLOW` |
| Impassable | stairs, wall, gap, cliff, water | Full avoidance pipeline |

### Pan-Tilt 360° Scan

| | Old (chassis rotation) | New (pan-tilt sweep) |
|---|---|---|
| Mechanism | 8 × 45° full chassis turns | Pan-tilt sweep + 1 × 180° chassis turn |
| Frames captured | 16 | Up to 42 |
| Time | 45–90s | 15–25s |
| Drift | High | Minimal |

### Async Cosmos Calls

Cosmos calls run in a `ThreadPoolExecutor` with 2 workers — nav checks and 360° scan analysis can overlap. While Cosmos processes frames, Eric continues sensor reads, void checks, motor control, and GUI updates.

---

## Challenges

**Cosmos inference blocks everything.** 5–9 second calls block Python entirely. Solved with `ThreadPoolExecutor` (2 workers) and persistent `_CameraReader` daemon threads per camera that drain the V4L2 buffer continuously — otherwise V4L2 stalls and produces `select() timeout` errors during every Cosmos call.

**V4L2 camera stalls on JetPack 6.2.** `cv2.VideoCapture` fills its kernel buffer when nobody reads. Fix: one daemon thread per camera in a tight `cap.read()` loop, storing the latest frame in a 1-slot buffer. `capture_frame()` just grabs from the buffer. GStreamer pipeline (`v4l2src io-mode=2`, `appsink drop=1 max-buffers=1`) is tried first.

**Void detection was backwards.** Original LiDAR void check treated `999m (no return) = clear`. At a staircase top the laser falls through open air and returns nothing. Fix: count valid returns in the front arc — < 15% ratio = void.

**Motor direction is inverted.** UGV Beast negative speed = forward. Correction layer in `motors.py` so all callers use natural semantics (`forward()`, `backward()`).

**UART byte corruption on JetPack 6.2.** Commands sent as strings occasionally corrupted at the ESP32 end. Fix: send every command byte-by-byte with 1ms inter-byte delay.

**Wide-angle camera loses small objects.** Lego figures are only a few pixels wide at 640×480. Fix: `multi_zoom_scan()` — one wide frame for context plus 4 digitally cropped and upscaled strips in the same Cosmos call.

**Cosmos JSON output is inconsistent.** Sometimes markdown fences, sometimes explanation text, sometimes a list instead of an object. `_parse_json()` handles all cases: strips fences, finds `{...}`, handles array-wrapped objects, falls back to safe defaults rather than crashing the mission loop.

**TTS blocks motor control.** `feed(text).play()` blocks until audio finishes. Fix: dedicated background TTS worker thread and a 1-slot queue. `eric_say()` clears stale items and returns instantly.

**Alarm tones — no internet, no audio files.** All tones generated mathematically at runtime using `struct.pack` to build raw PCM waveforms at 22050 Hz, played via pygame. Zero external files, fully offline.
