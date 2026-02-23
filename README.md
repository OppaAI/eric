# ERIC — Edge Robotics Innovation by Cosmos

**NVIDIA Cosmos Cookoff 2026 Entry**

ERIC is a search and rescue ground robot powered by **NVIDIA Cosmos Reason 2 (2B W4A16)**, running fully at the edge on a ~$750 CAD Jetson Orin Nano Super 8GB. No cloud. No server. Just a tracked robot reasoning about the physical world in real time.

**Cosmos Reason 2 is the mission brain** — it sees the world through Eric's cameras, reasons about what it finds, decides where to go, and now actively directs the escape route when obstacles are hit. The D500 LiDAR and OAK-D Lite provide an independent safety layer. ROS2 Nav2 handles path planning when enabled.

## Demo

**Mission: Find Princess Leia**

Eric navigates a Star Wars Lego scene autonomously — scanning with dual cameras, reasoning with Cosmos Reason 2 about what it sees, talking to characters to gather mission information, and manoeuvring around obstacles using the new 3-layer avoidance system. Entire demo recorded from the Gradio GUI screen — no outdoor filming needed. Judges see both live camera feeds, reasoning, motor telemetry, and LiDAR status simultaneously.

## Hardware

| Component | Details |
|---|---|
| SBC | Jetson Orin Nano Super 8GB |
| Robot | Waveshare UGV Beast (tracked) |
| LiDAR | D500 (360°, reactive obstacle safety + arc map) |
| Depth Camera | OAK-D Lite (3D perception, 3×3 depth grid) |
| Camera 1 | Webcam (close-up scanning) |
| Camera 2 | Pan-tilt wide-angle (navigation + overview) |
| TTS | Piper danny-low (CPU, zero VRAM) |
| Total | ~$750 CAD |
| Location | Vancouver BC, Canada |

## Stack

| Component | Role |
|---|---|
| **Cosmos Reason 2 (2B W4A16) via vLLM** | **Vision + physical reasoning — mission brain + avoidance director** |
| ROS2 Humble + Nav2 | Autonomous path planning (optional) |
| D500 LiDAR → ROS2 /scan | Reactive obstacle safety + 4-arc distance map for avoidance |
| OAK-D Lite | Stereo depth — 3×3 grid fed to Cosmos + avoidance planner |
| avoidance.py | 3-layer smart avoidance: backup → arc scan → Cosmos decision |
| Piper via RealtimeTTS | Streaming TTS, CPU only, zero VRAM |
| Gradio | Dual camera + LiDAR status + mission control UI |
| Waveshare ESP32 serial UART | Motor + OLED + LED + pan-tilt control |

---

## Architecture

### System Overview

```mermaid
flowchart TD
    COSMOS["🧠  COSMOS REASON 2 — 2B W4A16\n══════════════════════════════\nMISSION BRAIN + AVOIDANCE DIRECTOR\nSees  ·  Reasons  ·  Decides  ·  Escapes"]

    subgraph SENSORS["Sensor Layer"]
        LIDAR["📡 D500 LiDAR\n360° /scan topic\nArc map: F / L / R / Rear"]
        OAKD["📷 OAK-D Lite\nStereo Depth\n3×3 depth grid"]
        CAM1["🎥 Webcam\nClose-up"]
        CAM2["🎥 Pan-tilt Cam\nWide angle"]
    end

    subgraph AVOIDANCE["🚧 Smart Avoidance — avoidance.py"]
        AV1["Layer 1 · Instant backup\nNo Cosmos needed"]
        AV2["Layer 2 · LiDAR arc scan\nPick clearest direction"]
        AV3["Layer 3 · Cosmos decides\nCamera + all sensor data"]
        AV1 --> AV2 --> AV3
    end

    subgraph SAFETY["⚠️ Independent Safety Layer"]
        LIDAR_MON["LiDAR Safety Monitor\nlidar.py — instant stop / slow"]
        OAKD_MON["OAK-D Depth Monitor\noakd.py"]
    end

    subgraph NAV["Navigation Layer"]
        NAV2["🗺️ ROS2 Nav2\nPath Planning + SLAM\n(optional)"]
        DIRECT["⚡ Direct Motor Control\n(fallback)"]
    end

    MOTORS["🤖 ESP32 Motors — Waveshare UGV Beast UART"]

    CAM1 --> COSMOS
    CAM2 --> COSMOS
    OAKD --> OAKD_MON
    LIDAR --> LIDAR_MON
    LIDAR --> AV2
    OAKD --> AV2
    CAM2 --> AV3

    COSMOS -->|"goal pose / direction"| NAV2
    COSMOS -->|"fallback: move cmds"| DIRECT
    COSMOS -->|"triggers avoidance"| AVOIDANCE
    AV3 -->|"Cosmos escape decision\nturn_left/right/back + turn_sec"| MOTORS

    NAV2 -->|cmd_vel| MOTORS
    DIRECT --> MOTORS

    LIDAR_MON -->|"< 0.30m → STOP · < 0.60m → SLOW"| MOTORS
    OAKD_MON -->|"depth context"| COSMOS
    AVOIDANCE --> MOTORS

    style COSMOS fill:#76b900,stroke:#5a8a00,color:#000000,font-weight:bold
    style AV3 fill:#4a7a00,stroke:#76b900,color:#ffffff
    style SAFETY fill:#3a1a1a,stroke:#cc4444
    style AVOIDANCE fill:#1a2a1a,stroke:#76b900
```

### Smart Obstacle Avoidance Pipeline

```mermaid
flowchart TD
    TRIGGER["🚧 Obstacle Detected\nLiDAR < 0.30m  or  visual scan hits wall"]

    L1["⚡ LAYER 1 — Instant Hardware Reaction\nmotors.backward()  ×  1.5 s\nNo Cosmos · No delay · Always runs first"]

    L2["📡 LAYER 2 — Sensor Arc Scan\nD500 LiDAR: front / left / right / rear arcs\nOAK-D: 3×3 depth grid\npick_clearest_turn() → best escape direction"]

    COSMOS_AV["🧠  COSMOS REASON 2 — 2B W4A16\n══════════════════════════════\nINPUT: camera frame + LiDAR arc map + OAK-D grid\nOUTPUT: turn_left | turn_right | turn_back | forward\n         + turn_sec  (exactly how long to turn)"]

    COSMOS_WIN{"Cosmos replied\nwithin 20 s?"}

    TURN_COSMOS["▶ Execute Cosmos direction\n  for Cosmos turn_sec"]
    TURN_ARC["▶ Execute arc-based direction\n  escalating turn duration"]

    VERIFY["🔍 Verify path clear\nLiDAR front arc + OAK-D depth + quick visual scan"]

    CLEAR{"Path\nclear?"}

    RESUME["✅ Resume forward motion\nReset attempt counter"]
    RETRY["↩ Retry — longer turn\nattempt N+1"]
    FORCE360["🔄 Force full 360° scan\nMAX_AVOID_ATTEMPTS reached"]

    TRIGGER --> L1 --> L2 --> COSMOS_AV
    COSMOS_AV --> COSMOS_WIN
    COSMOS_WIN -->|"Yes"| TURN_COSMOS
    COSMOS_WIN -->|"No / timeout"| TURN_ARC
    TURN_COSMOS --> VERIFY
    TURN_ARC --> VERIFY
    VERIFY --> CLEAR
    CLEAR -->|"Yes"| RESUME
    CLEAR -->|"No · attempt < MAX"| RETRY
    CLEAR -->|"No · attempt = MAX"| FORCE360
    RETRY --> L1

    style COSMOS_AV fill:#76b900,stroke:#5a8a00,color:#000000,font-weight:bold
    style TRIGGER fill:#3a0000,stroke:#cc0000,color:#ffffff
    style RESUME fill:#0a2a0a,stroke:#76b900,color:#ffffff
    style FORCE360 fill:#2a1a00,stroke:#ff6600,color:#ffffff
    style L1 fill:#1a1a2a,stroke:#4444cc,color:#ffffff
    style L2 fill:#1a2a2a,stroke:#44aacc,color:#ffffff
```

### Obstacle Safety — Full Decision Flow

```mermaid
flowchart LR
    SCAN["D500 /scan\n360° reading"]
    ARC_FRONT["Extract front arc\n±60° = 120° total"]
    MIN["min_distance\nin front arc"]

    MIN -->|"< 0.30 m"| STOP["🛑 motors.stop()\nHARD STOP\navoidance.py takes over"]
    MIN -->|"0.30 – 0.60 m"| SLOW["🐢 motors.slow()\nREDUCE SPEED"]
    MIN -->|"> 0.60 m"| CLEAR_ACT["✅ No action"]

    STOP --> AVPIPE["avoidance.py\nBackup · Arc scan · Cosmos"]
    AVPIPE --> COSMOS_D["🧠 COSMOS REASON 2\n2B W4A16\nDecides escape route"]
    COSMOS_D --> ESCAPE["Execute turn\nResume mission"]

    CLEAR_ACT --> COSMOS_NAV["🧠 COSMOS REASON 2\n2B W4A16\nDrives mission forward"]

    SCAN --> ARC_FRONT --> MIN

    style STOP fill:#3a0000,stroke:#cc0000,color:#ffffff
    style SLOW fill:#2a1a00,stroke:#ff6600,color:#ffffff
    style CLEAR_ACT fill:#0a2a0a,stroke:#76b900,color:#ffffff
    style COSMOS_D fill:#76b900,stroke:#5a8a00,color:#000000,font-weight:bold
    style COSMOS_NAV fill:#76b900,stroke:#5a8a00,color:#000000,font-weight:bold
```

### Mission State Machine

```mermaid
stateDiagram-v2
    [*] --> Idle : System start
    Idle --> Initialising : ENGAGE pressed

    Initialising --> Scanning : 360° initial scan

    Scanning --> Reasoning : Cosmos Reason 2 receives\nframes + sensor context
    Reasoning --> Moving : Cosmos decides direction
    Reasoning --> Interacting : Target / character spotted

    Moving --> Scanning : SCAN_INTERVAL elapsed
    Moving --> Avoiding : LiDAR obstacle < 0.30m

    Avoiding --> Avoiding : Still blocked — retry\nCosmos picks escape route each time
    Avoiding --> Scanning : Path clear — resume
    Avoiding --> Scanning : MAX attempts → force 360°

    Interacting --> WaitingForInput : Eric speaks to character
    WaitingForInput --> Reasoning : User types character reply
    Reasoning --> MissionComplete : Objective confirmed by Cosmos

    MissionComplete --> Idle : DISENGAGE
    Idle --> [*]
```

---

## Sequence Diagrams

### Startup Sequence

```mermaid
sequenceDiagram
    participant User
    participant main.py
    participant Nav2
    participant LiDAR
    participant OAKD as OAK-D Lite
    participant Cosmos as ★ Cosmos Reason 2 (2B W4A16) ★
    participant GUI

    User->>main.py: uv run main.py
    main.py->>Nav2: init_nav2() [if USE_NAV2]
    Nav2-->>main.py: ✅ connected / ⚠️ fallback
    main.py->>LiDAR: init_lidar() [if USE_LIDAR]
    LiDAR-->>main.py: ✅ /scan subscribed + arc map ready
    main.py->>OAKD: init_oakd() [if USE_OAKD]
    OAKD-->>main.py: ✅ stereo depth active
    main.py->>Cosmos: ask_cosmos("ERIC online and ready.")
    Cosmos-->>main.py: "ERIC online and ready."
    main.py->>GUI: launch() [Gradio on :7860]
    GUI-->>User: Browser opens
```

### Mission Loop — One Reasoning Cycle

```mermaid
sequenceDiagram
    participant GUI
    participant Mission
    participant Cosmos as ★ Cosmos Reason 2 (2B W4A16) ★
    participant Avoidance as avoidance.py
    participant Motors
    participant LiDAR
    participant TTS

    GUI->>Mission: ENGAGE (mission selected)
    Mission->>Motors: 360° rotation scan
    Mission->>Cosmos: capture_frame() + NAV_PROMPT + sensor_context()
    Cosmos-->>Mission: reasoning + move decision (JSON)

    alt Move forward — path clear
        Mission->>LiDAR: obstacle_close()?
        LiDAR-->>Mission: false
        Mission->>Motors: forward()
    else Obstacle detected
        LiDAR->>Motors: stop() ← instant safety
        Mission->>Avoidance: avoid_obstacle(wall_ahead=True)
        Note over Avoidance: Layer 1: backward(1.5s)
        Avoidance->>LiDAR: get_arc_distances() F/L/R/Rear
        LiDAR-->>Avoidance: arc distance map
        Avoidance->>Cosmos: frame + arcs + OAK-D grid
        Cosmos-->>Avoidance: turn_right, turn_sec=2.1s
        Avoidance->>Motors: right(2.1s) → stop → verify
        Avoidance-->>Mission: path clear ✅
    end

    Mission->>Cosmos: capture_frame() [after SCAN_INTERVAL]
    Cosmos-->>Mission: character spotted!
    Mission->>Motors: stop()
    Mission->>TTS: speak("Greetings, I am ERIC...")
    Mission->>GUI: await character_input
    GUI-->>Mission: user typed character response
    Mission->>Cosmos: evaluate_response(character_text)
    Cosmos-->>Mission: has_info=true / objective_found=true
    Mission->>TTS: speak("Mission complete!")
```

### Avoidance Deep-Dive — Cosmos as Escape Director

```mermaid
sequenceDiagram
    participant LiDAR
    participant OAKD as OAK-D Lite
    participant Avoidance as avoidance.py
    participant Cosmos as ★ Cosmos Reason 2 (2B W4A16) ★
    participant Motors
    participant Mission

    Note over LiDAR,Mission: Obstacle detected — avoidance.py called

    Avoidance->>Motors: stop() + backward(1.5s)
    Note over Avoidance: Layer 1 complete — Eric is safe

    par Read all sensors simultaneously
        Avoidance->>LiDAR: get_arc_distances()
        LiDAR-->>Avoidance: front=0.18m left=0.92m right=0.41m rear=1.2m
    and
        Avoidance->>OAKD: get_depth_map()
        OAKD-->>Avoidance: 3×3 depth grid
    end

    Note over Avoidance: Layer 2: pick_clearest_turn() → "left" (0.92m clearance)

    Avoidance->>Cosmos: camera frame + arc map + OAK-D grid\n+ "Clearest sensor direction: left"
    Note over Cosmos: Sees image, reads real metric distances,\nreasons about best escape route
    Cosmos-->>Avoidance: action=turn_left  turn_sec=1.8s\nreasoning="Left has 0.92m — most clearance"

    Avoidance->>Motors: left(1.8s) → stop

    Avoidance->>LiDAR: min_front_distance()
    LiDAR-->>Avoidance: 1.4m ✅ clear
    Avoidance-->>Mission: return False (no 360 needed)
    Mission->>Motors: forward() — resume mission
```

### TTS Pipeline

```mermaid
sequenceDiagram
    participant Mission
    participant speak()
    participant Queue
    participant Worker
    participant Piper
    participant gTTS

    Mission->>speak(): speak("Hello world")
    speak()->>Queue: clear stale items
    speak()->>Queue: put(text)
    speak()-->>Mission: returns instantly (non-blocking)

    loop Background worker
        Worker->>Queue: get(timeout=1s)
        Queue-->>Worker: text

        alt Piper available
            Worker->>Piper: feed(text).play()
            Piper-->>Worker: audio complete (blocking)
        else gTTS fallback
            Worker->>gTTS: gTTS(text)
            gTTS-->>Worker: .mp3 via pygame
        end

        Worker->>Queue: task_done()
    end
```

---

## Project Structure

```
eric/
├── main.py                   # Entry point — initializes Nav2, LiDAR, Cosmos, GUI
├── config.py                 # All configuration (env vars + ROS2 flags)
├── cosmos.py                 # Cosmos Reason 2 API, camera, digital zoom crop
├── motors.py                 # Waveshare serial motor control + OLED + LED
├── tts.py                    # Piper streaming TTS (CPU, zero VRAM)
├── mission.py                # Mission state machine + YAML loader
├── avoidance.py              # 3-layer smart avoidance — Cosmos as escape director (NEW)
├── nav2.py                   # ROS2 Nav2 integration (graceful fallback)
├── lidar.py                  # D500 LiDAR safety monitor + raw scan → arc map
├── oakd.py                   # OAK-D Lite stereo depth + 3×3 depth grid
├── gui.py                    # Gradio dual-camera + LiDAR status UI
├── missions/
│   ├── template.yaml         # Start here — fully commented
│   ├── star_wars.yaml        # Find Princess Leia
│   ├── anakin_training.yaml  # Eric IS Anakin, faces dark side choice
│   ├── search_rescue.yaml    # Lost hiker rescue
│   └── office_mystery.yaml   # Missing USB drive
├── launch/
│   └── cosmos.sh             # vLLM Docker launch script
├── env.example               # Environment config template
└── pyproject.toml            # uv dependencies
```

## Quick Start

```bash
git clone https://github.com/OppaAi/eric
cd eric
uv sync

cp env.example .env
nano .env   # set SERIAL_PORT, camera indices, Piper paths

# 1. Start Cosmos vLLM (takes ~3 minutes to load)
bash launch/cosmos.sh
docker logs -f vllm-server   # wait for "Application startup complete"

# 2. (Optional) Start ROS2 Nav2 + LiDAR
ros2 launch ugv_tools navigation.launch.py   # Nav2 + SLAM
ros2 launch ugv_tools lidar.launch.py        # D500 LiDAR

# 3. Start Eric
uv run main.py

# 4. Open GUI
http://JETSON_IP:7860
```

## .env Configuration

```bash
# Required
SERIAL_PORT=/dev/ttyTHS1
PIPER_BINARY=/home/oppa-ai/piper/piper
PIPER_MODEL=/home/oppa-ai/piper/voices/en_US-danny-low.onnx

# Camera indices (check with: python3 -c "import cv2; [print(i, cv2.VideoCapture(i).read()[0]) for i in range(4)]")
CAMERA_WEBCAM=2
CAMERA_PANTILT=0

# Optional Nav2 + LiDAR + OAK-D (set true after ros2 launch)
USE_NAV2=false
USE_LIDAR=false
USE_OAKD=false
LIDAR_STOP_DIST=0.30
LIDAR_SLOW_DIST=0.60
```

## GUI

```
┌─────────────────────────────────────────────────────┐
│              🚨 EMERGENCY STOP 🚨                   │
├──────────────────┬──────────────────────────────────┤
│  Pan-tilt feed   │  📋 Mission (dropdown / text)    │
│  (wide angle)    │  [🚀 ENGAGE]  [🛑 DISENGAGE]    │
├──────────────────┤  Status                          │
│  Webcam feed     │  🔊 Eric Says                   │
│  (close-up)      │  💬 Character Interaction        │
├──────────────────┤  🕹️ Manual Controls             │
│  🚗 Telemetry   │  🛠️ Utilities                   │
│  📡 LiDAR       │  📜 Mission Log                  │
└──────────────────┴──────────────────────────────────┘
```

## Wide-Angle Camera & Object Detection

The pan-tilt camera is wide-angle — small objects like Lego figures can be hard for Cosmos to identify. Eric handles this automatically:

1. **Wide scan first** — Cosmos Reason 2 sees full scene context
2. **Digital zoom crop** — if something detected, `capture_zoomed()` crops and zooms that region
3. **Multi-zoom scan** — `multi_zoom_scan()` sends 1 wide + 4 cropped frames in one Cosmos call
4. **Webcam close-up** — when stopped and interacting, webcam gives high-detail close-up view

## Missions

Missions are YAML files in `missions/`. Select from dropdown — Cosmos Reason 2's reasoning adapts completely to the scenario.

| File | Mission |
|---|---|
| `star_wars.yaml` | Find Princess Leia in a Star Wars Lego scene |
| `anakin_training.yaml` | Eric IS Anakin Skywalker, faces the dark side choice |
| `search_rescue.yaml` | Find a missing hiker |
| `office_mystery.yaml` | Locate a missing USB drive |

**Create your own:** copy `missions/template.yaml` — no coding required.

## How a Mission Works

1. Select mission → press **ENGAGE**
2. Eric does initial 360° scan — Cosmos Reason 2 analyses all frames
3. Cosmos analyses camera frames + sensor context while Eric moves
4. LiDAR safety monitor runs independently — stops Eric if wall within 30cm
5. If blocked, `avoidance.py` fires: backup → LiDAR arc scan → **Cosmos Reason 2 picks escape route**
6. Nav2 handles path planning around obstacles (when enabled)
7. Eric stops at characters — type as character in GUI
8. Cosmos evaluates response, gets info, politely exits if off-topic
9. Mission continues until objective found

## Smart Obstacle Avoidance

`avoidance.py` implements a 3-layer pipeline. Cosmos Reason 2 is the escape director at Layer 3.

| Layer | What happens | Latency |
|---|---|---|
| **1 — Instant backup** | `motors.stop()` + `motors.backward(1.5s)` — no Cosmos, no delay | Immediate |
| **2 — Sensor arc scan** | LiDAR reads front/left/right/rear arcs + OAK-D 3×3 depth grid → `pick_clearest_turn()` | ~50 ms |
| **3 — Cosmos Reason 2** | Camera frame + arc map + OAK-D grid → `turn_left/right/back` + exact `turn_sec` | 5–9 s |

After the turn, `_path_is_clear()` cross-checks LiDAR + OAK-D + a quick visual scan. Still blocked → retry with longer turn. After `MAX_AVOID_ATTEMPTS` (3) → force full 360° scan.

**Small obstacles** get a dedicated step-around (right → forward → left arc) before escalating.

**Cosmos timeout fallback:** if Cosmos takes > 20s, the arc-based direction runs instead — Eric is never left stuck waiting.

## Cosmos Reason 2 — Every Role It Plays

| Situation | Cosmos Reason 2 input | Cosmos Reason 2 output |
|---|---|---|
| Moving — nav check | Camera frame + sensor context | `forward` or `stop` |
| Stopped scan | Dual camera frames + sensor data | Target / terrain / action JSON |
| 360° scan | 16 frames (8 pos × 2 tilts) | Best direction + target info |
| **Obstacle hit** | **Camera + LiDAR arcs + OAK-D grid** | **Escape direction + turn_sec** |
| Character spotted | Scene description | Greeting + mission question |
| Character replies | Conversation history | Continue or move on |
| Mission complete | Target confirmed | Triumphant announcement |

## Cosmos Performance (Jetson Orin Nano 8GB)

| Metric | Value |
|---|---|
| Model | `embedl/Cosmos-Reason2-2B-W4A16` |
| TPS | ~16–17 tokens/second |
| Vision inference | ~5–9 seconds per frame |
| GPU utilization | 0.75 |
| Startup time | ~3 minutes |
| VRAM for model | ~2.3 GB |
| Avoidance call timeout | 20 s (arc fallback if exceeded) |

## Dependencies

```bash
uv add gradio pyyaml pyserial requests python-dotenv \
        opencv-python-headless gtts pygame RealtimeTTS
# ROS2 Nav2 and LiDAR via apt (ros-humble-nav2-*, ros-humble-rplidar-ros)
```

## Built by

Solo developer — Vancouver BC, Canada
Built for the NVIDIA Cosmos Cookoff 2026
https://github.com/OppaAi/eric