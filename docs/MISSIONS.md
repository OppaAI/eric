# ERIC — Mission System

← [Back to README](../README.md)

---

Missions are YAML files in the `missions/` folder. Drop a new `.yaml` file in, click ↺ refresh in the GUI, and it appears instantly. No code required. Cosmos reads the briefing as plain English — no structured format needed.

---

## How It Works

When you press ENGAGE:

1. **Cosmos parses the briefing** into an ordered list of `MissionStep` objects — targets, sequencing, and action types extracted from natural language
2. **KV cache warm-up** fires a tiny pre-fill request so all subsequent Cosmos calls this mission pay only the incremental token cost
3. **Eric executes steps sequentially** — advancing only when each one is confirmed complete
4. The **`alarm_type` field** controls what fires on a find — LED pattern, audio tone, TTS prefix, follow-on behaviour
5. At mission end, **`_mission_report()`** delivers a summary of all finds

### Multi-Step Example

```
Briefing: "First find R2-D2 and speak to him. Then find Luke and wait for his response.
           Finally, locate Princess Leia and photograph her."

→ Step 1 of 3: target=R2-D2  action=speak_to
→ Step 2 of 3: target=Luke   action=wait_for_response
→ Step 3 of 3: target=Leia   action=photograph

Eric will not advance to Step 2 until Step 1 is complete.
```

---

## YAML Schema

```yaml
# Required
name: "Mission Name"
briefing: |
  Your full mission briefing here. Cosmos reads every word.
  Write it like you're briefing a human agent.

# Recommended
description: "One line for logs and README"
author: "Your name"

# Alarm type — controls what fires when a target is found
alarm_type: none   # none | hazard | siren | suspicious | nature

# What Cosmos watches for in every frame
target_objects:
  - person
  - robot

# Behaviour on find
photo_on_find:     false  # save timestamped photo to missions/photos/ (dual-cam, blur-checked)
announce_location: false  # TTS location announcement (nature missions: respects wildlife)
stay_with_target:  false  # SAR: stay and repeat broadcast every 15s
back_away_on_find: false  # security: back 3m + turn 180°
generate_report:   false  # mission end: summary of all finds

# 360° scan strategy — controls which scan mode Eric uses
scan_strategy: target_hunt   # target_hunt (default) | video_sweep
# target_hunt: async per-position pan-tilt sweep, early-exit on first confirmed target
#              Best for: search & rescue, find missions, security
# video_sweep: continuous chassis rotation + single panoramic video inference, no early-exit
#              Best for: nature explorer, inspection, patrol, survey

# Approach behaviour
approach_distance: 0.65   # metres — OAK-D/YOLO distance threshold for "arrived" (default 0.65m)
approach_on_detect: true  # YOLO: approach on detection (false = report only, stay in place)
detect_distance: 2.0      # metres — YOLO distance above which Eric steers toward (not stops)

# Obstacle circumnavigation (experimental)
circumnavigate_on_empty: false   # peek around blocking obstacle before doing full 360
circum_step_sec: 1.8             # seconds to side-step (tune for room size)
circum_dist_m: 0.4               # estimated step distance in metres
circum_forward_sec: 0.0          # optional forward nudge before side-stepping

# Narrative mission behaviour
wait_for_dismiss: false   # stay in place after greeting until operator presses STOP

# Characters (played by operator in GUI)
characters:
  - name: "R2-D2"
    hint: "Speaks in beeps. Knows where Luke is. Will help if asked nicely."

# Stage-by-stage goals — Cosmos sees the current stage goal during that step
mission_stages:
  - goal: "Find R2-D2 and ask him where Luke is hiding"
  - goal: "Find Luke and brief him on the mission"
  - goal: "Locate Princess Leia using Luke's directions"

# Terrain Eric will encounter (affects speed)
terrain:
  - "Smooth tile (normal speed)"
  - "Carpet (slow down)"

# GM notes — Eric ignores this section entirely
notes: |
  Setup: place R2-D2 in the living room, Luke in the kitchen.
  Character scripts: R2 beeps and says "boo-weep-boop" until asked about Luke.
```

---

## Action Types

| Action | What happens |
|---|---|
| `find_and_approach` | Navigate to target, mark done on arrival |
| `deliver_message` | Speak `message` to target, wait `wait_sec` seconds, advance |
| `speak_to` | Greet + initiate conversation, wait `wait_sec` seconds for operator to type reply |
| `wait_for_response` | Stop and wait `wait_sec` seconds — operator types character reply in GUI |
| `photograph` | Capture `photo_count` sharp blur-checked photos to `missions/photos/` |

---

## Alarm Types

| Alarm | LED | Audio | TTS Prefix | Follow-on |
|---|---|---|---|---|
| `siren` | Rapid red strobe | Rising oscillating tone | "EMERGENCY! EMERGENCY!" | Stay with target, repeat every 15s (if `stay_with_target: true`) |
| `hazard` | Slow amber pulse | Triple warning beep | "WARNING! HAZARD DETECTED!" | Log severity (CRITICAL/WARNING/ADVISORY), continue patrol |
| `suspicious` | Medium red strobe | Urgent staccato beeps | "ALERT! SUSPICIOUS OBJECT!" | Back away 3m, turn 180°, hold (if `back_away_on_find: true`) |
| `nature` | Gentle green pulse | None | *(no prefix — just narration)* | Photograph, narrate, continue |
| `none` | None | None | None | Find-and-approach → confirm description → eye contact → greet → photograph |

> All alarm tones are generated mathematically at runtime (raw PCM via `struct.pack`) — no audio files, no internet required.

---

## Narrative Missions (`alarm_type: none`)

With `alarm_type: none`, Eric runs the full target confirmation pipeline on arrival:

1. Cosmos checks if the person matches the description in the `characters` hints
2. If description doesn't match: Eric asks the stranger for directions, then resumes search
3. If it matches: tilt sweep to find face → eye contact gate → greet → dual-cam photos
4. If `wait_for_dismiss: true`: Eric stays in place after greeting until operator presses STOP

The `characters` list provides Cosmos with description hints for identity checking. The `owner` keyword in a character name triggers the description-match logic (e.g. `name: "Creator / Owner"`).

---

## Mission Library

| File | Name | Alarm | Scan | Description |
|---|---|---|---|---|
| `template.yaml` | Template | — | — | Fully commented starting point |
| `find_leia.yaml` | Operation Find Leia | none | target_hunt | 3-step: find R2 → brief Luke → locate Leia |
| `jedi_training.yaml` | Operation Chosen One | none | target_hunt | Eric IS Anakin — faces Palpatine's dark side offer |
| `protect_john_connor.yaml` | Protect John Connor | 🔴 suspicious | target_hunt | You are the T-800 — locate John, identify T-1000 |
| `fetch_slippers.yaml` | Fetch My Slippers | none | target_hunt | 360° sweep to find slippers on the floor |
| `find_yellow_pen.yaml` | Find the Yellow Pen | none | target_hunt | Colour-contrast search — yellow on green |
| `office_mystery.yaml` | Operation Missing Drive | none | target_hunt | Talk to staff, follow leads, find red USB drive |
| `search_and_rescue.yaml` | Search and Rescue | 🚨 siren | target_hunt | Find casualty, sound siren, stay and broadcast |
| `disaster_life_search.yaml` | Disaster Life Search | 🚨 siren | target_hunt | Simulated disaster sweep — survivor search |
| `hazard_patrol.yaml` | Hazard Patrol | ⚠️ hazard | target_hunt | Full-area safety inspection |
| `room_safety_check.yaml` | Room Safety Check | ⚠️ hazard | target_hunt | Single-room audit — PASS / CONDITIONAL / FAIL |
| `nature_explorer.yaml` | Nature Explorer | 🌿 nature | video_sweep | Wildlife + plants — poetic narration, photo each find |
| `security_sweep.yaml` | Security Sweep | 🔴 suspicious | target_hunt | Suspicious objects — automatic back-away protocol |
| `terrain_assessment.yaml` | Terrain Assessment | ⚠️ hazard | target_hunt | Map traversability, flag hazards, recommend route |

---

## Writing Your Own Mission

1. Copy `missions/template.yaml`
2. Write your `briefing` in plain English — Cosmos reads it as-is
3. Set `alarm_type` to match what Eric should do on a find
4. Set `scan_strategy: video_sweep` if this is an observation/patrol mission; leave default for find missions
5. Add `characters` with `hint` so you know what to type during the demo
6. Add `mission_stages` if you want Cosmos to have a focused sub-goal per step
7. Click ↺ refresh in the GUI — your mission appears in the dropdown immediately

### Tips

- **Multi-step missions:** write the briefing in sequence ("First find X. Then find Y. Finally..."). Cosmos will parse the steps automatically.
- **Find-and-greet (`alarm_type: none`):** add a character with `name: "Creator / Owner"` and a physical description in `hint`. Eric will check the description before greeting.
- **Observation missions:** set `scan_strategy: video_sweep` — Eric sweeps the whole room in one continuous rotation and sends the panoramic video to Cosmos.
- **Circumnavigation:** add `circumnavigate_on_empty: true` if your target might be hidden behind a box. Eric will peek around obstacles before doing a full 360.
