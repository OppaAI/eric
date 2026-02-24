# ERIC — Mission System

← [Back to README](../README.md)

---

Missions are YAML files in the `missions/` folder. Drop a new `.yaml` file in, click ↺ refresh in the GUI, and it appears instantly. No code required. Cosmos reads the briefing as plain English — no structured format needed.

---

## How It Works

When you press ENGAGE:

1. **Cosmos parses the briefing** into an ordered list of `MissionStep` objects — targets, sequencing, and action types extracted from natural language
2. **Eric executes steps sequentially** — advancing only when each one is confirmed complete
3. The **`alarm_type` field** controls what fires on a find — LED pattern, audio tone, TTS prefix, follow-on behaviour
4. At mission end, **`_mission_report()`** delivers a summary of all finds

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
photo_on_find:     false  # save timestamped photo to missions/photos/
announce_location: false  # TTS location announcement
stay_with_target:  false  # SAR: stay and repeat broadcast every 15s
back_away_on_find: false  # security: back 3m + turn 180°
generate_report:   false  # mission end: summary of all finds

# Characters (played by operator in GUI)
characters:
  - name: "R2-D2"
    hint: "Speaks in beeps. Knows where Luke is. Will help if asked nicely."

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
| `deliver_message` | Speak `message` to target, wait for acknowledgement |
| `speak_to` | Greet + initiate conversation, wait for operator to type reply |
| `wait_for_response` | Stop and wait — operator types character reply in GUI |
| `photograph` | Capture sharp frames, save to `missions/photos/` |

---

## Alarm Types

| Alarm | LED | Audio | TTS Prefix | Follow-on |
|---|---|---|---|---|
| `siren` | Rapid red strobe | Rising oscillating tone | "EMERGENCY! EMERGENCY!" | Stay with target, repeat every 15s |
| `hazard` | Slow amber pulse | Triple warning beep | "WARNING! HAZARD DETECTED!" | Log severity, continue patrol |
| `suspicious` | Medium red strobe | Urgent staccato beeps | "ALERT! SUSPICIOUS OBJECT!" | Back away 3m, turn 180°, hold |
| `nature` | Gentle green pulse | None | *(no prefix — just narration)* | Photograph, narrate, continue |
| `none` | None | None | None | Standard find-and-approach |

> All alarm tones are generated mathematically at runtime (raw PCM via `struct.pack`) — no audio files, no internet required.

---

## Mission Library

| File | Name | Alarm | Description |
|---|---|---|---|
| `template.yaml` | Template | — | Fully commented starting point |
| `find_leia.yaml` | Operation Find Leia | none | 3-step: find R2 → brief Luke → locate Leia |
| `jedi_training.yaml` | Operation Chosen One | none | Eric IS Anakin — faces Palpatine's dark side offer |
| `protect_john_connor.yaml` | Protect John Connor | 🔴 suspicious | You are the T-800 — locate John, identify T-1000 |
| `fetch_slippers.yaml` | Fetch My Slippers | none | 360° sweep to find slippers on the floor |
| `find_yellow_pen.yaml` | Find the Yellow Pen | none | Colour-contrast search — yellow on green |
| `office_mystery.yaml` | Operation Missing Drive | none | Talk to staff, follow leads, find red USB drive |
| `search_and_rescue.yaml` | Search and Rescue | 🚨 siren | Find casualty, sound siren, stay and broadcast |
| `disaster_life_search.yaml` | Disaster Life Search | 🚨 siren | Simulated disaster sweep — survivor search |
| `hazard_patrol.yaml` | Hazard Patrol | ⚠️ hazard | Full-area safety inspection |
| `room_safety_check.yaml` | Room Safety Check | ⚠️ hazard | Single-room audit — PASS / CONDITIONAL / FAIL |
| `nature_explorer.yaml` | Nature Explorer | 🌿 nature | Wildlife + plants — poetic narration, photo each find |
| `security_sweep.yaml` | Security Sweep | 🔴 suspicious | Suspicious objects — automatic back-away protocol |
| `terrain_assessment.yaml` | Terrain Assessment | ⚠️ hazard | Map traversability, flag hazards, recommend route |

---

## Writing Your Own Mission

1. Copy `missions/template.yaml`
2. Write your `briefing` in plain English — Cosmos reads it as-is
3. Set `alarm_type` to match what Eric should do on a find
4. Add `characters` with `hint` so you know what to type during the demo
5. Click ↺ refresh in the GUI — your mission appears in the dropdown immediately
