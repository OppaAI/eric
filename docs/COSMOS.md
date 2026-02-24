# How ERIC Uses NVIDIA Cosmos Reason 2

← [Back to README](../README.md)

---

Cosmos Reason 2 is not a module Eric calls occasionally. It **is** Eric's brain. Every decision — where to go, what to say, how to escape an obstacle, whether to greet someone — flows through Cosmos. There is no separate navigation algorithm, no rule-based object classifier, no hardcoded route planner. Cosmos sees, reasons, and decides.

---

## The Model

ERIC runs `embedl/Cosmos-Reason2-2B-W4A16` — a 4-bit weight, 16-bit activation quantized version of Cosmos Reason 2 2B — via vLLM on the Jetson Orin Nano Super 8GB. This quantization is what makes running a frontier vision-language model fully at the edge on an 8GB device possible.

| Metric | Value |
|---|---|
| Text tokens/sec | ~40–50 |
| Vision tokens/sec | ~16–17 (640×480) |
| Vision call latency | ~5–9 seconds |
| GPU utilisation | ~75% |
| VRAM used | ~6.8 GB / 7.4 GB available |
| Cloud required | None |

---

## The System Prompt

Every single Cosmos call — nav check, scan, character conversation, obstacle escape — carries the mission briefing as a persistent system prompt:

```python
system_prompt = BASE_IDENTITY + "\n\nMission briefing:\n" + mission_briefing
```

When you select `find_leia.yaml` and press ENGAGE, the briefing becomes Cosmos's identity for the entire mission. Cosmos has no memory between calls — the system prompt provides continuity. Every decision is made with full mission context. Eric never forgets what it is doing.

---

## The Mission Overlay

On top of the system prompt, `_get_mission_scan_overlay()` injects mission-type-specific instructions into every scan prompt:

- `siren` alarm → Cosmos is told to look for injured people, rate severity as CRITICAL
- `suspicious` alarm → Cosmos is given a precise description of suspicious objects and told not to approach
- `nature` alarm → Cosmos is told to narrate poetically and photograph each find

This changes what Cosmos pays attention to per mission with zero code changes.

---

## Role 1: Navigation Brain

While moving, Cosmos receives a pan-tilt camera frame every 4 seconds alongside a sensor context block — LiDAR arc distances, OAK-D depth grid, current terrain, void warnings. It outputs a structured JSON decision:

```json
{
  "action": "forward",
  "wall_ahead": false,
  "void_ahead": false,
  "object": "person",
  "target_visible": false,
  "distance": "far",
  "terrain": "carpet",
  "physical_reasoning": "Path is clear ahead. Carpet visible — reducing speed."
}
```

The mission loop reads this JSON and acts on it. No hardcoded routes, no waypoints. Where Eric goes next is always a Cosmos decision. The `terrain` field drives automatic speed adjustment — 57 terrain keywords mapped to 4 speed tiers.

---

## Role 2: Mission Step Parser

When a mission starts, Cosmos reads the raw briefing text and parses it into an ordered `MissionStep` array:

```
"First find R2-D2 and speak to him. Then find Luke and wait for his response."

→ [MissionStep(target="R2-D2", action="speak_to"),
   MissionStep(target="Luke",   action="wait_for_response")]
```

Eric executes these steps in order, advancing only when each is confirmed complete. No structured mission file required — Cosmos reads English.

---

## Role 3: 360° Scan Analyst

When Eric stops for a full scan, the pan-tilt head sweeps 7 pan positions at 3 tilt angles (30° down, 10° down, −10° horizon) — up to 42 frames batched into a single Cosmos multi-image call. Cosmos analyses the full panorama and outputs:

- Best direction to move toward the mission target
- Whether the target is visible and where
- Terrain type per direction and recommended speed
- Void or drop hazards per direction
- Physical reasoning explaining the decision

---

## Role 4: Escape Director

When the 3-layer avoidance pipeline fires, Cosmos is Layer 3. It receives a camera frame plus LiDAR arc distances (front / left / right / rear) and the OAK-D 3×3 depth grid, and outputs a specific escape direction with an exact turn duration:

```json
{
  "action": "turn_left",
  "turn_sec": 1.8,
  "physical_reasoning": "Left arc has 0.92m clearance vs 0.18m front and 0.41m right"
}
```

Layers 1 and 2 (instant backup + sensor arc scan) provide immediate safety. Cosmos provides the intelligent, context-aware escape route. If Cosmos times out (20s limit), the arc-based direction runs instead — Eric is never stuck waiting.

---

## Role 5: Eye-Contact Gate

Before Eric greets any person, a rapid single-frame Cosmos check asks one question:

```json
{
  "close_and_facing": true,
  "reasoning": "Person is approximately 1m away, face visible and oriented toward camera"
}
```

If `close_and_facing` is false — the person is far away, has their back turned, or is not looking — Eric moves on silently. Eliminates greetings shouted across rooms.

---

## Role 6: Character Conversation Handler

When a person or character responds in the GUI, Cosmos receives the full conversation history and the mission briefing. It decides:
- Did this person give useful mission information? Extract it.
- Has this conversation run its course? Exit politely and resume.
- Should Eric ask a follow-up question?

---

## Role 7: Target Confirmation

When Eric believes it has found its target, a final Cosmos check confirms: is this genuinely the target from the briefing, or a false positive from a shadow or partial view? Only after confirmation does `_trigger_mission_alarm()` fire.

---

## Role 8: Void Detection (Visual Layer)

Every scan prompt includes a `void_ahead` field. Cosmos is instructed to examine the lower third of every frame for stair edges, floor-texture endings, and open-air gaps. This is the third void detection layer on top of OAK-D floor-drop (hardware) and LiDAR return sparsity (hardware).

---

## Role 9: Mission Completion Announcement

When all steps are done, Cosmos generates the final announcement — in character, in voice, in the context of the specific mission that just ran. A search and rescue completion sounds different from a nature explorer summary or a security sweep.

---

## Summary

| When | Cosmos receives | Cosmos outputs |
|---|---|---|
| Mission start | Raw briefing text | Ordered `MissionStep[]` array |
| Moving (every 4s) | Camera frame + sensor context + mission overlay | `forward/stop` + terrain + `void_ahead` + reasoning |
| Full 360° scan | Up to 42 frames + sensor context + mission overlay | Direction + target info + void flags |
| Obstacle hit | Camera + LiDAR arcs + OAK-D grid | Escape direction + `turn_sec` |
| Eye-contact check | Single close frame | `close_and_facing: true/false` |
| Character reply | Conversation history + briefing | Extract info / continue / exit |
| Target spotted | Scene frame + mission context | `target_visible: true/false` |
| Mission complete | All steps confirmed | Final in-character announcement |
