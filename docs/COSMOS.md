# How ERIC Uses NVIDIA Cosmos Reason 2

← [Back to README](../README.md)

---

Cosmos Reason 2 is not a module Eric calls occasionally. It **is** Eric's brain. Every decision — where to go, what to say, how to escape an obstacle, whether to greet someone — flows through Cosmos. There is no rule-based object classifier, no separate vision pipeline, no scripted dialogue system. The mission loop provides structure (move, scan, approach), but every decision within it — what Eric sees, whether the path is clear, whether that's the target, what to say — is a Cosmos reasoning output.

---

## The Model

ERIC runs `embedl/Cosmos-Reason2-2B-W4A16-Edge2` — a 4-bit weight, 16-bit activation quantized version of Cosmos Reason 2 2B — via vLLM on the Jetson Orin Nano Super 8GB. This quantization is what makes running a frontier vision-language model fully at the edge on an 8GB device possible.

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

The system prompt includes a strict JSON output ruleset that names every valid field, warns against aliases (`"reasoning"` → must be `"physical_reasoning"`), and prohibits markdown fences. `_finalize_result()` still catches Cosmos inventions and remaps them at parse time.

**KV Cache Warm-up:** At mission start, before the acknowledgement call, a `max_tokens=1` dummy request fires with the full system prompt. This pre-populates vLLM's KV cache so every subsequent call this mission pays only the incremental token cost (scene frame + prompt delta), not the full system prompt. Reduces TTFT from ~1.5s to ~300ms per call on the Orin Nano.

---

## The Mission Overlay

On top of the system prompt, `_get_mission_scan_overlay()` injects mission-type-specific instructions into every scan prompt:

- `siren` alarm → Cosmos is told to look for injured people or specific named targets, rate severity as CRITICAL
- `suspicious` alarm → Cosmos is given a precise description of suspicious objects and told not to approach
- `hazard` alarm → Cosmos classifies finds as CRITICAL / WARNING / ADVISORY
- `nature` alarm → Cosmos is told to narrate poetically and photograph each find
- `none` alarm (narrative missions) → Cosmos is told to look for a specific person by description, approach any person to confirm identity

The overlay also appends two context blocks:
- **`_get_character_context()`** — per-character hints from the YAML `characters` list so Eric knows how to interact with each person
- **`_get_stage_context()`** — the current stage goal from `mission_stages` in the YAML, updated as steps advance

---

## Role 1: Navigation Brain

While moving, Cosmos receives buffered pan-tilt camera frames every 6 seconds alongside a sensor context block — LiDAR arc distances, OAK-D depth, YOLO Layer 2 detections, current terrain, void warnings. The call is **async** — fired immediately after capturing frames, then collected when needed. While Cosmos processes frames, Eric continues moving, polling YOLO and LiDAR at 100ms intervals.

Cosmos outputs a structured JSON decision:

```json
{
  "action": "forward",
  "wall_ahead": false,
  "void_ahead": false,
  "person_visible": false,
  "physical_reasoning": "Path is clear ahead. Carpet visible — reducing speed."
}
```

The mission loop reads this JSON and acts on it. No hardcoded routes, no waypoints. Where Eric goes next is always a Cosmos decision.

---

## Role 2: Mission Step Parser

When a mission starts, Cosmos reads the raw briefing text and parses it into an ordered `MissionStep` array:

```
"First find R2-D2 and speak to him. Then find Luke and wait for his response."

→ [MissionStep(target="R2-D2", action="speak_to", wait_sec=20),
   MissionStep(target="Luke",   action="wait_for_response", wait_sec=20)]
```

Eric executes these steps in order, advancing only when each is confirmed complete. No structured mission file required — Cosmos reads English.

---

## Role 3: 360° Scan Analyst

Two scan modes depending on `scan_strategy` in mission YAML:

**`target_hunt` (default):** Pan-tilt sweeps 7 positions at TILT_SEARCH (+10° — face/torso height). One sharp frame per position, each submitted to Cosmos **async immediately after capture**. Pan moves to the next position while inference runs — inference and movement fully overlap. First candidate triggers webcam confirmation. Total: ~8–15s.

**`video_sweep` (observation):** Chassis rotates 360° continuously while recording. All frames sent as one panoramic Cosmos call. No early-exit — observation missions survey everything. Total: ~17s.

Both provide: best direction, target visibility and location, terrain per direction, void/hazard flags, physical reasoning.

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

If `close_and_facing` is false — the person is far away, has their back turned, or is not looking — Eric moves on silently. Eliminates greetings shouted across rooms. This gate runs in navigation checks, approach scans, and the `_confirm_and_photograph_target()` pipeline.

---

## Role 6: Target Confirmation & Photography

When Eric arrives at the target, a multi-step confirmation pipeline runs before greeting:

1. **Description confirm** (low tilt, full body) — Cosmos checks if person matches YAML description
2. **Face sweep** — tilt steps from −15° to +25°, Cosmos checks at each angle if face is clearly visible
3. **Eye contact gate** — up to 8 attempts × 3s waiting for direct gaze
4. **Centre check** — before saving each photo, Cosmos checks if the target is in the middle third of the frame; pan nudges by 12° if off-centre

This pipeline fires for both pan-tilt and webcam photos independently, producing two sharp, centred, correctly-framed images per find.

---

## Role 7: Character Conversation Handler

When a person or character responds in the GUI, Cosmos receives the full conversation history and the mission briefing. It decides:
- Did this person give useful mission information? Extract it.
- Has this conversation run its course? Exit politely and resume (tag `[MOVE_ON]` in response).
- Should Eric ask a follow-up question?

Introduction/identity questions are detected by keyword and answered with a fixed identity prompt (no creative drift, `temperature=0.1`) so Eric always introduces himself correctly.

---

## Role 8: Target Confirmation Gate

When Eric believes it has found its target, a final Cosmos check confirms: is this genuinely the target from the briefing, or a false positive from a shadow or partial view? Only after confirmation does `_trigger_mission_alarm()` fire.

A **flip-flop persistence guard** (`target_spotted_count`) prevents single-frame misses from aborting an approach: if Cosmos says `target_visible=false` but the count is still positive, the target lock is maintained and the miss doesn't count against the invisible limit.

---

## Role 9: Void Detection (Visual Layer)

Every scan prompt includes a `void_ahead` field. Cosmos is instructed to examine the lower third of every frame for stair edges, floor-texture endings, and open-air gaps. This is the visual void detection layer.

> **Note:** Hardware void detection (OAK-D floor-drop and LiDAR return sparsity) is disabled for the cookoff build due to false positives on low-texture floors. Cosmos visual void detection remains active.

---

## Role 10: Mission Completion Announcement

When all steps are done, Cosmos generates the final announcement — in character, in voice, in the context of the specific mission that just ran. A search and rescue completion sounds different from a nature explorer summary or a security sweep.

---

## Summary

| When | Cosmos receives | Cosmos outputs |
|---|---|---|
| Mission start | Raw briefing text | Ordered `MissionStep[]` array |
| KV warm-up | System prompt (pre-fill only) | 1 token (discarded) |
| Moving (every 6s) | 6 buffered frames + sensor context + mission overlay | `forward/stop` + `person_visible` + `void_ahead` + reasoning |
| Full 360° scan (`target_hunt`) | 1 frame per pan position (async, up to 14) + mission overlay | Per-position: target visible + direction |
| Full 360° scan (`video_sweep`) | Full panoramic video (~17 frames) + mission overlay | Direction + environment summary + reasoning |
| Obstacle hit | Camera + LiDAR arcs + OAK-D grid | Escape direction + `turn_sec` |
| Eye-contact check | Single close frame | `close_and_facing: true/false` |
| Target arrive — description | Full body frame + description | `confirmed: true/false` |
| Target arrive — face sweep | Frame at each tilt angle | `face_visible: true/false` |
| Target arrive — photo centre | Photo frame | `centred: true/false` + `offset: left/right/centre` |
| Character reply | Conversation history + briefing | Extract info / continue / exit with `[MOVE_ON]` |
| Target spotted | Scene frame + mission context | `target_visible: true/false` |
| Mission complete | All steps confirmed | Final in-character announcement |
