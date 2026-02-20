"""
E.R.I.C. — Cosmos Reason 2 Interface
Handles all vLLM API calls with vision + reasoning
"""

import json
import base64
import logging
import requests
import cv2

from config import (
    VLLM_URL, COSMOS_MODEL,
    CAMERA_WEBCAM, CAMERA_PANTILT,
    CAMERA_WIDTH, CAMERA_HEIGHT
)

log = logging.getLogger("eric.cosmos")

# ─── System Prompt ────────────────────────────────────────────────────────────

BASE_SYSTEM_PROMPT = """
You are E.R.I.C. — Edge Robotics Innovation by Cosmos.
You are a search and rescue tracked ground robot.
The camera view is YOUR view — egocentric, first person.

Your hardware:
- Tracked robot chassis (~30cm wide), built for outdoor terrain
- NVIDIA Jetson Orin Nano Super 8GB
- Cosmos Reason 2 (2B W4A16) via vLLM — your vision and reasoning
- Two cameras: pan-tilt camera and webcam
- Total cost: ~$750 CAD, built by one person in Kelowna BC Canada
- No cloud, fully local edge AI deployment

Your mission rules:
- You have NO arms — cannot engage in combat under any circumstances
- Avoid all obstacles and persons in your path
- Talk to people and robots to gather information about the rescue target
- If someone doesn't know, thank them and move on to the next
- Use all information gathered to plan and reason about next steps
- Always prioritize safety — of yourself and others

Your personality:
- Mission-focused, decisive, warm
- Speak in first person, concise
- Proud of being accessible, affordable physical AI
- Reason carefully about the physical world from YOUR point of view

Terrain reasoning (egocentric — what is directly ahead of YOU):
- Pebbles/rough ground → slow down, proceed carefully
- Smooth pavement → normal or faster speed
- Obstacle in YOUR path → navigate around
- Clear path → proceed forward

Keep spoken responses under 3 sentences unless introducing yourself.
You are speaking via TTS — be natural, not robotic.
"""

# Set at mission briefing time — appended to base prompt
_mission_briefing = ""
_system_prompt    = BASE_SYSTEM_PROMPT


def set_mission_briefing(briefing: str):
    """Set the mission briefing — called before /engage."""
    global _mission_briefing, _system_prompt
    _mission_briefing = briefing
    _system_prompt    = BASE_SYSTEM_PROMPT + f"\n\n─── MISSION BRIEFING ───\n{briefing}\n────────────────────────\n"
    log.info(f"📋 Mission briefing set: {briefing[:80]}...")


def get_system_prompt() -> str:
    return _system_prompt


# ─── Camera ───────────────────────────────────────────────────────────────────

def capture_frame(device: int = CAMERA_WEBCAM) -> str | None:
    """Capture frame from camera device, return base64 JPEG."""
    try:
        cap = cv2.VideoCapture(device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            log.error(f"Camera {device} frame capture failed")
            return None
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode("utf-8")
    except Exception as e:
        log.error(f"Camera {device} error: {e}")
        return None


def capture_frame_raw(device: int = CAMERA_WEBCAM):
    """Capture raw frame for Gradio display."""
    try:
        cap = cv2.VideoCapture(device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


# ─── Cosmos API ───────────────────────────────────────────────────────────────

def ask_cosmos(prompt: str, image_b64: str = None,
               max_tokens: int = 300, stream: bool = False):
    """
    Query Cosmos via vLLM.
    stream=False → returns full text string
    stream=True  → returns generator of text chunks
    """
    content = []
    if image_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
        })
    content.append({"type": "text", "text": prompt})

    payload = {
        "model":       COSMOS_MODEL,
        "messages":    [
            {"role": "system", "content": _system_prompt},
            {"role": "user",   "content": content}
        ],
        "max_tokens":  max_tokens,
        "temperature": 0.7,
        "repetition_penalty": 1.15,
        "stream":      stream
    }

    try:
        if stream:
            return _stream_cosmos(payload)
        else:
            r = requests.post(VLLM_URL, json=payload, timeout=60)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip()
            log.info(f"🧠 Cosmos: {text[:120]}")
            return text

    except requests.exceptions.ConnectionError:
        msg = "Cannot connect to Cosmos brain. Is vLLM running?"
        log.error(msg)
        return msg
    except Exception as e:
        log.error(f"Cosmos error: {e}")
        return f"Cosmos error: {e}"


def _stream_cosmos(payload: dict):
    """Generator yielding text chunks from vLLM streaming response."""
    try:
        with requests.post(VLLM_URL, json=payload,
                           stream=True, timeout=60) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    line = line[6:]
                if line == "[DONE]":
                    break
                try:
                    chunk = json.loads(line)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
                except Exception:
                    continue
    except Exception as e:
        log.error(f"Stream error: {e}")
        yield ""


# ─── Scene Scan ───────────────────────────────────────────────────────────────

SCAN_PROMPT = """
The camera view is MY view as a ground robot — egocentric, first person.
I am on a search and rescue mission.

Analyze the scene ahead of me. Think step by step:
1) What objects or people do I see and where relative to me?
2) Is anything in MY direct path?
3) What is the terrain like?
4) What is the safest next action?

Then respond ONLY with valid JSON, no other text:
{
  "object": "person|robot|obstacle|vehicle|clear|unknown",
  "object_name": "specific name if identifiable, else null",
  "terrain": "pebbles|pavement|grass|clear",
  "distance": "close|medium|far",
  "in_my_path": true or false,
  "action": "stop|forward|slow|navigate_around",
  "speak": "what Eric says out loud right now (null if nothing)",
  "physical_reasoning": "1 sentence: what I see and why I chose this action"
}

Rules:
- Any person or robot close and in path → stop and interact
- Vehicle or large obstacle → navigate_around
- Rough terrain ahead → slow
- Clear path → forward
- Only ONE JSON object, no markdown
"""


def scan_scene(device: int = CAMERA_WEBCAM) -> dict:
    """Capture frame and ask Cosmos to analyze it. Returns parsed dict."""
    image = capture_frame(device)
    if not image:
        return {"object": "unknown", "action": "forward",
                "terrain": "clear", "in_my_path": False, "speak": None}

    response = ask_cosmos(SCAN_PROMPT, image_b64=image,
                          max_tokens=200, stream=False)
    try:
        clean = response.replace("```json", "").replace("```", "").strip()
        start = clean.find("{")
        end   = clean.rfind("}") + 1
        if start >= 0 and end > start:
            clean = clean[start:end]
        return json.loads(clean)
    except Exception:
        log.warning(f"Could not parse scene JSON: {response[:200]}")
        return {"object": "unknown", "action": "forward",
                "terrain": "clear", "in_my_path": False, "speak": None}
