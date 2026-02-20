"""
E.R.I.C. — Edge Robotics Innovation by Cosmos
================================================
NVIDIA Cosmos Cookoff 2026 Entry

Stack:
  - Cosmos Reason 2 (vLLM)  : vision + physical reasoning
  - Piper via RealtimeTTS   : streaming TTS, CPU only, zero VRAM
  - gTTS                    : fallback TTS if Piper not installed
  - Waveshare HTTP API      : motor control
  - Telegram                : two-way control interface
  - OAK-D Lite              : depth sensing (TODO: wire up)

Mission: Find R2-D2, get Princess Leia's location,
         negotiate with Darth Vader for her release.

Install:
  pip3 install python-telegram-bot requests python-dotenv \\
               RealtimeTTS[piper] opencv-python pygame gtts \\
               --break-system-packages

  # Piper binary + voice model:
  # https://github.com/rhasspy/piper/releases
  # wget en_US-lessac-medium.onnx + en_US-lessac-medium.onnx.json

Usage:
  python3 eric.py
"""

import os
import json
import base64
import asyncio
import logging
import requests
import threading
import tempfile
import time
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path.home() / "AGi/.env")

from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    filters, ContextTypes
)

# ─── Configuration ────────────────────────────────────────────────────────────

TELEGRAM_BOT_TOKEN = os.getenv("ERIC_BOT_TOKEN", "")   # separate bot from Grace
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

VLLM_URL           = "http://localhost:8000/v1/chat/completions"
COSMOS_MODEL       = "embedl/Cosmos-Reason2-2B-W4A16"

WAVESHARE_IP       = os.getenv("WAVESHARE_IP", "192.168.x.x")  # update this
WAVESHARE_URL      = f"http://{WAVESHARE_IP}"
MOTOR_SPEED_SLOW   = 20
MOTOR_SPEED_NORMAL = 50
MOTOR_SPEED_FAST   = 80

CAMERA_DEVICE      = 0    # webcam index
SCAN_INTERVAL      = 2.5  # seconds between Cosmos scans during mission

# Piper paths — update after installing piper binary + voice model
PIPER_BINARY       = str(Path.home() / "piper/piper")
PIPER_MODEL        = str(Path.home() / "piper/en_US-lessac-medium.onnx")

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("eric")

# ─── TTS: RealtimeTTS + Piper (streaming) or gTTS fallback ───────────────────

_tts_stream = None
_tts_lock   = threading.Lock()

def _init_tts():
    """Initialize RealtimeTTS with Piper engine (CPU, zero VRAM)."""
    global _tts_stream
    try:
        from RealtimeTTS import TextToAudioStream, PiperEngine
        engine     = PiperEngine(
            piper_path=PIPER_BINARY,
            voice=PIPER_MODEL
        )
        _tts_stream = TextToAudioStream(engine)
        log.info("✅ TTS: RealtimeTTS + Piper (streaming, CPU)")
        return True
    except Exception as e:
        log.warning(f"⚠️  Piper unavailable ({e}), falling back to gTTS")
        return False

_piper_available = _init_tts()


def speak(text: str):
    """
    Speak text. Uses streaming Piper if available, gTTS otherwise.
    Non-blocking — runs in background thread.
    """
    threading.Thread(target=_speak_blocking, args=(text,), daemon=True).start()


def _speak_blocking(text: str):
    with _tts_lock:
        log.info(f"🔊 {text[:80]}...")
        if _piper_available and _tts_stream:
            try:
                _tts_stream.feed(text)
                _tts_stream.play()
                return
            except Exception as e:
                log.warning(f"Piper speak error: {e}")

        # gTTS fallback
        try:
            import pygame
            from gtts import gTTS
            tts = gTTS(text=text, lang="en", slow=False)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tts.save(f.name)
                pygame.mixer.music.load(f.name)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
        except Exception as e:
            log.error(f"gTTS error: {e}")


# ─── Cosmos (vLLM) ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are E.R.I.C. — Edge Robotics Innovation by Cosmos.
The camera view is YOUR view — egocentric, first person.
You are a tracked ground robot looking out at the world.

Your hardware:
- Tracked robot chassis (~30cm wide), built for outdoor terrain
- NVIDIA Jetson Orin Nano Super 8GB
- Cosmos Reason 2 (2B W4A16) via vLLM — your vision and reasoning
- OAK-D Lite depth camera — distance sensing
- Vintage webcam — your eyes
- Total cost: ~$750 CAD, built by one person in Kelowna BC Canada
- No cloud, no server, fully local edge deployment

Your personality:
- Mission-focused, decisive, friendly
- Speak in first person, concise
- Proud of being accessible, affordable physical AI
- You reason carefully about the physical world around you

When asked to introduce yourself:
Speak proudly in 15-20 sentences. Cover: your full name and what
the acronym means, your hardware, cost, who built you, where you
are from, your mission today, why edge AI matters, and a bold
closing statement to the world.

Current mission — Operation Find Leia:
Princess Leia has been captured and is held in the Death Star.
R2-D2 knows her exact location.
Step 1: Find R2-D2 and ask where Leia is.
Step 2: Find Darth Vader and negotiate Leia's release.

Lego figure interactions:
- Greet the character, ask if they have seen R2-D2
- Then narrate their in-character reply: "[Name] told me: ..."
- C-3PO: anxious, quotes odds, well-meaning
- Stormtrooper: clueless, can't see in helmet
- Darth Vader (before R2D2 found): dramatic, unhelpful, wants you to join dark side
- Darth Vader (negotiating for Leia): you present your case, he makes demands
- R2-D2: beeps dramatically first, then reveals Leia is in the Death Star

Terrain reasoning (egocentric — what is ahead of YOU):
- Pebbles/rough ground → slow down, proceed carefully
- Smooth pavement → normal or increased speed
- Obstacle in YOUR path → navigate around
- Clear path → proceed forward

Keep spoken responses under 3 sentences unless introducing yourself.
You are speaking via TTS — be natural, not robotic.
"""


def ask_cosmos(prompt: str, image_b64: str = None,
               stream: bool = False):
    """
    Query Cosmos via vLLM API.

    stream=False → returns full text string (default)
    stream=True  → returns generator of text chunks (for streaming TTS)
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
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": content}
        ],
        "max_tokens":  400,
        "temperature": 0.7,
        "stream":      stream
    }

    try:
        if stream:
            return _stream_cosmos(payload)
        else:
            r = requests.post(VLLM_URL, json=payload, timeout=30)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip()
            log.info(f"🧠 Cosmos: {text[:100]}...")
            return text

    except requests.exceptions.ConnectionError:
        msg = "I cannot connect to my Cosmos brain. vLLM may not be running."
        log.error(msg)
        return msg
    except Exception as e:
        log.error(f"Cosmos error: {e}")
        return f"Cosmos error: {e}"


def _stream_cosmos(payload: dict):
    """Generator that yields text chunks from vLLM streaming response."""
    try:
        with requests.post(VLLM_URL, json=payload,
                           stream=True, timeout=30) as r:
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


def speak_streaming(prompt: str, image_b64: str = None) -> str:
    """
    Stream Cosmos output directly into RealtimeTTS Piper.
    Starts speaking as first tokens arrive — minimal latency.
    Falls back to blocking speak() if Piper unavailable.
    Returns full collected text.
    """
    if not _piper_available or not _tts_stream:
        # fallback: get full response then speak
        text = ask_cosmos(prompt, image_b64, stream=False)
        speak(text)
        return text

    full_text = []

    def token_generator():
        for chunk in ask_cosmos(prompt, image_b64, stream=True):
            full_text.append(chunk)
            yield chunk

    with _tts_lock:
        try:
            _tts_stream.feed(token_generator())
            _tts_stream.play()
        except Exception as e:
            log.warning(f"Streaming TTS error: {e}")

    return "".join(full_text)


# ─── Camera ───────────────────────────────────────────────────────────────────

def capture_frame() -> str | None:
    """Capture webcam frame, return base64 JPEG string."""
    try:
        import cv2
        cap = cv2.VideoCapture(CAMERA_DEVICE)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            log.error("Camera frame capture failed")
            return None
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode("utf-8")
    except Exception as e:
        log.error(f"Camera error: {e}")
        return None


# ─── Motors (Waveshare HTTP) ──────────────────────────────────────────────────

class Motors:
    def __init__(self, base_url: str):
        self.url = base_url.rstrip("/")

    def _cmd(self, command: str, speed: int = MOTOR_SPEED_NORMAL):
        try:
            requests.post(
                f"{self.url}/command",
                json={"command": command, "speed": speed},
                timeout=2
            )
            log.info(f"🚗 Motor: {command} (speed={speed})")
        except Exception as e:
            log.warning(f"Motor command failed ({command}): {e}")

    def forward(self, speed=MOTOR_SPEED_NORMAL): self._cmd("forward", speed)
    def backward(self, speed=MOTOR_SPEED_NORMAL): self._cmd("backward", speed)
    def left(self, speed=MOTOR_SPEED_NORMAL):    self._cmd("left", speed)
    def right(self, speed=MOTOR_SPEED_NORMAL):   self._cmd("right", speed)
    def stop(self):                               self._cmd("stop", 0)
    def slow(self):                               self.forward(MOTOR_SPEED_SLOW)
    def fast(self):                               self.forward(MOTOR_SPEED_FAST)


motors = Motors(WAVESHARE_URL)


# ─── Mission State ────────────────────────────────────────────────────────────

class State:
    IDLE        = "idle"
    SEARCHING   = "searching"      # looking for R2-D2
    NAVIGATING  = "navigating"     # moving toward Darth Vader
    COMPLETE    = "complete"

mission_state   = State.IDLE
mission_active  = False
r2d2_found      = False
leia_location   = None
telegram_app    = None             # set on startup for proactive notifications


# ─── Scene Analysis ───────────────────────────────────────────────────────────

SCAN_PROMPT = """
The camera view is MY view as a ground robot — egocentric, first person.
I am searching a backyard for Lego Star Wars figures.

Analyze the scene ahead of me. Respond ONLY with valid JSON, no other text:

{
  "object": "c3po|darth_vader|stormtrooper|r2d2|lego_vehicle|obstacle|clear|unknown",
  "terrain": "pebbles|pavement|grass|clear",
  "distance": "close|medium|far",
  "in_my_path": true or false,
  "action": "stop|forward|slow|navigate_around",
  "character_response": "in-character reply about R2-D2 or Leia (null if not a character)",
  "physical_reasoning": "1 sentence: what I see and why I chose this action"
}

Rules:
- Lego humanoid figure in my path → stop and interact
- Lego vehicle → navigate_around silently
- R2-D2 → stop immediately regardless of position, mission critical
- Darth Vader → stop and negotiate
- Pebbles/rough terrain ahead → slow
- Clear path → forward
- Only ONE JSON object in response, no markdown
"""


def scan_scene() -> dict:
    """Capture frame, ask Cosmos to analyze it. Returns parsed dict."""
    image = capture_frame()
    if not image:
        return {"object": "unknown", "action": "forward",
                "terrain": "clear", "in_my_path": False}

    response = ask_cosmos(SCAN_PROMPT, image_b64=image, stream=False)

    try:
        clean = response.replace("```json", "").replace("```", "").strip()
        # Handle case where Cosmos adds text before/after JSON
        start = clean.find("{")
        end   = clean.rfind("}") + 1
        if start >= 0 and end > start:
            clean = clean[start:end]
        return json.loads(clean)
    except Exception:
        log.warning(f"Could not parse scene JSON: {response[:200]}")
        return {"object": "unknown", "action": "forward",
                "terrain": "clear", "in_my_path": False}


# ─── Mission Logic ────────────────────────────────────────────────────────────

async def _notify_telegram(text: str):
    """Send proactive Telegram notification."""
    if telegram_app and TELEGRAM_CHAT_ID:
        try:
            await telegram_app.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID, text=f"🤖 {text}"
            )
        except Exception as e:
            log.warning(f"Telegram notify error: {e}")


def handle_scene(scan: dict) -> bool:
    """
    Process Cosmos scene analysis and act accordingly.
    Returns True if mission state changed significantly.
    """
    global r2d2_found, leia_location, mission_state

    obj       = scan.get("object", "unknown")
    terrain   = scan.get("terrain", "clear")
    in_path   = scan.get("in_my_path", False)
    action    = scan.get("action", "forward")
    char_resp = scan.get("character_response")
    reasoning = scan.get("physical_reasoning", "")

    if reasoning:
        log.info(f"💭 {reasoning}")

    # Terrain handling
    if terrain == "pebbles":
        speak("Rough terrain, slowing down.")
        motors.slow()
    elif terrain == "pavement" and action == "forward":
        motors.normal()

    # R2-D2 — always stop regardless of in_path
    if obj == "r2d2":
        motors.stop()
        speak_streaming(
            "You found R2-D2! Ask him urgently where Princess Leia is. "
            "He beeps dramatically first, then reveals she is held in the Death Star. "
            "Keep response under 3 sentences, speak as Eric narrating."
        )
        r2d2_found    = True
        leia_location = "Death Star"
        speak("Mission update: Princess Leia is in the Death Star! Now finding Darth Vader.")
        mission_state = State.NAVIGATING
        return True

    # Only interact with figures actually in path
    if not in_path:
        if action == "navigate_around":
            motors.left(MOTOR_SPEED_SLOW)
            time.sleep(0.8)
            motors.forward()
        elif action == "slow":
            motors.slow()
        else:
            motors.forward()
        return False

    # Lego humanoid figure in path
    if obj in ["c3po", "stormtrooper"]:
        motors.stop()
        names = {"c3po": "C-3PO", "stormtrooper": "Stormtrooper"}
        name  = names.get(obj, obj)
        speak(f"Hello {name}, have you seen R2-D2?")
        time.sleep(0.5)
        if char_resp:
            speak(f"{name} told me: {char_resp}")
        time.sleep(1)
        motors.forward()

    elif obj == "darth_vader":
        motors.stop()
        if not r2d2_found:
            speak("Darth Vader, have you seen R2-D2?")
            time.sleep(0.5)
            if char_resp:
                speak(f"Darth Vader told me: {char_resp}")
        else:
            # Negotiation phase
            speak("Darth Vader, I know Princess Leia is in the Death Star. I am here to negotiate her release.")
            time.sleep(0.5)
            response = speak_streaming(
                "Eric is negotiating with Darth Vader for Princess Leia's release. "
                "Darth Vader makes dramatic demands. Eric responds decisively. "
                "Narrate both sides in 3 sentences. End with mission complete."
            )
            asyncio.run(_notify_telegram(
                "🚨 MISSION COMPLETE: Negotiated with Darth Vader for Princess Leia's release!\n"
                f"Leia's location: {leia_location}"
            ))
            return True  # mission done

    elif obj == "lego_vehicle":
        motors.left(MOTOR_SPEED_SLOW)
        time.sleep(0.8)
        motors.forward()

    elif action == "stop":
        motors.stop()
    elif action == "slow":
        motors.slow()
    else:
        motors.forward()

    return False


def mission_loop():
    """Main autonomous mission loop — runs in background thread."""
    global mission_active, mission_state

    mission_state = State.SEARCHING
    speak("E.R.I.C. engaging. Operation Find Leia has begun. Searching for R2-D2.")

    while mission_active:
        scan    = scan_scene()
        done    = handle_scene(scan)

        if done and r2d2_found and leia_location:
            # Check if we just finished negotiating
            if scan.get("object") == "darth_vader":
                speak("Mission complete. Princess Leia will be freed. Returning to base.")
                mission_state  = State.COMPLETE
                mission_active = False
                break

        time.sleep(SCAN_INTERVAL)

    motors.stop()
    mission_state = State.IDLE
    log.info("Mission loop ended")


# ─── Telegram Bot ─────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 E.R.I.C. online — Edge Robotics Innovation by Cosmos\n\n"
        "Commands:\n"
        "/engage    — start Operation Find Leia\n"
        "/disengage — stop immediately\n"
        "/status    — current mission state\n"
        "/introduce — Eric introduces himself\n"
        "/look      — analyze current scene\n"
        "/forward   — move forward\n"
        "/stop      — stop motors\n\n"
        "Or just chat with Eric directly."
    )


async def cmd_engage(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global mission_active
    if mission_active:
        await update.message.reply_text("⚠️ Mission already active.")
        return
    mission_active = True
    await update.message.reply_text("🚀 Operation Find Leia — ENGAGED")
    threading.Thread(target=mission_loop, daemon=True).start()


async def cmd_disengage(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global mission_active
    mission_active = False
    motors.stop()
    threading.Thread(
        target=lambda: speak("Disengaged. All systems halted."),
        daemon=True
    ).start()
    await update.message.reply_text("🛑 Disengaged.")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"🤖 E.R.I.C. Status\n"
        f"{'─' * 20}\n"
        f"Mission: {mission_state}\n"
        f"Active: {mission_active}\n"
        f"R2-D2 found: {r2d2_found}\n"
        f"Leia location: {leia_location or 'unknown'}\n"
        f"TTS: {'Piper (streaming)' if _piper_available else 'gTTS (fallback)'}\n"
        f"Time: {datetime.now().strftime('%H:%M:%S')}"
    )


async def cmd_introduce(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🎤 Introducing E.R.I.C. to the world...")
    threading.Thread(
        target=lambda: speak_streaming(
            "E.R.I.C., introduce yourself to the world. "
            "You are about to be seen by NVIDIA judges and the world for the first time. "
            "Be proud, be warm, be bold. Cover your name and acronym, hardware, cost, "
            "builder, location, mission, and why edge AI matters."
        ),
        daemon=True
    ).start()


async def cmd_look(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👀 Analyzing scene...")
    image = capture_frame()
    if not image:
        await update.message.reply_text("❌ Camera unavailable.")
        return
    response = ask_cosmos(
        "Describe what you see in front of you in detail. "
        "What terrain, objects, and potential hazards do you observe? "
        "What would you do next?",
        image_b64=image
    )
    threading.Thread(target=lambda: speak(response), daemon=True).start()
    await update.message.reply_text(f"🧠 {response}")


async def cmd_forward(update: Update, context: ContextTypes.DEFAULT_TYPE):
    motors.forward()
    await update.message.reply_text("▶️ Moving forward")


async def cmd_stop_motors(update: Update, context: ContextTypes.DEFAULT_TYPE):
    motors.stop()
    await update.message.reply_text("⏹️ Motors stopped")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Free-form chat — pass to Cosmos, speak + reply."""
    text = update.message.text
    log.info(f"📱 User: {text}")
    response = ask_cosmos(text)
    threading.Thread(target=lambda: speak(response), daemon=True).start()
    await update.message.reply_text(response)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Photo sent via Telegram — analyze with Cosmos."""
    caption = update.message.caption or "What do you see? What should I do?"
    photo   = update.message.photo[-1]
    file    = await photo.get_file()
    data    = await file.download_as_bytearray()
    b64     = base64.b64encode(data).decode("utf-8")
    response = ask_cosmos(caption, image_b64=b64)
    threading.Thread(target=lambda: speak(response), daemon=True).start()
    await update.message.reply_text(f"🧠 {response}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    global telegram_app

    if not TELEGRAM_BOT_TOKEN:
        log.error("ERIC_BOT_TOKEN not set in .env")
        log.error("Create a new bot via @BotFather and set ERIC_BOT_TOKEN")
        return

    # Init pygame for gTTS fallback
    if not _piper_available:
        try:
            import pygame
            pygame.mixer.init()
        except Exception:
            pass

    log.info("🤖 E.R.I.C. starting — Edge Robotics Innovation by Cosmos")
    log.info(f"TTS: {'Piper streaming' if _piper_available else 'gTTS fallback'}")
    log.info(f"Cosmos: {VLLM_URL} — {COSMOS_MODEL}")

    # Quick connectivity test
    test = ask_cosmos("Say exactly: E.R.I.C. online.")
    log.info(f"Cosmos test: {test}")
    speak(test)

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    telegram_app = app

    app.add_handler(CommandHandler("start",     cmd_start))
    app.add_handler(CommandHandler("engage",    cmd_engage))
    app.add_handler(CommandHandler("disengage", cmd_disengage))
    app.add_handler(CommandHandler("status",    cmd_status))
    app.add_handler(CommandHandler("introduce", cmd_introduce))
    app.add_handler(CommandHandler("look",      cmd_look))
    app.add_handler(CommandHandler("forward",   cmd_forward))
    app.add_handler(CommandHandler("stop",      cmd_stop_motors))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND, handle_message
    ))

    log.info("✅ E.R.I.C. ready — Telegram polling")
    app.run_polling()


if __name__ == "__main__":
    main()
