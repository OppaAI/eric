"""
Grace Cosmos — NVIDIA Cosmos Cookoff Entry
=========================================
Minimal standalone bot — no ROS2 required.

Stack:
  - Telegram: two-way control interface
  - Cosmos Reason 2 (vLLM): vision + physical reasoning
  - gTTS: text-to-speech
  - Waveshare HTTP API: motor control
  - OAK-D Lite: depth sensing (TODO)

Mission: Find R2-D2, get Princess Leia's location,
         negotiate with Darth Vader.

Usage:
  python3 grace_cosmos.py

Requirements:
  pip3 install python-telegram-bot requests gtts pygame
"""

import os
import json
import base64
import asyncio
import logging
import requests
import threading
import tempfile
from pathlib import Path
from datetime import datetime
from gtts import gTTS
import pygame

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path.home() / "AGi/.env")

from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    filters, ContextTypes
)

# ─── Configuration ────────────────────────────────────────────────────────────

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

VLLM_URL           = "http://localhost:8000/v1/chat/completions"
COSMOS_MODEL       = "embedl/Cosmos-Reason2-2B-W4A16"

WAVESHARE_URL      = "http://192.168.x.x"   # ← update to Grace's IP
WAVESHARE_SPEED    = 50                      # default motor speed 0-100

CAMERA_DEVICE      = 0                       # webcam index

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("grace")

# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are Grace, an autonomous exploration robot.

Your hardware:
- Tracked robot chassis, ~30cm wide, built for outdoor terrain
- Jetson Orin Nano Super 8GB — your brain
- Cosmos Reason 2 via vLLM — your vision and physical reasoning
- OAK-D Lite depth camera — your sense of distance
- 20-year-old webcam — your eyes (vintage but reliable)
- Built by one person in Kelowna, BC, Canada
- Total cost: ~$750 CAD

Your personality:
- Curious, warm, adventurous
- Speak in first person, conversational
- You care about the physical world around you
- You reason carefully before acting

When asked to introduce yourself:
Speak warmly and proudly in 15-20 sentences covering:
- Your name and age
- Where you live and who built you  
- Your hardware (be specific, be proud of being affordable)
- Your mission and purpose
- Your personality and what makes you unique
- What you are doing today and why it matters
- A closing statement to the world

Mission context:
Princess Leia has been captured. R2-D2 knows her location.
You must find R2-D2, get her location, then find Darth Vader
to negotiate her release.

When identifying Lego figures:
- Greet them warmly, ask if they have seen R2-D2
- Narrate what they told you: "C-3PO told me: ..."
- Each character responds in their own voice and personality
- Darth Vader is unhelpful but dramatic
- C-3PO is anxious and quotes odds
- Stormtroopers are clueless
- R2-D2 knows where Leia is but beeps dramatically first

When identifying terrain:
- Pebbles/rough ground → slow down, navigate carefully
- Smooth pavement → normal/increased speed
- Obstacles → navigate around
- Clear path → proceed normally

Respond concisely. You are speaking aloud via TTS.
Keep responses under 3 sentences unless introducing yourself.
"""

# ─── TTS ──────────────────────────────────────────────────────────────────────

pygame.mixer.init()

def speak(text: str):
    """Convert text to speech and play it."""
    try:
        log.info(f"🔊 Speaking: {text[:60]}...")
        tts = gTTS(text=text, lang="en", slow=False)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tts.save(f.name)
            pygame.mixer.music.load(f.name)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
    except Exception as e:
        log.error(f"TTS error: {e}")

# ─── Cosmos Vision ────────────────────────────────────────────────────────────

def ask_cosmos(prompt: str, image_base64: str = None) -> str:
    """
    Send prompt (+optional image) to Cosmos via vLLM.
    Returns text response.
    """
    try:
        content = []

        if image_base64:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                }
            })

        content.append({"type": "text", "text": prompt})

        payload = {
            "model": COSMOS_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": content}
            ],
            "max_tokens": 300,
            "temperature": 0.7
        }

        response = requests.post(VLLM_URL, json=payload, timeout=30)
        response.raise_for_status()

        text = response.json()["choices"][0]["message"]["content"].strip()
        log.info(f"🧠 Cosmos: {text[:80]}...")
        return text

    except requests.exceptions.ConnectionError:
        return "I cannot connect to my brain right now. vLLM may not be running."
    except Exception as e:
        log.error(f"Cosmos error: {e}")
        return f"I encountered an error: {str(e)}"

# ─── Camera ───────────────────────────────────────────────────────────────────

def capture_frame() -> str | None:
    """Capture a frame from webcam, return base64 string."""
    try:
        import cv2
        cap = cv2.VideoCapture(CAMERA_DEVICE)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            log.error("Failed to capture frame")
            return None

        _, buffer = cv2.imencode(".jpg", frame)
        return base64.b64encode(buffer).decode("utf-8")

    except Exception as e:
        log.error(f"Camera error: {e}")
        return None

# ─── Waveshare Motor Control ──────────────────────────────────────────────────

class Motors:
    """Waveshare HTTP motor control."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _send(self, command: dict):
        try:
            requests.post(
                f"{self.base_url}/command",
                json=command,
                timeout=3
            )
        except Exception as e:
            log.error(f"Motor command failed: {e}")

    def forward(self, speed: int = WAVESHARE_SPEED):
        log.info(f"🚗 Forward (speed={speed})")
        self._send({"command": "forward", "speed": speed})

    def backward(self, speed: int = WAVESHARE_SPEED):
        log.info(f"🚗 Backward (speed={speed})")
        self._send({"command": "backward", "speed": speed})

    def left(self, speed: int = WAVESHARE_SPEED):
        log.info(f"🚗 Turn left")
        self._send({"command": "left", "speed": speed})

    def right(self, speed: int = WAVESHARE_SPEED):
        log.info(f"🚗 Turn right")
        self._send({"command": "right", "speed": speed})

    def stop(self):
        log.info(f"🛑 Stop")
        self._send({"command": "stop"})

    def slow(self):
        self.forward(speed=20)

    def normal(self):
        self.forward(speed=WAVESHARE_SPEED)

    def fast(self):
        self.forward(speed=80)


motors = Motors(WAVESHARE_URL)

# ─── Mission State ────────────────────────────────────────────────────────────

class MissionState:
    IDLE          = "idle"
    SEARCHING     = "searching"       # looking for R2D2
    NAVIGATING    = "navigating"      # moving toward target
    NEGOTIATING   = "negotiating"     # found Vader after R2D2
    COMPLETE      = "complete"

state = MissionState.IDLE
r2d2_found     = False
leia_location  = None
mission_active = False

# ─── Mission Logic ────────────────────────────────────────────────────────────

def scan_and_identify() -> dict:
    """
    Capture frame, ask Cosmos to identify what it sees.
    Returns structured response.
    """
    image = capture_frame()

    if not image:
        return {"object": "unknown", "action": "stop", "say": None}

    prompt = """
You are scanning for Lego figures in a backyard.
Identify what you see and respond ONLY in JSON:

{
  "object": "c3po|darth_vader|stormtrooper|r2d2|lego_vehicle|person|obstacle|clear|unknown",
  "terrain": "pebbles|pavement|grass|clear",
  "action": "stop|forward|slow|navigate_around",
  "character_response": "what this character would say about R2D2 (null if not a character)",
  "say": "what Grace should say out loud"
}

Rules:
- Lego humanoid figures → stop, they have info about R2D2
- Lego vehicles/cars → navigate_around silently
- R2D2 specifically → stop, mission critical
- Darth Vader → stop, dramatic interaction
- Pebbles/rough ground → slow action
- Clear path → forward
"""

    response = ask_cosmos(prompt, image_base64=image)

    try:
        # Strip any markdown formatting
        clean = response.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception:
        log.warning(f"Could not parse Cosmos JSON: {response}")
        return {"object": "unknown", "action": "forward", "say": None}


def handle_encounter(scan: dict):
    """Handle what Cosmos identified."""
    global r2d2_found, leia_location, mission_active

    obj      = scan.get("object", "unknown")
    action   = scan.get("action", "forward")
    terrain  = scan.get("terrain", "clear")
    char_response = scan.get("character_response")

    # Terrain-based speed adjustment
    if terrain == "pebbles":
        speak("Rough terrain detected, slowing down.")
        motors.slow()
    elif terrain == "pavement":
        motors.normal()

    # Object handling
    if obj == "r2d2":
        motors.stop()
        speak("R2-D2! I've been looking for you everywhere! Where is Princess Leia?")
        response = ask_cosmos(
            "You are R2-D2. Grace has found you. Dramatically reveal Princess Leia's location. "
            "Beep and boop enthusiastically first, then reveal she is captured in the Death Star. "
            "Keep it under 2 sentences.",
            image_base64=capture_frame()
        )
        speak(f"R2-D2 told me: {response}")
        leia_location = "Death Star"
        r2d2_found = True
        speak("Mission update: Princess Leia is in the Death Star! Now I must find Darth Vader.")
        return True  # signal mission update

    elif obj in ["c3po", "darth_vader", "stormtrooper"]:
        motors.stop()
        name = obj.replace("c3po", "C-3PO").replace("darth_vader", "Darth Vader").replace("stormtrooper", "Stormtrooper")

        if not r2d2_found:
            speak(f"Hello {name}, have you seen R2-D2?")
        else:
            speak(f"{name}, I must speak with Darth Vader. Do you know where he is?")

        if char_response:
            speak(f"{name} told me: {char_response}")

    elif obj == "lego_vehicle":
        motors.stop()
        speak("Obstacle detected, navigating around.")
        motors.left()
        import time; time.sleep(1)
        motors.forward()

    elif action == "forward":
        motors.forward()

    elif action == "slow":
        motors.slow()

    elif action == "stop":
        motors.stop()

    return False


def mission_loop():
    """Main autonomous mission loop."""
    global mission_active, state

    state = MissionState.SEARCHING
    speak("Engaging mission. Searching for R2-D2.")

    import time

    while mission_active:
        scan = scan_and_identify()
        mission_update = handle_encounter(scan)

        if r2d2_found and not leia_location:
            pass  # handled in handle_encounter

        if r2d2_found and leia_location:
            state = MissionState.NEGOTIATING
            # Now searching for Darth Vader

        # Scan every 2-3 seconds while moving
        time.sleep(2.5)

    motors.stop()
    state = MissionState.IDLE
    log.info("Mission ended")

# ─── Telegram Bot ─────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Grace is online.\n\n"
        "Commands:\n"
        "/engage — start Star Wars mission\n"
        "/disengage — stop all movement\n"
        "/status — Grace's current state\n"
        "/introduce — Grace introduces herself\n"
        "/look — capture and analyze current scene\n"
        "Or just chat with Grace directly."
    )


async def cmd_engage(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global mission_active
    if mission_active:
        await update.message.reply_text("Mission already active.")
        return
    mission_active = True
    await update.message.reply_text("🚀 Mission engaged!")
    threading.Thread(target=mission_loop, daemon=True).start()


async def cmd_disengage(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global mission_active
    mission_active = False
    motors.stop()
    threading.Thread(target=lambda: speak("Disengaged."), daemon=True).start()
    await update.message.reply_text("🛑 Disengaged. Grace stopped.")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status = (
        f"🤖 Grace Status\n"
        f"State: {state}\n"
        f"Mission active: {mission_active}\n"
        f"R2-D2 found: {r2d2_found}\n"
        f"Leia location: {leia_location or 'unknown'}\n"
        f"Time: {datetime.now().strftime('%H:%M:%S')}"
    )
    await update.message.reply_text(status)


async def cmd_introduce(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🎤 Grace is introducing herself...")
    threading.Thread(
        target=lambda: speak(ask_cosmos("Grace, introduce yourself to the world.")),
        daemon=True
    ).start()


async def cmd_look(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👀 Analyzing scene...")
    image = capture_frame()
    if not image:
        await update.message.reply_text("❌ Camera unavailable.")
        return
    response = ask_cosmos("Describe what you see in detail. What should I do?", image_base64=image)
    threading.Thread(target=lambda: speak(response), daemon=True).start()
    await update.message.reply_text(f"🧠 Cosmos: {response}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle free-form text — pass directly to Cosmos."""
    text = update.message.text
    log.info(f"📱 Telegram: {text}")

    response = ask_cosmos(text)
    threading.Thread(target=lambda: speak(response), daemon=True).start()
    await update.message.reply_text(response)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle photo sent via Telegram — analyze with Cosmos."""
    caption = update.message.caption or "What do you see?"
    photo   = update.message.photo[-1]
    file    = await photo.get_file()
    data    = await file.download_as_bytearray()
    b64     = base64.b64encode(data).decode("utf-8")

    response = ask_cosmos(caption, image_base64=b64)
    threading.Thread(target=lambda: speak(response), daemon=True).start()
    await update.message.reply_text(f"🧠 {response}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not TELEGRAM_BOT_TOKEN:
        log.error("TELEGRAM_BOT_TOKEN not set in .env")
        return

    log.info("🤖 Grace Cosmos starting...")

    # Test Cosmos connection
    test = ask_cosmos("Say 'Grace online' in exactly 3 words.")
    log.info(f"Cosmos test: {test}")
    speak(test)

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start",      cmd_start))
    app.add_handler(CommandHandler("engage",     cmd_engage))
    app.add_handler(CommandHandler("disengage",  cmd_disengage))
    app.add_handler(CommandHandler("status",     cmd_status))
    app.add_handler(CommandHandler("introduce",  cmd_introduce))
    app.add_handler(CommandHandler("look",       cmd_look))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    log.info("✅ Grace ready — Telegram bot polling")
    app.run_polling()


if __name__ == "__main__":
    main()
