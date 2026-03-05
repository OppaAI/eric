"""
ERIC — Telegram Bot Handler
eric.agi.robot@gmail.com → @YourEricBot

Features:
  - Text commands → mission execution
  - Voice messages → faster-whisper transcribe → execute
  - Slash commands: /sar /greet /stop /status /email /photo /missions
  - Mission updates pushed to owner in real time
  - Photos sent inline when target found
  - Approximate location updates
  - Owner-only — all other users rejected

Security:
  - TELEGRAM_OWNER_ID whitelist — only owner can control Eric
  - Bot token in .env — never hardcoded

Setup:
  1. Message @BotFather → /newbot → copy token
  2. Message @userinfobot → copy your user ID
  3. Add to .env:
       TELEGRAM_BOT_TOKEN=xxxx
       TELEGRAM_OWNER_ID=123456789

Install:
  uv add python-telegram-bot
"""

import logging
import threading
import asyncio
import pathlib
import tempfile
import time
from typing import Optional

log = logging.getLogger("eric.telegram")

from config import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_OWNER_ID,
    TELEGRAM_ENABLED,
)

# ─── State ────────────────────────────────────────────────────────────────────
_app              = None
_loop             = None
_bot_thread       = None
_running          = False
_last_location    = "unknown"   # updated by mission.py callbacks


# ─── Init ─────────────────────────────────────────────────────────────────────

def init_telegram() -> bool:
    """Load bot and verify token. Returns True if ready."""
    if not TELEGRAM_ENABLED:
        log.info("Telegram: disabled in config")
        return False
    if not TELEGRAM_BOT_TOKEN:
        log.warning("Telegram: TELEGRAM_BOT_TOKEN not set")
        return False
    if not TELEGRAM_OWNER_ID:
        log.warning("Telegram: TELEGRAM_OWNER_ID not set")
        return False
    try:
        from telegram.ext import Application
        log.info("Telegram: bot initializing...")
        return True
    except ImportError:
        log.error("Telegram: python-telegram-bot not installed — run: uv add python-telegram-bot")
        return False


def telegram_available() -> bool:
    return TELEGRAM_ENABLED and bool(TELEGRAM_BOT_TOKEN) and bool(TELEGRAM_OWNER_ID)


# ─── Owner guard ──────────────────────────────────────────────────────────────

def _is_owner(update) -> bool:
    """Only the configured owner can control Eric."""
    return update.effective_user.id == int(TELEGRAM_OWNER_ID)


async def _reject(update):
    await update.message.reply_text(
        "⛔ Unauthorized. I only respond to my creator."
    )


# ─── Slash command handlers ───────────────────────────────────────────────────

async def cmd_start(update, context):
    if not _is_owner(update):
        await _reject(update)
        return
    await update.message.reply_text(
        "🤖 *ERIC online.*\n"
        "Edge Robotics Innovation by Cosmos.\n\n"
        "Commands:\n"
        "/sar — Search and Rescue\n"
        "/greet — Greet Owner\n"
        "/stop — Stop mission\n"
        "/status — System status\n"
        "/missions — List available missions\n"
        "/email — Check email\n"
        "/photo — Take a photo now\n"
        "/location — Report position\n\n"
        "Or just send me a mission in plain text.",
        parse_mode="Markdown"
    )


async def cmd_sar(update, context):
    if not _is_owner(update): await _reject(update); return
    await _start_named_mission(update, "search_and_rescue")


async def cmd_greet(update, context):
    if not _is_owner(update): await _reject(update); return
    await _start_named_mission(update, "greet_owner")


async def cmd_stop(update, context):
    if not _is_owner(update): await _reject(update); return
    try:
        from mission import stop_mission
        stop_mission()
        await update.message.reply_text("⏹ Mission stopped.")
    except Exception as e:
        await update.message.reply_text(f"❌ Stop error: {e}")


async def cmd_status(update, context):
    if not _is_owner(update): await _reject(update); return
    status = _build_status_text()
    await update.message.reply_text(status, parse_mode="Markdown")


async def cmd_missions(update, context):
    if not _is_owner(update): await _reject(update); return
    try:
        from mission import list_missions
        missions = list_missions()
        if not missions:
            await update.message.reply_text("No missions found.")
            return
        text = "📋 *Available missions:*\n" + "\n".join(f"• `{m}`" for m in missions)
        await update.message.reply_text(text, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ {e}")


async def cmd_email(update, context):
    if not _is_owner(update): await _reject(update); return
    try:
        from config import EMAIL_ENABLED
        if not EMAIL_ENABLED:
            await update.message.reply_text("📧 Email is disabled.")
            return
        from email_handler import check_inbox, check_approvals
        await update.message.reply_text("📧 Checking email...")
        msgs = check_inbox()
        check_approvals()
        if not msgs:
            await update.message.reply_text("📭 No new messages.")
        else:
            count = len(msgs)
            text = f"📬 {count} new message{'s' if count > 1 else ''}:\n"
            for m in msgs[:3]:
                text += f"\n• From: {m['sender']}\n  Subject: {m['subject']}\n"
            if count > 3:
                text += f"\n...and {count - 3} more."
            text += "\n\nDrafts forwarded to your email for approval."
            await update.message.reply_text(text)
    except Exception as e:
        await update.message.reply_text(f"❌ Email error: {e}")


async def cmd_photo(update, context):
    if not _is_owner(update): await _reject(update); return
    await update.message.reply_text("📸 Taking photo...")
    try:
        from cosmos import capture_frame, CAMERA_PANTILT, CAMERA_WEBCAM
        import io
        frame_pt = capture_frame(CAMERA_PANTILT)
        frame_wc = capture_frame(CAMERA_WEBCAM)
        for label, frame in [("Pan-tilt", frame_pt), ("Webcam", frame_wc)]:
            if frame is not None:
                buf = _frame_to_bytes(frame)
                if buf:
                    await update.message.reply_photo(
                        photo=buf,
                        caption=f"📷 {label} — {time.strftime('%H:%M:%S')}"
                    )
    except Exception as e:
        await update.message.reply_text(f"❌ Photo error: {e}")


async def cmd_location(update, context):
    if not _is_owner(update): await _reject(update); return
    await update.message.reply_text(f"📍 Last known position: {_last_location}")


# ─── Text message handler ─────────────────────────────────────────────────────

async def handle_text(update, context):
    if not _is_owner(update): await _reject(update); return

    text = update.message.text.strip()
    if not text:
        return

    log.info(f"Telegram: text from owner → {text!r}")

    # Check email commands
    try:
        from config import EMAIL_ENABLED
        if EMAIL_ENABLED:
            from email_handler import handle_voice_email_command
            resp = handle_voice_email_command(text)
            if resp:
                await update.message.reply_text(f"📧 {resp}")
                return
    except Exception:
        pass

    # Route to mission
    try:
        from mission import get_mission_active, handle_character_response, start_mission
        if get_mission_active():
            resp = handle_character_response("Operator", text)
            if resp:
                await update.message.reply_text(f"🤖 {resp}")
        else:
            await update.message.reply_text(f"🎯 Starting mission: _{text}_", parse_mode="Markdown")
            start_mission(text)
    except Exception as e:
        await update.message.reply_text(f"❌ {e}")


# ─── Voice message handler ────────────────────────────────────────────────────

async def handle_voice(update, context):
    if not _is_owner(update): await _reject(update); return

    await update.message.reply_text("🎙 Transcribing voice message...")

    try:
        # Download voice file
        voice = update.message.voice
        tg_file = await context.bot.get_file(voice.file_id)

        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp_path = tmp.name

        await tg_file.download_to_drive(tmp_path)

        # Transcribe with faster-whisper
        from faster_whisper import WhisperModel
        from config import ASR_MODEL, ASR_DEVICE

        # Reuse loaded model from voice.py if available
        whisper = None
        try:
            from voice import _whisper_model
            whisper = _whisper_model
        except Exception:
            pass

        if whisper is None:
            whisper = WhisperModel(ASR_MODEL, device=ASR_DEVICE, compute_type="int8")

        segments, _ = whisper.transcribe(tmp_path, beam_size=1, vad_filter=True)
        text = " ".join(s.text for s in segments).strip()

        pathlib.Path(tmp_path).unlink(missing_ok=True)

        if not text:
            await update.message.reply_text("🔇 Couldn't hear anything.")
            return

        await update.message.reply_text(f"💬 Heard: _{text}_", parse_mode="Markdown")

        # Route same as text
        try:
            from mission import get_mission_active, handle_character_response, start_mission
            if get_mission_active():
                resp = handle_character_response("Operator", text)
                if resp:
                    await update.message.reply_text(f"🤖 {resp}")
            else:
                await update.message.reply_text(f"🎯 Starting mission: _{text}_", parse_mode="Markdown")
                start_mission(text)
        except Exception as e:
            await update.message.reply_text(f"❌ {e}")

    except Exception as e:
        log.error(f"Telegram: voice handler error — {e}")
        await update.message.reply_text(f"❌ Voice error: {e}")


# ─── Mission start helper ─────────────────────────────────────────────────────

async def _start_named_mission(update, mission_name: str):
    """Start a mission by YAML name."""
    try:
        from mission import start_mission, get_briefing_from_file
        briefing = get_briefing_from_file(mission_name)
        if not briefing:
            await update.message.reply_text(f"❌ Mission `{mission_name}` not found.", parse_mode="Markdown")
            return
        await update.message.reply_text(f"🚀 Starting _{mission_name}_...", parse_mode="Markdown")
        start_mission(briefing.strip(), mission_name=mission_name)
    except Exception as e:
        await update.message.reply_text(f"❌ {e}")


# ─── Push notifications (called from mission.py) ──────────────────────────────

def notify(text: str):
    """Send a text notification to owner. Fire-and-forget, thread-safe."""
    if not telegram_available() or _app is None:
        return
    asyncio.run_coroutine_threadsafe(
        _app.bot.send_message(chat_id=int(TELEGRAM_OWNER_ID), text=text),
        _loop
    )


def notify_photo(image_path: str, caption: str = ""):
    """Send a photo to owner."""
    if not telegram_available() or _app is None:
        return
    path = pathlib.Path(image_path)
    if not path.exists():
        return
    async def _send():
        with open(path, "rb") as f:
            await _app.bot.send_photo(
                chat_id=int(TELEGRAM_OWNER_ID),
                photo=f,
                caption=caption or path.name
            )
    asyncio.run_coroutine_threadsafe(_send(), _loop)


def notify_mission_update(event: str, detail: str = ""):
    """Called by mission.py on key events."""
    icons = {
        "started":   "🚀",
        "found":     "🎯",
        "alarm":     "🚨",
        "photo":     "📸",
        "complete":  "✅",
        "stopped":   "⏹",
        "error":     "❌",
        "nav":       "🚗",
    }
    icon = icons.get(event, "ℹ")
    msg  = f"{icon} *{event.upper()}*"
    if detail:
        msg += f"\n{detail}"
    notify(msg)


def set_location(location: str):
    """Update Eric's last known position string."""
    global _last_location
    _last_location = location


# ─── Status builder ───────────────────────────────────────────────────────────

def _build_status_text() -> str:
    lines = ["🤖 *ERIC Status*\n"]

    try:
        from mission import get_mission_active, get_mission_state
        active = get_mission_active()
        state  = get_mission_state() if active else "idle"
        lines.append(f"Mission: {'🟢 ' + state.upper() if active else '⚫ IDLE'}")
    except Exception:
        lines.append("Mission: unknown")

    try:
        import requests
        from config import VLLM_URL
        r = requests.get(VLLM_URL.replace("/v1/chat/completions", "/health"), timeout=1.5)
        lines.append(f"Cosmos: {'🟢 ONLINE' if r.status_code == 200 else '🔴 OFFLINE'}")
    except Exception:
        lines.append("Cosmos: 🔴 OFFLINE")

    try:
        from lidar import lidar_available, get_status as ls
        if lidar_available():
            s = ls()
            d = s.get("min_distance", 999)
            lines.append(f"LiDAR: 🟢 {d:.2f}m")
        else:
            lines.append("LiDAR: 🔴 OFFLINE")
    except Exception:
        lines.append("LiDAR: ⚫ N/A")

    try:
        from tts import piper_available
        lines.append(f"TTS: {'🟢 PIPER' if piper_available() else '🟡 GTTS'}")
    except Exception:
        lines.append("TTS: ⚫ N/A")

    try:
        from voice import voice_available, is_session_active
        if voice_available():
            session = "active" if is_session_active() else "sleeping"
            lines.append(f"Voice: 🟢 {session.upper()}")
        else:
            lines.append("Voice: 🔴 OFFLINE")
    except Exception:
        lines.append("Voice: ⚫ N/A")

    lines.append(f"\n📍 Position: {_last_location}")
    lines.append(f"🕐 {time.strftime('%Y-%m-%d %H:%M:%S')}")

    return "\n".join(lines)


# ─── Frame helper ─────────────────────────────────────────────────────────────

def _frame_to_bytes(frame) -> Optional[bytes]:
    """Convert numpy/PIL frame to JPEG bytes for Telegram."""
    try:
        import io
        import numpy as np
        if hasattr(frame, "tobytes"):
            # PIL Image
            buf = io.BytesIO()
            frame.save(buf, format="JPEG")
            return buf.getvalue()
        elif isinstance(frame, np.ndarray):
            import cv2
            _, encoded = cv2.imencode(".jpg", frame)
            return encoded.tobytes()
    except Exception as e:
        log.debug(f"Telegram: frame_to_bytes error — {e}")
    return None


# ─── Bot runner ───────────────────────────────────────────────────────────────

def start_telegram_bot():
    """Start the Telegram bot in a background thread."""
    global _bot_thread, _running
    if not telegram_available():
        return False
    _running = True
    _bot_thread = threading.Thread(
        target=_run_bot_loop,
        daemon=True,
        name="telegram-bot"
    )
    _bot_thread.start()
    log.info("Telegram: bot thread started")
    return True


def stop_telegram_bot():
    global _running
    _running = False
    if _app and _loop:
        asyncio.run_coroutine_threadsafe(_app.stop(), _loop)


def _run_bot_loop():
    """Run the async bot in a dedicated event loop."""
    global _app, _loop

    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)

    try:
        from telegram.ext import (
            Application, CommandHandler, MessageHandler,
            filters
        )

        _app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

        # Register handlers
        _app.add_handler(CommandHandler("start",    cmd_start))
        _app.add_handler(CommandHandler("sar",      cmd_sar))
        _app.add_handler(CommandHandler("greet",    cmd_greet))
        _app.add_handler(CommandHandler("stop",     cmd_stop))
        _app.add_handler(CommandHandler("status",   cmd_status))
        _app.add_handler(CommandHandler("missions", cmd_missions))
        _app.add_handler(CommandHandler("email",    cmd_email))
        _app.add_handler(CommandHandler("photo",    cmd_photo))
        _app.add_handler(CommandHandler("location", cmd_location))

        _app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
        _app.add_handler(MessageHandler(filters.VOICE, handle_voice))

        log.info("Telegram: bot polling started")
        _app.run_polling(stop_signals=None)

    except Exception as e:
        log.error(f"Telegram: bot loop error — {e}")
    finally:
        _loop.close()
