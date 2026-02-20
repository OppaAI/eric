"""
E.R.I.C. — Text to Speech
Piper streaming (CPU, zero VRAM) with gTTS fallback
"""

import threading
import tempfile
import logging

from config import PIPER_BINARY, PIPER_MODEL

log = logging.getLogger("eric.tts")

_tts_stream      = None
_tts_lock        = threading.Lock()
_piper_available = False


def init_tts() -> bool:
    global _tts_stream, _piper_available
    try:
        from RealtimeTTS import TextToAudioStream, PiperEngine, PiperVoice
        engine = PiperEngine(
            piper_path=PIPER_BINARY,
            voice=PiperVoice(
                model_file=PIPER_MODEL,
                config_file=PIPER_MODEL + ".json"
            )
        )
        _tts_stream      = TextToAudioStream(engine)
        _piper_available = True
        log.info("✅ TTS: Piper streaming (CPU)")
        return True
    except Exception as e:
        log.warning(f"⚠️  Piper unavailable ({e}) — using gTTS fallback")
        _piper_available = False
        return False


def speak(text: str):
    """Non-blocking speak — runs in background thread."""
    threading.Thread(target=_speak_blocking, args=(text,), daemon=True).start()


def _speak_blocking(text: str):
    with _tts_lock:
        log.info(f"🔊 {text[:80]}")
        if _piper_available and _tts_stream:
            try:
                _tts_stream.feed(text)
                _tts_stream.play()
                return
            except Exception as e:
                log.warning(f"Piper error: {e}")
        _gtts_speak(text)


def speak_streaming(token_generator) -> str:
    """
    Feed a token generator directly into Piper TTS.
    Starts speaking as first tokens arrive — minimal latency.
    Returns full collected text.
    Falls back to blocking speak if Piper unavailable.
    """
    full_text = []

    if _piper_available and _tts_stream:
        def _gen():
            for chunk in token_generator:
                full_text.append(chunk)
                yield chunk

        with _tts_lock:
            try:
                _tts_stream.feed(_gen())
                _tts_stream.play()
                return "".join(full_text)
            except Exception as e:
                log.warning(f"Streaming TTS error: {e}")

    # Fallback — collect all tokens then speak
    for chunk in token_generator:
        full_text.append(chunk)
    text = "".join(full_text)
    _speak_blocking(text)
    return text


def _gtts_speak(text: str):
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


def piper_available() -> bool:
    return _piper_available
