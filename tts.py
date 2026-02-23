"""
ERIC — Text to Speech
Piper via RealtimeTTS (CPU, zero VRAM) with gTTS fallback
Based on working Spencer/Grace voice chatbot implementation
"""

import time
import tempfile
import threading
import logging

from config import PIPER_BINARY, PIPER_MODEL

log = logging.getLogger("eric.tts")

_talk_stream     = None
_tts_lock        = threading.Lock()
_piper_available = False


def init_tts() -> bool:
    """Initialize Piper via RealtimeTTS with warm-up."""
    global _talk_stream, _piper_available
    try:
        from RealtimeTTS import TextToAudioStream, PiperEngine, PiperVoice

        voice = PiperVoice(
            model_file=PIPER_MODEL,
            config_file=PIPER_MODEL + ".json"
        )
        engine = PiperEngine(
            piper_path=PIPER_BINARY,
            voice=voice
        )

        _talk_stream = TextToAudioStream(engine, frames_per_buffer=1024, output_sample_rate=22050)

        # Warm up — prevents first sentence being cut off
        _talk_stream.feed("warm up").play(muted=True)

        _piper_available = True
        log.info("✅ TTS: Piper streaming (CPU, zero VRAM, warmed up)")
        return True

    except Exception as e:
        log.warning(f"⚠️  Piper unavailable ({e}) — using gTTS fallback")
        _piper_available = False
        try:
            import pygame
            pygame.mixer.init()
        except Exception:
            pass
        return False


def _play_kwargs():
    """Shared play() parameters that prevent sentence cutoff."""
    return dict(
        fast_sentence_fragment=False,
        fast_sentence_fragment_allsentences=False,
        fast_sentence_fragment_allsentences_multiple=False,
        buffer_threshold_seconds=1.0,
        minimum_sentence_length=25,
        minimum_first_fragment_length=20,
        force_first_fragment_after_words=25,
        comma_silence_duration=0.5,
        sentence_silence_duration=1.0,
        default_silence_duration=1.0
    )


def speak(text: str):
    """Non-blocking speak — fires in background thread."""
    threading.Thread(target=_speak_blocking, args=(text,), daemon=True).start()


def _speak_blocking(text: str):
    """Blocking speak — waits until audio fully finishes."""
    with _tts_lock:
        log.info(f"🔊 {text[:80]}")
        if _piper_available and _talk_stream:
            try:
                _talk_stream.feed(text).play(**_play_kwargs())
                # Wait until fully done
                while _talk_stream.is_playing():
                    time.sleep(0.1)
                return
            except Exception as e:
                log.warning(f"Piper error: {e}")
        _gtts_speak(text)


def speak_streaming(token_gen) -> str:
    """
    Feed token generator into Piper using play_async.
    Starts speaking as tokens arrive.
    Returns full collected text.
    """
    full = []

    if _piper_available and _talk_stream:
        def _gen():
            for chunk in token_gen:
                full.append(chunk)
                yield chunk
        try:
            _talk_stream.feed(_gen()).play_async(**_play_kwargs())
            while _talk_stream.is_playing():
                time.sleep(0.1)
            return "".join(full)
        except Exception as e:
            log.warning(f"Streaming TTS error: {e}")

    # Fallback — collect all then speak
    for chunk in token_gen:
        full.append(chunk)
    text = "".join(full)
    _gtts_speak(text)
    return text


def wait_speak_stop():
    """Block until TTS finishes playing."""
    if _talk_stream:
        while _talk_stream.is_playing():
            time.sleep(0.1)


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
                time.sleep(0.1)
    except Exception as e:
        log.error(f"gTTS error: {e}")


def piper_available() -> bool:
    return _piper_available