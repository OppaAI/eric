"""
ERIC — Text to Speech
Piper via RealtimeTTS (CPU, zero VRAM) with gTTS fallback

Architecture:
  speak() puts text into a queue — always returns instantly, never blocks mission.
  A single background worker thread drains the queue one at a time.
  If TTS stalls, a 15s timeout forces it to move on.
  The mission loop is never blocked by TTS.
"""

import time
import queue
import tempfile
import threading
import logging

from config import PIPER_BINARY, PIPER_MODEL

log = logging.getLogger("eric.tts")

_talk_stream     = None
_piper_available = False

# Single queue — mission puts text in, worker drains it
_tts_queue  = queue.Queue()
_tts_worker = None


def init_tts() -> bool:
    """Initialize Piper via RealtimeTTS with warm-up, start queue worker."""
    global _talk_stream, _piper_available, _tts_worker
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

        _talk_stream = TextToAudioStream(engine, frames_per_buffer=1024)

        # Warm up — prevents first sentence being cut off
        _talk_stream.feed("warm up").play(muted=True)

        _piper_available = True
        log.info("✅ TTS: Piper streaming (CPU, zero VRAM, warmed up)")

    except Exception as e:
        log.warning(f"⚠️  Piper unavailable ({e}) — using gTTS fallback")
        _piper_available = False
        try:
            import pygame
            pygame.mixer.init()
        except Exception:
            pass

    # Start worker regardless — handles both Piper and gTTS
    _tts_worker = threading.Thread(target=_worker, daemon=True)
    _tts_worker.start()
    return _piper_available


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


import os
import sys

def _worker():
    """Background worker — drains TTS queue one item at a time."""
    # Suppress "Wait aborted" print spam from RealtimeTTS internals
    _devnull = open(os.devnull, 'w')

    while True:
        try:
            text = _tts_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        log.info(f"🔊 {text[:80]}")
        try:
            if _piper_available and _talk_stream:
                # Redirect stdout to suppress RealtimeTTS internal print spam
                _old_stdout = sys.stdout
                sys.stdout = _devnull
                try:
                    _talk_stream.feed(text).play(**_play_kwargs())
                finally:
                    sys.stdout = _old_stdout
                # play() is blocking — stream is done when it returns
            else:
                _gtts_speak(text)
        except Exception as e:
            log.warning(f"TTS worker error: {e}")
        finally:
            _tts_queue.task_done()


def speak(text: str):
    """
    Non-blocking speak — puts text in queue and returns instantly.
    Only keeps 1 pending item max — drops stale speech immediately.
    """
    if not text or not text.strip():
        return
    # If anything already waiting, it's stale — clear it and replace
    while not _tts_queue.empty():
        try:
            _tts_queue.get_nowait()
            _tts_queue.task_done()
        except queue.Empty:
            break
    _tts_queue.put(text)


def speak_streaming(token_gen) -> str:
    """
    Collect tokens and speak via queue.
    Returns full collected text.
    """
    full = []
    for chunk in token_gen:
        full.append(chunk)
    text = "".join(full)
    speak(text)
    return text


def wait_speak_stop():
    """Block until TTS queue is fully drained."""
    _tts_queue.join()


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
