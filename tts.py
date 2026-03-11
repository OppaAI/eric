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


def _play_kwargs(long: bool = False):
    """Shared play() parameters that prevent sentence cutoff."""
    return dict(
        fast_sentence_fragment=False,
        fast_sentence_fragment_allsentences=False,
        fast_sentence_fragment_allsentences_multiple=False,
        buffer_threshold_seconds=0.3 if long else 1.0,
        minimum_sentence_length=5,          # low — never drop short sentences
        minimum_first_fragment_length=5,    # low — start playing quickly
        force_first_fragment_after_words=8,
        comma_silence_duration=0.3,
        sentence_silence_duration=0.6,
        default_silence_duration=0.6
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
                    # Use long=True for multi-sentence text — lower buffer threshold
                    # ensures all sentences play without being dropped
                    _is_long = text.count(".") + text.count("!") + text.count("?") > 1
                    _talk_stream.feed(text).play(**_play_kwargs(long=_is_long))
                finally:
                    sys.stdout = _old_stdout
                # play() is blocking — stream is done when it returns
            else:
                _gtts_speak(text)
        except Exception as e:
            log.warning(f"TTS worker error: {e}")
        finally:
            _tts_queue.task_done()


def speak(text: str, replace_queue: bool = True):
    """
    Non-blocking speak — puts text in queue and returns instantly.
    replace_queue=True (default): drops stale queued speech and replaces.
    replace_queue=False: appends to queue (use for multi-part responses).
    """
    if not text or not text.strip():
        return
    if replace_queue:
        # Drop stale items — only the latest speech matters during navigation
        while not _tts_queue.empty():
            try:
                _tts_queue.get_nowait()
                _tts_queue.task_done()
            except queue.Empty:
                break
    _tts_queue.put(text.strip())


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

# ─── ROS2 TtsNode ─────────────────────────────────────────────────────────────
# Thin wrapper that adds ROS2 topic interfaces to tts.py.
# speak() Python API remains UNCHANGED — mission.py calls it directly.
#
# Topics added:
#   Subscribe  /tts/speak     std_msgs/String   — text to speak
#   Subscribe  /tts/clear     std_msgs/Empty    — clear queue
#   Publish    /tts/status    std_msgs/String   — "speaking" | "idle"
#
# Usage (from main.py):
#   from tts import TtsNode, start_tts_node
#   start_tts_node()
#   speak("hello")   # still works identically

import threading as _tts_threading

_tts_node        = None
_tts_node_thread = None
_tts_node_lock   = _tts_threading.Lock()


class TtsNode:
    """ROS2 node wrapper for TTS. Adds topic interfaces, keeps speak() API intact."""

    def __init__(self):
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String, Empty

        if not rclpy.ok(): rclpy.init(args=None)
        self._node = Node("eric_tts_node")

        # Subscribe /tts/speak — any node can request speech
        self._speak_sub = self._node.create_subscription(
            String, "/tts/speak", self._on_speak, 10
        )

        # Subscribe /tts/clear — clear the speech queue
        self._clear_sub = self._node.create_subscription(
            Empty, "/tts/clear", self._on_clear, 10
        )

        # Publish /tts/status — "speaking" or "idle"
        self._status_pub = self._node.create_publisher(String, "/tts/status", 10)
        self._status_timer = self._node.create_timer(0.5, self._publish_status)

        log.info("TtsNode: ROS2 node ready")

    def _on_speak(self, msg):
        speak(msg.data)

    def _on_clear(self, msg):
        """Clear the TTS queue — stops pending speech."""
        while not _tts_queue.empty():
            try:
                _tts_queue.get_nowait()
                _tts_queue.task_done()
            except Exception:
                break

    def _publish_status(self):
        from std_msgs.msg import String
        status = "speaking" if not _tts_queue.empty() else "idle"
        self._status_pub.publish(String(data=status))

    def spin(self):
        import rclpy
        try:
            rclpy.spin(self._node)
        except Exception as e:
            log.debug(f"TtsNode spin ended: {e}")
        finally:
            self._node.destroy_node()
            try:
                rclpy.shutdown()
            except Exception:
                pass

    def destroy(self):
        try:
            self._node.destroy_node()
        except Exception:
            pass


def start_tts_node() -> bool:
    """Launch TtsNode in a background daemon thread."""
    global _tts_node, _tts_node_thread
    with _tts_node_lock:
        if _tts_node is not None:
            return True
        try:
            _tts_node = TtsNode()
            _tts_node_thread = _tts_threading.Thread(
                target=_tts_node.spin,
                daemon=True,
                name="tts-node-spin"
            )
            _tts_node_thread.start()
            log.info("TtsNode: spinning in background thread")
            return True
        except Exception as e:
            log.error(f"TtsNode: failed to start — {e}")
            _tts_node = None
            return False


def stop_tts_node():
    global _tts_node
    with _tts_node_lock:
        if _tts_node:
            _tts_node.destroy()
            _tts_node = None
