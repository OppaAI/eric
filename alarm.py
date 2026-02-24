"""
ERIC — Alarm / Alert System
=============================
Multi-modal alerts triggered by specific mission events:

  SIREN       — search and rescue: injured person found
                TTS urgent announcement + rapid LED flash + rising tone
  HAZARD      — hazard patrol: dangerous object/substance found
                TTS safety warning + slow LED pulse + warning tone
  SUSPICIOUS  — anti-terror: suspicious object found
                TTS alert + LED strobe + urgent tone sequence
  NATURE      — nature explore: wildlife / scenic photo opportunity
                TTS excited comment + gentle LED pulse (non-alarming)
  CLEAR       — all-clear after alarm
                TTS confirmation + steady green LED

Audio strategy:
  Primary:  pygame.mixer.Sound with generated sine/square tones (no external files needed)
  Fallback: TTS only if pygame unavailable

LED strategy:
  motors.lights() rapid flash patterns
  All patterns run in a background daemon thread — never blocks mission loop

Usage:
  from alarm import sound_alarm, AlarmType

  sound_alarm(AlarmType.SIREN,      "Injured person found near staircase")
  sound_alarm(AlarmType.HAZARD,     "Gas canister detected — possible leak")
  sound_alarm(AlarmType.SUSPICIOUS, "Unidentified package under desk")
  sound_alarm(AlarmType.NATURE,     "Red fox spotted near stream")
"""

import threading
import time
import logging
import struct
import math

log = logging.getLogger("eric.alarm")

# ── Alarm types ───────────────────────────────────────────────────────────────
class AlarmType:
    SIREN      = "siren"       # search and rescue — found injured person
    HAZARD     = "hazard"      # environment hazard found
    SUSPICIOUS = "suspicious"  # suspicious / terror object found
    NATURE     = "nature"      # wildlife / scenic discovery
    CLEAR      = "clear"       # all-clear / situation resolved
    NONE       = "none"        # no alarm — silent find (e.g. narrative missions)


# ── Alarm config ──────────────────────────────────────────────────────────────
_ALARM_CONFIG = {
    AlarmType.SIREN: {
        "voice_prefix": "EMERGENCY! EMERGENCY! ",
        "led_pattern":  "strobe_fast",   # rapid on/off
        "tone":         "siren",         # rising oscillating tone
        "led_r": 255, "led_g": 0,        # red flash
        "repeat": 3,
    },
    AlarmType.HAZARD: {
        "voice_prefix": "WARNING! HAZARD DETECTED! ",
        "led_pattern":  "pulse_slow",    # slow pulse
        "tone":         "warning",       # triple beep
        "led_r": 255, "led_g": 128,      # amber
        "repeat": 2,
    },
    AlarmType.SUSPICIOUS: {
        "voice_prefix": "ALERT! SUSPICIOUS OBJECT! ",
        "led_pattern":  "strobe_medium", # medium strobe
        "tone":         "alert",         # urgent staccato
        "led_r": 255, "led_g": 0,        # red
        "repeat": 3,
    },
    AlarmType.NATURE: {
        "voice_prefix": "",              # no alarm prefix for nature — just excited
        "led_pattern":  "pulse_gentle",  # calm pulse
        "tone":         None,            # no alarm tone for nature
        "led_r": 0, "led_g": 200,        # gentle green
        "repeat": 1,
    },
    AlarmType.CLEAR: {
        "voice_prefix": "All clear. ",
        "led_pattern":  "steady",
        "tone":         "clear",         # double ascending beep
        "led_r": 0, "led_g": 255,
        "repeat": 1,
    },
    AlarmType.NONE: {
        "voice_prefix": "",              # silent — no alarm for narrative missions
        "led_pattern":  None,
        "tone":         None,
        "led_r": 0, "led_g": 0,
        "repeat": 0,
    },
}

_alarm_thread: threading.Thread | None = None
_alarm_stop   = threading.Event()
_alarm_lock   = threading.Lock()

# ── Public API ────────────────────────────────────────────────────────────────

def sound_alarm(alarm_type: str, detail: str = "", blocking: bool = False):
    """
    Trigger a multi-modal alarm.

    alarm_type: AlarmType constant
    detail:     situation description (spoken after prefix)
    blocking:   if True, wait for alarm to finish before returning

    Always non-blocking by default — mission loop keeps running.
    """
    cfg = _ALARM_CONFIG.get(alarm_type, _ALARM_CONFIG[AlarmType.HAZARD])

    # Silent mode — NONE alarm type means no audio/LED, just log
    if alarm_type == AlarmType.NONE:
        log.info(f"ALARM [NONE — silent]: {detail}")
        return

    def _run():
        try:
            _alarm_stop.clear()
            # Run LED + tone in parallel with TTS
            led_thread  = threading.Thread(target=_led_pattern,
                                           args=(cfg["led_pattern"], cfg["repeat"]),
                                           daemon=True)
            tone_thread = threading.Thread(target=_play_tone,
                                           args=(cfg["tone"], cfg["repeat"]),
                                           daemon=True)
            led_thread.start()
            tone_thread.start()

            # TTS announcement
            if detail or cfg["voice_prefix"]:
                from tts import speak
                msg = cfg["voice_prefix"] + (detail or "")
                speak(msg)
                log.info(f"ALARM [{alarm_type.upper()}]: {msg}")

            led_thread.join(timeout=15)
            tone_thread.join(timeout=15)
        except Exception as e:
            log.error(f"Alarm error: {e}")

    with _alarm_lock:
        global _alarm_thread
        if _alarm_thread and _alarm_thread.is_alive():
            _alarm_stop.set()   # stop previous alarm
            _alarm_thread.join(timeout=3)
        _alarm_thread = threading.Thread(target=_run, daemon=True)
        _alarm_thread.start()

    if blocking:
        _alarm_thread.join(timeout=20)


def stop_alarm():
    """Immediately stop any running alarm pattern."""
    _alarm_stop.set()
    try:
        from motors import motors
        motors.lights(0, 0)
    except Exception:
        pass


# ── LED patterns ──────────────────────────────────────────────────────────────

def _led_pattern(pattern: str, repeat: int = 2):
    if pattern is None or repeat == 0:
        return
    try:
        from motors import motors
    except Exception:
        return

    if pattern == "strobe_fast":
        for _ in range(repeat * 8):
            if _alarm_stop.is_set(): break
            motors.lights(255, 255); time.sleep(0.1)
            motors.lights(0, 0);    time.sleep(0.1)

    elif pattern == "strobe_medium":
        for _ in range(repeat * 5):
            if _alarm_stop.is_set(): break
            motors.lights(255, 255); time.sleep(0.2)
            motors.lights(0, 0);    time.sleep(0.15)

    elif pattern == "pulse_slow":
        for _ in range(repeat * 3):
            if _alarm_stop.is_set(): break
            motors.lights(255, 255); time.sleep(0.5)
            motors.lights(0, 0);    time.sleep(0.5)

    elif pattern == "pulse_gentle":
        for _ in range(repeat * 2):
            if _alarm_stop.is_set(): break
            motors.lights(128, 128); time.sleep(0.8)
            motors.lights(0, 0);     time.sleep(0.4)

    elif pattern == "steady":
        motors.lights(128, 255)
        time.sleep(2.0)
        motors.lights(0, 0)

    try:
        motors.lights(0, 0)
    except Exception:
        pass


# ── Audio tones via pygame (no external sound files needed) ──────────────────

def _play_tone(tone_type: str | None, repeat: int = 1):
    """Generate and play tones using pygame. Silently skips if pygame unavailable."""
    if tone_type is None:
        return
    try:
        import pygame
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)

        def _gen_wave(freq: float, duration_ms: int,
                      shape: str = "sine", volume: float = 0.6) -> "pygame.mixer.Sound":
            sample_rate = 22050
            n_samples   = int(sample_rate * duration_ms / 1000)
            buf         = []
            for i in range(n_samples):
                t = i / sample_rate
                if shape == "sine":
                    val = math.sin(2 * math.pi * freq * t)
                elif shape == "square":
                    val = 1.0 if math.sin(2 * math.pi * freq * t) >= 0 else -1.0
                else:
                    val = math.sin(2 * math.pi * freq * t)
                buf.append(int(val * 32767 * volume))
            raw = struct.pack(f"<{n_samples}h", *buf)
            return pygame.mixer.Sound(buffer=raw)

        patterns = {
            "siren": [
                (440, 300, "sine"), (520, 300, "sine"), (620, 300, "sine"),
                (720, 300, "sine"), (620, 200, "sine"), (520, 200, "sine"),
            ],
            "warning": [
                (880, 200, "square"), (0, 80, "sine"),
                (880, 200, "square"), (0, 80, "sine"),
                (880, 400, "square"),
            ],
            "alert": [
                (1000, 100, "square"), (0, 50, "sine"),
                (1000, 100, "square"), (0, 50, "sine"),
                (1200, 150, "square"), (0, 80, "sine"),
                (1200, 150, "square"),
            ],
            "clear": [
                (523, 200, "sine"), (0, 60, "sine"),
                (659, 200, "sine"), (0, 60, "sine"),
                (784, 400, "sine"),
            ],
        }

        seq = patterns.get(tone_type, patterns["warning"])

        for _ in range(repeat):
            if _alarm_stop.is_set():
                break
            for entry in seq:
                if _alarm_stop.is_set():
                    break
                freq, dur_ms, shape = entry
                if freq > 0:
                    sound = _gen_wave(freq, dur_ms, shape)
                    sound.play()
                    time.sleep(dur_ms / 1000.0)
                else:
                    time.sleep(dur_ms / 1000.0)

    except ImportError:
        log.debug("pygame not available — tone skipped")
    except Exception as e:
        log.warning(f"Tone playback error: {e}")
