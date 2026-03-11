"""
ERIC — Voice Pipeline
Always-on hands-free voice interaction. No GUI required.

Pipeline:
  1. silero-vad          : always listening, very low CPU (~50MB RAM)
  2. faster-whisper      : transcribes speech chunks (~166MB RAM distil-small.en)
  3. keyword check       : "hey eric" / "hi eric" in transcript → wake
  4. ECAPA-TDNN          : speaker verification (optional, flag: ASR_VERIFY_SPEAKER)
  5. active session      : keep listening, transcribe each utterance, pass to mission
  6. timeout             : 2-3 min no voice → back to wake word mode

Memory budget on Jetson Orin Nano (after vLLM ~6.8GB):
  silero-vad       ~50MB
  faster-whisper   ~166MB  (distil-small.en default)
  ECAPA-TDNN       ~100MB  (only when ASR_VERIFY_SPEAKER=true)
  Total:           ~316MB  (fits in <600MB remaining)

Usage:
  from voice import init_voice, start_voice_pipeline, stop_voice_pipeline
  init_voice()
  start_voice_pipeline(on_utterance=my_callback)
"""

import logging
import threading
import time
import queue
import numpy as np
from typing import Optional, Callable

log = logging.getLogger("eric.voice")

from config import (
    ASR_MODEL, ASR_DEVICE, ASR_LANGUAGE, ASR_SAMPLE_RATE,
    ASR_ENABLED, ASR_VERIFY_SPEAKER, ASR_SPEAKER_EMBEDDING,
    ASR_WAKE_WORDS, ASR_SESSION_TIMEOUT_SEC,
    ASR_VERIFY_THRESHOLD, ASR_MIC_DEVICE
)

# ─── Constants ────────────────────────────────────────────────────────────────
CHUNK_MS          = 96          # silero-vad chunk size (must be 32/64/96ms at 16kHz)
CHUNK_SAMPLES     = int(ASR_SAMPLE_RATE * CHUNK_MS / 1000)
VAD_THRESHOLD     = 0.45        # silero confidence threshold
SPEECH_PAD_MS     = 400         # pad silence after speech ends before cutting
MAX_RECORD_SEC    = 15          # max single utterance length
SILENCE_CHUNKS    = int(SPEECH_PAD_MS / CHUNK_MS)

# ─── State ────────────────────────────────────────────────────────────────────
_vad_model        = None
_whisper_model    = None
_ecapa_model      = None
_speaker_embedding = None       # enrolled speaker embedding

_pipeline_running = False
_session_active   = False
_session_timer    = None
_pipeline_thread  = None
_audio_queue: queue.Queue = queue.Queue(maxsize=500)

_on_utterance_cb: Optional[Callable[[str, bool], None]] = None
# callback signature: on_utterance(text: str, is_wake: bool)
# is_wake=True on first activation, False on subsequent session utterances

_on_state_change_cb: Optional[Callable[[str], None]] = None
# callback signature: on_state_change(state: str)
# states: "sleeping" | "listening" | "active" | "processing"


# ─── Init ─────────────────────────────────────────────────────────────────────

def init_voice() -> bool:
    """
    Load all voice models. Call once at startup.
    Returns True if minimum pipeline (vad + whisper) is ready.
    """
    ok = _load_vad() and _load_whisper()
    if ASR_VERIFY_SPEAKER:
        _load_ecapa()
        _load_speaker_embedding()
    return ok


def _load_vad() -> bool:
    global _vad_model
    try:
        import torch
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        _vad_model = (model, utils)
        log.info("Voice: silero-vad loaded")
        return True
    except Exception as e:
        log.error(f"Voice: silero-vad load failed — {e}")
        return False


def _load_whisper() -> bool:
    global _whisper_model
    try:
        from faster_whisper import WhisperModel
        log.info(f"Voice: loading faster-whisper {ASR_MODEL} on {ASR_DEVICE}...")
        _whisper_model = WhisperModel(
            ASR_MODEL,
            device=ASR_DEVICE,
            compute_type="int8",
        )
        log.info("Voice: faster-whisper ready")
        return True
    except Exception as e:
        log.error(f"Voice: faster-whisper load failed — {e}")
        return False


def _load_ecapa() -> bool:
    global _ecapa_model
    try:
        from speechbrain.pretrained import SpeakerRecognition
        _ecapa_model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/tmp/ecapa_model",
        )
        log.info("Voice: ECAPA-TDNN speaker verification loaded")
        return True
    except Exception as e:
        log.error(f"Voice: ECAPA-TDNN load failed — {e}")
        _ecapa_model = None
        return False


def _load_speaker_embedding():
    """Load pre-enrolled speaker embedding from file if it exists."""
    global _speaker_embedding
    import pathlib
    path = pathlib.Path(ASR_SPEAKER_EMBEDDING)
    if not path.exists():
        log.warning(f"Voice: no speaker embedding at {path} — run enroll_speaker() first")
        return
    try:
        import torch
        _speaker_embedding = torch.load(path)
        log.info(f"Voice: speaker embedding loaded from {path}")
    except Exception as e:
        log.error(f"Voice: failed to load speaker embedding — {e}")


def voice_available() -> bool:
    return _vad_model is not None and _whisper_model is not None


def speaker_verification_available() -> bool:
    return _ecapa_model is not None and _speaker_embedding is not None


# ─── Speaker enrollment ───────────────────────────────────────────────────────

def enroll_speaker(audio_path: Optional[str] = None, duration_sec: int = 5) -> bool:
    """
    Enroll the authorised speaker. Records duration_sec of audio (or uses
    existing file) and saves the embedding to ASR_SPEAKER_EMBEDDING.
    Call this once from CLI before using speaker verification.

    Usage:
        from voice import init_voice, enroll_speaker
        init_voice()
        enroll_speaker()   # will record 5 seconds from mic
    """
    global _speaker_embedding
    if _ecapa_model is None:
        log.error("Voice: ECAPA not loaded — cannot enroll")
        return False

    import pathlib, torch

    if audio_path is None:
        # Record fresh enrollment audio
        log.info(f"Voice: recording {duration_sec}s enrollment audio — speak now...")
        try:
            import sounddevice as sd
            audio = sd.rec(
                int(duration_sec * ASR_SAMPLE_RATE),
                samplerate=ASR_SAMPLE_RATE,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            audio = audio.flatten()
            tmp = "/tmp/eric_enroll.wav"
            _save_wav(audio, tmp)
            audio_path = tmp
            log.info("Voice: enrollment recording complete")
        except Exception as e:
            log.error(f"Voice: enrollment recording failed — {e}")
            return False

    try:
        embedding = _ecapa_model.encode_batch(
            _ecapa_model.load_audio(audio_path).unsqueeze(0)
        )
        out_path = pathlib.Path(ASR_SPEAKER_EMBEDDING)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embedding, out_path)
        _speaker_embedding = embedding
        log.info(f"Voice: speaker enrolled → {out_path}")
        return True
    except Exception as e:
        log.error(f"Voice: enrollment failed — {e}")
        return False


def _verify_speaker(audio: np.ndarray) -> bool:
    """Returns True if audio matches enrolled speaker above threshold."""
    if not speaker_verification_available():
        return True   # verification unavailable → allow through
    try:
        import torch
        tmp = "/tmp/eric_verify.wav"
        _save_wav(audio, tmp)
        score, prediction = _ecapa_model.verify_files(
            ASR_SPEAKER_EMBEDDING, tmp
        )
        result = float(score) >= ASR_VERIFY_THRESHOLD
        log.debug(f"Voice: speaker score={float(score):.3f} threshold={ASR_VERIFY_THRESHOLD} → {'MATCH' if result else 'REJECT'}")
        return result
    except Exception as e:
        log.warning(f"Voice: speaker verification error — {e}")
        return True   # fail open


# ─── VAD helpers ─────────────────────────────────────────────────────────────

def _vad_confidence(chunk: np.ndarray) -> float:
    """Run silero-vad on a single chunk. Returns confidence 0-1."""
    try:
        import torch
        model, _ = _vad_model
        tensor = torch.from_numpy(chunk).float()
        with torch.no_grad():
            conf = model(tensor, ASR_SAMPLE_RATE).item()
        return conf
    except Exception:
        return 0.0


# ─── Transcription ────────────────────────────────────────────────────────────

def _transcribe(audio: np.ndarray) -> Optional[str]:
    """Transcribe audio array. Returns text or None."""
    if _whisper_model is None:
        return None
    try:
        segments, _ = _whisper_model.transcribe(
            audio,
            language=ASR_LANGUAGE if ASR_LANGUAGE else None,
            beam_size=1,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
        )
        text = " ".join(s.text for s in segments).strip()
        return text if text else None
    except Exception as e:
        log.error(f"Voice: transcription error — {e}")
        return None


# ─── Wake word check ──────────────────────────────────────────────────────────

def _is_wake_word(text: str) -> bool:
    """Check if transcript contains any configured wake word."""
    lower = text.lower()
    for phrase in ASR_WAKE_WORDS:
        if phrase.lower() in lower:
            return True
    return False


# ─── Session timer ────────────────────────────────────────────────────────────

def _reset_session_timer():
    global _session_timer, _session_active
    if _session_timer is not None:
        _session_timer.cancel()
    _session_timer = threading.Timer(ASR_SESSION_TIMEOUT_SEC, _session_timeout)
    _session_timer.daemon = True
    _session_timer.start()


def _session_timeout():
    global _session_active
    if _session_active:
        log.info(f"Voice: session timeout after {ASR_SESSION_TIMEOUT_SEC}s — returning to wake word mode")
        _session_active = False
        _notify_state("sleeping")
        try:
            from tts import speak
            speak("Going to sleep. Say 'Hey Eric' to wake me.")
        except Exception:
            pass


# ─── Audio capture thread ─────────────────────────────────────────────────────

def _audio_capture_thread():
    """Capture audio from default mic into queue. Runs as daemon thread."""
    try:
        import sounddevice as sd

        def _callback(indata, frames, time_info, status):
            if status:
                log.debug(f"Voice audio status: {status}")
            try:
                _audio_queue.put_nowait(indata.copy().flatten())
            except queue.Full:
                pass   # drop oldest implicitly — queue is big enough

        with sd.InputStream(
            samplerate=ASR_SAMPLE_RATE,
            device=ASR_MIC_DEVICE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            callback=_callback,
        ):
            log.info("Voice: audio capture started")
            while _pipeline_running:
                time.sleep(0.05)

    except Exception as e:
        log.error(f"Voice: audio capture failed — {e}")


# ─── Main pipeline loop ───────────────────────────────────────────────────────

def _pipeline_loop():
    """
    Main voice pipeline. Runs as daemon thread.
    States: sleeping (wake word only) ↔ active (full session)
    """
    global _session_active

    speech_buffer   = []
    silence_count   = 0
    in_speech       = False

    log.info("Voice: pipeline started — waiting for wake word")
    _notify_state("sleeping")

    while _pipeline_running:
        try:
            chunk = _audio_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        conf = _vad_confidence(chunk)
        is_speech = conf >= VAD_THRESHOLD

        if is_speech:
            silence_count = 0
            if not in_speech:
                in_speech = True
                speech_buffer = []
                log.debug("Voice: speech start detected")
            speech_buffer.append(chunk)

            # Safety cap — don't buffer forever
            if len(speech_buffer) * CHUNK_MS / 1000 >= MAX_RECORD_SEC:
                log.debug("Voice: max record length reached — forcing transcribe")
                _process_utterance(speech_buffer, force=True)
                speech_buffer = []
                in_speech = False

        else:
            if in_speech:
                silence_count += 1
                speech_buffer.append(chunk)   # pad with silence
                if silence_count >= SILENCE_CHUNKS:
                    # End of utterance
                    in_speech = False
                    silence_count = 0
                    _process_utterance(speech_buffer)
                    speech_buffer = []


def _process_utterance(chunks: list, force: bool = False):
    """Transcribe and route an utterance."""
    global _session_active

    audio = np.concatenate(chunks).flatten()

    # Too short — skip
    if len(audio) < ASR_SAMPLE_RATE * 0.4:
        return

    _notify_state("processing")

    text = _transcribe(audio)
    if not text:
        _notify_state("active" if _session_active else "sleeping")
        return

    log.info(f"Voice: heard → {text!r} (session={'active' if _session_active else 'sleeping'})")

    if _session_active:
        # Active session — pass everything through
        # Still check for explicit sleep command
        lower = text.lower()
        if any(w in lower for w in ["goodbye eric", "bye eric", "sleep eric", "go to sleep"]):
            _session_active = False
            if _session_timer:
                _session_timer.cancel()
            _notify_state("sleeping")
            log.info("Voice: sleep command received")
            try:
                from tts import speak
                speak("Goodbye. Going to sleep.")
            except Exception:
                pass
            return

        _reset_session_timer()
        _notify_state("active")
        if _on_utterance_cb:
            _on_utterance_cb(text, False)

    else:
        # Sleeping — check for wake word
        if _is_wake_word(text):
            # Speaker verification if enabled
            if ASR_VERIFY_SPEAKER:
                if not _verify_speaker(audio):
                    log.info("Voice: wake word detected but speaker not verified — ignoring")
                    _notify_state("sleeping")
                    return

            _session_active = True
            _reset_session_timer()
            _notify_state("listening")
            log.info("Voice: wake word confirmed — session active")
            try:
                from tts import speak
                speak("Yes?")
            except Exception:
                pass
            if _on_utterance_cb:
                _on_utterance_cb(text, True)
        else:
            _notify_state("sleeping")


# ─── Public API ───────────────────────────────────────────────────────────────

def start_voice_pipeline(
    on_utterance: Optional[Callable[[str, bool], None]] = None,
    on_state_change: Optional[Callable[[str], None]] = None,
):
    """
    Start the always-on voice pipeline in background threads.
    on_utterance(text, is_wake) — called with each transcribed utterance
    on_state_change(state)      — called on state transitions
    """
    global _pipeline_running, _pipeline_thread
    global _on_utterance_cb, _on_state_change_cb

    if not voice_available():
        log.error("Voice: models not loaded — call init_voice() first")
        return False

    if _pipeline_running:
        log.warning("Voice: pipeline already running")
        return True

    _on_utterance_cb   = on_utterance
    _on_state_change_cb = on_state_change
    _pipeline_running  = True

    threading.Thread(target=_audio_capture_thread, daemon=True, name="voice-capture").start()
    threading.Thread(target=_pipeline_loop,        daemon=True, name="voice-pipeline").start()

    log.info(f"Voice: pipeline running — wake words: {ASR_WAKE_WORDS}")
    log.info(f"Voice: speaker verification: {'ON' if ASR_VERIFY_SPEAKER else 'OFF'}")
    return True


def stop_voice_pipeline():
    """Stop the voice pipeline."""
    global _pipeline_running, _session_active
    _pipeline_running = False
    _session_active   = False
    if _session_timer:
        _session_timer.cancel()
    log.info("Voice: pipeline stopped")


def force_wake():
    """Programmatically activate session without wake word — useful for testing."""
    global _session_active
    _session_active = True
    _reset_session_timer()
    _notify_state("active")
    log.info("Voice: session force-activated")


def is_session_active() -> bool:
    return _session_active


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _notify_state(state: str):
    if _on_state_change_cb:
        try:
            _on_state_change_cb(state)
        except Exception:
            pass


def _save_wav(audio: np.ndarray, path: str):
    """Save float32 audio array as 16kHz mono WAV."""
    import wave, struct
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(ASR_SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
