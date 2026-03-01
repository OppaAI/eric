"""
ERIC — Robot Teleoperation Controller (Gradio 6.6)
Single file: web UI + motor control + camera + PS3 gamepad.

Usage:
    uv run teleop.py

Open http://aurora:TELEOP_PORT in a browser (default: http://aurora:8889)

PS3 Controller:
  Left stick        → drive robot continuously (tank mix, hold = keep moving)
  Right stick       → pan-tilt camera continuously (hold = keep moving)
  D-pad             → fine movement steps (one step per press, then stops)
  L3 (L-stick btn)  → stop motors
  R3 (R-stick btn)  → centre camera
  L1 / R1           → spin left / spin right (continuous while held)
  L2 / R2           → cycle speed mode (one step per press)
  Cross (×)         → take photo  (saved to PHOTO_DIR)
  Circle (○)        → start/stop video recording  (saved to VIDEO_DIR)
  Square (□)        → toggle base LED
  Triangle (△)      → toggle head LED
  Start             → toggle stream + fullscreen
  Select            → switch camera

Keyboard:
  W/↑ fwd   S/↓ back   A/← left   D/→ right
  Q spin←   E spin→    SPACE stop
  1=slow  2=normal  3=fast
"""

import json
import logging
import os
import threading
import time
from datetime import datetime

import cv2
import gradio as gr
from dotenv import load_dotenv
import numpy as np
from PIL import Image as PILImage

from motors import motors

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("eric.teleop")

# ── Load .env config ──────────────────────────────────────────────────────────
load_dotenv()

SPEED_CYCLE = ["slow", "normal", "fast"]
SPEEDS      = {"slow": 0.2, "normal": 0.4, "fast": 0.6}

# Fine D-pad step speed (fraction of normal speed — smaller = more precise)
DPAD_SPEED_FACTOR = 0.7   # D-pad moves at 70% of selected speed

PAN_CENTRE  =  0
TILT_CENTRE =  0
PAN_STEP    =  5   # smaller step for fine D-pad pan-tilt control
TILT_STEP   =  5

# D-pad drive: short burst duration in seconds (then auto-stop)
DPAD_BURST_MS = 220   # ms of movement per D-pad press

_pantilt_pos  = {"pan": PAN_CENTRE, "tilt": TILT_CENTRE}
_pantilt_lock = threading.Lock()

# ── Directories from .env ─────────────────────────────────────────────────────
_PHOTO_DIR = os.path.expanduser(os.getenv("PHOTO_DIR", "~/photos"))
_VIDEO_DIR = os.path.expanduser(os.getenv("VIDEO_DIR", "~/videos"))
os.makedirs(_PHOTO_DIR, exist_ok=True)
os.makedirs(_VIDEO_DIR, exist_ok=True)
log.info("Photo dir : %s", _PHOTO_DIR)
log.info("Video dir : %s", _VIDEO_DIR)

# ── Gradio server config from .env ────────────────────────────────────────────
# Use TELEOP_PORT / TELEOP_HOST — different from GRADIO_PORT used by other apps
_TELEOP_PORT = int(os.getenv("TELEOP_PORT", "8889"))
_TELEOP_HOST = os.getenv("TELEOP_HOST", "0.0.0.0")
log.info("Teleop server: %s:%d", _TELEOP_HOST, _TELEOP_PORT)

# ── Camera ────────────────────────────────────────────────────────────────────
_cap           = None
_cap_lock      = threading.Lock()
_recorder      = None
_recording     = False
_recorder_lock = threading.Lock()
_stream_active = False          # stream off by default
_blank_frame   = PILImage.fromarray(np.zeros((240, 320, 3), dtype=np.uint8))

_CAM_PANTILT   = f"/dev/video{os.getenv('CAMERA_PANTILT', '0')}"
_CAM_WEBCAM    = f"/dev/video{os.getenv('CAMERA_WEBCAM',  '2')}"
CAMERA_DEVICES = [_CAM_PANTILT, _CAM_WEBCAM]
_cam_device    = _CAM_PANTILT

# Stream resolution — lower = faster FPS, less lag
STREAM_W, STREAM_H = 320, 240
STREAM_QUALITY     = 60        # JPEG quality 1-100

def _get_cap():
    global _cap
    if _cap is not None and _cap.isOpened():
        return _cap
    cap = cv2.VideoCapture(_cam_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  STREAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ret, _ = cap.read()
    if ret:
        _cap = cap
        log.info("Camera opened: %s", _cam_device)
    else:
        cap.release()
        log.warning("Camera %s not available", _cam_device)
        _cap = None
    return _cap


def switch_camera() -> tuple:
    """Toggle between camera devices and restart the capture."""
    global _cap, _cam_device
    with _cap_lock:
        if _cap is not None:
            _cap.release()
            _cap = None
        idx = CAMERA_DEVICES.index(_cam_device) if _cam_device in CAMERA_DEVICES else 0
        _cam_device = CAMERA_DEVICES[(idx + 1) % len(CAMERA_DEVICES)]
        log.info("Switched camera to %s", _cam_device)
    is_webcam = (_cam_device == _CAM_WEBCAM)
    label     = "🔭 Pan-Tilt Cam" if is_webcam else "📷 Webcam"
    cam_num   = _cam_device.replace('/dev/video', '')
    status_   = f'<div id="status-box">📷 CAM {cam_num} — {"WEBCAM" if is_webcam else "PAN-TILT"}</div>'
    return status_, gr.update(value=label)

def _compress_frame(frame_bgr):
    """Encode as JPEG at reduced quality, decode back to PIL — cuts bandwidth."""
    import io
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_QUALITY]
    _, buf = cv2.imencode('.jpg', frame_bgr, encode_param)
    return PILImage.open(io.BytesIO(buf.tobytes()))

def _read_frame():
    if not _stream_active:
        return _blank_frame
    with _cap_lock:
        cap = _get_cap()
        if cap is None:
            return _blank_frame
        ret, frame = cap.read()
    if not ret or frame is None:
        return _blank_frame
    if _cam_device == _CAM_WEBCAM:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = cv2.resize(frame, (STREAM_W, STREAM_H), interpolation=cv2.INTER_LINEAR)
    with _recorder_lock:
        if _recording and _recorder is not None:
            _recorder.write(frame)
    return _compress_frame(frame)

def toggle_stream() -> str:
    global _stream_active
    _stream_active = not _stream_active
    if _stream_active:
        log.info("Stream started")
        return '<div id="status-box">📷 &nbsp;STREAM ON</div>'
    else:
        log.info("Stream stopped")
        return '<div id="status-box">⏹ &nbsp;STREAM OFF</div>'

def take_photo() -> str:
    with _cap_lock:
        cap = _get_cap()
        if cap is None:
            return "⚠ Camera unavailable"
        ret, frame = cap.read()
    if not ret or frame is None:
        return "⚠ Failed to capture"
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(_PHOTO_DIR, f"photo_{ts}.jpg")
    cv2.imwrite(path, frame)
    log.info("Photo saved: %s", path)
    return f"📷 {path}"

def toggle_recording() -> str:
    """
    Record frames to a temp .avi (XVID — reliable on all ARM OpenCV builds),
    then on stop, remux to a proper .mp4 container via ffmpeg stream-copy
    (no re-encode, instant, produces a fully playable MP4).
    """
    global _recorder, _recording, _recording_tmp_path, _recording_final_path
    with _recorder_lock:
        if _recording:
            _recording = False
            if _recorder:
                _recorder.release()
                _recorder = None
            tmp   = _recording_tmp_path
            final = _recording_final_path
            def _remux():
                try:
                    import subprocess
                    result = subprocess.run(
                        ["ffmpeg", "-y", "-i", tmp,
                         "-c:v", "copy", "-movflags", "+faststart", final],
                        capture_output=True, timeout=60
                    )
                    if result.returncode == 0:
                        os.remove(tmp)
                        log.info("Remuxed to %s", final)
                    else:
                        log.warning("ffmpeg remux failed: %s", result.stderr.decode())
                except FileNotFoundError:
                    log.warning("ffmpeg not found, keeping raw avi: %s", tmp)
                except Exception as e:
                    log.error("remux error: %s", e)
            threading.Thread(target=_remux, daemon=True).start()
            return "⏹ Saved → " + os.path.basename(final)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _recording_tmp_path   = os.path.join(_VIDEO_DIR, f".tmp_{ts}.avi")
        _recording_final_path = os.path.join(_VIDEO_DIR, f"video_{ts}.mp4")
        _recorder = cv2.VideoWriter(
            _recording_tmp_path,
            cv2.VideoWriter_fourcc(*'XVID'),
            10.0, (640, 480)
        )
        if not _recorder.isOpened():
            _recorder = None
            return "⚠ VideoWriter failed to open"
        _recording = True
        log.info("Recording started: %s", _recording_final_path)
        return "🎥 REC → " + os.path.basename(_recording_final_path)

# Track temp/final paths for the remux step
_recording_tmp_path   = ""
_recording_final_path = ""


# ── Battery ──────────────────────────────────────────────────────────────────
# The Waveshare MCU broadcasts battery voltage continuously on /dev/ttyTHS1.
#
# UART sharing fix:
#   Previously this module opened /dev/ttyTHS1 directly, competing with
#   motors.py (which also owns the port for motor commands) and odom.py
#   (which read encoder feedback). Three openers on one serial port causes
#   byte theft and corrupted JSON.
#
#   Fix: motors.py owns the port and routes all incoming packets by T-type.
#   Battery packets don't have a consistent T value across firmware versions,
#   so we register a catch-all queue that receives every unmatched packet.
#   We extract the voltage with a regex on the raw JSON string — same logic
#   as before, now reading from the queue instead of the serial port.
#
# 3S LiPo thresholds:
#   HIGH     > 75%   12.19–12.60 V   bright cyan  #00ffff
#   MEDIUM   > 40%   11.67–12.19 V   mid cyan     #00aacc
#   LOW      > 15%   11.07–11.67 V   dark cyan    #006688
#   CRITICAL ≤ 15%   10.50–11.07 V   magenta-red  #ff0066

import re as _re
import queue as _queue

_BATT_V_MAX      = 12.6
_BATT_V_MIN      = 10.5
_battery_voltage = None
_battery_lock    = threading.Lock()
_battery_queue   = _queue.Queue(maxsize=20)


def _battery_poll_loop():
    """
    Drain battery voltage packets from the motors UART router.
    Motors owns /dev/ttyTHS1 — we subscribe to unmatched packets via
    the catch-all queue, then regex-extract the voltage value.
    Falls back gracefully if motors.py is not available (simulation mode).
    """
    global _battery_voltage

    # Register catch-all queue with motors UART router
    try:
        from motors import motors as _m
        _m.subscribe_uart_catchall(_battery_queue)
        log.info("Battery: subscribed to motors UART router (catch-all)")
    except Exception as e:
        log.warning("Battery: could not subscribe to UART router (%s) — voltage unavailable", e)
        return

    while True:
        try:
            data = _battery_queue.get(timeout=2.0)
            # Voltage is embedded in the JSON as a 4-digit integer (e.g. 1185 = 11.85V).
            # Scan all numeric values in the packet to find one in the 9.00-13.50V range.
            raw = json.dumps(data)
            for m in _re.findall(r'(\d+)', raw):
                val = int(m)
                if 900 <= val <= 1350:      # 9.00–13.50 V sanity range
                    with _battery_lock:
                        _battery_voltage = round(val / 100.0, 2)
                    break
        except _queue.Empty:
            pass   # no packet — voltage stays at last known value
        except Exception as e:
            log.debug("Battery poll error: %s", e)
            time.sleep(0.5)

threading.Thread(target=_battery_poll_loop, daemon=True, name="battery").start()


def _battery_level(pct: int) -> tuple[str, str, str]:
    """Return (label, color, glow_color) for a battery percentage."""
    if pct > 75:
        return "HIGH",     "#00ffff", "#00ffff55"   # bright cyan
    elif pct > 40:
        return "MEDIUM",   "#00aacc", "#00aacc55"   # mid cyan
    elif pct > 15:
        return "LOW",      "#006688", "#00668855"   # dark cyan
    else:
        return "CRITICAL", "#ff0066", "#ff006655"   # stop button magenta-red


def get_battery_html() -> str:
    with _battery_lock:
        v = _battery_voltage
    mono = "font-family:'Share Tech Mono',monospace"
    if v is None:
        return (
            f'<div style="{mono};font-size:0.72rem;color:#4a5568;'
            f'background:#0f1215;border:1px solid #1e2530;border-radius:4px;'
            f'padding:6px 10px;letter-spacing:0.08em">'
            f'⚡ BATT &nbsp;<span style="color:#2a3545">-- . - V</span>'
            f'</div>'
        )
    pct              = max(0, min(100, int((v - _BATT_V_MIN) / (_BATT_V_MAX - _BATT_V_MIN) * 100)))
    label, color, glow = _battery_level(pct)
    filled           = round(pct / 10)
    bar_filled       = f'<span style="color:{color}">{"█" * filled}</span>'
    bar_empty        = f'<span style="color:#2a3545">{"░" * (10 - filled)}</span>'
    return (
        f'<div style="{mono};font-size:0.72rem;'
        f'background:#0a0c0e;border:1px solid {color}66;border-radius:4px;'
        f'padding:6px 12px;box-shadow:0 0 12px {glow}, 0 0 24px {color}22;letter-spacing:0.06em;'
        f'display:flex;align-items:center;gap:10px">'
        f'<span style="color:{color};font-size:0.8em;letter-spacing:0.2em;'
        f'text-shadow:0 0 8px {color},0 0 16px {color}88">⚡ {label}</span>'
        f'<span style="color:#c8d6e5;font-size:1.05em;font-weight:bold;'
        f'text-shadow:0 0 6px #c8d6e5aa">{v:.2f}V</span>'
        f'<span style="font-size:0.85em;letter-spacing:0;'
        f'text-shadow:0 0 6px {color}66">{bar_filled}{bar_empty}</span>'
        f'</div>'
    )


# ── Lights ────────────────────────────────────────────────────────────────────
_lights = {"base": False, "head": False}

def toggle_base(state):
    _lights["base"] = state
    motors.lights(base=255 if state else 0, head=255 if _lights["head"] else 0)
    return state

def toggle_head(state):
    _lights["head"] = state
    motors.lights(base=255 if _lights["base"] else 0, head=255 if state else 0)
    return state

def toggle_both(state):
    _lights["base"] = _lights["head"] = state
    motors.lights(base=255 if state else 0, head=255 if state else 0)
    return state, state, state


# ── Pan-tilt ──────────────────────────────────────────────────────────────────
def _apply_pantilt():
    try:
        with _pantilt_lock:
            pan  = _pantilt_pos["pan"]
            tilt = _pantilt_pos["tilt"]
        log.info(f"pantilt → pan={pan} tilt={tilt}")
        motors.pantilt(pan, tilt)
    except Exception as e:
        log.warning(f"pan_tilt FAILED: {e}", exc_info=True)

def pantilt_move(direction: str) -> str:
    with _pantilt_lock:
        if   direction == "up":     _pantilt_pos["tilt"] = min( 90, _pantilt_pos["tilt"] + TILT_STEP)
        elif direction == "down":   _pantilt_pos["tilt"] = max(-90, _pantilt_pos["tilt"] - TILT_STEP)
        elif direction == "left":   _pantilt_pos["pan"]  = max(-90, _pantilt_pos["pan"]  - PAN_STEP)
        elif direction == "right":  _pantilt_pos["pan"]  = min( 90, _pantilt_pos["pan"]  + PAN_STEP)
        elif direction == "centre":
            _pantilt_pos["pan"]  = PAN_CENTRE
            _pantilt_pos["tilt"] = TILT_CENTRE
        pan  = _pantilt_pos["pan"]
        tilt = _pantilt_pos["tilt"]
    _apply_pantilt()
    return f'<div id="status-box">🎥 PAN {pan:+d}° TILT {tilt:+d}°</div>'

def pantilt_centre() -> str:
    return pantilt_move("centre")

def pantilt_raw(pan: float, tilt: float) -> str:
    """
    Continuous pan-tilt from R-stick: accumulate position based on stick deflection.
    pan/tilt here are velocity values (-90..90 range mapped from stick -1..1).
    We clamp the accumulated position to ±90°.
    """
    with _pantilt_lock:
        _pantilt_pos["pan"]  = max(-90, min(90, int(pan)))
        _pantilt_pos["tilt"] = max(-90, min(90, int(tilt)))
        p = _pantilt_pos["pan"]
        t = _pantilt_pos["tilt"]
    _apply_pantilt()
    return f'<div id="status-box">🎥 PAN {p:+d}° TILT {t:+d}°</div>'

def pantilt_step_raw(dpan: float, dtilt: float) -> str:
    """
    Incremental pan-tilt step — used by D-pad buttons for fine control.
    dpan / dtilt are deltas (e.g. ±PAN_STEP).
    """
    with _pantilt_lock:
        _pantilt_pos["pan"]  = max(-90, min(90, _pantilt_pos["pan"]  + int(dpan)))
        _pantilt_pos["tilt"] = max(-90, min(90, _pantilt_pos["tilt"] + int(dtilt)))
        p = _pantilt_pos["pan"]
        t = _pantilt_pos["tilt"]
    _apply_pantilt()
    return f'<div id="status-box">🎥 PAN {p:+d}° TILT {t:+d}°</div>'


# ── Motors ────────────────────────────────────────────────────────────────────
def send_motor(cmd: str, speed: str) -> str:
    s = SPEEDS.get(speed, 0.4)
    cmds = {
        "forward":    (-s,           -s          ),
        "backward":   ( s,            s          ),
        "left":       ( s * 0.5,     -s * 0.5    ),
        "right":      (-s * 0.5,      s * 0.5    ),
        "spin_left":  ( s,           -s          ),
        "spin_right": (-s,            s          ),
        "stop":       ( 0.0,          0.0        ),
    }
    L, R = cmds.get(cmd, (0.0, 0.0))
    log.info(f"send_motor cmd={cmd!r} speed={speed!r} → L={L} R={R}")
    icons = {"forward":"▲","backward":"▼","left":"◀","right":"▶",
             "spin_left":"↺","spin_right":"↻","stop":"■"}
    try:
        motors._send(L, R)
        if cmd == "stop":
            return '<div id="status-box">■ &nbsp;STOPPED</div>'
        return f'<div id="status-box">{icons.get(cmd,"?")} &nbsp;{cmd.upper().replace("_"," ")} &nbsp; L={L:+.3f} &nbsp; R={R:+.3f}</div>'
    except Exception as e:
        log.error(f"motors._send FAILED: {e}", exc_info=True)
        return f'<div id="status-box">⚠ {e}</div>'

def send_motor_raw(L: float, R: float) -> str:
    """Continuous drive from analogue stick — no auto-stop."""
    try:
        motors._send(L, R)
        if L == 0 and R == 0:
            return '<div id="status-box">■ &nbsp;STOPPED</div>'
        return f'<div id="status-box">🕹 L={L:+.3f} R={R:+.3f}</div>'
    except Exception as e:
        return f'<div id="status-box">⚠ {e}</div>'

def send_motor_dpad(cmd: str, speed: str) -> str:
    """
    Fine D-pad movement: run motors briefly then stop.
    This gives a short controlled burst for precise repositioning.
    The burst timing is handled on the JS side; Python just drives then stops.
    """
    s = SPEEDS.get(speed, 0.4) * DPAD_SPEED_FACTOR
    cmds = {
        "forward":  (-s,      -s     ),
        "backward": ( s,       s     ),
        "left":     ( s*0.5,  -s*0.5 ),
        "right":    (-s*0.5,   s*0.5 ),
    }
    L, R = cmds.get(cmd, (0.0, 0.0))
    icons = {"forward":"▲","backward":"▼","left":"◀","right":"▶"}
    try:
        motors._send(L, R)
        # Auto-stop after burst (non-blocking — JS will also send stop)
        def _stop_after():
            time.sleep(DPAD_BURST_MS / 1000.0)
            motors._send(0.0, 0.0)
        threading.Thread(target=_stop_after, daemon=True).start()
        return (f'<div id="status-box">{icons.get(cmd,"?")} &nbsp;FINE '
                f'{cmd.upper()} &nbsp; L={L:+.3f} &nbsp; R={R:+.3f}</div>')
    except Exception as e:
        log.error(f"send_motor_dpad FAILED: {e}", exc_info=True)
        return f'<div id="status-box">⚠ {e}</div>'


# ── CSS ───────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@300;400;600;700;900&display=swap');

:root {
    --bg:     #050508;
    --panel:  #161622;
    --border: #00ffff22;
    --accent: #00ffff;
    --magenta:#ff00ff;
    --green:  #39ff14;
    --danger: #ff0066;
    --warn:   #ff9900;
    --text:   #ccffff;
    --dim:    #7a9aaa;
}

*, *::before, *::after { box-sizing: border-box; }

/* Rendered but invisible — lets JS click it via _clickId */
.hidden-offscreen { position:fixed !important; left:-9999px !important; top:-9999px !important;
    width:1px !important; height:1px !important; overflow:hidden !important;
    pointer-events:none !important; opacity:0 !important; }

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'Share Tech Mono', 'Courier New', monospace !important;
    margin: 0 !important; padding: 0 !important;
    background-image:
        linear-gradient(rgba(0,255,255,0.06) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,255,0.06) 1px, transparent 1px),
        radial-gradient(ellipse at 50% 0%, rgba(0,255,255,0.06) 0%, transparent 50%);
    background-size: 40px 40px, 40px 40px, 100% 100%;
}
/* CRT scanline */
body::after {
    content:''; position:fixed; inset:0; pointer-events:none; z-index:9998;
    background: repeating-linear-gradient(0deg,
        rgba(0,255,255,0.012) 0px, rgba(0,255,255,0.012) 1px,
        transparent 1px, transparent 3px);
}
footer, .built-with { display:none !important; }

#eric-header {
    font-family:'Share Tech Mono',monospace;
    display:flex; align-items:center; justify-content:space-between;
    padding:10px 20px 8px;
    border-bottom:1px solid #ff00ff44;
    margin-bottom:0;
}
#eric-title {
    color:#00ffff; letter-spacing:0.3em; font-size:1.3rem;
    text-shadow: 0 0 10px #00ffff, 0 0 20px #00ffffaa, 0 0 40px #00ffff66;
}

/* Battery bar — sits right below the header title bar, full-width strip */
#batt-html {
    padding: 0 20px 8px !important;
    margin: 0 !important;
    display: flex !important;
    justify-content: flex-end !important;
    border-bottom: 1px solid #ff00ff22 !important;
    margin-bottom: 8px !important;
    background: transparent !important;
}
#batt-html > div { width: auto !important; }

#status-box {
    font-family:'Share Tech Mono',monospace !important; font-size:0.85rem !important;
    color:var(--accent) !important; background:var(--panel) !important;
    border:1px solid var(--border) !important; border-radius:4px !important;
    padding:10px 16px !important; text-align:center; min-height:42px !important;
    margin-bottom:10px;
    text-shadow: 0 0 8px #00ffff55;
}

#camera-feed img { display:block !important; width:100% !important; border-radius:0 !important; }

/* ── Stream / cam switch buttons ── */
#btn-stream button {
    font-family:'Share Tech Mono',monospace !important; font-size:0.75rem !important;
    letter-spacing:0.12em !important; width:100% !important; margin-top:6px !important;
    background:#002a1a !important; border:1px solid #39ff1488 !important;
    color:#39ff14 !important; border-radius:3px !important; transition:all 0.15s !important;
    box-shadow: 0 0 10px rgba(57,255,20,0.18), inset 0 0 8px rgba(57,255,20,0.06) !important;
    text-shadow: 0 0 10px #39ff1499 !important;
}
#btn-stream button:hover { box-shadow:0 0 20px rgba(57,255,20,0.35), 0 0 40px rgba(57,255,20,0.12) !important; border-color:#39ff14 !important; text-shadow:0 0 12px #39ff14 !important; }

#btn-cam-switch button {
    font-family:'Share Tech Mono',monospace !important; font-size:0.75rem !important;
    letter-spacing:0.08em !important; width:100% !important; margin-top:6px !important;
    background:#001a2a !important; border:1px solid #00ffff88 !important;
    color:#00eeff !important; border-radius:3px !important; transition:all 0.15s !important;
    box-shadow: 0 0 10px rgba(0,255,255,0.15), inset 0 0 8px rgba(0,255,255,0.05) !important;
    text-shadow: 0 0 10px #00ffff99 !important;
}
#btn-cam-switch button:hover { box-shadow:0 0 20px rgba(0,255,255,0.3), 0 0 40px rgba(0,255,255,0.1) !important; border-color:#00ffff !important; text-shadow:0 0 12px #00ffff !important; }

/* ── Drive buttons ── */
.dpad-btn {
    font-family:'Share Tech Mono',monospace !important; font-size:1.4rem !important;
    background:#0a0a0f !important; border:1px solid #00ffff44 !important;
    border-radius:2px !important; color:#4a99bb !important;
    width:76px !important; height:76px !important; min-width:76px !important;
    transition:all 0.1s !important;
    box-shadow: 0 0 6px rgba(0,255,255,0.08), inset 0 0 8px rgba(0,255,255,0.03) !important;
    text-shadow: 0 0 8px #00ffff44 !important;
}
.dpad-btn:hover { background:#001a1a !important; border-color:#00ffff !important;
    color:#00ffff !important; box-shadow:0 0 20px rgba(0,255,255,0.35), 0 0 40px rgba(0,255,255,0.15), inset 0 0 12px rgba(0,255,255,0.08) !important;
    text-shadow: 0 0 12px #00ffffff !important; }
.dpad-btn:active { background:#003333 !important; transform:scale(0.95) !important; box-shadow:0 0 30px rgba(0,255,255,0.5) !important; }

.stop-btn {
    font-family:'Share Tech Mono',monospace !important; font-size:1.1rem !important;
    background:#0a0005 !important; border:1px solid #ff006644 !important;
    border-radius:2px !important; color:#cc2255 !important;
    width:76px !important; height:76px !important; min-width:76px !important;
    transition:all 0.1s !important;
    box-shadow: 0 0 6px rgba(255,0,102,0.1), inset 0 0 8px rgba(255,0,102,0.04) !important;
    text-shadow: 0 0 8px #ff006644 !important;
}
.stop-btn:hover { border-color:#ff0066 !important; color:#ff0066 !important;
    box-shadow:0 0 20px rgba(255,0,102,0.4), 0 0 40px rgba(255,0,102,0.15), inset 0 0 12px rgba(255,0,102,0.1) !important;
    text-shadow: 0 0 12px #ff0066ff !important; }
.stop-btn:active { transform:scale(0.95) !important; box-shadow:0 0 30px rgba(255,0,102,0.6) !important; }

.spin-btn {
    font-family:'Share Tech Mono',monospace !important; font-size:1.3rem !important;
    background:#0a0a0f !important; border:1px solid #ff00ff44 !important;
    border-radius:2px !important; color:#bb22bb !important;
    width:76px !important; height:76px !important; min-width:76px !important;
    transition:all 0.1s !important;
    box-shadow: 0 0 6px rgba(255,0,255,0.08), inset 0 0 8px rgba(255,0,255,0.03) !important;
    text-shadow: 0 0 8px #ff00ff44 !important;
}
.spin-btn:hover { border-color:#ff00ff !important; color:#ff00ff !important;
    box-shadow:0 0 20px rgba(255,0,255,0.35), 0 0 40px rgba(255,0,255,0.15), inset 0 0 12px rgba(255,0,255,0.08) !important;
    text-shadow: 0 0 12px #ff00ffff !important; }
.spin-btn:active { transform:scale(0.95) !important; box-shadow:0 0 30px rgba(255,0,255,0.5) !important; }

/* ── Pan-tilt buttons — amber/gold ── */
.pt-btn {
    font-family:'Share Tech Mono',monospace !important; font-size:1.2rem !important;
    background:#0a0a0f !important; border:1px solid #ff990044 !important;
    border-radius:2px !important; color:#cc8800 !important;
    width:60px !important; height:60px !important; min-width:60px !important;
    transition:all 0.1s !important;
    box-shadow: 0 0 5px rgba(255,153,0,0.08), inset 0 0 6px rgba(255,153,0,0.03) !important;
    text-shadow: 0 0 8px #ff990044 !important;
}
.pt-btn:hover { border-color:#ff9900 !important; color:#ff9900 !important;
    box-shadow:0 0 18px rgba(255,153,0,0.35), 0 0 36px rgba(255,153,0,0.12), inset 0 0 10px rgba(255,153,0,0.08) !important;
    text-shadow: 0 0 10px #ff9900ff !important; }
.pt-btn:active { transform:scale(0.95) !important; box-shadow:0 0 28px rgba(255,153,0,0.5) !important; }

.pt-centre-btn {
    font-family:'Share Tech Mono',monospace !important; font-size:0.7rem !important;
    background:#0a0a0f !important; border:1px solid #ff990044 !important;
    border-radius:2px !important; color:#cc9922 !important;
    width:60px !important; height:60px !important; min-width:60px !important;
    letter-spacing:0.08em !important; transition:all 0.1s !important;
    box-shadow: 0 0 5px rgba(255,153,0,0.06) !important;
}
.pt-centre-btn:hover { border-color:#ff9900 !important; color:#ff9900 !important;
    box-shadow:0 0 15px rgba(255,153,0,0.3) !important; text-shadow:0 0 8px #ff9900 !important; }
.pt-centre-btn:active { transform:scale(0.95) !important; }

/* ── Speed radio ── */
.speed-radio span.svelte-1gfkfd6 { display:none !important; }
.speed-radio .wrap { gap:6px !important; justify-content:center !important; }
.speed-radio label {
    font-family:'Share Tech Mono',monospace !important; font-size:0.72rem !important;
    letter-spacing:0.15em !important; text-transform:uppercase !important;
    color:#aaddee !important; background:var(--panel) !important;
    border:1px solid #00ffff44 !important; border-radius:2px !important;
    padding:6px 16px !important; cursor:pointer !important;
    transition: all 0.15s !important;
}
.speed-radio label:hover { border-color:var(--accent) !important; color:var(--text) !important;
    box-shadow: 0 0 10px rgba(0,255,255,0.2) !important; text-shadow: 0 0 6px #00ffff88 !important; }

/* ── Photo & Record buttons ── */
#btn-photo button {
    font-family:'Share Tech Mono',monospace !important; font-size:0.75rem !important;
    background:#001a2a !important; border:1px solid #00ffff44 !important;
    color:#00ffff !important; border-radius:3px !important; transition:all 0.15s !important;
    box-shadow: 0 0 6px rgba(0,255,255,0.1) !important; text-shadow: 0 0 8px #00ffff55 !important;
    width:100% !important; margin-bottom:4px !important;
}
#btn-photo button:hover { box-shadow:0 0 16px rgba(0,255,255,0.3), 0 0 32px rgba(0,255,255,0.1) !important;
    border-color:#00ffff !important; text-shadow:0 0 10px #00ffff !important; }

#circle-btn button {
    font-family:'Share Tech Mono',monospace !important; font-size:0.75rem !important;
    background:#2a0010 !important; border:1px solid #ff006644 !important;
    color:#ff0066 !important; border-radius:3px !important; transition:all 0.15s !important;
    box-shadow: 0 0 6px rgba(255,0,102,0.12) !important; text-shadow: 0 0 8px #ff006655 !important;
    width:100% !important;
}
#circle-btn button:hover { box-shadow:0 0 16px rgba(255,0,102,0.3), 0 0 32px rgba(255,0,102,0.1) !important;
    border-color:#ff0066 !important; text-shadow:0 0 10px #ff0066 !important; }

/* ── Side panels ── */
.side-panel {
    background:var(--panel); border:1px solid #ff00ff22;
    border-radius:4px; padding:14px;
    box-shadow: 0 0 10px rgba(255,0,255,0.04), inset 0 0 20px rgba(0,0,0,0.3);
}
.side-panel label {
    font-family:'Share Tech Mono',monospace !important; font-size:0.78rem !important;
    letter-spacing:0.1em !important; text-transform:uppercase !important;
    color:#aaddee !important; transition:color 0.15s !important;
}
.side-panel input:checked ~ label { color:#ff00ff !important; text-shadow:0 0 8px #ff00ff66 !important; }
.section-header {
    font-family:'Share Tech Mono',monospace; font-size:0.82rem; letter-spacing:0.18em;
    color:#ff44ff; text-transform:uppercase; padding-bottom:8px; text-shadow:0 0 10px #ff00ff99;
    border-bottom:1px solid #ff00ff44; margin-bottom:10px;
}
#gamepad-status {
    font-family:'Share Tech Mono',monospace; font-size:0.68rem; color:var(--dim);
    text-align:center; padding:6px 8px; border:1px solid var(--border);
    border-radius:3px; background:var(--panel); margin-top:8px; transition:all 0.3s;
}
#gamepad-status.connected { color:var(--green); border-color:var(--green); box-shadow:0 0 8px rgba(57,255,20,0.2); }
#key-hints {
    font-family:'Share Tech Mono',monospace; font-size:0.6rem; color:var(--dim);
    text-align:center; letter-spacing:0.06em; padding:8px 0 2px;
    border-top:1px solid var(--border); margin-top:6px;
}

/* ════════════════════════════════
   MOBILE
   ════════════════════════════════ */
@media (max-width: 640px) {
    html, body { overflow:hidden !important; height:100% !important; width:100% !important; position:fixed !important; }
    .gradio-container { padding:0 !important; margin:0 !important; height:100dvh !important; width:100vw !important; overflow:hidden !important; }
    #eric-header { display:none !important; }
    .mobile-cam { position:fixed !important; inset:0 !important; width:100vw !important; height:100dvh !important; z-index:1 !important; padding:0 !important; margin:0 !important; }
    .mobile-controls, .mobile-side { display:none !important; }
    .mobile-cam #camera-feed,
    .mobile-cam #camera-feed > div,
    .mobile-cam #camera-feed > div > div,
    .mobile-cam #camera-feed > div > div > div {
        width:100vw !important; height:100dvh !important;
        padding:0 !important; margin:0 !important;
    }
    .mobile-cam #camera-feed img {
        width:100vw !important; height:100dvh !important;
        object-fit:cover !important; display:block !important;
    }

    #mobile-hud {
        position:fixed; top:0; left:0; right:0; z-index:100;
        display:flex !important; align-items:center; justify-content:space-between;
        padding:10px 14px;
        background:linear-gradient(to bottom, rgba(5,5,8,0.85) 0%, transparent 100%);
        font-family:'Share Tech Mono',monospace; font-size:0.72rem; color:#00ffff; letter-spacing:0.15em;
    }
    #mobile-hud-title { font-size:0.85rem; letter-spacing:0.3em; text-shadow:0 0 10px #00ffff88; }

    #mobile-speed {
        position:fixed; top:44px; right:14px; z-index:100;
        font-family:'Share Tech Mono',monospace; font-size:0.65rem; color:#ff9900;
        background:rgba(10,12,14,0.8); border:1px solid #ff990055; border-radius:4px;
        padding:3px 8px; backdrop-filter:blur(4px);
    }
    #mobile-rec {
        position:fixed; top:44px; left:14px; z-index:100;
        font-family:'Share Tech Mono',monospace; font-size:0.65rem; color:var(--danger);
        background:rgba(10,12,14,0.8); border:1px solid var(--danger); border-radius:4px;
        padding:3px 8px; backdrop-filter:blur(4px); display:none;
    }
    #mobile-rec.active { display:block; }
    #mobile-cmd {
        position:fixed; bottom:58px; left:50%; transform:translateX(-50%); z-index:100;
        font-family:'Share Tech Mono',monospace; font-size:0.75rem; color:var(--accent);
        background:rgba(10,12,14,0.8); border:1px solid var(--border); border-radius:6px;
        padding:5px 14px; backdrop-filter:blur(4px); white-space:nowrap;
    }
    #mobile-gp {
        position:fixed; bottom:20px; left:50%; transform:translateX(-50%); z-index:100;
        font-family:'Share Tech Mono',monospace; font-size:0.7rem; color:var(--dim);
        background:rgba(10,12,14,0.75); border:1px solid var(--border); border-radius:20px;
        padding:6px 16px; backdrop-filter:blur(4px); white-space:nowrap;
    }
    #mobile-gp.connected { color:var(--green); border-color:var(--green); box-shadow:0 0 12px rgba(57,255,20,0.2); }
    #gp-debug {
        position:fixed; bottom:90px; left:8px; right:8px; z-index:200;
        font-family:'Share Tech Mono',monospace; font-size:0.58rem; color:#ff9900;
        background:rgba(0,0,0,0.85); border:1px solid #ff990055;
        border-radius:4px; padding:4px 8px; display:none;
        word-break:break-all; line-height:1.4;
    }
    #mobile-toggle {
        position:fixed; bottom:16px; right:16px; z-index:200;
        width:44px; height:44px; background:rgba(10,12,14,0.85); border:1px solid var(--border);
        border-radius:8px; display:flex !important; align-items:center; justify-content:center;
        font-size:1.2rem; cursor:pointer; backdrop-filter:blur(4px); color:var(--dim);
    }
    #mobile-toggle:active { transform:scale(0.9); color:var(--accent); border-color:var(--accent); }
    #mobile-fullscreen {
        position:fixed; top:10px; right:14px; z-index:200;
        width:36px; height:36px; background:rgba(10,12,14,0.8); border:1px solid var(--border);
        border-radius:6px; display:flex !important; align-items:center; justify-content:center;
        font-size:1.1rem; cursor:pointer; backdrop-filter:blur(4px); color:var(--dim);
    }
    #mobile-fullscreen:active { transform:scale(0.9); color:var(--accent); border-color:var(--accent); }
    #mobile-drawer {
        position:fixed; bottom:0; left:0; right:0; z-index:150;
        background:rgba(10,12,14,0.96); border-top:1px solid var(--border);
        backdrop-filter:blur(8px); padding:14px 14px 28px;
        transform:translateY(100%); transition:transform 0.3s cubic-bezier(0.4,0,0.2,1);
        border-radius:16px 16px 0 0; overflow-y:auto; max-height:80dvh;
    }
    #mobile-drawer.open { transform:translateY(0); }
    #mobile-drawer .dpad-btn, #mobile-drawer .stop-btn, #mobile-drawer .spin-btn {
        width:58px !important; height:58px !important; min-width:58px !important; font-size:1.2rem !important;
    }
    #mobile-drawer .pt-btn, #mobile-drawer .pt-centre-btn {
        width:52px !important; height:52px !important; min-width:52px !important;
    }
}
@media (min-width: 641px) {
    #mobile-hud,#mobile-gp,#mobile-cmd,#mobile-speed,#mobile-rec,
    #mobile-toggle,#mobile-fullscreen,#mobile-drawer { display:none !important; }
}
"""

# ── JavaScript ────────────────────────────────────────────────────────────────
CONTROL_JS = f"""
() => {{
    if (window._ericBound) return;
    window._ericBound = true;

    // ── Constants mirrored from Python ───────────────────────────────────────
    const DPAD_BURST_MS = {DPAD_BURST_MS};   // ms for D-pad motor burst before auto-stop

    // ── Speed cycle ───────────────────────────────────────────────────────────
    const SPEED_CYCLE = ['slow', 'normal', 'fast'];
    let speedIdx = 1;  // start at 'normal'

    function getSpeed() {{ return SPEED_CYCLE[speedIdx]; }}

    function applySpeed() {{
        document.querySelectorAll('.speed-radio input[type=radio]').forEach(r => {{
            if (r.value === SPEED_CYCLE[speedIdx]) r.click();
        }});
        _syncSpeedBtn();
        const el = document.getElementById('mobile-speed');
        if (el) el.textContent = SPEED_CYCLE[speedIdx].toUpperCase();
        setStatus('⚡ SPEED: ' + SPEED_CYCLE[speedIdx].toUpperCase());
    }}

    function speedLeft()  {{ speedIdx = ((speedIdx - 1) + 3) % 3; applySpeed(); }}
    function speedRight() {{ speedIdx = (speedIdx + 1) % 3;       applySpeed(); }}

    // ── Status helpers ────────────────────────────────────────────────────────
    function setStatus(msg) {{
        const el = document.getElementById('status-box');
        if (el) el.innerHTML = msg;
        const mel = document.getElementById('mobile-cmd');
        if (mel) mel.textContent = msg.replace(/<[^>]+>/g, '');
    }}

    function setGpStatus(msg, connected) {{
        ['gamepad-status','mobile-gp'].forEach(id => {{
            const el = document.getElementById(id);
            if (el) {{ el.textContent = msg; el.className = connected ? 'connected' : ''; }}
        }});
    }}

    // ── Click Gradio buttons by elem_id ───────────────────────────────────────
    function _clickId(id) {{
        const wrapper = document.getElementById(id);
        if (!wrapper) {{ console.warn('btn not found:', id); return; }}
        const btn = wrapper.tagName === 'BUTTON' ? wrapper : wrapper.querySelector('button');
        if (btn) btn.click();
        else console.warn('no <button> inside:', id);
    }}

    const CMD_ID = {{
        forward:    'btn-forward',
        backward:   'btn-backward',
        left:       'btn-left',
        right:      'btn-right',
        spin_left:  'btn-spin-left',
        spin_right: 'btn-spin-right',
        stop:       'btn-stop',
        // Fine D-pad movement buttons (separate from main drive)
        dp_forward: 'btn-dp-forward',
        dp_backward:'btn-dp-backward',
        dp_left:    'btn-dp-left',
        dp_right:   'btn-dp-right',
        photo:      'btn-photo',
        rec:        'circle-btn',
        stream:     'btn-stream',
        pt_up:      'btn-pt-up',
        pt_down:    'btn-pt-down',
        pt_left:    'btn-pt-left',
        pt_right:   'btn-pt-right',
        pt_centre:  'btn-pt-centre',
    }};

    function clickCmd(cmd) {{
        const id = CMD_ID[cmd];
        if (id) _clickId(id);
        else console.warn('no id for cmd:', cmd);
    }}

    // ── Continuous L-stick drive ──────────────────────────────────────────────
    // L-stick sends raw motor values continuously at SEND_HZ.
    // We directly fire the hidden raw-drive buttons which call send_motor_raw().
    // No auto-stop — when stick returns to centre (both ~0) we send stop.
    let _rawDriveBusy  = false;
    let _rawPTBusy     = false;
    let _lastL = 0, _lastR = 0;
    let _lastPan = 0, _lastTilt = 0;

    // Accumulated pan/tilt position for continuous R-stick control
    let _ptAccPan  = 0;
    let _ptAccTilt = 0;

    // Gradio textbox setter — must use the native input value setter to
    // trigger Svelte's reactive binding, then fire both 'input' and 'change'.
    function _gradioSet(elemId, val) {{
        const box = document.getElementById(elemId);
        if (!box) {{ console.warn('gradioSet: not found', elemId); return false; }}
        const inp = box.querySelector('textarea') || box.querySelector('input');
        if (!inp) {{ console.warn('gradioSet: no input in', elemId); return false; }}
        // Use the native setter so Svelte picks up the change
        const nativeSetter = Object.getOwnPropertyDescriptor(
            inp.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype,
            'value'
        ).set;
        nativeSetter.call(inp, val);
        inp.dispatchEvent(new Event('input',  {{bubbles: true}}));
        inp.dispatchEvent(new Event('change', {{bubbles: true}}));
        return true;
    }}

    function _setRawDrive(L, R) {{
        _gradioSet('raw-drive-input', L.toFixed(3) + ':' + R.toFixed(3));
    }}

    function _setRawPT(pan, tilt) {{
        _gradioSet('raw-pt-input', pan.toFixed(1) + ':' + tilt.toFixed(1));
    }}

    // ── D-pad fine movement: press fires a timed burst then auto-stops ────────
    let _dpStopTimer = null;

    function sendDpad(cmd) {{
        // Cancel any pending stop from previous press
        if (_dpStopTimer) {{ clearTimeout(_dpStopTimer); _dpStopTimer = null; }}
        clickCmd('dp_' + cmd);   // fires send_motor_dpad() on Python side
        setStatus(({{forward:'▲',backward:'▼',left:'◀',right:'▶'}}[cmd]||'?')
                  + ' FINE ' + cmd.toUpperCase());
        // JS-side stop after burst — Python also sends stop via threading
        _dpStopTimer = setTimeout(() => {{
            clickCmd('stop');
            _dpStopTimer = null;
        }}, DPAD_BURST_MS + 20);
    }}

    // ── Keyboard ──────────────────────────────────────────────────────────────
    const KEY_MAP = {{
        'ArrowUp':'forward','w':'forward','W':'forward',
        'ArrowDown':'backward','s':'backward','S':'backward',
        'ArrowLeft':'left','a':'left','A':'left',
        'ArrowRight':'right','d':'right','D':'right',
        'q':'spin_left','Q':'spin_left',
        'e':'spin_right','E':'spin_right',
        ' ':'stop',
    }};
    let held = new Set();
    document.addEventListener('keydown', e => {{
        if (['INPUT','TEXTAREA'].includes(e.target.tagName)) return;
        if (e.key === '1') {{ speedIdx = 0; applySpeed(); return; }}
        if (e.key === '2') {{ speedIdx = 1; applySpeed(); return; }}
        if (e.key === '3') {{ speedIdx = 2; applySpeed(); return; }}
        const cmd = KEY_MAP[e.key];
        if (!cmd) return;
        e.preventDefault();
        if (held.has(e.key)) return;
        held.add(e.key);
        clickCmd(cmd);
        setStatus(({{forward:'▲',backward:'▼',left:'◀',right:'▶',
                     spin_left:'↺',spin_right:'↻',stop:'■'}}[cmd]||'?')
                  + ' ' + cmd.toUpperCase().replace('_',' '));
    }});
    document.addEventListener('keyup', e => {{
        const cmd = KEY_MAP[e.key];
        if (!cmd) return;
        e.preventDefault();
        held.delete(e.key);
        if (cmd !== 'stop') clickCmd('stop');
    }});

    // ── Gamepad ───────────────────────────────────────────────────────────────
    const DEADZONE    = 0.15;
    const SEND_HZ     = 20;         // tick rate
    const PT_RATE     = 1.5;        // degrees per tick per unit of stick deflection
    const SPEED_SCALE = 0.6;        // max analogue stick motor speed

    let gpIndex = null, gpLoop = null;
    let btnPrev = [];
    let _l2fired = false, _r2fired = false;

    function dz(v) {{ return Math.abs(v) < DEADZONE ? 0 : v; }}

    function tankMix(ly, lx) {{
        const fwd = -dz(ly), turn = -dz(lx);
        let L = fwd - turn, R = fwd + turn;
        const mx = Math.max(Math.abs(L), Math.abs(R), 1);
        return [parseFloat((-(L/mx)*SPEED_SCALE).toFixed(3)),
                parseFloat((-(R/mx)*SPEED_SCALE).toFixed(3))];
    }}

    function updateGpDebug(gp) {{
        const el = document.getElementById('gp-debug');
        if (!el || !gp) return;
        const pressed = gp.buttons.map((b,i) => b.pressed ? i : null).filter(i => i !== null);
        const axes    = Array.from(gp.axes).map((a,i) => `A${{i}}:${{a.toFixed(2)}}`).join(' ');
        el.textContent = `BTN:${{pressed.join(',')||'-'}}  ${{axes}}`;
        el.style.display = 'block';
    }}

    function gpTick() {{
        if (gpIndex === null) return;
        const gp = navigator.getGamepads()[gpIndex];
        if (!gp) return;

        updateGpDebug(gp);

        const btns = gp.buttons.map((b, i) => {{
            if (typeof b === 'number') return b > 0.5;
            if (i === 6 || i === 7) return (b.value ?? (b.pressed ? 1 : 0)) > 0.5;
            return b.pressed;
        }});
        const edge = i => !!(btns[i]) && !(btnPrev[i]);

        // ── PS3 button mapping ───────────────────────────────────────────────
        // ×=0  ○=1  □=2  △=3  L1=4  R1=5  L2=6  R2=7
        // Select=8  Start=9  L3=10  R3=11
        // D-Up=12  D-Down=13  D-Left=14  D-Right=15

        if (edge(0))  doPhoto();
        if (edge(1))  toggleRec();
        if (edge(2))  toggleBaseLED();
        if (edge(3))  toggleHeadLED();

        if (edge(10)) {{ clickCmd('stop');    setStatus('■ STOPPED'); }}
        if (edge(11)) {{ centreCamera(); }}

        if (edge(9))  toggleStreamBtn();
        if (edge(8))  switchCameraBtn();

        // L1/R1 — spin continuously while held, stop on release.
        // Send raw motor values directly (same path as L-stick) to avoid
        // hammering Gradio button clicks on every 50ms tick.
        const spinSpeed = SPEED_SCALE * 0.8;
        if (!!(btns[4])) {{
            // L1 = spin left: left motor forward, right motor back
            _setRawDrive(spinSpeed, -spinSpeed);
            setStatus('↺ SPIN LEFT');
        }} else if (!!(btns[5])) {{
            // R1 = spin right: left motor back, right motor forward
            _setRawDrive(-spinSpeed, spinSpeed);
            setStatus('↻ SPIN RIGHT');
        }} else if (!!(btnPrev[4]) || !!(btnPrev[5])) {{
            // Released — stop
            _setRawDrive(0, 0);
        }}

        // L2/R2 — cycle speed (latched per press, indices 6 and 7)
        // These are analog triggers on PS3, use value threshold
        const l2val = (typeof gp.buttons[6] === 'object') ? (gp.buttons[6].value||0) : (gp.buttons[6]||0);
        const r2val = (typeof gp.buttons[7] === 'object') ? (gp.buttons[7].value||0) : (gp.buttons[7]||0);
        const l2on = l2val > 0.5, r2on = r2val > 0.5;
        if (!l2on) _l2fired = false;
        if (!r2on) _r2fired = false;
        if (l2on && !_l2fired) {{ _l2fired = true; speedLeft();  }}
        if (r2on && !_r2fired) {{ _r2fired = true; speedRight(); }}

        // ── D-pad → fine movement steps (one burst per press) ────────────────
        // Button d-pad
        const dpU = !!(btns[12]) || (gp.axes.length > 7 && gp.axes[7] < -0.5);
        const dpD = !!(btns[13]) || (gp.axes.length > 7 && gp.axes[7] >  0.5);
        const dpL = !!(btns[14]) || (gp.axes.length > 6 && gp.axes[6] < -0.5);
        const dpR = !!(btns[15]) || (gp.axes.length > 6 && gp.axes[6] >  0.5);

        const prevDpU = !!(btnPrev[12]) || (gp.axes.length > 7 && (btnPrev._a7||0) < -0.5);
        const prevDpD = !!(btnPrev[13]) || (gp.axes.length > 7 && (btnPrev._a7||0) >  0.5);
        const prevDpL = !!(btnPrev[14]) || (gp.axes.length > 6 && (btnPrev._a6||0) < -0.5);
        const prevDpR = !!(btnPrev[15]) || (gp.axes.length > 6 && (btnPrev._a6||0) >  0.5);

        // Fire on rising edge only — one burst per press, no hold-repeat
        if (dpU && !prevDpU) sendDpad('forward');
        if (dpD && !prevDpD) sendDpad('backward');
        if (dpL && !prevDpL) sendDpad('left');
        if (dpR && !prevDpR) sendDpad('right');

        btnPrev = [...btns];
        btnPrev._a6 = gp.axes[6] || 0;
        btnPrev._a7 = gp.axes[7] || 0;

        // ── Left stick → continuous drive (raw motor values) ─────────────────
        // Send at every tick while stick is deflected; send stop when returning to centre.
        const [L, R] = tankMix(gp.axes[1]??0, gp.axes[0]??0);
        if (Math.abs(L - _lastL) > 0.01 || Math.abs(R - _lastR) > 0.01) {{
            if (!_rawDriveBusy) {{
                _rawDriveBusy = true;
                _setRawDrive(L, R);
                _lastL = L; _lastR = R;
                setTimeout(() => {{ _rawDriveBusy = false; }}, 40);
            }}
        }}

        // ── Right stick → continuous pan-tilt ────────────────────────────────
        // Accumulate position each tick based on stick deflection rate.
        // Clamped to ±90°. When stick returns to centre, position holds.
        const rsX2 = dz(gp.axes[2]??0), rsY3 = dz(gp.axes[3]??0);
        const rsX3 = dz(gp.axes[3]??0), rsY4 = dz(gp.axes[4]??0);
        const useAlt = (Math.abs(rsX3) + Math.abs(rsY4)) > (Math.abs(rsX2) + Math.abs(rsY3));
        const rsPan  = useAlt ? rsX3 : rsX2;
        const rsTilt = useAlt ? rsY4 : rsY3;

        if (Math.abs(rsPan) > 0 || Math.abs(rsTilt) > 0) {{
            _ptAccPan  = Math.max(-90, Math.min(90, _ptAccPan  + rsPan  * PT_RATE));
            _ptAccTilt = Math.max(-90, Math.min(90, _ptAccTilt - rsTilt * PT_RATE));
            if (!_rawPTBusy) {{
                _rawPTBusy = true;
                _setRawPT(_ptAccPan, _ptAccTilt);
                setTimeout(() => {{ _rawPTBusy = false; }}, 40);
            }}
        }}
    }}

    function connectGamepad(gp) {{
        gpIndex = gp.index;
        const name = gp.id.length > 36 ? gp.id.substring(0,36)+'…' : gp.id;
        setGpStatus('🎮 ' + name, true);
        if (gpLoop) clearInterval(gpLoop);
        gpLoop = setInterval(gpTick, 1000/SEND_HZ);
    }}

    window.addEventListener('gamepadconnected', e => connectGamepad(e.gamepad));
    window.addEventListener('gamepaddisconnected', e => {{
        if (e.gamepad.index === gpIndex) {{
            gpIndex = null; clearInterval(gpLoop); gpLoop = null;
            _lastL = 0; _lastR = 0;
            setGpStatus('🎮 no gamepad', false);
            clickCmd('stop');
        }}
    }});

    // ── Battery mirror to mobile HUD ──────────────────────────────────────────
    const battObserver = new MutationObserver(() => {{
        const src = document.querySelector('#batt-html');
        const dst = document.getElementById('mobile-hud-batt');
        if (src && dst) dst.innerHTML = src.innerHTML;
    }});
    setTimeout(() => {{
        const el = document.querySelector('#batt-html');
        if (el) battObserver.observe(el, {{childList: true, subtree: true, characterData: true}});
    }}, 2000);

    // ── Poll for gamepad (mobile Chrome/Firefox need polling) ─────────────────
    function startGamepadPoller() {{
        setInterval(() => {{
            if (gpIndex !== null) return;
            const pads = navigator.getGamepads ? navigator.getGamepads() : [];
            for (let i = 0; i < pads.length; i++) {{
                if (pads[i] && pads[i].connected) {{
                    gpIndex = i;
                    const name = pads[i].id.length > 36 ? pads[i].id.substring(0,36)+'…' : pads[i].id;
                    setGpStatus('🎮 ' + name, true);
                    if (!gpLoop) gpLoop = setInterval(gpTick, 1000/SEND_HZ);
                    break;
                }}
            }}
        }}, 500);
    }}
    startGamepadPoller();

    // ── Action helpers ────────────────────────────────────────────────────────
    let _recording = false;

    function doPhoto() {{
        clickCmd('photo');
        setStatus('📷 capturing...');
    }}

    function toggleRec() {{
        clickCmd('rec');
        _recording = !_recording;
        setStatus(_recording ? '🎥 RECORDING' : '⏹ REC stopped');
        const rec = document.getElementById('mobile-rec');
        if (rec) rec.classList.toggle('active', _recording);
    }}

    function _toggleCheckbox(elemId) {{
        const wrapper = document.getElementById(elemId);
        if (!wrapper) return;
        const inp = wrapper.querySelector('input[type=checkbox]');
        if (inp) inp.click();
    }}

    let _baseLED = false, _headLED = false;

    function toggleBaseLED() {{
        _baseLED = !_baseLED;
        _toggleCheckbox('chk-light-base');
        setStatus('💡 BASE LED ' + (_baseLED ? 'ON' : 'OFF'));
    }}

    function toggleHeadLED() {{
        _headLED = !_headLED;
        _toggleCheckbox('chk-light-head');
        setStatus('💡 HEAD LED ' + (_headLED ? 'ON' : 'OFF'));
    }}

    function centreCamera() {{
        // Reset accumulated position
        _ptAccPan = 0; _ptAccTilt = 0;
        clickCmd('pt_centre');
        setStatus('🎥 PAN-TILT CENTRED');
    }}

    function toggleStreamBtn() {{
        const wrapper = document.getElementById('btn-stream');
        if (!wrapper) return;
        const btn = wrapper.tagName === 'BUTTON' ? wrapper : wrapper.querySelector('button');
        if (!btn) return;
        const turningOn = btn.textContent.trim().includes('Start');
        btn.click();
        if (turningOn) {{ enterFullscreen(); }} else {{ exitFullscreen(); }}
    }}

    function enterFullscreen() {{
        const el = document.documentElement;
        if (!document.fullscreenElement && !document.webkitFullscreenElement) {{
            const req = el.requestFullscreen || el.webkitRequestFullscreen || el.mozRequestFullScreen;
            if (req) req.call(el).catch(() => {{}});
        }}
    }}

    function exitFullscreen() {{
        if (document.fullscreenElement || document.webkitFullscreenElement) {{
            const exit = document.exitFullscreen || document.webkitExitFullscreen;
            if (exit) exit.call(document).catch(() => {{}});
        }}
    }}

    function switchCameraBtn() {{
        _clickId('btn-switch-cam');
        setStatus('📷 switching camera...');
    }}

    // ── Mobile drawer ─────────────────────────────────────────────────────────
    window._drawerOpen = false;
    window.toggleDrawer = function() {{
        window._drawerOpen = !window._drawerOpen;
        const drawer = document.getElementById('mobile-drawer');
        const btn    = document.getElementById('mobile-toggle');
        if (drawer) drawer.classList.toggle('open', window._drawerOpen);
        if (btn)    btn.textContent = window._drawerOpen ? '✕' : '⊞';
    }};

    if (window.innerWidth <= 640) setTimeout(injectMobileUI, 1500);

    function injectMobileUI() {{
        if (document.getElementById('mobile-hud')) return;

        const hud = document.createElement('div');
        hud.id = 'mobile-hud';
        hud.innerHTML = '<span id="mobile-hud-title">ERIC</span><span id="mobile-hud-batt"></span>';
        document.body.appendChild(hud);

        const spd = document.createElement('div');
        spd.id = 'mobile-speed'; spd.textContent = 'NORMAL';
        document.body.appendChild(spd);

        const rec = document.createElement('div');
        rec.id = 'mobile-rec'; rec.textContent = '● REC';
        document.body.appendChild(rec);

        const cmd = document.createElement('div');
        cmd.id = 'mobile-cmd'; cmd.textContent = '■ STANDBY';
        document.body.appendChild(cmd);

        const gps = document.createElement('div');
        gps.id = 'mobile-gp'; gps.textContent = '🎮 no gamepad';
        document.body.appendChild(gps);

        const dbg = document.createElement('div');
        dbg.id = 'gp-debug';
        document.body.appendChild(dbg);

        const fsBtn = document.createElement('div');
        fsBtn.id = 'mobile-fullscreen';
        fsBtn.innerHTML = '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/></svg>';
        fsBtn.title = 'Fullscreen';
        fsBtn.addEventListener('click', () => {{
            const el = document.documentElement;
            if (!document.fullscreenElement && !document.webkitFullscreenElement) {{
                const req = el.requestFullscreen || el.webkitRequestFullscreen || el.mozRequestFullScreen;
                if (req) req.call(el);
                if (screen.orientation && screen.orientation.lock)
                    screen.orientation.lock('landscape').catch(()=>{{}});
            }} else {{
                const exit = document.exitFullscreen || document.webkitExitFullscreen;
                if (exit) exit.call(document);
            }}
        }});
        document.body.appendChild(fsBtn);

        const toggle = document.createElement('div');
        toggle.id = 'mobile-toggle';
        toggle.textContent = '⊞';
        toggle.addEventListener('click', window.toggleDrawer);
        document.body.appendChild(toggle);

        const drawer = document.createElement('div');
        drawer.id = 'mobile-drawer';
        setTimeout(() => {{
            const ctrl = document.querySelector('.mobile-controls');
            const side = document.querySelector('.mobile-side');
            if (ctrl) drawer.appendChild(ctrl.cloneNode(true));
            if (side) drawer.appendChild(side.cloneNode(true));
            drawer.querySelectorAll('button').forEach(btn => {{
                btn.addEventListener('click', () => {{
                    const orig = Array.from(document.querySelectorAll(
                        '.mobile-controls button, .mobile-side button'
                    )).find(b => b.textContent.trim() === btn.textContent.trim());
                    if (orig) orig.click();
                }});
            }});
        }}, 500);
        document.body.appendChild(drawer);

        const battSrc = document.querySelector('#batt-html');
        if (battSrc) {{
            const battDst = document.getElementById('mobile-hud-batt');
            if (battDst) battDst.innerHTML = battSrc.innerHTML;
        }}
    }}

    window.addEventListener('resize', () => {{ if (window.innerWidth <= 640) injectMobileUI(); }});
}}
"""

# ── Command handler ───────────────────────────────────────────────────────────
def handle_raw_drive(raw: str) -> str:
    """Handle continuous L-stick raw drive: 'L:R' float values."""
    if not raw or ':' not in raw:
        return ""
    try:
        parts = raw.split(':')
        return send_motor_raw(float(parts[0]), float(parts[1]))
    except Exception as e:
        return f'<div id="status-box">⚠ {e}</div>'

def handle_raw_pt(raw: str) -> str:
    """Handle continuous R-stick raw pan-tilt: 'pan:tilt' float values."""
    if not raw or ':' not in raw:
        return ""
    try:
        parts = raw.split(':')
        return pantilt_raw(float(parts[0]), float(parts[1]))
    except Exception as e:
        return f'<div id="status-box">⚠ {e}</div>'


# ── Build UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(title="ERIC — Robot Teleoperation", css=CUSTOM_CSS) as demo:

    gr.HTML('''
    <div id="eric-header">
        <div id="eric-title">ERIC <span style="color:#6a7a8a">/ CONTROL</span></div>
    </div>''')
    batt_html = gr.HTML(get_battery_html(), elem_id="batt-html")

    with gr.Row(equal_height=False):

        # ── Camera ────────────────────────────────────────────────────────────
        with gr.Column(scale=3, min_width=300, elem_classes=["mobile-cam"]):
            cam_feed = gr.Image(
                label="Camera 0", elem_id="camera-feed",
                show_label=False, height=480, type="pil",
                container=False, value=None,
            )
            with gr.Row(equal_height=True):
                stream_btn = gr.Button("▶ Start Stream", elem_id="btn-stream",
                                       variant="primary", size="sm")
                cam_sw_btn = gr.Button("📷 Webcam", elem_id="btn-cam-switch",
                                       variant="secondary", size="sm")

        # ── Controls ──────────────────────────────────────────────────────────
        with gr.Column(scale=2, min_width=280, elem_classes=["mobile-controls"]):
            status = gr.HTML('<div id="status-box">■ &nbsp;STANDBY</div>')

            gr.HTML('<div class="section-header">🕹 L-STICK · Continuous &nbsp;&nbsp;&nbsp; D-PAD · Fine Step</div>')
            with gr.Row(equal_height=True):
                gr.HTML('<div style="width:76px"></div>')
                fwd_btn = gr.Button("▲", elem_classes=["dpad-btn"], elem_id="btn-forward")
                gr.HTML('<div style="width:76px"></div>')
            with gr.Row(equal_height=True):
                left_btn  = gr.Button("◀", elem_classes=["dpad-btn"], elem_id="btn-left")
                stop_btn  = gr.Button("■", elem_classes=["stop-btn"], elem_id="btn-stop")
                right_btn = gr.Button("▶", elem_classes=["dpad-btn"], elem_id="btn-right")
            with gr.Row(equal_height=True):
                sl_btn  = gr.Button("↺", elem_classes=["spin-btn"], elem_id="btn-spin-left")
                bwd_btn = gr.Button("▼", elem_classes=["dpad-btn"], elem_id="btn-backward")
                sr_btn  = gr.Button("↻", elem_classes=["spin-btn"], elem_id="btn-spin-right")

            # Hidden fine D-pad buttons — use CSS offscreen (NOT visible=False which removes from DOM)
            with gr.Row(elem_classes=["hidden-offscreen"]):
                dp_fwd_btn  = gr.Button("dp▲", elem_id="btn-dp-forward")
                dp_bwd_btn  = gr.Button("dp▼", elem_id="btn-dp-backward")
                dp_left_btn = gr.Button("dp◀", elem_id="btn-dp-left")
                dp_rgt_btn  = gr.Button("dp▶", elem_id="btn-dp-right")

            speed = gr.Radio(
                choices=["slow", "normal", "fast"], value="normal",
                label="Speed", elem_classes=["speed-radio"],
            )

            gr.HTML('<div class="section-header" style="margin-top:12px">🎥 R-STICK · Continuous &nbsp;&nbsp;&nbsp; BUTTONS · Fine Step</div>')
            with gr.Row(equal_height=True):
                gr.HTML('<div style="width:60px"></div>')
                pt_up_btn = gr.Button("▲", elem_classes=["pt-btn"], elem_id="btn-pt-up")
                gr.HTML('<div style="width:60px"></div>')
            with gr.Row(equal_height=True):
                pt_left_btn   = gr.Button("◀",   elem_classes=["pt-btn"], elem_id="btn-pt-left")
                pt_centre_btn = gr.Button("CTR", elem_classes=["pt-centre-btn"], elem_id="btn-pt-centre")
                pt_right_btn  = gr.Button("▶",   elem_classes=["pt-btn"], elem_id="btn-pt-right")
            with gr.Row(equal_height=True):
                gr.HTML('<div style="width:60px"></div>')
                pt_down_btn = gr.Button("▼", elem_classes=["pt-btn"], elem_id="btn-pt-down")
                gr.HTML('<div style="width:60px"></div>')

            gr.HTML('<div id="gamepad-status">🎮 no gamepad connected</div>')
            gr.HTML(f'''<div id="key-hints">
                W/↑ fwd · S/↓ back · A/← left · D/→ right · SPACE stop<br>
                L-STICK=continuous drive · R-STICK=continuous cam · D-PAD=fine steps (burst {DPAD_BURST_MS}ms)<br>
                ×=photo · ○=rec · □=base LED · △=head LED<br>
                L1=spin← · R1=spin→ · L2/R2=speed◀▶ · L3=stop · R3=centre
            </div>''')

        # ── Side: Lights + Battery + Capture ─────────────────────────────────
        with gr.Column(scale=1, min_width=170, elem_classes=["mobile-side"]):
            with gr.Group(elem_classes=["side-panel"]):
                gr.HTML('<div class="section-header">💡 &nbsp; Lights</div>')
                light_both = gr.Checkbox(label="All Lights", value=False, elem_id="chk-light-both")
                light_base = gr.Checkbox(label="Base LED",   value=False, elem_id="chk-light-base")
                light_head = gr.Checkbox(label="Head LED",   value=False, elem_id="chk-light-head")

            gr.HTML('<div style="height:10px"></div>')

            with gr.Group(elem_classes=["side-panel"]):
                gr.HTML('<div class="section-header">📡 &nbsp; Telemetry</div>')
                gr.HTML('<div id="telemetry-info" style="font-family:\'Share Tech Mono\',monospace;font-size:0.68rem;color:#5a7a8a;padding:4px 0">voltage in header ↑</div>')

            gr.HTML('<div style="height:10px"></div>')

            with gr.Group(elem_classes=["side-panel"]):
                gr.HTML('<div class="section-header">📷 &nbsp; Capture</div>')
                photo_btn  = gr.Button("📷 Photo",  variant="secondary", elem_id="btn-photo")
                circle_btn = gr.Button("⏺ Record", variant="stop", elem_id="circle-btn")
                capture_status = gr.HTML(
                    '<div style="font-family:\'Share Tech Mono\',monospace;'
                    'font-size:0.7rem;color:#7a9aaa;padding:4px 0">ready</div>'
                )
                gr.HTML(f'<div style="font-family:\'Share Tech Mono\',monospace;'
                        f'font-size:0.58rem;color:#4a6677;padding:2px 0 0">'
                        f'📁 {_PHOTO_DIR}<br>🎥 {_VIDEO_DIR}</div>')

    # ── Timers ────────────────────────────────────────────────────────────────
    gr.Timer(0.066).tick(fn=_read_frame, outputs=[cam_feed])
    gr.Timer(5.0).tick(fn=get_battery_html, outputs=[batt_html])

    # ── Stream toggle ─────────────────────────────────────────────────────────
    def _toggle_stream_btn():
        msg   = toggle_stream()
        label = "⏹ Stop Stream" if _stream_active else "▶ Start Stream"
        return msg, gr.update(value=label)
    stream_btn.click(fn=_toggle_stream_btn, outputs=[status, stream_btn])

    # ── Continuous drive buttons (keyboard / UI clicks — hold-to-drive) ───────
    for btn, cmd in [
        (fwd_btn,   "forward"),   (bwd_btn,   "backward"),
        (left_btn,  "left"),      (right_btn, "right"),
        (sl_btn,    "spin_left"), (sr_btn,    "spin_right"),
        (stop_btn,  "stop"),
    ]:
        btn.click(fn=lambda s, c=cmd: send_motor(c, s), inputs=[speed], outputs=[status])

    # ── Fine D-pad buttons (triggered by gamepad D-pad) ───────────────────────
    for btn, cmd in [
        (dp_fwd_btn,  "forward"),
        (dp_bwd_btn,  "backward"),
        (dp_left_btn, "left"),
        (dp_rgt_btn,  "right"),
    ]:
        btn.click(fn=lambda s, c=cmd: send_motor_dpad(c, s), inputs=[speed], outputs=[status])

    # ── Pan-tilt step buttons (UI / D-pad pan-tilt on gamepad) ────────────────
    pt_up_btn.click(    fn=lambda: pantilt_move("up"),    outputs=[status])
    pt_down_btn.click(  fn=lambda: pantilt_move("down"),  outputs=[status])
    pt_left_btn.click(  fn=lambda: pantilt_move("left"),  outputs=[status])
    pt_right_btn.click( fn=lambda: pantilt_move("right"), outputs=[status])
    pt_centre_btn.click(fn=pantilt_centre,                outputs=[status])

    # ── Capture ───────────────────────────────────────────────────────────────
    photo_btn.click( fn=take_photo,       outputs=[capture_status])
    circle_btn.click(fn=toggle_recording, outputs=[capture_status])

    # ── Lights ────────────────────────────────────────────────────────────────
    light_both.change(fn=toggle_both, inputs=[light_both], outputs=[light_both, light_base, light_head])
    light_base.change(fn=toggle_base, inputs=[light_base], outputs=[light_base])
    light_head.change(fn=toggle_head, inputs=[light_head], outputs=[light_head])

    # ── Camera switch ─────────────────────────────────────────────────────────
    cam_sw_btn.click(fn=switch_camera, outputs=[status, cam_sw_btn])
    switch_cam_btn = gr.Button("switch-cam", elem_id="btn-switch-cam", elem_classes=["hidden-offscreen"])
    switch_cam_btn.click(fn=switch_camera, outputs=[status, cam_sw_btn])

    # ── Continuous raw-drive textbox (updated by JS L-stick tick) ────────────
    # Must be in the DOM (not visible=False) so JS getElementById works.
    raw_drive = gr.Textbox(value="", elem_id="raw-drive-input",
                           elem_classes=["hidden-offscreen"], show_label=False)
    raw_drive.change(fn=handle_raw_drive, inputs=[raw_drive], outputs=[status])

    # ── Continuous raw pan-tilt textbox (updated by JS R-stick tick) ─────────
    raw_pt = gr.Textbox(value="", elem_id="raw-pt-input",
                        elem_classes=["hidden-offscreen"], show_label=False)
    raw_pt.change(fn=handle_raw_pt, inputs=[raw_pt], outputs=[status])

    demo.load(fn=None, js=CONTROL_JS)
    demo.load(fn=pantilt_centre, outputs=[status])


if __name__ == "__main__":
    demo.launch(server_name=_TELEOP_HOST, server_port=_TELEOP_PORT)