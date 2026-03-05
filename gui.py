"""
ERIC — Cockpit Dashboard GUI
Cyberpunk mission control for 1680x1050 single-screen operation.
"""

import threading
import logging

import gradio as gr

from config import GRADIO_HOST, GRADIO_PORT, CAMERA_WEBCAM, CAMERA_PANTILT, MISSIONS_DIR, ASR_ENABLED
from cosmos import capture_frame_raw, ask_cosmos
from motors import motors
from tts import speak, init_tts, piper_available
from mission import (
    start_mission, stop_mission, resume_after_interaction,
    handle_character_response, register_ui_callbacks,
    list_missions, get_briefing_from_file, get_mission_metadata,
    get_mission_active, get_mission_state, get_conversation_history,
    _ms as _mission_state,
)

log = logging.getLogger("eric.gui")

# ─── Battery (regex reader — works around Jetson THS1 frame-drop) ────────────
import re as _re, os as _os, time as _time

_BATT_V_MAX      = 12.6
_BATT_V_MIN      = 10.5
_battery_voltage = None
_battery_lock    = threading.Lock()

# ── Serial access lock shared with motors ────────────────────────────────────
# We reuse motors._ser so there's only ONE connection to the MCU.
# A dedicated lock prevents interleaving with motor commands.
_serial_lock = threading.Lock()

def _query_battery_via_motors():
    """
    Send T:105 through the motors serial port and extract voltage from reply.
    Returns voltage in volts, or None on failure.
    Uses the regex tail method from test_bat.py:  (\d+)\}  → val/100.
    """
    try:
        ser = motors._ser
        if ser is None or not ser.is_open:
            return None
        with _serial_lock:
            ser.reset_input_buffer()
            ser.write(b'{"T":105}\n')
            _time.sleep(0.12)          # give MCU time to respond
            raw = b""
            deadline = _time.time() + 0.4
            while _time.time() < deadline:
                n = ser.in_waiting
                if n:
                    raw += ser.read(n)
                else:
                    _time.sleep(0.01)
        text    = raw.replace(b'\x00', b'').decode('utf-8', errors='replace')
        compact = ''.join(text.split())
        for m in _re.findall(r'(\d+)\}', compact):
            val = int(m)
            if 900 <= val <= 1350:
                return round(val / 100.0, 2)
    except Exception as exc:
        log.debug("battery query error: %s", exc)
    return None


def _battery_poll_loop():
    global _battery_voltage
    # Wait for motors to initialise before first attempt
    _time.sleep(5)
    log.info("GUI battery poller started")
    while True:
        v = _query_battery_via_motors()
        if v is not None:
            with _battery_lock:
                _battery_voltage = v
            log.debug("Battery: %.2f V", v)
        else:
            log.debug("Battery query returned no data")
        _time.sleep(30)   # poll every 30 s — battery changes slowly

threading.Thread(target=_battery_poll_loop, daemon=True, name="gui-battery").start()


def _battery_level_gui(pct):
    if pct > 75:   return "HIGH",     "#00ffff", "#00ffff33"
    elif pct > 40: return "MEDIUM",   "#39ff14", "#39ff1433"
    elif pct > 15: return "LOW",      "#ff9900", "#ff990033"
    else:          return "CRITICAL", "#ff0066", "#ff006633"


def get_battery_html():
    with _battery_lock:
        v = _battery_voltage
    mono = "font-family:'Courier New',monospace"
    if v is None:
        return (
            f'<div style="background:#0a0a0f;border:1px solid #ff00ff22;border-radius:4px;'
            f'padding:5px 8px;{mono};font-size:0.68em;color:#333;letter-spacing:0.08em">'
            f'⚡ BATTERY &nbsp;<span style="color:#1a1a22">-- . - V</span></div>'
        )
    pct              = max(0, min(100, int((v - _BATT_V_MIN) / (_BATT_V_MAX - _BATT_V_MIN) * 100)))
    label, color, glow = _battery_level_gui(pct)
    filled           = round(pct / 10)
    bar = (f'<span style="color:{color}">{"█" * filled}</span>'
           f'<span style="color:#1a1a22">{"░" * (10 - filled)}</span>')
    return (
        f'<div style="background:#0a0a0f;border:1px solid {color}33;border-radius:4px;'
        f'padding:5px 8px;box-shadow:0 0 8px {glow};{mono};font-size:0.68em;letter-spacing:0.06em">'
        f'<span style="color:{color};letter-spacing:0.18em;font-weight:bold">{label}</span>'
        f'&nbsp;&nbsp;'
        f'<span style="color:#ccffff;font-weight:bold;font-size:1.1em">{v:.2f}V</span>'
        f'&nbsp;&nbsp;{bar}&nbsp;&nbsp;'
        f'<span style="color:{color}">{pct}%</span>'
        f'</div>'
    )


# ─── ASR state ────────────────────────────────────────────────────────────────
_asr_recording   = False
_asr_mode        = "MISSION"   # "MISSION" fills briefing box | "COMMS" fills char reply
_asr_last_text   = ""
_asr_lock        = threading.Lock()
_voice_state     = "sleeping"  # sleeping | listening | active | processing

# ─── Shared state ─────────────────────────────────────────────────────────────
_eric_says  = ""
_status     = "IDLE"
_log_text   = ""

_motor_state = {"direction": "stopped", "left": 0.0, "right": 0.0}

_TEMPLATE_MISSION = """\
You are ERIC — Edge Robotics Innovation by Cosmos.
Search and rescue mission. Identify persons, robots, objects of interest.
Approach and interact. NO combat. Avoid obstacles. Talk and gather intel."""

def _set_eric_says(t):
    global _eric_says
    _eric_says = t

def _set_status(t):
    global _status
    _status = str(t).upper()

def _append_log(t):
    global _log_text
    lines = (_log_text + "\n" + str(t)).strip().split("\n")
    _log_text = "\n".join(lines[-60:])

def set_voice_state(state: str):
    """Called by voice pipeline on state transitions."""
    global _voice_state
    _voice_state = state

def get_voice_state_html() -> str:
    icons = {
        "sleeping":    ("💤", "#444",    "SLEEPING"),
        "listening":   ("👂", "#ffff00", "LISTENING"),
        "active":      ("🎙", "#00ff88", "ACTIVE"),
        "processing":  ("⚙",  "#ff9900", "PROCESSING"),
    }
    icon, color, label = icons.get(_voice_state, ("?", "#444", _voice_state.upper()))
    return (
        f'<div style="background:#0a0a0f;border:1px solid {color}44;border-radius:4px;'
        f'padding:5px 8px;font-family:\'Share Tech Mono\',monospace;font-size:0.75em;'
        f'letter-spacing:0.08em">'
        f'<span style="color:{color}">{icon} VOICE: {label}</span>'
        f'</div>'
    )

register_ui_callbacks(
    eric_says=_set_eric_says,
    status=_set_status,
    log=_append_log
)

def get_webcam():   return capture_frame_raw(CAMERA_WEBCAM)
def get_pantilt():  return capture_frame_raw(CAMERA_PANTILT)
def get_eric():     return _eric_says
def get_status():   return _status
def get_log():      return _log_text

# ─── Module status ────────────────────────────────────────────────────────────

def _battery_row() -> str:
    """Render a battery status row matching _dot() style for the system status box."""
    with _battery_lock:
        v = _battery_voltage
    mono = "font-family:'Courier New',monospace"
    if v is None:
        return f"""
    <div style="display:flex;align-items:center;gap:8px;padding:2px 0;{mono}">
        <div style="width:8px;height:8px;border-radius:50%;background:#333;flex-shrink:0;"></div>
        <span style="color:#aaa;font-size:0.7em;letter-spacing:0.04em;min-width:90px">BATTERY</span>
        <span style="color:#444;font-size:0.65em;letter-spacing:0.06em">--.-V</span>
    </div>"""
    pct   = max(0, min(100, int((v - _BATT_V_MIN) / (_BATT_V_MAX - _BATT_V_MIN) * 100)))
    label, color, _ = _battery_level_gui(pct)
    filled = round(pct / 10)
    bar = (f'<span style="color:{color}">{"█" * filled}</span>'
           f'<span style="color:#1a1a22">{"░" * (10 - filled)}</span>')
    return f"""
    <div style="padding:2px 0 4px 0;{mono}">
        <div style="display:flex;align-items:center;gap:8px">
            <div style="width:8px;height:8px;border-radius:50%;background:{color};box-shadow:0 0 12px {color};flex-shrink:0;"></div>
            <span style="color:#ddd;font-size:0.78em;letter-spacing:0.04em;min-width:90px">BATTERY</span>
            <span style="color:{color};font-size:0.76em;letter-spacing:0.06em;font-weight:bold">{label}</span>
            <span style="font-size:0.72em;margin-left:4px">{bar}</span>
            <span style="color:#ccffff;font-size:0.78em;font-weight:bold;margin-left:4px">{v:.2f}V</span>
            <span style="color:{color};font-size:0.72em;font-weight:bold;margin-left:2px">{pct}%</span>
        </div>
    </div>"""


def _dot(active: bool, label: str, detail: str = "") -> str:
    color  = "#00ffff" if active else "#ff0066"
    glow   = f"0 0 12px {color}" if active else "none"
    state  = "ONLINE" if active else "OFFLINE"
    detail_html = f'<span style="color:#999;font-size:0.8em;margin-left:4px">{detail}</span>' if detail else ""
    return f"""
    <div style="display:flex;align-items:center;gap:8px;padding:3px 0;font-family:'Courier New',monospace">
        <div style="width:9px;height:9px;border-radius:50%;background:{color};box-shadow:{glow};flex-shrink:0;"></div>
        <span style="color:#eee;font-size:0.84em;letter-spacing:0.04em;min-width:90px">{label}</span>
        <span style="color:{color};font-size:0.82em;letter-spacing:0.06em;font-weight:bold">{state}</span>
        {detail_html}
    </div>"""

def get_module_status_html() -> str:
    try:
        import requests
        from config import VLLM_URL
        r = requests.get(VLLM_URL.replace("/v1/chat/completions", "/health"), timeout=1.5)
        cosmos_ok = r.status_code == 200
    except Exception:
        cosmos_ok = False

    try:
        from mission import _ms as _ms_ref
        mission_ok = bool(_ms_ref.mission_active)
    except Exception:
        mission_ok = False

    try:
        from lidar import lidar_available, get_status as ls
        lidar_ok = lidar_available()
        lidar_detail = ""
        if lidar_ok:
            s = ls()
            d = s.get("min_distance", 999)
            lidar_detail = f"{d:.2f}m" if d < 999 else "clear"
    except Exception:
        lidar_ok, lidar_detail = False, ""

    try:
        from oakd import oakd_available
        oakd_ok = oakd_available()
    except Exception as e:
        log.debug("oakd_available check failed: %s", e)
        oakd_ok = False
    oakd_detail = ""
    if oakd_ok:
        try:
            from oakd import get_front_depth
            d = get_front_depth()
            oakd_detail = f"{d:.2f}m" if d is not None else "—"
        except Exception as e:
            log.debug("get_front_depth failed: %s", e)

    try:
        from nav2 import nav2_available, is_navigating
        nav2_ok = nav2_available()
        nav2_detail = "NAV" if (nav2_ok and is_navigating()) else ""
    except Exception:
        nav2_ok, nav2_detail = False, ""

    try:
        tts_ok = piper_available()
        tts_detail = "PIPER" if tts_ok else "GTTS"
    except Exception:
        tts_ok, tts_detail = False, ""

    try:
        motors_ok = motors._ser is not None and motors._ser.is_open
    except Exception:
        motors_ok = False

    rows = (
        _dot(cosmos_ok,  "COSMOS", "vLLM") +
        _dot(mission_ok, "MISSION", _status[:8]) +
        _dot(lidar_ok,   "LIDAR", lidar_detail) +
        _dot(oakd_ok,    "OAK-D", oakd_detail) +
        _dot(nav2_ok,    "NAV2", nav2_detail) +
        _dot(tts_ok,     "TTS", tts_detail) +
        _dot(motors_ok,  "MOTORS", "UART") +
        _battery_row()
    )

    return f"""
    <div style="background:#0a0a0f;border:1px solid #ff00ff44;border-radius:4px;padding:8px 10px;">
        <div style="color:#ff44ff;font-size:0.82em;letter-spacing:0.2em;margin-bottom:6px;border-bottom:1px solid #ff00ff55;padding-bottom:4px;font-weight:bold;text-shadow:0 0 10px #ff00ff88">SYSTEM STATUS</div>
        {rows}
    </div>"""

# ─── Motor telemetry ──────────────────────────────────────────────────────────

def _motor_telemetry_html(direction: str, left: float, right: float) -> str:
    color = {
        "forward":  "#00ffff", "backward": "#ff00ff",
        "left":     "#ffff00", "right":    "#ffff00",
        "stopped":  "#444444", "spinning": "#ff0066",
    }.get(direction, "#444444")

    arrow = {
        "forward":  "▲", "backward": "▼",
        "left":     "◀", "right":    "▶",
        "stopped":  "■", "spinning": "↺",
    }.get(direction, "■")

    speed = max(abs(left), abs(right))
    pct   = int(speed / 0.50 * 100)

    return f"""
    <div style="background:#0a0a0f;border:1px solid {color}66;border-radius:4px;padding:8px 10px;font-family:'Courier New',monospace;">
        <div style="color:{color};font-size:0.7em;font-weight:bold;letter-spacing:0.15em;margin-bottom:6px;border-bottom:1px solid {color}33;padding-bottom:4px">DRIVE SYSTEM</div>
        <div style="display:flex;align-items:center;gap:10px">
            <div style="font-size:1.8em;color:{color};width:32px;text-align:center;text-shadow:0 0 10px {color}66">{arrow}</div>
            <div style="flex:1">
                <div style="color:{color};font-size:0.9em;font-weight:bold;letter-spacing:0.1em;text-transform:uppercase">{direction}</div>
                <div style="color:#888;font-size:0.75em;margin:2px 0">
                    L<span style="color:#fff;margin:0 4px">{left:+.2f}</span>
                    R<span style="color:#fff;margin:0 4px">{right:+.2f}</span>
                    <span style="color:#fff">{speed:.2f}m/s</span>
                </div>
                <div style="height:4px;background:#1a1a1f;border-radius:2px;overflow:hidden;margin-top:3px">
                    <div style="height:100%;width:{pct}%;background:{color};box-shadow:0 0 8px {color};border-radius:2px;transition:width 0.3s"></div>
                </div>
            </div>
        </div>
    </div>"""

def get_motor_telemetry():
    return _motor_telemetry_html(
        _motor_state["direction"],
        _motor_state["left"],
        _motor_state["right"]
    )

# ─── Sensor panels ────────────────────────────────────────────────────────────

def _sensor_panel(label: str, body_html: str, border_color: str = "#00ffff") -> str:
    return f"""
    <div style="background:#0a0a0f;border:1px solid {border_color}44;border-radius:4px;padding:6px 8px;font-family:'Courier New',monospace;">
        <div style="color:{border_color};font-size:0.72em;font-weight:bold;letter-spacing:0.1em;margin-bottom:4px;border-bottom:1px solid {border_color}33;padding-bottom:2px">{label}</div>
        <div style="text-align:center;padding:2px 0">{body_html}</div>
    </div>"""

def get_lidar_html() -> str:
    try:
        from lidar import lidar_available, get_status as ls
        if lidar_available():
            s    = ls()
            d    = s.get("min_distance", 999)
            dist = f"{d:.2f}m" if d < 999 else "—"
            if s.get("obstacle_close"):
                color, state = "#ff0066", "STOP"
            elif s.get("obstacle_near"):
                color, state = "#ffff00", "CAUTION"
            else:
                color, state = "#00ffff", "CLEAR"
            body = f'<span style="color:{color};font-size:0.78em;text-shadow:0 0 8px {color}66">{state}</span> <span style="color:#fff;font-size:1em;font-weight:bold">{dist}</span>'
            return _sensor_panel("LIDAR D500", body, color)
    except Exception:
        pass
    return _sensor_panel("LIDAR D500", '<span style="color:#333;font-size:0.75em">OFFLINE</span>', "#444")

def get_oakd_html() -> str:
    try:
        from oakd import oakd_available
        if oakd_available():
            # Module is up — now try to get depth reading separately
            try:
                from oakd import get_front_depth
                d = get_front_depth()
            except Exception as e:
                log.debug("get_front_depth failed: %s", e)
                d = None
            if d is None:
                body = '<span style="color:#aaa;font-size:0.75em">NO READING</span>'
                color = "#00ffff"
            elif d < 0.30:
                color = "#ff0066"
                body  = f'<span style="color:{color};font-size:0.8em;text-shadow:0 0 8px {color}66">CRITICAL</span> <span style="color:#fff;font-weight:bold">{d:.2f}m</span>'
            elif d < 0.60:
                color = "#ffff00"
                body  = f'<span style="color:{color};font-size:0.8em;text-shadow:0 0 8px {color}66">WARNING</span> <span style="color:#fff;font-weight:bold">{d:.2f}m</span>'
            else:
                color = "#00ffff"
                body  = f'<span style="color:{color};font-size:0.8em;text-shadow:0 0 8px {color}66">OPTIMAL</span> <span style="color:#fff;font-weight:bold">{d:.2f}m</span>'
            return _sensor_panel("OAK-D DEPTH", body, color)
    except Exception as e:
        log.debug("oakd_available failed: %s", e)
    return _sensor_panel("OAK-D LITE", '<span style="color:#333;font-size:0.75em">OFFLINE</span>', "#444")

# ─── Mission helpers ──────────────────────────────────────────────────────────

def load_mission_choices():
    missions = list_missions()
    return missions if missions else ["(no missions found)"]

_selected_mission_name = ""

def on_mission_select(name: str):
    global _selected_mission_name
    if not name or name == "(no missions found)":
        _selected_mission_name = ""
        return ""
    _selected_mission_name = name
    briefing = get_briefing_from_file(name)
    return briefing or ""

def _default_mission_text():
    try:
        briefing = get_briefing_from_file("template")
        if briefing and briefing.strip():
            return briefing.strip()
    except Exception:
        pass
    return _TEMPLATE_MISSION.strip()

def _default_mission_choice():
    choices = load_mission_choices()
    if "template" in choices:
        return "template"
    return choices[0] if choices else None

# ─── Actions ──────────────────────────────────────────────────────────────────

def action_engage(briefing: str):
    if not briefing.strip():
        return "⚠ Enter mission briefing."
    resp = start_mission(briefing.strip(), mission_name=_selected_mission_name)
    return resp

def action_disengage():
    """Stop mission but keep systems online."""
    try:
        stop_mission()
        return "◼ MISSION DISENGAGED — Systems standby"
    except Exception as e:
        return f"❌ Disengage error: {e}"

def action_stop():
    try:
        stop_mission()
    except Exception:
        pass
    try:
        motors.stop()
    except Exception:
        pass

    layer0_result = "⚠ Layer 0 serial not reached"
    try:
        import serial as _serial
        import glob as _glob
        import json as _json

        BAUD = 115200
        ths2  = ["/dev/ttyTHS2"] if _glob.glob("/dev/ttyTHS2") else []
        ports = ths2 + sorted(_glob.glob("/dev/ttyUSB*")) + sorted(_glob.glob("/dev/ttyACM*"))
        port  = ports[0] if ports else None

        if port:
            ser = _serial.Serial(port, BAUD, timeout=1, rtscts=False, xonxoff=False)
            ser.reset_input_buffer()
            ser.reset_output_buffer()
            cmd = _json.dumps({"T": 1, "L": 0.0, "R": 0.0}) + "\n"
            for byte in cmd.encode("utf-8"):
                ser.write(bytes([byte]))
                import time as _time; _time.sleep(0.001)
            ser.close()
            layer0_result = f"✓ Layer 0 stop → {port}"
            log.warning(f"🛑 LAYER 0 E-HALT fired via {port}")
        else:
            layer0_result = "❌ No serial port found"
            log.error("🛑 LAYER 0 E-HALT: no port found")

    except Exception as e:
        layer0_result = f"❌ Layer 0 error: {e}"
        log.error(f"🛑 LAYER 0 E-HALT error: {e}")

    return f"🛑 E-HALT EXECUTED\n{layer0_result}"

def action_char_reply(char_name: str, char_says: str):
    if not char_name.strip() or not char_says.strip():
        return "⚠ Enter name and message.", char_says
    resp = handle_character_response(char_name.strip(), char_says.strip())
    resume_after_interaction()
    return resp, ""   # resp → char_reply box, "" clears char_says input

# ─── CSS & layout ─────────────────────────────────────────────────────────────

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

/* ── Global cyberpunk reset with dark cyan grid ─────────────────────── */
body, .gradio-container {
    background: #050508 !important;
    font-family: 'Share Tech Mono', 'Courier New', monospace !important;
    background-image:
        /* Dark cyan grid lines */
        linear-gradient(rgba(0,255,255,0.32) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,255,0.32) 1px, transparent 1px),
        /* Subtle radial glows */
        radial-gradient(ellipse at 50% 0%, rgba(0,255,255,0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 100%, rgba(0,255,255,0.05) 0%, transparent 50%);
    background-size: 40px 40px, 40px 40px, 100% 100%, 100% 100%;
}
footer { display:none !important; }

/* CRT scanline effect */
body::after {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg,
        rgba(0,255,255,0.015) 0px,
        rgba(0,255,255,0.015) 1px,
        transparent 1px,
        transparent 3px
    );
    pointer-events: none;
    z-index: 9999;
}

/* ── Header with BIGGER title ───────────────────────────────────────── */
.eric-header {
    display: flex;
    align-items: baseline;
    gap: 16px;
    padding: 4px 0 4px 0;
    border-bottom: 1px solid #ff00ff44;
    margin-bottom: 6px;
}
.eric-title {
    font-size: 2.2em;  /* BIGGER */
    font-weight: bold;
    letter-spacing: 0.3em;
    color: #00ffff;
    font-family: 'Share Tech Mono', monospace;
    text-shadow: 
        0 0 10px #00ffff,
        0 0 20px #00ffffaa,
        0 0 40px #00ffff66,
        0 0 80px #00ffff33;
}
.eric-sub {
    font-size: 1.0em;
    color: #ff55ff;
    letter-spacing: 0.1em;
    font-family: 'Share Tech Mono', monospace;
    text-shadow: 0 0 10px #ff00ffaa;
}

/* ── STOP button — PERFECT CIRCLE ───────────────────────────────────── */
#stop-btn {
    width: 72px !important;
    height: 72px !important;
    min-width: 72px !important;
    min-height: 72px !important;
    max-width: 72px !important;
    max-height: 72px !important;
    border-radius: 50% !important;
    background: radial-gradient(circle at 35% 35%, #ff0066, #660022) !important;
    border: 3px solid #ff0066 !important;
    color: #fff !important;
    font-size: 0.75em !important;
    font-weight: bold !important;
    letter-spacing: 0.08em !important;
    font-family: 'Share Tech Mono', monospace !important;
    box-shadow: 
        0 0 25px #ff0066aa,
        0 0 50px #ff006644,
        inset 0 2px 4px #ff669944 !important;
    transition: all 0.15s !important;
    line-height: 1 !important;
    padding: 0 !important;
    aspect-ratio: 1 / 1 !important;
}
#stop-btn:hover {
    box-shadow: 
        0 0 35px #ff0066ff,
        0 0 70px #ff006666,
        inset 0 2px 4px #ff669944 !important;
    transform: scale(1.08) !important;
    border-color: #ff3399 !important;
}
#stop-btn:active {
    transform: scale(0.95) !important;
    box-shadow: 0 0 20px #ff006688 !important;
}

/* ── EMERGENCY label above STOP ─────────────────────────────────────── */
#emergency-label {
    color: #ff1155;
    font-size: 0.9em;
    font-weight: bold;
    letter-spacing: 0.16em;
    text-align: center;
    font-family: 'Share Tech Mono', monospace;
    text-shadow: 0 0 14px #ff005599, 0 0 28px #ff005544;
    padding: 0;
    white-space: nowrap;
}

/* ── ENGAGE button ──────────────────────────────────────────────────── */
#engage-btn {
    background: linear-gradient(135deg, #006666, #00ffff) !important;
    border: 1px solid #00ffff !important;
    color: #000 !important;
    letter-spacing: 0.12em !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.85em !important;
    font-weight: bold !important;
    box-shadow: 0 0 15px #00ffff44 !important;
}
#engage-btn:hover {
    box-shadow: 0 0 25px #00ffffaa !important;
    transform: translateY(-1px) !important;
}

/* ── DISENGAGE button ───────────────────────────────────────────────── */
#disengage-btn {
    background: linear-gradient(135deg, #660066, #ff00ff) !important;
    border: 1px solid #ff00ff !important;
    color: #fff !important;
    letter-spacing: 0.12em !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.85em !important;
    font-weight: bold !important;
    box-shadow: 0 0 15px #ff00ff44 !important;
}
#disengage-btn:hover {
    box-shadow: 0 0 25px #ff00ffaa !important;
    transform: translateY(-1px) !important;
}

/* ── Textboxes ──────────────────────────────────────────────────────── */
textarea, input[type=text] {
    background: #0a0a0f !important;
    border: 1px solid #00ffff44 !important;
    color: #ccffff !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.85em !important;
    border-radius: 3px !important;
}
textarea:focus, input[type=text]:focus {
    border-color: #00ffff !important;
    box-shadow: 0 0 12px #00ffff55 !important;
    outline: none !important;
}

/* ── Labels ──────────────────────────────────────────────────────────── */
label span, .gr-label {
    color: #ff00ff !important;
    font-size: 0.75em !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-family: 'Share Tech Mono', monospace !important;
    text-shadow: 0 0 8px #ff00ff44 !important;
}

/* ── Eric Says panel ─────────────────────────────────────────────────── */
#eric-says textarea {
    border-left: 2px solid #00ffff !important;
    color: #e0ffff !important;
    font-size: 0.95em !important;
    font-weight: bold !important;
    min-height: 60px !important;
    background: #0a0a0f !important;
    text-shadow: 0 0 10px #00ffff44 !important;
}

/* ── Log panel ───────────────────────────────────────────────────────── */
#sys-log textarea {
    color: #88cccc !important;
    font-size: 0.75em !important;
    min-height: 100px !important;
    background: #0a0a0f !important;
}

/* ── Pure HTML D-pad cross layout ────────────────────────────────────── */
#dpad-container {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 40px;
    margin: 6px 0 8px 28px;
}
#dpad-cross {
    display: flex;
    flex-direction: column;
    gap: 4px;
    flex-shrink: 0;
}
.dpad-row {
    display: flex;
    flex-direction: row;
    gap: 4px;
    align-items: center;
}
.dpad-cell {
    width: 46px;
    height: 46px;
    flex-shrink: 0;
}
.dpad-key {
    width: 46px;
    height: 46px;
    flex-shrink: 0;
    background: #080812;
    border: 1.5px solid #00ffff55;
    border-radius: 4px;
    color: #00ffff;
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.25em;
    font-weight: bold;
    cursor: pointer;
    text-shadow: 0 0 8px #00ffff88;
    transition: all 0.1s;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0;
    line-height: 1;
}
.dpad-key:hover {
    border-color: #00ffff;
    background: #001a1a;
    box-shadow: 0 0 14px #00ffff55;
    transform: scale(1.1);
}
.dpad-key:active {
    transform: scale(0.9);
    box-shadow: 0 0 6px #00ffff44;
}
.dpad-key-stop {
    border-color: #ff006655;
    color: #ff0066;
    text-shadow: 0 0 8px #ff006688;
}
.dpad-key-stop:hover {
    border-color: #ff0066;
    background: #1a0008;
    box-shadow: 0 0 12px #ff006655;
}
.dpad-key-spin {
    color: #ff00ff;
    border-color: #ff00ff44;
    text-shadow: 0 0 8px #ff00ff88;
}
.dpad-key-spin:hover {
    border-color: #ff00ff;
    background: #1a001a;
    box-shadow: 0 0 12px #ff00ff55;
}
/* RIGHT side: EMERGENCY label + STOP circle */
#dpad-emergency {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 6px;
    flex-shrink: 0;
}
/* STOP circle — sized to roughly match d-pad height (3 × 46px + 2 × 4px gap = 146px) */
.stop-circle {
    width: 130px;
    height: 130px;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 35%, #ff2266, #550011);
    border: 3px solid #ff0066;
    color: #fff;
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.0em;
    font-weight: bold;
    letter-spacing: 0.12em;
    cursor: pointer;
    box-shadow: 0 0 28px #ff0066aa, 0 0 56px #ff006644;
    transition: all 0.15s;
    line-height: 1;
}
.stop-circle:hover {
    box-shadow: 0 0 40px #ff0066ff, 0 0 80px #ff006666;
    transform: scale(1.05);
    border-color: #ff3399;
}
.stop-circle:active {
    transform: scale(0.92);
    box-shadow: 0 0 18px #ff006688;
}
/* Hidden Gradio buttons — kept in DOM for event binding, invisible */
.hidden-offscreen {
    position: absolute !important;
    left: -9999px !important;
    top: -9999px !important;
    width: 1px !important;
    height: 1px !important;
    overflow: hidden !important;
    pointer-events: none !important;
    opacity: 0 !important;
}
/* ctrl-btn for refresh + transmit buttons */
.ctrl-btn button {
    background: #0a0a0f !important;
    border: 1px solid #00ffff44 !important;
    color: #00ffff !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.9em !important;
    border-radius: 2px !important;
    padding: 4px 8px !important;
    transition: all 0.15s !important;
}
.ctrl-btn button:hover {
    border-color: #00ffff !important;
    background: #001a1a !important;
    box-shadow: 0 0 12px #00ffff55 !important;
}
.ctrl-wide button {
    font-size: 0.85em !important;
    padding: 6px 10px !important;
}

/* ── Camera images ───────────────────────────────────────────────────── */
.cam-label {
    color: #ff44ff;
    font-size: 0.88em;
    font-weight: bold;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-family: 'Share Tech Mono', monospace;
    margin-bottom: 1px;
    margin-top: 2px;
    text-shadow: 0 0 10px #ff00ff99;
}

/* ── Section headers ─────────────────────────────────────────────────── */
.panel-head {
    color: #00ffff;
    font-size: 0.9em;
    font-weight: bold;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    font-family: 'Share Tech Mono', monospace;
    border-bottom: 1px solid #00ffff66;
    padding-bottom: 2px;
    margin-bottom: 3px;
    margin-top: 3px;
    text-shadow: 0 0 12px #00ffff88;
}

/* ── Dropdown ────────────────────────────────────────────────────────── */
.gr-dropdown {
    background: #0a0a0f !important;
    border: 1px solid #ff00ff44 !important;
    color: #ffccff !important;
}
.gr-dropdown:focus {
    border-color: #ff00ff !important;
    box-shadow: 0 0 10px #ff00ff55 !important;
}

/* ── Slider ──────────────────────────────────────────────────────────── */
input[type=range] {
    accent-color: #00ffff !important;
}
.gr-slider {
    background: #0a0a0f !important;
}

/* ── Reduce padding globally ─────────────────────────────────────────── */
.gr-box, .gr-column, .gr-row {
    gap: 4px !important;
}
.gr-form {
    gap: 3px !important;
}

/* ── Compact right column spacing ───────────────────────────────────── */

/* ── ASR mic button ──────────────────────────────────────────────────── */
#asr-mic-btn button {
    background: #0a0a0f !important;
    border: 1.5px solid #ff00ff55 !important;
    color: #ff44ff !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.85em !important;
    letter-spacing: 0.1em !important;
    border-radius: 3px !important;
    transition: all 0.15s !important;
}
#asr-mic-btn button:hover {
    border-color: #ff00ff !important;
    box-shadow: 0 0 14px #ff00ff66 !important;
    background: #1a001a !important;
}
#asr-mic-btn.recording button {
    background: #330011 !important;
    border-color: #ff0066 !important;
    color: #ff0066 !important;
    box-shadow: 0 0 20px #ff006699 !important;
}

.compact-col {
    margin-top: 0 !important;
    padding-top: 0 !important;
}
"""

_HEADER_HTML = """
<div class="eric-header">
    <span class="eric-title">E.R.I.C.</span>
    <span class="eric-sub">
        EDGE ROBOTICS INNOVATION // JETSON ORIN NANO // WAVESHARE UGV ROVER
    </span>
</div>
"""

# ─── Build UI ─────────────────────────────────────────────────────────────────


# ─── ASR actions ──────────────────────────────────────────────────────────────

def _asr_status_html(recording: bool, last_text: str = "") -> str:
    if recording:
        color, label, dot = "#ff0066", "RECORDING...", "🔴"
    else:
        color, label, dot = "#ff44ff", "PRESS & HOLD TO SPEAK", "🎙"
    text_line = f'<div style="color:#aaa;font-size:0.7em;margin-top:3px;font-style:italic">{last_text[:60]}</div>' if last_text else ""
    return (
        f'<div style="background:#0a0a0f;border:1px solid {color}44;border-radius:4px;'
        f'padding:5px 8px;font-family:\'Share Tech Mono\',monospace;font-size:0.75em;">'
        f'<span style="color:{color};letter-spacing:0.1em">{dot} {label}</span>'
        f'{text_line}</div>'
    )


def action_asr_start():
    """Called on mic button press — start recording."""
    global _asr_recording
    if not ASR_ENABLED:
        return _asr_status_html(False, "ASR disabled in config")
    try:
        from asr import init_asr, start_recording, asr_available
        if not asr_available():
            init_asr()
        ok = start_recording()
        _asr_recording = ok
        return _asr_status_html(ok, "")
    except Exception as e:
        log.error(f"ASR start error: {e}")
        return _asr_status_html(False, str(e))


def action_asr_stop(briefing: str, char_name_val: str, mode: str):
    """
    Called on mic button release — stop and transcribe.
    Returns (updated_briefing, updated_char_says, asr_status_html).
    mode: "MISSION" fills briefing box | "COMMS" fills char says box.
    """
    global _asr_recording, _asr_last_text
    _asr_recording = False
    if not ASR_ENABLED:
        return briefing, "", _asr_status_html(False, "ASR disabled")
    try:
        from asr import stop_and_transcribe
        text = stop_and_transcribe()
        if not text:
            return briefing, "", _asr_status_html(False, "(nothing heard)")
        _asr_last_text = text
        if mode == "COMMS":
            return briefing, text, _asr_status_html(False, text)
        else:
            return text, "", _asr_status_html(False, text)
    except Exception as e:
        log.error(f"ASR stop error: {e}")
        return briefing, "", _asr_status_html(False, str(e))


def build_ui():
    default_text    = _default_mission_text()
    default_mission = _default_mission_choice()

    with gr.Blocks(title="ERIC — Mission Control", css=_CSS) as demo:

        gr.HTML(_HEADER_HTML)

        with gr.Row(equal_height=False):

            # ═══════════════════════════════════════════════════════════════
            # LEFT COLUMN — cameras + status + sensors + power
            # ═══════════════════════════════════════════════════════════════
            with gr.Column(scale=2, min_width=180):

                gr.HTML('<div class="cam-label">◉ PAN-TILT FEED</div>')
                pantilt_img = gr.Image(streaming=True, height=175, label="", show_label=False)

                gr.HTML('<div class="cam-label">◉ WEBCAM FEED</div>')
                webcam_img = gr.Image(streaming=True, height=145, label="", show_label=False)

                gr.HTML('<div class="panel-head" style="margin-top:2px">◈ MODULE STATUS</div>')
                module_status = gr.HTML(value=get_module_status_html())
                voice_status  = gr.HTML(value=get_voice_state_html())

                with gr.Row(equal_height=True):
                    lidar_display = gr.HTML(value=get_lidar_html())
                    oakd_display  = gr.HTML(value=get_oakd_html())

            # ═══════════════════════════════════════════════════════════════
            # CENTRE COLUMN — mission + comms
            # ═══════════════════════════════════════════════════════════════
            with gr.Column(scale=3, min_width=300):

                gr.HTML('<div class="panel-head">◈ MISSION BRIEFING</div>')
                with gr.Row():
                    mission_dd = gr.Dropdown(
                        choices=load_mission_choices(),
                        value=default_mission,
                        label="",
                        show_label=False,
                        scale=4
                    )
                    refresh_btn = gr.Button("↻", scale=0, min_width=36, elem_classes=["ctrl-btn"])

                briefing_box = gr.Textbox(
                    value=default_text,
                    label="",
                    show_label=False,
                    lines=5,
                    max_lines=7,
                    placeholder="Mission briefing..."
                )

                with gr.Row():
                    engage_btn    = gr.Button("▶ ENGAGE",    elem_id="engage-btn",    variant="primary",   scale=1)
                    disengage_btn = gr.Button("◼ DISENGAGE", elem_id="disengage-btn", variant="secondary", scale=1)

                gr.HTML('<div class="panel-head">◈ VOICE INPUT</div>')
                with gr.Row():
                    asr_mode_dd = gr.Dropdown(
                        choices=["MISSION", "COMMS"],
                        value="MISSION",
                        label="",
                        show_label=False,
                        scale=1,
                        info="MISSION → briefing box · COMMS → character reply"
                    )
                    asr_mic_btn = gr.Button("🎙 HOLD TO SPEAK", scale=2, elem_id="asr-mic-btn")
                asr_status = gr.HTML(value=_asr_status_html(False, ""))

                gr.HTML('<div class="panel-head">◈ ERIC TRANSMISSION</div>')
                eric_says_box = gr.Textbox(
                    value="", label="", show_label=False,
                    interactive=False, lines=4, max_lines=5,
                    elem_id="eric-says", placeholder="Awaiting transmission..."
                )

                gr.HTML('<div class="panel-head">◈ CHARACTER COMMS</div>')
                gr.HTML('<div style="color:#ff44ff;font-size:0.88em;font-family:\'Share Tech Mono\',monospace;letter-spacing:0.06em;font-weight:bold;text-shadow:0 0 10px #ff00ffaa;margin-bottom:4px;padding:2px 0;border-left:3px solid #ff00ff;padding-left:8px">⟶ WHEN ERIC STOPS — TYPE AS CHARACTER</div>')
                with gr.Row():
                    char_name = gr.Textbox(placeholder="Name...",    label="", show_label=False, scale=1)
                    char_says = gr.Textbox(placeholder="Message...", label="", show_label=False, scale=3)
                with gr.Row():
                    char_btn   = gr.Button("⟶ TRANSMIT", variant="secondary", scale=1, elem_classes=["ctrl-btn", "ctrl-wide"])
                    char_reply = gr.Textbox(label="", show_label=False, interactive=False, lines=2, scale=3, placeholder="Response...")

            # ═══════════════════════════════════════════════════════════════
            # RIGHT COLUMN — manual override + drive + log
            # ═══════════════════════════════════════════════════════════════
            with gr.Column(scale=2, min_width=200):

                gr.HTML('<div class="panel-head" style="margin-top:0">◈ MANUAL OVERRIDE</div>')
                speed_slider = gr.Slider(minimum=0.05, maximum=0.50, value=0.20, step=0.05, label="SPEED m/s")

                # ── D-pad: pure HTML cross + hidden Gradio triggers ──────
                # Visual d-pad rendered in HTML/CSS so buttons stay square.
                # Hidden off-screen Gradio buttons receive JS click events.
                gr.HTML("""
<div id="dpad-container">

  <!-- LEFT: 3x3 cross grid -->
  <div id="dpad-cross">
    <div class="dpad-row">
      <div class="dpad-cell"></div>
      <button class="dpad-key" onclick="document.getElementById('gr-btn-fwd').click()">▲</button>
      <div class="dpad-cell"></div>
    </div>
    <div class="dpad-row">
      <button class="dpad-key" onclick="document.getElementById('gr-btn-left').click()">◀</button>
      <button class="dpad-key dpad-key-stop" onclick="document.getElementById('gr-btn-halt').click()">■</button>
      <button class="dpad-key" onclick="document.getElementById('gr-btn-right').click()">▶</button>
    </div>
    <div class="dpad-row">
      <button class="dpad-key dpad-key-spin" onclick="document.getElementById('gr-btn-spinl').click()">↺</button>
      <button class="dpad-key" onclick="document.getElementById('gr-btn-back').click()">▼</button>
      <button class="dpad-key dpad-key-spin" onclick="document.getElementById('gr-btn-spinr').click()">↻</button>
    </div>
  </div>

  <!-- RIGHT: EMERGENCY label + STOP circle, full height of d-pad -->
  <div id="dpad-emergency">
    <div id="emergency-label">EMERGENCY</div>
    <button class="stop-circle" onclick="document.getElementById('gr-btn-stop').click()">STOP</button>
  </div>

</div>
""")
                # Hidden Gradio buttons — in DOM but invisible, receive JS clicks
                with gr.Row(elem_classes=["hidden-offscreen"]):
                    btn_fwd    = gr.Button("fwd",   elem_id="gr-btn-fwd")
                    btn_back   = gr.Button("back",  elem_id="gr-btn-back")
                    btn_left   = gr.Button("left",  elem_id="gr-btn-left")
                    btn_right  = gr.Button("right", elem_id="gr-btn-right")
                    btn_halt   = gr.Button("halt",  elem_id="gr-btn-halt")
                    btn_spin_l = gr.Button("spinl", elem_id="gr-btn-spinl")
                    btn_spin_r = gr.Button("spinr", elem_id="gr-btn-spinr")
                    stop_btn   = gr.Button("stop",  elem_id="gr-btn-stop")

                motor_status = gr.Textbox(
                    label="", show_label=False,
                    interactive=False, max_lines=1,
                    value="STOPPED", placeholder="Motor status"
                )

                gr.HTML('<div class="panel-head">◈ DRIVE SYSTEM</div>')
                motor_display = gr.HTML(value=_motor_telemetry_html("stopped", 0.0, 0.0))

                gr.HTML('<div class="panel-head">◈ SYSTEM LOG</div>')
                log_box = gr.Textbox(
                    value="", label="", show_label=False,
                    interactive=False, lines=8, max_lines=12,
                    elem_id="sys-log", placeholder="System log..."
                )

        # ── Wire up buttons ───────────────────────────────────────────────
        # ASR mic — mousedown starts recording, mouseup stops + transcribes
        asr_mic_btn.click(
            action_asr_stop,
            inputs=[briefing_box, char_name, asr_mode_dd],
            outputs=[briefing_box, char_says, asr_status]
        )
        # Note: Gradio doesn't support mousedown/mouseup natively.
        # The mic button is a toggle — first click starts, second click stops + transcribes.
        # TODO: replace with JS mousedown/mouseup when Gradio adds pointer event support.

        engage_btn.click(action_engage,    inputs=[briefing_box], outputs=[motor_status])
        disengage_btn.click(action_disengage,                     outputs=[motor_status])
        stop_btn.click(action_stop,                               outputs=[motor_status])

        refresh_btn.click(lambda: gr.update(choices=load_mission_choices()), outputs=[mission_dd])
        mission_dd.change(on_mission_select, inputs=[mission_dd], outputs=[briefing_box])

        char_btn.click(action_char_reply, inputs=[char_name, char_says], outputs=[char_reply, char_says])
        char_says.submit(action_char_reply, inputs=[char_name, char_says], outputs=[char_reply, char_says])

        btn_fwd.click(   lambda s: (motors.forward(s),             f"▲ FWD {s:.2f}")[1], inputs=[speed_slider], outputs=[motor_status])
        btn_back.click(  lambda s: (motors.backward(s),            f"▼ BWD {s:.2f}")[1], inputs=[speed_slider], outputs=[motor_status])
        btn_left.click(  lambda s: (motors.left(s),                f"◀ L {s:.2f}")[1],   inputs=[speed_slider], outputs=[motor_status])
        btn_right.click( lambda s: (motors.right(s),               f"▶ R {s:.2f}")[1],   inputs=[speed_slider], outputs=[motor_status])
        btn_halt.click(  lambda:   (motors.stop(),                  "■ STOP")[1],                               outputs=[motor_status])
        btn_spin_l.click(lambda s: (motors._send(-s, s),           f"↺ SPIN {s:.2f}")[1], inputs=[speed_slider], outputs=[motor_status])
        btn_spin_r.click(lambda s: (motors._send(s, -s),           f"↻ SPIN {s:.2f}")[1], inputs=[speed_slider], outputs=[motor_status])

        # Live polling
        gr.Timer(1.0).tick(get_webcam,            outputs=webcam_img)
        gr.Timer(1.0).tick(get_pantilt,           outputs=pantilt_img)
        gr.Timer(1.0).tick(get_eric,              outputs=eric_says_box)
        gr.Timer(1.5).tick(get_log,              outputs=log_box)
        gr.Timer(0.5).tick(get_motor_telemetry,  outputs=motor_display)
        gr.Timer(1.0).tick(get_lidar_html,       outputs=lidar_display)
        gr.Timer(1.0).tick(get_oakd_html,        outputs=oakd_display)
        gr.Timer(3.0).tick(get_module_status_html, outputs=module_status)
        gr.Timer(1.0).tick(get_voice_state_html,   outputs=voice_status)

    return demo

# ─── Launch ───────────────────────────────────────────────────────────────────

def launch():
    init_tts()
    demo = build_ui()
    log.info(f"🌐 ERIC Mission Control → http://{GRADIO_HOST}:{GRADIO_PORT}")
    demo.launch(
        server_name=GRADIO_HOST,
        server_port=GRADIO_PORT,
        show_error=True,
        quiet=False,
        theme=gr.themes.Base(primary_hue="cyan", neutral_hue="slate"),
        css=_CSS,
    )
