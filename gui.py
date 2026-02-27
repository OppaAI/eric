"""
ERIC — Cockpit Dashboard GUI
Cyberpunk mission control for 1680x1050 single-screen operation.
"""

import threading
import logging

import gradio as gr

from config import GRADIO_HOST, GRADIO_PORT, CAMERA_WEBCAM, CAMERA_PANTILT, MISSIONS_DIR
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

def _dot(active: bool, label: str, detail: str = "") -> str:
    color  = "#00ffff" if active else "#ff0066"
    glow   = f"0 0 12px {color}" if active else "none"
    state  = "ONLINE" if active else "OFFLINE"
    detail_html = f'<span style="color:#555;font-size:0.65em;margin-left:4px">{detail}</span>' if detail else ""
    return f"""
    <div style="display:flex;align-items:center;gap:8px;padding:2px 0;font-family:'Courier New',monospace">
        <div style="width:8px;height:8px;border-radius:50%;background:{color};box-shadow:{glow};flex-shrink:0;"></div>
        <span style="color:#aaa;font-size:0.7em;letter-spacing:0.04em;min-width:90px">{label}</span>
        <span style="color:{color};font-size:0.65em;letter-spacing:0.06em">{state}</span>
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
        from oakd import oakd_available, get_front_depth
        oakd_ok = oakd_available()
        oakd_detail = ""
        if oakd_ok:
            d = get_front_depth()
            oakd_detail = f"{d:.2f}m" if d is not None else "—"
    except Exception:
        oakd_ok, oakd_detail = False, ""

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
        _dot(motors_ok,  "MOTORS", "UART")
    )

    return f"""
    <div style="background:#0a0a0f;border:1px solid #ff00ff44;border-radius:4px;padding:8px 10px;">
        <div style="color:#ff00ff;font-size:0.6em;letter-spacing:0.2em;margin-bottom:6px;border-bottom:1px solid #ff00ff33;padding-bottom:4px">SYSTEM STATUS</div>
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
    <div style="background:#0a0a0f;border:1px solid {border_color}44;border-radius:4px;padding:8px 10px;font-family:'Courier New',monospace;margin-bottom:4px">
        <div style="color:{border_color};font-size:0.7em;font-weight:bold;letter-spacing:0.12em;margin-bottom:5px;border-bottom:1px solid {border_color}33;padding-bottom:3px">{label}</div>
        {body_html}
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
            body = f"""
            <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="color:{color};font-size:0.8em;letter-spacing:0.08em;text-shadow:0 0 8px {color}66">{state}</span>
                <span style="color:#fff;font-size:1em;font-weight:bold">{dist}</span>
            </div>"""
            return _sensor_panel("LIDAR D500", body, color)
    except Exception:
        pass
    return _sensor_panel("LIDAR D500", '<span style="color:#333;font-size:0.75em">OFFLINE</span>', "#444")

def get_oakd_html() -> str:
    try:
        from oakd import oakd_available, get_front_depth
        if oakd_available():
            d = get_front_depth()
            if d is None:
                body = '<span style="color:#555;font-size:0.75em">NO READING</span>'
                color = "#444"
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
    except Exception:
        pass
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
        return "⚠ Enter character data.", ""
    resp = handle_character_response(char_name.strip(), char_says.strip())
    resume_after_interaction()
    return resp, ""

# ─── CSS & layout ─────────────────────────────────────────────────────────────

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

/* ── Global cyberpunk reset with dark cyan grid ─────────────────────── */
body, .gradio-container {
    background: #050508 !important;
    font-family: 'Share Tech Mono', 'Courier New', monospace !important;
    background-image:
        /* Dark cyan grid lines */
        linear-gradient(rgba(0,255,255,0.08) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,255,0.08) 1px, transparent 1px),
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
    font-size: 0.7em;
    color: #ff00ff;
    letter-spacing: 0.1em;
    font-family: 'Share Tech Mono', monospace;
    text-shadow: 0 0 10px #ff00ff66;
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

/* ── SQUARE joystick buttons ────────────────────────────────────────── */
.ctrl-btn button {
    background: #0a0a0f !important;
    border: 1px solid #00ffff66 !important;
    color: #00ffff !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 1.1em !important;
    font-weight: bold !important;
    border-radius: 2px !important;
    /* SQUARE: equal width and height */
    width: 44px !important;
    height: 44px !important;
    min-width: 44px !important;
    min-height: 44px !important;
    max-width: 44px !important;
    max-height: 44px !important;
    padding: 0 !important;
    text-shadow: 0 0 8px #00ffff88 !important;
    transition: all 0.15s !important;
}
.ctrl-btn button:hover {
    border-color: #00ffff !important;
    background: #001a1a !important;
    box-shadow: 0 0 15px #00ffff55 !important;
    transform: scale(1.05) !important;
}
.ctrl-stop button {
    border-color: #ff0066 !important;
    color: #ff0066 !important;
    text-shadow: 0 0 8px #ff006688 !important;
}
.ctrl-stop button:hover {
    background: #1a0008 !important;
    border-color: #ff3399 !important;
    box-shadow: 0 0 15px #ff006655 !important;
}
/* Wide buttons for spin - also square but wider */
.ctrl-wide button {
    width: 44px !important;
    height: 44px !important;
    min-width: 44px !important;
    min-height: 44px !important;
    font-size: 0.85em !important;
}

/* ── Camera images ───────────────────────────────────────────────────── */
.cam-label {
    color: #ff00ff;
    font-size: 0.65em;
    font-weight: bold;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-family: 'Share Tech Mono', monospace;
    margin-bottom: 2px;
    text-shadow: 0 0 10px #ff00ff66;
}

/* ── Section headers ─────────────────────────────────────────────────── */
.panel-head {
    color: #00ffff;
    font-size: 0.7em;
    font-weight: bold;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    font-family: 'Share Tech Mono', monospace;
    border-bottom: 1px solid #00ffff44;
    padding-bottom: 3px;
    margin-bottom: 5px;
    margin-top: 6px;
    text-shadow: 0 0 10px #00ffff44;
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
.compact-col {
    margin-top: 0 !important;
    padding-top: 0 !important;
}
"""

_HEADER_HTML = """
<div class="eric-header">
    <span class="eric-title">E.R.I.C.</span>
    <span class="eric-sub">
        EDGE ROBOTICS INNOVATION // COSMOS COOKOFF 2026 // JETSON ORIN NANO // WAVESHARE UGV // VANCOUVER BC // ~$750 CAD
    </span>
</div>
"""

# ─── Build UI ─────────────────────────────────────────────────────────────────

def build_ui():
    default_text    = _default_mission_text()
    default_mission = _default_mission_choice()

    with gr.Blocks(title="ERIC — Mission Control", css=_CSS) as demo:

        gr.HTML(_HEADER_HTML)

        with gr.Row(equal_height=False):

            # ═══════════════════════════════════════════════════════════════
            # LEFT COLUMN — cameras + status + sensors
            # ═══════════════════════════════════════════════════════════════
            with gr.Column(scale=2, min_width=180):

                gr.HTML('<div class="cam-label">◉ PAN-TILT FEED</div>')
                pantilt_img = gr.Image(streaming=True, height=140, label="", show_label=False)

                gr.HTML('<div class="cam-label">◉ WEBCAM FEED</div>')
                webcam_img = gr.Image(streaming=True, height=110, label="", show_label=False)

                gr.HTML('<div class="panel-head" style="margin-top:4px">◈ MODULE STATUS</div>')
                module_status = gr.HTML(value=get_module_status_html())

                gr.HTML('<div class="panel-head">◈ SENSORS</div>')
                lidar_display = gr.HTML(value=get_lidar_html())
                oakd_display  = gr.HTML(value=get_oakd_html())

            # ═══════════════════════════════════════════════════════════════
            # CENTRE COLUMN — mission + comms (log moved to right)
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

                # ENGAGE and DISENGAGE side by side
                with gr.Row():
                    engage_btn = gr.Button("▶ ENGAGE", elem_id="engage-btn", variant="primary", scale=1)
                    disengage_btn = gr.Button("◼ DISENGAGE", elem_id="disengage-btn", variant="secondary", scale=1)

                gr.HTML('<div class="panel-head">◈ ERIC TRANSMISSION</div>')
                eric_says_box = gr.Textbox(
                    value="",
                    label="",
                    show_label=False,
                    interactive=False,
                    lines=4,
                    max_lines=5,
                    elem_id="eric-says",
                    placeholder="Awaiting transmission..."
                )

                gr.HTML('<div class="panel-head">◈ CHARACTER COMMS</div>')
                gr.HTML('<span style="color:#ff00ff88;font-size:0.65em;font-family:\'Courier New\',monospace;letter-spacing:0.05em">WHEN ERIC STOPS — TYPE AS CHARACTER</span>')
                with gr.Row():
                    char_name = gr.Textbox(placeholder="Name...", label="", show_label=False, scale=1)
                    char_says = gr.Textbox(placeholder="Message...", label="", show_label=False, scale=3)
                with gr.Row():
                    char_btn = gr.Button("⟶ TRANSMIT", variant="secondary", scale=1, elem_classes=["ctrl-btn", "ctrl-wide"])
                    char_reply = gr.Textbox(label="", show_label=False, interactive=False, lines=2, scale=3, placeholder="Response...")

            # ═══════════════════════════════════════════════════════════════
            # RIGHT COLUMN — joystick + STOP + drive + LOG (moved here)
            # ═══════════════════════════════════════════════════════════════
            with gr.Column(scale=2, min_width=200):

                gr.HTML('<div class="panel-head" style="margin-top:0">◈ MANUAL OVERRIDE</div>')

                # Speed slider at top
                speed_slider = gr.Slider(minimum=0.05, maximum=0.50, value=0.20, step=0.05, label="SPEED m/s")

                # Joystick layout with STOP on right
                with gr.Row(equal_height=True):
                    # Left side: D-pad + spin buttons integrated
                    with gr.Column(scale=3, min_width=0):
                        # Row 1: Forward
                        with gr.Row():
                            gr.HTML('<div style="flex:1"></div>')
                            btn_fwd = gr.Button("▲", elem_classes=["ctrl-btn"])
                            gr.HTML('<div style="flex:1"></div>')
                        
                        # Row 2: Left, Halt, Right
                        with gr.Row():
                            btn_left = gr.Button("◀", elem_classes=["ctrl-btn"])
                            btn_halt = gr.Button("■", elem_classes=["ctrl-btn", "ctrl-stop"])
                            btn_right = gr.Button("▶", elem_classes=["ctrl-btn"])
                        
                        # Row 3: Spin Left, Down, Spin Right
                        with gr.Row():
                            btn_spin_l = gr.Button("↺", elem_classes=["ctrl-btn"])
                            btn_back = gr.Button("▼", elem_classes=["ctrl-btn"])
                            btn_spin_r = gr.Button("↻", elem_classes=["ctrl-btn"])

                    # Right side: STOP button (perfect circle)
                    with gr.Column(scale=1, min_width=0):
                        gr.HTML('<div style="height:45px"></div>')
                        stop_btn = gr.Button("STOP", elem_id="stop-btn")
                        gr.HTML('<div style="color:#ff0066;font-size:0.6em;letter-spacing:0.12em;text-align:center;margin-top:6px;font-family:\'Courier New\',monospace;text-shadow:0 0 10px #ff006666">E-HALT</div>')

                motor_status = gr.Textbox(
                    label="", show_label=False,
                    interactive=False, max_lines=1,
                    value="STOPPED",
                    placeholder="Motor status"
                )

                gr.HTML('<div class="panel-head">◈ DRIVE SYSTEM</div>')
                motor_display = gr.HTML(value=_motor_telemetry_html("stopped", 0.0, 0.0))

                # SYSTEM LOG moved to bottom of right column
                gr.HTML('<div class="panel-head">◈ SYSTEM LOG</div>')
                log_box = gr.Textbox(
                    value="",
                    label="",
                    show_label=False,
"""
ERIC — Cockpit Dashboard GUI
Cyberpunk mission control for 1680x1050 single-screen operation.
"""

import threading
import logging

import gradio as gr

from config import GRADIO_HOST, GRADIO_PORT, CAMERA_WEBCAM, CAMERA_PANTILT, MISSIONS_DIR
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

def _dot(active: bool, label: str, detail: str = "") -> str:
    color  = "#00ffff" if active else "#ff0066"
    glow   = f"0 0 12px {color}" if active else "none"
    state  = "ONLINE" if active else "OFFLINE"
    detail_html = f'<span style="color:#888;font-size:0.78em;margin-left:4px">{detail}</span>' if detail else ""
    return f"""
    <div style="display:flex;align-items:center;gap:8px;padding:2px 0;font-family:'Courier New',monospace">
        <div style="width:9px;height:9px;border-radius:50%;background:{color};box-shadow:{glow};flex-shrink:0;"></div>
        <span style="color:#ccc;font-size:0.82em;letter-spacing:0.04em;min-width:90px">{label}</span>
        <span style="color:{color};font-size:0.78em;letter-spacing:0.06em">{state}</span>
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
        from oakd import oakd_available, get_front_depth
        oakd_ok = oakd_available()
        oakd_detail = ""
        if oakd_ok:
            d = get_front_depth()
            oakd_detail = f"{d:.2f}m" if d is not None else "—"
    except Exception:
        oakd_ok, oakd_detail = False, ""

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
        _dot(motors_ok,  "MOTORS", "UART")
    )

    return f"""
    <div style="background:#0a0a0f;border:1px solid #ff00ff44;border-radius:4px;padding:8px 10px;">
        <div style="color:#ff00ff;font-size:0.75em;letter-spacing:0.15em;margin-bottom:6px;border-bottom:1px solid #ff00ff33;padding-bottom:4px">SYSTEM STATUS</div>
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
        <div style="color:{color};font-size:0.82em;font-weight:bold;letter-spacing:0.12em;margin-bottom:6px;border-bottom:1px solid {color}33;padding-bottom:4px">DRIVE SYSTEM</div>
        <div style="display:flex;align-items:center;gap:10px">
            <div style="font-size:1.8em;color:{color};width:32px;text-align:center;text-shadow:0 0 10px {color}66">{arrow}</div>
            <div style="flex:1">
                <div style="color:{color};font-size:0.9em;font-weight:bold;letter-spacing:0.1em;text-transform:uppercase">{direction}</div>
                <div style="color:#aaa;font-size:0.82em;margin:2px 0">
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
    <div style="background:#0a0a0f;border:1px solid {border_color}44;border-radius:4px;padding:8px 10px;font-family:'Courier New',monospace;margin-bottom:4px">
        <div style="color:{border_color};font-size:0.82em;font-weight:bold;letter-spacing:0.10em;margin-bottom:5px;border-bottom:1px solid {border_color}33;padding-bottom:3px">{label}</div>
        {body_html}
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
            body = f"""
            <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="color:{color};font-size:0.9em;letter-spacing:0.08em;text-shadow:0 0 8px {color}66">{state}</span>
                <span style="color:#fff;font-size:1em;font-weight:bold">{dist}</span>
            </div>"""
            return _sensor_panel("LIDAR D500", body, color)
    except Exception:
        pass
    return _sensor_panel("LIDAR D500", '<span style="color:#333;font-size:0.75em">OFFLINE</span>', "#444")

def get_oakd_html() -> str:
    try:
        from oakd import oakd_available, get_front_depth
        if oakd_available():
            d = get_front_depth()
            if d is None:
                body = '<span style="color:#555;font-size:0.75em">NO READING</span>'
                color = "#444"
            elif d < 0.30:
                color = "#ff0066"
                body  = f'<span style="color:{color};font-size:0.9em;text-shadow:0 0 8px {color}66">CRITICAL</span> <span style="color:#fff;font-weight:bold">{d:.2f}m</span>'
            elif d < 0.60:
                color = "#ffff00"
                body  = f'<span style="color:{color};font-size:0.9em;text-shadow:0 0 8px {color}66">WARNING</span> <span style="color:#fff;font-weight:bold">{d:.2f}m</span>'
            else:
                color = "#00ffff"
                body  = f'<span style="color:{color};font-size:0.9em;text-shadow:0 0 8px {color}66">OPTIMAL</span> <span style="color:#fff;font-weight:bold">{d:.2f}m</span>'
            return _sensor_panel("OAK-D DEPTH", body, color)
    except Exception:
        pass
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
        return "⚠ Enter character data.", ""
    resp = handle_character_response(char_name.strip(), char_says.strip())
    resume_after_interaction()
    return resp, ""

# ─── CSS & layout ─────────────────────────────────────────────────────────────

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

/* ── Global cyberpunk reset with dark cyan grid ─────────────────────── */
body, .gradio-container {
    background: #050508 !important;
    font-family: 'Share Tech Mono', 'Courier New', monospace !important;
    background-image:
        /* Dark cyan grid lines */
        linear-gradient(rgba(0,255,255,0.08) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,255,0.08) 1px, transparent 1px),
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
    font-size: 0.82em;
    color: #ff88ff;
    letter-spacing: 0.08em;
    font-family: 'Share Tech Mono', monospace;
    text-shadow: none;
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
    color: #ff66ff !important;
    font-size: 0.82em !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-family: 'Share Tech Mono', monospace !important;
    text-shadow: none !important;
}

/* ── Eric Says panel ─────────────────────────────────────────────────── */
#eric-says textarea {
    border-left: 2px solid #00ffff !important;
    color: #e8ffff !important;
    font-size: 0.95em !important;
    font-weight: bold !important;
    min-height: 60px !important;
    background: #0a0a0f !important;
}

/* ── Log panel ───────────────────────────────────────────────────────── */
#sys-log textarea {
    color: #aadddd !important;
    font-size: 0.82em !important;
    min-height: 100px !important;
    background: #0a0a0f !important;
}

/* ── SQUARE joystick buttons ────────────────────────────────────────── */
.ctrl-btn button {
    background: #0a0a0f !important;
    border: 1px solid #00ffff66 !important;
    color: #00ffff !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 1.1em !important;
    font-weight: bold !important;
    border-radius: 2px !important;
    /* SQUARE: equal width and height */
    width: 44px !important;
    height: 44px !important;
    min-width: 44px !important;
    min-height: 44px !important;
    max-width: 44px !important;
    max-height: 44px !important;
    padding: 0 !important;
    text-shadow: 0 0 8px #00ffff88 !important;
    transition: all 0.15s !important;
}
.ctrl-btn button:hover {
    border-color: #00ffff !important;
    background: #001a1a !important;
    box-shadow: 0 0 15px #00ffff55 !important;
    transform: scale(1.05) !important;
}
.ctrl-stop button {
    border-color: #ff0066 !important;
    color: #ff0066 !important;
    text-shadow: 0 0 8px #ff006688 !important;
}
.ctrl-stop button:hover {
    background: #1a0008 !important;
    border-color: #ff3399 !important;
    box-shadow: 0 0 15px #ff006655 !important;
}
/* Wide buttons for spin - also square but wider */
.ctrl-wide button {
    width: 44px !important;
    height: 44px !important;
    min-width: 44px !important;
    min-height: 44px !important;
    font-size: 0.85em !importr, outputs=motor_status)
        btn_right.click(lambda s: (motors.right(s), f"▶ R {s:.2f}")[1], inputs=speed_slider, outputs=motor_status)
        btn_halt.click(lambda: (motors.stop(), "■ STOP")[1], outputs=motor_status)
        btn_spin_l.click(lambda s: (motors._send(-s, s), f"↺ SPIN {s:.2f}")[1], inputs=speed_slider, outputs=motor_status)
        btn_spin_r.click(lambda s: (motors._send(s, -s), f"↻ SPIN {s:.2f}")[1], inputs=speed_slider, outputs=motor_status)

        # Live polling
        gr.Timer(1.0).tick(get_webcam, outputs=webcam_img)
        gr.Timer(1.0).tick(get_pantilt, outputs=pantilt_img)
        gr.Timer(1.0).tick(get_eric, outputs=eric_says_box)
        gr.Timer(1.5).tick(get_log, outputs=log_box)
        gr.Timer(0.5).tick(get_motor_telemetry, outputs=motor_display)
        gr.Timer(1.0).tick(get_lidar_html, outputs=lidar_display)
        gr.Timer(1.0).tick(get_oakd_html, outputs=oakd_display)
        gr.Timer(2.0).tick(get_module_status_html, outputs=module_status)

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
