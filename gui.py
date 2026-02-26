"""
ERIC — Cockpit Dashboard GUI
Mission control interface styled as a tracked robot operations centre.
Three-column layout:
  LEFT   — narrow dual camera feeds
  CENTRE — mission briefing + character comms
  RIGHT  — module status lights, gauges, manual controls

Key changes from previous GUI:
  - Module status indicator lights (green = active, red = inactive)
  - Round red STOP button replacing rectangular emergency stop + disengage
  - AI speech (Eric Says) and system log are separate panels
  - Mission defaults to template on startup with template text pre-loaded
  - scan360 / nav / system events go to LOG only — AI speech panel stays clean
  - Camera column is narrow (scale=2) vs centre+right (scale=3 each)
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
_eric_says  = ""          # AI speech only — never overwritten by system events
_status     = "IDLE"
_log_text   = ""          # System events, scan results, nav checks

# Motor state for telemetry
_motor_state = {"direction": "stopped", "left": 0.0, "right": 0.0}

# Template default mission text shown at startup
_TEMPLATE_MISSION = """\
You are ERIC — Edge Robotics Innovation by Cosmos.
You are on a search and rescue mission.

OBJECTIVE:
Search the area systematically. Identify any persons, robots, or
objects of interest. Approach and interact with anything relevant.
Report your findings.

RULES:
- You have NO arms — never engage in combat
- Avoid all obstacles — use your LiDAR and depth camera
- Talk to anyone you find and gather information
- If someone cannot help, thank them and move on
- Reason carefully from your egocentric camera view
"""

# ─── UI callbacks — called from mission.py background threads ─────────────────

def _set_eric_says(t):
    """AI speech only — set from Cosmos responses and Eric's spoken lines."""
    global _eric_says
    _eric_says = t

def _set_status(t):
    global _status
    _status = str(t).upper()

def _append_log(t):
    """System events: scan results, nav checks, avoidance, state changes."""
    global _log_text
    lines = (_log_text + "\n" + str(t)).strip().split("\n")
    _log_text = "\n".join(lines[-120:])   # keep last 120 lines


register_ui_callbacks(
    eric_says=_set_eric_says,
    status=_set_status,
    log=_append_log
)


# ─── Camera feeds ─────────────────────────────────────────────────────────────
def get_webcam():   return capture_frame_raw(CAMERA_WEBCAM)
def get_pantilt():  return capture_frame_raw(CAMERA_PANTILT)
def get_eric():     return _eric_says
def get_status():   return _status
def get_log():      return _log_text


# ─── Module status lights ──────────────────────────────────────────────────────

def _dot(active: bool, label: str, detail: str = "") -> str:
    """Single status indicator row: coloured circle + label + optional detail."""
    color  = "#76b900" if active else "#cc2200"
    glow   = f"0 0 8px {color}88" if active else "none"
    state  = "ONLINE" if active else "OFFLINE"
    detail_html = f'<span style="color:#555;font-size:0.72em;margin-left:6px">{detail}</span>' if detail else ""
    return f"""
    <div style="display:flex;align-items:center;gap:10px;padding:4px 0;font-family:'Courier New',monospace">
        <div style="
            width:11px;height:11px;border-radius:50%;
            background:{color};box-shadow:{glow};
            flex-shrink:0;
        "></div>
        <span style="color:#ccc;font-size:0.82em;letter-spacing:0.04em;min-width:110px">{label}</span>
        <span style="color:{color};font-size:0.72em;letter-spacing:0.08em">{state}</span>
        {detail_html}
    </div>"""


def get_module_status_html() -> str:
    """Build the full module status panel with live indicator lights."""

    # ── Cosmos / vLLM ──────────────────────────────────────────────────────
    try:
        import requests
        from config import VLLM_URL
        r = requests.get(VLLM_URL.replace("/v1/chat/completions", "/health"), timeout=1.5)
        cosmos_ok = r.status_code == 200
    except Exception:
        cosmos_ok = False

    # ── Mission active ──────────────────────────────────────────────────────
    try:
        from mission import _ms as _ms_ref
        mission_ok = bool(_ms_ref.mission_active)
    except Exception:
        mission_ok = False

    # ── LiDAR ──────────────────────────────────────────────────────────────
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

    # ── OAK-D ──────────────────────────────────────────────────────────────
    try:
        from oakd import oakd_available, get_front_depth
        oakd_ok = oakd_available()
        oakd_detail = ""
        if oakd_ok:
            d = get_front_depth()
            oakd_detail = f"{d:.2f}m" if d is not None else "no reading"
    except Exception:
        oakd_ok, oakd_detail = False, ""

    # ── Nav2 ───────────────────────────────────────────────────────────────
    try:
        from nav2 import nav2_available, is_navigating
        nav2_ok = nav2_available()
        nav2_detail = "navigating" if (nav2_ok and is_navigating()) else ""
    except Exception:
        nav2_ok, nav2_detail = False, ""

    # ── TTS ────────────────────────────────────────────────────────────────
    try:
        tts_ok = piper_available()
        tts_detail = "Piper" if tts_ok else "gTTS"
    except Exception:
        tts_ok, tts_detail = False, ""

    # ── Motors serial ──────────────────────────────────────────────────────
    try:
        motors_ok = motors._ser is not None and motors._ser.is_open
    except Exception:
        motors_ok = False

    rows = (
        _dot(cosmos_ok,  "COSMOS REASON 2", "vLLM") +
        _dot(mission_ok, "MISSION",         _status) +
        _dot(lidar_ok,   "LIDAR D500",      lidar_detail) +
        _dot(oakd_ok,    "OAK-D LITE",      oakd_detail) +
        _dot(nav2_ok,    "NAV2 / SLAM",     nav2_detail) +
        _dot(tts_ok,     "TTS PIPER",       tts_detail) +
        _dot(motors_ok,  "MOTORS ESP32",    "UART")
    )

    return f"""
    <div style="
        background:#0d0d0d;
        border:1px solid #1e3a1e;
        border-radius:6px;
        padding:12px 14px;
        font-family:'Courier New',monospace;
    ">
        <div style="
            color:#76b900;font-size:0.7em;letter-spacing:0.15em;
            margin-bottom:10px;border-bottom:1px solid #1e3a1e;padding-bottom:6px
        ">SYSTEM STATUS</div>
        {rows}
    </div>"""


# ─── Motor telemetry ──────────────────────────────────────────────────────────

def _motor_telemetry_html(direction: str, left: float, right: float) -> str:
    color = {
        "forward":  "#76b900", "backward": "#ff6600",
        "left":     "#00aaff", "right":    "#00aaff",
        "stopped":  "#444444", "spinning": "#aa00ff",
    }.get(direction, "#444444")

    arrow = {
        "forward":  "▲", "backward": "▼",
        "left":     "◀", "right":    "▶",
        "stopped":  "■", "spinning": "↺",
    }.get(direction, "■")

    speed = max(abs(left), abs(right))
    pct   = int(speed / 0.50 * 100)

    return f"""
    <div style="
        background:#0d0d0d;border:1px solid {color}88;
        border-radius:6px;padding:10px 12px;font-family:'Courier New',monospace;
    ">
        <div style="color:#88d400;font-size:0.82em;font-weight:bold;letter-spacing:0.12em;
                    margin-bottom:8px;border-bottom:1px solid #2d5a2d;padding-bottom:5px">
            DRIVE SYSTEM
        </div>
        <div style="display:flex;align-items:center;gap:14px">
            <div style="font-size:2.2em;color:{color};width:40px;text-align:center">{arrow}</div>
            <div style="flex:1">
                <div style="color:{color};font-size:1.05em;font-weight:bold;letter-spacing:0.1em;text-transform:uppercase">{direction}</div>
                <div style="color:#aaa;font-size:0.85em;margin:4px 0">
                    L <span style="color:#fff;font-weight:bold">{left:+.2f}</span>
                    &nbsp;·&nbsp;
                    R <span style="color:#fff;font-weight:bold">{right:+.2f}</span>
                    &nbsp;·&nbsp;
                    <span style="color:#fff;font-weight:bold">{speed:.2f} m/s</span>
                </div>
                <div style="height:6px;background:#1a1a1a;border-radius:3px;overflow:hidden;margin-top:4px">
                    <div style="height:100%;width:{pct}%;background:{color};border-radius:3px;transition:width 0.3s"></div>
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

def _sensor_panel(label: str, body_html: str, border_color: str = "#2d5a2d") -> str:
    return f"""
    <div style="background:#0d0d0d;border:1px solid {border_color};border-radius:6px;
                padding:10px 12px;font-family:'Courier New',monospace;margin-bottom:6px">
        <div style="color:#88d400;font-size:0.82em;font-weight:bold;letter-spacing:0.12em;
                    margin-bottom:7px;border-bottom:1px solid {border_color};padding-bottom:5px">
            {label}
        </div>
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
                color, state = "#cc2200", "🚧 STOP ZONE"
            elif s.get("obstacle_near"):
                color, state = "#ff6600", "⚠ CAUTION"
            else:
                color, state = "#76b900", "✓ CLEAR"
            body = f"""
            <div style="display:flex;justify-content:space-between;align-items:center">
                <span style="color:{color};font-size:0.85em;letter-spacing:0.08em">{state}</span>
                <span style="color:#fff;font-size:1.1em">{dist}</span>
            </div>
            <div style="color:#444;font-size:0.68em;margin-top:4px">
                STOP &lt;{s.get('obstacle_close') and '0.30' or '0.30'}m · SLOW &lt;0.60m
            </div>"""
            return _sensor_panel("LIDAR D500 — FRONT ARC", body, color + "44")
    except Exception:
        pass
    return _sensor_panel("LIDAR D500", '<span style="color:#333;font-size:0.8em">NOT CONNECTED</span>')


def get_oakd_html() -> str:
    try:
        from oakd import oakd_available, get_front_depth
        if oakd_available():
            d = get_front_depth()
            if d is None:
                body = '<span style="color:#555;font-size:0.8em">NO READING</span>'
                color = "#333"
            elif d < 0.30:
                color = "#cc2200"
                body  = f'<span style="color:{color};font-size:0.85em">🚧 VERY CLOSE</span> <span style="color:#fff">{d:.2f}m</span>'
            elif d < 0.60:
                color = "#ff6600"
                body  = f'<span style="color:{color};font-size:0.85em">⚠ CLOSE</span> <span style="color:#fff">{d:.2f}m</span>'
            else:
                color = "#76b900"
                body  = f'<span style="color:{color};font-size:0.85em">✓ CLEAR</span> <span style="color:#fff">{d:.2f}m</span>'
            return _sensor_panel("OAK-D DEPTH — CENTRE FWD", body, color + "44")
    except Exception:
        pass
    return _sensor_panel("OAK-D LITE", '<span style="color:#333;font-size:0.8em">NOT CONNECTED</span>')


# ─── Mission helpers ──────────────────────────────────────────────────────────

def load_mission_choices():
    missions = list_missions()
    return missions if missions else ["(no missions found)"]


_selected_mission_name = ""   # tracks which YAML was selected (for start_mission)


def on_mission_select(name: str):
    global _selected_mission_name
    if not name or name == "(no missions found)":
        _selected_mission_name = ""
        return ""
    _selected_mission_name = name
    briefing = get_briefing_from_file(name)
    return briefing or ""


def _default_mission_text():
    """Try to load template.yaml at startup, fall back to built-in default."""
    try:
        briefing = get_briefing_from_file("template")
        if briefing and briefing.strip():
            return briefing.strip()
    except Exception:
        pass
    return _TEMPLATE_MISSION.strip()


def _default_mission_choice():
    """Return 'template' if it exists in missions list, else first choice."""
    choices = load_mission_choices()
    if "template" in choices:
        return "template"
    return choices[0] if choices else None


# ─── Actions ──────────────────────────────────────────────────────────────────

def action_engage(briefing: str):
    if not briefing.strip():
        return "⚠ Enter a mission briefing first."
    resp = start_mission(briefing.strip(), mission_name=_selected_mission_name)
    return resp


def action_stop():
    """
    Layer 0 emergency halt — bypasses all software layers.
    Opens the ESP32 serial port directly (same method as test_layer0.py)
    and sends a raw zero-velocity command byte-by-byte.
    Also kills the mission and the motors object as belt-and-suspenders.
    """
    # ── Belt and suspenders: kill mission + motors object ──────────────
    try:
        stop_mission()
    except Exception:
        pass
    try:
        motors.stop()
    except Exception:
        pass

    # ── Layer 0: raw serial direct to ESP32 ────────────────────────────
    # Mirrors test_layer0.py exactly: byte-by-byte, 1ms inter-byte delay,
    # no flow control — the only method proven reliable on JetPack 6.2.
    layer0_result = "⚠ Layer 0 serial not reached"
    try:
        import serial as _serial
        import glob as _glob
        import json as _json

        BAUD = 115200
        # Port priority: ttyTHS2 (Jetson HW UART) → ttyUSB* → ttyACM*
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
            layer0_result = f"✅ Layer 0 stop sent → {port}"
            log.warning(f"🛑 LAYER 0 E-HALT fired via {port}")
        else:
            layer0_result = "❌ Layer 0: no serial port found"
            log.error("🛑 LAYER 0 E-HALT: no port found")

    except Exception as e:
        layer0_result = f"❌ Layer 0 error: {e}"
        log.error(f"🛑 LAYER 0 E-HALT error: {e}")

    return f"🛑 E-HALT — ALL STOP\n{layer0_result}"


def action_introduce():
    resp = ask_cosmos(
        "Introduce yourself to the world. NVIDIA judges are watching. "
        "Cover: full name and acronym, hardware, cost, builder, location, "
        "mission, why edge AI matters. Be proud, warm, bold. 15-20 sentences.",
        max_tokens=500
    )
    threading.Thread(target=lambda: speak(resp), daemon=True).start()
    return resp


def action_look():
    from cosmos import capture_frame
    image = capture_frame(CAMERA_WEBCAM)
    if not image:
        return "❌ Camera unavailable."
    resp = ask_cosmos(
        "Describe in detail what you see. Terrain, objects, hazards. What would you do next?",
        image_b64=image,
        max_tokens=200
    )
    threading.Thread(target=lambda: speak(resp), daemon=True).start()
    return resp


def action_char_reply(char_name: str, char_says: str):
    if not char_name.strip() or not char_says.strip():
        return "⚠ Enter character name and what they say.", ""
    resp = handle_character_response(char_name.strip(), char_says.strip())
    resume_after_interaction()
    return resp, ""


def action_status():
    history = "\n".join(
        f"  • {e['character']}: {e['said'][:60]}"
        for e in get_conversation_history()[-5:]
    ) or "  (none yet)"
    sensor_lines = []
    try:
        from lidar import lidar_available, get_status as ls
        if lidar_available():
            s = ls()
            d = s.get("min_distance", 999)
            sensor_lines.append(f"  LiDAR front: {d:.2f}m" if d < 999 else "  LiDAR front: clear")
    except Exception:
        pass
    try:
        from oakd import oakd_available, get_front_depth
        if oakd_available():
            d = get_front_depth()
            sensor_lines.append(f"  OAK-D front: {d:.2f}m" if d is not None else "  OAK-D front: no reading")
    except Exception:
        pass
    try:
        from nav2 import nav2_available, is_navigating, get_pose
        if nav2_available():
            p = get_pose()
            sensor_lines.append(f"  Nav2: navigating={is_navigating()} pose=({p['x']:.2f},{p['y']:.2f})")
    except Exception:
        pass
    sensors = "\n".join(sensor_lines) or "  (sensors not enabled)"
    return (
        f"State:  {get_mission_state()}\n"
        f"Active: {get_mission_active()}\n"
        f"TTS:    {'Piper streaming' if piper_available() else 'gTTS fallback'}\n"
        f"\nSensors:\n{sensors}"
        f"\nRecent conversations:\n{history}"
    )


# ─── CSS & layout constants ───────────────────────────────────────────────────

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

/* ── Global reset ──────────────────────────────────────────────────── */
body, .gradio-container {
    background: #080b08 !important;
    font-family: 'Share Tech Mono', 'Courier New', monospace !important;
    background-image:
        linear-gradient(rgba(45,90,45,0.12) 1px, transparent 1px),
        linear-gradient(90deg, rgba(45,90,45,0.12) 1px, transparent 1px);
    background-size: 40px 40px;
}
footer { display:none !important; }

/* Subtle scanline */
body::after {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg,
        rgba(0,255,0,0.015) 0px,
        rgba(0,255,0,0.015) 1px,
        transparent 1px,
        transparent 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* ── Header ─────────────────────────────────────────────────────────── */
.eric-header {
    display: flex;
    align-items: baseline;
    gap: 18px;
    padding: 10px 0 4px 0;
    border-bottom: 1px solid #2d5a2d;
    margin-bottom: 10px;
}
.eric-title {
    font-size: 1.6em;
    font-weight: bold;
    letter-spacing: 0.2em;
    color: #88d400;
    font-family: 'Share Tech Mono', 'Courier New', monospace;
    text-shadow: 0 0 10px #88d40055;
}
.eric-sub {
    font-size: 0.82em;
    color: #7aaa7a;
    letter-spacing: 0.05em;
    font-family: 'Share Tech Mono', 'Courier New', monospace;
}

/* ── STOP button — round red, sits beside joystick ──────────────────── */
#stop-btn {
    width: 72px !important;
    height: 72px !important;
    min-width: 72px !important;
    border-radius: 50% !important;
    background: radial-gradient(circle at 38% 35%, #ff3333, #880000) !important;
    border: 3px solid #cc0000 !important;
    color: #fff !important;
    font-size: 0.78em !important;
    font-weight: bold !important;
    letter-spacing: 0.15em !important;
    font-family: 'Share Tech Mono', 'Courier New', monospace !important;
    box-shadow: 0 0 18px #cc000066, inset 0 2px 4px #ff666644 !important;
    transition: box-shadow 0.15s, transform 0.1s !important;
    line-height: 1.2 !important;
    padding: 0 !important;
}
#stop-btn:hover {
    box-shadow: 0 0 32px #ff0000aa, inset 0 2px 4px #ff666644 !important;
    transform: scale(1.04) !important;
}
#stop-btn:active {
    transform: scale(0.97) !important;
    box-shadow: 0 0 12px #cc000088 !important;
}

/* ── ENGAGE button ───────────────────────────────────────────────────── */
#engage-btn {
    background: linear-gradient(135deg, #3a6600, #76b900) !important;
    border: 1px solid #76b900 !important;
    color: #000 !important;
    letter-spacing: 0.12em !important;
    font-family: 'Share Tech Mono', 'Courier New', monospace !important;
    font-size: 0.95em !important;
    font-weight: bold !important;
}
#engage-btn:hover {
    box-shadow: 0 0 14px #76b90066 !important;
}

/* ── Textboxes ───────────────────────────────────────────────────────── */
textarea, input[type=text] {
    background: #0d130d !important;
    border: 1px solid #2d5a2d !important;
    color: #d8f0d8 !important;
    font-family: 'Share Tech Mono', 'Courier New', monospace !important;
    font-size: 0.95em !important;
    border-radius: 4px !important;
}
textarea:focus, input[type=text]:focus {
    border-color: #76b900 !important;
    box-shadow: 0 0 8px #76b90033 !important;
}

/* ── Labels ──────────────────────────────────────────────────────────── */
label span, .gr-label {
    color: #7aaa7a !important;
    font-size: 0.85em !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-family: 'Share Tech Mono', 'Courier New', monospace !important;
}

/* ── Eric Says speech panel ──────────────────────────────────────────── */
#eric-says textarea {
    border-left: 3px solid #76b900 !important;
    color: #eefff0 !important;
    font-size: 1.05em !important;
    font-weight: bold !important;
    min-height: 90px !important;
}

/* ── Log panel ───────────────────────────────────────────────────────── */
#sys-log textarea {
    color: #90c890 !important;
    font-size: 0.88em !important;
    min-height: 110px !important;
}

/* ── Manual control buttons — compact squares ────────────────────────── */
.ctrl-btn button {
    background: #0d1a0d !important;
    border: 1px solid #2d5a2d !important;
    color: #88d400 !important;
    font-family: 'Share Tech Mono', 'Courier New', monospace !important;
    font-size: 1.1em !important;
    font-weight: bold !important;
    border-radius: 4px !important;
    width: 44px !important;
    height: 44px !important;
    min-width: 44px !important;
    padding: 0 !important;
}
.ctrl-btn button:hover {
    border-color: #76b900 !important;
    background: #1a3300 !important;
}
.ctrl-stop button {
    border-color: #883300 !important;
    color: #ff8844 !important;
}
.ctrl-stop button:hover {
    background: #1a0a00 !important;
    border-color: #ff6600 !important;
}
.ctrl-wide button {
    width: 96px !important;
    height: 44px !important;
    min-width: 96px !important;
    font-size: 0.82em !important;
}

/* ── Accordion ───────────────────────────────────────────────────────── */
.gr-accordion {
    background: #0a0f0a !important;
    border: 1px solid #2d5a2d !important;
    border-radius: 4px !important;
}

/* ── Camera images ───────────────────────────────────────────────────── */
.cam-label {
    color: #7aaa7a;
    font-size: 0.78em;
    font-weight: bold;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-family: 'Share Tech Mono', 'Courier New', monospace;
    margin-bottom: 1px;
    padding-left: 2px;
}

/* ── Section headers ─────────────────────────────────────────────────── */
.panel-head {
    color: #88d400;
    font-size: 0.78em;
    font-weight: bold;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-family: 'Share Tech Mono', 'Courier New', monospace;
    border-bottom: 1px solid #2d5a2d;
    padding-bottom: 4px;
    margin-bottom: 6px;
    margin-top: 8px;
}
"""

_HEADER_HTML = """
<div class="eric-header">
    <span class="eric-title">E.R.I.C.</span>
    <span class="eric-sub">
        EDGE ROBOTICS INNOVATION BY COSMOS &nbsp;&#x2502;&nbsp;
        NVIDIA COSMOS COOKOFF 2026 &nbsp;&#x2502;&nbsp;
        JETSON ORIN NANO SUPER 8GB &nbsp;&#x2502;&nbsp;
        WAVESHARE UGV BEAST &nbsp;&#x2502;&nbsp;
        VANCOUVER BC &nbsp;&#x2502;&nbsp; ~$750 CAD
    </span>
</div>
"""


# ─── Build UI ─────────────────────────────────────────────────────────────────

def build_ui():
    default_text    = _default_mission_text()
    default_mission = _default_mission_choice()

    with gr.Blocks(title="ERIC — Mission Control") as demo:

        gr.HTML(_HEADER_HTML)

        with gr.Row(equal_height=False):

            # ═══════════════════════════════════════════════════════════════
            # LEFT COLUMN — cameras + module status + sensors
            # ═══════════════════════════════════════════════════════════════
            with gr.Column(scale=2, min_width=200):

                gr.HTML('<div class="cam-label">📡 PAN-TILT</div>')
                pantilt_img = gr.Image(
                    streaming=True, height=180, label="",
                    show_label=False
                )

                gr.HTML('<div class="cam-label" style="margin-top:4px">🔬 WEBCAM</div>')
                webcam_img = gr.Image(
                    streaming=True, height=140, label="",
                    show_label=False
                )

                # Module status — moved here from right column
                gr.HTML('<div class="panel-head" style="margin-top:6px">MODULE STATUS</div>')
                module_status = gr.HTML(value=get_module_status_html())

                # Sensors below module status
                gr.HTML('<div class="panel-head">SENSORS</div>')
                lidar_display = gr.HTML(value=get_lidar_html())
                oakd_display  = gr.HTML(value=get_oakd_html())

            # ═══════════════════════════════════════════════════════════════
            # CENTRE COLUMN — mission briefing + comms + log
            # ═══════════════════════════════════════════════════════════════
            with gr.Column(scale=3, min_width=320):

                gr.HTML('<div class="panel-head">MISSION BRIEFING</div>')
                with gr.Row():
                    mission_dd = gr.Dropdown(
                        choices=load_mission_choices(),
                        value=default_mission,
                        label="",
                        show_label=False,
                        scale=3
                    )
                    refresh_btn = gr.Button("↺", scale=0, min_width=40)

                briefing_box = gr.Textbox(
                    value=default_text,
                    label="",
                    show_label=False,
                    lines=8,
                    max_lines=14,
                    placeholder="Mission briefing…"
                )

                engage_btn = gr.Button(
                    "▶  ENGAGE MISSION",
                    elem_id="engage-btn",
                    variant="primary"
                )

                gr.HTML('<div class="panel-head">ERIC — TRANSMISSION</div>')
                eric_says_box = gr.Textbox(
                    value="",
                    label="",
                    show_label=False,
                    interactive=False,
                    lines=4,
                    max_lines=6,
                    elem_id="eric-says",
                    placeholder="Awaiting transmission…"
                )

                gr.HTML('<div class="panel-head">CHARACTER COMMS</div>')
                gr.HTML('<span style="color:#2a4a2a;font-size:0.72em;font-family:\'Courier New\',monospace">WHEN ERIC STOPS — TYPE AS THE CHARACTER</span>')
                with gr.Row():
                    char_name = gr.Textbox(
                        placeholder="Character name…",
                        label="", show_label=False, scale=1
                    )
                    char_says = gr.Textbox(
                        placeholder="What they say…",
                        label="", show_label=False, scale=3
                    )
                char_btn   = gr.Button("⟶  TRANSMIT", variant="secondary")
                char_reply = gr.Textbox(
                    label="", show_label=False,
                    interactive=False, lines=3,
                    placeholder="Eric responds…"
                )

                gr.HTML('<div class="panel-head">SYSTEM LOG</div>')
                log_box = gr.Textbox(
                    value="",
                    label="",
                    show_label=False,
                    interactive=False,
                    lines=7,
                    max_lines=12,
                    elem_id="sys-log",
                    placeholder="System events, nav checks, scan results…"
                )

            # ═══════════════════════════════════════════════════════════════
            # RIGHT COLUMN — joystick + STOP (top), drive gauge, utilities
            # ═══════════════════════════════════════════════════════════════
            with gr.Column(scale=3, min_width=280):

                gr.HTML('<div class="panel-head">MANUAL OVERRIDE</div>')

                # ── Joystick + STOP side by side at top ───────────────────
                with gr.Row(equal_height=True):

                    # Joystick d-pad (left side)
                    with gr.Column(scale=3, min_width=0):
                        speed_slider = gr.Slider(
                            minimum=0.05, maximum=0.50, value=0.20, step=0.05,
                            label="Speed (m/s)"
                        )
                        with gr.Row():
                            gr.HTML('<div style="flex:1"></div>')
                            btn_fwd = gr.Button("▲", min_width=44, elem_classes=["ctrl-btn"])
                            gr.HTML('<div style="flex:1"></div>')
                        with gr.Row():
                            btn_left  = gr.Button("◀", min_width=44, elem_classes=["ctrl-btn"])
                            btn_halt  = gr.Button("■", min_width=44, elem_classes=["ctrl-btn", "ctrl-stop"])
                            btn_right = gr.Button("▶", min_width=44, elem_classes=["ctrl-btn"])
                        with gr.Row():
                            gr.HTML('<div style="flex:1"></div>')
                            btn_back = gr.Button("▼", min_width=44, elem_classes=["ctrl-btn"])
                            gr.HTML('<div style="flex:1"></div>')
                        with gr.Row():
                            btn_spin_l = gr.Button("↺ L", min_width=44, elem_classes=["ctrl-btn", "ctrl-wide"])
                            btn_spin_r = gr.Button("↻ R", min_width=44, elem_classes=["ctrl-btn", "ctrl-wide"])

                    # STOP button (right side, vertically centred)
                    with gr.Column(scale=1, min_width=0):
                        gr.HTML('<div style="height:20px"></div>')
                        stop_btn = gr.Button("STOP", elem_id="stop-btn")
                        gr.HTML(
                            '<div style="color:#cc2200;font-size:0.6em;letter-spacing:0.12em;'
                            'text-align:center;margin-top:4px;font-family:\'Courier New\',monospace">'
                            'E-HALT</div>'
                        )

                # Motor status text
                motor_status = gr.Textbox(
                    label="", show_label=False,
                    interactive=False, max_lines=1,
                    value="STOPPED",
                    placeholder="Motor status"
                )

                # ── Drive system gauge — directly below joystick ──────────
                gr.HTML('<div class="panel-head">DRIVE SYSTEM</div>')
                motor_display = gr.HTML(value=_motor_telemetry_html("stopped", 0.0, 0.0))

                # ── Utilities ─────────────────────────────────────────────
                with gr.Accordion("UTILITIES", open=False):
                    util_out = gr.Textbox(label="", show_label=False, lines=5, interactive=False)
                    with gr.Row():
                        gr.Button("INTRODUCE", elem_classes=["ctrl-btn", "ctrl-wide"]).click(
                            action_introduce, outputs=util_out
                        )
                        gr.Button("LOOK",      elem_classes=["ctrl-btn", "ctrl-wide"]).click(
                            action_look, outputs=util_out
                        )
                        gr.Button("STATUS",    elem_classes=["ctrl-btn", "ctrl-wide"]).click(
                            action_status, outputs=util_out
                        )

        # ── Event wiring ───────────────────────────────────────────────────

        mission_dd.change(on_mission_select, inputs=mission_dd, outputs=briefing_box)
        refresh_btn.click(
            lambda: gr.update(choices=load_mission_choices()),
            outputs=mission_dd
        )

        stop_btn.click(action_stop, outputs=eric_says_box)
        engage_btn.click(action_engage, inputs=briefing_box, outputs=eric_says_box)

        char_btn.click(
            action_char_reply,
            inputs=[char_name, char_says],
            outputs=[char_reply, char_says]
        )

        # Manual drive
        btn_fwd.click(
            lambda s: (motors.forward(s),   f"▲ FWD {s:.2f}m/s")[1],
            inputs=speed_slider, outputs=motor_status
        )
        btn_back.click(
            lambda s: (motors.backward(s),  f"▼ REV {s:.2f}m/s")[1],
            inputs=speed_slider, outputs=motor_status
        )
        btn_left.click(
            lambda s: (motors.left(s),      f"◀ LEFT {s:.2f}m/s")[1],
            inputs=speed_slider, outputs=motor_status
        )
        btn_right.click(
            lambda s: (motors.right(s),     f"▶ RIGHT {s:.2f}m/s")[1],
            inputs=speed_slider, outputs=motor_status
        )
        btn_halt.click(
            lambda: (motors.stop(), "■ STOPPED")[1],
            outputs=motor_status
        )
        btn_spin_l.click(
            lambda s: (motors._send(-s, s), f"↺ SPIN L {s:.2f}m/s")[1],
            inputs=speed_slider, outputs=motor_status
        )
        btn_spin_r.click(
            lambda s: (motors._send(s, -s), f"↻ SPIN R {s:.2f}m/s")[1],
            inputs=speed_slider, outputs=motor_status
        )

        # ── Live polling timers ────────────────────────────────────────────
        gr.Timer(1.0).tick(get_webcam,             outputs=webcam_img)
        gr.Timer(1.0).tick(get_pantilt,            outputs=pantilt_img)
        gr.Timer(1.0).tick(get_eric,               outputs=eric_says_box)
        gr.Timer(1.5).tick(get_log,                outputs=log_box)
        gr.Timer(0.5).tick(get_motor_telemetry,    outputs=motor_display)
        gr.Timer(1.0).tick(get_lidar_html,         outputs=lidar_display)
        gr.Timer(1.0).tick(get_oakd_html,          outputs=oakd_display)
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
        theme=gr.themes.Base(primary_hue="green", neutral_hue="green"),
        css=_CSS,
    )
