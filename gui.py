"""
ERIC — Gradio GUI
Dual camera feeds, mission briefing, character interaction
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
    list_missions, get_briefing_from_file,
    mission_active, mission_state, conversation_history
)

log = logging.getLogger("eric.gui")

# ─── Shared state updated by mission callbacks ────────────────────────────────
_eric_says = "ERIC ready."
_status    = "🔴 IDLE"
_log_text  = ""


def _motor_telemetry_html(direction: str, left: float, right: float) -> str:
    """Generate HTML motor telemetry display."""
    color = {
        "forward":  "#76b900",
        "backward": "#ff6600",
        "left":     "#00aaff",
        "right":    "#00aaff",
        "stopped":  "#666666",
        "spinning": "#aa00ff",
    }.get(direction, "#666666")

    arrow = {
        "forward":  "▲",
        "backward": "▼",
        "left":     "◀",
        "right":    "▶",
        "stopped":  "■",
        "spinning": "↺",
    }.get(direction, "■")

    speed = max(abs(left), abs(right))

    return f"""
    <div style="
        background:#1a1a1a;
        border:1px solid {color};
        border-radius:8px;
        padding:12px;
        font-family:monospace;
    ">
        <div style="display:flex;align-items:center;gap:16px">
            <div style="
                font-size:2.5em;
                color:{color};
                width:48px;
                text-align:center;
            ">{arrow}</div>
            <div>
                <div style="color:{color};font-size:1.1em;font-weight:bold;text-transform:uppercase">
                    {direction}
                </div>
                <div style="color:#aaa;font-size:0.85em;margin-top:2px">
                    Speed: <span style="color:#fff">{speed:.2f} m/s</span>
                </div>
                <div style="color:#aaa;font-size:0.85em">
                    L: <span style="color:#fff">{left:+.2f}</span>
                    &nbsp;|&nbsp;
                    R: <span style="color:#fff">{right:+.2f}</span>
                </div>
            </div>
            <div style="flex:1">
                <div style="
                    height:8px;
                    background:#333;
                    border-radius:4px;
                    overflow:hidden;
                ">
                    <div style="
                        height:100%;
                        width:{int(speed / 0.50 * 100)}%;
                        background:{color};
                        border-radius:4px;
                        transition:width 0.3s;
                    "></div>
                </div>
                <div style="color:#555;font-size:0.75em;margin-top:2px">
                    0.0 ──────────────── 0.5 m/s
                </div>
            </div>
        </div>
    </div>
    """


# Track current motor state for telemetry display
_motor_state = {"direction": "stopped", "left": 0.0, "right": 0.0}


def get_motor_telemetry():
    return _motor_telemetry_html(
        _motor_state["direction"],
        _motor_state["left"],
        _motor_state["right"]
    )


def _set_eric_says(t): global _eric_says; _eric_says = t
def _set_status(t):    global _status;    _status    = t
def _append_log(t):
    global _log_text
    lines = (_log_text + "\n" + t).strip().split("\n")
    _log_text = "\n".join(lines[-60:])


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


# ─── Mission file helpers ─────────────────────────────────────────────────────
def load_mission_choices():
    missions = list_missions()
    return missions if missions else ["(no missions found)"]


def on_mission_select(name: str):
    """Load briefing text when user selects a mission from dropdown."""
    if not name or name == "(no missions found)":
        return ""
    briefing = get_briefing_from_file(name)
    return briefing or ""


# ─── Actions ──────────────────────────────────────────────────────────────────
def action_engage(briefing: str):
    if not briefing.strip():
        return "⚠️ Enter a mission briefing first.", _status
    resp = start_mission(briefing.strip())
    return resp, _status


def action_disengage():
    stop_mission()
    return "Mission stopped.", _status


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
        return "⚠️ Enter character name and what they say.", ""
    resp = handle_character_response(char_name.strip(), char_says.strip())
    resume_after_interaction()
    return resp, ""


def action_status():
    history = "\n".join(
        f"  • {e['character']}: {e['said'][:60]}"
        for e in conversation_history[-5:]
    ) or "  (none yet)"
    return (
        f"State:  {mission_state}\n"
        f"Active: {mission_active}\n"
        f"TTS:    {'Piper streaming' if piper_available() else 'gTTS fallback'}\n"
        f"\nRecent conversations:\n{history}"
    )


# ─── Build UI ─────────────────────────────────────────────────────────────────
def build_ui():
    with gr.Blocks(
        title="ERIC — Edge Robotics Innovation by Cosmos",
    ) as demo:

        gr.HTML('<div class="title">🤖 ERIC — Edge Robotics Innovation by Cosmos</div>')
        gr.HTML('<div class="sub">NVIDIA Cosmos Cookoff 2026 · Jetson Orin Nano Super 8GB · Waveshare UGV Beast w/Tracked Wheels· Kelowna Canada BC · ~$750 CAD</div>')
        gr.HTML("""<style>
            body { background:#111; }
            .title { font-size:1.4em; font-weight:bold; color:#76b900; }
            .sub   { color:#888; font-size:0.85em; margin-bottom:12px; }
            .says  { border-left:3px solid #76b900; padding-left:8px; font-size:1.05em; }
            footer { display:none !important; }
            #estop { background:#cc0000 !important; color:white !important;
                     font-size:1.2em !important; font-weight:bold !important;
                     height:52px !important; border:2px solid #ff4444 !important;
                     letter-spacing:2px; margin-bottom:8px; }
            #estop:hover { background:#ff0000 !important; box-shadow:0 0 12px #ff0000 !important; }
        </style>""")
        estop_btn = gr.Button("🚨  EMERGENCY STOP  🚨", variant="stop", elem_id="estop")

        with gr.Row():

            # ── LEFT: Camera feeds ───────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 📷 Pan-Tilt Camera")
                pantilt_img = gr.Image(streaming=True, height=240, label="Pan-Tilt")
                gr.Markdown("### 📷 Webcam — Navigation")
                webcam_img  = gr.Image(streaming=True, height=240, label="Webcam")

                gr.HTML("<hr style='border-color:#333;margin:6px 0'>")
                gr.Markdown("### 🚗 Motor Telemetry")
                motor_display = gr.HTML(value=_motor_telemetry_html("stopped", 0.0, 0.0))

            # ── RIGHT: Control panel ─────────────────────────────────────────
            with gr.Column(scale=1):

                gr.Markdown("### 📋 Mission")
                with gr.Row():
                    mission_dd = gr.Dropdown(
                        choices=load_mission_choices(),
                        label="Load from file",
                        scale=2
                    )
                    refresh_btn = gr.Button("🔄", scale=0, min_width=48)

                briefing_box = gr.Textbox(
                    placeholder=(
                        "Type your mission briefing here — or select a file above to load one.\n\n"
                        "Example: Princess Leia has been captured. R2-D2 may know her location. "
                        "You cannot engage in combat. Seek Luke Skywalker's help."
                    ),
                    label="Mission Briefing",
                    lines=5
                )

                with gr.Row():
                    engage_btn    = gr.Button("🚀 ENGAGE",    variant="primary", scale=2)
                    disengage_btn = gr.Button("🛑 DISENGAGE", variant="stop",    scale=1)

                gr.HTML("<hr style='border-color:#333;margin:6px 0'>")

                status_box    = gr.Textbox(value=_status,    label="Status",      interactive=False, max_lines=1)
                eric_says_box = gr.Textbox(value=_eric_says, label="🔊 Eric Says", interactive=False, lines=4, elem_classes=["says"])

                gr.HTML("<hr style='border-color:#333;margin:6px 0'>")

                gr.Markdown("### 💬 Character Interaction")
                gr.Markdown("*When Eric stops — type as the character below*")
                with gr.Row():
                    char_name = gr.Textbox(placeholder="Character (e.g. R2-D2)", label="Character", scale=1)
                    char_says = gr.Textbox(placeholder="What they say...",        label="They say",  scale=3)
                char_btn   = gr.Button("📨 Send as Character", variant="secondary")
                char_reply = gr.Textbox(label="Eric responds", interactive=False, lines=3)

                with gr.Accordion("🕹️ Manual Controls", open=True):
                    speed_slider = gr.Slider(
                        minimum=0.05, maximum=0.50, value=0.25, step=0.05,
                        label="Speed (m/s)",
                        info="Slow=0.05 · Normal=0.25 · Fast=0.50"
                    )
                    with gr.Row():
                        btn_left   = gr.Button("◀️ Left",    scale=1)
                        btn_fwd    = gr.Button("▲ Forward",  scale=2, variant="primary")
                        btn_right  = gr.Button("▶️ Right",   scale=1)
                    with gr.Row():
                        btn_back   = gr.Button("▼ Backward", scale=1)
                        btn_stop   = gr.Button("⏹️ STOP",    scale=2, variant="stop")
                        btn_spin_l = gr.Button("🔄 Spin L",  scale=1)
                    motor_status = gr.Textbox(
                        label="Motor Status", interactive=False,
                        max_lines=1, value="Stopped"
                    )

                with gr.Accordion("🛠️ Utilities", open=False):
                    with gr.Row():
                        gr.Button("🎤 Introduce").click(action_introduce, outputs=gr.Textbox(label="Output", lines=5, interactive=False))
                        gr.Button("👀 Look").click(action_look, outputs=gr.Textbox(label="Output", lines=5, interactive=False))
                        gr.Button("📊 Status").click(action_status, outputs=gr.Textbox(label="Output", lines=5, interactive=False))

                with gr.Accordion("📜 Mission Log", open=False):
                    log_box = gr.Textbox(label="", lines=10, interactive=False)

        # ── Event wiring ──────────────────────────────────────────────────────
        mission_dd.change(on_mission_select, inputs=mission_dd, outputs=briefing_box)
        refresh_btn.click(lambda: gr.update(choices=load_mission_choices()), outputs=mission_dd)

        estop_btn.click(
            lambda: (stop_mission(), motors.stop(), "🚨 EMERGENCY STOP — All systems halted")[2],
            outputs=eric_says_box
        )

        engage_btn.click(action_engage, inputs=briefing_box, outputs=[eric_says_box, status_box])
        disengage_btn.click(action_disengage, outputs=[eric_says_box, status_box])
        char_btn.click(action_char_reply, inputs=[char_name, char_says], outputs=[char_reply, char_says])

        btn_fwd.click(   lambda s: (motors.forward(s),           f"▲ Forward  {s} m/s")[1],  inputs=speed_slider, outputs=motor_status)
        btn_back.click(  lambda s: (motors.backward(s),          f"▼ Backward {s} m/s")[1],  inputs=speed_slider, outputs=motor_status)
        btn_left.click(  lambda s: (motors.left(s),              f"◀️ Left     {s} m/s")[1],  inputs=speed_slider, outputs=motor_status)
        btn_right.click( lambda s: (motors.right(s),             f"▶️ Right    {s} m/s")[1],  inputs=speed_slider, outputs=motor_status)
        btn_stop.click(  lambda:   (motors.stop(),               "⏹️ Stopped")[1],                                  outputs=motor_status)
        btn_spin_l.click(lambda s: (motors._send(-s, s),         f"🔄 Spin L  {s} m/s")[1],  inputs=speed_slider, outputs=motor_status)

        # ── Live updates using gr.Timer (Gradio 6.0+) ─────────────────────────
        gr.Timer(1.0).tick(get_webcam,          outputs=webcam_img)
        gr.Timer(1.0).tick(get_pantilt,         outputs=pantilt_img)
        gr.Timer(1.0).tick(get_eric,            outputs=eric_says_box)
        gr.Timer(1.0).tick(get_status,          outputs=status_box)
        gr.Timer(2.0).tick(get_log,             outputs=log_box)
        gr.Timer(0.5).tick(get_motor_telemetry, outputs=motor_display)

    return demo


def launch():
    init_tts()
    demo = build_ui()
    log.info(f"🌐 Gradio UI → http://{GRADIO_HOST}:{GRADIO_PORT}")
    demo.launch(
        server_name=GRADIO_HOST,
        server_port=GRADIO_PORT,
        theme=gr.themes.Base(primary_hue="green"),
        css="""
            body { background:#111; }
            .title { font-size:1.4em; font-weight:bold; color:#76b900; }
            .sub   { color:#888; font-size:0.85em; margin-bottom:12px; }
            .says  { border-left:3px solid #76b900; padding-left:8px; font-size:1.05em; }
            footer { display:none !important; }
        """,
        show_error=True,
        quiet=False
    )