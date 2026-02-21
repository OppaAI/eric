"""
E.R.I.C. — Gradio GUI
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
_eric_says = "E.R.I.C. ready."
_status    = "🔴 IDLE"
_log_text  = ""


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


def action_fwd():   motors.forward(); return "▶️ Forward"
def action_stop():  motors.stop();    return "⏹️ Stop"
def action_left():  motors.left();    return "◀️ Left"
def action_right(): motors.right();   return "▶️ Right"


# ─── Build UI ─────────────────────────────────────────────────────────────────
def build_ui():
    with gr.Blocks(
        title="E.R.I.C. — Edge Robotics Innovation by Cosmos",
        theme=gr.themes.Base(primary_hue="green"),
        css="""
            body { background:#111; }
            .title { font-size:1.4em; font-weight:bold; color:#76b900; }
            .sub   { color:#888; font-size:0.85em; margin-bottom:12px; }
            .says  { border-left:3px solid #76b900; padding-left:8px; font-size:1.05em; }
            footer { display:none !important; }
        """
    ) as demo:

        gr.HTML('<div class="title">🤖 E.R.I.C. — Edge Robotics Innovation by Cosmos</div>')
        gr.HTML('<div class="sub">NVIDIA Cosmos Cookoff 2026 · Jetson Orin Nano Super 8GB · Kelowna BC · ~$750 CAD</div>')

        with gr.Row():

            # ── LEFT: Camera feeds ───────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 📷 Pan-Tilt Camera")
                pantilt_img = gr.Image(streaming=True, height=240, label="Pan-Tilt")
                gr.Markdown("### 📷 Webcam — Navigation")
                webcam_img  = gr.Image(streaming=True, height=240, label="Webcam")

            # ── RIGHT: Control panel ─────────────────────────────────────────
            with gr.Column(scale=1):

                # ── Mission selection + briefing (TOP) ───────────────────────
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

                # Status + Eric says
                status_box    = gr.Textbox(value=_status,    label="Status",     interactive=False, max_lines=1)
                eric_says_box = gr.Textbox(value=_eric_says, label="🔊 Eric Says", interactive=False, lines=4, elem_classes=["says"])

                gr.HTML("<hr style='border-color:#333;margin:6px 0'>")

                # ── Character interaction ─────────────────────────────────────
                gr.Markdown("### 💬 Character Interaction")
                gr.Markdown("*When Eric stops — type as the character below*")
                with gr.Row():
                    char_name = gr.Textbox(placeholder="Character (e.g. R2-D2)", label="Character", scale=1)
                    char_says = gr.Textbox(placeholder="What they say...",        label="They say",  scale=3)
                char_btn   = gr.Button("📨 Send as Character", variant="secondary")
                char_reply = gr.Textbox(label="Eric responds", interactive=False, lines=3)

                # ── Manual controls ───────────────────────────────────────────
                with gr.Accordion("🕹️ Manual Controls", open=False):
                    with gr.Row():
                        gr.Button("◀️ Left").click(action_left,  outputs=status_box)
                        gr.Button("▶️ Fwd").click(action_fwd,   outputs=status_box)
                        gr.Button("▶️ Right").click(action_right, outputs=status_box)
                    gr.Button("⏹️ Stop", variant="stop").click(action_stop, outputs=status_box)

                # ── Utilities ─────────────────────────────────────────────────
                with gr.Accordion("🛠️ Utilities", open=False):
                    with gr.Row():
                        gr.Button("🎤 Introduce").click(action_introduce, outputs=gr.Textbox(label="Output", lines=5, interactive=False))
                        gr.Button("👀 Look").click(action_look, outputs=gr.Textbox(label="Output", lines=5, interactive=False))
                        gr.Button("📊 Status").click(action_status, outputs=gr.Textbox(label="Output", lines=5, interactive=False))

                # ── Mission log ───────────────────────────────────────────────
                with gr.Accordion("📜 Mission Log", open=False):
                    log_box = gr.Textbox(label="", lines=10, interactive=False)

        # ── Event wiring ──────────────────────────────────────────────────────
        mission_dd.change(on_mission_select,  inputs=mission_dd, outputs=briefing_box)
        refresh_btn.click(lambda: gr.update(choices=load_mission_choices()), outputs=mission_dd)

        engage_btn.click(action_engage, inputs=briefing_box, outputs=[eric_says_box, status_box])
        disengage_btn.click(action_disengage,                outputs=[eric_says_box, status_box])
        char_btn.click(action_char_reply, inputs=[char_name, char_says], outputs=[char_reply, char_says])

        # Live updates every second
        demo.load(get_webcam,  outputs=webcam_img,    every=1)
        demo.load(get_pantilt, outputs=pantilt_img,   every=1)
        demo.load(get_eric,    outputs=eric_says_box, every=1)
        demo.load(get_status,  outputs=status_box,    every=1)
        demo.load(get_log,     outputs=log_box,       every=2)

    return demo


def launch():
    init_tts()
    demo = build_ui()
    log.info(f"🌐 Gradio UI → http://{GRADIO_HOST}:{GRADIO_PORT}")
    demo.launch(server_name=GRADIO_HOST, server_port=GRADIO_PORT,
                show_error=True, quiet=False)
