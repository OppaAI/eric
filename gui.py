"""
E.R.I.C. — Gradio GUI
Dual camera feeds + mission control + character interaction
"""

import time
import threading
import logging

import gradio as gr

from config import GRADIO_HOST, GRADIO_PORT, CAMERA_WEBCAM, CAMERA_PANTILT
from cosmos import capture_frame_raw, ask_cosmos
from motors import motors
from tts import speak, init_tts, piper_available
from mission import (
    start_mission, stop_mission, resume_after_interaction,
    handle_character_response, register_ui_callbacks,
    mission_active, mission_state, conversation_history
)

log = logging.getLogger("eric.gui")

# ─── Shared UI state (updated by mission callbacks) ──────────────────────────
_eric_says_text = "E.R.I.C. ready."
_status_text    = "🔴 IDLE"
_log_text       = ""


def _set_eric_says(text: str):
    global _eric_says_text
    _eric_says_text = text


def _set_status(text: str):
    global _status_text
    _status_text = text


def _append_log(text: str):
    global _log_text
    _log_text = _log_text + "\n" + text
    # Keep last 50 lines
    lines = _log_text.strip().split("\n")
    _log_text = "\n".join(lines[-50:])


register_ui_callbacks(
    eric_says=_set_eric_says,
    status=_set_status,
    log=_append_log
)


# ─── Camera feeds ─────────────────────────────────────────────────────────────

def get_webcam_feed():
    frame = capture_frame_raw(CAMERA_WEBCAM)
    return frame


def get_pantilt_feed():
    frame = capture_frame_raw(CAMERA_PANTILT)
    return frame


# ─── Mission actions ──────────────────────────────────────────────────────────

def action_engage(briefing: str):
    if not briefing.strip():
        return "⚠️ Please enter a mission briefing first.", _status_text
    response = start_mission(briefing.strip())
    return response, _status_text


def action_disengage():
    stop_mission()
    return "Mission stopped.", _status_text


def action_introduce():
    response = ask_cosmos(
        "Introduce yourself to the world. You are about to be seen by NVIDIA judges. "
        "Cover: full name and acronym, hardware, cost, builder, location, mission, "
        "why edge AI matters. Be proud, warm, bold. 15-20 sentences.",
        max_tokens=500
    )
    threading.Thread(target=lambda: speak(response), daemon=True).start()
    return response


def action_look():
    from cosmos import capture_frame, SCAN_PROMPT
    image = capture_frame(CAMERA_WEBCAM)
    if not image:
        return "❌ Camera unavailable."
    response = ask_cosmos(
        "Describe in detail what you see in front of you. "
        "What terrain, objects, potential hazards? What would you do next?",
        image_b64=image,
        max_tokens=200
    )
    threading.Thread(target=lambda: speak(response), daemon=True).start()
    return response


def action_character_reply(character_name: str, character_says: str):
    """User types as a Lego character — Eric responds."""
    if not character_name.strip() or not character_says.strip():
        return "⚠️ Enter character name and what they say.", ""
    response = handle_character_response(
        character_name.strip(),
        character_says.strip()
    )
    resume_after_interaction()
    return response, ""


def action_manual_forward():
    motors.forward()
    return "▶️ Moving forward"


def action_manual_stop():
    motors.stop()
    return "⏹️ Stopped"


def action_manual_left():
    motors.left()
    return "◀️ Turning left"


def action_manual_right():
    motors.right()
    return "▶️ Turning right"


def action_status():
    from mission import mission_state, conversation_history
    history = "\n".join([
        f"  • {e['character']}: {e['said'][:60]}"
        for e in conversation_history[-5:]
    ]) or "  (none yet)"
    return (
        f"State: {mission_state}\n"
        f"Active: {mission_active}\n"
        f"TTS: {'Piper streaming' if piper_available() else 'gTTS fallback'}\n"
        f"\nRecent conversations:\n{history}"
    )


def get_eric_says():
    return _eric_says_text


def get_status():
    return _status_text


def get_log():
    return _log_text


# ─── Build Gradio UI ──────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(
        title="E.R.I.C. — Edge Robotics Innovation by Cosmos",
        theme=gr.themes.Base(primary_hue="green"),
        css="""
            .eric-title { font-size: 1.4em; font-weight: bold; color: #76b900; }
            .eric-says  { font-size: 1.1em; border-left: 3px solid #76b900; padding-left: 8px; }
            footer { display: none !important; }
        """
    ) as demo:

        gr.HTML('<div class="eric-title">🤖 E.R.I.C. — Edge Robotics Innovation by Cosmos</div>')
        gr.HTML('<div style="color:#aaa;font-size:0.85em">NVIDIA Cosmos Cookoff 2026 · Jetson Orin Nano · Kelowna BC · $750 CAD</div>')

        with gr.Row():
            # ── Left: Dual camera feeds ──────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 📷 Pan-Tilt Camera")
                pantilt_feed = gr.Image(
                    label="Pan-Tilt",
                    streaming=True,
                    height=240
                )
                gr.Markdown("### 📷 Webcam (Navigation)")
                webcam_feed = gr.Image(
                    label="Webcam",
                    streaming=True,
                    height=240
                )

            # ── Right: Control panel ─────────────────────────────────────────
            with gr.Column(scale=1):

                # ── MISSION BRIEFING — top, prominent ────────────────────────
                gr.Markdown("### 📋 Mission Briefing")
                briefing_input = gr.Textbox(
                    placeholder=(
                        "Type your mission briefing here before engaging...\n\n"
                        "Example: Princess Leia has been captured and is held in "
                        "the Death Star. R2-D2 may know her exact location — find "
                        "him first. Darth Vader is planning something evil and may "
                        "have taken her. You cannot engage in combat — seek the "
                        "help of Luke Skywalker. Good luck."
                    ),
                    label="",
                    lines=5
                )

                with gr.Row():
                    engage_btn    = gr.Button("🚀 ENGAGE",    variant="primary", scale=2)
                    disengage_btn = gr.Button("🛑 DISENGAGE", variant="stop",    scale=1)

                gr.HTML("<hr style='margin:8px 0; border-color:#333'>")

                # Status bar
                status_box = gr.Textbox(
                    value=_status_text,
                    label="Mission Status",
                    interactive=False,
                    max_lines=1
                )

                # Eric speaks
                gr.Markdown("### 🔊 Eric Says")
                eric_says_box = gr.Textbox(
                    value=_eric_says_text,
                    label="",
                    interactive=False,
                    lines=4,
                    elem_classes=["eric-says"]
                )

                # Character interaction
                gr.Markdown("### 💬 Character Interaction")
                gr.Markdown("*When Eric stops to talk — type as the character*")
                with gr.Row():
                    char_name_input = gr.Textbox(
                        placeholder="Character name (e.g. R2-D2)",
                        label="Character",
                        scale=1
                    )
                    char_says_input = gr.Textbox(
                        placeholder="What the character says...",
                        label="Character says",
                        scale=3
                    )
                char_reply_btn = gr.Button("📨 Send as Character")
                char_response_box = gr.Textbox(
                    label="Eric responds",
                    interactive=False,
                    lines=3
                )

                # Manual controls
                with gr.Accordion("🕹️ Manual Controls", open=False):
                    with gr.Row():
                        gr.Button("◀️ Left").click(action_manual_left, outputs=status_box)
                        gr.Button("▶️ Forward").click(action_manual_forward, outputs=status_box)
                        gr.Button("▶️ Right").click(action_manual_right, outputs=status_box)
                    gr.Button("⏹️ Stop", variant="stop").click(action_manual_stop, outputs=status_box)

                # Utilities
                with gr.Accordion("🛠️ Utilities", open=False):
                    with gr.Row():
                        introduce_btn = gr.Button("🎤 Introduce")
                        look_btn      = gr.Button("👀 Look")
                        status_btn    = gr.Button("📊 Status")
                    util_output = gr.Textbox(label="Output", lines=5, interactive=False)

                # Mission log
                with gr.Accordion("📜 Mission Log", open=False):
                    log_box = gr.Textbox(
                        label="",
                        lines=10,
                        interactive=False,
                        value=""
                    )

        # ── Event handlers ────────────────────────────────────────────────────

        engage_btn.click(
            action_engage,
            inputs=[briefing_input],
            outputs=[eric_says_box, status_box]
        )

        disengage_btn.click(
            action_disengage,
            outputs=[eric_says_box, status_box]
        )

        char_reply_btn.click(
            action_character_reply,
            inputs=[char_name_input, char_says_input],
            outputs=[char_response_box, char_says_input]
        )

        introduce_btn.click(action_introduce, outputs=util_output)
        look_btn.click(action_look, outputs=util_output)
        status_btn.click(action_status, outputs=util_output)

        # ── Live updates every second ─────────────────────────────────────────
        demo.load(
            get_webcam_feed,
            outputs=webcam_feed,
            every=1
        )
        demo.load(
            get_pantilt_feed,
            outputs=pantilt_feed,
            every=1
        )
        demo.load(
            get_eric_says,
            outputs=eric_says_box,
            every=1
        )
        demo.load(
            get_status,
            outputs=status_box,
            every=1
        )
        demo.load(
            get_log,
            outputs=log_box,
            every=2
        )

    return demo


def launch():
    init_tts()
    demo = build_ui()
    log.info(f"🌐 Gradio UI: http://{GRADIO_HOST}:{GRADIO_PORT}")
    demo.launch(
        server_name=GRADIO_HOST,
        server_port=GRADIO_PORT,
        show_error=True,
        quiet=False
    )
