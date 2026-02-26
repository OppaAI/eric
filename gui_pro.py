import logging
import gradio as gr

from config import GRADIO_HOST, GRADIO_PORT
from motors import motors
from tts import init_tts
from mission import (
    start_mission, stop_mission, list_missions,
    get_briefing_from_file, get_mission_active, get_mission_state, _ms
)

log = logging.getLogger("eric.gui")

# --- SPENCER'S ONE-SCREEN TACTICAL CSS ---
SPENCER_CSS = """
.gradio-container { 
    background-color: #0d1117 !important; 
    color: #c9d1d9 !important; 
    font-family: 'Courier New', monospace !important; 
    max-height: 100vh !important;
    overflow: hidden !important;
}

/* THE STOP BUTTON: Circle Force */
#stop_btn_container {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
}
#stop_btn { 
    background: linear-gradient(180deg, #ff4444 0%, #cc0000 100%) !important; 
    color: white !important;
    font-weight: bold !important;
    border: 3px solid #30363d !important;
    width: 90px !important;
    height: 90px !important;
    min-width: 90px !important;
    max-width: 90px !important;
    border-radius: 50% !important;
    box-shadow: 0 0 15px rgba(255, 0, 0, 0.4) !important;
    cursor: pointer !important;
}

/* Chat & Briefing Panels */
#eric_says_box textarea { font-size: 1.1em !important; color: #fee100 !important; background: #1c2128 !important; }
#briefing_box textarea { background: #161b22 !important; color: #8b949e !important; }

/* Constrain images to fit on one screen */
.preview-image { max-height: 200px !important; object-fit: contain !important; }
footer { display: none !important; }
"""

def get_module_status_html():
    active = get_mission_active()
    state = get_mission_state()
    # High-glow LEDs based on state
    if not active:
        glow, shadow, label = "#444", "none", "OFFLINE"
    elif "APPROACH" in state.name:
        glow, shadow, label = "#fee100", "0 0 15px #fee100", f"LOCKED: {state.name}"
    else:
        glow, shadow, label = "#76b900", "0 0 15px #76b900", state.name

    return f"""
    <div style="padding:10px; border:1px solid #333; background:#161b22; display:flex; align-items:center; border-radius:4px;">
        <div style="width:14px; height:14px; border-radius:50%; background:{glow}; box-shadow:{shadow}; margin-right:12px;"></div>
        <div style="line-height:1;">
            <span style="font-size:0.6rem; color:#8b949e; text-transform:uppercase; display:block;">Mission Status</span>
            <span style="font-size:0.85rem; font-weight:bold; color:#c9d1d9;">{label}</span>
        </div>
    </div>
    """

def get_lidar_html():
    try:
        dist = getattr(_ms, 'last_lidar_dist', 0.0)
        pct = min(100, int((dist / 2.0) * 100))
        color = "#ff4444" if dist < 0.5 else "#76b900"
        return f"""
        <div style="font-size:0.7rem; color:#8b949e; margin-bottom:4px;">PROXIMITY: {dist:.2f}m</div>
        <div style="width:100%; background:#333; height:8px; border-radius:4px;">
            <div style="width:{pct}%; background:{color}; height:100%; border-radius:4px; transition:width 0.3s;"></div>
        </div>
        """
    except: return ""

def build_ui():
    with gr.Blocks(title="ERIC HUD v2", css=SPENCER_CSS) as demo:
        # HEADER ROW
        with gr.Row():
            with gr.Column(scale=4):
                gr.HTML("<h1 style='color:#76b900; margin:0; letter-spacing:1px;'>🤖 ERIC COMMAND HUD</h1>")
            with gr.Column(scale=2):
                status_led = gr.HTML()
            with gr.Column(scale=1, elem_id="stop_btn_container"):
                btn_stop = gr.Button("STOP", elem_id="stop_btn")

        # MAIN HUD GRID
        with gr.Row():
            # LEFT: Sensor Feeds
            with gr.Column(scale=2):
                # We use placeholder functions for images since we're using your existing capture logic
                cam_p = gr.Image(label="PAN-TILT", height=200)
                cam_w = gr.Image(label="WEBCAM", height=200)
                lidar_ui = gr.HTML()

            # CENTER: AI Reasoning
            with gr.Column(scale=3):
                eric_says = gr.Textbox(label="COSMOS REASONING", elem_id="eric_says_box", lines=15)
                log_output = gr.Textbox(label="L1 SYSTEM LOG", lines=3)

            # RIGHT: Mission Briefing
            with gr.Column(scale=3):
                mission_select = gr.Dropdown(choices=list_missions(), label="SELECT PROTOCOL")
                mission_brief = gr.Textbox(label="INTRODUCTION / PARAMETERS", elem_id="briefing_box", lines=15)
                btn_start = gr.Button("⚡ INITIATE MISSION", variant="primary")

        # Callbacks
        mission_select.change(get_briefing_from_file, inputs=mission_select, outputs=mission_brief)
        btn_start.click(start_mission, inputs=mission_brief)
        btn_stop.click(stop_mission)

        # Telemetry Polling
        gr.Timer(1.0).tick(get_module_status_html, outputs=status_led)
        gr.Timer(0.5).tick(get_lidar_html, outputs=lidar_ui)
        gr.Timer(1.0).tick(lambda: getattr(_ms, 'last_speech', "Ready..."), outputs=eric_says)

    return demo

def launch():
    init_tts()
    ui = build_ui()
    ui.launch(server_name=GRADIO_HOST, server_port=GRADIO_PORT, share=False, prevent_thread_lock=True)

if __name__ == "__main__":
    launch()
