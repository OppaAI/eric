"""
ERIC — Tactical Command Dashboard (Spencer Edition)
Industrial Cyberpunk style with high-contrast telemetry and neon status indicators.
"""

import logging
import gradio as gr

from config import GRADIO_HOST, GRADIO_PORT
from motors import motors
from tts import init_tts
from mission import (
    start_mission, stop_mission, resume_after_interaction,
    handle_character_response, register_ui_callbacks,
    list_missions, get_briefing_from_file,
    get_mission_active, get_mission_state, _ms
)

log = logging.getLogger("eric.gui")

# --- Tactical CSS Overhaul ---
SPENCER_CSS = """
.gradio-container { 
    background-color: #0d1117 !important; 
    color: #c9d1d9 !important; 
    font-family: 'Courier New', Courier, monospace !important; 
}
.box, .form, .gr-panel { 
    border: 1px solid #30363d !important; 
    background: #161b22 !important; 
    border-radius: 4px !important;
}
.label { 
    color: #58a6ff !important; 
    text-transform: uppercase !important; 
    font-weight: bold !important;
    letter-spacing: 1px;
}
#eric_says_box { 
    border-left: 6px solid #fee100 !important; 
    background: #1c2128 !important;
}
#stop_btn { 
    background: linear-gradient(180deg, #ff4444 0%, #cc0000 100%) !important; 
    color: white !important; 
    border: 1px solid #30363d !important;
    font-weight: bold !important;
    font-size: 1.2em !important;
}
footer {display: none !important;}
"""

# --- Telemetry & Status Logic ---

def get_module_status_html():
    """Returns tactical LED-style status indicators with neon glow."""
    active = get_mission_active()
    state = get_mission_state()
    
    # Glow logic: Green for search, Yellow for Pikachu, Red/Gray for Idle
    if not active:
        glow_color = "#444"
        shadow = "none"
        label = "SYSTEM IDLE"
    elif "APPROACH" in state.name:
        glow_color = "#fee100" # Pikachu Yellow
        shadow = "0 0 15px #fee100, 0 0 5px #fff"
        label = f"TARGET LOCKED: {state.name}"
    else:
        glow_color = "#76b900" # Nvidia Green
        shadow = "0 0 15px #76b900, 0 0 5px #fff"
        label = f"MISSION: {state.name}"

    return f"""
    <div style="padding: 15px; background: #0d1117; border: 1px solid #30363d; border-radius: 4px;">
        <div style="display: flex; align-items: center;">
            <div style="width: 14px; height: 14px; border-radius: 50%; background: {glow_color}; box-shadow: {shadow}; margin-right: 12px;"></div>
            <div>
                <span style="color: #8b949e; font-size: 0.7rem; text-transform: uppercase; display: block;">Core Status</span>
                <span style="color: #c9d1d9; font-weight: bold; font-size: 0.9rem;">{label}</span>
            </div>
        </div>
    </div>
    """

def get_lidar_html():
    """Renders a tactical proximity bar."""
    try:
        dist = getattr(_ms, 'last_lidar_dist', 0.0)
        # Convert distance to a percentage for a bar (max 2m)
        pct = min(100, int((dist / 2.0) * 100))
        color = "#ff4444" if dist < 0.5 else "#76b900"
        return f"""
        <div style="color: #8b949e; font-size: 0.7rem; margin-bottom: 4px;">LIDAR PROXIMITY: {dist:.2f}m</div>
        <div style="width: 100%; background: #30363d; height: 10px; border-radius: 5px;">
            <div style="width: {pct}%; background: {color}; height: 100%; border-radius: 5px; transition: width 0.3s;"></div>
        </div>
        """
    except: return "LIDAR OFFLINE"

# --- UI Builders ---

def build_ui():
    with gr.Blocks(title="ERIC COMMAND CENTER", css=SPENCER_CSS) as demo:
        gr.HTML("<h2 style='color: #76b900; margin-bottom: 0;'>🤖 ERIC TACTICAL INTERFACE</h2>")
        gr.Markdown("Local Edge Inference • Industrial S&R Protocol")

        with gr.Row():
            # LEFT: Dual Feeds
            with gr.Column(scale=2):
                cam_p = gr.Image(label="PAN-TILT (PRIMARY)", source="webcam", streaming=True)
                cam_w = gr.Image(label="WEBCAM (FLOOR)")
            
            # CENTRE: Mission Control
            with gr.Column(scale=3):
                eric_says = gr.Textbox(label="ERIC SPEECH (COSMOS REASONING)", elem_id="eric_says_box", lines=4)
                with gr.Row():
                    mission_select = gr.Dropdown(choices=list_missions(), label="SELECT MISSION")
                    btn_start = gr.Button("⚡ INITIATE", variant="primary")
                
                mission_brief = gr.Textbox(label="MISSION PARAMETERS", lines=8)
                log_output = gr.Textbox(label="SYSTEM LOG", lines=6)

            # RIGHT: Analytics & Safety
            with gr.Column(scale=2):
                status_led = gr.HTML(label="MISSION STATUS")
                lidar_ui = gr.HTML(label="SENSORS")
                btn_stop = gr.Button("🛑 EMERGENCY STOP", elem_id="stop_btn")
                
                with gr.Accordion("Manual Overrides", open=False):
                    speed = gr.Slider(0, 1.0, value=0.2, label="Drive Power")
                    with gr.Row():
                        gr.Button("↺").click(lambda s: motors.spin_left(s), inputs=speed)
                        gr.Button("↑").click(lambda s: motors.forward(s), inputs=speed)
                        gr.Button("↻").click(lambda s: motors.spin_right(s), inputs=speed)

        # Wire up events
        mission_select.change(get_briefing_from_file, inputs=mission_select, outputs=mission_brief)
        btn_start.click(start_mission, inputs=mission_brief)
        btn_stop.click(stop_mission)

        # Timers for the "Live" feel
        gr.Timer(1.0).tick(get_module_status_html, outputs=status_led)
        gr.Timer(0.5).tick(get_lidar_html, outputs=lidar_ui)
        # Logic to pull speech from _ms
        gr.Timer(1.0).tick(lambda: getattr(_ms, 'last_speech', ""), outputs=eric_says)

    return demo

def launch():
    init_tts()
    ui = build_ui()
    ui.launch(server_name=GRADIO_HOST, server_port=GRADIO_PORT, share=False, prevent_thread_lock=True)

if __name__ == "__main__":
    launch()
