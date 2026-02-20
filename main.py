"""
E.R.I.C. — Edge Robotics Innovation by Cosmos
===============================================
NVIDIA Cosmos Cookoff 2026

Stack:
  - Cosmos Reason 2 (vLLM)  : vision + physical reasoning
  - Piper via RealtimeTTS   : streaming TTS, CPU only, zero VRAM
  - gTTS                    : fallback TTS
  - Waveshare UGV           : tracked robot via serial UART → ESP32
  - Gradio                  : dual camera GUI + mission control

Hardware:
  - Jetson Orin Nano Super 8GB
  - ~$750 CAD total cost
  - Kelowna BC Canada

Usage:
  uv run main.py
  # Then open http://JETSON_IP:7860
"""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("eric")


def main():
    log.info("🤖 E.R.I.C. starting — Edge Robotics Innovation by Cosmos")

    # Quick Cosmos connectivity test
    from cosmos import ask_cosmos
    test = ask_cosmos("Say exactly: E.R.I.C. online and ready.", max_tokens=20)
    log.info(f"Cosmos test: {test}")

    # Launch Gradio GUI (blocking)
    from gui import launch
    launch()


if __name__ == "__main__":
    main()
