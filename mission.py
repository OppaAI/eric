"""
E.R.I.C. — Edge Robotics Innovation by Cosmos
===============================================
NVIDIA Cosmos Cookoff 2026

Stack:
  Cosmos Reason 2 (vLLM)  — vision + physical reasoning
  Piper via RealtimeTTS   — streaming TTS, CPU only, zero VRAM
  gTTS                    — fallback TTS
  Waveshare UGV           — tracked robot via serial UART to ESP32
  Gradio                  — dual camera GUI + mission control

Quick start:
  bash launch/cosmos.sh     # start vLLM (wait ~3 min)
  uv run main.py            # start Eric
  open http://JETSON_IP:7860
"""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger("eric")


def main():
    log.info("🤖 E.R.I.C. starting — Edge Robotics Innovation by Cosmos")

    # Quick Cosmos connectivity check
    from cosmos import ask_cosmos
    test = ask_cosmos("Say exactly: E.R.I.C. online.", max_tokens=20)
    log.info(f"Cosmos: {test}")

    # Launch Gradio (blocking)
    from gui import launch
    launch()


if __name__ == "__main__":
    main()
