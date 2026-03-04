#!/usr/bin/env python3
"""
narrate.py — Generate MP3 narration using Eric's Piper voice.

Usage:
    uv run narrate.py

Edit SCRIPT below with your narration text, then run.
Output: narration.mp3 in current directory.
"""

import subprocess
import tempfile
import os

# ── Load Piper paths from Eric's config ──────────────────────────────────────
from config import PIPER_BINARY, PIPER_MODEL

# ─────────────────────────────────────────────────────────────────────────────
# PASTE YOUR NARRATION SCRIPT HERE
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT = """
ERIC is a mission-based autonomous robot powered by NVIDIA Cosmos Reason 2,
running fully at the edge on a Jetson Orin Nano Super — no cloud, no internet.

Give it a mission in plain English, press Engage, and it navigates, reasons,
avoids obstacles, identifies targets, and announces its findings —
all driven by a single vision-language model on consumer hardware under one thousand dollars.

In this demo, the operator lies on the floor as the casualty.
Eric engages the Search and Rescue mission, scans the environment,
locates the casualty, fires the siren, and stays on station broadcasting location
until help arrives.

Every decision — where to go, what it sees, whether to approach, what to say —
flows through Cosmos Reason 2.
"""
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_MP3 = "narration.mp3"


def main():
    print(f"Piper binary : {PIPER_BINARY}")
    print(f"Piper model  : {PIPER_MODEL}")
    print(f"Output       : {OUTPUT_MP3}")
    print()

    # Write script to temp text file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(SCRIPT.strip())
        txt_path = f.name

    wav_tmp = txt_path.replace(".txt", ".wav")

    try:
        # Step 1: Piper text → WAV
        print("Running Piper...")
        with open(txt_path) as stdin_f:
            result = subprocess.run(
                [
                    PIPER_BINARY,
                    "--model", PIPER_MODEL,
                    "--output_file", wav_tmp,
                ],
                stdin=stdin_f,
                capture_output=True,
            )
        if result.returncode != 0:
            print(f"Piper error:\n{result.stderr.decode()}")
            return

        # Step 2: WAV → MP3 via ffmpeg
        print("Converting to MP3...")
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", wav_tmp,
                "-codec:a", "libmp3lame",
                "-qscale:a", "2",
                OUTPUT_MP3,
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            print(f"ffmpeg error:\n{result.stderr.decode()}")
            return

        size_kb = os.path.getsize(OUTPUT_MP3) // 1024
        print(f"\n✅ Done — {OUTPUT_MP3} ({size_kb} KB)")
        print("Copy to your editing machine and drop into your video timeline.")

    finally:
        for f in [txt_path, wav_tmp]:
            try:
                os.unlink(f)
            except Exception:
                pass


if __name__ == "__main__":
    main()
