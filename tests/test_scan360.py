"""
ERIC — 360° Video Scan Test
============================
Standalone script to:
  1. Fix pan-tilt at a low angle (looking slightly down at floor level)
  2. Slowly rotate body 360° while capturing video from webcam
  3. Send the full clip to Cosmos Reason 2
  4. Print out everything Cosmos identifies in the scene

Usage:
  python3 scan360_test.py

Tune these constants at the top before running:
  TURN_SPEED      — how fast to spin (m/s)
  TURN_DURATION   — how many seconds for a full 360° (tune on your floor)
  TILT_ANGLE      — pan-tilt tilt angle in degrees (negative = down)
  CAPTURE_FPS     — frames per second during capture
  CAMERA          — which camera index to use (webcam=0, pantilt=1)
"""

import json
import time
import base64
import sys
import serial
import cv2
import numpy as np
import requests

# ─── Tune These ───────────────────────────────────────────────────────────────
TARGET        = "R2D2"  # what Eric is looking for — used in prompt and result highlight
TURN_SPEED    = 0.15    # m/s — slow spin for smooth video
TURN_DURATION = 9.0     # seconds — tune until you get a true 360° on your floor
TILT_ANGLE    = -5     # degrees — negative = tilts down toward floor level
CAPTURE_FPS   = 5       # frames per second during turn (2fps × 9s = ~18 frames)
CAMERA        = 0       # webcam index

# ─── Hardware ─────────────────────────────────────────────────────────────────
SERIAL_PORT  = "/dev/ttyTHS1"
SERIAL_BAUD  = 115200
VLLM_URL     = "http://AIVA:8000/v1/chat/completions"
COSMOS_MODEL = "embedl/Cosmos-Reason2-2B-W4A16"

# ─── Cosmos Prompt ────────────────────────────────────────────────────────────
def build_prompt():
    return f"""
These frames are from a continuous 360-degree video scan of my surroundings.
I am a ground robot slowly rotating in place. The camera is fixed at a low angle,
pointing slightly downward to see objects on the floor.

The frames are in chronological order — they cover a full 360° rotation.
Frame 1 = where I started (north). I rotated clockwise.

I am specifically searching for: {TARGET}

Carefully analyze ALL frames and tell me:
1. Every object, figure, or character you can identify — be specific, name them if possible
2. Whether my search target "{TARGET}" is visible — which frame and direction
3. Which direction each object is from me (north=front, east=right, south=back, west=left)
4. How far away each object looks (close <1m, medium 1-3m, far >3m)

Respond ONLY with valid JSON:
{{
  "target": "{TARGET}",
  "target_found": false,
  "target_direction": "north|east|south|west|unknown",
  "target_distance": "close|medium|far|unknown",
  "target_frame": null,
  "scene_description": "overall description of the environment",
  "objects_found": [
    {{
      "name": "object or character name",
      "description": "what it looks like",
      "direction": "north|east|south|west",
      "distance": "close|medium|far",
      "confidence": "high|medium|low"
    }}
  ],
  "floor_terrain": "carpet|tiles|wood|concrete|mixed",
  "lighting": "bright|normal|dim",
  "notes": "anything else notable"
}}
"""


# ─── Serial / Motor ───────────────────────────────────────────────────────────

def connect_serial():
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1,
                            rtscts=False, xonxoff=False)
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        print(f"✅ Serial connected: {SERIAL_PORT} @ {SERIAL_BAUD}")
        return ser
    except Exception as e:
        print(f"❌ Serial failed: {e}")
        return None


def send_cmd(ser, data: dict):
    if not ser:
        print(f"  [SIM] {data}")
        return
    cmd = json.dumps(data) + "\n"
    for byte in cmd.encode("utf-8"):
        ser.write(bytes([byte]))
        time.sleep(0.001)


def spin(ser, speed=TURN_SPEED):
    """Spin right in place."""
    send_cmd(ser, {"T": 1, "L": -speed, "R": speed})


def stop(ser):
    send_cmd(ser, {"T": 1, "L": 0.0, "R": 0.0})


def pantilt_set(ser, pan=0, tilt=0, speed=30):
    send_cmd(ser, {"T": 133, "X": pan, "Y": tilt, "SPD": speed, "ACC": 10})


def lights(ser, base=0, head=255):
    send_cmd(ser, {"T": 132, "IO4": base, "IO5": head})


# ─── Camera ───────────────────────────────────────────────────────────────────

def open_camera(index=CAMERA, width=640, height=480):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"❌ Camera {index} failed to open")
        return None
    print(f"✅ Camera {index} opened ({width}x{height})")
    return cap


def encode_frame(frame) -> str:
    """Rotate webcam frame (mounted sideways) and encode to base64 JPEG."""
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode("utf-8")


# ─── Capture Video During 360° Turn ──────────────────────────────────────────

def capture_360_clip(ser, cap) -> list[str]:
    """
    Spin 360° while capturing frames continuously.
    Motor spin runs in a separate thread so cap.read() never causes a pause.
    Returns list of base64 JPEG frames.
    """
    import threading

    print(f"\n🎬 Starting 360° turn — {TURN_DURATION}s @ {TURN_SPEED} m/s")
    print(f"   Camera: {CAMERA}, FPS: {CAPTURE_FPS}, Tilt: {TILT_ANGLE}°")
    print(f"   Expected frames: ~{int(TURN_DURATION * CAPTURE_FPS)}")

    frames     = []
    interval   = 1.0 / CAPTURE_FPS
    spinning   = threading.Event()
    spinning.set()

    def keep_spinning():
        """Send spin command repeatedly so motor never times out."""
        while spinning.is_set():
            spin(ser)
            time.sleep(0.2)   # re-send every 200ms to keep motor alive

    # Start motor thread FIRST, then capture immediately
    motor_thread = threading.Thread(target=keep_spinning, daemon=True)
    motor_thread.start()

    end_time    = time.time() + TURN_DURATION
    frame_count = 0

    while time.time() < end_time:
        t0 = time.time()

        cap.grab()  # flush stale buffer
        ret, frame = cap.read()
        if ret:
            b64 = encode_frame(frame)
            frames.append(b64)
            frame_count += 1
            remaining = end_time - time.time()
            print(f"  📸 Frame {frame_count} captured  ({remaining:.1f}s remaining)")
        else:
            print("  ⚠️  Frame read failed")

        elapsed = time.time() - t0
        sleep_t = max(0.0, interval - elapsed)
        if sleep_t > 0:
            time.sleep(sleep_t)

    # Stop motor
    spinning.clear()
    motor_thread.join(timeout=1.0)
    stop(ser)
    time.sleep(0.3)

    print(f"\n✅ Capture complete — {len(frames)} frames")
    return frames


# ─── Cosmos Inference ─────────────────────────────────────────────────────────

def ask_cosmos(frames: list[str]) -> dict:
    """Send all frames to Cosmos Reason 2 and return parsed JSON result."""
    print(f"\n🧠 Sending {len(frames)} frames to Cosmos Reason 2...")
    print(f"   Model: {COSMOS_MODEL}")
    print(f"   URL:   {VLLM_URL}")

    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}}
        for f in frames
    ]
    content.append({"type": "text", "text": build_prompt()})

    payload = {
        "model":    COSMOS_MODEL,
        "messages": [
            {"role": "system", "content": "You are ERIC, a ground robot analyzing your surroundings via a 360° video scan. Be precise and specific about what you see."},
            {"role": "user",   "content": content}
        ],
        "max_tokens":  600,
        "temperature": 0.2,
    }

    try:
        t0 = time.time()
        r  = requests.post(VLLM_URL, json=payload, timeout=180)
        r.raise_for_status()
        elapsed = time.time() - t0
        text = r.json()["choices"][0]["message"]["content"].strip()
        print(f"   ⏱️  Inference time: {elapsed:.1f}s")
        return text
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Cosmos — is vLLM running?")
        print("   Start with: bash launch/cosmos.sh")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Cosmos error: {e}")
        sys.exit(1)


def parse_result(raw: str) -> dict:
    """Parse JSON from Cosmos response."""
    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        s = clean.find("{")
        e = clean.rfind("}") + 1
        if s >= 0 and e > s:
            return json.loads(clean[s:e])
    except Exception:
        pass
    return {"raw_response": raw}


def print_result(result: dict):
    """Pretty print what Cosmos found."""
    print("\n" + "═" * 60)
    print("🔍 COSMOS SCAN RESULT")
    print("═" * 60)

    if "raw_response" in result:
        print("⚠️  Could not parse JSON — raw response:")
        print(result["raw_response"])
        return

    # ── Target found? ──────────────────────────────────────────
    target_found = result.get("target_found", False)
    if target_found:
        print(f"\n🎯🎯🎯 TARGET FOUND: {result.get('target', TARGET)}")
        print(f"   Direction: {result.get('target_direction', '?')}")
        print(f"   Distance:  {result.get('target_distance', '?')}")
        print(f"   Frame:     {result.get('target_frame', '?')}")
    else:
        print(f"\n❌ Target NOT found: {result.get('target', TARGET)}")

    print(f"\n📍 Scene:    {result.get('scene_description', 'N/A')}")
    print(f"🏠 Terrain:  {result.get('floor_terrain', 'N/A')}")
    print(f"💡 Lighting: {result.get('lighting', 'N/A')}")

    objects = result.get("objects_found", [])
    if objects:
        print(f"\n📦 All Objects Found ({len(objects)}):")
        print("─" * 60)
        for i, obj in enumerate(objects, 1):
            conf = obj.get("confidence", "?")
            conf_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(conf, "⚪")
            is_target = TARGET.lower() in obj.get("name", "").lower()
            prefix = "🎯 " if is_target else "   "
            print(f"  {i}. {prefix}{conf_icon} {obj.get('name', 'Unknown')}")
            print(f"       Description: {obj.get('description', '')}")
            print(f"       Direction:   {obj.get('direction', '?')}")
            print(f"       Distance:    {obj.get('distance', '?')}")
            print(f"       Confidence:  {conf}")
    else:
        print("\n⚠️  No objects identified")

    notes = result.get("notes")
    if notes:
        print(f"\n📝 Notes: {notes}")

    print("\n" + "═" * 60)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("ERIC — 360° Video Scan Test")
    print("=" * 60)
    print(f"Turn duration:  {TURN_DURATION}s  (tune if not a full 360°)")
    print(f"Turn speed:     {TURN_SPEED} m/s")
    print(f"Tilt angle:     {TILT_ANGLE}°")
    print(f"Capture FPS:    {CAPTURE_FPS}")
    print(f"Camera:         {CAMERA}")
    print()

    # Connect hardware
    ser = connect_serial()
    cap = open_camera(CAMERA)
    if cap is None:
        sys.exit(1)

    # Set pan-tilt to fixed low angle and leave it there
    print(f"\n📷 Setting pan-tilt to tilt={TILT_ANGLE}° (fixed low angle)...")
    pantilt_set(ser, pan=0, tilt=TILT_ANGLE, speed=30)
    time.sleep(1.5)  # wait for pan-tilt to settle

    # Turn on lights
    lights(ser, base=0, head=255)
    time.sleep(0.3)

    # Countdown
    print("\n⏳ Starting in 3 seconds — place your objects now...")
    for i in range(3, 0, -1):
        print(f"   {i}...")
        time.sleep(1.0)

    # Capture 360° video clip
    frames = capture_360_clip(ser, cap)

    # Turn off lights
    lights(ser, base=0, head=0)

    if not frames:
        print("❌ No frames captured — check camera")
        sys.exit(1)

    # Send to Cosmos
    raw = ask_cosmos(frames)

    # Parse and print
    result = parse_result(raw)
    print_result(result)

    # Save raw response for debugging
    with open("scan360_result.json", "w") as f:
        json.dump({"frames_captured": len(frames), "cosmos_raw": raw, "parsed": result}, f, indent=2)
    print(f"\n💾 Full result saved to scan360_result.json")

    # Cleanup
    cap.release()
    if ser:
        ser.close()


if __name__ == "__main__":
    main()