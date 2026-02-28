#!/usr/bin/env python3
"""
ERIC — Battery Test v6

The consistent head-drop ('s":0}' every time, never the '{') means
something is consuming the START of every packet before we read it.

This script:
  1. Checks for processes holding /dev/ttyTHS1 open
  2. Decodes what we DO receive to work out the true packet length
     and how many bytes are being stolen
  3. Uses the known tail to reconstruct the voltage anyway
  4. Provides a get_voltage() function you can paste into teleop.py

Run:  uv run test_battery2.py
      (teleop service should already be stopped)
"""

import json, os, re, subprocess, time, sys, serial

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PORT = os.getenv("SERIAL_PORT", "/dev/ttyTHS1")
BAUD = 115200

print(f"\n{'='*55}")
print(f"  ERIC Battery Test v6  —  {PORT} @ {BAUD}")
print(f"{'='*55}\n")


def open_port(path):
    return serial.Serial(path, BAUD,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=0)


def read_for(port, duration):
    deadline = time.time() + duration
    buf = b""
    while time.time() < deadline:
        n = port.in_waiting
        if n:
            buf += port.read(n)
        else:
            time.sleep(0.002)
    return buf


def frame_payloads(raw, frame=128):
    parts = []
    for i in range(0, len(raw), frame):
        chunk = raw[i:i+frame]
        p = bytes(b for b in chunk if b != 0)
        if p:
            parts.append(p.decode('utf-8', errors='replace'))
    return ''.join(parts)


# ── STEP 1: Who else has the port open? ───────────────────────
print("STEP 1 — Processes with /dev/ttyTHS1 open:")
try:
    result = subprocess.run(
        ["lsof", PORT],
        capture_output=True, text=True)
    lines = result.stdout.strip().splitlines()
    if lines:
        for line in lines:
            print(f"  {line}")
    else:
        print("  (none — port is exclusively ours)")
except Exception as e:
    print(f"  lsof failed: {e}")

print()

# Also check if teleop service is running
try:
    result = subprocess.run(
        ["systemctl", "is-active", "teleop"],
        capture_output=True, text=True)
    status = result.stdout.strip()
    print(f"  teleop service status: {status}")
    if status == "active":
        print("  *** teleop is RUNNING — it is stealing packet headers!")
        print("  *** Run: sudo systemctl stop teleop")
except Exception as e:
    print(f"  systemctl check failed: {e}")

print()

# ── STEP 2: Decode the known tail to get voltage anyway ───────
print("STEP 2 — Extract voltage from tail fragments (workaround)")
print("  We know the MCU sends: {...,\"v\":NNNN}")
print("  We receive the tail:   'NNN..N}'")
print("  So we extract the number directly.\n")

ser = open_port(PORT)
ser.reset_input_buffer()
time.sleep(0.2)

voltages = []
for attempt in range(5):
    ser.reset_input_buffer()
    ser.write(b'{"T":105}\n')
    time.sleep(0.08)
    raw = read_for(ser, 0.5)
    text = frame_payloads(raw)
    compact = ''.join(text.split())
    print(f"  attempt {attempt+1}: {compact!r}")

    # Extract all runs of digits immediately followed by '}'
    # e.g. '1192}' -> 1192
    matches = re.findall(r'(\d+)\}', compact)
    for m in matches:
        val = int(m)
        if 900 <= val <= 1300:  # sane range for 3S LiPo in 10mV units
            volts = val / 100.0
            print(f"    -> voltage fragment {val} -> {volts:.2f} V")
            voltages.append(volts)

print()
if voltages:
    avg = sum(voltages) / len(voltages)
    print(f"  Average voltage from {len(voltages)} readings: {avg:.2f} V")
    print()
    print("  *** VOLTAGE CONFIRMED (from tail fragments) ***")
    print(f"  *** {avg:.2f} V  (raw units / 100) ***")
else:
    print("  No voltage fragments found in range 9.00-13.00 V")

print()

# ── STEP 3: Check what a full uninterrupted packet looks like ─
print("STEP 3 — Sniff passive stream (no command) for 5s")
print("  Looking for the longest consecutive fragment...")
ser.reset_input_buffer()
raw_passive = read_for(ser, 5.0)
text_passive = frame_payloads(raw_passive)
compact_passive = ''.join(text_passive.split())
print(f"  Full passive text: {compact_passive!r}")
print()

# Find longest fragment between whitespace
fragments = [f for f in re.split(r'[\r\n]+', text_passive) if f.strip()]
longest = max(fragments, key=len) if fragments else ""
print(f"  Longest fragment: {longest.strip()!r}")
print(f"  -> This is the TAIL of a JSON packet.")
print(f"  -> The FULL packet probably looks like:")
tail = longest.strip()
# Try to reconstruct: common Waveshare patterns
candidates = [
    f'{{"T":1,"v":{tail}',          # if tail is just the value+brace
    f'{{"T":105,"v":{tail}',
]
for c in candidates:
    try:
        d = json.loads(c)
        print(f"     {c}  -> valid JSON: {d}")
    except Exception:
        pass

ser.close()
print()
print("="*55)
print("  SUMMARY")
print("="*55)
if voltages:
    print(f"  Battery voltage: {sum(voltages)/len(voltages):.2f} V")
    print()
    print("  To use in teleop.py, add this function:")
    print("""
def get_battery_voltage(ser):
    import re
    ser.reset_input_buffer()
    ser.write(b'{\"T\":105}\\n')
    time.sleep(0.08)
    deadline = time.time() + 0.5
    buf = b""
    while time.time() < deadline:
        n = ser.in_waiting
        if n:
            buf += ser.read(n)
        else:
            time.sleep(0.002)
    text = buf.replace(b'\\x00', b'').decode('utf-8', errors='replace')
    matches = re.findall(r'(\\d+)\\}', ''.join(text.split()))
    for m in matches:
        val = int(m)
        if 900 <= val <= 1300:
            return val / 100.0
    return None
""")
else:
    print("  Could not determine voltage.")
    print("  Check that teleop service is stopped: sudo systemctl stop teleop")
print("="*55)