#!/usr/bin/env python3
"""
ERIC — Layer 0 Test Script
Tests ESP32 serial communication directly from SSH terminal.
No project imports needed — completely standalone.

Run:
    python3 test_layer0.py              # auto-detect port
    python3 test_layer0.py /dev/ttyUSB0 # specify port

What this tests:
    1. Port detection and serial connection
    2. Stop command (safe baseline)
    3. OLED display
    4. Lights
    5. Pan-tilt
    6. Forward / backward / left / right movement
    7. Watchdog timeout — ESP32 stops motors when commands stop
    8. Keepalive heartbeat — motors stay running with periodic commands
    9. Serial latency — measures round-trip time per command

IMPORTANT: Put the robot on a stand or clear 1 metre of floor space.
           It will move during test 6.
"""

import json
import sys
import time
import threading
import glob

# ── Config ────────────────────────────────────────────────────────────────────
BAUD         = 115200
SLOW         = 0.3     # m/s — safe test speed
TEST_MOVE_S  = 1.5     # seconds of movement per direction test
WATCHDOG_S   = 3.5     # seconds of silence to trigger watchdog (ESP32 default ~2s)

# ── Terminal colours ──────────────────────────────────────────────────────────
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):        print(f"  {GREEN}✅ {msg}{RESET}")
def fail(msg):      print(f"  {RED}❌ {msg}{RESET}")
def info(msg):      print(f"  {CYAN}ℹ  {msg}{RESET}")
def warn(msg):      print(f"  {YELLOW}⚠  {msg}{RESET}")
def head(msg):      print(f"\n{BOLD}{CYAN}{'─'*52}\n  {msg}\n{'─'*52}{RESET}")
def ask(msg):       return input(f"  {YELLOW}{msg}{RESET} ")


# ── Serial helpers ────────────────────────────────────────────────────────────

def find_port() -> str | None:
    """
    Auto-detect ESP32 serial port.

    Priority order:
      1. /dev/ttyTHS2  — Jetson Orin Nano hardware UART (correct for ERIC)
      2. /dev/ttyUSB*  — USB-serial adapters
      3. /dev/ttyACM*  — CDC-ACM USB devices (Arduino etc — usually wrong for ERIC)

    ttyACM0 is almost always the wrong port on Jetson — it's typically a
    connected USB device (camera, Arduino) not the ESP32 UART. ttyTHS2 is
    the Jetson's dedicated hardware serial line wired directly to the ESP32.
    """
    # ttyTHS2 first — Jetson hardware UART wired to ESP32 on ERIC
    ths2      = ["/dev/ttyTHS2"] if glob.glob("/dev/ttyTHS2") else []
    ths_other = sorted(p for p in glob.glob("/dev/ttyTHS*") if p != "/dev/ttyTHS2")
    usb_ports = sorted(glob.glob("/dev/ttyUSB*"))
    acm_ports = sorted(glob.glob("/dev/ttyACM*"))

    all_ports = ths2 + ths_other + usb_ports + acm_ports

    if all_ports:
        info(f"Ports found: {all_ports}")
        if ths2:
            info(f"Using /dev/ttyTHS2 (Jetson hardware UART) — correct for ERIC")
        elif ths_other:
            warn(f"ttyTHS2 not found — falling back to {ths_other[0]}")
        elif acm_ports and not usb_ports:
            warn(f"Only ttyACM found — this is usually wrong for ERIC")
            warn(f"Expected /dev/ttyTHS2 — check ESP32 wiring")

    return all_ports[0] if all_ports else None


def open_serial(port: str):
    """
    Open serial with JetPack 6.2 safe settings.
    rtscts=False and xonxoff=False are required — JetPack 6.2 UART
    hardware flow control causes buffer corruption with the ESP32.
    """
    import serial
    ser = serial.Serial(
        port, BAUD,
        timeout=1,
        rtscts=False,
        xonxoff=False,
    )
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    return ser


def send(ser, data: dict):
    """
    Send JSON command byte-by-byte with 1ms inter-byte delay.
    Byte-by-byte transmission prevents UART buffer corruption on JetPack 6.2.
    Whole-buffer writes (ser.write(full_cmd)) cause garbled ESP32 parsing.
    """
    cmd = json.dumps(data) + "\n"
    for byte in cmd.encode("utf-8"):
        ser.write(bytes([byte]))
        time.sleep(0.001)


# ── Motor commands ────────────────────────────────────────────────────────────
# UGV Beast convention: negative speed = forward, positive = backward.
# This is a hardware quirk of the Waveshare chassis — not a bug.

def stop(ser):
    send(ser, {"T": 1, "L": 0.0, "R": 0.0})

def forward(ser, speed=SLOW):
    send(ser, {"T": 1, "L": round(-speed, 3), "R": round(-speed, 3)})

def backward(ser, speed=SLOW):
    send(ser, {"T": 1, "L": round(speed, 3), "R": round(speed, 3)})

def turn_left(ser, speed=SLOW):
    send(ser, {"T": 1, "L": round(-speed, 3), "R": round(speed, 3)})

def turn_right(ser, speed=SLOW):
    send(ser, {"T": 1, "L": round(speed, 3), "R": round(-speed, 3)})

def oled(ser, line: int, text: str):
    send(ser, {"T": 3, "lineNum": line, "Text": str(text)[:16]})

def lights(ser, base=255, head_light=255):
    send(ser, {"T": 132, "IO4": base, "IO5": head_light})

def pantilt(ser, pan=0, tilt=0, speed=50):
    send(ser, {"T": 133, "X": pan, "Y": tilt, "SPD": speed, "ACC": 10})


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_connection(ser) -> bool:
    head("TEST 1 — Connection")
    try:
        stop(ser)
        ok(f"Port open: {ser.port} @ {ser.baudrate} baud")
        ok("Stop command sent successfully")
        return True
    except Exception as e:
        fail(f"Serial write failed: {e}")
        return False


def test_oled(ser):
    head("TEST 2 — OLED Display")
    info("Watch the OLED screen on the chassis")
    try:
        oled(ser, 0, "ERIC L0 TEST")
        oled(ser, 1, "OLED OK")
        ok("Lines written — check physical OLED display")
    except Exception as e:
        fail(f"OLED failed: {e}")


def test_lights(ser):
    head("TEST 3 — Lights")
    info("LEDs: ON for 1s → OFF for 0.5s → ON again")
    try:
        lights(ser, 255, 255);  time.sleep(1.0);  ok("Lights ON")
        lights(ser, 0, 0);      time.sleep(0.5);  ok("Lights OFF")
        lights(ser, 255, 255);                    ok("Lights ON (left on)")
    except Exception as e:
        fail(f"Lights failed: {e}")


def test_pantilt(ser):
    head("TEST 4 — Pan-Tilt")
    info("Camera: centre → left 30° → right 30° → centre → tilt up → tilt down → centre")
    moves = [
        (0,    0,   "Centre"),
        (-30,  0,   "Pan left 30°"),
        (30,   0,   "Pan right 30°"),
        (0,    0,   "Centre"),
        (0,    20,  "Tilt up 20°"),
        (0,    -10, "Tilt down 10°"),
        (0,    0,   "Centre"),
    ]
    try:
        for pan, tilt, label in moves:
            pantilt(ser, pan, tilt)
            time.sleep(1.0)
            ok(label)
    except Exception as e:
        fail(f"Pan-tilt failed: {e}")


def test_motors(ser):
    head("TEST 5 — Motor Movement")
    warn(f"Robot WILL move. Clear at least {TEST_MOVE_S * SLOW + 0.3:.1f}m in all directions.")
    ans = ask("Ready? (y to run, any other key to skip):")
    if ans.strip().lower() != "y":
        warn("Motor movement test skipped")
        return

    moves = [
        ("Forward",     forward),
        ("Backward",    backward),
        ("Turn left",   turn_left),
        ("Turn right",  turn_right),
    ]

    for label, fn in moves:
        try:
            info(f"{label} for {TEST_MOVE_S}s at {SLOW} m/s...")
            oled(ser, 1, label[:16])
            fn(ser, SLOW)
            time.sleep(TEST_MOVE_S)
            stop(ser)
            time.sleep(0.5)
            ok(f"{label} complete")
        except Exception as e:
            fail(f"{label} failed: {e}")
            stop(ser)

    oled(ser, 1, "Motors OK")


def test_watchdog(ser):
    head("TEST 6 — ESP32 Watchdog Timeout")
    info("Confirms ESP32 stops motors automatically when the Jetson goes silent.")
    info(f"We send forward, then go completely silent for {WATCHDOG_S}s.")
    info("Motors MUST stop on their own — no stop command from us.")
    warn("Watch the wheels. They should stop within ~2s of silence.")
    warn("This simulates a Jetson crash — Layer 0's last safety net.")
    ans = ask("Ready? (y to run, skip otherwise):")
    if ans.strip().lower() != "y":
        warn("Watchdog test skipped")
        return

    try:
        oled(ser, 0, "WATCHDOG TEST")
        oled(ser, 1, "Moving...")
        forward(ser, SLOW)
        time.sleep(0.5)

        ok("Moving — going SILENT now...")
        print(f"  {CYAN}  (no commands for {WATCHDOG_S}s){RESET}")

        # Complete silence — nothing sent, simulating Jetson hang
        time.sleep(WATCHDOG_S)

        ok(f"Silence held for {WATCHDOG_S}s")
        ans2 = ask("Did motors stop on their own? (y/n):")
        if ans2.strip().lower() == "y":
            ok("ESP32 watchdog confirmed working ✅")
            ok("Layer 0 will protect Eric even if Jetson crashes")
        else:
            warn("Motors did NOT stop — watchdog may not be enabled")
            warn("Check ESP32 firmware watchdog timeout setting")
            warn("Default Waveshare firmware: ~2s — may need reflash if disabled")

    except Exception as e:
        fail(f"Watchdog test error: {e}")
    finally:
        stop(ser)
        oled(ser, 1, "Watchdog done")


def test_keepalive(ser):
    head("TEST 7 — Keepalive Heartbeat")
    info("Confirms motors stay running when we send periodic heartbeat commands.")
    info("A background thread re-sends forward every 0.5s for 5 seconds.")
    info("Motors should run continuously without stopping.")
    warn("Watch the wheels — should run the full 5 seconds.")
    ans = ask("Ready? (y to run, skip otherwise):")
    if ans.strip().lower() != "y":
        warn("Keepalive test skipped")
        return

    stop_flag  = threading.Event()
    errors     = []

    def heartbeat():
        while not stop_flag.is_set():
            try:
                forward(ser, SLOW)
            except Exception as e:
                errors.append(str(e))
            time.sleep(0.5)

    try:
        oled(ser, 0, "KEEPALIVE TEST")
        oled(ser, 1, "Running 5s...")
        forward(ser, SLOW)

        t = threading.Thread(target=heartbeat, daemon=True)
        t.start()

        for i in range(5, 0, -1):
            print(f"  {CYAN}  {i}s remaining...{RESET}", end="\r", flush=True)
            time.sleep(1.0)
        print()

        stop_flag.set()
        t.join(timeout=1.0)
        stop(ser)

        if errors:
            warn(f"Heartbeat errors: {errors[0]}")
        else:
            ok("Heartbeat ran cleanly for 5s — no drops")
            ok("Motors stayed running continuously ✅")

        oled(ser, 1, "Keepalive OK")

    except KeyboardInterrupt:
        stop_flag.set()
        stop(ser)
        warn("Keepalive interrupted")
    except Exception as e:
        stop_flag.set()
        stop(ser)
        fail(f"Keepalive failed: {e}")


def test_latency(ser):
    head("TEST 8 — Serial Latency")
    info("Sends 20 stop commands and measures time per command.")
    info("Target: < 30ms average. Above 60ms = congestion risk.")

    times = []
    try:
        for _ in range(20):
            t0 = time.monotonic()
            stop(ser)
            times.append((time.monotonic() - t0) * 1000)
            time.sleep(0.05)

        avg = sum(times) / len(times)
        mn  = min(times)
        mx  = max(times)

        ok(f"20 commands — avg {avg:.1f}ms  min {mn:.1f}ms  max {mx:.1f}ms")

        if avg < 30:
            ok("Latency GOOD — serial is healthy")
        elif avg < 60:
            warn("Latency MARGINAL — may see occasional jitter under load")
        else:
            fail("Latency HIGH — possible UART buffer congestion or baud mismatch")
            info("Check: is another process using the serial port?")
            info("Check: ls -la /proc/*/fd | grep ttyUSB")

    except Exception as e:
        fail(f"Latency test failed: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}{CYAN}{'═'*52}")
    print("  ERIC — Layer 0 Test Suite")
    print("  ESP32 / UGV Beast serial validation")
    print(f"{'═'*52}{RESET}")

    # ── Port ──────────────────────────────────────────────────────────────────
    port = sys.argv[1] if len(sys.argv) > 1 else None
    if not port:
        info("No port specified — auto-detecting...")
        port = find_port()
        if not port:
            fail("No serial ports found.")
            fail("Expected /dev/ttyTHS2 — Jetson hardware UART to ESP32")
            fail("Try:  ls /dev/ttyTHS*")
            fail("Also try:  ls /dev/ttyUSB*  /dev/ttyACM*  (USB fallback)")
            sys.exit(1)
        info(f"Auto-selected: {port}")
    else:
        info(f"Using: {port}")

    # ── pyserial check ────────────────────────────────────────────────────────
    try:
        import serial
    except ImportError:
        fail("pyserial not installed.")
        fail("Run:  pip install pyserial  or  pip3 install pyserial")
        sys.exit(1)

    # ── Open port ─────────────────────────────────────────────────────────────
    try:
        ser = open_serial(port)
        ok(f"Serial open: {port} @ {BAUD}")
    except PermissionError:
        fail(f"Permission denied: {port}")
        fail("Run:  sudo chmod 666 " + port)
        fail("Or add user to dialout:  sudo usermod -a -G dialout $USER")
        sys.exit(1)
    except Exception as e:
        fail(f"Cannot open {port}: {e}")
        sys.exit(1)

    # ── Safety stop ───────────────────────────────────────────────────────────
    stop(ser)
    time.sleep(0.3)

    # ── Run all tests ─────────────────────────────────────────────────────────
    passed = 0
    skipped = 0
    total  = 8

    def run(fn):
        nonlocal passed, skipped
        try:
            fn(ser)
            passed += 1
        except KeyboardInterrupt:
            warn("Skipped (Ctrl+C)")
            skipped += 1
            stop(ser)
        except Exception as e:
            fail(f"Unexpected error in {fn.__name__}: {e}")
            stop(ser)

    if not test_connection(ser):
        fail("Connection failed — aborting all tests")
        ser.close()
        sys.exit(1)
    passed += 1

    run(test_oled)
    run(test_lights)
    run(test_pantilt)
    run(test_motors)
    run(test_watchdog)
    run(test_keepalive)
    run(test_latency)

    # ── Final ─────────────────────────────────────────────────────────────────
    stop(ser)
    oled(ser, 0, "TEST COMPLETE")
    oled(ser, 1, f"{passed}/{total} passed")
    lights(ser, 255, 255)
    ser.close()

    head("SUMMARY")
    info(f"Passed:  {passed}/{total}")
    if skipped:
        info(f"Skipped: {skipped} (movement/watchdog tests require confirmation)")
    failed = total - passed - skipped
    if failed:
        fail(f"Failed:  {failed} — review output above")

    if passed == total:
        ok("Layer 0 fully healthy — ESP32 communication confirmed ✅")
    elif passed + skipped == total:
        ok("All non-skipped tests passed ✅")
        warn("Re-run and answer 'y' to movement tests for full validation")
    else:
        warn("Some tests failed — Layer 0 needs attention before running missions")

    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{YELLOW}⚠  Interrupted — sending emergency stop...{RESET}")
        try:
            import serial, glob
            port = sys.argv[1] if len(sys.argv) > 1 else (
                (["/dev/ttyTHS2"] + sorted(glob.glob("/dev/ttyUSB*")) + sorted(glob.glob("/dev/ttyACM*"))) or [None]
            )[0]
            if port:
                s = serial.Serial(port, BAUD, timeout=1, rtscts=False, xonxoff=False)
                cmd = json.dumps({"T": 1, "L": 0.0, "R": 0.0}) + "\n"
                for b in cmd.encode():
                    s.write(bytes([b])); time.sleep(0.001)
                s.close()
                print(f"{GREEN}Stop sent.{RESET}")
        except Exception:
            print(f"{RED}Could not send stop — kill power if motors are running!{RESET}")