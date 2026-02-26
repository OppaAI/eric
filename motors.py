"""
ERIC — Motor Control
Waveshare UGV via serial UART to ESP32
Command format: {"T":1,"L":speed,"R":speed}  speed in m/s

Fixes for JetPack 6.2 UART issues:
- rtscts=False, xonxoff=False to disable flow control
- Byte-by-byte transmission with 1ms delay to prevent buffer corruption
- Motor directions corrected (negative = forward on UGV Beast)
"""
import json
import time
import threading
import logging
from config import SERIAL_PORT, SERIAL_BAUD, MOTOR_SPEED_SLOW, MOTOR_SPEED_NORMAL, MOTOR_SPEED_FAST

log = logging.getLogger("eric.motors")


class Motors:
    """
    Controls Waveshare UGV tracked robot via serial UART.
    Gracefully simulates if serial port unavailable.
    """

    def __init__(self):
        self._ser           = None
        self._lock          = threading.Lock()
        self._current_left  = 0.0   # track last commanded speeds
        self._current_right = 0.0   # so slow() knows if robot is moving
        self._connect()

    def _connect(self):
        try:
            import serial
            self._ser = serial.Serial(
                SERIAL_PORT, SERIAL_BAUD,
                timeout=1,
                rtscts=False,
                xonxoff=False
            )
            self._ser.reset_input_buffer()
            self._ser.reset_output_buffer()
            log.info(f"✅ Motors: {SERIAL_PORT} @ {SERIAL_BAUD}")
        except Exception as e:
            log.warning(f"⚠️  Motors unavailable ({e}) — simulation mode")

    def _write(self, cmd: str):
        """Send a raw command string byte by byte."""
        if not self._ser:
            log.info(f"[SIM] {cmd.strip()}")
            return
        with self._lock:
            try:
                for byte in cmd.encode("utf-8"):
                    self._ser.write(bytes([byte]))
                    time.sleep(0.001)
            except Exception as e:
                log.error(f"Serial write error: {e}")

    def _send_raw(self, data: dict):
        """Send any arbitrary JSON command to ESP32."""
        cmd = json.dumps(data) + "\n"
        self._write(cmd)

    def _send(self, left: float, right: float):
        # Track current speed so slow() can check if we're moving
        self._current_left  = left
        self._current_right = right

        cmd = json.dumps({"T": 1, "L": round(left, 3), "R": round(right, 3)}) + "\n"
        self._write(cmd)

        # Determine direction label
        if left == 0 and right == 0:
            direction = "stopped"
        elif left < 0 and right < 0:
            direction = "forward"
        elif left > 0 and right > 0:
            direction = "backward"
        elif left < 0 and right > 0:
            direction = "left"
        elif left > 0 and right < 0:
            direction = "right"
        else:
            direction = "spinning"

        # Update telemetry state for GUI display
        try:
            from gui import _motor_state
            _motor_state["left"]      = round(left, 3)
            _motor_state["right"]     = round(right, 3)
            _motor_state["direction"] = direction
        except ImportError:
            pass

        # Structured activity log — only log meaningful changes to avoid spam
        try:
            from logger import log_action as _log_action
            _log_action("MOTOR", f"{direction} L={left:.3f} R={right:.3f}")
        except Exception:
            pass

    def oled(self, line: int, text: str):
        """Write text to ESP32 OLED display (max 16 chars per line)."""
        cmd = json.dumps({"T": 3, "lineNum": line, "Text": str(text)[:16]}) + "\n"
        if not self._ser:
            log.info(f"[OLED SIM] line {line}: {text}")
            return
        self._write(cmd)

    def lights(self, base: int = 255, head: int = 255):
        """Control LED lights. Values 0-255."""
        self._send_raw({"T": 132, "IO4": base, "IO5": head})

    def pantilt(self, pan: int = 0, tilt: int = 0, speed: int = 50):
        """Pan-tilt control. pan/tilt in degrees from center."""
        self._send_raw({"T": 133, "X": pan, "Y": tilt, "SPD": speed, "ACC": 10})

    # NOTE: positive speed = forward on UGV Beast hardware (corrected)
    def backward(self, speed=MOTOR_SPEED_NORMAL):  self._send(speed, speed)
    def forward(self, speed=MOTOR_SPEED_NORMAL):   self._send(-speed, -speed)
    def left(self, speed=MOTOR_SPEED_SLOW):        self._send(-speed, speed)   # left track back, right track forward
    def right(self, speed=MOTOR_SPEED_SLOW):       self._send(speed, -speed)   # right track back, left track forward
    def stop(self):                                self._send(0.0, 0.0)
    def fast(self):                                self.forward(MOTOR_SPEED_FAST)

    def slow(self):
        """
        Reduce speed if already driving forward — never command movement from rest.
        Called by LiDAR safety monitor when an obstacle enters the SLOW_DIST zone.
        If the robot is stationary, this is a no-op to prevent unwanted movement.
        """
        # Only apply if currently driving forward (both tracks negative = forward)
        if self._current_left < 0 and self._current_right < 0:
            self.forward(MOTOR_SPEED_SLOW)
        # If stopped, turning, or reversing — do nothing


motors = Motors()