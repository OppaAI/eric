"""
E.R.I.C. — Motor Control
Waveshare UGV via serial UART → ESP32
"""

import json
import threading
import logging

from config import SERIAL_PORT, SERIAL_BAUD, MOTOR_SPEED_NORMAL, MOTOR_SPEED_SLOW, MOTOR_SPEED_FAST

log = logging.getLogger("eric.motors")


class Motors:
    """
    Sends JSON commands to ESP32 via GPIO UART serial.
    Command format: {"T":1,"L":speed,"R":speed}
    Speed in m/s. Positive = forward, negative = backward, 0 = stop.
    Gracefully simulates if serial unavailable.
    """

    def __init__(self):
        self._ser  = None
        self._lock = threading.Lock()
        self._connect()

    def _connect(self):
        try:
            import serial
            self._ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
            log.info(f"✅ Motors: serial {SERIAL_PORT} @ {SERIAL_BAUD}")
        except Exception as e:
            log.warning(f"⚠️  Motors unavailable ({e}) — running in simulation mode")
            self._ser = None

    def _send(self, left: float, right: float):
        cmd = json.dumps({"T": 1, "L": round(left, 3), "R": round(right, 3)}) + "\n"
        if not self._ser:
            log.info(f"[MOTOR SIM] {cmd.strip()}")
            return
        with self._lock:
            try:
                self._ser.write(cmd.encode("utf-8"))
            except Exception as e:
                log.error(f"Motor send error: {e}")

    def oled(self, line: int, text: str):
        """Display text on ESP32 OLED screen."""
        cmd = json.dumps({"T": 3, "lineNum": line, "Text": text[:16]}) + "\n"
        if not self._ser:
            log.info(f"[OLED SIM] line {line}: {text}")
            return
        with self._lock:
            try:
                self._ser.write(cmd.encode("utf-8"))
            except Exception as e:
                log.error(f"OLED send error: {e}")

    def forward(self, speed=MOTOR_SPEED_NORMAL): self._send(speed, speed)
    def backward(self, speed=MOTOR_SPEED_NORMAL): self._send(-speed, -speed)
    def left(self, speed=MOTOR_SPEED_SLOW):       self._send(-speed, speed)
    def right(self, speed=MOTOR_SPEED_SLOW):      self._send(speed, -speed)
    def stop(self):                                self._send(0, 0)
    def slow(self):                                self.forward(MOTOR_SPEED_SLOW)
    def fast(self):                                self.forward(MOTOR_SPEED_FAST)


motors = Motors()
