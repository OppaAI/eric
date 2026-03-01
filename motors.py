"""
ERIC — Motor Control
Waveshare UGV via serial UART to ESP32
Command format: {"T":1,"L":speed,"R":speed}  speed in m/s

UART sharing fix:
  Previously, both motors.py and teleop.py's battery poller opened
  /dev/ttyTHS1 independently. odom.py also tried to read from it.
  Three processes competing for one serial port causes byte theft,
  JSON parse errors, and missed battery or odometry packets.

  Fix: Motors owns the ONE open serial port. A background reader thread
  reads every incoming JSON packet and routes it by T-type to registered
  subscriber queues. Any module that needs UART data calls:

    motors.subscribe_uart(type_id, queue)   # receive packets of that T type
    motors.unsubscribe_uart(type_id)

  Battery subscriber (teleop.py):  subscribe_uart_catchall()
  Odom subscriber (odom.py):       T=1001 (wheel speed feedback)
  IMU subscriber (future):         T=1003

  The byte-by-byte write with 1ms delay is kept — it fixes JetPack 6.2
  UART buffer corruption on the Jetson ttyTHS1.
"""
import json
import queue
import time
import threading
import logging
from config import SERIAL_PORT, SERIAL_BAUD, MOTOR_SPEED_SLOW, MOTOR_SPEED_NORMAL, MOTOR_SPEED_FAST

log = logging.getLogger("eric.motors")


class Motors:
    """
    Controls Waveshare UGV tracked robot via serial UART.
    Owns the single open /dev/ttyTHS1 port.
    Routes incoming JSON packets to registered subscribers by T-type.
    Gracefully simulates if serial port unavailable.
    """

    def __init__(self):
        self._ser           = None
        self._write_lock    = threading.Lock()
        self._current_left  = 0.0
        self._current_right = 0.0

        # UART packet router — {T_value: queue.Queue}
        self._subscribers      = {}   # int → queue.Queue
        self._subscriber_lock  = threading.Lock()

        # Catch-all queue — receives every packet not matched by T-type.
        # Battery voltage packets use this (T value varies by firmware version).
        self._catchall_queue   = None

        self._connect()
        self._start_reader()

    # ── Serial connection ──────────────────────────────────────────────────────

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

    # ── UART packet router ─────────────────────────────────────────────────────

    def subscribe_uart(self, type_id: int, q: queue.Queue):
        """
        Register a queue to receive all incoming UART packets with T==type_id.
        Packets are delivered as parsed dicts via put_nowait (non-blocking —
        if the queue is full the oldest item is dropped, so subscribers must
        drain regularly). Thread-safe.

        Example:
            odom_q = queue.Queue(maxsize=10)
            motors.subscribe_uart(1001, odom_q)
            # odom_q.get() will return {"T":1001,"L":0.3,"R":0.3,...}
        """
        with self._subscriber_lock:
            self._subscribers[type_id] = q
        log.debug(f"UART subscriber registered for T={type_id}")

    def unsubscribe_uart(self, type_id: int):
        """Remove a T-type subscriber."""
        with self._subscriber_lock:
            self._subscribers.pop(type_id, None)

    def subscribe_uart_catchall(self, q: queue.Queue):
        """
        Register a queue to receive all packets whose T-type has no specific
        subscriber. Battery voltage packets use this — their T value varies
        across Waveshare firmware versions so matching by type is unreliable.
        Only one catch-all queue is supported.
        """
        self._catchall_queue = q

    def _start_reader(self):
        """Start the background UART reader/router thread."""
        if self._ser is None:
            return   # simulation mode — nothing to read
        t = threading.Thread(
            target=self._reader_loop,
            daemon=True,
            name="motors-uart-reader"
        )
        t.start()
        log.info("📡 Motors UART reader/router started")

    def _reader_loop(self):
        """
        Read incoming JSON lines from ESP32 and route to subscribers.
        Handles newline-framed JSON — accumulates bytes until \n then parses.
        On serial error: logs once, waits 1s, continues — never crashes.
        """
        buf = b""
        while True:
            try:
                if not self._ser or not self._ser.is_open:
                    time.sleep(0.5)
                    continue

                n = self._ser.in_waiting
                if n == 0:
                    time.sleep(0.01)   # 100 Hz idle poll
                    continue

                buf += self._ser.read(n)

                # Process all complete newline-terminated packets
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.replace(b"\x00", b"").strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line.decode("utf-8", errors="replace"))
                        self._route_packet(data)
                    except json.JSONDecodeError:
                        pass   # firmware debug output — ignore silently

                # Discard accumulation if no newline after 512 bytes
                if len(buf) > 512:
                    buf = buf[-256:]

            except Exception as e:
                log.debug(f"UART reader error: {e}")
                time.sleep(1.0)

    def _route_packet(self, data: dict):
        """Deliver a parsed packet to the appropriate subscriber queue."""
        t_val = data.get("T")

        with self._subscriber_lock:
            q = self._subscribers.get(t_val)

        if q is not None:
            try:
                q.put_nowait(data)
            except queue.Full:
                pass   # subscriber not draining — drop packet silently
        elif self._catchall_queue is not None:
            try:
                self._catchall_queue.put_nowait(data)
            except queue.Full:
                pass

    # ── Serial write ───────────────────────────────────────────────────────────

    def _write(self, cmd: str):
        """Send a command string byte by byte with 1ms inter-byte delay.
        Required for JetPack 6.2 / ttyTHS1 UART buffer stability."""
        if not self._ser:
            log.info(f"[SIM] {cmd.strip()}")
            return
        with self._write_lock:
            try:
                for byte in cmd.encode("utf-8"):
                    self._ser.write(bytes([byte]))
                    time.sleep(0.001)
            except Exception as e:
                log.error(f"Serial write error: {e}")

    def _send_raw(self, data: dict):
        """Send any arbitrary JSON command to ESP32."""
        self._write(json.dumps(data) + "\n")

    def _send(self, left: float, right: float):
        self._current_left  = left
        self._current_right = right

        self._write(json.dumps({"T": 1, "L": round(left, 3), "R": round(right, 3)}) + "\n")

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

        try:
            from gui import _motor_state
            _motor_state["left"]      = round(left, 3)
            _motor_state["right"]     = round(right, 3)
            _motor_state["direction"] = direction
        except ImportError:
            pass

        try:
            from logger import log_action as _log_action
            _log_action("MOTOR", f"{direction} L={left:.3f} R={right:.3f}")
        except Exception:
            pass

    # ── High-level motor commands ──────────────────────────────────────────────

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

    # negative = forward on UGV Beast hardware
    def backward(self, speed=MOTOR_SPEED_NORMAL):  self._send( speed,  speed)
    def forward(self, speed=MOTOR_SPEED_NORMAL):   self._send(-speed, -speed)
    def left(self, speed=MOTOR_SPEED_SLOW):        self._send(-speed,  speed)
    def right(self, speed=MOTOR_SPEED_SLOW):       self._send( speed, -speed)
    def stop(self):                                self._send(0.0, 0.0)
    def fast(self):                                self.forward(MOTOR_SPEED_FAST)

    def slow(self):
        """
        Reduce speed if already driving forward — never command movement from rest.
        Called by LiDAR safety monitor when obstacle enters SLOW_DIST zone.
        """
        if self._current_left < 0 and self._current_right < 0:
            self.forward(MOTOR_SPEED_SLOW)


motors = Motors()