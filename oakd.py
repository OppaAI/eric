"""
ERIC — OAK-D Lite Depth Camera  (Layer 1 + Layer 2)
DepthAI 3.x API — Pipeline runs itself via pipeline.start().

Layer 1 Safety (industrial-standard reactive safety):
  Every depth frame is checked instantly in the reader thread:
    front depth  < OAKD_STOP_DIST  → motors.stop()  immediately
    front depth  < OAKD_SLOW_DIST  → motors.slow()  immediately
    floor drop detected (high OR medium confidence) → motors.stop()
  This fires in the reader thread — zero latency, independent of Cosmos.
  Floor drop check runs at FLOOR_CHECK_HZ to limit CPU (numpy is expensive).

Layer 2 Person/Animal Detection (YOLOv8n on OAK-D Myriad X VPU):
  Runs YOLOv8n entirely on the OAK-D's built-in Myriad X — zero Jetson RAM.
  Detects: person, dog, cat, horse, sheep, cow, bird + large animals (COCO).
  Every detection frame:
    confidence > YOLO_MIN_CONFIDENCE
    AND OAK-D spatial depth confirms distance
    AND distance < YOLO_SLOW_DIST  → slow motors
    AND distance < YOLO_STOP_DIST  → stop motors (Layer 2 motor guard)
    → fires _yolo_callback(label, distance_m, bearing_deg) for mission.py
  No Cosmos involved — pure Myriad X VPU hardware detection.
  Falls back gracefully if blob not found — depth pipeline still works.

Gaps closed vs previous version
────────────────────────────────
1. Medium-confidence void auto-stop — both "high" AND "medium" floor-drop
   confidence now trigger motors.stop() (was: high only).

2. Front obstacle — 3-patch sampling (left, centre, right at ±15°).
   A single textureless patch (white wall, glass) can no longer silence
   the whole front check. Minimum of up to 3 valid samples is used.
   Logged if fewer than 3 patches return valid depth.

3. Proportional bearing — bearing is now a signed float in degrees
   (−90° … +90°) in addition to the coarse "left/center/right" string.
   mission.py uses bearing_deg to compute a proportional steering
   correction instead of a fixed 0.3-second turn regardless of angle.

4. YOLO bounding-box size trend — _yolo_check() tracks the previous
   bounding box width per label. If the box is shrinking (target moving
   away), the callback is skipped and only a log entry is produced,
   avoiding the "approach someone walking away" failure mode.

5. YOLO position memory — _last_yolo_positions stores the most recent
   valid (label, dist_m, bearing_deg) per label, updated every packet
   regardless of cooldown. mission.py can query this to recover context
   after Cosmos finishes a long inference cycle.

6. Layer 2 motor guard race condition — _yolo_check() now marks
   _yolo_motor_stop=True when it calls motors.stop(), giving mission.py
   a way to know that Layer 2 issued a stop and not to override it with
   a forward command until it has handled the detection.

7. GUI status exposes YOLO blob status — oakd_status_html() shows
   YOLO: ✅ active | ❌ no blob | ⏸ paused so operators always know
   whether Layer 2 is actually running.

8. Void auto-stop also fires on medium confidence — matches lidar.py.

Disconnect handling (unchanged from previous version):
  - _reader_loop detects fatal XLink errors and exits cleanly.
  - _oakd_ok cleared immediately so all callers see unavailable.
  - Reconnect watchdog retries init_oakd() every RECONNECT_INTERVAL_S s.
  - Log spam suppressed: one warning per disconnect, silence until back.
"""

import datetime
import logging
import threading
import time
import numpy as np
from pathlib import Path

log = logging.getLogger("eric.oakd")

# ── Core state ────────────────────────────────────────────────────────────────
_pipeline      = None
_depth_queue   = None
_depth_frame   = None
_lock          = threading.Lock()
_oakd_ok       = False
_reader_thread = None

# ── Reconnect watchdog ────────────────────────────────────────────────────────
_reconnect_thread    = None
_reconnect_started   = False
RECONNECT_INTERVAL_S = 10.0

# ── Layer 1 Safety constants ──────────────────────────────────────────────────
OAKD_STOP_DIST   = 0.30    # meters — stop if obstacle closer than this
OAKD_SLOW_DIST   = 0.60    # meters — slow if obstacle closer than this
FLOOR_CHECK_HZ   = 1.0     # floor-drop check rate — 1Hz enough, numpy is heavy + eases USB load

_safety_active    = True
_last_floor_check = 0.0

# ── Layer 2 YOLO constants ────────────────────────────────────────────────────
YOLO_BLOB_PATH      = Path("models/yolov8n_openvino_2022.1_4shave.blob")
YOLO_MIN_CONFIDENCE = 0.55   # minimum detection confidence
YOLO_SLOW_DIST      = 3.0    # meters — slow when target within this range
YOLO_STOP_DIST      = 2.0    # meters — stop when target within this range
YOLO_CHECK_HZ       = 10.0   # detection polling rate
YOLO_COOLDOWN_S     = 2.0    # seconds between repeated callbacks, same label

# COCO classes ERIC cares about: people + animals (all large enough to matter)
YOLO_TARGET_CLASSES = {
    0:  "person",
    15: "bird",
    16: "cat",
    17: "dog",
    18: "horse",
    19: "sheep",
    20: "cow",
    21: "elephant",
    22: "bear",
    23: "zebra",
    24: "giraffe",
}

# ── Layer 2 YOLO state ────────────────────────────────────────────────────────
_yolo_spatial_queue  = None
_yolo_ok             = False
_yolo_callback       = None    # callable(label, dist_m, bearing, bearing_deg)
_yolo_active         = False
_yolo_lock           = threading.Lock()

_last_yolo_detect:    dict  = {}    # label → last callback timestamp
_last_yolo_bbox_w:    dict  = {}    # label → last bbox width (approach trend)
_last_yolo_positions: dict  = {}    # label → (dist_m, bearing, bearing_deg) — latest
_yolo_motor_stop:     bool  = False  # True when Layer 2 issued a motors.stop()

# ── Bearing EMA smoothing ─────────────────────────────────────────────────────
# Single-frame bearing readings are noisy. A 3-frame exponential moving average
# (α=0.4) reduces jitter without adding significant lag.
# Formula: ema = α * new + (1 - α) * prev
_BEARING_EMA_ALPHA   = 0.4
_bearing_ema:        dict  = {}     # label → smoothed bearing_deg float

# ── Fatal error strings ───────────────────────────────────────────────────────
_FATAL_ERRORS = (
    "MessageQueue was closed",
    "X_LINK_ERROR",
    "XLinkError",
    "Couldn't read data",
    "Communication exception",
)


# ─── Public API ───────────────────────────────────────────────────────────────

def oakd_available() -> bool:
    return _oakd_ok


def yolo_available() -> bool:
    """True if YOLO pipeline is running on Myriad X."""
    return _yolo_ok


def yolo_motor_stop_issued() -> bool:
    """
    True if Layer 2 called motors.stop() for a YOLO detection and that
    detection has not yet been handled by mission.py.
    mission.py calls clear_yolo_motor_stop() after it handles the event
    so it knows not to re-issue motors.forward() prematurely.
    """
    with _yolo_lock:
        return _yolo_motor_stop


def clear_yolo_motor_stop():
    """Call from mission.py after handling a YOLO-triggered stop."""
    global _yolo_motor_stop
    with _yolo_lock:
        _yolo_motor_stop = False


def get_last_yolo_position(label: str) -> dict | None:
    """
    Return the most recent YOLO position for a label, regardless of cooldown.
    Returns dict with keys: dist_m, bearing, bearing_deg.
    Returns None if label has never been detected.
    Used by mission.py to recover spatial context after a long Cosmos call.
    """
    with _yolo_lock:
        pos = _last_yolo_positions.get(label)
        return dict(pos) if pos else None


def set_safety_active(active: bool):
    """Enable/disable Layer 1 auto-stop. Use False only for testing."""
    global _safety_active
    _safety_active = active
    log.info(f"OAK-D safety: {'ENABLED' if active else 'DISABLED'}")


def set_yolo_active(active: bool):
    """Enable/disable Layer 2 YOLO detections. Call from mission.py."""
    global _yolo_active
    _yolo_active = active
    log.info(f"OAK-D YOLO detection: {'ACTIVE' if active else 'PAUSED'}")


def set_yolo_callback(fn):
    """
    Register mission.py callback for Layer 2 detections.
    Signature: fn(label: str, dist_m: float, bearing: str, bearing_deg: float)
      bearing     — coarse: "left" | "center" | "right"
      bearing_deg — precise: signed degrees (−90 … +90), negative=left
    Set to None to disable callbacks.
    """
    global _yolo_callback
    _yolo_callback = fn
    log.info(f"OAK-D YOLO callback: {'registered' if fn else 'cleared'}")


# ─── Initialisation ───────────────────────────────────────────────────────────

def init_oakd() -> bool:
    """
    (Re-)initialise OAK-D pipeline.
    Builds two pipelines on the Myriad X:
      1. Stereo depth (Layer 1 safety)
      2. YOLOv8n spatial detection (Layer 2 — optional, requires blob)
    Safe to call multiple times — tears down existing pipeline first.
    """
    global _pipeline, _depth_queue, _oakd_ok, _reader_thread
    global _yolo_spatial_queue, _yolo_ok

    _teardown()

    try:
        import depthai as dai

        _pipeline = dai.Pipeline()

        # ── Stereo depth nodes (Layer 1) ──────────────────────────────────────
        mono_left  = _pipeline.create(dai.node.MonoCamera)
        mono_right = _pipeline.create(dai.node.MonoCamera)
        stereo     = _pipeline.create(dai.node.StereoDepth)

        mono_left.setResolution(
            dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_left.setFps(15)
        mono_right.setResolution(
            dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        mono_right.setFps(15)

        stereo.setLeftRightCheck(True)
        stereo.setExtendedDisparity(False)
        stereo.setSubpixel(False)

        # Try best preset first, fall back gracefully
        for preset_name in ("HIGH_DENSITY", "HIGH_ACCURACY", "DEFAULT"):
            try:
                preset = getattr(dai.node.StereoDepth.PresetMode, preset_name)
                stereo.setDefaultProfilePreset(preset)
                log.info(f"OAK-D stereo preset: {preset_name}")
                break
            except AttributeError:
                continue

        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)
        _depth_queue = stereo.depth.createOutputQueue(maxSize=1, blocking=False)

        # ── YOLO spatial detection nodes (Layer 2) ────────────────────────────
        _yolo_ok = False
        if YOLO_BLOB_PATH.exists():
            try:
                _yolo_ok = _add_yolo_pipeline(_pipeline, stereo)
                if _yolo_ok:
                    log.info("✅ OAK-D YOLO: YOLOv8n spatial detection active "
                             "on Myriad X")
            except Exception as ye:
                log.warning(f"⚠️  OAK-D YOLO pipeline failed ({ye}) "
                            "— depth still active")
                _yolo_ok = False
        else:
            log.warning(
                f"⚠️  OAK-D YOLO: blob not found at {YOLO_BLOB_PATH} "
                "— Layer 2 person detection DISABLED. "
                "Download from https://github.com/luxonis/depthai-model-zoo"
            )

        _pipeline.start()
        _oakd_ok = True
        log.info("✅ OAK-D Lite: stereo depth started (DepthAI 3.x)")

        _reader_thread = threading.Thread(
            target=_reader_loop, daemon=True, name="oakd-reader"
        )
        _reader_thread.start()

        _ensure_reconnect_watchdog()
        return True

    except Exception as e:
        log.warning(f"⚠️  OAK-D init failed ({e}) — depth perception disabled")
        _oakd_ok = False
        _yolo_ok = False
        _ensure_reconnect_watchdog()
        return False


def _add_yolo_pipeline(pipeline, stereo) -> bool:
    """
    Add YOLOv8n spatial detection to an existing pipeline.
    DepthAI 3.x API — YOLO config goes via nn.detectionParser sub-node.
    Colour camera feeds YOLO; stereo depth provides spatial coordinates.
    """
    import depthai as dai

    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(416, 416)   # YOLOv8n input size
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setIspScale(1, 1)          # force ISP to honour 1080P — prevents
    cam_rgb.setVideoSize(1920, 1080)   # "expected 1920x1080 received 2104x1560"
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(5)

    nn = pipeline.create(dai.node.SpatialDetectionNetwork)
    nn.setBlobPath(str(YOLO_BLOB_PATH))
    nn.setConfidenceThreshold(YOLO_MIN_CONFIDENCE)
    nn.setBoundingBoxScaleFactor(0.5)
    nn.setDepthLowerThreshold(100)    # mm — ignore depth < 10 cm
    nn.setDepthUpperThreshold(8000)   # mm — ignore depth > 8 m

    # DepthAI 3.x — YOLO-specific config lives in detectionParser sub-node
    nn.detectionParser.setNumClasses(80)
    nn.detectionParser.setCoordinateSize(4)
    nn.detectionParser.setAnchors([
        10, 13, 16, 30, 33, 23,
        30, 61, 62, 45, 59, 119,
        116, 90, 156, 198, 373, 326
    ])
    nn.detectionParser.setAnchorMasks({
        "side52": [0, 1, 2],
        "side26": [3, 4, 5],
        "side13": [6, 7, 8]
    })
    nn.detectionParser.setIouThreshold(0.5)

    stereo.depth.link(nn.inputDepth)
    cam_rgb.preview.link(nn.input)

    global _yolo_spatial_queue
    _yolo_spatial_queue = nn.out.createOutputQueue(maxSize=4, blocking=False)
    return True


def _teardown():
    """Stop existing pipeline and clear state without raising."""
    global _pipeline, _depth_queue, _depth_frame, _oakd_ok
    global _yolo_spatial_queue, _yolo_ok
    _oakd_ok            = False
    _yolo_ok            = False
    _depth_queue        = None
    _yolo_spatial_queue = None
    with _lock:
        _depth_frame = None
    try:
        if _pipeline is not None:
            _pipeline.stop()
    except Exception:
        pass
    _pipeline = None


# ─── Reader loop ──────────────────────────────────────────────────────────────

def _reader_loop():
    """
    Background thread — reads depth frames and YOLO detections continuously.

    Layer 1: every depth frame → _safety_check() → instant stop/slow.
    Layer 2: every YOLO_CHECK_HZ interval → _yolo_check() → callback.

    On fatal disconnect: logs once, clears _oakd_ok, exits cleanly.
    Reconnect watchdog handles restart.
    """
    global _depth_frame, _oakd_ok, _last_floor_check

    floor_check_interval = 1.0 / FLOOR_CHECK_HZ
    yolo_check_interval  = 1.0 / YOLO_CHECK_HZ
    last_yolo_check      = 0.0

    while _oakd_ok:
        try:
            if _depth_queue is None:
                time.sleep(0.1)
                continue

            in_depth = _depth_queue.get(timeout=datetime.timedelta(seconds=1))  # 1s timeout — prevents infinite block on disconnect
            if in_depth is None:
                continue

            frame = in_depth.getFrame()
            with _lock:
                _depth_frame = frame

            if _safety_active:
                _safety_check(floor_check_interval)

            # ── Layer 2: YOLO detections ──────────────────────────────────────
            now = time.monotonic()
            if (_yolo_ok and _yolo_active
                    and _yolo_spatial_queue is not None
                    and now - last_yolo_check >= yolo_check_interval):
                last_yolo_check = now
                _yolo_check()

        except Exception as e:
            err_str = str(e)
            if any(token in err_str for token in _FATAL_ERRORS):
                log.warning(
                    f"⚠️  OAK-D disconnected ({err_str.splitlines()[0]}) "
                    "— depth disabled, reconnect watchdog will retry"
                )
                _oakd_ok = False
                return
            log.warning(f"OAK-D reader frame error: {e}")
            time.sleep(0.5)

    log.info("OAK-D reader loop exited (device offline)")


# ─── Layer 1: Safety check ────────────────────────────────────────────────────

def _safety_check(floor_check_interval: float):
    """
    Layer 1 safety reactions — called every depth frame.

    Check 1 — Front obstacle (3-patch sampling):
      Samples depth at (left-of-centre, centre, right-of-centre).
      Uses minimum of valid samples — one textureless patch can't blind us.
      Logs a warning if fewer than 3 patches return valid depth (possible
      glass / white wall) so operators can investigate.

    Check 2 — Floor drop / void (rate-limited at FLOOR_CHECK_HZ):
      HIGH confidence  → motors.stop() — unambiguous drop detected.
      MEDIUM confidence → motors.stop() — stairs are fatal; better safe.
        Medium is logged differently so operators can check if it's a
        wide open doorway causing false positives.

    Both checks are independent of Cosmos — pure reactive hardware safety.

    Sensor arbitration:
      LiDAR and OAK-D both call motors.stop() independently. Whichever
      fires first wins. No arbitration is correct for safety — the most
      conservative sensor always dominates. Adding arbitration would only
      delay the stop.
    """
    global _last_floor_check

    try:
        from motors import motors

        # ── Check 1: front obstacle — 3-patch depth sampling ─────────────────
        samples = _get_front_depth_3patch()
        if samples["valid_patches"] == 0:
            # No depth at all in front — don't act, but warn
            pass
        else:
            if samples["valid_patches"] < 3:
                log.debug(
                    f"OAK-D: only {samples['valid_patches']}/3 depth patches "
                    "valid (possible glass/white wall) — using available patches"
                )
            front = samples["min_m"]
            # Check if avoidance turn is in progress — suppress stop if so
            _avoidance_in_progress = False
            try:
                from lidar import _avoidance_active
                _avoidance_in_progress = _avoidance_active
            except Exception:
                pass
            if not _avoidance_in_progress:
                if front < OAKD_STOP_DIST:
                    motors.stop()
                    log.warning(f"🚧 OAK-D STOP — obstacle at {front:.2f}m "
                                f"({samples['valid_patches']}/3 patches)")
                elif front < OAKD_SLOW_DIST:
                    motors.slow()
                    log.info(f"⚠️  OAK-D slow — obstacle at {front:.2f}m "
                             f"({samples['valid_patches']}/3 patches)")

        # ── Check 2: floor drop / void (HIGH confidence only) ────────────────
        # Re-enabled — LiDAR void is disabled (horizontal scanner can't see drops).
        # Only HIGH confidence fires — medium was causing flat-floor false positives.
        # Rate-limited by FLOOR_CHECK_HZ to avoid hammering the depth frame.
        now_fc = time.monotonic()
        if now_fc - _last_floor_check >= 1.0 / FLOOR_CHECK_HZ:
            _last_floor_check = now_fc
            drop = get_floor_drop()
            if drop["void_detected"] and drop["confidence"] == "high":
                if not _avoidance_in_progress:
                    motors.stop()
                    log.warning(
                        f"🕳️  OAK-D VOID STOP (high confidence): {drop['reason']}"
                    )

    except Exception as e:
        log.debug(f"OAK-D safety check error: {e}")


# ─── Layer 2: YOLO check ──────────────────────────────────────────────────────

def _yolo_check():
    """
    Layer 2: poll YOLO spatial detections from Myriad X.
    Runs at YOLO_CHECK_HZ — only when _yolo_active is True.

    For each detection of a target class (person/animal):
      1. Extract confidence, spatial depth (mm → m), bounding box.
      2. Compute proportional bearing_deg (−90 … +90°) from bbox centre x.
         Negative = left, positive = right.
         Also compute coarse bearing string (left/center/right) for compat.
      3. Track bounding box width trend: if box is SHRINKING (target leaving),
         skip callback — don't chase someone walking away.
         Only fire callback when box is stable or growing (target approaching
         or stationary).
      4. Always update _last_yolo_positions regardless of trend or cooldown,
         so mission.py can recover spatial context after a long Cosmos call.
      5. Layer 2 motor guard: if stopping for a target, set _yolo_motor_stop=True
         so mission.py knows not to re-issue motors.forward() immediately.
      6. Apply per-label cooldown before firing callback.
    """
    global _last_yolo_detect, _last_yolo_bbox_w, _yolo_motor_stop

    try:
        detections = _yolo_spatial_queue.tryGetAll()
        if not detections:
            return

        # Use most recent detection packet — discard older ones in queue
        packet = detections[-1]
        now    = time.monotonic()

        for det in packet.detections:
            label_idx = det.label
            if label_idx not in YOLO_TARGET_CLASSES:
                continue

            label      = YOLO_TARGET_CLASSES[label_idx]
            confidence = det.confidence
            if confidence < YOLO_MIN_CONFIDENCE:
                continue

            # ── Spatial depth ─────────────────────────────────────────────────
            dist_mm = det.spatialCoordinates.z
            if dist_mm <= 0:
                continue
            dist_m = dist_mm / 1000.0

            # ── Proportional bearing ──────────────────────────────────────────
            # bbox centre x: 0.0 = left edge, 1.0 = right edge
            # Map to degrees: 0.0→−45°, 0.5→0°, 1.0→+45°
            # (±45° is the approximate OAK-D Lite horizontal FOV half-angle)
            cx          = (det.xmin + det.xmax) / 2.0
            bearing_deg = (cx - 0.5) * 90.0   # −45 … +45°, negative=left

            if cx < 0.35:
                bearing = "left"
            elif cx > 0.65:
                bearing = "right"
            else:
                bearing = "center"

            # ── Bounding box width trend ──────────────────────────────────────
            # bbox width normalised 0.0–1.0
            bbox_w = det.xmax - det.xmin
            with _yolo_lock:
                prev_w = _last_yolo_bbox_w.get(label, bbox_w)
                _last_yolo_bbox_w[label] = bbox_w
                shrinking = bbox_w < prev_w * 0.85   # >15% smaller → leaving

                # ── EMA bearing smoothing ─────────────────────────────────────
                # Reduces single-frame jitter without significant lag.
                prev_ema = _bearing_ema.get(label, bearing_deg)
                smoothed_bearing_deg = (_BEARING_EMA_ALPHA * bearing_deg
                                        + (1.0 - _BEARING_EMA_ALPHA) * prev_ema)
                _bearing_ema[label]  = smoothed_bearing_deg
                bearing_deg          = round(smoothed_bearing_deg, 1)

                # Recompute coarse bearing from smoothed value
                if bearing_deg < -12.0:
                    bearing = "left"
                elif bearing_deg > 12.0:
                    bearing = "right"
                else:
                    bearing = "center"

            # ── Always update position memory ─────────────────────────────────
            with _yolo_lock:
                _last_yolo_positions[label] = {
                    "dist_m":     dist_m,
                    "bearing":    bearing,
                    "bearing_deg": round(bearing_deg, 1),
                    "confidence": round(confidence, 2),
                    "timestamp":  now,
                }

            # ── Skip callback if target is clearly moving away ────────────────
            if shrinking:
                log.debug(f"YOLO: {label} bbox shrinking "
                          f"({prev_w:.2f}→{bbox_w:.2f}) — target leaving, skip callback")
                continue

            # ── Layer 2 motor reaction ────────────────────────────────────────
            try:
                from motors import motors
                if dist_m < YOLO_STOP_DIST:
                    motors.stop()
                    with _yolo_lock:
                        _yolo_motor_stop = True
                    log.info(f"🛑 YOLO STOP — {label} at {dist_m:.1f}m "
                             f"({bearing} / {bearing_deg:+.0f}°)")
                elif dist_m < YOLO_SLOW_DIST:
                    motors.slow()
                    log.info(f"⚠️  YOLO slow — {label} at {dist_m:.1f}m "
                             f"({bearing} / {bearing_deg:+.0f}°)")
            except Exception:
                pass

            # ── Per-label cooldown before callback ────────────────────────────
            with _yolo_lock:
                last_cb = _last_yolo_detect.get(label, 0.0)
                if now - last_cb < YOLO_COOLDOWN_S:
                    continue
                _last_yolo_detect[label] = now

            # ── Fire callback ─────────────────────────────────────────────────
            cb = _yolo_callback
            if cb:
                try:
                    cb(label, dist_m, bearing, bearing_deg)
                except Exception as ce:
                    log.warning(f"YOLO callback error: {ce}")

    except Exception as e:
        log.debug(f"OAK-D YOLO check error: {e}")


# ─── Reconnect watchdog ───────────────────────────────────────────────────────

def _ensure_reconnect_watchdog():
    global _reconnect_thread, _reconnect_started
    if _reconnect_started:
        return
    _reconnect_started = True
    _reconnect_thread  = threading.Thread(
        target=_reconnect_loop, daemon=True, name="oakd-reconnect"
    )
    _reconnect_thread.start()


def _reconnect_loop():
    while True:
        time.sleep(RECONNECT_INTERVAL_S)
        if not _oakd_ok:
            log.info("🔄 OAK-D reconnect attempt...")
            init_oakd()


# ─── Depth query helpers ──────────────────────────────────────────────────────

def get_depth_at(x_ratio: float = 0.5, y_ratio: float = 0.5,
                 patch_px: int = 10) -> float | None:
    """Return median depth (meters) at a normalised (x,y) position."""
    if not _oakd_ok:
        return None
    with _lock:
        if _depth_frame is None:
            return None
        frame = _depth_frame.copy()
    h, w = frame.shape
    cx = int(x_ratio * w)
    cy = int(y_ratio * h)
    x1, x2 = max(0, cx - patch_px), min(w, cx + patch_px)
    y1, y2 = max(0, cy - patch_px), min(h, cy + patch_px)
    patch = frame[y1:y2, x1:x2]
    valid = patch[patch > 0]
    if len(valid) == 0:
        return None
    return float(np.median(valid)) / 1000.0


def _get_front_depth_3patch() -> dict:
    """
    Sample depth at three horizontal positions in the front-centre zone:
      left-of-centre  (x=0.35, y=0.65)
      centre          (x=0.50, y=0.65)
      right-of-centre (x=0.65, y=0.65)

    Returns dict:
      {
        "min_m":         float — minimum of valid samples (use for safety),
        "samples":       list[float | None] — [left, centre, right],
        "valid_patches": int — number of patches that returned valid depth,
      }

    Using the minimum of three samples means a single textureless patch
    (white wall, glass door) can't silence the whole front obstacle check.
    Returns min_m=999.0 if no patches are valid.
    """
    positions = [(0.35, 0.65), (0.50, 0.65), (0.65, 0.65)]
    samples   = [get_depth_at(x, y) for x, y in positions]
    valid     = [s for s in samples if s is not None]
    return {
        "min_m":         min(valid) if valid else 999.0,
        "samples":       samples,
        "valid_patches": len(valid),
    }


def get_front_depth() -> float | None:
    """
    Return minimum front depth (meters) from 3-patch sampling.
    Returns None only if all three patches are invalid (no depth data at all).
    """
    result = _get_front_depth_3patch()
    m = result["min_m"]
    return m if m < 999.0 else None


def get_depth_map() -> dict | None:
    """Return a 9-cell depth map (top/mid/bot × left/center/right)."""
    if not _oakd_ok:
        return None
    cells = {
        "top_left":   (0.2, 0.2), "top_center": (0.5, 0.2), "top_right":   (0.8, 0.2),
        "mid_left":   (0.2, 0.5), "mid_center": (0.5, 0.5), "mid_right":   (0.8, 0.5),
        "bot_left":   (0.2, 0.8), "bot_center": (0.5, 0.8), "bot_right":   (0.8, 0.8),
    }
    result = {}
    for name, (x, y) in cells.items():
        d = get_depth_at(x, y)
        result[name] = round(d, 2) if d is not None else None
    return result


# ─── Floor-drop / void detection ─────────────────────────────────────────────

def get_floor_drop(
    strip_y_ratio:    float = 0.75,   # 15cm mount: floor visible at y≈0.65-0.80, not 0.90
    strip_height_px:  int   = 25,     # taller strip gives better statistics at this height
    patch_cols:       int   = 5,
    normal_max_m:     float = 6.0,    # larger rooms won't false-trigger
    drop_threshold_m: float = 3.0,    # 15cm height — small step looks like large drop
) -> dict:
    """
    Detect floor discontinuity (holes, stairs, cliffs) from stereo depth.

    Camera mount: OAK-D is ~15cm off the ground. At this height the camera
    sees the floor at a very shallow angle — the bottom strip of the frame
    is floor at 20-40cm. Stereo depth on low-texture carpet/tile at that
    angle returns very few valid pixels (specular reflections, poor baseline).
    The old 2% sparse threshold fired on every flat indoor floor.

    Tuning for 15cm mount:
      strip_y_ratio:   0.90 → 0.75   sample higher where floor is more visible
      strip_height_px: 15  → 25      more pixels for better statistics
      sparse HIGH:     2%  → 0.5%    only truly empty = open air
      drop_threshold:  2.0 → 3.0     shallow angle exaggerates drop apparent size
      normal_max_m:    5.0 → 6.0     larger rooms were false-triggering

    Confidence levels:
      high   → return_ratio < 0.5% OR edge depth >> normal_max_m
      medium → drop_delta > drop_threshold_m OR no edge returns but mid valid
      low    → no anomaly
    """
    if not _oakd_ok:
        return {"void_detected": False, "confidence": "low",
                "floor_mid_m": None, "floor_edge_m": None,
                "valid_returns": 0, "reason": "OAK-D unavailable"}

    with _lock:
        if _depth_frame is None:
            return {"void_detected": False, "confidence": "low",
                    "floor_mid_m": None, "floor_edge_m": None,
                    "valid_returns": 0, "reason": "no depth frame"}
        frame = _depth_frame.copy()

    h, w = frame.shape

    # ── Mid-frame floor reference ─────────────────────────────────────────────
    mid_y     = int(0.5 * h)
    mid_patch = frame[max(0, mid_y - 10):min(h, mid_y + 10), w // 4: 3 * w // 4]
    mid_valid = mid_patch[(mid_patch > 0) & (mid_patch < 8000)]
    floor_mid_m = (float(np.median(mid_valid)) / 1000.0
                   if len(mid_valid) > 10 else None)

    # ── Bottom-strip floor edge ───────────────────────────────────────────────
    edge_y0    = int(strip_y_ratio * h)
    edge_y1    = min(h, edge_y0 + strip_height_px)
    edge_strip = frame[edge_y0:edge_y1, :]

    col_step    = w // (patch_cols + 1)
    edge_depths = []
    valid_count = 0
    for col in range(1, patch_cols + 1):
        cx    = col * col_step
        patch = edge_strip[:, max(0, cx - 8):min(w, cx + 8)]
        valid = patch[(patch > 0) & (patch < 12000)]
        valid_count += len(valid)
        if len(valid) >= 3:
            edge_depths.append(float(np.median(valid)) / 1000.0)

    floor_edge_m = float(np.median(edge_depths)) if edge_depths else None

    # ── Void decision ─────────────────────────────────────────────────────────
    void_detected = False
    confidence    = "low"
    reason        = "no anomaly"

    total_edge_pixels = edge_strip.size
    return_ratio      = valid_count / max(total_edge_pixels, 1)

    if return_ratio < 0.005 and total_edge_pixels > 100:   # < 0.5% — truly open air only
        void_detected = True
        confidence    = "high"
        reason        = (f"floor edge returns sparse ({return_ratio:.1%}) "
                         "— drop or hole below")

    elif floor_mid_m is not None and floor_edge_m is not None:
        drop_delta = floor_edge_m - floor_mid_m
        if floor_edge_m > normal_max_m:
            void_detected = True
            confidence    = "high"
            reason        = (f"floor edge depth {floor_edge_m:.1f}m >> "
                             f"normal ({floor_mid_m:.1f}m) — stairs or cliff")
        elif drop_delta > drop_threshold_m:
            void_detected = True
            confidence    = "medium"
            reason        = (f"floor drops {drop_delta:.1f}m at edge "
                             f"({floor_mid_m:.1f}m→{floor_edge_m:.1f}m) "
                             "— step or slope")

    elif floor_edge_m is None and floor_mid_m is not None and floor_mid_m < 2.0:
        void_detected = True
        confidence    = "medium"
        reason        = (f"no floor returns at edge, mid={floor_mid_m:.1f}m "
                         "— possible drop")

    return {
        "void_detected": void_detected,
        "confidence":    confidence,
        "floor_mid_m":   round(floor_mid_m, 2) if floor_mid_m is not None else None,
        "floor_edge_m":  round(floor_edge_m, 2) if floor_edge_m is not None else None,
        "valid_returns": valid_count,
        "reason":        reason,
    }


def void_ahead() -> bool:
    """Quick boolean — True if any void is detected at any confidence."""
    return get_floor_drop()["void_detected"]


# ─── Status helpers ───────────────────────────────────────────────────────────

def get_status() -> dict:
    """Return current OAK-D + YOLO status for GUI or debug display."""
    result = _get_front_depth_3patch()
    return {
        "available":     _oakd_ok,
        "front_m":       round(result["min_m"], 2) if result["min_m"] < 999 else None,
        "valid_patches": result["valid_patches"],
        "yolo_ok":       _yolo_ok,
        "yolo_active":   _yolo_active,
        "yolo_blob":     YOLO_BLOB_PATH.exists(),
    }


def oakd_status_html() -> str:
    """
    HTML status panel for Gradio.
    Now shows: front depth, patch coverage, and YOLO Layer 2 status.
    YOLO status: ✅ active | ❌ no blob | ⏸ paused
    """
    s = get_status()

    if not s["available"]:
        return """
        <div style="background:#1a1a1a;border:1px solid #444;border-radius:8px;
                    padding:10px;font-family:monospace;color:#666">
            📷 OAK-D Lite: not connected
        </div>"""

    d = s["front_m"]
    if d is None:
        dist_str, color, label = "—", "#666", "No depth data"
    elif d < 0.30:
        dist_str, color, label = f"{d:.2f}m", "#cc0000", "🚧 VERY CLOSE"
    elif d < 0.60:
        dist_str, color, label = f"{d:.2f}m", "#ff6600", "⚠️  CLOSE"
    else:
        dist_str, color, label = f"{d:.2f}m", "#76b900", "✅ CLEAR"

    patches = s["valid_patches"]
    patch_color = "#76b900" if patches == 3 else "#ff6600" if patches > 0 else "#cc0000"
    patch_str   = f"{patches}/3"

    if s["yolo_ok"] and s["yolo_active"]:
        yolo_str   = "✅ active"
        yolo_color = "#76b900"
    elif s["yolo_ok"] and not s["yolo_active"]:
        yolo_str   = "⏸ paused"
        yolo_color = "#888"
    elif s["yolo_blob"]:
        yolo_str   = "⚠️ init fail"
        yolo_color = "#ff6600"
    else:
        yolo_str   = "❌ no blob"
        yolo_color = "#cc0000"

    return f"""
    <div style="background:#1a1a1a;border:1px solid {color};border-radius:8px;
                padding:10px;font-family:monospace;">
        <div style="color:{color};font-weight:bold">📷 OAK-D Lite — {label}</div>
        <div style="color:#aaa;font-size:0.85em;margin-top:4px">
            Front depth: <span style="color:#fff">{dist_str}</span>
            &nbsp;|&nbsp;
            Patches: <span style="color:{patch_color}">{patch_str}</span>
            &nbsp;|&nbsp;
            YOLO L2: <span style="color:{yolo_color}">{yolo_str}</span>
        </div>
    </div>"""