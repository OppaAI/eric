"""
ERIC — OAK-D Lite Depth Camera
DepthAI 3.x API — Pipeline runs itself via pipeline.start().
No separate Device creation needed.

Layer 1 Safety (industrial standard — mirrors LiDAR safety monitor):
  Every depth frame is checked instantly:
    front depth  < OAKD_STOP_DIST  → motors.stop()  immediately
    front depth  < OAKD_SLOW_DIST  → motors.slow()  immediately
    floor drop detected (high confidence) → motors.stop() immediately
  This fires in the reader thread — zero latency, independent of Cosmos.
  Floor drop check runs at FLOOR_CHECK_HZ to limit CPU (numpy is expensive).

Disconnect handling:
  - _reader_loop detects "MessageQueue was closed" / XLinkError and exits cleanly
  - _oakd_ok is cleared immediately so all callers see unavailable
  - A reconnect watchdog retries init_oakd() every RECONNECT_INTERVAL_S seconds
  - Log spam is suppressed: only one warning per disconnect event, then silence
    until the device comes back online
"""

import logging
import threading
import time
import numpy as np

log = logging.getLogger("eric.oakd")

_pipeline      = None
_depth_queue   = None
_depth_frame   = None
_lock          = threading.Lock()
_oakd_ok       = False
_reader_thread = None

# Reconnect watchdog
_reconnect_thread    = None
_reconnect_started   = False
RECONNECT_INTERVAL_S = 10.0   # seconds between reconnection attempts

# ── Layer 1 Safety constants ──────────────────────────────────────────────────
OAKD_STOP_DIST   = 0.30   # meters — stop if obstacle closer than this
OAKD_SLOW_DIST   = 0.60   # meters — slow if obstacle closer than this
FLOOR_CHECK_HZ   = 5.0    # how often to run floor-drop check (per second)
                           # get_floor_drop() is numpy-heavy — don't run every frame

_safety_active   = True   # can be disabled for testing
_last_floor_check = 0.0   # timestamp of last floor drop check

# Error strings that mean the queue/pipeline is permanently closed
_FATAL_ERRORS = (
    "MessageQueue was closed",
    "X_LINK_ERROR",
    "XLinkError",
    "Couldn't read data",
    "Communication exception",
)


def oakd_available() -> bool:
    return _oakd_ok


def set_safety_active(active: bool):
    """Enable/disable Layer 1 auto-stop. Use False only for testing."""
    global _safety_active
    _safety_active = active
    log.info(f"OAK-D safety: {'ENABLED' if active else 'DISABLED'}")


# ─── Initialisation ───────────────────────────────────────────────────────────

def init_oakd() -> bool:
    """
    (Re-)initialise OAK-D pipeline.
    Safe to call multiple times — tears down existing pipeline first.
    Returns True on success.
    """
    global _pipeline, _depth_queue, _oakd_ok, _reader_thread

    # Tear down any existing pipeline cleanly before re-init
    _teardown()

    try:
        import depthai as dai

        _pipeline  = dai.Pipeline()

        mono_left  = _pipeline.create(dai.node.MonoCamera)
        mono_right = _pipeline.create(dai.node.MonoCamera)
        stereo     = _pipeline.create(dai.node.StereoDepth)

        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        stereo.setLeftRightCheck(True)
        stereo.setExtendedDisparity(False)
        stereo.setSubpixel(False)

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

        # DepthAI 3.x — createOutputQueue on the node output, then start pipeline
        _depth_queue = stereo.depth.createOutputQueue(maxSize=1, blocking=False)
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
        _ensure_reconnect_watchdog()
        return False


def _teardown():
    """Stop the existing pipeline and clear state without raising."""
    global _pipeline, _depth_queue, _depth_frame, _oakd_ok
    _oakd_ok     = False
    _depth_queue = None
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
    Background thread — reads depth frames from the DepthAI output queue.

    Layer 1 Safety (industrial standard):
      After storing each frame, immediately checks:
        1. Front obstacle distance → stop/slow motors instantly (no Cosmos, no latency)
        2. Floor drop / void       → stop motors instantly (staircase protection)
      Floor drop check is rate-limited to FLOOR_CHECK_HZ to avoid hammering numpy.

    On a clean disconnect (MessageQueue closed / XLinkError) we:
      1. Log ONE warning.
      2. Set _oakd_ok = False so all callers immediately see unavailable.
      3. Exit the loop — the reconnect watchdog will call init_oakd() again.
    """
    global _depth_frame, _oakd_ok, _last_floor_check

    floor_check_interval = 1.0 / FLOOR_CHECK_HZ

    while _oakd_ok:
        try:
            if _depth_queue is None:
                time.sleep(0.1)
                continue

            in_depth = _depth_queue.get()
            if in_depth is None:
                continue

            frame = in_depth.getFrame()
            with _lock:
                _depth_frame = frame

            # ── Layer 1: instant safety reaction ─────────────────────────────
            # Mirrors LiDAR _scan_callback() — fires every frame, zero latency.
            if _safety_active:
                _safety_check(floor_check_interval)

        except Exception as e:
            err_str = str(e)

            # Fatal disconnect — log once, mark unavailable, exit thread
            if any(token in err_str for token in _FATAL_ERRORS):
                log.warning(
                    f"⚠️  OAK-D disconnected ({err_str.splitlines()[0]}) "
                    "— depth disabled, reconnect watchdog will retry"
                )
                _oakd_ok = False
                return   # ← exit loop; no more spam

            # Transient / unexpected error — log and back off briefly
            log.warning(f"OAK-D reader frame error: {e}")
            time.sleep(0.5)

    log.info("OAK-D reader loop exited (device offline)")


def _safety_check(floor_check_interval: float):
    """
    Layer 1 safety reactions — called every depth frame from _reader_loop().

    Check 1 — Front obstacle:
      Reads front depth (center of frame). If closer than thresholds →
      motors.stop() or motors.slow() instantly, same as LiDAR does.

    Check 2 — Floor drop / void (rate-limited):
      Runs get_floor_drop() at FLOOR_CHECK_HZ. High-confidence void →
      motors.stop() instantly. This is the staircase protection.

    Both checks are independent of Cosmos — pure reactive hardware safety.
    """
    global _last_floor_check

    try:
        from motors import motors

        # ── Check 1: front obstacle ───────────────────────────────────────────
        front = get_front_depth()
        if front is not None:
            if front < OAKD_STOP_DIST:
                motors.stop()
                log.warning(f"🚧 OAK-D STOP — obstacle at {front:.2f}m")
            elif front < OAKD_SLOW_DIST:
                motors.slow()
                log.info(f"⚠️  OAK-D slow — obstacle at {front:.2f}m")

        # ── Check 2: floor drop / void (rate-limited) ─────────────────────────
        now = time.monotonic()
        if now - _last_floor_check >= floor_check_interval:
            _last_floor_check = now
            drop = get_floor_drop()
            if drop["void_detected"] and drop["confidence"] == "high":
                motors.stop()
                log.warning(
                    f"🕳️  OAK-D VOID STOP — {drop['reason']}"
                )

    except Exception as e:
        # Never let safety check crash the reader loop
        log.debug(f"OAK-D safety check error: {e}")


# ─── Reconnect watchdog ───────────────────────────────────────────────────────

def _ensure_reconnect_watchdog():
    """Start the reconnect watchdog thread once, if not already running."""
    global _reconnect_thread, _reconnect_started
    if _reconnect_started:
        return
    _reconnect_started = True
    _reconnect_thread  = threading.Thread(
        target=_reconnect_loop, daemon=True, name="oakd-reconnect"
    )
    _reconnect_thread.start()


def _reconnect_loop():
    """
    Periodically attempts to reconnect the OAK-D when it goes offline.
    Stays quiet while the device is connected; only logs when it tries.
    """
    while True:
        time.sleep(RECONNECT_INTERVAL_S)
        if not _oakd_ok:
            log.info("🔄 OAK-D reconnect attempt...")
            init_oakd()


# ─── Depth query helpers ──────────────────────────────────────────────────────

def get_depth_at(x_ratio: float = 0.5, y_ratio: float = 0.5,
                 patch_px: int = 10) -> float | None:
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


def get_front_depth() -> float | None:
    return get_depth_at(0.5, 0.65)


def get_depth_map() -> dict | None:
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
    strip_y_ratio: float = 0.85,
    strip_height_px: int = 20,
    patch_cols: int = 5,
    normal_max_m: float = 3.0,
    drop_threshold_m: float = 1.2,
) -> dict:
    """
    Detect floor discontinuity (holes, stairs, cliffs, balconies) using the
    OAK-D stereo depth map.

    Strategy:
      - Sample a horizontal strip of depth near the bottom of the frame
        (y_ratio=0.85 → lower quarter, where floor should appear when pan-tilt
        is at ground-looking angle).
      - Compare mid-frame floor depth (y≈0.5) vs bottom-strip floor depth.
        A sudden large jump (floor disappears) → void ahead.
      - Also flag if the bottom strip has very few valid depth returns
        (sparse returns = open air below the camera edge = drop).

    Returns dict:
      {
        "void_detected": bool,
        "confidence":    "high" | "medium" | "low",
        "floor_mid_m":   float | None,
        "floor_edge_m":  float | None,
        "valid_returns": int,
        "reason":        str,
      }
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
    floor_mid_m = float(np.median(mid_valid)) / 1000.0 if len(mid_valid) > 10 else None

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

    if return_ratio < 0.05 and total_edge_pixels > 100:
        void_detected = True
        confidence    = "high"
        reason        = f"floor edge returns sparse ({return_ratio:.1%}) — drop or hole below"

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
                             f"({floor_mid_m:.1f}m→{floor_edge_m:.1f}m) — step or slope")

    elif floor_edge_m is None and floor_mid_m is not None and floor_mid_m < 2.0:
        void_detected = True
        confidence    = "medium"
        reason        = f"no floor returns at edge, mid={floor_mid_m:.1f}m — possible drop"

    return {
        "void_detected": void_detected,
        "confidence":    confidence,
        "floor_mid_m":   round(floor_mid_m, 2) if floor_mid_m is not None else None,
        "floor_edge_m":  round(floor_edge_m, 2) if floor_edge_m is not None else None,
        "valid_returns": valid_count,
        "reason":        reason,
    }


def void_ahead() -> bool:
    """Quick boolean check — True if a floor drop/void is detected. Use in safety gates."""
    return get_floor_drop()["void_detected"]


# ─── Status helpers ───────────────────────────────────────────────────────────

def get_status() -> dict:
    """Return current OAK-D status for GUI or debug display."""
    front = get_front_depth()
    return {
        "available": _oakd_ok,
        "front_m":   round(front, 2) if front is not None else None,
    }


def oakd_status_html() -> str:
    s = get_status()
    if not s["available"]:
        return """
        <div style="background:#1a1a1a;border:1px solid #444;border-radius:8px;
                    padding:10px;font-family:monospace;color:#666">
            📷 OAK-D Lite: not connected
        </div>"""
    d = s["front_m"]
    if d is None:
        dist_str, color, label = "—", "#666", "No measurement"
    elif d < 0.30:
        dist_str, color, label = f"{d:.2f}m", "#cc0000", "🚧 VERY CLOSE"
    elif d < 0.60:
        dist_str, color, label = f"{d:.2f}m", "#ff6600", "⚠️  CLOSE"
    else:
        dist_str, color, label = f"{d:.2f}m", "#76b900", "✅ CLEAR"
    return f"""
    <div style="background:#1a1a1a;border:1px solid {color};border-radius:8px;
                padding:10px;font-family:monospace;">
        <div style="color:{color};font-weight:bold">📷 OAK-D Lite — {label}</div>
        <div style="color:#aaa;font-size:0.85em;margin-top:4px">
            Front depth: <span style="color:#fff">{dist_str}</span>
        </div>
    </div>"""
