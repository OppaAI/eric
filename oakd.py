"""
ERIC — OAK-D Lite Depth Camera
DepthAI 3.x API — Pipeline runs itself via pipeline.start().
No separate Device creation needed.
"""

import logging
import threading
import numpy as np

log = logging.getLogger("eric.oakd")

_pipeline    = None
_depth_queue = None
_depth_frame = None
_lock        = threading.Lock()
_oakd_ok     = False
_reader_thread = None


def oakd_available() -> bool:
    return _oakd_ok


def init_oakd() -> bool:
    global _pipeline, _depth_queue, _oakd_ok, _reader_thread

    try:
        import depthai as dai

        _pipeline = dai.Pipeline()

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

        _reader_thread = threading.Thread(target=_reader_loop, daemon=True, name="oakd-reader")
        _reader_thread.start()
        return True

    except Exception as e:
        log.warning(f"⚠️  OAK-D init failed ({e}) — depth perception disabled")
        _oakd_ok = False
        return False


def _reader_loop():
    global _depth_frame
    while True:
        try:
            if _depth_queue is None:
                import time; time.sleep(0.1)
                continue
            in_depth = _depth_queue.get()
            if in_depth is None:
                continue
            frame = in_depth.getFrame()
            with _lock:
                _depth_frame = frame
        except Exception as e:
            log.warning(f"OAK-D reader frame error: {e}")
            import time; time.sleep(0.1)


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


def get_floor_drop(
    strip_y_ratio: float = 0.85,    # sample near bottom of frame (floor edge)
    strip_height_px: int = 20,      # pixel rows to average for floor strip
    patch_cols: int = 5,            # number of horizontal sample columns
    normal_max_m: float = 3.0,      # anything beyond this = floor gone / void
    drop_threshold_m: float = 1.2,  # floor-to-void jump that triggers alert
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
        "floor_mid_m":   float | None,   # depth at frame center
        "floor_edge_m":  float | None,   # depth at bottom strip (where floor should be)
        "valid_returns": int,             # number of valid pixels in bottom strip
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

    # ── Mid-frame floor reference (y=0.5, centre horizontal strip) ───────────
    mid_y    = int(0.5 * h)
    mid_patch = frame[max(0, mid_y - 10):min(h, mid_y + 10), w // 4: 3 * w // 4]
    mid_valid = mid_patch[(mid_patch > 0) & (mid_patch < 8000)]   # <8m valid
    floor_mid_m = float(np.median(mid_valid)) / 1000.0 if len(mid_valid) > 10 else None

    # ── Bottom-strip floor edge (where the floor should end / stair lip begins) ─
    edge_y0   = int(strip_y_ratio * h)
    edge_y1   = min(h, edge_y0 + strip_height_px)
    edge_strip = frame[edge_y0:edge_y1, :]

    # Sample patch_cols evenly spaced columns for robustness
    col_step   = w // (patch_cols + 1)
    edge_depths = []
    valid_count = 0
    for col in range(1, patch_cols + 1):
        cx = col * col_step
        patch = edge_strip[:, max(0, cx - 8):min(w, cx + 8)]
        valid = patch[(patch > 0) & (patch < 12000)]
        valid_count += len(valid)
        if len(valid) >= 3:
            edge_depths.append(float(np.median(valid)) / 1000.0)

    floor_edge_m = float(np.median(edge_depths)) if edge_depths else None

    # ── Void decision logic ────────────────────────────────────────────────────
    void_detected = False
    confidence    = "low"
    reason        = "no anomaly"

    # Case 1: Bottom strip has almost no valid returns → open air / void below
    total_edge_pixels = edge_strip.size
    return_ratio = valid_count / max(total_edge_pixels, 1)
    if return_ratio < 0.05 and total_edge_pixels > 100:
        void_detected = True
        confidence    = "high"
        reason        = f"floor edge returns sparse ({return_ratio:.1%}) — drop or hole below"

    # Case 2: Floor suddenly gets much further away at the edge vs centre
    elif floor_mid_m is not None and floor_edge_m is not None:
        drop_delta = floor_edge_m - floor_mid_m
        if floor_edge_m > normal_max_m:
            void_detected = True
            confidence    = "high"
            reason        = f"floor edge depth {floor_edge_m:.1f}m >> normal ({floor_mid_m:.1f}m) — stairs or cliff"
        elif drop_delta > drop_threshold_m:
            void_detected = True
            confidence    = "medium"
            reason        = f"floor drops {drop_delta:.1f}m at edge ({floor_mid_m:.1f}m→{floor_edge_m:.1f}m) — step or slope"

    # Case 3: No edge depth readings at all (camera over void — nothing to bounce off)
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
    result = get_floor_drop()
    return result["void_detected"]
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
