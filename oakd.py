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


def get_status() -> dict:
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