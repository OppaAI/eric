"""
ERIC — OAK-D Lite Depth Camera
DepthAI stereo depth perception for metric obstacle distances.

Architecture:
  OAK-D Lite → DepthAI stereo pipeline → depth frame (uint16, mm)
  Background reader thread keeps latest frame fresh.
  get_front_depth() / get_depth_at() return meters — used by mission.py
  to give Cosmos real metric distances instead of visual guesses.

If OAK-D is unavailable, all functions return None gracefully — no crash.
"""

import logging
import threading
import numpy as np

log = logging.getLogger("eric.oakd")

_device      = None
_depth_frame = None   # latest depth frame, numpy uint16 (mm units)
_lock        = threading.Lock()
_oakd_ok     = False
_reader_thread = None


def oakd_available() -> bool:
    return _oakd_ok


def init_oakd() -> bool:
    """
    Initialize OAK-D Lite stereo depth pipeline.
    Non-blocking — starts background reader thread.
    Returns True if OAK-D found and pipeline started.
    """
    global _device, _oakd_ok, _reader_thread

    try:
        import depthai as dai

        pipeline = dai.Pipeline()

        # ── Mono cameras ───────────────────────────────────────────────────────
        mono_left  = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)

        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # ── Stereo depth ───────────────────────────────────────────────────────
        stereo = pipeline.create(dai.node.StereoDepth)
        # HIGH_DENSITY was added in newer depthai versions — fall back gracefully
        try:
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        except AttributeError:
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
        stereo.setLeftRightCheck(True)          # reduce artifacts
        stereo.setExtendedDisparity(False)
        stereo.setSubpixel(False)
        # Align depth to RGB socket so pixel coords match camera view
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

        # ── Output ─────────────────────────────────────────────────────────────
        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")

        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)
        stereo.depth.link(xout_depth.input)

        _device = dai.Device(pipeline)
        _oakd_ok = True
        log.info("✅ OAK-D Lite: stereo depth pipeline started")

        _reader_thread = threading.Thread(target=_reader_loop, daemon=True, name="oakd-reader")
        _reader_thread.start()
        return True

    except ImportError:
        log.warning("⚠️  DepthAI not installed — OAK-D disabled (pip install depthai)")
        _oakd_ok = False
        return False
    except Exception as e:
        log.warning(f"⚠️  OAK-D init failed ({e}) — depth perception disabled")
        _oakd_ok = False
        return False


def _reader_loop():
    """Background thread — continuously reads depth frames from OAK-D."""
    global _depth_frame
    try:
        q = _device.getOutputQueue("depth", maxSize=1, blocking=False)
        while True:
            try:
                in_depth = q.get()
                frame = in_depth.getFrame()   # numpy uint16, mm units
                with _lock:
                    _depth_frame = frame
            except Exception as e:
                log.warning(f"OAK-D reader frame error: {e}")
    except Exception as e:
        log.error(f"OAK-D reader loop died: {e}")


def get_depth_at(x_ratio: float = 0.5, y_ratio: float = 0.5,
                 patch_px: int = 10) -> float | None:
    """
    Return depth in METERS at a normalized image position.
    x_ratio: 0.0 (left)  → 1.0 (right)
    y_ratio: 0.0 (top)   → 1.0 (bottom)
    patch_px: half-size of sampling patch — median reduces noise.
    Returns None if OAK-D unavailable, frame not ready, or all pixels invalid.
    """
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

    valid = patch[patch > 0]   # 0 mm = invalid / no measurement
    if len(valid) == 0:
        return None

    return float(np.median(valid)) / 1000.0   # mm → meters


def get_front_depth() -> float | None:
    """
    Depth at center-bottom of frame — what is directly ahead of the robot.
    y=0.65 captures the ground-level view typical of the pan-tilt camera angle.
    """
    return get_depth_at(0.5, 0.65)


def get_depth_map() -> dict | None:
    """
    Return a coarse 3×3 depth grid (meters) covering the full frame.
    Keys: "top_left", "top_center", "top_right",
          "mid_left",  "mid_center",  "mid_right",
          "bot_left",  "bot_center",  "bot_right"
    Returns None if OAK-D not available.
    """
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
    """Return current OAK-D status for GUI / sensor_context()."""
    front = get_front_depth()
    return {
        "available": _oakd_ok,
        "front_m":   round(front, 2) if front is not None else None,
    }


def oakd_status_html() -> str:
    """HTML status display for Gradio (matches lidar_status_html style)."""
    s = get_status()

    if not s["available"]:
        return """
        <div style="background:#1a1a1a;border:1px solid #444;border-radius:8px;
                    padding:10px;font-family:monospace;color:#666">
            📷 OAK-D Lite: not connected
        </div>"""

    d = s["front_m"]
    if d is None:
        dist_str = "—"
        color = "#666"
        label = "No measurement"
    elif d < 0.30:
        dist_str = f"{d:.2f}m"
        color = "#cc0000"
        label = "🚧 VERY CLOSE"
    elif d < 0.60:
        dist_str = f"{d:.2f}m"
        color = "#ff6600"
        label = "⚠️  CLOSE"
    else:
        dist_str = f"{d:.2f}m"
        color = "#76b900"
        label = "✅ CLEAR"

    return f"""
    <div style="background:#1a1a1a;border:1px solid {color};border-radius:8px;
                padding:10px;font-family:monospace;">
        <div style="color:{color};font-weight:bold">📷 OAK-D Lite — {label}</div>
        <div style="color:#aaa;font-size:0.85em;margin-top:4px">
            Front depth: <span style="color:#fff">{dist_str}</span>
        </div>
    </div>"""