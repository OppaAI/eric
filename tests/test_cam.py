"""
Camera diagnostic — run this on the robot BEFORE starting controller.py

    python cam_test.py

Saves test.jpg if successful. Shows exactly what's failing.
"""

import time
import cv2
import numpy as np

def try_cap(index, backend, fourcc_str, w, h):
    tag = f"[cam{index} {backend.__name__ if hasattr(backend,'__name__') else backend} {fourcc_str}]"
    print(f"\n{tag} Opening...")
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        print(f"{tag} ❌ Failed to open")
        return None

    if fourcc_str:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"{tag} Opened — actual res={actual_w}x{actual_h} fps={actual_fps}")

    # Warm up — read and discard 10 frames
    print(f"{tag} Warming up (10 frames)...")
    for i in range(10):
        ret, frame = cap.read()
        status = "ok" if ret else "FAIL"
        mean = frame.mean() if (ret and frame is not None) else "N/A"
        print(f"  warm {i+1}/10: ret={status}  mean={mean:.1f}" if ret else f"  warm {i+1}/10: ret={status}")
        time.sleep(0.05)

    # Read 5 real frames
    print(f"{tag} Reading 5 frames...")
    good = 0
    for i in range(5):
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"  frame {i+1}: ❌ ret=False")
            continue
        m = frame.mean()
        shape = frame.shape
        print(f"  frame {i+1}: ✅ shape={shape}  mean={m:.2f}  min={frame.min()}  max={frame.max()}")
        if m > 2:
            good += 1
            if good == 1:
                # Save the first good frame
                cv2.imwrite("test.jpg", frame)
                print(f"  → Saved test.jpg")
        else:
            print(f"  frame {i+1}: ⚠️  BLACK FRAME (mean<2)")
        time.sleep(0.1)

    cap.release()
    return good

print("=" * 60)
print("ERIC Camera Diagnostic")
print("=" * 60)

# Test combinations
combos = [
    (0, cv2.CAP_V4L2, "MJPG", 640, 480),
    (0, cv2.CAP_V4L2, "YUYV", 640, 480),
    (0, cv2.CAP_V4L2, "",     640, 480),   # default fourcc
    (0, cv2.CAP_ANY,  "",     640, 480),   # let OpenCV pick backend
]

results = []
for args in combos:
    good = try_cap(*args)
    results.append((args, good))

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for args, good in results:
    idx, backend, fourcc, w, h = args
    name = {cv2.CAP_V4L2: "V4L2", cv2.CAP_ANY: "ANY"}.get(backend, str(backend))
    status = f"✅ {good}/5 good frames" if good else "❌ 0 good frames"
    print(f"  cam{idx} {name:6s} {fourcc or 'default':6s}  {status}")

print("\nIf test.jpg was saved, open it to verify the image.")
print("If all show 0 good frames → hardware/driver issue.")
print("If YUYV works but MJPG doesn't → change fourcc in controller.py")
print("If CAP_ANY works but V4L2 doesn't → remove CAP_V4L2 from controller.py")