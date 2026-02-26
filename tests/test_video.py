#!/usr/bin/env python3
"""
describe_cosmos.py — Full video description using Cosmos Reason 2
Samples frames across the entire video, processes them in batches,
and prints a plain-text description of each segment plus a full summary.

Usage:
    uv run describe_cosmos.py [--video test.mp4] [--fps 0.5] [--batch 4]
"""

import argparse
import base64
import json
import sys
import time

import cv2
import requests

# ─── Config ───────────────────────────────────────────────────────────────────
DEFAULT_VLLM_URL   = "http://localhost:8000/v1/chat/completions"
DEFAULT_MODEL      = "embedl/Cosmos-Reason2-2B-W4A16-Edge2"
DEFAULT_VIDEO      = "./test.mp4"
DEFAULT_FPS_SAMPLE = 0.5    # 1 frame every 2 seconds
DEFAULT_BATCH_SIZE = 4      # frames per LLM call (keep low to avoid 400s)
DEFAULT_MAX_TOKENS = 300
JPEG_QUALITY       = 70
MAX_IMAGE_PX       = 65_536  # ~256x256 max per frame

SEGMENT_PROMPT = (
    "These are frames sampled from a video between {start}s and {end}s. "
    "In 2-3 sentences, describe what you see: the setting, any people or objects, "
    "and what appears to be happening. Write naturally, no lists or JSON."
)

FINAL_PROMPT = (
    "Below are descriptions of consecutive segments of a video. "
    "Write a single flowing paragraph describing the entire video from start to finish, "
    "as if narrating it to someone who hasn't seen it. "
    "Cover the setting, people, objects, and how things change over time. "
    "Prose only — no bullet points, no headers.\n\nSegments:\n{segments}"
)


# ─── Video sampling ───────────────────────────────────────────────────────────

def sample_all_frames(video_path: str, fps_sample: float):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌  Cannot open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    video_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = max(1, int(video_fps / fps_sample))
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration   = total / video_fps

    print(f"📹  Video: {video_path}")
    print(f"    {total} frames @ {video_fps:.1f} fps → {duration:.1f}s ({duration/60:.1f} min)")
    print(f"    Sampling every {frame_step} frames ({fps_sample} fps)\n")

    frames, timestamps = [], []
    idx = 0
    while idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        if w * h > MAX_IMAGE_PX:
            scale = (MAX_IMAGE_PX / (w * h)) ** 0.5
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        frames.append(base64.b64encode(buf).decode("utf-8"))
        timestamps.append(round(idx / video_fps, 1))
        idx += frame_step

    cap.release()
    print(f"    Extracted {len(frames)} frames total\n")
    return frames, timestamps


# ─── LLM helpers ─────────────────────────────────────────────────────────────

def call_llm(content: list, url: str, model: str, max_tokens: int) -> str:
    payload = {
        "model":       model,
        "messages":    [{"role": "user", "content": content}],
        "max_tokens":  max_tokens,
        "temperature": 0.2,
        "stream":      True,
    }
    full_text = []
    try:
        with requests.post(url, json=payload, stream=True, timeout=180) as r:
            if not r.ok:
                try:
                    err = r.json()
                except Exception:
                    err = r.text[:300]
                print(f"\n  ❌  HTTP {r.status_code}: {err}")
                return ""
            for line in r.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    line = line[6:]
                if line == "[DONE]":
                    break
                try:
                    delta = json.loads(line)["choices"][0]["delta"].get("content", "")
                    if delta:
                        full_text.append(delta)
                except Exception:
                    continue
    except requests.exceptions.ConnectionError:
        print("❌  Cannot connect to vLLM. Is it running?")
        sys.exit(1)
    except Exception as e:
        print(f"❌  Request error: {e}")
        return ""
    return "".join(full_text)


def call_llm_with_retry(content: list, url: str, model: str, max_tokens: int) -> str:
    result = call_llm(content, url, model, max_tokens)
    if result:
        return result
    images = [c for c in content if c.get("type") == "image_url"]
    texts  = [c for c in content if c.get("type") == "text"]
    if len(images) > 2:
        print(f"  🔁  Retrying with 2 frames instead of {len(images)}...")
        reduced = images[::max(1, len(images) // 2)][:2] + texts
        result = call_llm(reduced, url, model, max_tokens)
    return result


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Full video description with Cosmos Reason 2")
    parser.add_argument("--video",  default=DEFAULT_VIDEO)
    parser.add_argument("--fps",    type=float, default=DEFAULT_FPS_SAMPLE)
    parser.add_argument("--batch",  type=int,   default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--tokens", type=int,   default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--url",    default=DEFAULT_VLLM_URL)
    parser.add_argument("--model",  default=DEFAULT_MODEL)
    args = parser.parse_args()

    print("=" * 65)
    print("  Cosmos Reason 2 — Full Video Description")
    print("=" * 65)
    print(f"  Model: {args.model}")
    print(f"  fps sample: {args.fps}  |  batch: {args.batch} frames")
    print("=" * 65 + "\n")

    frames, timestamps = sample_all_frames(args.video, args.fps)

    batches = [
        (frames[i:i + args.batch], timestamps[i:i + args.batch])
        for i in range(0, len(frames), args.batch)
    ]

    print(f"Processing {len(batches)} segments...\n")
    print("─" * 65)

    segment_texts = []

    for seg_idx, (batch_frames, batch_times) in enumerate(batches, 1):
        t_start   = batch_times[0]
        t_end     = batch_times[-1]
        approx_kb = sum(len(f) * 3 // 4 for f in batch_frames) // 1024
        print(f"\n🎬  Segment {seg_idx}/{len(batches)}  [{t_start}s – {t_end}s]  (~{approx_kb} KB)", flush=True)

        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}}
            for f in batch_frames
        ]
        content.append({"type": "text", "text": SEGMENT_PROMPT.format(start=t_start, end=t_end)})

        t0      = time.perf_counter()
        text    = call_llm_with_retry(content, args.url, args.model, args.tokens)
        elapsed = time.perf_counter() - t0
        text    = text.strip()

        if text:
            print(f"  ✅  {elapsed:.1f}s")
            print(f"  {text}")
            segment_texts.append(f"[{t_start}s–{t_end}s] {text}")
        else:
            print(f"  ❌  No response after {elapsed:.1f}s")

        time.sleep(0.3)

    # ── Final full-video narrative ────────────────────────────────────────────
    if segment_texts:
        print("\n" + "=" * 65)
        print("  FULL VIDEO DESCRIPTION")
        print("=" * 65 + "\n")

        final_content = [{"type": "text", "text": FINAL_PROMPT.format(segments="\n".join(segment_texts))}]
        t0         = time.perf_counter()
        final_text = call_llm_with_retry(final_content, args.url, args.model, max_tokens=600)
        elapsed    = time.perf_counter() - t0

        print(final_text.strip())
        print(f"\n⏱️  Generated in {elapsed:.1f}s")

        # Save plain text log
        out_path = "video_description.txt"
        with open(out_path, "w") as f:
            f.write("SEGMENT DESCRIPTIONS\n" + "=" * 40 + "\n\n")
            f.write("\n\n".join(segment_texts))
            f.write("\n\n\nFULL VIDEO DESCRIPTION\n" + "=" * 40 + "\n\n")
            f.write(final_text.strip() + "\n")
        print(f"\n💾  Saved to: {out_path}")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()