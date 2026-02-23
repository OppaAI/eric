#!/usr/bin/env python3
"""
bench_cosmos.py — Cosmos Reason 2 video inference speed benchmark
Tests vision inference using frames sampled from a local .mp4 file.

Usage:
    uv run bench_cosmos.py [--video test.mp4] [--runs 10] [--fps 1] [--frames 5]

Metrics reported per run:
  - Time to first token (TTFT)
  - Total inference time
  - Tokens per second (TPS)
  - Total tokens generated

Summary: mean, min, max, stddev across all runs.
"""

import argparse
import base64
import json
import statistics
import sys
import time

import cv2
import requests

# ─── Config ───────────────────────────────────────────────────────────────────
DEFAULT_VLLM_URL    = "http://localhost:8000/v1/chat/completions"
DEFAULT_MODEL       = "embedl/Cosmos-Reason2-2B-W4A16"
DEFAULT_VIDEO       = "./test.mp4"
DEFAULT_RUNS        = 10
DEFAULT_FPS_SAMPLE  = 1.0   # frames per second to sample from video
DEFAULT_MAX_FRAMES  = 5     # max frames sent per inference call
DEFAULT_MAX_TOKENS  = 150

PROMPT = """
These are sampled frames from a video clip captured by a ground robot camera.
Analyze what you observe across the frames.

Respond ONLY with valid JSON:
{
  "objects_seen": ["list of objects or people visible"],
  "motion": "stationary | slow | fast | approaching | retreating",
  "terrain": "tiles | carpet | pavement | grass | wood | unknown",
  "hazards": ["list of hazards or obstacles, empty list if none"],
  "summary": "1 sentence describing the overall scene and any notable changes across frames"
}
"""


# ─── Video sampling ───────────────────────────────────────────────────────────

def sample_frames(video_path: str, fps_sample: float, max_frames: int) -> list[str]:
    """Sample frames from video, return as list of base64 JPEGs."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌  Cannot open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    video_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = max(1, int(video_fps / fps_sample))
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration   = total / video_fps

    print(f"📹  Video: {video_path}")
    print(f"    {total} frames @ {video_fps:.1f} fps → {duration:.1f}s")
    print(f"    Sampling every {frame_step} frames ({fps_sample} fps) → up to {max_frames} frames sent")

    frames = []
    idx    = 0
    while len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break

        # Resize to stay within Cosmos 256k pixel budget
        h, w = frame.shape[:2]
        max_px = 256_000
        if w * h > max_px:
            scale = (max_px / (w * h)) ** 0.5
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frames.append(base64.b64encode(buf).decode("utf-8"))
        idx += frame_step

    cap.release()
    print(f"    Extracted {len(frames)} frames\n")
    return frames


# ─── Single inference run ─────────────────────────────────────────────────────

def run_inference(frames: list[str], url: str, model: str,
                  max_tokens: int, run_num: int) -> dict:
    """Send frames to Cosmos, measure TTFT and total time. Returns metrics dict."""

    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}}
        for f in frames
    ]
    content.append({"type": "text", "text": PROMPT.strip()})

    payload = {
        "model":              model,
        "messages":           [{"role": "user", "content": content}],
        "max_tokens":         max_tokens,
        "temperature":        0.1,
        "repetition_penalty": 1.15,
        "stream":             True,
    }

    print(f"  Run {run_num:2d}/{run_num} → ", end="", flush=True)

    t_start     = time.perf_counter()
    t_first     = None
    tokens      = []
    full_text   = []

    try:
        with requests.post(url, json=payload, stream=True, timeout=120) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    line = line[6:]
                if line == "[DONE]":
                    break
                try:
                    chunk = json.loads(line)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        if t_first is None:
                            t_first = time.perf_counter()
                        full_text.append(delta)
                        # Rough token count — vLLM doesn't stream token counts
                        tokens.append(delta)
                except Exception:
                    continue

    except requests.exceptions.ConnectionError:
        print("❌  Cannot connect to vLLM. Is it running? (bash launch/cosmos.sh)")
        sys.exit(1)
    except Exception as e:
        print(f"❌  Request error: {e}")
        return None

    t_end        = time.perf_counter()
    text         = "".join(full_text)
    total_time   = t_end - t_start
    ttft         = (t_first - t_start) if t_first else total_time

    # Approximate token count from word count (rough but consistent)
    token_count  = len(text.split())
    tps          = token_count / total_time if total_time > 0 else 0

    print(f"TTFT {ttft:.2f}s  |  total {total_time:.2f}s  |  ~{token_count} tokens  |  {tps:.1f} tok/s")

    # Try to parse JSON response
    try:
        clean = text.replace("```json", "").replace("```", "").strip()
        s = clean.find("{"); e = clean.rfind("}") + 1
        parsed = json.loads(clean[s:e]) if s >= 0 and e > s else None
    except Exception:
        parsed = None

    return {
        "run":          run_num,
        "ttft_s":       round(ttft, 3),
        "total_s":      round(total_time, 3),
        "tokens":       token_count,
        "tps":          round(tps, 2),
        "text":         text[:300],
        "parsed_ok":    parsed is not None,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cosmos Reason 2 video inference benchmark")
    parser.add_argument("--video",    default=DEFAULT_VIDEO,      help="Path to .mp4 file")
    parser.add_argument("--runs",     type=int, default=DEFAULT_RUNS,       help="Number of inference runs")
    parser.add_argument("--fps",      type=float, default=DEFAULT_FPS_SAMPLE, help="Frames per second to sample")
    parser.add_argument("--frames",   type=int, default=DEFAULT_MAX_FRAMES,  help="Max frames per call")
    parser.add_argument("--tokens",   type=int, default=DEFAULT_MAX_TOKENS,  help="Max output tokens")
    parser.add_argument("--url",      default=DEFAULT_VLLM_URL,   help="vLLM endpoint URL")
    parser.add_argument("--model",    default=DEFAULT_MODEL,       help="Model name")
    args = parser.parse_args()

    print("=" * 60)
    print("  Cosmos Reason 2 — Video Inference Benchmark")
    print("=" * 60)
    print(f"  Model:   {args.model}")
    print(f"  URL:     {args.url}")
    print(f"  Runs:    {args.runs}")
    print(f"  Frames:  up to {args.frames} per call @ {args.fps} fps sample")
    print(f"  Tokens:  max {args.tokens} output")
    print("=" * 60 + "\n")

    # Sample frames once — reuse across all runs (same input, measures pure inference)
    frames = sample_frames(args.video, args.fps, args.frames)

    print(f"Running {args.runs} inference calls...\n")
    results = []
    for i in range(1, args.runs + 1):
        result = run_inference(frames, args.url, args.model, args.tokens, i)
        if result:
            results.append(result)
        time.sleep(0.5)  # brief pause between runs

    if not results:
        print("\n❌  No successful runs.")
        sys.exit(1)

    # ── Summary statistics ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)

    for metric, label, unit in [
        ("ttft_s",  "Time to first token (TTFT)", "s"),
        ("total_s", "Total inference time",        "s"),
        ("tps",     "Tokens per second",           "tok/s"),
        ("tokens",  "Output tokens",               ""),
    ]:
        vals = [r[metric] for r in results]
        mean = statistics.mean(vals)
        mn   = min(vals)
        mx   = max(vals)
        std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
        unit_str = f" {unit}" if unit else ""
        print(f"\n  {label}:")
        print(f"    mean={mean:.2f}{unit_str}  min={mn:.2f}{unit_str}  max={mx:.2f}{unit_str}  σ={std:.2f}")

    parsed_ok = sum(1 for r in results if r["parsed_ok"])
    print(f"\n  JSON parse success: {parsed_ok}/{len(results)}")
    print(f"  Successful runs:    {len(results)}/{args.runs}")

    print("\n" + "=" * 60)
    print("  PER-RUN TABLE")
    print("  {:>4}  {:>8}  {:>8}  {:>7}  {:>8}  {}".format(
        "Run", "TTFT(s)", "Total(s)", "Tokens", "Tok/s", "JSON"))
    print("  " + "-" * 52)
    for r in results:
        print("  {:>4}  {:>8.2f}  {:>8.2f}  {:>7}  {:>8.2f}  {}".format(
            r["run"], r["ttft_s"], r["total_s"], r["tokens"],
            r["tps"], "✅" if r["parsed_ok"] else "❌"))
    print("=" * 60)

    # Show last response as sample
    if results:
        print("\n  Sample response (last run):")
        print("  " + results[-1]["text"][:400].replace("\n", "\n  "))
        print()


if __name__ == "__main__":
    main()