#!/usr/bin/env python3
"""
ERIC — Cosmos Image Inference Benchmark
Runs ./test.jpg through Cosmos 10 times and calculates TPS.
Usage: python image_bench.py [--image path/to/image.jpg] [--iterations 10]
"""
import time
import json
import base64
import urllib.request
import argparse
import sys
import os

SERVER = "http://localhost:8000"
MODEL  = "embedl/Cosmos-Reason2-2B-W4A16"

PROMPT = (
    "You are ERIC, a search and rescue robot. "
    "Describe exactly what you see in this image. "
    "Identify: objects, people, terrain, obstacles, and what action you would take next. "
    "Be specific and concise — 3-4 sentences."
)


def load_image(path: str) -> tuple[str, str]:
    """Load image and return (base64_data, mime_type)."""
    if not os.path.exists(path):
        print(f"❌ Image not found: {path}")
        sys.exit(1)
    ext  = path.rsplit(".", 1)[-1].lower()
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png",  "webp": "image/webp"}.get(ext, "image/jpeg")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), mime


def infer(img_b64: str, mime: str) -> tuple[str, float, int, int, float]:
    """Run one inference. Returns (reply, elapsed, in_tok, out_tok, tps)."""
    payload = json.dumps({
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                {"type": "text", "text": PROMPT}
            ]
        }],
        "max_tokens": 200,
        "temperature": 0.0,
    }).encode()

    req = urllib.request.Request(
        f"{SERVER}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"}
    )

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        data = json.loads(r.read())
    t1 = time.perf_counter()

    elapsed   = t1 - t0
    reply     = data["choices"][0]["message"]["content"].strip()
    usage     = data.get("usage", {})
    in_tok    = usage.get("prompt_tokens", 0)
    out_tok   = usage.get("completion_tokens", 0)
    tps       = out_tok / elapsed if elapsed > 0 else 0

    return reply, elapsed, in_tok, out_tok, tps


def check_server():
    try:
        urllib.request.urlopen(f"{SERVER}/health", timeout=3)
        return True
    except:
        return False


def main():
    parser = argparse.ArgumentParser(description="Cosmos image inference benchmark")
    parser.add_argument("--image",      default="./test.jpg", help="Image path (default: ./test.jpg)")
    parser.add_argument("--iterations", type=int, default=10,  help="Number of iterations (default: 10)")
    args = parser.parse_args()

    print("=" * 65)
    print("  ERIC — Cosmos Image Inference Benchmark")
    print(f"  Model:      {MODEL}")
    print(f"  Server:     {SERVER}")
    print(f"  Image:      {args.image}")
    print(f"  Iterations: {args.iterations}")
    print("=" * 65)

    if not check_server():
        print("❌ Cosmos server not reachable. Start with: bash launch/cosmos.sh")
        sys.exit(1)
    print("✅ Server ready\n")

    print(f"Loading image: {args.image}")
    img_b64, mime = load_image(args.image)
    size_kb = len(img_b64) * 3 / 4 / 1024
    print(f"Image loaded: {size_kb:.1f} KB ({mime})\n")

    times      = []
    out_tokens = []
    in_tokens  = []

    for i in range(1, args.iterations + 1):
        print(f"Run {i:2d}/{args.iterations} — inferring...", end="", flush=True)

        try:
            reply, elapsed, in_tok, out_tok, tps = infer(img_b64, mime)
            times.append(elapsed)
            out_tokens.append(out_tok)
            in_tokens.append(in_tok)

            print(f"\r Run {i:2d}/{args.iterations} — {elapsed:.2f}s | in:{in_tok} out:{out_tok} tok | TPS: {tps:.1f}")

            # Show first response in full, truncate rest
            if i == 1:
                print(f"         Response: {reply}\n")
            else:
                print(f"         Response: {reply[:80]}{'...' if len(reply) > 80 else ''}\n")

        except Exception as e:
            print(f"\r Run {i:2d}/{args.iterations} — ❌ Error: {e}\n")
            continue

    if not times:
        print("No successful runs.")
        return

    # Summary statistics
    avg_time  = sum(times) / len(times)
    min_time  = min(times)
    max_time  = max(times)
    avg_out   = sum(out_tokens) / len(out_tokens)
    avg_tps   = avg_out / avg_time
    total_tps = sum(out_tokens) / sum(times)  # True throughput TPS

    # Exclude first run (warmup) if more than 1 run
    if len(times) > 1:
        warm_times  = times[1:]
        warm_tokens = out_tokens[1:]
        warm_avg    = sum(warm_tokens) / sum(warm_times)
    else:
        warm_avg = avg_tps

    print("=" * 65)
    print(f"  RESULTS — {len(times)}/{args.iterations} successful runs")
    print(f"  Avg time:         {avg_time:.2f}s")
    print(f"  Min / Max time:   {min_time:.2f}s / {max_time:.2f}s")
    print(f"  Avg output tokens:{avg_out:.0f}")
    print(f"  TPS (all runs):   {total_tps:.1f}")
    print(f"  TPS (ex warmup):  {warm_avg:.1f}")
    print(f"  Avg input tokens: {sum(in_tokens)/len(in_tokens):.0f}  (image + prompt)")
    print("=" * 65)


if __name__ == "__main__":
    main()