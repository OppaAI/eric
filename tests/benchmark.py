#!/usr/bin/env python3
"""
E.R.I.C. — Cosmos Inference Benchmark
Times text and image inference against local vllm server
Usage: python bench.py
"""
import time
import base64
import json
import urllib.request
import urllib.error
import argparse
import sys

SERVER = "http://localhost:8000"
MODEL  = "embedl/Cosmos-Reason2-2B-W4A16"

def post(endpoint, payload):
    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        f"{SERVER}{endpoint}",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())

def bench_text(prompt, runs=3):
    print(f"\n{'='*50}")
    print(f"TEXT BENCHMARK ({runs} runs)")
    print(f"Prompt: {prompt[:80]}...")
    print('='*50)

    times = []
    tokens = []

    for i in range(runs):
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.0
        }
        t0 = time.perf_counter()
        resp = post("/v1/chat/completions", payload)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        usage   = resp.get("usage", {})
        out_tok = usage.get("completion_tokens", 0)
        tps     = out_tok / elapsed if elapsed > 0 else 0

        times.append(elapsed)
        tokens.append(out_tok)

        print(f"  Run {i+1}: {elapsed:.2f}s | {out_tok} tokens | {tps:.1f} TPS")
        if i == 0:
            content = resp["choices"][0]["message"]["content"]
            print(f"  Response: {content[:150]}...")

    avg_time = sum(times) / len(times)
    avg_tok  = sum(tokens) / len(tokens)
    avg_tps  = avg_tok / avg_time
    print(f"\n  AVG: {avg_time:.2f}s | {avg_tok:.0f} tokens | {avg_tps:.1f} TPS")


def bench_image(image_path, prompt="Describe this image briefly.", runs=2):
    print(f"\n{'='*50}")
    print(f"IMAGE BENCHMARK ({runs} runs)")
    print(f"Image: {image_path}")
    print(f"Prompt: {prompt}")
    print('='*50)

    # Load and encode image
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = image_path.rsplit(".", 1)[-1].lower()
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
    except FileNotFoundError:
        print(f"  ❌ Image not found: {image_path}")
        return

    times = []
    tokens = []

    for i in range(runs):
        payload = {
            "model": MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                    {"type": "text", "text": prompt}
                ]
            }],
            "max_tokens": 200,
            "temperature": 0.0
        }
        t0 = time.perf_counter()
        resp = post("/v1/chat/completions", payload)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        usage   = resp.get("usage", {})
        out_tok = usage.get("completion_tokens", 0)
        tps     = out_tok / elapsed if elapsed > 0 else 0

        times.append(elapsed)
        tokens.append(out_tok)

        print(f"  Run {i+1}: {elapsed:.2f}s | {out_tok} tokens | {tps:.1f} TPS")
        if i == 0:
            content = resp["choices"][0]["message"]["content"]
            print(f"  Response: {content[:150]}...")

    avg_time = sum(times) / len(times)
    avg_tok  = sum(tokens) / len(tokens)
    avg_tps  = avg_tok / avg_time
    print(f"\n  AVG: {avg_time:.2f}s | {avg_tok:.0f} tokens | {avg_tps:.1f} TPS")


def check_server():
    try:
        req = urllib.request.Request(f"{SERVER}/health")
        with urllib.request.urlopen(req, timeout=5):
            return True
    except:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cosmos inference benchmark")
    parser.add_argument("--image", type=str, default=None, help="Path to image file for vision benchmark")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs (default: 3)")
    parser.add_argument("--text-only", action="store_true", help="Skip image benchmark")
    args = parser.parse_args()

    print("🚀 Cosmos Inference Benchmark")
    print(f"   Server: {SERVER}")
    print(f"   Model:  {MODEL}")

    if not check_server():
        print("\n❌ Server not reachable at", SERVER)
        print("   Make sure Cosmos is running: docker logs -f vllm-server")
        sys.exit(1)

    print("   ✅ Server reachable\n")

    # Text benchmark
    bench_text(
        prompt="Explain in detail what a robot needs to navigate autonomously in an indoor environment.",
        runs=args.runs
    )

    # Image benchmark
    if not args.text_only:
        image_path = args.image or "/home/oppa-ai/ugv_jetson/templates/pictures/photo_2025-12-01_22-41-24.jpg"
        bench_image(image_path, runs=min(args.runs, 2))

    print("\n✅ Benchmark complete!")
