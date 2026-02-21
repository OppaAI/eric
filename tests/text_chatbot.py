#!/usr/bin/env python3
"""
ERIC — Cosmos Chatbot TPS Benchmark
Simple interactive chatbot loop that measures tokens per second.
Usage: python chat_bench.py
"""
import time
import json
import urllib.request
import sys

SERVER = "http://localhost:8000"
MODEL  = "embedl/Cosmos-Reason2-2B-W4A16"

SYSTEM = (
    "You are ERIC — Edge Robotics Innovation by Cosmos. "
    "A search and rescue tracked ground robot running on a Jetson Orin Nano. "
    "Be helpful, concise, and natural."
)

history = []


def chat(user_msg: str, max_tokens: int = 300):
    history.append({"role": "user", "content": user_msg})

    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM}] + history,
        "max_tokens": max_tokens,
        "temperature": 0.7,
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
    total_tok = usage.get("total_tokens", 0)
    tps       = out_tok / elapsed if elapsed > 0 else 0

    history.append({"role": "assistant", "content": reply})
    return reply, elapsed, in_tok, out_tok, tps


def check_server():
    try:
        urllib.request.urlopen(f"{SERVER}/health", timeout=3)
        return True
    except:
        return False


def main():
    print("=" * 60)
    print("  ERIC — Cosmos Chatbot TPS Benchmark")
    print(f"  Model:  {MODEL}")
    print(f"  Server: {SERVER}")
    print("=" * 60)

    if not check_server():
        print("❌ Cosmos server not reachable. Start with: bash launch/cosmos.sh")
        sys.exit(1)

    print("✅ Server ready. Type your message (or 'quit' to exit, 'reset' to clear history)\n")

    session_times  = []
    session_tokens = []
    turn = 0

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "reset":
            history.clear()
            session_times.clear()
            session_tokens.clear()
            turn = 0
            print("🔄 History cleared.\n")
            continue

        turn += 1
        print(f"\nEric [thinking...]", end="", flush=True)

        try:
            reply, elapsed, in_tok, out_tok, tps = chat(user_input)
        except Exception as e:
            print(f"\r❌ Error: {e}\n")
            continue

        session_times.append(elapsed)
        session_tokens.append(out_tok)

        # Clear the thinking line and print reply
        print(f"\rEric: {reply}")
        print(
            f"  ⏱  {elapsed:.2f}s | "
            f"in:{in_tok} out:{out_tok} tok | "
            f"TPS: {tps:.1f} | "
            f"turn #{turn}"
        )

        if len(session_times) > 1:
            avg_tps = sum(session_tokens) / sum(session_times)
            print(f"  📊 Session avg TPS: {avg_tps:.1f} over {turn} turns")
        print()

    # Final summary
    if session_times:
        avg_tps  = sum(session_tokens) / sum(session_times)
        avg_time = sum(session_times) / len(session_times)
        print("\n" + "=" * 60)
        print(f"  SESSION SUMMARY — {turn} turns")
        print(f"  Avg TPS:          {avg_tps:.1f}")
        print(f"  Avg response time:{avg_time:.2f}s")
        print(f"  Total tokens out: {sum(session_tokens)}")
        print("=" * 60)


if __name__ == "__main__":
    main()