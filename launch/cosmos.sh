#!/bin/bash
# E.R.I.C. — Launch Cosmos Reason 2 via vLLM on Jetson Orin Nano
# Run this first and wait ~3 minutes before starting main.py

docker stop vllm-server 2>/dev/null
docker rm vllm-server 2>/dev/null

docker run -d \
  --restart unless-stopped \
  --network host \
  --shm-size=8g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --runtime=nvidia \
  --name=vllm-server \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -e HF_HOME=/root/.cache/huggingface \
  ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin \
  vllm serve "embedl/Cosmos-Reason2-2B-W4A16" \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.68 \
  --max-num-seqs 1 \
  --mm-processor-kwargs '{"max_pixels":256000}'

echo "⏳ Cosmos loading (~3 minutes)..."
echo "Monitor: docker logs -f vllm-server"
echo "Ready when you see: Application startup complete"
