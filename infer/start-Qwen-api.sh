#!/usr/bin/env bash
set -e

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# GPU0
CUDA_VISIBLE_DEVICES=0 python -m uvicorn Qwen-Image-Edit-2509-api:app \
  --host 0.0.0.0 --port 8001 &

# GPU1
CUDA_VISIBLE_DEVICES=1 python -m uvicorn Qwen-Image-Edit-2509-api:app \
  --host 0.0.0.0 --port 8002 &

# GPU2
CUDA_VISIBLE_DEVICES=2 python -m uvicorn Qwen-Image-Edit-2509-api:app \
  --host 0.0.0.0 --port 8003 &

# GPU3
CUDA_VISIBLE_DEVICES=3 python -m uvicorn Qwen-Image-Edit-2509-api:app \
  --host 0.0.0.0 --port 8004 &

wait
