#!/usr/bin/env bash
set -euo pipefail

# -------- config --------
IMAGE_NAME="embedding-cuda"
CONTAINER_NAME="embedding-cuda"

# Resolve project root (two levels up from this script)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_CACHE="${PROJECT_ROOT}/hf_cache"

DOCKERFILE="${PROJECT_ROOT}/docker/cuda/Dockerfile"

# -------- build image --------
echo "Building Docker image..."
docker build \
  -t "${IMAGE_NAME}" \
  -f "${DOCKERFILE}" \
  "${PROJECT_ROOT}"

# -------- run container --------
echo "Running container..."
docker run --rm \
  --name "${CONTAINER_NAME}" \
  --gpus '"device=0"' \
  -it \
  -w /app \
  -v "${PROJECT_ROOT}:/app" \
  -v "${HF_CACHE}:/hf_cache" \
  -e HF_HOME=/hf_cache \
  -e HF_DATASETS_CACHE=/hf_cache/datasets \
  -e TRANSFORMERS_CACHE=/hf_cache/transformers \
  -e TMPDIR=/hf_cache/tmp \
  -e DEVICE=cuda \
  "${IMAGE_NAME}" \
  python3 main.py
