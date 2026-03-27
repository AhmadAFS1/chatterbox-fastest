#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "Missing required tool: uv" >&2
  echo "Install it first: https://docs.astral.sh/uv/" >&2
  exit 1
fi

if [[ ! -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
  echo "[setup] Creating virtual environment..."
  uv venv
fi

echo "[setup] Syncing dependencies..."
uv sync

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export CHATTERBOX_MODEL_VARIANT="${CHATTERBOX_MODEL_VARIANT:-multilingual}"
export CHATTERBOX_GPU_MEMORY_UTILIZATION="${CHATTERBOX_GPU_MEMORY_UTILIZATION:-0.8}"
export CHATTERBOX_MAX_BATCH_SIZE="${CHATTERBOX_MAX_BATCH_SIZE:-10}"
export CHATTERBOX_MAX_MODEL_LEN="${CHATTERBOX_MAX_MODEL_LEN:-10000}"
export CHATTERBOX_SPLIT_SENTENCES_DEFAULT="${CHATTERBOX_SPLIT_SENTENCES_DEFAULT:-true}"
export CHATTERBOX_CONDS_CACHE_SIZE="${CHATTERBOX_CONDS_CACHE_SIZE:-32}"
export CHATTERBOX_COMPILE="${CHATTERBOX_COMPILE:-false}"
export CHATTERBOX_API_HOST="${CHATTERBOX_API_HOST:-0.0.0.0}"
export CHATTERBOX_API_PORT="${CHATTERBOX_API_PORT:-8000}"

echo "[startup] Chatterbox TTS"
echo "[startup] repo=$SCRIPT_DIR"
echo "[startup] host=${CHATTERBOX_API_HOST} port=${CHATTERBOX_API_PORT}"
echo "[startup] variant=${CHATTERBOX_MODEL_VARIANT} gpu_mem=${CHATTERBOX_GPU_MEMORY_UTILIZATION}"

exec "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/tts_api_server.py"
