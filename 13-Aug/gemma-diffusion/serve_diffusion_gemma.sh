#!/usr/bin/env bash
# Serve DiffusionGemma 26B-A4B-IT through vLLM's OpenAI-compatible API.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-/spinning/tnguyenho/llm/diffusiongemma-26B-A4B-it}"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL")}"
PORT="${PORT:-8000}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
API_KEY="${API_KEY:-14June2026}"
UV_BIN="${UV_BIN:-}"

if [[ -z "$UV_BIN" ]]; then
    if command -v uv >/dev/null 2>&1; then
        UV_BIN="$(command -v uv)"
    elif [[ -x /home/tnguyenho/miniforge3/bin/uv ]]; then
        UV_BIN=/home/tnguyenho/miniforge3/bin/uv
    else
        echo "error: uv was not found" >&2
        exit 1
    fi
fi

if [[ ! -d "$MODEL" ]]; then
    echo "error: model directory not found: $MODEL" >&2
    exit 1
fi

# Miniforge's libstdc++ is needed by scipy pulled in with transformers 5.x.
export LD_LIBRARY_PATH="/home/tnguyenho/miniforge3/lib:${LD_LIBRARY_PATH:-}"
# Diffusion sampling needs additional allocation headroom during warmup.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec "$UV_BIN" run --project "$SCRIPT_DIR" vllm serve "$MODEL" \
    --served-model-name "$MODEL_NAME" \
    --host "$VLLM_HOST" \
    --port "$PORT" \
    --gpu-memory-utilization 0.8 \
    --max-model-len 64000 \
    --limit-mm-per-prompt '{"image": 100}' \
    --data-parallel-size 2 \
    --dtype bfloat16 \
    --enforce-eager \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    "$@"
