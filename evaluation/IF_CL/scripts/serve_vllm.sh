#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?MODEL_PATH is required}"
: "${MODEL_NAME:?MODEL_NAME is required}"

PYTHON_BIN="${PYTHON_BIN:-python}"
PORT="${VLLM_PORT:-${PORT:-8000}}"
TP_SIZE="${TP_SIZE:-${TENSOR_PARALLEL_SIZE:-1}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-150000}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
DTYPE="${DTYPE:-auto}"

export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"

cmd=(
    "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server
    --model "$MODEL_PATH"
    --served-model-name "$MODEL_NAME"
    --host "${VLLM_HOST:-0.0.0.0}"
    --port "$PORT"
    --tensor-parallel-size "$TP_SIZE"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --dtype "$DTYPE"
    --trust-remote-code
    --max-model-len "$MAX_MODEL_LEN"
    --enable-chunked-prefill
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --max-num-seqs "$MAX_NUM_SEQS"
)

if [ "${VLLM_LANGUAGE_MODEL_ONLY:-1}" = "1" ]; then
    cmd+=(--language-model-only)
fi
if [ -n "${VLLM_REASONING_PARSER:-}" ]; then
    cmd+=(--reasoning-parser "$VLLM_REASONING_PARSER")
fi
if [ -n "${VLLM_HF_OVERRIDES:-}" ]; then
    cmd+=(--hf-overrides "$VLLM_HF_OVERRIDES")
fi

# shellcheck disable=SC2206
extra_args=(${VLLM_EXTRA_ARGS:-})
cmd+=("${extra_args[@]}")

exec "${cmd[@]}"
