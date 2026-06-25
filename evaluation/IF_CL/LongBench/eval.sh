#!/usr/bin/env bash
set -euo pipefail
set -x

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$BENCH_DIR/.." && pwd)"
cd "$BENCH_DIR"

if [ -n "${CONDA_ENV:-}" ]; then
    if [ -n "${CONDA_SH:-}" ]; then
        source "$CONDA_SH"
    elif command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
    else
        echo "CONDA_ENV is set but conda is not available. Set CONDA_SH or unset CONDA_ENV." >&2
        exit 1
    fi
    conda activate "$CONDA_ENV"
fi
export PYTHONPATH="$BENCH_DIR:${PYTHONPATH:-}"

: "${MODEL_PATH:?MODEL_PATH not set}"
: "${MODEL_NAME:?MODEL_NAME not set}"

source "$ROOT_DIR/wait_vllm.sh"
start_vllm_and_wait "serve_vllm_${MODEL_NAME}.log" "${VLLM_PORT:-8000}" "${VLLM_SLEEP:-1800}" "$ROOT_DIR/scripts/serve_vllm.sh" || exit 1

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_COUNT="${LONGBENCH_RUN_COUNT:-3}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/../datasets/data}"
DATA_DIR="${LONGBENCH_DATA_DIR:-$DATA_ROOT/longbench_v2}"
WORKERS="${LONGBENCH_WORKERS:-32}"

for run_id in $(seq 1 "$RUN_COUNT"); do
    "$PYTHON_BIN" pred.py \
        --cot \
        --model "$MODEL_NAME" \
        --output_model "${MODEL_NAME}_run${run_id}" \
        --data_dir "$DATA_DIR" \
        --tokenizer "${TOKENIZER_PATH:-$MODEL_PATH}" \
        --base_url "http://localhost:${VLLM_PORT:-8000}/v1" \
        --api_key "${OPENAI_API_KEY:-EMPTY}" \
        --workers "$WORKERS"
done

"$PYTHON_BIN" result.py --average_model "$MODEL_NAME" --run_count "$RUN_COUNT" --result_suffix cot
