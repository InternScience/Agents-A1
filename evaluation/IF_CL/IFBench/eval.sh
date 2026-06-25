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
export NLTK_DATA="${NLTK_DATA:-$BENCH_DIR/.nltk_data}"
mkdir -p "$NLTK_DATA" outputs/responses outputs/eval
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/../datasets/data}"
INPUT_FILE="${IFBENCH_DATA_FILE:-$DATA_ROOT/ifbench/IFBench_test.jsonl}"

if [ "${AUTO_DOWNLOAD_NLTK:-0}" = "1" ]; then
    "$PYTHON_BIN" "$ROOT_DIR/scripts/download_nltk_data.py"
fi

: "${MODEL_PATH:?MODEL_PATH not set}"
: "${MODEL_NAME:?MODEL_NAME not set}"

source "$ROOT_DIR/wait_vllm.sh"
start_vllm_and_wait "serve_vllm_${MODEL_NAME}.log" "${VLLM_PORT:-8000}" "${VLLM_SLEEP:-1800}" "$ROOT_DIR/scripts/serve_vllm.sh" || exit 1

generate_cmd=(
    "$PYTHON_BIN"
    generate_responses.py
    --api-base "http://localhost:${VLLM_PORT:-8000}/v1"
    --model "$MODEL_NAME"
    --input-file "$INPUT_FILE"
    --output-file outputs/responses/"$MODEL_NAME".jsonl
    --workers "${IFBENCH_WORKERS:-64}"
    --resume
)
"${generate_cmd[@]}"

"$PYTHON_BIN" -m run_eval \
    --input_data "$INPUT_FILE" \
    --input_response_data outputs/responses/"$MODEL_NAME".jsonl \
    --output_dir outputs/eval
