#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage:
  bash eval_long.sh --model_name <served-name> --model_path <hf-path>

Environment:
  BENCHMARKS              Space-separated benchmark list. Default: "longbench ifbench ifeval"
  TOKENIZER_PATH          Tokenizer path/name. Default: MODEL_PATH
  VLLM_SLEEP              vLLM readiness timeout in seconds. Default: 500
  LONGBENCH_TP_SIZE       Tensor parallel size for LongBench. Default: 8
  IFBENCH_TP_SIZE         Tensor parallel size for IFBench. Default: 4
  IFEVAL_TP_SIZE          Tensor parallel size for IFEval. Default: 4
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME=""
MODEL_PATH=""

while [ $# -gt 0 ]; do
    case "$1" in
        --model_name) MODEL_NAME="$2"; shift 2 ;;
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [ -z "$MODEL_NAME" ] || [ -z "$MODEL_PATH" ]; then
    usage
    exit 1
fi

export MODEL_NAME MODEL_PATH
export TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
export VLLM_SLEEP="${VLLM_SLEEP:-500}"

run_benchmark() {
    local name="$1"
    local script="$2"
    local tp_size="$3"

    echo "[eval_long] running ${name} with TP_SIZE=${tp_size}"
    (
        export TP_SIZE="$tp_size"
        bash -ex "$ROOT_DIR/$script"
    )
}

for benchmark in ${BENCHMARKS:-longbench ifbench ifeval}; do
    case "$benchmark" in
        longbench)
            run_benchmark longbench LongBench/eval.sh "${LONGBENCH_TP_SIZE:-8}"
            ;;
        ifbench)
            run_benchmark ifbench IFBench/eval.sh "${IFBENCH_TP_SIZE:-4}"
            ;;
        ifeval)
            run_benchmark ifeval IFEval/eval.sh "${IFEVAL_TP_SIZE:-4}"
            ;;
        *)
            echo "Unknown benchmark: $benchmark" >&2
            echo "Allowed values: longbench ifbench ifeval" >&2
            exit 1
            ;;
    esac
done
