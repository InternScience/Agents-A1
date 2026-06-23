#!/bin/bash
# run.sh - Main entry point for inference and evaluation
#
# Usage:
#   bash run.sh <model_path> [dataset_path]
#
# Examples:
#   bash run.sh /path/to/model                     # Uses default GAIA dataset
#   bash run.sh /path/to/model /path/to/data.jsonl  # Custom dataset
#
# Environment variables are loaded from .env file.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment variables from .env
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

MODEL_NAME=${1:-$MODEL_NAME}
MODEL_PATH=${1:-$MODEL_PATH}
DATASET_PATH=${2:-${DATASET:-./datasets/data/gaia-2023-validation-text-103/standardized_data.jsonl}}

if [ -z "$MODEL_PATH" ] || [ "$MODEL_PATH" = "/path/to/your/model" ]; then
    echo "Error: MODEL_PATH not provided."
    echo "Usage: bash run.sh <model_path> [dataset_path]"
    exit 1
fi

OUTPUT_PATH=${OUTPUT_PATH:-./results}

echo "=========================================="
echo "Inference + Evaluation"
echo "=========================================="
echo "Model Name:   $MODEL_NAME"
echo "Model Path:   $MODEL_PATH"
echo "Dataset: $DATASET_PATH"
echo "Output:  $OUTPUT_PATH"
echo "=========================================="

# Step 1: Run inference
echo ""
echo ">>> Step 1: Running inference..."
cd inference
PYTHONPATH="$(pwd)" bash run_inference.sh "$MODEL_NAME" "$MODEL_PATH" "$DATASET_PATH"
cd "$SCRIPT_DIR"

# Step 2: Run evaluation
model_name=$(basename "${MODEL_PATH%/}")
dataset_name=$(basename "$(dirname "$DATASET_PATH")")
eval_dir="${OUTPUT_PATH}/${model_name}/${dataset_name}"

if [ -f "${eval_dir}/iter1.jsonl" ]; then
    # Copy iter1 to iter2/iter3 for single-rollout evaluation
    ROLLOUT_COUNT=${ROLLOUT_COUNT:-1}
    if [ "$ROLLOUT_COUNT" -eq 1 ]; then
        cp "${eval_dir}/iter1.jsonl" "${eval_dir}/iter2.jsonl"
        cp "${eval_dir}/iter1.jsonl" "${eval_dir}/iter3.jsonl"
        echo "Single rollout: copied iter1.jsonl to iter2.jsonl and iter3.jsonl"
    fi

    echo ""
    echo ">>> Step 2: Running evaluation..."

    # Determine dataset type for evaluation
    EVAL_DATASET="gaia"
    case "$dataset_name" in
        *browsecomp_200*|*browsecomp_en*) EVAL_DATASET="browsecomp_en_full" ;;
        *browsecomp_zh*)                  EVAL_DATASET="browsecomp_zh" ;;
        *seal*)                           EVAL_DATASET="seal-0" ;;
        *xbench*)                         EVAL_DATASET="xbench-deepsearch" ;;
        *hle*)                            EVAL_DATASET="hle" ;;
        *gaia*)                           EVAL_DATASET="gaia" ;;
    esac

    cd evaluation
    python evaluate.py \
        --input-folder "${eval_dir}/" \
        --dataset "$EVAL_DATASET" \
        2>&1 | tee "${eval_dir}/evaluation_results.txt"
    cd "$SCRIPT_DIR"

    echo ""
    echo "Evaluation results saved to: ${eval_dir}/evaluation_results.txt"
else
    echo "Warning: iter1.jsonl not found in ${eval_dir}. Skipping evaluation."
fi

echo ""
echo "Done!"
