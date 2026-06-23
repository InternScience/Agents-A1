#!/bin/bash
# run_inference.sh - Universal entry point for inference
#
# Usage: bash run_inference.sh <model_path_or_name> <dataset_path>
#
# Modes:
#   Local vLLM (default): Starts a vLLM server, waits for readiness, runs inference.
#   Remote API:           Set AGENT_API_BASE_URL in .env to skip vLLM startup and
#                         send requests to the remote endpoint directly.
#
# Environment variables (from .env):
#   AGENT_API_KEY, AGENT_API_BASE_URL  — remote API credentials
#   MAX_WORKERS, TEMPERATURE, PRESENCE_PENALTY, OUTPUT_PATH, ROLLOUT_COUNT

# Load environment variables from .env file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"

if [ -f "$ENV_FILE" ]; then
    echo "Loading environment variables from .env file..."
    set -a
    source "$ENV_FILE"
    set +a
fi

MODEL_NAME=$1
MODEL_PATH=$2
DATASET_PATH=$3

if [ -z "$MODEL_NAME" ] || [ -z "$DATASET_PATH" ]; then
    echo "Usage: bash run_inference.sh <model_path_or_name> <dataset_path>"
    exit 1
fi

echo "Model Name: $MODEL_NAME"
echo "Model Path: $MODEL_PATH"
echo "Dataset: $DATASET_PATH"

cd "$SCRIPT_DIR"

if [ -n "$AGENT_API_BASE_URL" ]; then
    ######################################
    ### Remote API Mode                ###
    ######################################
    echo "Mode: Remote API (AGENT_API_BASE_URL=$AGENT_API_BASE_URL)"
    echo ""
    echo "Starting inference..."

    python -u run_inference.py \
        --dataset "$DATASET_PATH" \
        --output "${OUTPUT_PATH:-../results}" \
        --max-workers ${MAX_WORKERS:-16} \
        --model "$MODEL_NAME" \
        --model-path "$MODEL_PATH" \
        --temperature ${TEMPERATURE:-0.85} \
        --top-p ${TOP_P:-0.95} \
        --top-k ${TOP_K:--1} \
        --presence-penalty ${PRESENCE_PENALTY:-1.1} \
        --total-splits ${WORLD_SIZE:-1} \
        --worker-split $((${RANK:-0} + 1)) \
        --roll-out-count ${ROLLOUT_COUNT:-1}
else
    ######################################
    ### Local vLLM Mode                ###
    ######################################
    echo "Mode: Local vLLM"
    echo "Starting vLLM server..."

    # Adjust CUDA_VISIBLE_DEVICES and --tensor-parallel-size for your hardware:
    #   - For 30B-A3B models: 2 GPUs per instance (tp=2)
    #   - For larger models: increase tensor-parallel-size accordingly
    #   - For single GPU: set CUDA_VISIBLE_DEVICES=0 and tp=1
    #
    # To run multiple instances on different ports, duplicate this line with
    # different CUDA_VISIBLE_DEVICES and port numbers.
    VLLM_USE_AOT_COMPILE=0 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve $MODEL_NAME \
        --host 0.0.0.0 \
        --port 6001 \
        --disable-log-requests \
        --tensor-parallel-size 8 \
        --gpu_memory_utilization 0.95 \
        --reasoning-parser qwen3 \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        --trust-remote-code &

    ######################################
    ### Wait for server readiness      ###
    ######################################

    timeout=6000
    start_time=$(date +%s)

    # Add more ports here if running multiple vLLM instances
    main_ports=(6001)

    declare -A server_status
    for port in "${main_ports[@]}"; do
        server_status[$port]=false
    done

    echo "Waiting for server(s) to start..."

    while true; do
        all_ready=true

        for port in "${main_ports[@]}"; do
            if [ "${server_status[$port]}" = "false" ]; then
                if curl -s -f http://localhost:$port/v1/models > /dev/null 2>&1; then
                    echo "Server (port $port) is ready!"
                    server_status[$port]=true
                else
                    all_ready=false
                fi
            fi
        done

        if [ "$all_ready" = "true" ]; then
            echo "All servers are ready!"
            break
        fi

        current_time=$(date +%s)
        elapsed=$((current_time - start_time))
        if [ $elapsed -gt $timeout ]; then
            echo "Error: Server startup timeout after ${timeout} seconds"
            for port in "${main_ports[@]}"; do
                if [ "${server_status[$port]}" = "false" ]; then
                    echo "  - Server on port $port failed to start"
                fi
            done
            exit 1
        fi

        printf '.'
        sleep 10
    done

    ######################################
    ### Run inference                   ###
    ######################################

    echo ""
    echo "Starting inference..."

    # Match --ports to main_ports above
    PORTS=$(IFS=,; echo "${main_ports[*]}")

    python -u run_inference.py \
        --dataset "$DATASET_PATH" \
        --output "${OUTPUT_PATH:-../results}" \
        --max-workers ${MAX_WORKERS:-16} \
        --model $MODEL_NAME \
        --model-path "$MODEL_PATH" \
        --temperature ${TEMPERATURE:-0.85} \
        --top-p ${TOP_P:-0.95} \
        --top-k ${TOP_K:--1} \
        --presence-penalty ${PRESENCE_PENALTY:-1.1} \
        --total-splits ${WORLD_SIZE:-1} \
        --worker-split $((${RANK:-0} + 1)) \
        --roll-out-count ${ROLLOUT_COUNT:-1} \
        --ports "$PORTS"
fi
