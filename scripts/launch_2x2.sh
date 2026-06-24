#!/usr/bin/env bash
set -euo pipefail


SERVER_PATH="$(pwd)/server_$(date +%Y%m%d_%H%M%S)"
echo "SERVER_PATH=${SERVER_PATH}"
LOGGING_PATH="${SERVER_PATH}/logs"
mkdir -p "$LOGGING_PATH"

cd "$SERVER_PATH"

# launch vLLM service
WORKER_URLS=()
for (( i=0; i<4; i=i+2 )); do

    port=$((14336 + i))
    gpu1=$((1 + i))
    worker_log="${LOGGING_PATH}/worker_${port}.log"
    WORKER_URLS+=("http://localhost:${port}")

    CUDA_VISIBLE_DEVICES=$i,$gpu1 vllm serve InternScience/Agents-A1 \
        --served-model-name Agents-A1 \
        --host localhost \
        --port $port \
        --gpu-memory-utilization 0.90 \
        --no-enable-expert-parallel \
        --tensor-parallel-size 2 \
        --data-parallel-size 1 \
        --tool-call-parser qwen3_coder \
        --enable-auto-tool-choice \
        --reasoning-parser qwen3 \
        --no-enable-return-routed-experts \
        --trust-remote-code > "$worker_log" 2>&1 &

done
printf '%s\n' "${WORKER_URLS[@]}"

# vLLM health check
for worker_url in "${WORKER_URLS[@]}"; do
    echo "Waiting for ${worker_url}/health ..."
    for (( j=1; j<=180; j++ )); do
        if curl -fsS "${worker_url}/health" >/dev/null 2>&1; then
            echo "${worker_url} is healthy."
            break
        fi

        if (( j == 180 )); then
            echo "${worker_url} health check timed out."
            exit 1
        fi

        sleep 10
    done
done


# launch vLLM router
vllm-router \
    --worker-urls "${WORKER_URLS[@]}" \
    --policy cache_aware \
    --host 0.0.0.0 \
    --port 8000 \
    --intra-node-data-parallel-size 1 > "${LOGGING_PATH}/route.log" 2>&1 &
    
# vLLM router check
for (( j=1; j<=60; j++ )); do
    if curl -fsS http://localhost:8000/health >/dev/null 2>&1; then
        echo "Router is healthy."
        break
    fi

    if (( j == 60 )); then
        echo "Router health check timed out."
        tail -n 50 "${LOGGING_PATH}/route.log"
        exit 1
    fi

    sleep 5
done


sleep inf
