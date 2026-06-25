#!/bin/bash
# Shared helper: start serve_vllm.sh in background, wait until ready, and fail fast
# if the vllm process dies before becoming ready.
#
# Usage (source this file, then call the function):
#   source /path/to/repo/wait_vllm.sh
#   start_vllm_and_wait "serve_vllm_${MODEL_NAME}.log" 8000 "${VLLM_SLEEP:-1800}" scripts/serve_vllm.sh
#
# Semantics of third arg: MAX wait seconds for readiness (not a fixed sleep).
# When vllm is ready earlier, the function returns immediately.
# On EXIT of the calling shell, the vllm process group is terminated.

start_vllm_and_wait() {
    local log_file="${1:-serve_vllm.log}"
    local port="${2:-8000}"
    local timeout="${3:-1800}"
    local serve_script="${4:-serve_vllm.sh}"
    local poll_interval=5

    # setsid gives vllm its own process group so we can kill all children on exit
    setsid bash "$serve_script" > "$log_file" 2>&1 &
    VLLM_PID=$!
    echo "[wait_vllm] started vllm pid=$VLLM_PID using ${serve_script}; polling http://localhost:${port}/v1/models (timeout=${timeout}s, log=$log_file)"

    _wait_vllm_cleanup() {
        if [ -n "${VLLM_PID:-}" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "[wait_vllm] terminating vllm pgid=$VLLM_PID"
            kill -TERM "-$VLLM_PID" 2>/dev/null || kill -TERM "$VLLM_PID" 2>/dev/null || true
            sleep 5
            kill -KILL "-$VLLM_PID" 2>/dev/null || kill -KILL "$VLLM_PID" 2>/dev/null || true
        fi
    }
    trap _wait_vllm_cleanup EXIT

    local start=$SECONDS
    while :; do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "[wait_vllm] ERROR: vllm process exited before becoming ready. Last 120 lines of $log_file:" >&2
            tail -n 120 "$log_file" >&2 || true
            return 1
        fi
        if curl -fsS --max-time 5 "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
            echo "[wait_vllm] vllm ready after $((SECONDS - start))s"
            return 0
        fi
        if (( SECONDS - start >= timeout )); then
            echo "[wait_vllm] ERROR: vllm not ready after ${timeout}s. Last 120 lines of $log_file:" >&2
            tail -n 120 "$log_file" >&2 || true
            return 1
        fi
        sleep "$poll_interval"
    done
}
