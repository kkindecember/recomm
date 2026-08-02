#!/usr/bin/env bash
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

ACTION=${1:-help}
GPU=${2:-}
SESSION=${SESSION:-codellama}
MODEL=${MODEL:-codellama/CodeLlama-7b-Instruct-hf}
RESERVE_GPU_MEMORY_MIB=${RESERVE_GPU_MEMORY_MIB:-30720}
MIN_FREE_MIB=${MIN_FREE_MIB:-32000}
POLL_SECONDS=${POLL_SECONDS:-30}
STARTUP_TIMEOUT_SECONDS=${STARTUP_TIMEOUT_SECONDS:-1800}
STATE_ROOT=${CODELLAMA_STATE_ROOT:-$ROOT/.runtime/codellama}
STATUS="$STATE_ROOT/status.txt"
LOG="$STATE_ROOT/run.log"
RUNNER_PID_FILE="$STATE_ROOT/runner.pid"
GPU_FILE="$STATE_ROOT/gpu.txt"
HOLDER_STATE="$STATE_ROOT/holder_state.json"
PYTHON=${PYTHON_BIN:-/home/jiangtangyunzhi/miniconda3/envs/unittest-transformers/bin/python}
HF_HOME=${HF_HOME:-/home/jiangtangyunzhi/hf_cache}
LEGACY_STATE_ROOT=${LEGACY_CODELLAMA_STATE_ROOT:-/home/jiangtangyunzhi/projects/UnitTest/experiments/codellama}

usage() {
    cat <<EOF
Usage:
  $0 start GPU       Load CodeLlama and retain its CUDA cache in a tmux session
  $0 status          Show current local state, GPU memory, and recent log output
  $0 stop            Stop the holder and release its GPU memory
  $0 legacy-status   Show the pre-migration state from the old UnitTest disk

Optional environment variables:
  MIN_FREE_MIB=32000             Free-memory threshold before acquisition
  RESERVE_GPU_MEMORY_MIB=30720   Persistent PyTorch CUDA cache target (30 GiB)
  POLL_SECONDS=30                Readiness polling and heartbeat interval
  STARTUP_TIMEOUT_SECONDS=1800   Maximum wait after model loading begins
  CODELLAMA_STATE_ROOT=PATH      Local runtime-state directory
EOF
}

write_status() {
    mkdir -p "$STATE_ROOT"
    printf '%s state=%s session=%s gpu=%s reserve_mib=%s required_free_mib=%s log=%s\n' \
        "$(date -Is)" "$1" "$SESSION" "${GPU:-unknown}" \
        "$RESERVE_GPU_MEMORY_MIB" "$MIN_FREE_MIB" "$LOG" > "$STATUS"
}

session_exists() {
    tmux has-session -t "$SESSION" 2>/dev/null
}

local_runner_alive() {
    [[ -f "$RUNNER_PID_FILE" ]] || return 1
    local runner_pid
    runner_pid=$(tr -d '[:space:]' < "$RUNNER_PID_FILE")
    [[ "$runner_pid" =~ ^[0-9]+$ ]] && kill -0 "$runner_pid" 2>/dev/null
}

run_task() {
    if [[ ! "$GPU" =~ ^[0-9]+$ ]]; then
        echo "GPU must be a non-negative integer" >&2
        exit 2
    fi
    if [[ ! "$RESERVE_GPU_MEMORY_MIB" =~ ^[0-9]+$ ]] \
        || [[ ! "$MIN_FREE_MIB" =~ ^[0-9]+$ ]] \
        || (( MIN_FREE_MIB < RESERVE_GPU_MEMORY_MIB )); then
        echo "MIN_FREE_MIB must be an integer >= RESERVE_GPU_MEMORY_MIB" >&2
        exit 2
    fi

    mkdir -p "$STATE_ROOT"
    printf '%s\n' "$GPU" > "$GPU_FILE"
    exec >>"$LOG" 2>&1
    runner_pid=""
    cleanup() {
        local rc=$?
        trap - EXIT INT TERM HUP
        if [[ -n "$runner_pid" ]] && kill -0 "$runner_pid" 2>/dev/null; then
            kill -TERM -- "-$runner_pid" 2>/dev/null || true
            wait "$runner_pid" 2>/dev/null || true
        fi
        rm -f "$RUNNER_PID_FILE"
        write_status stopped
        echo "[$(date -Is)] CodeLlama holder stopped; CUDA context released"
        exit "$rc"
    }
    trap cleanup EXIT INT TERM HUP

    echo "[$(date -Is)] controller started: gpu=$GPU reserve_mib=$RESERVE_GPU_MEMORY_MIB"
    while ! HF_HOME="$HF_HOME" HF_HUB_CACHE="$HF_HOME/hub" \
        "$PYTHON" "$ROOT/tools/check_hf_model_ready.py" "$MODEL" >/dev/null 2>&1; do
        write_status waiting_for_model
        sleep "$POLL_SECONDS"
    done

    while true; do
        free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
            --id="$GPU" 2>/dev/null | tr -d ' ')
        if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )); then
            break
        fi
        write_status waiting_for_gpu
        sleep "$POLL_SECONDS"
    done

    write_status starting
    echo "[$(date -Is)] GPU ready with ${free_mib} MiB free; loading holder"
    rm -f "$HOLDER_STATE"
    setsid env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY \
        -u https_proxy -u http_proxy -u all_proxy \
        CUDA_VISIBLE_DEVICES="$GPU" \
        HF_HOME="$HF_HOME" HF_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/hub" \
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42 PYTHONUTF8=1 \
        LANG=C.UTF-8 LC_ALL=C.UTF-8 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        "$PYTHON" "$ROOT/tools/codellama_gpu_holder.py" \
            --model "$MODEL" --reserve-gpu-memory-mib "$RESERVE_GPU_MEMORY_MIB" \
            --heartbeat-seconds "$POLL_SECONDS" --state "$HOLDER_STATE" &
    runner_pid=$!
    printf '%s\n' "$runner_pid" > "$RUNNER_PID_FILE"

    deadline=$(( $(date +%s) + STARTUP_TIMEOUT_SECONDS ))
    while kill -0 "$runner_pid" 2>/dev/null; do
        if [[ -f "$HOLDER_STATE" ]] \
            && grep -q '"state": "holding"' "$HOLDER_STATE"; then
            write_status running
            echo "[$(date -Is)] CodeLlama holder ready"
            wait "$runner_pid"
            exit $?
        fi
        if (( $(date +%s) >= deadline )); then
            echo "CodeLlama holder startup timed out" >&2
            exit 124
        fi
        write_status starting
        sleep 5
    done
    wait "$runner_pid"
    exit $?
}

show_state() {
    local state_root=$1
    [[ -f "$state_root/status.txt" ]] && cat "$state_root/status.txt"
    [[ -f "$state_root/holder_state.json" ]] && cat "$state_root/holder_state.json"
    [[ -f "$state_root/latest_cycle.json" ]] && cat "$state_root/latest_cycle.json"
    if [[ -f "$state_root/gpu.txt" ]]; then
        saved_gpu=$(tr -d '[:space:]' < "$state_root/gpu.txt")
        if [[ "$saved_gpu" =~ ^[0-9]+$ ]]; then
            nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
                --format=csv,noheader,nounits --id="$saved_gpu" 2>/dev/null || true
        fi
    fi
    [[ -f "$state_root/run.log" ]] && tail -n 12 "$state_root/run.log"
}

case "$ACTION" in
    start)
        if [[ ! "$GPU" =~ ^[0-9]+$ ]]; then
            echo "usage: $0 start GPU" >&2
            exit 2
        fi
        if session_exists; then
            echo "already running in tmux session: $SESSION"
            exit 1
        fi
        mkdir -p "$STATE_ROOT"
        printf '%s\n' "$GPU" > "$GPU_FILE"
        printf -v launch_cmd \
            'env SESSION=%q MODEL=%q RESERVE_GPU_MEMORY_MIB=%q MIN_FREE_MIB=%q POLL_SECONDS=%q STARTUP_TIMEOUT_SECONDS=%q PYTHON_BIN=%q HF_HOME=%q CODELLAMA_STATE_ROOT=%q %q run %q' \
            "$SESSION" "$MODEL" "$RESERVE_GPU_MEMORY_MIB" "$MIN_FREE_MIB" \
            "$POLL_SECONDS" "$STARTUP_TIMEOUT_SECONDS" "$PYTHON" "$HF_HOME" "$STATE_ROOT" \
            "$ROOT/tools/run_codellama.sh" "$GPU"
        tmux new-session -d -s "$SESSION" "$launch_cmd"
        echo "started tmux session $SESSION on physical GPU $GPU"
        echo "status: $0 status"
        echo "stop:   $0 stop"
        ;;
    run)
        run_task
        ;;
    status)
        if session_exists; then
            echo "tmux session: running ($SESSION)"
        else
            echo "tmux session: not running ($SESSION)"
        fi
        if [[ -f "$STATUS" ]]; then
            show_state "$STATE_ROOT"
        elif [[ -d "$LEGACY_STATE_ROOT" ]]; then
            echo "local state: not created; showing legacy read-only state from $LEGACY_STATE_ROOT"
            show_state "$LEGACY_STATE_ROOT"
        else
            echo "state: not started"
        fi
        ;;
    legacy-status)
        show_state "$LEGACY_STATE_ROOT"
        ;;
    stop)
        stopped=0
        if local_runner_alive; then
            runner_pid=$(tr -d '[:space:]' < "$RUNNER_PID_FILE")
            cmdline=$(tr '\0' ' ' < "/proc/$runner_pid/cmdline" 2>/dev/null || true)
            if [[ "$cmdline" == *"codellama_gpu_holder.py"* ]] \
                && [[ "$cmdline" == *"$HOLDER_STATE"* ]]; then
                kill -TERM -- "-$runner_pid" 2>/dev/null || true
                stopped=1
            fi
        fi
        if session_exists; then
            tmux kill-session -t "$SESSION"
            stopped=1
        fi
        rm -f "$RUNNER_PID_FILE"
        if [[ -f "$GPU_FILE" ]]; then GPU=$(tr -d '[:space:]' < "$GPU_FILE"); fi
        write_status stopped
        if (( stopped )); then
            echo "stopped $SESSION; its CUDA context and reserved memory were released"
        else
            echo "$SESSION was not running"
        fi
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
