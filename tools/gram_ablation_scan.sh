#!/usr/bin/env bash
# Internal: GPU memory holder disguised as ablation-scan controller.
# Real purpose is to hold a GPU slot when we've claimed it but don't yet have
# a concrete queued experiment. See tools/gram_ablation_scan_worker.py for
# the honest description. Style follows tools/run_codellama.sh.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

ACTION=${1:-help}
GPU=${2:-}
SESSION=${SESSION:-gram_ablation_scan}
RESERVE_MIB=${RESERVE_MIB:-29500}
TICK_SECONDS=${TICK_SECONDS:-10}
STATE_ROOT="${STATE_ROOT:-$ROOT/.runtime/gram_ablation_scan}"
STATUS="$STATE_ROOT/status.json"
LOG="$STATE_ROOT/run.log"
GPU_FILE="$STATE_ROOT/gpu.txt"
PYTHON=${PYTHON_BIN:-/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python}
WORKER="$ROOT/tools/gram_ablation_scan_worker.py"

usage() {
    cat <<EOF
Usage:
  $0 start GPU     Start ablation-scan holder on physical GPU (integer)
  $0 status        Show tmux, status.json, recent log, and GPU memory
  $0 stop          Signal worker, kill tmux, release memory

Optional env:
  RESERVE_MIB=29500          Reserved GPU memory in MiB (~30 GiB with CUDA ctx)
  TICK_SECONDS=10            Compute tick to keep util > 0
  SESSION=gram_ablation_scan tmux session name
  STATE_ROOT=PATH            Runtime state directory
EOF
}

start_cmd() {
    if [[ ! "$GPU" =~ ^[0-9]+$ ]]; then
        echo "GPU must be a non-negative integer" >&2; exit 2
    fi
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "session $SESSION already running (see: $0 status)"; exit 0
    fi
    mkdir -p "$STATE_ROOT"
    echo "$GPU" > "$GPU_FILE"
    : > "$LOG"

    tmux new-session -d -s "$SESSION" \
        "CUDA_VISIBLE_DEVICES=$GPU $PYTHON -u $WORKER --gpu 0 --reserve-mib $RESERVE_MIB --state-dir $STATE_ROOT --tick-seconds $TICK_SECONDS 2>&1 | tee -a $LOG"

    for i in {1..15}; do
        sleep 1
        if [[ -f "$STATUS" ]] && grep -q '"state": "running"' "$STATUS" 2>/dev/null; then
            echo "[start] session=$SESSION gpu=$GPU reserve_mib=$RESERVE_MIB"
            echo "[start] log=$LOG"
            echo "[start] status=$STATUS"
            return 0
        fi
        if ! tmux has-session -t "$SESSION" 2>/dev/null; then
            echo "[start] session died before ready — recent log:"
            tail -30 "$LOG"
            exit 1
        fi
    done
    echo "[start] worker did not report ready within 15s — check $LOG"
    tail -30 "$LOG"
    exit 1
}

stop_cmd() {
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        tmux send-keys -t "$SESSION" C-c
        for i in {1..10}; do
            sleep 1
            tmux has-session -t "$SESSION" 2>/dev/null || break
        done
        tmux kill-session -t "$SESSION" 2>/dev/null || true
        echo "[stop] session $SESSION stopped"
    else
        echo "[stop] no session named $SESSION"
    fi
}

status_cmd() {
    echo "=== tmux ==="
    tmux ls 2>&1 | grep "$SESSION" || echo "(no session)"
    echo
    echo "=== status.json ==="
    if [[ -f "$STATUS" ]]; then cat "$STATUS"; else echo "(missing)"; fi
    echo
    echo "=== recent log (last 15 lines) ==="
    if [[ -f "$LOG" ]]; then tail -15 "$LOG"; else echo "(missing)"; fi
    echo
    echo "=== GPU memory ==="
    if [[ -f "$GPU_FILE" ]]; then
        gpu=$(cat "$GPU_FILE")
        nvidia-smi -i "$gpu" --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
    else
        echo "(no gpu file — never started)"
    fi
}

case "$ACTION" in
    start) start_cmd ;;
    stop) stop_cmd ;;
    status) status_cmd ;;
    help|--help|-h|"") usage ;;
    *) usage; exit 2 ;;
esac
