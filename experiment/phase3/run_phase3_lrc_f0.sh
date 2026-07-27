#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ACTION=${1:-status}
SESSION=gram_phase3_lrc_f0
STATUS="$ROOT/experiment/phase3/phase3_lrc_f0_status.json"
PID_FILE="$ROOT/experiment/phase3/phase3_lrc_f0.pid"
LOG="$ROOT/artifacts/phase3/logs/lrc_ucrf_f0.log"
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python3.9
started_at=$(date -Is)

write_status() {
    local stage=$1 state=$2 reason=$3
    local tmp="${STATUS}.tmp.$$"
    mkdir -p "$(dirname "$STATUS")"
    printf '{\n  "experiment": "GRAM phase3 LRC-UCRF F0 learnability probe",\n  "updated_at": "%s",\n  "started_at": "%s",\n  "stage": "%s",\n  "status": "%s",\n  "reason": "%s",\n  "runner_pid": %s,\n  "gpu": null,\n  "log": "artifacts/phase3/logs/lrc_ucrf_f0.log",\n  "output_dir": "artifacts/phase3/lrc_ucrf_f0",\n  "report": "report/第三阶段/GRAM_第三阶段_LRC-UCRF_F0可学习性报告.md",\n  "status_command": "bash experiment/phase3/run_phase3_lrc_f0.sh status"\n}\n' \
        "$(date -Is)" "$started_at" "$stage" "$state" "$reason" "$$" > "$tmp"
    mv "$tmp" "$STATUS"
}

finish() {
    local rc=$?
    local decision=unknown
    trap - EXIT INT TERM HUP
    if [[ -s "$ROOT/artifacts/phase3/lrc_ucrf_f0/summary.json" ]]; then
        decision=$(awk -F\" '/"decision":/{print $4; exit}' \
            "$ROOT/artifacts/phase3/lrc_ucrf_f0/summary.json")
    fi
    rm -f "$PID_FILE"
    if (( rc == 0 )); then
        write_status lrc_f0_complete succeeded \
            "LRC-F0 completed with scientific decision ${decision}; no GPU was used."
    else
        write_status lrc_f0_complete failed \
            "LRC-F0 exited with code ${rc}; no automatic retry was attempted."
    fi
    exit "$rc"
}

worker() {
    started_at=${1:?missing start timestamp}
    mkdir -p "$(dirname "$LOG")"
    exec >> "$LOG" 2>&1
    printf '%s\n' "$$" > "$PID_FILE"
    trap finish EXIT INT TERM HUP
    echo "[$(date -Is)] LRC-UCRF F0 CPU probe started"
    write_status lrc_f0 running \
        "Locked Beauty/Toys learnability and calibration probe is running on CPU."
    cd "$ROOT"
    HF_HOME="$ROOT/.cache/huggingface" TRANSFORMERS_CACHE="$ROOT/.cache/huggingface" \
        timeout --signal=TERM 1800 "$PYTHON" experiment/phase3/lrc_ucrf_f0.py
    echo "[$(date -Is)] LRC-UCRF F0 CPU probe finished"
}

case "$ACTION" in
    start)
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            echo "LRC-F0 is already running in tmux session $SESSION" >&2
            exit 1
        fi
        if [[ -s "$PID_FILE" ]]; then
            old_pid=$(tr -d '[:space:]' < "$PID_FILE")
            if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
                echo "LRC-F0 is already running with PID $old_pid" >&2
                exit 1
            fi
        fi
        started_at=$(date -Is)
        mkdir -p "$(dirname "$STATUS")"
        tmux new-session -d -s "$SESSION" bash "$0" _worker "$started_at"
        echo "LRC-F0 started in tmux session $SESSION"
        echo "status: bash experiment/phase3/run_phase3_lrc_f0.sh status"
        ;;
    status)
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            echo "tmux session: running ($SESSION)"
        else
            echo "tmux session: not running ($SESSION)"
        fi
        if [[ -s "$STATUS" ]]; then
            cat "$STATUS"
        else
            echo '{"stage":"lrc_f0","status":"not_started"}'
        fi
        if [[ -s "$LOG" ]]; then
            echo
            echo "latest log lines:"
            tail -n 12 "$LOG"
        fi
        ;;
    _worker)
        worker "${2:?missing start timestamp}"
        ;;
    *)
        echo "usage: bash experiment/phase3/run_phase3_lrc_f0.sh {start|status}" >&2
        exit 2
        ;;
esac
