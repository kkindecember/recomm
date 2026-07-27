#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ACTION=${1:-status}
SESSION=gram_phase3_s0b
STATUS="$ROOT/experiment/phase3/phase3_s0b_status.json"
PID_FILE="$ROOT/experiment/phase3/phase3_s0b.pid"
LOG="$ROOT/artifacts/phase3/logs/s0b_reliability_probe.log"
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python3.9
started_at=$(date -Is)

write_status() {
    local stage=$1
    local state=$2
    local reason=$3
    local tmp="${STATUS}.tmp.$$"
    mkdir -p "$(dirname "$STATUS")"
    printf '{\n  "experiment": "GRAM phase3 S0b reliability-abstention probe",\n  "updated_at": "%s",\n  "started_at": "%s",\n  "stage": "%s",\n  "status": "%s",\n  "reason": "%s",\n  "runner_pid": %s,\n  "gpu": null,\n  "locked_config_count": 16,\n  "log": "artifacts/phase3/logs/s0b_reliability_probe.log",\n  "output_dir": "artifacts/phase3/s0b",\n  "report": "report/第三阶段/GRAM_第三阶段_S0b可靠性拒绝探针报告.md",\n  "status_command": "bash experiment/phase3/run_phase3_s0b.sh status"\n}\n' \
        "$(date -Is)" "$started_at" "$stage" "$state" "$reason" "$$" > "$tmp"
    mv "$tmp" "$STATUS"
}

worker() {
    started_at=${1:?missing start timestamp}
    local rc=0 decision=unknown
    mkdir -p "$(dirname "$LOG")"
    exec >> "$LOG" 2>&1
    printf '%s\n' "$$" > "$PID_FILE"
    trap 'rc=$?; rm -f "$PID_FILE"; if (( rc == 0 )); then decision=$(awk -F\" '\''/"decision":/{print $4; exit}'\'' "$ROOT/artifacts/phase3/s0b/summary.json"); write_status s0b_complete succeeded "S0b completed with scientific decision ${decision}; no GPU was used."; else write_status s0b_complete failed "S0b exited with code ${rc}; no automatic retry was attempted."; fi; exit "$rc"' EXIT INT TERM HUP
    echo "[$(date -Is)] S0b 16-config CPU probe started"
    write_status s0b_reliability_probe running \
        "Locked 16-config Beauty/Toys validation probe is running on CPU."
    cd "$ROOT"
    HF_HOME="$ROOT/.cache/huggingface" TRANSFORMERS_CACHE="$ROOT/.cache/huggingface" \
        timeout --signal=TERM 1800 "$PYTHON" experiment/phase3/s0b_reliability_probe.py
    echo "[$(date -Is)] S0b CPU probe finished"
}

case "$ACTION" in
    start)
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            echo "S0b is already running in tmux session $SESSION" >&2
            exit 1
        fi
        if [[ -s "$PID_FILE" ]]; then
            old_pid=$(tr -d '[:space:]' < "$PID_FILE")
            if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
                echo "S0b is already running with PID $old_pid" >&2
                exit 1
            fi
        fi
        started_at=$(date -Is)
        mkdir -p "$(dirname "$STATUS")"
        tmux new-session -d -s "$SESSION" bash "$0" _worker "$started_at"
        echo "S0b started in tmux session $SESSION"
        echo "status: bash experiment/phase3/run_phase3_s0b.sh status"
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
            echo '{"stage":"s0b","status":"not_started"}'
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
        echo "usage: bash experiment/phase3/run_phase3_s0b.sh {start|status}" >&2
        exit 2
        ;;
esac
