#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ACTION=${1:-status}
SESSION=gram_phase4_ccrr_r0
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python3.9
STATUS="$ROOT/experiment/phase4/phase4_ccrr_r0_status.json"
PID_FILE="$ROOT/experiment/phase4/phase4_ccrr_r0.pid"
LOG="$ROOT/artifacts/phase4/logs/ccrr_r0.log"
OUTPUT="$ROOT/artifacts/phase4/ccrr_r0"
started_at=$(date -Is)
workload_pid=""

write_status() {
    local state=$1 reason=$2
    local workload_json=${workload_pid:-null}
    local tmp="${STATUS}.tmp.$$"
    printf '{\n  "experiment": "GRAM phase4 CCRR R0",\n  "updated_at": "%s",\n  "started_at": "%s",\n  "status": "%s",\n  "reason": "%s",\n  "runner_pid": %s,\n  "workload_pid": %s,\n  "log": "artifacts/phase4/logs/ccrr_r0.log",\n  "output": "artifacts/phase4/ccrr_r0/summary.json",\n  "resource": "CPU-only; GPU reservation unchanged"\n}\n' \
        "$(date -Is)" "$started_at" "$state" "$reason" "$$" "$workload_json" > "$tmp"
    mv "$tmp" "$STATUS"
}

finish() {
    local experiment_rc=$?
    trap - EXIT INT TERM HUP
    if (( experiment_rc == 0 )); then
        write_status succeeded "CCRR R0 completed; inspect summary.json."
    else
        write_status failed "CCRR R0 exit=${experiment_rc}; no retry and no scientific decision inferred."
    fi
    rm -f "$PID_FILE"
    exit "$experiment_rc"
}

worker() {
    started_at=${1:?missing start time}
    trap finish EXIT INT TERM HUP
    mkdir -p "$(dirname "$LOG")" "$OUTPUT"
    exec >> "$LOG" 2>&1
    cd "$ROOT"
    printf '%s\n' "$$" > "$PID_FILE"
    write_status running "Calibration-only logistic fit is running; audit remains locked until calibration passes."
    "$PYTHON" -m experiment.phase4.ccrr_r0 \
        --config "$ROOT/artifacts/phase4/configs/ccrr_r0_preregistered.json" \
        --output-dir "$OUTPUT" &
    workload_pid=$!
    write_status running "Calibration-only logistic fit is running; audit remains locked until calibration passes."
    wait "$workload_pid"
    workload_pid=""
}

case "$ACTION" in
    start)
        mkdir -p "$(dirname "$LOG")"
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            echo "session already exists: $SESSION"
            exit 1
        fi
        tmux new-session -d -s "$SESSION" \
            "bash '$ROOT/experiment/phase4/run_phase4_ccrr_r0.sh' worker '$started_at'"
        echo "started $SESSION"
        ;;
    worker)
        worker "${2:?missing start time}"
        ;;
    status)
        [[ -f "$STATUS" ]] && cat "$STATUS" || echo "not started"
        ;;
    *)
        echo "usage: $0 {start|status}"
        exit 2
        ;;
esac
