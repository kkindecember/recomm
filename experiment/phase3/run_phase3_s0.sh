#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ACTION=${1:-status}
STATUS="$ROOT/experiment/phase3/phase3_s0_status.json"
PID_FILE="$ROOT/experiment/phase3/phase3_s0.pid"
LOG="$ROOT/artifacts/phase3/logs/s0_toys_validation.log"
OUTPUT_DIR="$ROOT/artifacts/phase3/s0/Toys/validation"
PREDICTIONS="$ROOT/GRAM/preds/20260722_020042_Toys_sequential_pred_validation.tsv"
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python3.9
TMUX_SESSION=gram_phase3_s0

write_initial_status() {
    local runner_pid=$1
    local now
    now=$(date -Is)
    local tmp="${STATUS}.tmp.$$"
    printf '{\n  "experiment": "GRAM phase3 S0 offline diagnostics",\n  "updated_at": "%s",\n  "started_at": "%s",\n  "stage": "s0_toys_validation",\n  "status": "running",\n  "reason": "CPU-only Toys validation diagnostics are running in background.",\n  "runner_pid": %s,\n  "gpu": null,\n  "log": "artifacts/phase3/logs/s0_toys_validation.log",\n  "output_dir": "artifacts/phase3/s0/Toys/validation",\n  "report": "report/第三阶段/GRAM_第三阶段_S0离线诊断报告.md"\n}\n' \
        "$now" "$now" "$runner_pid" > "$tmp"
    mv "$tmp" "$STATUS"
}

worker() {
    local started_at=$1
    local runner_pid=$$
    local rc=0
    mkdir -p "$(dirname "$LOG")" "$OUTPUT_DIR" "$(dirname "$STATUS")"
    printf '%s\n' "$runner_pid" > "$PID_FILE"
    exec >> "$LOG" 2>&1
    echo "[$(date -Is)] S0 Toys validation CPU diagnostics started"
    if HF_HOME="$ROOT/.cache/huggingface" TRANSFORMERS_CACHE="$ROOT/.cache/huggingface" \
        "$PYTHON" "$ROOT/experiment/phase3/s0_offline_diagnostics.py" \
        --dataset Toys \
        --predictions "$PREDICTIONS" \
        --mode validation \
        --output-dir "$OUTPUT_DIR" \
        --local-files-only; then
        :
    else
        rc=$?
    fi
    if (( rc == 0 )); then
        if "$PYTHON" "$ROOT/experiment/phase3/s0_build_report.py"; then
            :
        else
            rc=$?
        fi
    fi
    local state reason
    if (( rc == 0 )); then
        state=partial
        reason="Toys validation diagnostics completed, but overall S0 is incomplete because Beauty validation remains pending."
    else
        state=failed
        reason="S0 Toys validation diagnostics exited with code ${rc}; no automatic retry was attempted."
    fi
    local tmp="${STATUS}.tmp.$$"
    printf '{\n  "experiment": "GRAM phase3 S0 offline diagnostics",\n  "updated_at": "%s",\n  "started_at": "%s",\n  "stage": "s0_toys_validation_complete",\n  "status": "%s",\n  "exit_code": %s,\n  "reason": "%s",\n  "runner_pid": %s,\n  "gpu": null,\n  "log": "artifacts/phase3/logs/s0_toys_validation.log",\n  "output_dir": "artifacts/phase3/s0/Toys/validation",\n  "report": "report/第三阶段/GRAM_第三阶段_S0离线诊断报告.md"\n}\n' \
        "$(date -Is)" "$started_at" "$state" "$rc" "$reason" "$runner_pid" > "$tmp"
    mv "$tmp" "$STATUS"
    rm -f "$PID_FILE"
    echo "[$(date -Is)] S0 Toys validation CPU diagnostics exit code: $rc"
    exit "$rc"
}

case "$ACTION" in
    start)
        if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
            echo "S0 is already running in tmux session $TMUX_SESSION"
            exit 1
        fi
        if [[ -s "$PID_FILE" ]]; then
            old_pid=$(cat "$PID_FILE")
            if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
                echo "S0 is already running with PID $old_pid"
                exit 1
            fi
        fi
        mkdir -p "$(dirname "$LOG")" "$(dirname "$STATUS")"
        started_at=$(date -Is)
        tmux new-session -d -s "$TMUX_SESSION" bash "$0" _worker "$started_at"
        runner_pid=$(tmux display-message -p -t "$TMUX_SESSION" '#{pane_pid}')
        write_initial_status "$runner_pid"
        echo "S0 started in tmux session $TMUX_SESSION with PID $runner_pid"
        echo "status: bash experiment/phase3/run_phase3_s0.sh status"
        ;;
    status)
        if [[ -s "$STATUS" ]]; then
            cat "$STATUS"
        else
            echo '{"stage":"s0","status":"not_started"}'
        fi
        if [[ -s "$LOG" ]]; then
            echo
            echo "latest log lines:"
            tail -n 8 "$LOG"
        fi
        ;;
    _worker)
        worker "${2:?missing start timestamp}"
        ;;
    *)
        echo "usage: bash experiment/phase3/run_phase3_s0.sh {start|status}" >&2
        exit 2
        ;;
esac
