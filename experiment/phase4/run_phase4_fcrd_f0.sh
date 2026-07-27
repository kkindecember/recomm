#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ACTION=${1:-status}
GPU=3
MIN_FREE_MIB=30720
SESSION=gram_phase4_fcrd_f0
RESERVER=/home/jiangtangyunzhi/projects/UnitTest/tools/run_codellama.sh
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python3.9
STATUS="$ROOT/experiment/phase4/phase4_fcrd_f0_status.json"
PID_FILE="$ROOT/experiment/phase4/phase4_fcrd_f0.pid"
LOG="$ROOT/artifacts/phase4/logs/fcrd_f0.log"
OUTPUT="$ROOT/artifacts/phase4/fcrd_f0"
started_at=$(date -Is)
workload_pid=""
reservation_state=unchanged

write_status() {
    local state=$1 reason=$2
    local workload_json=${workload_pid:-null}
    local tmp="${STATUS}.tmp.$$"
    printf '{\n  "experiment": "GRAM phase4 FCRD F0",\n  "updated_at": "%s",\n  "started_at": "%s",\n  "status": "%s",\n  "reason": "%s",\n  "gpu_selected": 3,\n  "runner_pid": %s,\n  "workload_pid": %s,\n  "log": "artifacts/phase4/logs/fcrd_f0.log",\n  "output": "artifacts/phase4/fcrd_f0/summary.json",\n  "resource_reservation": "%s"\n}\n' \
        "$(date -Is)" "$started_at" "$state" "$reason" "$$" \
        "$workload_json" "$reservation_state" > "$tmp"
    mv "$tmp" "$STATUS"
}

restore_resource() {
    reservation_state=restoring
    write_status restoring_resource "FCRD F0 ended; restoring CodeLlama on physical GPU3."
    for attempt in 1 2 3; do
        if "$RESERVER" start "$GPU"; then
            reservation_state=restored
            return 0
        fi
        sleep 2
    done
    reservation_state=restore_failed
    return 1
}

finish() {
    local experiment_rc=$? restore_rc=0
    trap - EXIT INT TERM HUP
    restore_resource || restore_rc=$?
    if (( experiment_rc == 0 && restore_rc == 0 )); then
        write_status succeeded "FCRD F0 completed; inspect summary.json."
    elif (( restore_rc != 0 )); then
        write_status failed_to_restore_resource "F0 exit=${experiment_rc}; resource restore failed."
    else
        write_status failed "F0 exit=${experiment_rc}; no scientific decision inferred."
    fi
    rm -f "$PID_FILE"
    (( experiment_rc == 0 )) || exit "$experiment_rc"
    exit "$restore_rc"
}

worker() {
    started_at=${1:?missing start time}
    trap finish EXIT INT TERM HUP
    mkdir -p "$(dirname "$LOG")" "$OUTPUT"
    exec >> "$LOG" 2>&1
    cd "$ROOT"
    printf '%s\n' "$$" > "$PID_FILE"
    write_status releasing_resource "Acquiring physical GPU3 under the locked resource protocol."
    "$RESERVER" stop
    reservation_state=released_for_experiment
    local free_mib=0
    for attempt in $(seq 1 24); do
        free_mib=$(nvidia-smi --query-gpu=memory.free \
            --format=csv,noheader,nounits --id="$GPU" | tr -d ' ')
        if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )); then
            break
        fi
        sleep 5
    done
    if [[ ! "$free_mib" =~ ^[0-9]+$ ]] || (( free_mib < MIN_FREE_MIB )); then
        write_status blocked "GPU3 has ${free_mib:-unknown} MiB free; 30720 MiB required."
        exit 4
    fi
    write_status running "Frozen SASRec full-catalog residual inference and locked audit are running."
    CUDA_VISIBLE_DEVICES="$GPU" \
        HF_HOME="$ROOT/.cache/huggingface" \
        TRANSFORMERS_CACHE="$ROOT/.cache/huggingface" \
        "$PYTHON" -m experiment.phase4.fcrd_f0 \
        --config "$ROOT/artifacts/phase4/configs/fcrd_f0_preregistered.json" \
        --output-dir "$OUTPUT" \
        --device cuda &
    workload_pid=$!
    write_status running "Frozen SASRec full-catalog residual inference and locked audit are running."
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
            "bash '$ROOT/experiment/phase4/run_phase4_fcrd_f0.sh' worker '$started_at'"
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
