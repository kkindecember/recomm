#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ACTION=${1:-status}
GPU=3
MIN_FREE_MIB=30720
MIN_DISK_KIB=52428800
SESSION=gram_phase4_ialc_n1
RESERVER=/home/jiangtangyunzhi/projects/UnitTest/tools/run_codellama.sh
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python3.9
CONFIG="$ROOT/artifacts/phase4/configs/ialc_n1_preregistered.json"
OUTPUT="$ROOT/artifacts/phase4/ialc_n1"
STATUS="$ROOT/experiment/phase4/phase4_ialc_n1_status.json"
PID_FILE="$ROOT/experiment/phase4/phase4_ialc_n1.pid"
LOG="$ROOT/artifacts/phase4/logs/ialc_n1.log"
started_at=$(date -Is)
reservation_state=unchanged

write_status() {
    local state=$1 reason=$2
    local tmp="${STATUS}.tmp.$$"
    printf '{"experiment":"GRAM phase4 IALC N1","updated_at":"%s","started_at":"%s","status":"%s","reason":"%s","runner_pid":%s,"gpu_selected":3,"resource_reservation":"%s","validation_test_allowed":false,"sports_allowed":false,"output":"artifacts/phase4/ialc_n1/summary.json"}\n' \
        "$(date -Is)" "$started_at" "$state" "$reason" "$$" "$reservation_state" > "$tmp"
    mv "$tmp" "$STATUS"
}

restore_resource() {
    reservation_state=restoring
    write_status restoring_resource "N1 ended; restoring CodeLlama on physical GPU3."
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
        write_status succeeded "IALC N1 completed; inspect summary.json."
    elif (( restore_rc != 0 )); then
        write_status failed_to_restore_resource "N1 exit=${experiment_rc}; resource restoration failed."
    else
        write_status failed "N1 exit=${experiment_rc}; no automatic retry; resource restored."
    fi
    rm -f "$PID_FILE"
    (( experiment_rc == 0 )) || exit "$experiment_rc"
    exit "$restore_rc"
}

worker() {
    started_at=${1:?missing timestamp}
    trap finish EXIT INT TERM HUP
    mkdir -p "$OUTPUT" "$(dirname "$LOG")"
    exec >> "$LOG" 2>&1
    printf '%s\n' "$$" > "$PID_FILE"
    local available
    available=$(df --output=avail /home | tail -n 1 | tr -d ' ')
    if [[ ! "$available" =~ ^[0-9]+$ ]] || (( available < MIN_DISK_KIB )); then
        write_status blocked "Disk guard failed: ${available:-unknown} KiB available."
        exit 2
    fi
    write_status releasing_resource "Stopping CodeLlama before IALC N1."
    "$RESERVER" stop
    reservation_state=released_for_experiment
    local free_mib=""
    for attempt in $(seq 1 24); do
        free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GPU" | tr -d ' ')
        [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )) && break
        sleep 5
    done
    if [[ ! "$free_mib" =~ ^[0-9]+$ ]] || (( free_mib < MIN_FREE_MIB )); then
        write_status blocked "GPU3 free memory ${free_mib:-unknown} MiB below ${MIN_FREE_MIB} MiB."
        exit 3
    fi
    write_status running "Running locked frozen IALC N1 training-prefix audit."
    CUDA_VISIBLE_DEVICES="$GPU" HF_HOME="$ROOT/.cache/huggingface" \
        TRANSFORMERS_CACHE="$ROOT/.cache/huggingface" \
        timeout --signal=TERM 14400 \
        "$PYTHON" "$ROOT/experiment/phase4/ialc_n1.py" \
        --config "$CONFIG" --output-root "$OUTPUT"
}

case "$ACTION" in
    start)
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            echo "IALC N1 already running in $SESSION" >&2
            exit 1
        fi
        started_at=$(date -Is)
        tmux new-session -d -s "$SESSION" bash "$0" _worker "$started_at"
        reservation_state=scheduled_for_release
        write_status starting "Persistent IALC N1 session started."
        echo "IALC N1 started in tmux session $SESSION"
        ;;
    status)
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            echo "tmux session: running ($SESSION)"
        else
            echo "tmux session: not running ($SESSION)"
        fi
        [[ -s "$STATUS" ]] && sed -n '1p' "$STATUS" || echo '{"status":"not_started"}'
        [[ -s "$LOG" ]] && tail -n 20 "$LOG"
        true
        ;;
    _worker)
        worker "${2:?missing timestamp}"
        ;;
    *)
        echo "usage: bash experiment/phase4/run_phase4_ialc_n1.sh {start|status}" >&2
        exit 2
        ;;
esac
