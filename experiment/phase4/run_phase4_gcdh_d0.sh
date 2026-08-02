#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ACTION=${1:-status}
GPU=3
MIN_FREE_MIB=30720
MIN_DISK_KIB=52428800
SESSION=gram_phase4_gcdh_d0
RESERVER="$ROOT/tools/run_codellama.sh"
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python3.9
CONFIG="$ROOT/artifacts/phase4/configs/gcdh_d0_preregistered.json"
OUTPUT="$ROOT/artifacts/phase4/gcdh_d0/summary.json"
STATUS="$ROOT/experiment/phase4/phase4_gcdh_d0_status.json"
PID_FILE="$ROOT/experiment/phase4/phase4_gcdh_d0.pid"
LOG="$ROOT/artifacts/phase4/logs/gcdh_d0.log"
started_at=$(date -Is)
reservation_state=unchanged

write_status() {
    local state=$1 reason=$2
    local tmp="${STATUS}.tmp.$$"
    printf '{"experiment":"GRAM phase4 GCDH D0","updated_at":"%s","started_at":"%s","status":"%s","reason":"%s","runner_pid":%s,"gpu_selected":3,"resource_reservation":"%s","test_data_allowed":false,"output":"artifacts/phase4/gcdh_d0/summary.json"}\n' \
        "$(date -Is)" "$started_at" "$state" "$reason" "$$" "$reservation_state" > "$tmp"
    mv "$tmp" "$STATUS"
}

restore_resource() {
    reservation_state=restoring
    write_status restoring_resource "D0 ended; restoring CodeLlama on physical GPU3."
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
        write_status succeeded "GCDH D0 completed; inspect summary.json."
    elif (( restore_rc != 0 )); then
        write_status failed_to_restore_resource "D0 exit=${experiment_rc}; resource restoration failed."
    else
        write_status failed "D0 exit=${experiment_rc}; no automatic retry; resource restored."
    fi
    rm -f "$PID_FILE"
    (( experiment_rc == 0 )) || exit "$experiment_rc"
    exit "$restore_rc"
}

worker() {
    started_at=${1:?missing timestamp}
    trap finish EXIT INT TERM HUP
    mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$LOG")"
    exec >> "$LOG" 2>&1
    printf '%s\n' "$$" > "$PID_FILE"
    local available
    available=$(df --output=avail /home | tail -n 1 | tr -d ' ')
    if [[ ! "$available" =~ ^[0-9]+$ ]] || (( available < MIN_DISK_KIB )); then
        write_status blocked "Disk guard failed: ${available:-unknown} KiB available."
        exit 2
    fi
    write_status releasing_resource "Stopping CodeLlama before D0."
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
    write_status running "Running locked no-update GCDH D0."
    CUDA_VISIBLE_DEVICES="$GPU" HF_HOME="$ROOT/.cache/huggingface" \
        TRANSFORMERS_CACHE="$ROOT/.cache/huggingface" \
        timeout --signal=TERM 14400 \
        "$PYTHON" "$ROOT/experiment/phase4/gcdh_d0.py" \
        --config "$CONFIG" --output "$OUTPUT"
}

case "$ACTION" in
    start)
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            echo "GCDH D0 already running in $SESSION" >&2
            exit 1
        fi
        started_at=$(date -Is)
        tmux new-session -d -s "$SESSION" bash "$0" _worker "$started_at"
        reservation_state=scheduled_for_release
        write_status starting "Persistent GCDH D0 session started."
        echo "GCDH D0 started in tmux session $SESSION"
        ;;
    status)
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            echo "tmux session: running ($SESSION)"
        else
            echo "tmux session: not running ($SESSION)"
        fi
        [[ -s "$STATUS" ]] && cat "$STATUS" || echo '{"status":"not_started"}'
        [[ -s "$LOG" ]] && tail -n 20 "$LOG"
        true
        ;;
    _worker)
        worker "${2:?missing timestamp}"
        ;;
    *)
        echo "usage: bash experiment/phase4/run_phase4_gcdh_d0.sh {start|status}" >&2
        exit 2
        ;;
esac
