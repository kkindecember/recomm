#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ACTION=${1:-status}
GPU=3
MIN_FREE_MIB=30720
SESSION=gram_phase3_lei_f0
RESERVER=/home/jiangtangyunzhi/projects/UnitTest/tools/run_codellama.sh
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python3.9
STATUS="$ROOT/experiment/phase3/phase3_lei_f0_status.json"
PID_FILE="$ROOT/experiment/phase3/phase3_lei_f0.pid"
LOG="$ROOT/artifacts/phase3/logs/lei_f0.log"
BOARD_LOG="$ROOT/experiment/phase3/lei_f0_gpu_board.csv"
PROCESS_LOG="$ROOT/experiment/phase3/lei_f0_gpu_process.csv"
DISK_LOG="$ROOT/experiment/phase3/lei_f0_disk.csv"
started_at=$(date -Is)
workload_pid=""
monitor_pid=""
reservation_state=unchanged

write_status() {
    local state=$1 reason=$2
    local workload_json=${workload_pid:-null}
    local tmp="${STATUS}.tmp.$$"
    printf '{\n  "experiment": "GRAM phase3 LEI F0-D",\n  "updated_at": "%s",\n  "started_at": "%s",\n  "status": "%s",\n  "reason": "%s",\n  "gpu_selected": 3,\n  "runner_pid": %s,\n  "workload_pid": %s,\n  "log": "artifacts/phase3/logs/lei_f0.log",\n  "output": "artifacts/phase3/lei_f0/summary.json",\n  "resource_reservation": "%s"\n}\n' \
        "$(date -Is)" "$started_at" "$state" "$reason" "$$" \
        "$workload_json" "$reservation_state" > "$tmp"
    mv "$tmp" "$STATUS"
}

restore_resource() {
    reservation_state=restoring
    write_status restoring_resource "LEI F0-D ended; restoring CodeLlama on physical GPU3."
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
    if [[ -n "$monitor_pid" ]] && kill -0 "$monitor_pid" 2>/dev/null; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
    restore_resource || restore_rc=$?
    if (( experiment_rc == 0 && restore_rc == 0 )); then
        write_status succeeded "LEI F0-D completed; inspect the locked decision in summary.json."
    elif (( restore_rc != 0 )); then
        write_status failed_to_restore_resource "F0-D exit=${experiment_rc}; resource restore failed."
    else
        write_status failed "F0-D exit=${experiment_rc}; no scientific decision was inferred."
    fi
    rm -f "$PID_FILE"
    (( experiment_rc == 0 )) || exit "$experiment_rc"
    exit "$restore_rc"
}

monitor_resources() {
    echo "timestamp,index,memory.used,memory.free,utilization.gpu" > "$BOARD_LOG"
    echo "timestamp,pid,used_gpu_memory,is_workload" > "$PROCESS_LOG"
    echo "timestamp,filesystem,size_kib,used_kib,available_kib,capacity,mount" > "$DISK_LOG"
    local tick=0
    while true; do
        local timestamp pid used is_workload
        timestamp=$(date -Is)
        nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
            --format=csv,noheader,nounits --id="$GPU" 2>/dev/null \
            | sed "s/^/${timestamp},/" >> "$BOARD_LOG" || true
        while IFS=',' read -r pid used; do
            pid=${pid//[[:space:]]/}
            used=${used//[[:space:]]/}
            [[ "$pid" =~ ^[0-9]+$ ]] || continue
            is_workload=0
            [[ "$pid" == "$workload_pid" ]] && is_workload=1
            printf '%s,%s,%s,%s\n' "$timestamp" "$pid" "$used" "$is_workload" >> "$PROCESS_LOG"
        done < <(nvidia-smi --query-compute-apps=pid,used_gpu_memory \
            --format=csv,noheader,nounits --id="$GPU" 2>/dev/null || true)
        if (( tick % 60 == 0 )); then
            df -Pk "$ROOT" | awk -v ts="$timestamp" 'NR==2 {print ts","$1","$2","$3","$4","$5","$6}' \
                >> "$DISK_LOG" || true
        fi
        tick=$((tick + 1))
        sleep 5
    done
}

worker() {
    started_at=${1:?missing start time}
    trap finish EXIT INT TERM HUP
    mkdir -p "$(dirname "$LOG")"
    exec >> "$LOG" 2>&1
    printf '%s\n' "$$" > "$PID_FILE"
    write_status releasing_resource "Acquiring physical GPU3 under the locked resource protocol."
    "$RESERVER" stop
    reservation_state=released_for_experiment
    local free_mib="" wait_attempt
    for wait_attempt in $(seq 1 24); do
        free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
            --id="$GPU" | tr -d ' ')
        if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )); then
            break
        fi
        write_status releasing_resource \
            "Waiting for GPU3 CUDA context release (${free_mib:-unknown} MiB free; attempt ${wait_attempt}/24)."
        sleep 5
    done
    if [[ ! "$free_mib" =~ ^[0-9]+$ ]] || (( free_mib < MIN_FREE_MIB )); then
        write_status blocked "GPU3 has ${free_mib:-unknown} MiB free; 30720 MiB required."
        exit 4
    fi
    monitor_resources &
    monitor_pid=$!
    write_status running "Frozen Toys/Beauty span scoring is running; no training, beam generation, or test access."
    CUDA_VISIBLE_DEVICES="$GPU" \
        HF_HOME="$ROOT/.cache/huggingface" \
        TRANSFORMERS_CACHE="$ROOT/.cache/huggingface" \
        "$PYTHON" "$ROOT/experiment/phase3/lei_f0.py" \
        --config "$ROOT/artifacts/phase3/configs/lei_f0_d_preregistered.json" &
    workload_pid=$!
    write_status running "Frozen Toys/Beauty span scoring is running; no training, beam generation, or test access."
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
            "bash '$ROOT/experiment/phase3/run_phase3_lei_f0.sh' worker '$started_at'"
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
