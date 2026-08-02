#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ACTION=${1:-status}
GPU=3
MIN_FREE_MIB=30720
MIN_DISK_KIB=52428800
SESSION=gram_phase4_gcdh_p0
RESERVER="$ROOT/tools/run_codellama.sh"
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python3.9
CONFIG="$ROOT/artifacts/phase4/configs/gcdh_p0_preregistered.json"
OUTPUT="$ROOT/artifacts/phase4/gcdh_p0"
SPLITS="$ROOT/artifacts/phase4/gcdh_p0_splits"
STATUS="$ROOT/experiment/phase4/phase4_gcdh_p0_status.json"
PID_FILE="$ROOT/experiment/phase4/phase4_gcdh_p0.pid"
LOG="$ROOT/artifacts/phase4/logs/gcdh_p0.log"
BOARD_LOG="$ROOT/experiment/phase4/gcdh_p0_gpu_board.csv"
PROCESS_LOG="$ROOT/experiment/phase4/gcdh_p0_gpu_process.csv"
DISK_LOG="$ROOT/experiment/phase4/gcdh_p0_disk.csv"
started_at=$(date -Is)
workload_pid=""
monitor_pid=""
reservation_state=unchanged
current_stage=none
current_dataset=none
current_control=none

write_status() {
    local state=$1 reason=$2
    local workload_json=${workload_pid:-null}
    local tmp="${STATUS}.tmp.$$"
    mkdir -p "$(dirname "$STATUS")"
    printf '{\n  "experiment": "GRAM phase4 GCDH P0",\n  "updated_at": "%s",\n  "started_at": "%s",\n  "stage": "%s",\n  "status": "%s",\n  "reason": "%s",\n  "current_dataset": "%s",\n  "current_control": "%s",\n  "gpu_selected": 3,\n  "minimum_free_mib": 30720,\n  "runner_pid": %s,\n  "workload_pid": %s,\n  "log": "artifacts/phase4/logs/gcdh_p0.log",\n  "output_dir": "artifacts/phase4/gcdh_p0",\n  "resource_reservation": "%s",\n  "test_data_allowed": false,\n  "status_command": "bash experiment/phase4/run_phase4_gcdh_p0.sh status"\n}\n' \
        "$(date -Is)" "$started_at" "$current_stage" "$state" "$reason" \
        "$current_dataset" "$current_control" "$$" "$workload_json" \
        "$reservation_state" > "$tmp"
    mv "$tmp" "$STATUS"
}

restore_resource() {
    reservation_state=restoring
    current_stage=restore_resource
    write_status restoring_resource "GCDH P0 ended; restoring CodeLlama on physical GPU3."
    for attempt in 1 2 3; do
        if "$RESERVER" start "$GPU"; then
            reservation_state=restored
            return 0
        fi
        echo "[$(date -Is)] restore attempt $attempt failed"
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
    if [[ -n "$workload_pid" ]] && kill -0 "$workload_pid" 2>/dev/null; then
        kill -TERM "$workload_pid" 2>/dev/null || true
        wait "$workload_pid" 2>/dev/null || true
    fi
    restore_resource || restore_rc=$?
    current_dataset=none
    current_control=none
    current_stage=complete
    if (( experiment_rc == 0 && restore_rc == 0 )); then
        write_status succeeded "All GCDH P0 stages completed; inspect summary.json."
    elif (( restore_rc != 0 )); then
        write_status failed_to_restore_resource \
            "Experiment exit=${experiment_rc}; CodeLlama restoration failed."
    else
        write_status failed \
            "Experiment exit=${experiment_rc}; no automatic retry; CodeLlama restored."
    fi
    rm -f "$PID_FILE"
    (( experiment_rc == 0 )) || exit "$experiment_rc"
    exit "$restore_rc"
}

monitor_resources() {
    local sample=0
    [[ -s "$BOARD_LOG" ]] || echo "timestamp,index,memory.used,memory.free,utilization.gpu" > "$BOARD_LOG"
    [[ -s "$PROCESS_LOG" ]] || echo "timestamp,pid,used_gpu_memory" > "$PROCESS_LOG"
    [[ -s "$DISK_LOG" ]] || echo "timestamp,filesystem,available_kib" > "$DISK_LOG"
    while true; do
        local timestamp available
        timestamp=$(date -Is)
        nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
            --format=csv,noheader,nounits --id="$GPU" 2>/dev/null \
            | while IFS= read -r row; do printf '%s,%s\n' "$timestamp" "$row"; done \
            >> "$BOARD_LOG" || true
        nvidia-smi --query-compute-apps=pid,used_gpu_memory \
            --format=csv,noheader,nounits --id="$GPU" 2>/dev/null \
            | while IFS= read -r row; do printf '%s,%s\n' "$timestamp" "$row"; done \
            >> "$PROCESS_LOG" || true
        if (( sample % 60 == 0 )); then
            available=$(df --output=avail /home | tail -n 1 | tr -d ' ')
            printf '%s,/home,%s\n' "$timestamp" "$available" >> "$DISK_LOG"
            if [[ ! "$available" =~ ^[0-9]+$ ]] || (( available < MIN_DISK_KIB )); then
                echo "[$timestamp] disk guard triggered available_kib=$available" >> "$LOG"
                kill -TERM "$$" 2>/dev/null || true
                return 1
            fi
        fi
        sample=$((sample + 1))
        sleep 5
    done
}

run_stage() {
    local stage=$1 dataset=$2 control=${3:-}
    current_stage=$stage
    current_dataset=$dataset
    current_control=${control:-none}
    write_status running "Running locked ${stage} for ${dataset}/${current_control}."
    local command=(
        "$PYTHON" "$ROOT/experiment/phase4/gcdh_p0.py"
        --config "$CONFIG" --stage "$stage" --dataset "$dataset"
        --output-root "$OUTPUT"
    )
    [[ -n "$control" ]] && command+=(--control "$control")
    CUDA_VISIBLE_DEVICES="$GPU" HF_HOME="$ROOT/.cache/huggingface" \
        TRANSFORMERS_CACHE="$ROOT/.cache/huggingface" \
        timeout --signal=TERM 43200 "${command[@]}" &
    workload_pid=$!
    write_status running "Running locked ${stage} for ${dataset}/${current_control}."
    wait "$workload_pid"
    workload_pid=""
}

run_control() {
    local dataset=$1 control=$2
    local dir="$OUTPUT/$dataset/$control"
    if [[ -s "$dir/training_summary.json" && -s "$dir/model.pt" \
        && -s "$dir/validation_summary.json" && -s "$dir/validation_per_user.csv" ]]; then
        echo "[$(date -Is)] RESUME_SKIP $dataset/$control complete"
        return 0
    fi
    if [[ -s "$dir/training_summary.json" && -s "$dir/model.pt" ]]; then
        run_stage validate "$dataset" "$control"
    else
        run_stage train "$dataset" "$control"
    fi
}

worker() {
    started_at=${1:?missing timestamp}
    trap finish EXIT INT TERM HUP
    mkdir -p "$OUTPUT" "$(dirname "$LOG")" "$ROOT/.cache/huggingface"
    exec >> "$LOG" 2>&1
    printf '%s\n' "$$" > "$PID_FILE"
    local required
    for required in "$CONFIG" \
        "$ROOT/GRAM/log/Toys/1_20260720_1830/id_0_rec_30/model_rec_phase_1_epoch_30.pt" \
        "$ROOT/GRAM/log/Beauty/4_20260718_2153/id_0_rec_30/model_rec_phase_1_epoch_25.pt" \
        "$SPLITS/Toys/manifest.json" "$SPLITS/Beauty/manifest.json"; do
        if [[ ! -s "$required" ]]; then
            current_stage=precondition
            write_status blocked "Required locked material missing: $required"
            exit 2
        fi
    done
    current_stage=release_resource
    write_status releasing_resource "Stopping CodeLlama before acquiring physical GPU3."
    "$RESERVER" stop
    reservation_state=released_for_experiment
    local free_mib=""
    for attempt in $(seq 1 24); do
        free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
            --id="$GPU" | tr -d ' ')
        [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )) && break
        sleep 5
    done
    if [[ ! "$free_mib" =~ ^[0-9]+$ ]] || (( free_mib < MIN_FREE_MIB )); then
        current_stage=acquire_gpu
        write_status blocked "GPU3 free memory ${free_mib:-unknown} MiB below 30720 MiB."
        exit 4
    fi
    monitor_resources &
    monitor_pid=$!
    local dataset control
    for dataset in Toys Beauty; do
        if [[ ! -s "$OUTPUT/$dataset/smoke/smoke.json" ]]; then
            run_stage smoke "$dataset"
        fi
        for control in C0 C1; do
            run_control "$dataset" "$control"
        done
    done
    current_stage=analysis
    current_dataset=both
    current_control=C0_vs_C1
    write_status running "Applying paired bootstrap and locked dual-head gates."
    "$PYTHON" "$ROOT/experiment/phase4/gcdh_p0_analyze.py" \
        --config "$CONFIG" --input-root "$OUTPUT" \
        --output "$OUTPUT/summary.json"
}

case "$ACTION" in
    start)
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            echo "GCDH P0 already running in $SESSION" >&2
            exit 1
        fi
        if [[ -s "$PID_FILE" ]]; then
            old_pid=$(tr -d '[:space:]' < "$PID_FILE")
            if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
                echo "GCDH P0 already running with PID $old_pid" >&2
                exit 1
            fi
        fi
        started_at=$(date -Is)
        tmux new-session -d -s "$SESSION" bash "$0" _worker "$started_at"
        reservation_state=scheduled_for_release
        current_stage=starting
        write_status starting "Persistent session started; preparing GCDH P0."
        echo "GCDH P0 started in tmux session $SESSION"
        echo "status: bash experiment/phase4/run_phase4_gcdh_p0.sh status"
        ;;
    status)
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            echo "tmux session: running ($SESSION)"
        else
            echo "tmux session: not running ($SESSION)"
        fi
        [[ -s "$STATUS" ]] && cat "$STATUS" || echo '{"status":"not_started"}'
        if [[ -s "$LOG" ]]; then
            echo
            echo "latest log lines:"
            tail -n 20 "$LOG"
        fi
        echo
        nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
            --format=csv,noheader,nounits --id="$GPU" 2>/dev/null || true
        ;;
    _worker)
        worker "${2:?missing timestamp}"
        ;;
    *)
        echo "usage: bash experiment/phase4/run_phase4_gcdh_p0.sh {start|status}" >&2
        exit 2
        ;;
esac
