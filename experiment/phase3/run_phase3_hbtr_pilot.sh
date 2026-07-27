#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ACTION=${1:-status}
GPU=3
MIN_FREE_MIB=30720
MIN_DISK_KIB=52428800
SESSION=gram_phase3_hbtr_pilot
RESERVER=/home/jiangtangyunzhi/projects/UnitTest/tools/run_codellama.sh
STATUS="$ROOT/experiment/phase3/phase3_hbtr_pilot_status.json"
PID_FILE="$ROOT/experiment/phase3/phase3_hbtr_pilot.pid"
LOG="$ROOT/artifacts/phase3/logs/hbtr_pilot.log"
BOARD_LOG="$ROOT/experiment/phase3/hbtr_pilot_gpu_board.csv"
PROCESS_LOG="$ROOT/experiment/phase3/hbtr_pilot_gpu_process.csv"
DISK_LOG="$ROOT/experiment/phase3/hbtr_pilot_disk.csv"
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python3.9
started_at=$(date -Is)
workload_pid=""
monitor_pid=""
reservation_state=unchanged
current_dataset=none
current_control=none
current_stage=none

write_status() {
    local state=$1 reason=$2
    local workload_json=${workload_pid:-null}
    local tmp="${STATUS}.tmp.$$"
    mkdir -p "$(dirname "$STATUS")"
    printf '{\n  "experiment": "GRAM phase3 HBTR 10%% pilot",\n  "updated_at": "%s",\n  "started_at": "%s",\n  "stage": "%s",\n  "status": "%s",\n  "reason": "%s",\n  "current_dataset": "%s",\n  "current_control": "%s",\n  "gpu_selected": 3,\n  "minimum_free_mib": 30720,\n  "minimum_disk_kib": 52428800,\n  "runner_pid": %s,\n  "workload_pid": %s,\n  "log": "artifacts/phase3/logs/hbtr_pilot.log",\n  "output_dir": "artifacts/phase3/hbtr_pilot",\n  "resource_reservation": "%s",\n  "test_data_allowed": false,\n  "status_command": "bash experiment/phase3/run_phase3_hbtr_pilot.sh status"\n}\n' \
        "$(date -Is)" "$started_at" "$current_stage" "$state" "$reason" \
        "$current_dataset" "$current_control" "$$" "$workload_json" \
        "$reservation_state" > "$tmp"
    mv "$tmp" "$STATUS"
}

restore_codellama() {
    local restored=1
    reservation_state=restoring
    current_stage=restore_resource
    write_status restoring_resource "Pilot ended; restoring CodeLlama reservation on physical GPU3."
    for attempt in 1 2 3; do
        if "$RESERVER" start "$GPU"; then
            restored=0
            reservation_state=restored
            break
        fi
        echo "[$(date -Is)] CodeLlama restore attempt $attempt failed"
        sleep 2
    done
    return "$restored"
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
    if [[ "$reservation_state" != restored ]]; then
        restore_codellama || restore_rc=$?
    fi
    current_dataset=none
    current_control=none
    current_stage=complete
    if (( experiment_rc == 0 && restore_rc == 0 )); then
        write_status succeeded "All locked pilot stages and analysis completed; CodeLlama was restored."
    elif (( restore_rc != 0 )); then
        reservation_state=failed
        write_status failed_to_restore_resource \
            "Pilot exited with code ${experiment_rc}; CodeLlama restoration failed after three attempts."
    else
        write_status failed \
            "Pilot exited with code ${experiment_rc}; no automatic retry was attempted and CodeLlama was restored."
    fi
    rm -f "$PID_FILE"
    if (( experiment_rc != 0 )); then
        exit "$experiment_rc"
    fi
    exit "$restore_rc"
}

monitor_resources() {
    local sample_index=0
    [[ -s "$BOARD_LOG" ]] || echo "timestamp,index,memory.used,memory.free,utilization.gpu" > "$BOARD_LOG"
    [[ -s "$PROCESS_LOG" ]] || echo "timestamp,pid,used_gpu_memory" > "$PROCESS_LOG"
    [[ -s "$DISK_LOG" ]] || echo "timestamp,filesystem,available_kib" > "$DISK_LOG"
    while true; do
        local timestamp board available_kib
        timestamp=$(date -Is)
        board=$(nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
            --format=csv,noheader,nounits --id="$GPU" 2>/dev/null || true)
        [[ -n "$board" ]] && printf '%s,%s\n' "$timestamp" "$board" >> "$BOARD_LOG"
        nvidia-smi --query-compute-apps=pid,used_gpu_memory \
            --format=csv,noheader,nounits --id="$GPU" 2>/dev/null \
            | while IFS= read -r row; do printf '%s,%s\n' "$timestamp" "$row"; done \
            >> "$PROCESS_LOG" || true
        if (( sample_index % 60 == 0 )); then
            available_kib=$(df --output=avail /home 2>/dev/null | tail -n 1 | tr -d ' ')
            if [[ "$available_kib" =~ ^[0-9]+$ ]]; then
                printf '%s,/home,%s\n' "$timestamp" "$available_kib" >> "$DISK_LOG"
                if (( available_kib < MIN_DISK_KIB )); then
                    echo "[$timestamp] disk guard triggered: available_kib=$available_kib" >> "$LOG"
                    kill -TERM "$$" 2>/dev/null || true
                    return 1
                fi
            fi
        fi
        sample_index=$((sample_index + 1))
        sleep 5
    done
}

run_stage() {
    local stage=$1 dataset=$2 control=${3:-}
    local timeout_seconds=21600
    current_stage=$stage
    current_dataset=$dataset
    current_control=${control:-none}
    [[ "$stage" == preflight ]] && timeout_seconds=900
    local available_kib
    available_kib=$(df --output=avail /home | tail -n 1 | tr -d ' ')
    if [[ ! "$available_kib" =~ ^[0-9]+$ ]] || (( available_kib < MIN_DISK_KIB )); then
        write_status blocked "Disk guard refused ${stage}: available_kib=${available_kib:-unknown}."
        return 5
    fi
    write_status running "Running locked ${stage} stage for ${dataset}/${current_control}."
    local command=(
        "$PYTHON" "$ROOT/experiment/phase3/hbtr_pilot.py"
        --stage "$stage" --dataset "$dataset"
        --output-root "$ROOT/artifacts/phase3/hbtr_pilot"
    )
    if [[ -n "$control" ]]; then
        command+=(--control "$control")
    fi
    CUDA_VISIBLE_DEVICES="$GPU" HF_HOME="$ROOT/.cache/huggingface" \
        TRANSFORMERS_CACHE="$ROOT/.cache/huggingface" \
        timeout --signal=TERM "$timeout_seconds" "${command[@]}" &
    workload_pid=$!
    write_status running "Running locked ${stage} stage for ${dataset}/${current_control}."
    wait "$workload_pid"
    workload_pid=""
}

run_control_resumable() {
    local dataset=$1 control=$2
    local control_dir="$ROOT/artifacts/phase3/hbtr_pilot/$dataset/$control"
    if [[ -s "$control_dir/training_summary.json" \
        && -s "$control_dir/model.pt" \
        && -s "$control_dir/validation_per_user.csv" \
        && -s "$control_dir/validation_summary.json" ]]; then
        echo "[$(date -Is)] RESUME_SKIP complete control dataset=$dataset control=$control"
        return 0
    fi
    if [[ -s "$control_dir/training_summary.json" && -s "$control_dir/model.pt" ]]; then
        echo "[$(date -Is)] RESUME_VALIDATE trained control dataset=$dataset control=$control"
        run_stage validate "$dataset" "$control"
        return 0
    fi
    echo "[$(date -Is)] RESUME_TRAIN incomplete control dataset=$dataset control=$control"
    run_stage train "$dataset" "$control"
}

worker() {
    started_at=${1:?missing start timestamp}
    trap finish EXIT INT TERM HUP
    mkdir -p "$(dirname "$LOG")" "$(dirname "$STATUS")" "$ROOT/.cache/huggingface"
    exec >> "$LOG" 2>&1
    printf '%s\n' "$$" > "$PID_FILE"

    local required
    for required in \
        "$ROOT/GRAM/log/Toys/1_20260720_1830/id_0_rec_30/model_rec_phase_1_epoch_30.pt" \
        "$ROOT/GRAM/log/Beauty/4_20260718_2153/id_0_rec_30/model_rec_phase_1_epoch_25.pt" \
        "$ROOT/artifacts/phase3/hbtr_pilot_splits/Toys/manifest.json" \
        "$ROOT/artifacts/phase3/hbtr_pilot_splits/Beauty/manifest.json" \
        "$ROOT/artifacts/phase3/hbtr_b1_smoke/Toys/negative_cache.json" \
        "$ROOT/artifacts/phase3/hbtr_b1_smoke/Beauty/negative_cache.json"; do
        if [[ ! -s "$required" ]]; then
            current_stage=precondition
            write_status blocked "Required locked material is missing: ${required}."
            exit 2
        fi
    done

    current_stage=release_resource
    write_status releasing_resource "Stopping CodeLlama reservation before acquiring physical GPU3."
    if ! "$RESERVER" stop; then
        write_status blocked "Failed to stop CodeLlama reservation."
        exit 3
    fi
    reservation_state=released_for_experiment

    local free_mib=""
    local release_check
    for release_check in $(seq 1 24); do
        free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
            --id="$GPU" 2>/dev/null | tr -d ' ')
        if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )); then
            break
        fi
        echo "[$(date -Is)] waiting for GPU3 release: free_mib=${free_mib:-unknown} check=${release_check}/24"
        sleep 5
    done
    if [[ ! "$free_mib" =~ ^[0-9]+$ ]] || (( free_mib < MIN_FREE_MIB )); then
        current_stage=acquire_gpu
        write_status blocked "Physical GPU3 did not reach the locked 30720 MiB free-memory threshold."
        exit 4
    fi

    monitor_resources &
    monitor_pid=$!
    run_stage preflight Toys
    run_stage preflight Beauty
    local dataset control
    for dataset in Toys Beauty; do
        if [[ -s "$ROOT/artifacts/phase3/hbtr_pilot/$dataset/cache/negative_cache.json" ]]; then
            echo "[$(date -Is)] RESUME_SKIP existing cache dataset=$dataset"
        else
            run_stage cache "$dataset"
        fi
        for control in C0 C1 C2 C3 C4; do
            run_control_resumable "$dataset" "$control"
        done
    done
    current_stage=analysis
    current_dataset=both
    current_control=all
    write_status running "Applying the locked bootstrap and promotion gate."
    "$PYTHON" "$ROOT/experiment/phase3/hbtr_pilot_analyze.py"
}

case "$ACTION" in
    start)
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            echo "HBTR pilot is already running in tmux session $SESSION" >&2
            exit 1
        fi
        if [[ -s "$PID_FILE" ]]; then
            old_pid=$(tr -d '[:space:]' < "$PID_FILE")
            if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
                echo "HBTR pilot is already running with PID $old_pid" >&2
                exit 1
            fi
        fi
        started_at=$(date -Is)
        tmux new-session -d -s "$SESSION" bash "$0" _worker "$started_at"
        reservation_state=scheduled_for_release
        current_stage=starting
        write_status starting "Persistent session started; preparing the locked HBTR 10% pilot."
        echo "HBTR pilot started in tmux session $SESSION"
        echo "status: bash experiment/phase3/run_phase3_hbtr_pilot.sh status"
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
            echo '{"stage":"hbtr_pilot","status":"not_started"}'
        fi
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
        worker "${2:?missing start timestamp}"
        ;;
    *)
        echo "usage: bash experiment/phase3/run_phase3_hbtr_pilot.sh {start|status}" >&2
        exit 2
        ;;
esac
