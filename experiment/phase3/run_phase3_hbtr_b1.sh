#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ACTION=${1:-status}
GPU=3
MIN_FREE_MIB=30720
SESSION=gram_phase3_hbtr_b1
RESERVER=/home/jiangtangyunzhi/projects/UnitTest/tools/run_codellama.sh
STATUS="$ROOT/experiment/phase3/phase3_hbtr_b1_status.json"
PID_FILE="$ROOT/experiment/phase3/phase3_hbtr_b1.pid"
LOG="$ROOT/artifacts/phase3/logs/hbtr_b1_smoke.log"
BOARD_LOG="$ROOT/experiment/phase3/hbtr_b1_gpu_board.csv"
PROCESS_LOG="$ROOT/experiment/phase3/hbtr_b1_gpu_process.csv"
DISK_LOG="$ROOT/experiment/phase3/hbtr_b1_disk.csv"
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python3.9
started_at=$(date -Is)
workload_pid=""
monitor_pid=""
reservation_state=unchanged
current_dataset=none

write_status() {
    local stage=$1 state=$2 reason=$3
    local workload_json=${workload_pid:-null}
    local tmp="${STATUS}.tmp.$$"
    mkdir -p "$(dirname "$STATUS")"
    printf '{\n  "experiment": "GRAM phase3 HBTR-B1 correctness smoke",\n  "updated_at": "%s",\n  "started_at": "%s",\n  "stage": "%s",\n  "status": "%s",\n  "reason": "%s",\n  "current_dataset": "%s",\n  "gpu_selected": 3,\n  "minimum_free_mib": 30720,\n  "runner_pid": %s,\n  "workload_pid": %s,\n  "log": "artifacts/phase3/logs/hbtr_b1_smoke.log",\n  "gpu_board_telemetry": "experiment/phase3/hbtr_b1_gpu_board.csv",\n  "gpu_process_telemetry": "experiment/phase3/hbtr_b1_gpu_process.csv",\n  "disk_telemetry": "experiment/phase3/hbtr_b1_disk.csv",\n  "output_dir": "artifacts/phase3/hbtr_b1_smoke",\n  "resource_reservation": "%s",\n  "status_command": "bash experiment/phase3/run_phase3_hbtr_b1.sh status"\n}\n' \
        "$(date -Is)" "$started_at" "$stage" "$state" "$reason" \
        "$current_dataset" "$$" "$workload_json" "$reservation_state" > "$tmp"
    mv "$tmp" "$STATUS"
}

restore_codellama() {
    local restored=1
    reservation_state=restoring
    write_status hbtr_b1_finished restoring_resource \
        "B1 smoke work ended; restoring CodeLlama reservation on physical GPU3."
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
    if (( experiment_rc == 0 && restore_rc == 0 )); then
        write_status hbtr_b1_complete succeeded \
            "Toys and Beauty B1 correctness smoke passed; weights were discarded and CodeLlama was restored."
    elif (( restore_rc != 0 )); then
        reservation_state=failed
        write_status hbtr_b1_complete failed_to_restore_resource \
            "B1 exited with code ${experiment_rc}; CodeLlama restoration failed after three attempts."
    else
        write_status hbtr_b1_complete failed \
            "B1 exited with code ${experiment_rc}; no automatic retry was attempted and CodeLlama was restored."
    fi
    rm -f "$PID_FILE"
    if (( experiment_rc != 0 )); then
        exit "$experiment_rc"
    fi
    exit "$restore_rc"
}

monitor_resources() {
    local sample_index=0
    echo "timestamp,index,memory.used,memory.free,utilization.gpu" > "$BOARD_LOG"
    echo "timestamp,pid,used_gpu_memory,is_workload" > "$PROCESS_LOG"
    echo "timestamp,filesystem,available_kib" > "$DISK_LOG"
    while true; do
        local timestamp board available_kib pid used_memory is_workload
        timestamp=$(date -Is)
        board=$(nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
            --format=csv,noheader,nounits --id="$GPU" 2>/dev/null || true)
        [[ -n "$board" ]] && printf '%s,%s\n' "$timestamp" "$board" >> "$BOARD_LOG"
        while IFS=',' read -r pid used_memory; do
            pid=${pid//[[:space:]]/}
            used_memory=${used_memory//[[:space:]]/}
            [[ "$pid" =~ ^[0-9]+$ ]] || continue
            is_workload=0
            [[ "$pid" == "$workload_pid" ]] && is_workload=1
            printf '%s,%s,%s,%s\n' "$timestamp" "$pid" "$used_memory" "$is_workload" >> "$PROCESS_LOG"
        done < <(nvidia-smi --query-compute-apps=pid,used_gpu_memory \
            --format=csv,noheader,nounits --id="$GPU" 2>/dev/null || true)
        if (( sample_index % 60 == 0 )); then
            available_kib=$(df --output=avail /home 2>/dev/null | tail -n 1 | tr -d ' ')
            [[ "$available_kib" =~ ^[0-9]+$ ]] \
                && printf '%s,/home,%s\n' "$timestamp" "$available_kib" >> "$DISK_LOG"
        fi
        sample_index=$((sample_index + 1))
        sleep 5
    done
}

run_dataset() {
    local dataset=$1
    current_dataset=$dataset
    write_status hbtr_b1_smoke running \
        "Locked correctness-only smoke is running for ${dataset}; metrics cannot support effect claims."
    CUDA_VISIBLE_DEVICES="$GPU" HF_HOME="$ROOT/.cache/huggingface" \
        TRANSFORMERS_CACHE="$ROOT/.cache/huggingface" \
        timeout --signal=TERM 900 "$PYTHON" "$ROOT/experiment/phase3/hbtr_b1_smoke.py" \
        --dataset "$dataset" \
        --max-samples 100 \
        --max-train-steps 2 \
        --output-dir "$ROOT/artifacts/phase3/hbtr_b1_smoke/$dataset" &
    workload_pid=$!
    write_status hbtr_b1_smoke running \
        "Locked correctness-only smoke is running for ${dataset}; metrics cannot support effect claims."
    wait "$workload_pid"
    workload_pid=""
}

worker() {
    started_at=${1:?missing start timestamp}
    trap finish EXIT INT TERM HUP
    mkdir -p "$(dirname "$LOG")" "$(dirname "$STATUS")"
    exec >> "$LOG" 2>&1
    printf '%s\n' "$$" > "$PID_FILE"

    for required in \
        "$ROOT/GRAM/log/Toys/1_20260720_1830/id_0_rec_30/model_rec_phase_1_epoch_30.pt" \
        "$ROOT/GRAM/log/Beauty/4_20260718_2153/id_0_rec_30/model_rec_phase_1_epoch_25.pt"; do
        if [[ ! -s "$required" ]]; then
            write_status hbtr_b1 blocked "Locked baseline checkpoint is missing: ${required}."
            exit 2
        fi
    done

    write_status hbtr_b1 releasing_resource \
        "Stopping CodeLlama reservation before acquiring physical GPU3."
    if ! "$RESERVER" stop; then
        write_status hbtr_b1 blocked "Failed to stop CodeLlama reservation."
        exit 3
    fi
    reservation_state=released_for_experiment

    local free_mib=""
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
        write_status hbtr_b1 blocked \
            "Physical GPU3 did not reach the locked 30720 MiB free-memory threshold."
        exit 4
    fi

    monitor_resources &
    monitor_pid=$!
    run_dataset Toys
    run_dataset Beauty
}

case "$ACTION" in
    start)
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            echo "HBTR-B1 is already running in tmux session $SESSION" >&2
            exit 1
        fi
        if [[ -s "$PID_FILE" ]]; then
            old_pid=$(tr -d '[:space:]' < "$PID_FILE")
            if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
                echo "HBTR-B1 is already running with PID $old_pid" >&2
                exit 1
            fi
        fi
        started_at=$(date -Is)
        tmux new-session -d -s "$SESSION" bash "$0" _worker "$started_at"
        reservation_state=scheduled_for_release
        write_status hbtr_b1 starting \
            "Persistent session started; preparing locked Toys and Beauty correctness smoke."
        echo "HBTR-B1 started in tmux session $SESSION"
        echo "status: bash experiment/phase3/run_phase3_hbtr_b1.sh status"
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
            echo '{"stage":"hbtr_b1","status":"not_started"}'
        fi
        if [[ -s "$LOG" ]]; then
            echo
            echo "latest log lines:"
            tail -n 16 "$LOG"
        fi
        echo
        nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
            --format=csv,noheader,nounits --id="$GPU" 2>/dev/null || true
        ;;
    _worker)
        worker "${2:?missing start timestamp}"
        ;;
    *)
        echo "usage: bash experiment/phase3/run_phase3_hbtr_b1.sh {start|status}" >&2
        exit 2
        ;;
esac
