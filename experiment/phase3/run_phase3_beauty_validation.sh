#!/usr/bin/env bash

# Complete phase-3 S0 on Beauty: validation inference -> CPU diagnostics -> report.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ACTION=${1:-status}
GPU=3
MIN_FREE_MIB=30720
SESSION=gram_phase3_s0_beauty_validation
RESERVER=/home/jiangtangyunzhi/projects/UnitTest/tools/run_codellama.sh
STATUS="$ROOT/experiment/phase3/phase3_s0_status.json"
PID_FILE="$ROOT/experiment/phase3/phase3_s0_beauty_validation.pid"
LOG="$ROOT/artifacts/phase3/logs/s0_beauty_validation.log"
BOARD_LOG="$ROOT/experiment/phase3/s0_beauty_validation_gpu_board.csv"
PROCESS_LOG="$ROOT/experiment/phase3/s0_beauty_validation_gpu_process.csv"
DISK_LOG="$ROOT/experiment/phase3/s0_beauty_validation_disk.csv"
CHECKPOINT="$ROOT/GRAM/log/Beauty/4_20260718_2153/id_0_rec_30/model_rec_phase_1_epoch_25.pt"
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python3.9
started_at=$(date -Is)
workload_pid=""
monitor_pid=""
reservation_state=unchanged

write_status() {
    local stage=$1
    local state=$2
    local reason=$3
    local workload_json=${workload_pid:-null}
    local tmp="${STATUS}.tmp.$$"
    mkdir -p "$(dirname "$STATUS")"
    printf '{\n  "experiment": "GRAM phase3 S0 Beauty validation and offline diagnostics",\n  "updated_at": "%s",\n  "started_at": "%s",\n  "stage": "%s",\n  "status": "%s",\n  "reason": "%s",\n  "gpu_selected": 3,\n  "minimum_free_mib": 30720,\n  "runner_pid": %s,\n  "workload_pid": %s,\n  "checkpoint": "GRAM/log/Beauty/4_20260718_2153/id_0_rec_30/model_rec_phase_1_epoch_25.pt",\n  "log": "artifacts/phase3/logs/s0_beauty_validation.log",\n  "gpu_board_telemetry": "experiment/phase3/s0_beauty_validation_gpu_board.csv",\n  "gpu_process_telemetry": "experiment/phase3/s0_beauty_validation_gpu_process.csv",\n  "disk_telemetry": "experiment/phase3/s0_beauty_validation_disk.csv",\n  "report": "report/第三阶段/GRAM_第三阶段_S0离线诊断报告.md",\n  "resource_reservation": "%s",\n  "status_command": "bash experiment/phase3/run_phase3_beauty_validation.sh status"\n}\n' \
        "$(date -Is)" "$started_at" "$stage" "$state" "$reason" "$$" \
        "$workload_json" "$reservation_state" > "$tmp"
    mv "$tmp" "$STATUS"
}

restore_codellama() {
    local restored=1
    reservation_state=restoring
    write_status s0_beauty_validation_finished restoring_resource \
        "Experiment work ended; restoring CodeLlama reservation on physical GPU3."
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
    local experiment_rc=$?
    local restore_rc=0
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

    if (( experiment_rc == 0 && restore_rc == 0 )); then
        write_status s0_complete succeeded \
            "Beauty validation, dual-dataset S0 analysis, and report update completed; CodeLlama reservation was restored."
    elif (( restore_rc != 0 )); then
        reservation_state=failed
        write_status s0_complete failed_to_restore_resource \
            "Experiment exited with code ${experiment_rc}; CodeLlama restoration failed after three attempts."
    else
        write_status s0_complete failed \
            "Beauty S0 pipeline exited with code ${experiment_rc}; CodeLlama reservation was restored; no automatic retry was attempted."
    fi
    rm -f "$PID_FILE"
    echo "[$(date -Is)] phase3 S0 Beauty pipeline exit=$experiment_rc restore=$restore_rc"
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

worker() {
    started_at=${1:?missing start timestamp}
    local start_epoch free_mib prediction prediction_epoch
    trap finish EXIT INT TERM HUP
    mkdir -p "$(dirname "$LOG")" "$(dirname "$STATUS")"
    exec >> "$LOG" 2>&1
    printf '%s\n' "$$" > "$PID_FILE"

    if [[ ! -s "$CHECKPOINT" ]]; then
        write_status s0_beauty_validation blocked "Locked Beauty epoch-25 checkpoint is missing or empty."
        exit 2
    fi

    echo "[$(date -Is)] stopping CodeLlama reservation before Beauty validation"
    write_status s0_beauty_validation releasing_resource \
        "Stopping CodeLlama reservation before acquiring physical GPU3."
    if ! "$RESERVER" stop; then
        write_status s0_beauty_validation blocked "Failed to stop CodeLlama reservation."
        exit 3
    fi
    reservation_state=released_for_experiment

    free_mib=""
    for release_check in $(seq 1 24); do
        free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
            --id="$GPU" 2>/dev/null | tr -d ' ')
        if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )); then
            break
        fi
        echo "[$(date -Is)] waiting for GPU3 release: free_mib=${free_mib:-unknown} check=${release_check}/24"
        sleep 5
    done
    if [[ ! "$free_mib" =~ ^[0-9]+$ ]]; then
        write_status s0_beauty_validation blocked "Could not read free memory on physical GPU3."
        exit 4
    fi
    if (( free_mib < MIN_FREE_MIB )); then
        write_status s0_beauty_validation blocked \
            "Physical GPU3 has ${free_mib} MiB free; 30720 MiB is required."
        exit 5
    fi

    echo "[$(date -Is)] Beauty epoch-25 validation starting on GPU3 with ${free_mib} MiB free"
    start_epoch=$(date +%s)
    cd "$ROOT/GRAM/command"
    PHYSICAL_GPU="$GPU" HF_HOME="$ROOT/.cache/huggingface" \
        TRANSFORMERS_CACHE="$ROOT/.cache/huggingface" \
        PYTHON_BIN="$PYTHON" BEST_CHECKPOINT="$CHECKPOINT" \
        timeout --signal=TERM 10800 bash validate_gram_beauty_best_single.sh &
    workload_pid=$!
    monitor_resources &
    monitor_pid=$!
    write_status s0_beauty_validation running \
        "Locked epoch-25 checkpoint is running full-ranking Beauty validation on physical GPU3."
    wait "$workload_pid"
    workload_pid=""

    if kill -0 "$monitor_pid" 2>/dev/null; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
    monitor_pid=""

    prediction=$(find "$ROOT/GRAM/preds" -maxdepth 1 -type f \
        -name '*_Beauty_sequential_pred_validation.tsv' -printf '%T@ %p\n' \
        | sort -nr | head -n 1 | cut -d' ' -f2-)
    if [[ -z "$prediction" || ! -s "$prediction" ]]; then
        echo "No Beauty validation prediction was produced" >&2
        exit 6
    fi
    prediction_epoch=$(stat -c %Y "$prediction")
    if (( prediction_epoch < start_epoch )); then
        echo "Newest Beauty validation prediction predates this run: $prediction" >&2
        exit 7
    fi
    echo "[$(date -Is)] validation prediction: $prediction"

    restore_codellama

    cd "$ROOT"
    write_status s0_beauty_offline_analysis running \
        "GPU validation completed; CPU-only Beauty diagnostics and report update are running."
    HF_HOME="$ROOT/.cache/huggingface" TRANSFORMERS_CACHE="$ROOT/.cache/huggingface" \
        "$PYTHON" experiment/phase3/s0_offline_diagnostics.py \
        --dataset Beauty \
        --predictions "$prediction" \
        --mode validation \
        --output-dir artifacts/phase3/s0/Beauty/validation \
        --local-files-only
    "$PYTHON" experiment/phase3/s0_build_report.py
}

case "$ACTION" in
    start)
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            echo "Beauty S0 is already running in tmux session $SESSION" >&2
            exit 1
        fi
        if [[ -s "$PID_FILE" ]]; then
            old_pid=$(tr -d '[:space:]' < "$PID_FILE")
            if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
                echo "Beauty S0 is already running with PID $old_pid" >&2
                exit 1
            fi
        fi
        started_at=$(date -Is)
        tmux new-session -d -s "$SESSION" bash "$0" _worker "$started_at"
        runner_pid=$(tmux display-message -p -t "$SESSION" '#{pane_pid}')
        workload_pid=""
        reservation_state=scheduled_for_release
        write_status s0_beauty_validation starting \
            "Persistent background session started; preparing to release CodeLlama and acquire GPU3."
        echo "Beauty S0 started in tmux session $SESSION with PID $runner_pid"
        echo "status: bash experiment/phase3/run_phase3_beauty_validation.sh status"
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
            echo '{"stage":"s0_beauty_validation","status":"not_started"}'
        fi
        if [[ -s "$LOG" ]]; then
            echo
            echo "latest log lines:"
            tail -n 12 "$LOG"
        fi
        echo
        nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
            --format=csv,noheader,nounits --id="$GPU" 2>/dev/null || true
        ;;
    _worker)
        worker "${2:?missing start timestamp}"
        ;;
    *)
        echo "usage: bash experiment/phase3/run_phase3_beauty_validation.sh {start|status}" >&2
        exit 2
        ;;
esac
