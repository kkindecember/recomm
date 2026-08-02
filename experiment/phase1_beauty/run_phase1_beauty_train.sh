#!/usr/bin/env bash

# Run the formal phase-1 Beauty training on one selected physical GPU and
# restore the user's CodeLlama reservation immediately after training exits.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GPU=${1:?usage: run_phase1_beauty_train.sh PHYSICAL_GPU}
RESERVER="$ROOT/tools/run_codellama.sh"
STATUS="$ROOT/experiment/phase1_beauty_status.json"
PID_FILE="$ROOT/experiment/phase1_beauty_train.pid"
GPU_LOG="$ROOT/experiment/phase1_beauty_train_gpu.csv"
LOG="$ROOT/artifacts/phase1_beauty/logs/train_seed2023.log"
monitor_pid=""
workload_pid=""
started_at=$(date -Is)

write_status() {
    local stage=$1
    local status=$2
    local reason=$3
    local reservation=$4
    local workload_json=${workload_pid:-null}
    local tmp="${STATUS}.tmp.$$"
    printf '{\n  "experiment": "GRAM phase1 Beauty single-GPU reproduction",\n  "updated_at": "%s",\n  "started_at": "%s",\n  "stage": "%s",\n  "status": "%s",\n  "reason": "%s",\n  "gpu_selected": %s,\n  "runner_pid": %s,\n  "workload_pid": %s,\n  "log": "artifacts/phase1_beauty/logs/train_seed2023.log",\n  "gpu_telemetry": "experiment/phase1_beauty_train_gpu.csv",\n  "resource_reservation": "%s"\n}\n' \
        "$(date -Is)" "$started_at" "$stage" "$status" "$reason" "$GPU" \
        "$$" "$workload_json" "$reservation" > "$tmp"
    mv "$tmp" "$STATUS"
}

restore_reservation() {
    local experiment_rc=$?
    local reserve_rc=1
    trap - EXIT INT TERM HUP

    if [[ -n "$monitor_pid" ]] && kill -0 "$monitor_pid" 2>/dev/null; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
    if [[ -n "$workload_pid" ]] && kill -0 "$workload_pid" 2>/dev/null; then
        kill -TERM "$workload_pid" 2>/dev/null || true
        wait "$workload_pid" 2>/dev/null || true
    fi

    write_status formal_training_finished restoring_resource \
        "Formal training exited with code ${experiment_rc}; restoring GPU reservation." restoring
    echo "[$(date -Is)] formal training exit code: $experiment_rc"
    echo "[$(date -Is)] restoring CodeLlama reservation on physical GPU $GPU"

    for attempt in 1 2 3; do
        if "$RESERVER" start "$GPU"; then
            reserve_rc=0
            break
        fi
        echo "[$(date -Is)] reservation restart attempt $attempt failed"
        sleep 2
    done

    if (( reserve_rc == 0 )); then
        if (( experiment_rc == 0 )); then
            write_status formal_training_complete succeeded \
                "Formal training completed and GPU reservation was restored." restored
        else
            write_status formal_training_complete failed \
                "Formal training failed with code ${experiment_rc}; GPU reservation was restored." restored
        fi
    else
        write_status formal_training_complete failed_to_restore_resource \
            "Formal training exited with code ${experiment_rc}; failed to restore GPU reservation after three attempts." failed
    fi

    rm -f "$PID_FILE"
    if (( experiment_rc != 0 )); then
        exit "$experiment_rc"
    fi
    exit "$reserve_rc"
}
trap restore_reservation EXIT INT TERM HUP

mkdir -p "$(dirname "$LOG")"
exec >> "$LOG" 2>&1
printf '%s\n' "$$" > "$PID_FILE"

echo "[$(date -Is)] formal phase-1 Beauty training starting on physical GPU $GPU"
echo "timestamp,index,memory.used,memory.free,utilization.gpu" > "$GPU_LOG"
(
    while true; do
        timestamp=$(date -Is)
        sample=$(nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
            --format=csv,noheader,nounits --id="$GPU" 2>/dev/null || true)
        [[ -n "$sample" ]] && printf '%s,%s\n' "$timestamp" "$sample" >> "$GPU_LOG"
        sleep 10
    done
) &
monitor_pid=$!

cd "$ROOT/GRAM/command"
PHYSICAL_GPU="$GPU" HF_HOME="$ROOT/.cache/huggingface" \
    TRANSFORMERS_CACHE="$ROOT/.cache/huggingface" \
    PYTHON_BIN=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python \
    bash train_gram_beauty_single.sh &
workload_pid=$!
write_status formal_training running \
    "Formal 30-epoch seed-2023 training is running on physical GPU ${GPU}." \
    released_for_experiment
wait "$workload_pid"
