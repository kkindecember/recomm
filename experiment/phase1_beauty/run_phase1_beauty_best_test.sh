#!/usr/bin/env bash

# Run phase-D best-checkpoint testing on one selected physical GPU and restore
# the user's CodeLlama reservation immediately after testing exits.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GPU=${1:?usage: run_phase1_beauty_best_test.sh PHYSICAL_GPU}
RESERVER="$ROOT/tools/run_codellama.sh"
STATUS="$ROOT/experiment/phase1_beauty_status.json"
PID_FILE="$ROOT/experiment/phase1_beauty_best_test.pid"
GPU_LOG="$ROOT/experiment/phase1_beauty_best_test_gpu.csv"
LOG="$ROOT/artifacts/phase1_beauty/logs/test_best_checkpoint.log"
BEST_CHECKPOINT="$ROOT/GRAM/log/Beauty/4_20260718_2153/id_0_rec_30/model_rec_phase_1_epoch_25.pt"
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
    printf '{\n  "experiment": "GRAM phase1 Beauty single-GPU reproduction",\n  "updated_at": "%s",\n  "started_at": "%s",\n  "stage": "%s",\n  "status": "%s",\n  "reason": "%s",\n  "gpu_selected": %s,\n  "runner_pid": %s,\n  "workload_pid": %s,\n  "checkpoint_epoch": 25,\n  "selection_metric": "validation_NDCG@10",\n  "checkpoint": "GRAM/log/Beauty/4_20260718_2153/id_0_rec_30/model_rec_phase_1_epoch_25.pt",\n  "log": "artifacts/phase1_beauty/logs/test_best_checkpoint.log",\n  "gpu_telemetry": "experiment/phase1_beauty_best_test_gpu.csv",\n  "resource_reservation": "%s"\n}\n' \
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

    write_status best_checkpoint_test_finished restoring_resource \
        "Best-checkpoint test exited with code ${experiment_rc}; restoring GPU reservation." restoring
    echo "[$(date -Is)] best-checkpoint test exit code: $experiment_rc"
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
            write_status best_checkpoint_test_complete succeeded \
                "Epoch-25 best-checkpoint test completed and GPU reservation was restored." restored
        else
            write_status best_checkpoint_test_complete failed \
                "Epoch-25 best-checkpoint test failed with code ${experiment_rc}; GPU reservation was restored." restored
        fi
    else
        write_status best_checkpoint_test_complete failed_to_restore_resource \
            "Epoch-25 best-checkpoint test exited with code ${experiment_rc}; failed to restore GPU reservation after three attempts." failed
    fi

    rm -f "$PID_FILE"
    if (( experiment_rc != 0 )); then
        exit "$experiment_rc"
    fi
    exit "$reserve_rc"
}
trap restore_reservation EXIT INT TERM HUP

if [[ ! -s "$BEST_CHECKPOINT" ]]; then
    write_status best_checkpoint_test blocked "Selected epoch-25 checkpoint is missing or empty." unchanged
    exit 2
fi

mkdir -p "$(dirname "$LOG")"
exec >> "$LOG" 2>&1
printf '%s\n' "$$" > "$PID_FILE"

echo "[$(date -Is)] phase-D Beauty epoch-25 best-checkpoint test starting on physical GPU $GPU"
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
    BEST_CHECKPOINT="$BEST_CHECKPOINT" \
    bash test_gram_beauty_best_single.sh &
workload_pid=$!
write_status best_checkpoint_test running \
    "Epoch-25 checkpoint selected by validation NDCG@10 is under full-ranking test on physical GPU ${GPU}." \
    released_for_experiment
wait "$workload_pid"
