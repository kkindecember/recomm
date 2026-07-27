#!/usr/bin/env bash

# Run the phase-1 Beauty smoke test on one selected physical GPU and restore
# the user's CodeLlama reservation immediately after the test exits.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GPU=${1:?usage: run_phase1_beauty_smoke.sh PHYSICAL_GPU}
RESERVER=/home/jiangtangyunzhi/projects/UnitTest/tools/run_codellama.sh
STATUS="$ROOT/experiment/phase1_beauty_status.json"
PID_FILE="$ROOT/experiment/phase1_beauty_smoke.pid"
GPU_LOG="$ROOT/experiment/phase1_beauty_smoke_gpu.csv"
LOG="$ROOT/artifacts/phase1_beauty/logs/smoke_test.log"
monitor_pid=""
workload_pid=""

write_status() {
    local stage=$1
    local status=$2
    local reason=$3
    local reservation=$4
    local tmp="${STATUS}.tmp.$$"
    local workload_json=${workload_pid:-null}
    printf '{\n  "experiment": "GRAM phase1 Beauty single-GPU reproduction",\n  "updated_at": "%s",\n  "stage": "%s",\n  "status": "%s",\n  "reason": "%s",\n  "gpu_selected": %s,\n  "runner_pid": %s,\n  "workload_pid": %s,\n  "log": "artifacts/phase1_beauty/logs/smoke_test.log",\n  "gpu_telemetry": "experiment/phase1_beauty_smoke_gpu.csv",\n  "resource_reservation": "%s"\n}\n' \
        "$(date -Is)" "$stage" "$status" "$reason" "$GPU" "$$" \
        "$workload_json" "$reservation" > "$tmp"
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

    write_status smoke_test_finished restoring_resource \
        "Smoke test exited with code ${experiment_rc}; restoring GPU reservation." restoring
    echo "[$(date -Is)] smoke test exit code: $experiment_rc"
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
            write_status smoke_test_complete succeeded \
                "Smoke test completed and GPU reservation was restored." restored
        else
            write_status smoke_test_complete failed \
                "Smoke test failed with code ${experiment_rc}; GPU reservation was restored." restored
        fi
    else
        write_status smoke_test_complete failed_to_restore_resource \
            "Smoke test exited with code ${experiment_rc}; failed to restore GPU reservation after three attempts." failed
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
write_status smoke_test running "Smoke test is running on physical GPU ${GPU}." released_for_experiment

echo "[$(date -Is)] phase-1 Beauty smoke test starting on physical GPU $GPU"
echo "timestamp,index,memory.used,memory.free,utilization.gpu" > "$GPU_LOG"
(
    while true; do
        timestamp=$(date -Is)
        sample=$(nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
            --format=csv,noheader,nounits --id="$GPU" 2>/dev/null || true)
        [[ -n "$sample" ]] && printf '%s,%s\n' "$timestamp" "$sample" >> "$GPU_LOG"
        sleep 5
    done
) &
monitor_pid=$!

cd "$ROOT/GRAM/command"
PHYSICAL_GPU="$GPU" HF_HOME="$ROOT/.cache/huggingface" \
    TRANSFORMERS_CACHE="$ROOT/.cache/huggingface" \
    PYTHON_BIN=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python \
    bash smoke_test_gram_beauty_single.sh &
workload_pid=$!
write_status smoke_test running \
    "Smoke test is running on physical GPU ${GPU}; workload_pid is the GPU Python process." \
    released_for_experiment
wait "$workload_pid"
