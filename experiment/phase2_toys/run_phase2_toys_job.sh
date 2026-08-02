#!/usr/bin/env bash

# Common phase-2 Toys background runner. Public wrappers select the job mode.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
EXPERIMENT_DIR="$ROOT/experiment/phase2_toys"
MODE=${1:?usage: run_phase2_toys_job.sh MODE PHYSICAL_GPU [BEST_CHECKPOINT]}
GPU=${2:?usage: run_phase2_toys_job.sh MODE PHYSICAL_GPU [BEST_CHECKPOINT]}
BEST_CHECKPOINT=${3:-}
RESERVER="$ROOT/tools/run_codellama.sh"
STATUS="$EXPERIMENT_DIR/phase2_toys_status.json"
MIN_FREE_MIB=${MIN_FREE_MIB:-30720}
monitor_pid=""
workload_pid=""
started_at=$(date -Is)

case "$MODE" in
    smoke)
        STAGE=smoke_test
        LABEL="Toys smoke test"
        COMMAND_SCRIPT=smoke_test_gram_toys_single.sh
        LOG="$ROOT/artifacts/phase2_toys/logs/smoke_test.log"
        PID_FILE="$EXPERIMENT_DIR/phase2_toys_smoke.pid"
        PREFIX=phase2_toys_smoke
        ;;
    train)
        STAGE=formal_training
        LABEL="Toys formal training"
        COMMAND_SCRIPT=train_gram_toys_single.sh
        LOG="$ROOT/artifacts/phase2_toys/logs/train_seed2023.log"
        PID_FILE="$EXPERIMENT_DIR/phase2_toys_train.pid"
        PREFIX=phase2_toys_train
        ;;
    best_test)
        STAGE=best_checkpoint_test
        LABEL="Toys best-checkpoint test"
        COMMAND_SCRIPT=test_gram_toys_best_single.sh
        LOG="$ROOT/artifacts/phase2_toys/logs/test_best_checkpoint.log"
        PID_FILE="$EXPERIMENT_DIR/phase2_toys_best_test.pid"
        PREFIX=phase2_toys_best_test
        if [[ ! -s "$BEST_CHECKPOINT" ]]; then
            echo "best checkpoint is missing or empty: $BEST_CHECKPOINT" >&2
            exit 2
        fi
        ;;
    *)
        echo "unsupported mode: $MODE" >&2
        exit 2
        ;;
esac

BOARD_LOG="$EXPERIMENT_DIR/${PREFIX}_gpu_board.csv"
PROCESS_LOG="$EXPERIMENT_DIR/${PREFIX}_gpu_process.csv"
DISK_LOG="$EXPERIMENT_DIR/${PREFIX}_disk.csv"

write_status() {
    local stage=$1
    local status=$2
    local reason=$3
    local reservation=$4
    local workload_json=${workload_pid:-null}
    local tmp="${STATUS}.tmp.$$"
    printf '{\n  "experiment": "GRAM phase2 Toys single-GPU reproduction",\n  "updated_at": "%s",\n  "started_at": "%s",\n  "stage": "%s",\n  "status": "%s",\n  "reason": "%s",\n  "mode": "%s",\n  "gpu_selected": %s,\n  "runner_pid": %s,\n  "workload_pid": %s,\n  "log": "%s",\n  "gpu_board_telemetry": "%s",\n  "gpu_process_telemetry": "%s",\n  "disk_telemetry": "%s",\n  "resource_reservation": "%s"\n}\n' \
        "$(date -Is)" "$started_at" "$stage" "$status" "$reason" "$MODE" \
        "$GPU" "$$" "$workload_json" "${LOG#"$ROOT/"}" \
        "${BOARD_LOG#"$ROOT/"}" "${PROCESS_LOG#"$ROOT/"}" \
        "${DISK_LOG#"$ROOT/"}" "$reservation" > "$tmp"
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

    write_status "${STAGE}_finished" restoring_resource \
        "${LABEL} exited with code ${experiment_rc}; restoring GPU reservation." restoring
    echo "[$(date -Is)] $LABEL exit code: $experiment_rc"
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
            write_status "${STAGE}_complete" succeeded \
                "${LABEL} completed and GPU reservation was restored." restored
        else
            write_status "${STAGE}_complete" failed \
                "${LABEL} failed with code ${experiment_rc}; GPU reservation was restored." restored
        fi
    else
        write_status "${STAGE}_complete" failed_to_restore_resource \
            "${LABEL} exited with code ${experiment_rc}; failed to restore GPU reservation." failed
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

free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
    --id="$GPU" 2>/dev/null | tr -d ' ')
if [[ ! "$free_mib" =~ ^[0-9]+$ ]]; then
    write_status "$STAGE" blocked "Could not read free memory on physical GPU ${GPU}." unchanged
    exit 3
fi
if (( free_mib < MIN_FREE_MIB )); then
    write_status "$STAGE" blocked \
        "Physical GPU ${GPU} has ${free_mib} MiB free; ${MIN_FREE_MIB} MiB is required." unchanged
    exit 4
fi

echo "[$(date -Is)] $LABEL starting on physical GPU $GPU with ${free_mib} MiB free"
echo "timestamp,index,memory.used,memory.free,utilization.gpu" > "$BOARD_LOG"
echo "timestamp,pid,used_gpu_memory,is_workload" > "$PROCESS_LOG"
echo "timestamp,filesystem,available_kib" > "$DISK_LOG"

cd "$ROOT/GRAM/command"
if [[ "$MODE" == best_test ]]; then
    PHYSICAL_GPU="$GPU" HF_HOME="$ROOT/.cache/huggingface" \
        TRANSFORMERS_CACHE="$ROOT/.cache/huggingface" \
        PYTHON_BIN=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python3.9 \
        BEST_CHECKPOINT="$BEST_CHECKPOINT" \
        bash "$COMMAND_SCRIPT" &
else
    PHYSICAL_GPU="$GPU" HF_HOME="$ROOT/.cache/huggingface" \
        TRANSFORMERS_CACHE="$ROOT/.cache/huggingface" \
        PYTHON_BIN=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python3.9 \
        bash "$COMMAND_SCRIPT" &
fi
workload_pid=$!

(
    sample_index=0
    while true; do
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
            printf '%s,%s,%s,%s\n' "$timestamp" "$pid" "$used_memory" "$is_workload" \
                >> "$PROCESS_LOG"
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
) &
monitor_pid=$!

write_status "$STAGE" running \
    "${LABEL} is running on physical GPU ${GPU}." released_for_experiment
wait "$workload_pid"
