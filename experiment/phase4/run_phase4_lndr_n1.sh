#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python3.9
CONFIG="$ROOT/artifacts/phase4/configs/lndr_n1_preregistered.json"
OUTPUT="$ROOT/artifacts/phase4/lndr_n1"
LOG="$ROOT/artifacts/phase4/logs/lndr_n1.log"
STATUS="$OUTPUT/status.json"
GPU=4
MIN_FREE_MIB=30720
MIN_DISK_KIB=52428800

mkdir -p "$OUTPUT" "$(dirname "$LOG")"
mark_failure() {
    local rc=$?
    if (( rc != 0 )); then
        printf '{"experiment_id":"GRAM_PHASE4_LNDR_N1","status":"failed","exit_code":%s,"no_automatic_retry":true}\n' "$rc" > "$STATUS"
    fi
}
trap mark_failure EXIT
available=$(df --output=avail /home | tail -n 1 | tr -d ' ')
if [[ ! "$available" =~ ^[0-9]+$ ]] || (( available < MIN_DISK_KIB )); then
    printf '{"experiment_id":"GRAM_PHASE4_LNDR_N1","status":"blocked","reason":"disk_guard"}\n' > "$STATUS"
    exit 2
fi
free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GPU" | tr -d ' ')
if [[ ! "$free_mib" =~ ^[0-9]+$ ]] || (( free_mib < MIN_FREE_MIB )); then
    printf '{"experiment_id":"GRAM_PHASE4_LNDR_N1","status":"blocked","reason":"gpu_guard"}\n' > "$STATUS"
    exit 3
fi
printf '{"experiment_id":"GRAM_PHASE4_LNDR_N1","status":"running"}\n' > "$STATUS"
CUDA_VISIBLE_DEVICES="$GPU" HF_HOME="$ROOT/.cache/huggingface" \
    TRANSFORMERS_CACHE="$ROOT/.cache/huggingface" \
    timeout --signal=TERM 14400 \
    "$PYTHON" "$ROOT/experiment/phase4/lndr_n1.py" \
    --config "$CONFIG" --output-root "$OUTPUT" 2>&1 | tee "$LOG"
