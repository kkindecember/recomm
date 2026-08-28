#!/usr/bin/env bash
set -u -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
CONFIG=experiment/phase16/configs/stage16_s2_saux_batch2048_sweep.json
OUTPUT="$ROOT/artifacts/phase16/s2_saux_batch2048_sweep/gpu2_a1"
STATUS="$OUTPUT/status.json"
LOG="$OUTPUT/run.log"
GPU=${1:-2}
MINIMUM_FREE=12288
HARD_TIMEOUT=600
EXACT_COMMAND="bash experiment/phase16/run_stage16_s2_saux_batch2048_sweep.sh $GPU"
STARTED_AT=$(date -Is)
ADMISSION_FREE=0

write_status() {
  local state=$1 code=$2 reason=$3 rc=$4
  local temporary="$STATUS.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE16_S2_SAUX_BATCH2048_MEMORY_SWEEP","attempt_id":"s16_s2_saux_batch2048_gpu2_a1","status":"%s","status_code":"%s","reason":"%s","started_at":"%s","updated_at":"%s","physical_gpu":%d,"visible_gpu":0,"gpu_count":1,"minimum_free_mib_per_gpu":12288,"admission_free_mib_per_gpu":[%d],"hard_timeout_seconds":600,"workload_rc":%d,"exit_code":%d,"test_read":false,"automatic_retry":false,"exact_start_command":"%s","output_dir":"artifacts/phase16/s2_saux_batch2048_sweep/gpu2_a1","log_path":"artifacts/phase16/s2_saux_batch2048_sweep/gpu2_a1/run.log","summary_path":"artifacts/phase16/s2_saux_batch2048_sweep/gpu2_a1/summary.json"}\n' \
    "$state" "$code" "$reason" "$STARTED_AT" "$(date -Is)" "$GPU" "$ADMISSION_FREE" "$rc" "$rc" "$EXACT_COMMAND" > "$temporary"
  mv "$temporary" "$STATUS"
}

cd "$ROOT" || exit 2
if [[ "$GPU" != "2" ]]; then
  echo "This frozen sweep is authorized only for physical GPU 2." >&2
  exit 7
fi
if [[ -e "$OUTPUT/summary.json" ]]; then
  echo "Refusing to overwrite a completed memory sweep." >&2
  exit 8
fi
mkdir -p "$OUTPUT"
ADMISSION_FREE=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
if (( ADMISSION_FREE < MINIMUM_FREE )); then
  write_status failed GPU_ADMISSION_FAILED "GPU 2 free memory is below 12288 MiB; no workload started and no automatic retry." 9
  exit 9
fi

write_status running RUNNING "Running syntax and frozen data-contract preflight." -1
timeout --signal=TERM --kill-after=10 300 \
  "$PYTHON" -m py_compile \
    experiment/phase16/protocol/official_specgr_runtime.py \
    experiment/phase16/protocol/specgr_faithful.py \
    experiment/phase16/protocol/saux_formal_train.py \
    experiment/phase16/protocol/saux_batch2048_sweep.py >> "$LOG" 2>&1
rc=$?
if (( rc != 0 )); then
  write_status failed PREFLIGHT_FAILED "Syntax preflight failed; no automatic retry." "$rc"
  exit "$rc"
fi
timeout --signal=TERM --kill-after=10 300 \
  "$PYTHON" -m unittest discover -s experiment/phase16/tests -p 'test_saux_formal.py' -v >> "$LOG" 2>&1
rc=$?
if (( rc != 0 )); then
  write_status failed PREFLIGHT_FAILED "Frozen data-contract tests failed; no automatic retry." "$rc"
  exit "$rc"
fi

write_status running RUNNING "Official UniSRec batch-2048 one-step memory calibration is running." -1
timeout --signal=TERM --kill-after=15 "$HARD_TIMEOUT" \
  env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" experiment/phase16/protocol/saux_batch2048_sweep.py --config "$CONFIG" >> "$LOG" 2>&1
rc=$?
if (( rc == 124 )); then
  write_status timeout TIMEOUT "Memory sweep exceeded its 600-second hard timeout; no automatic retry." 124
  exit 124
fi
if (( rc != 0 )); then
  write_status failed FAILED "Memory sweep exited non-zero; no automatic retry." "$rc"
  exit "$rc"
fi
write_status completed COMPLETED "PASS_S16_2_SAUX_BATCH2048_MEMORY_SWEEP" 0
