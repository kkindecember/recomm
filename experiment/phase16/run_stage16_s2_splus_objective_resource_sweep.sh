#!/usr/bin/env bash
set -u -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
CONFIG=experiment/phase16/configs/stage16_s2_splus_objective_resource_sweep.json
OUTPUT="$ROOT/artifacts/phase16/s2_splus_objective_resource_sweep/gpu5_a1"
STATUS="$OUTPUT/status.json"
LOG="$OUTPUT/run.log"
GPU=${1:-5}
MINIMUM_FREE=10240
HARD_TIMEOUT=600
STARTED_AT=$(date -Is)
ADMISSION_FREE=0

write_status() {
  local state=$1 code=$2 reason=$3 rc=$4 temporary="$STATUS.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE16_S2_SPLUS_OBJECTIVE_RESOURCE_SWEEP","attempt_id":"s16_s2_splus_resource_gpu5_a1","status":"%s","status_code":"%s","reason":"%s","started_at":"%s","updated_at":"%s","physical_gpu":%d,"visible_gpu":0,"minimum_free_mib":10240,"admission_free_mib":%d,"hard_timeout_seconds":600,"holder_released":false,"workload_rc":%d,"exit_code":%d,"test_read":false,"automatic_retry":false,"exact_start_command":"bash experiment/phase16/run_stage16_s2_splus_objective_resource_sweep.sh 5","output_dir":"artifacts/phase16/s2_splus_objective_resource_sweep/gpu5_a1","summary_path":"artifacts/phase16/s2_splus_objective_resource_sweep/gpu5_a1/summary.json","log_path":"artifacts/phase16/s2_splus_objective_resource_sweep/gpu5_a1/run.log"}\n' "$state" "$code" "$reason" "$STARTED_AT" "$(date -Is)" "$GPU" "$ADMISSION_FREE" "$rc" "$rc" > "$temporary"
  mv "$temporary" "$STATUS"
}

cd "$ROOT" || exit 2
if [[ "$GPU" != "5" ]]; then
  echo "This bounded sweep is frozen for physical GPU 5." >&2
  exit 7
fi
if [[ -e "$OUTPUT/summary.json" ]]; then
  echo "Refusing to overwrite completed S-PLUS resource sweep." >&2
  exit 8
fi
mkdir -p "$OUTPUT"
ADMISSION_FREE=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
if (( ADMISSION_FREE < MINIMUM_FREE )); then
  write_status failed GPU_ADMISSION_FAILED "GPU 5 free memory is below 10240 MiB; holder was not released and no workload started." 9
  exit 9
fi
write_status running RUNNING "Running syntax and Stage16 unit-test preflight; holder remains active." -1
timeout --signal=TERM --kill-after=10 300 "$PYTHON" -m py_compile experiment/phase16/protocol/splus_objective_resource_sweep.py >> "$LOG" 2>&1
rc=$?
if (( rc != 0 )); then write_status failed PREFLIGHT_FAILED "Python syntax preflight failed; no retry." "$rc"; exit "$rc"; fi
timeout --signal=TERM --kill-after=10 300 "$PYTHON" -m unittest discover -s experiment/phase16/tests -p 'test_*.py' -q >> "$LOG" 2>&1
rc=$?
if (( rc != 0 )); then write_status failed PREFLIGHT_FAILED "Stage16 tests failed; no retry." "$rc"; exit "$rc"; fi
write_status running RUNNING "Objective-complete S-PLUS/CTRL resource sweep running; no efficacy metric." -1
timeout --signal=TERM --kill-after=15 "$HARD_TIMEOUT" env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" experiment/phase16/protocol/splus_objective_resource_sweep.py --config "$CONFIG" >> "$LOG" 2>&1
rc=$?
if (( rc == 124 )); then write_status timeout TIMEOUT "Resource sweep exceeded 600-second hard timeout; no retry." 124; exit 124; fi
if (( rc != 0 )); then write_status failed FAILED "Resource sweep exited non-zero; no retry." "$rc"; exit "$rc"; fi
write_status completed COMPLETED "PASS_S16_2_SPLUS_OBJECTIVE_RESOURCE_SWEEP" 0
