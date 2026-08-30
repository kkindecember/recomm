#!/usr/bin/env bash
set -u -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
CONFIG=${S16_S3_CONFIG:-experiment/phase16/configs/stage16_s3_gfull_objective_resource_sweep.json}
ATTEMPT_ID=${S16_S3_ATTEMPT_ID:-s16_s3_gfull_resource_a1}
EXPERIMENT_ID=${S16_S3_EXPERIMENT_ID:-GRAM_PHASE16_S3_GFULL_OBJECTIVE_RESOURCE_SWEEP}
OUTPUT_REL=${S16_S3_OUTPUT_REL:-artifacts/phase16/s3_genrecedit/resource_sweep/toys_seed1502_a1}
OUTPUT="$ROOT/$OUTPUT_REL"
STATUS="$OUTPUT/status.json"
LOG="$OUTPUT/run.log"
PROGRESS_FILE="$OUTPUT/progress.json"
TELEMETRY="$OUTPUT/gpu_telemetry.csv"
EXACT_COMMAND="${S16_S3_EXACT_COMMAND:-bash experiment/phase16/run_stage16_s3_gfull_objective_resource_sweep.sh}"
LOG_REL="$OUTPUT_REL/run.log"
SUMMARY_REL="$OUTPUT_REL/summary.json"
MINIMUM_FREE=${S16_S3_MINIMUM_FREE:-12288}
EXPECTED_PEAK=${S16_S3_EXPECTED_PEAK:-8192}
HARD_TIMEOUT=${S16_S3_HARD_TIMEOUT:-360}
SUCCESS_CODE=${S16_S3_SUCCESS_CODE:-PASS_S16_3_GFULL_OBJECTIVE_RESOURCE_SWEEP}
SUCCESS_REASON=${S16_S3_SUCCESS_REASON:-S16-3 faithful resource sweep completed; formal Gate remains pending explicit GPU authorization.}
LINEAR_BLOCK_CODE=${S16_S3_LINEAR_BLOCK_CODE:-RESOURCE_BLOCKED_FAITHFUL_LINEAR_SYSTEM}
LINEAR_BLOCK_REASON=${S16_S3_LINEAR_BLOCK_REASON:-The fixed faithful no-ridge solve/rank/residual contract failed; no resampling, ridge, pseudoinverse, fallback, or retry was used.}
VALID_Z_BLOCK_CODE=${S16_S3_VALID_Z_BLOCK_CODE:-RESOURCE_BLOCKED_FAITHFUL_VALID_Z}
VALID_Z_BLOCK_REASON=${S16_S3_VALID_Z_BLOCK_REASON:-At least one fixed position resource subset produced no valid z; no outcome resampling or retry was used.}
FIXED_GPU=${S16_S3_FIXED_GPU:-}
EXCLUDED_GPUS=${S16_S3_EXCLUDED_GPUS:-0,5,7}
STARTED_AT=$(date -Is)
STAGE=preflight
PROGRESS=0
TOTAL=5
PROGRESS_UNIT=resource_contract_steps
SELECTED_GPU=null
ADMISSION_FREE=0
WORKLOAD_PID=0
TELEMETRY_PID=0
LAST_PROGRESS_AT="$STARTED_AT"
TERMINAL_WRITTEN=false

write_status() {
  local state=$1 code=$2 reason=$3 rc=$4 alive=$5 pending=$6
  local temporary="$STATUS.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"%s","attempt_id":"%s","status":"%s","status_code":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","last_progress_at":"%s","runner_pid":%d,"workload_pid":%d,"process_alive":%s,"physical_gpu":%s,"visible_gpu":%s,"gpu_count":%d,"minimum_free_mib_per_gpu":%d,"admission_free_mib_per_gpu":[%d],"expected_peak_mib_per_gpu":%d,"progress_current":%d,"progress_total":%d,"progress_unit":"%s","hard_timeout_seconds":%d,"workload_rc":%d,"exit_code":%d,"exit_code_pending":%s,"test_read":false,"validation_used":false,"scientific_efficacy_metric_produced":false,"automatic_retry":false,"exact_start_command":"%s","output_dir":"%s","log_path":"%s","summary_path":"%s"}\n' \
    "$EXPERIMENT_ID" "$ATTEMPT_ID" "$state" "$code" "$STAGE" "$reason" "$STARTED_AT" "$(date -Is)" "$LAST_PROGRESS_AT" $$ "$WORKLOAD_PID" "$alive" "$SELECTED_GPU" "$([[ "$SELECTED_GPU" == null ]] && echo null || echo 0)" "$([[ "$SELECTED_GPU" == null ]] && echo 0 || echo 1)" "$MINIMUM_FREE" "$ADMISSION_FREE" "$EXPECTED_PEAK" "$PROGRESS" "$TOTAL" "$PROGRESS_UNIT" "$HARD_TIMEOUT" "$rc" "$rc" "$pending" "$EXACT_COMMAND" "$OUTPUT_REL" "$LOG_REL" "$SUMMARY_REL" > "$temporary"
  mv "$temporary" "$STATUS"
}

terminal_status() {
  local state=$1 code=$2 reason=$3 rc=$4
  STAGE=finished
  LAST_PROGRESS_AT=$(date -Is)
  write_status "$state" "$code" "$reason" "$rc" false false
  TERMINAL_WRITTEN=true
}

stop_telemetry() {
  if (( TELEMETRY_PID <= 0 )); then
    return
  fi
  kill -TERM "$TELEMETRY_PID" 2>/dev/null || true
  for _ in 1 2 3 4 5; do
    if ! kill -0 "$TELEMETRY_PID" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if kill -0 "$TELEMETRY_PID" 2>/dev/null; then
    kill -KILL "$TELEMETRY_PID" 2>/dev/null || true
  fi
  wait "$TELEMETRY_PID" 2>/dev/null || true
  TELEMETRY_PID=0
}

handle_signal() {
  local signal=$1
  if (( WORKLOAD_PID > 0 )) && kill -0 "$WORKLOAD_PID" 2>/dev/null; then
    kill -TERM "$WORKLOAD_PID" 2>/dev/null || true
    wait "$WORKLOAD_PID" 2>/dev/null || true
  fi
  WORKLOAD_PID=0
  stop_telemetry
  terminal_status FAILED INTERRUPTED "Runner received ${signal}; no automatic retry." 143
  exit 143
}

handle_exit() {
  local rc=$?
  stop_telemetry
  if [[ "$TERMINAL_WRITTEN" != true ]]; then
    terminal_status FAILED RUNNER_EXIT "Runner exited before a controlled terminal status; no automatic retry." "$rc"
  fi
}

cd "$ROOT" || exit 2
if [[ -e "$OUTPUT" ]]; then
  echo "Refusing existing S16-3 attempt root for $ATTEMPT_ID; retries require a new attempt directory." >&2
  exit 8
fi
mkdir -p "$OUTPUT"
trap 'handle_signal TERM' TERM
trap 'handle_signal INT' INT
trap 'handle_signal HUP' HUP
trap handle_exit EXIT
write_status RUNNING RUNNING "Running S16-3 syntax and CPU contract preflight." -1 true true

timeout --signal=TERM --kill-after=5 30 \
  "$PYTHON" -m experiment.phase16.protocol.gfull_objective_resource_sweep \
    --config "$CONFIG" \
    --capture-identity-only >> "$LOG" 2>&1
rc=$?
if (( rc != 0 )); then
  terminal_status FAILED IDENTITY_FREEZE_FAILED "S16-3 preflight execution identity could not be frozen; no automatic retry." "$rc"
  exit "$rc"
fi

timeout --signal=TERM --kill-after=5 30 \
  "$PYTHON" -m py_compile \
    experiment/phase16/protocol/genrecedit_faithful.py \
    experiment/phase16/protocol/genrecedit_inspired.py \
    experiment/phase16/protocol/genrecedit_data.py \
    experiment/phase16/protocol/gfull_objective_resource_sweep.py \
    experiment/phase16/protocol/finalize_s3_gfull_resource_sweep.py \
    experiment/phase16/tests/test_genrecedit_faithful.py \
    experiment/phase16/tests/test_genrecedit_inspired.py \
    experiment/phase16/tests/test_genrecedit_data.py \
    experiment/phase16/tests/test_gfull_resource_contract.py >> "$LOG" 2>&1
rc=$?
if (( rc != 0 )); then
  terminal_status FAILED PREFLIGHT_FAILED "S16-3 syntax preflight failed; no automatic retry." "$rc"
  exit "$rc"
fi

timeout --signal=TERM --kill-after=5 30 \
  "$PYTHON" -m unittest discover -s experiment/phase16/tests -p 'test_*.py' -q >> "$LOG" 2>&1
rc=$?
if (( rc != 0 )); then
  terminal_status FAILED PREFLIGHT_FAILED "Stage16 CPU contract tests failed; no automatic retry." "$rc"
  exit "$rc"
fi
PROGRESS=1
LAST_PROGRESS_AT=$(date -Is)
STAGE=gpu_admission
write_status RUNNING RUNNING "Selecting one explicitly eligible GPU for the bounded S16-3 sweep." -1 true true

if [[ ! "$EXCLUDED_GPUS" =~ ^([0-9]+(,[0-9]+)*)?$ ]]; then
  terminal_status FAILED PREFLIGHT_FAILED "S16_S3_EXCLUDED_GPUS must be a comma-separated list of physical GPU indices." 2
  exit 2
fi

BEST_FREE=-1
BEST_UTIL=101
SELECTED=-1
while IFS=',' read -r index free util; do
  index=${index//[[:space:]]/}
  free=${free//[[:space:]]/}
  util=${util//[[:space:]]/}
  if [[ -n "$FIXED_GPU" && "$index" != "$FIXED_GPU" ]]; then
    continue
  fi
  if [[ ",$EXCLUDED_GPUS," == *",$index,"* ]]; then
    continue
  fi
  if (( free >= MINIMUM_FREE )) && (( free > BEST_FREE || (free == BEST_FREE && util < BEST_UTIL) )); then
    SELECTED=$index
    BEST_FREE=$free
    BEST_UTIL=$util
  fi
done < <(timeout --signal=TERM --kill-after=5 10 nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits)

if (( SELECTED < 0 )); then
  terminal_status BLOCKED GPU_ADMISSION_FAILED "The fixed/eligible GPU did not meet the frozen free-memory admission; no process was modified and no retry was attempted." 9
  exit 9
fi
SELECTED_GPU=$SELECTED
ADMISSION_FREE=$BEST_FREE
PROGRESS=2
LAST_PROGRESS_AT=$(date -Is)
STAGE=train_only_context_contract
write_status RUNNING RUNNING "Starting Stage16 train-only context build and objective-complete sweep." -1 true true
printf 'timestamp,index,memory_used_mib,memory_free_mib,utilization_gpu_percent\n' > "$TELEMETRY"
(
  while true; do
    timeout --signal=TERM --kill-after=2 5 nvidia-smi --id="$SELECTED" --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits >> "$TELEMETRY" 2>/dev/null || true
    sleep 30
  done
) &
TELEMETRY_PID=$!
env CUDA_VISIBLE_DEVICES="$SELECTED" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  timeout --signal=TERM --kill-after=10 "$HARD_TIMEOUT" \
  "$PYTHON" -m experiment.phase16.protocol.gfull_objective_resource_sweep \
    --config "$CONFIG" \
    --physical-gpu "$SELECTED" \
    --admission-free-mib "$BEST_FREE" \
    --admission-util-percent "$BEST_UTIL" \
    --worker-hard-timeout-seconds "$HARD_TIMEOUT" \
    --expected-peak-mib "$EXPECTED_PEAK" >> "$LOG" 2>&1 &
WORKLOAD_PID=$!
while kill -0 "$WORKLOAD_PID" 2>/dev/null; do
  sleep 15
  observed_age=""
  if [[ -s "$PROGRESS_FILE" ]]; then
    progress_row=$("$PYTHON" -c 'import datetime,json,sys; p=json.load(open(sys.argv[1])); t=datetime.datetime.fromisoformat(p["updated_at"].replace("Z","+00:00")); age=(datetime.datetime.now(datetime.timezone.utc)-t.astimezone(datetime.timezone.utc)).total_seconds(); print("\t".join(map(str,[p["stage"],p["progress_current"],p["progress_total"],p["progress_unit"],p["updated_at"],int(age)])))' "$PROGRESS_FILE" 2>> "$LOG") || progress_row=""
    if [[ -n "$progress_row" ]]; then
      IFS=$'\t' read -r observed_stage observed_current observed_total observed_unit observed_at observed_age <<< "$progress_row"
      if [[ "$observed_stage" != "$STAGE" || "$observed_current" != "$PROGRESS" || "$observed_total" != "$TOTAL" || "$observed_unit" != "$PROGRESS_UNIT" ]]; then
        LAST_PROGRESS_AT=$observed_at
      fi
      STAGE=$observed_stage
      PROGRESS=$observed_current
      TOTAL=$observed_total
      PROGRESS_UNIT=$observed_unit
    fi
  fi
  if [[ -n "${observed_age:-}" ]] && (( observed_age > 300 )); then
    write_status RUNNING STALE_HEARTBEAT "Worker progress heartbeat is older than five minutes; workload was not killed." -1 true true
  else
    write_status RUNNING RUNNING "S16-3 bounded workload alive; progress is worker-owned." -1 true true
  fi
done
wait "$WORKLOAD_PID"; rc=$?
WORKLOAD_PID=0
stop_telemetry
if (( rc == 124 )); then
  terminal_status TIMEOUT RESOURCE_BLOCKED_BOUNDED_TIMEOUT "S16-3 faithful resource diagnostic exceeded its frozen worker budget; partial diagnostics were retained and no retry was attempted." 124
  exit 124
fi
if (( rc == 9 )); then
  terminal_status BLOCKED GPU_ADMISSION_FAILED "Worker re-admission fell below the frozen minimum before model load; no retry or process modification." 9
  exit 9
fi
if (( rc == 10 )); then
  terminal_status BLOCKED "$LINEAR_BLOCK_CODE" "$LINEAR_BLOCK_REASON" 10
  exit 10
fi
if (( rc == 11 )); then
  terminal_status BLOCKED "$VALID_Z_BLOCK_CODE" "$VALID_Z_BLOCK_REASON" 11
  exit 11
fi
if (( rc != 0 )); then
  terminal_status FAILED FAILED "S16-3 bounded resource sweep exited non-zero; no automatic retry or Gate promotion." "$rc"
  exit "$rc"
fi

PROGRESS=4
TOTAL=5
PROGRESS_UNIT=resource_contract_steps
LAST_PROGRESS_AT=$(date -Is)
STAGE=artifact_contract
write_status RUNNING RUNNING "Finalizing S16-3 resource artifacts without promoting the formal admission Gate." -1 true true
timeout --signal=TERM --kill-after=5 30 \
  "$PYTHON" -m experiment.phase16.protocol.finalize_s3_gfull_resource_sweep --config "$CONFIG" >> "$LOG" 2>&1
rc=$?
if (( rc != 0 )); then
  terminal_status FAILED FAILED "S16-3 resource artifact finalization failed; no automatic retry." "$rc"
  exit "$rc"
fi

PROGRESS=5
terminal_status COMPLETED "$SUCCESS_CODE" "$SUCCESS_REASON" 0
echo "$SUCCESS_CODE"
