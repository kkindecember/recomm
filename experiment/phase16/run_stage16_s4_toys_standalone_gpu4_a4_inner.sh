#!/usr/bin/env bash
set -u -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
CONFIG=experiment/phase16/configs/stage16_s4_toys_standalone_gpu4_a4.json
VALIDATOR_MODULE=experiment.phase16.protocol.stage16_s4_toys_validation
FINALIZER_MODULE=experiment.phase16.protocol.finalize_stage16_s4_toys
OUTPUT_REL=artifacts/phase16/s4_toys_standalone/formal/toys_seed1502_gpu4_a4
OUTPUT="$ROOT/$OUTPUT_REL"
STATUS="$OUTPUT/status.json"
LOG="$OUTPUT/run.log"
TELEMETRY="$OUTPUT/gpu_telemetry.csv"
GPU=4
MINIMUM_FREE=19000
EXPECTED_PEAK=20000
DISK_RESERVATION_MIB=16384
PER_ARM_TIMEOUT=172800
FORMAL_TIMEOUT=604800
HEARTBEAT=30
EXACT_COMMAND="bash experiment/phase16/run_stage16_s4_toys_standalone_gpu4_a4.sh"
ATTEMPT_ID=s16_s4_toys_standalone_gpu4_a4
EXPERIMENT_ID=GRAM_PHASE16_S4_TOYS_STANDALONE_FROZEN_VALIDATION
ISOLATED_RUNTIME_ROOT=.runtime/phase16_s4_toys_gpu4_a4_runtime
REPEAT_SESSION=phase16_s4_toys_repeat_gpu4_a4
REPEAT_INNER=experiment/phase16/run_stage16_s4_toys_repeat_gpu4_a4_inner.sh
HOST_ROOT=$(dirname "$(readlink -f "$ROOT/artifacts")")
REPEAT_STATUS_REL=.runtime/phase16_s4_gpu4_repeat/status.json
REPEAT_STATUS="$HOST_ROOT/$REPEAT_STATUS_REL"
STARTED_AT=$(date -Is)
WORKLOAD_PID=0
TELEMETRY_PID=0
ADMISSION_FREE=0
STAGE=preflight
CURRENT_ARM=none
PROGRESS=0
TOTAL=9
PROGRESS_UNIT=steps
LAST_PROGRESS_AT="$STARTED_AT"
TERMINAL_WRITTEN=false
REPEAT_STARTED=false
LAST_STATE=FAILED
LAST_CODE=RUNNER_EXIT
LAST_REASON="S16-4 GPU4 a4 has not reached a controlled terminal state."
LAST_RC=1
COMPUTE_DEADLINE=0

export PYTHONDONTWRITEBYTECODE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

write_status() {
  local state=$1 code=$2 reason=$3 rc=$4 alive=$5 pending=$6
  local temporary="$STATUS.tmp.$$"
  printf '{"experiment_id":"%s","attempt_id":"%s","status":"%s","status_code":"%s","stage":"%s","current_arm":"%s","reason":"%s","started_at":"%s","updated_at":"%s","last_progress_at":"%s","runner_pid":%d,"workload_pid":%d,"process_alive":%s,"physical_gpu":4,"visible_gpu":0,"gpu_count":1,"minimum_free_mib_per_gpu":19000,"admission_free_mib_per_gpu":[%d],"expected_peak_reserved_mib":20000,"disk_reservation_mib":16384,"progress_current":%d,"progress_total":%d,"progress_unit":"%s","per_arm_hard_timeout_seconds":172800,"formal_hard_timeout_seconds":604800,"workload_rc":%d,"exit_code":%d,"exit_code_pending":%s,"test_read":false,"validation_used_for_state_selection_or_tuning":false,"scientific_efficacy_metric_produced":%s,"automatic_retry":false,"existing_processes_modified":false,"exact_start_command":"%s","output_dir":"%s","log_path":"%s/run.log","summary_path":"%s/summary.json","repeat_started":%s,"repeat_session":"%s","repeat_status_path":"%s","repeat_discard_output":true,"repeat_artifacts_saved":false,"repeat_affects_scientific_results":false,"isolated_runtime_root":"%s"}\n' \
    "$EXPERIMENT_ID" "$ATTEMPT_ID" "$state" "$code" "$STAGE" "$CURRENT_ARM" "$reason" "$STARTED_AT" "$(date -Is)" "$LAST_PROGRESS_AT" $$ "$WORKLOAD_PID" "$alive" "$ADMISSION_FREE" "$PROGRESS" "$TOTAL" "$PROGRESS_UNIT" "$rc" "$rc" "$pending" "$([[ "$state" == COMPLETED ]] && echo true || echo false)" "$EXACT_COMMAND" "$OUTPUT_REL" "$OUTPUT_REL" "$OUTPUT_REL" "$REPEAT_STARTED" "$REPEAT_SESSION" "$REPEAT_STATUS_REL" "$ISOLATED_RUNTIME_ROOT" > "$temporary"
  mv "$temporary" "$STATUS"
}

terminal_status() {
  local state=$1 code=$2 reason=$3 rc=$4
  STAGE=finished
  CURRENT_ARM=none
  LAST_PROGRESS_AT=$(date -Is)
  LAST_STATE=$state
  LAST_CODE=$code
  LAST_REASON=$reason
  LAST_RC=$rc
  write_status "$state" "$code" "$reason" "$rc" false false
  TERMINAL_WRITTEN=true
}

stop_telemetry() {
  if (( TELEMETRY_PID <= 0 )); then return; fi
  kill -TERM "$TELEMETRY_PID" 2>/dev/null || true
  wait "$TELEMETRY_PID" 2>/dev/null || true
  TELEMETRY_PID=0
}

terminate_own_workload() {
  if (( WORKLOAD_PID > 0 )) && kill -0 "$WORKLOAD_PID" 2>/dev/null; then
    kill -TERM "$WORKLOAD_PID" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      if ! kill -0 "$WORKLOAD_PID" 2>/dev/null; then break; fi
      sleep 1
    done
    if kill -0 "$WORKLOAD_PID" 2>/dev/null; then
      kill -KILL "$WORKLOAD_PID" 2>/dev/null || true
    fi
    wait "$WORKLOAD_PID" 2>/dev/null || true
  fi
  WORKLOAD_PID=0
}

handle_signal() {
  terminate_own_workload
  stop_telemetry
  terminal_status FAILED INTERRUPTED "Formal S16-4 GPU4 runner received a signal; partial a4 data remains isolated and no retry was started." 143
  exit 143
}

handle_exit() {
  local rc=$?
  stop_telemetry
  if [[ "$TERMINAL_WRITTEN" != true ]]; then
    terminal_status FAILED RUNNER_EXIT "Formal S16-4 GPU4 runner exited without a controlled terminal state; no retry was started." "$rc"
  fi
}

remaining_timeout() {
  local now remaining
  now=$(date +%s)
  remaining=$((COMPUTE_DEADLINE - now))
  if (( remaining <= 0 )); then
    echo 0
  elif (( remaining < PER_ARM_TIMEOUT )); then
    echo "$remaining"
  else
    echo "$PER_ARM_TIMEOUT"
  fi
}

wait_for_gpu4_memory() {
  STAGE=waiting_gpu4_memory
  while true; do
    ADMISSION_FREE=$(nvidia-smi --id="$GPU" --query-gpu=memory.free \
      --format=csv,noheader,nounits 2>> "$LOG" | tr -d ' ') || ADMISSION_FREE=0
    if [[ "$ADMISSION_FREE" =~ ^[0-9]+$ ]] && (( ADMISSION_FREE >= MINIMUM_FREE )); then
      return 0
    fi
    write_status RUNNING WAITING_GPU4_MEMORY "Waiting only for GPU4 free memory to reach 19000 MiB; utilization is not an admission condition and no existing process is modified." -1 false true
    sleep 5
  done
}

run_gpu_step() {
  local arm=$1 mode=$2 timeout_seconds rc
  wait_for_gpu4_memory
  timeout_seconds=$(remaining_timeout)
  if (( timeout_seconds <= 0 )); then return 124; fi
  CURRENT_ARM=$arm
  LAST_PROGRESS_AT=$(date -Is)
  if [[ "$mode" == smoke ]]; then
    STAGE=bounded_resource_smoke
    write_status RUNNING RUNNING "Running one discard-only bounded resource-smoke event for $arm on GPU4." -1 true true
    env CUDA_VISIBLE_DEVICES="$GPU" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      timeout --signal=TERM --kill-after=20 "$timeout_seconds" \
      "$PYTHON" -m "$VALIDATOR_MODULE" --config "$CONFIG" --arm "$arm" \
        --discard-output --event-limit 1 >> "$LOG" 2>&1 &
  else
    STAGE=formal_arm
    write_status RUNNING RUNNING "Running the write-once formal $arm validation arm on GPU4." -1 true true
    env CUDA_VISIBLE_DEVICES="$GPU" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      timeout --signal=TERM --kill-after=20 "$timeout_seconds" \
      "$PYTHON" -m "$VALIDATOR_MODULE" --config "$CONFIG" --arm "$arm" \
        --output-dir "$OUTPUT/arms/$arm" >> "$LOG" 2>&1 &
  fi
  WORKLOAD_PID=$!
  while kill -0 "$WORKLOAD_PID" 2>/dev/null; do
    sleep "$HEARTBEAT"
    write_status RUNNING RUNNING "S16-4 GPU4 $mode step for $arm is active; existing processes have not been modified." -1 true true
  done
  wait "$WORKLOAD_PID"; rc=$?
  WORKLOAD_PID=0
  return "$rc"
}

cd "$ROOT" || exit 2
if [[ -e "$OUTPUT" || -L "$OUTPUT" ]]; then
  echo "Refusing existing formal S16-4 GPU4 artifact root." >&2
  exit 8
fi
mkdir -p "$OUTPUT"
trap handle_signal TERM INT HUP
trap handle_exit EXIT
write_status RUNNING RUNNING "Verifying isolated runtime and disk before memory-only GPU4 admission." -1 false true

timeout --signal=TERM --kill-after=10 600 \
  "$PYTHON" -m experiment.phase16.protocol.prepare_stage16_s4_gpu4_a4_runtime \
    verify --snapshot-root "$ROOT" >> "$LOG" 2>&1
rc=$?
if (( rc != 0 )); then
  terminal_status FAILED RUNTIME_IDENTITY_FAILED "The immutable S16-4 GPU4 runtime failed verification; no GPU code started." "$rc"
  exit "$rc"
fi
cp --no-clobber "$ROOT/runtime_snapshot_manifest.json" "$OUTPUT/runtime_snapshot_manifest.json"
rc=$?
if (( rc == 0 )); then cp --no-clobber "$ROOT/$CONFIG" "$OUTPUT/config.json"; rc=$?; fi
if (( rc != 0 )); then
  terminal_status FAILED RUNTIME_IDENTITY_FAILED "The S16-4 execution identity could not be copied into the write-once formal root; no GPU code started." "$rc"
  exit "$rc"
fi
available_disk=$(df -Pm "$OUTPUT" | awk 'NR==2 {print $4}')
if (( available_disk < DISK_RESERVATION_MIB )); then
  terminal_status BLOCKED DISK_ADMISSION_FAILED "S16-4 requires 16384 MiB free disk; no GPU code started." 12
  exit 12
fi

CURRENT_ARM=none
wait_for_gpu4_memory
COMPUTE_DEADLINE=$(( $(date +%s) + FORMAL_TIMEOUT ))
printf 'timestamp,index,memory_used_mib,memory_free_mib,utilization_gpu_percent\n' > "$TELEMETRY"
(
  while true; do
    timeout --signal=TERM --kill-after=2 5 nvidia-smi --id="$GPU" \
      --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits >> "$TELEMETRY" 2>/dev/null || true
    sleep "$HEARTBEAT"
  done
) &
TELEMETRY_PID=$!

for arm in S-AUX S-PLUS-CTRL S-PLUS G-RIDGE; do
  run_gpu_step "$arm" smoke
  rc=$?
  if (( rc != 0 )); then
    code=BOUNDED_SMOKE_FAILED
    (( rc == 124 )) && code=TIMEOUT
    terminal_status FAILED "$code" "Discard-only bounded resource smoke failed for $arm; no formal arm data was created and no retry was started." "$rc"
    exit "$rc"
  fi
  PROGRESS=$((PROGRESS + 1))
done

for arm in S-AUX S-PLUS-CTRL S-PLUS G-RIDGE; do
  run_gpu_step "$arm" formal
  rc=$?
  if (( rc != 0 )); then
    code=FORMAL_ARM_FAILED
    (( rc == 124 )) && code=TIMEOUT
    terminal_status FAILED "$code" "The write-once formal $arm arm failed; partial a4 artifacts are retained and no retry was started." "$rc"
    exit "$rc"
  fi
  PROGRESS=$((PROGRESS + 1))
done

stop_telemetry
STAGE=formal_finalization
CURRENT_ARM=none
LAST_PROGRESS_AT=$(date -Is)
write_status RUNNING RUNNING "All four formal arms completed; validating the paired-bootstrap and artifact contract on CPU." -1 false true
timeout --signal=TERM --kill-after=20 3600 "$PYTHON" -m "$FINALIZER_MODULE" --config "$CONFIG" >> "$LOG" 2>&1
rc=$?
if (( rc != 0 )); then
  terminal_status FAILED ARTIFACT_CONTRACT_FAILED "Formal arms completed but S16-4 finalization failed; no result was promoted and no retry was started." "$rc"
  exit "$rc"
fi
PROGRESS=9
terminal_status COMPLETED COMPLETED_S16_4_TOYS_STANDALONE_FROZEN_VALIDATION "S16-4 Toys standalone validation completed on GPU4; all formal outputs are sealed and repeat compute remains non-promotional." 0

if ! tmux has-session -t "$REPEAT_SESSION" 2>/dev/null; then
  tmux new-session -d -s "$REPEAT_SESSION" "cd '$ROOT'; exec bash '$REPEAT_INNER'"
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    if [[ -s "$REPEAT_STATUS" ]]; then
      REPEAT_STARTED=true
      break
    fi
    sleep 1
  done
fi
if [[ "$REPEAT_STARTED" == true ]]; then
  write_status "$LAST_STATE" "$LAST_CODE" "S16-4 formal outputs are sealed; the GPU4 discard-only repeat queue has started and cannot write scientific artifacts." "$LAST_RC" false false
else
  write_status "$LAST_STATE" "$LAST_CODE" "S16-4 formal outputs are sealed; repeat queue launch was unavailable, without changing the formal result." "$LAST_RC" false false
fi
exit 0
