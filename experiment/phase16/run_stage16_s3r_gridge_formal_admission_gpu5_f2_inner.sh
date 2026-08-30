#!/usr/bin/env bash
set -u -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
CONFIG=${PHASE16_FORMAL_CONFIG:-experiment/phase16/configs/stage16_s3r_gridge_formal_admission_gpu5_f2.json}
ATTEMPT_ID=${PHASE16_FORMAL_ATTEMPT_ID:-s16_s3r_gridge_formal_gpu5_f2}
ATTEMPT_LABEL=${PHASE16_FORMAL_ATTEMPT_LABEL:-f2}
EXPERIMENT_ID=GRAM_PHASE16_S3R_GRIDGE_FORMAL_ADMISSION
OUTPUT_REL=${PHASE16_FORMAL_OUTPUT_REL:-artifacts/phase16/s3_genrecedit/inspired_ridge/admission/toys_seed1502_gpu5_f2}
OUTPUT="$ROOT/$OUTPUT_REL"
STATUS="$OUTPUT/status.json"
PROGRESS_FILE="$OUTPUT/progress.json"
LOG="$OUTPUT/run.log"
TELEMETRY="$OUTPUT/gpu_telemetry.csv"
GPU=5
MINIMUM_FREE=13312
EXPECTED_PEAK=8668
DISK_RESERVATION_MIB=32768
HARD_TIMEOUT=604800
STALL_ADVISORY=3600
EXACT_COMMAND=${PHASE16_FORMAL_EXACT_COMMAND:-"bash experiment/phase16/run_stage16_s3r_gridge_formal_admission_gpu5_f2.sh"}
STARTED_AT=$(date -Is)
WORKLOAD_PID=0
TELEMETRY_PID=0
ADMISSION_FREE=0
STAGE=preflight
PROGRESS=0
TOTAL=1
PROGRESS_UNIT=formal_preflight
LAST_PROGRESS_AT="$STARTED_AT"
TERMINAL_WRITTEN=false
STABILITY_QUEUE_STARTED=false
STABILITY_QUEUE_SESSION=${PHASE16_REPEAT_SESSION:-phase16_s3r_gridge_repeat_gpu5_f2}
STABILITY_QUEUE_STATUS=${PHASE16_REPEAT_STATUS_REL:-artifacts/phase16/s3_genrecedit/inspired_ridge/stability/toys_seed1502_gpu5_f2/status.json}
REPEAT_QUEUE_SCRIPT=${PHASE16_REPEAT_SCRIPT:-experiment/phase16/run_stage16_s3r_gridge_repeat_gpu5_f2.sh}
REPEAT_QUEUE_COMMAND="bash $REPEAT_QUEUE_SCRIPT"
ISOLATED_RUNTIME_ROOT=${PHASE16_ISOLATED_RUNTIME_ROOT:-.runtime/phase16_s3r_gridge_f2_runtime}
REPEAT_QUEUE_LAUNCH_ATTEMPTED=false
LAST_TERMINAL_STATE=FAILED
LAST_TERMINAL_CODE=RUNNER_EXIT
LAST_TERMINAL_REASON="$ATTEMPT_LABEL has not reached a controlled terminal state."
LAST_TERMINAL_RC=1
BASELINE_GPU_PIDS="$OUTPUT/gpu5_baseline_compute_pids.txt"

write_status() {
  local state=$1 code=$2 reason=$3 rc=$4 alive=$5 pending=$6
  local temporary="$STATUS.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"%s","attempt_id":"%s","status":"%s","status_code":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","last_progress_at":"%s","runner_pid":%d,"workload_pid":%d,"process_alive":%s,"physical_gpu":5,"visible_gpu":0,"gpu_count":1,"minimum_free_mib_per_gpu":13312,"admission_free_mib_per_gpu":[%d],"expected_peak_mib_per_gpu":8668,"disk_reservation_mib":32768,"progress_current":%d,"progress_total":%d,"progress_unit":"%s","hard_timeout_seconds":604800,"workload_rc":%d,"exit_code":%d,"exit_code_pending":%s,"test_read":false,"validation_used":false,"scientific_efficacy_metric_produced":false,"automatic_retry":false,"existing_processes_modified":false,"exact_start_command":"%s","output_dir":"%s","log_path":"%s/run.log","summary_path":"%s/summary.json","stability_queue_started":%s,"stability_queue_session":"%s","stability_queue_status_path":"%s","repeat_after_any_terminal":true,"repeat_affects_scientific_results":false,"repeat_promotion_eligible":false,"repeat_normal_experiment_priority":true,"isolated_runtime_root":"%s"}\n' \
    "$EXPERIMENT_ID" "$ATTEMPT_ID" "$state" "$code" "$STAGE" "$reason" "$STARTED_AT" "$(date -Is)" "$LAST_PROGRESS_AT" $$ "$WORKLOAD_PID" "$alive" "$ADMISSION_FREE" "$PROGRESS" "$TOTAL" "$PROGRESS_UNIT" "$rc" "$rc" "$pending" "$EXACT_COMMAND" "$OUTPUT_REL" "$OUTPUT_REL" "$OUTPUT_REL" "$STABILITY_QUEUE_STARTED" "$STABILITY_QUEUE_SESSION" "$STABILITY_QUEUE_STATUS" "$ISOLATED_RUNTIME_ROOT" > "$temporary"
  mv "$temporary" "$STATUS"
}

terminal_status() {
  local state=$1 code=$2 reason=$3 rc=$4
  STAGE=finished
  LAST_PROGRESS_AT=$(date -Is)
  write_status "$state" "$code" "$reason" "$rc" false false
  LAST_TERMINAL_STATE=$state
  LAST_TERMINAL_CODE=$code
  LAST_TERMINAL_REASON=$reason
  LAST_TERMINAL_RC=$rc
  TERMINAL_WRITTEN=true
}

launch_repeat_queue() {
  if [[ "$REPEAT_QUEUE_LAUNCH_ATTEMPTED" == true ]]; then return; fi
  REPEAT_QUEUE_LAUNCH_ATTEMPTED=true
  if bash "$REPEAT_QUEUE_SCRIPT" >> "$LOG" 2>&1; then
    STABILITY_QUEUE_STARTED=true
    write_status "$LAST_TERMINAL_STATE" "$LAST_TERMINAL_CODE" "$LAST_TERMINAL_REASON The isolated non-promotional repeat queue started after the $ATTEMPT_LABEL terminal state; it yields to new GPU5 priority processes." "$LAST_TERMINAL_RC" false false
  else
    write_status "$LAST_TERMINAL_STATE" "$LAST_TERMINAL_CODE" "$LAST_TERMINAL_REASON Repeat queue launch failed; the formal $ATTEMPT_LABEL terminal result is unchanged." "$LAST_TERMINAL_RC" false false
  fi
}

stop_telemetry() {
  if (( TELEMETRY_PID <= 0 )); then return; fi
  kill -TERM "$TELEMETRY_PID" 2>/dev/null || true
  for _ in 1 2 3 4 5; do
    if ! kill -0 "$TELEMETRY_PID" 2>/dev/null; then break; fi
    sleep 1
  done
  if kill -0 "$TELEMETRY_PID" 2>/dev/null; then kill -KILL "$TELEMETRY_PID" 2>/dev/null || true; fi
  wait "$TELEMETRY_PID" 2>/dev/null || true
  TELEMETRY_PID=0
}

terminate_workload() {
  if (( WORKLOAD_PID > 0 )) && kill -0 "$WORKLOAD_PID" 2>/dev/null; then
    kill -TERM "$WORKLOAD_PID" 2>/dev/null || true
    sleep 10
    kill -KILL "$WORKLOAD_PID" 2>/dev/null || true
    wait "$WORKLOAD_PID" 2>/dev/null || true
  fi
  WORKLOAD_PID=0
}

capture_gpu_baseline() {
  local gpu_uuid
  gpu_uuid=$(nvidia-smi --id="$GPU" --query-gpu=uuid --format=csv,noheader | tr -d ' ')
  {
    printf '# GPU5 baseline compute PIDs captured before formal %s; later repeat cycles yield to any new PID.\n' "$ATTEMPT_LABEL"
    nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits \
      | awk -F',' -v uuid="$gpu_uuid" '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($1==uuid) print $2}'
  } > "$BASELINE_GPU_PIDS.tmp.$$"
  mv "$BASELINE_GPU_PIDS.tmp.$$" "$BASELINE_GPU_PIDS"
}

handle_signal() {
  terminate_workload
  stop_telemetry
  terminal_status FAILED INTERRUPTED "Formal S16-3R runner received a signal; partial checkpoints remain isolated and no retry/resume was started." 143
  exit 143
}

handle_exit() {
  local rc=$?
  stop_telemetry
  if [[ "$TERMINAL_WRITTEN" != true ]]; then
    terminal_status FAILED RUNNER_EXIT "Formal S16-3R runner exited without a controlled terminal status; no automatic retry." "$rc"
  fi
  launch_repeat_queue
}

cd "$ROOT" || exit 2
if [[ -e "$OUTPUT" ]]; then
  echo "Refusing existing formal S16-3R artifact root." >&2
  exit 8
fi
mkdir -p "${OUTPUT%/*}"
BOOT_LOG="${OUTPUT%/*}/.formal_gpu5_${ATTEMPT_LABEL}_boot.log.$$"
trap handle_signal TERM INT HUP
trap handle_exit EXIT

# Needed so the user-authorized repeat queue can start after any terminal,
# including identity/preflight failures before the normal GPU admission point.
capture_gpu_baseline

timeout --signal=TERM --kill-after=5 60 \
  "$PYTHON" -m experiment.phase16.protocol.gridge_formal_admission \
    --config "$CONFIG" --capture-identity-only >> "$BOOT_LOG" 2>&1
rc=$?
if (( rc != 0 )); then
  terminal_status FAILED IDENTITY_FREEZE_FAILED "Formal S16-3R execution identity could not be frozen; no GPU workload started." "$rc"
  exit "$rc"
fi
mv "$BOOT_LOG" "$LOG"
write_status RUNNING RUNNING "Running formal S16-3R syntax, CPU regression, resource-parent, disk, and GPU5 admission checks." -1 false true

timeout --signal=TERM --kill-after=10 600 \
  "$PYTHON" -m py_compile \
    experiment/phase16/protocol/genrecedit_faithful.py \
    experiment/phase16/protocol/genrecedit_inspired.py \
    experiment/phase16/protocol/gridge_formal_admission.py \
    experiment/phase16/protocol/finalize_s3r_gridge_formal.py \
    experiment/phase16/tests/test_gridge_formal_admission.py >> "$LOG" 2>&1
rc=$?
if (( rc != 0 )); then
  terminal_status FAILED PREFLIGHT_FAILED "Formal S16-3R syntax preflight failed; no GPU workload started." "$rc"
  exit "$rc"
fi
timeout --signal=TERM --kill-after=10 600 \
  "$PYTHON" -m unittest discover -s experiment/phase16/tests -p 'test_*.py' -q >> "$LOG" 2>&1
rc=$?
if (( rc != 0 )); then
  terminal_status FAILED PREFLIGHT_FAILED "Stage16 CPU regression failed; no GPU workload started." "$rc"
  exit "$rc"
fi
available_disk=$(df -Pm "$OUTPUT" | awk 'NR==2 {print $4}')
if (( available_disk < DISK_RESERVATION_MIB )); then
  terminal_status BLOCKED DISK_ADMISSION_FAILED "Formal S16-3R requires 32768 MiB free disk; no GPU workload started." 12
  exit 12
fi
ADMISSION_FREE=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
if (( ADMISSION_FREE < MINIMUM_FREE )); then
  terminal_status BLOCKED GPU_ADMISSION_FAILED "GPU5 was below the frozen 13312 MiB admission; no process was modified and no retry was attempted." 9
  exit 9
fi

capture_gpu_baseline

STAGE=gpu_launch
PROGRESS=1
TOTAL=1
PROGRESS_UNIT=formal_preflight
LAST_PROGRESS_AT=$(date -Is)
write_status RUNNING RUNNING "Formal S16-3R preflight passed; launching the full 302400-request G-RIDGE workload on GPU5." -1 true true
printf 'timestamp,index,memory_used_mib,memory_free_mib,utilization_gpu_percent\n' > "$TELEMETRY"
(
  while true; do
    timeout --signal=TERM --kill-after=2 5 nvidia-smi --id="$GPU" --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits >> "$TELEMETRY" 2>/dev/null || true
    sleep 60
  done
) &
TELEMETRY_PID=$!

env CUDA_VISIBLE_DEVICES="$GPU" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  timeout --signal=TERM --kill-after=20 "$HARD_TIMEOUT" \
  "$PYTHON" -m experiment.phase16.protocol.gridge_formal_admission \
    --config "$CONFIG" \
    --physical-gpu "$GPU" \
    --admission-free-mib "$ADMISSION_FREE" \
    --expected-peak-mib "$EXPECTED_PEAK" >> "$LOG" 2>&1 &
WORKLOAD_PID=$!

while kill -0 "$WORKLOAD_PID" 2>/dev/null; do
  sleep 30
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
  if [[ -n "${observed_age:-}" ]] && (( observed_age > STALL_ADVISORY )); then
    write_status RUNNING STALL_SUSPECTED "Formal S16-3R progress has not changed for at least one hour; advisory only, workload continues." -1 true true
  else
    write_status RUNNING RUNNING "Formal S16-3R full computation is progressing; this is the authoritative admission attempt, not occupancy work." -1 true true
  fi
done
wait "$WORKLOAD_PID"; rc=$?
WORKLOAD_PID=0
stop_telemetry
if (( rc == 124 )); then
  terminal_status TIMEOUT TIMEOUT "Formal S16-3R exceeded the seven-day hard timeout; partial checkpoints were retained and no retry/resume was started." 124
  exit 124
fi
if (( rc == 9 )); then
  terminal_status BLOCKED GPU_ADMISSION_FAILED "GPU5 worker re-admission fell below 13312 MiB before model load; no process was modified." 9
  exit 9
fi
if (( rc == 10 )); then
  terminal_status BLOCKED FORMAL_BLOCKED_GRIDGE_LINEAR_SYSTEM "At least one full G-RIDGE system failed the frozen condition/Cholesky/residual contract; no fallback or retry was used." 10
  exit 10
fi
if (( rc == 11 )); then
  terminal_status BLOCKED FORMAL_BLOCKED_VALID_Z "At least one full position produced no valid z; no outcome resampling or retry was used." 11
  exit 11
fi
if (( rc != 0 )); then
  terminal_status FAILED FAILED "Formal S16-3R workload exited non-zero; partial evidence remains isolated and no automatic retry was used." "$rc"
  exit "$rc"
fi

STAGE=artifact_contract
PROGRESS=0
TOTAL=1
PROGRESS_UNIT=formal_finalization
LAST_PROGRESS_AT=$(date -Is)
write_status RUNNING RUNNING "Full G-RIDGE computation completed; validating the fail-closed formal artifact contract." -1 false true
timeout --signal=TERM --kill-after=20 1800 \
  "$PYTHON" -m experiment.phase16.protocol.finalize_s3r_gridge_formal \
    --config "$CONFIG" >> "$LOG" 2>&1
rc=$?
if (( rc != 0 )); then
  terminal_status FAILED ARTIFACT_CONTRACT_FAILED "Full computation finished, but formal artifact validation failed; S16-3 Gate was not promoted and no retry was used." "$rc"
  exit "$rc"
fi

PROGRESS=1
TOTAL=1
PROGRESS_UNIT=formal_finalization
FORMAL_REASON="S16-3 authoritative $ATTEMPT_LABEL is finished: all 302400 requests, six FP64 G-RIDGE solves, 7435 item-disjoint events, and 512 warm-preservation pairs completed; the formal contract Gate passed. Admission metrics are non-promotional and validation/test stayed sealed."
terminal_status COMPLETED PASS_S16_3R_GRIDGE_CONTRACT_ADMISSION "$FORMAL_REASON" 0
exit 0
