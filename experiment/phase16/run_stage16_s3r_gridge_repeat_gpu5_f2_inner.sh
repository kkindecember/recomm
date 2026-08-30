#!/usr/bin/env bash
set -u -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
QUEUE_ROOT_REL=${PHASE16_REPEAT_ROOT_REL:-artifacts/phase16/s3_genrecedit/inspired_ridge/stability/toys_seed1502_gpu5_f2}
QUEUE_ROOT="$ROOT/$QUEUE_ROOT_REL"
QUEUE_STATUS="$QUEUE_ROOT/status.json"
QUEUE_LOG="$QUEUE_ROOT/run.log"
FORMAL_STATUS_REL=${PHASE16_FORMAL_STATUS_REL:-artifacts/phase16/s3_genrecedit/inspired_ridge/admission/toys_seed1502_gpu5_f2/status.json}
FORMAL_STATUS="$ROOT/$FORMAL_STATUS_REL"
BASELINE_GPU_PIDS_REL=${PHASE16_BASELINE_GPU_PIDS_REL:-artifacts/phase16/s3_genrecedit/inspired_ridge/admission/toys_seed1502_gpu5_f2/gpu5_baseline_compute_pids.txt}
BASELINE_GPU_PIDS="$ROOT/$BASELINE_GPU_PIDS_REL"
FORMAL_ATTEMPT_ID=${PHASE16_FORMAL_ATTEMPT_ID:-s16_s3r_gridge_formal_gpu5_f2}
FORMAL_ATTEMPT_LABEL=${PHASE16_FORMAL_ATTEMPT_LABEL:-f2}
REPEAT_ATTEMPT_PREFIX=${PHASE16_REPEAT_ATTEMPT_PREFIX:-s16_s3r_gridge_repeat_gpu5_f2}
REPEAT_PROTOCOL_MODULE=${PHASE16_REPEAT_PROTOCOL_MODULE:-experiment.phase16.protocol.gridge_repeat_queue}
REPEAT_PROTOCOL_PATH=${PHASE16_REPEAT_PROTOCOL_PATH:-experiment/phase16/protocol/gridge_repeat_queue.py}
GPU=5
MINIMUM_FREE=13312
EXPECTED_PEAK=8668
DISK_RESERVATION_MIB=32768
HARD_TIMEOUT=604800
EXACT_COMMAND=${PHASE16_REPEAT_EXACT_COMMAND:-"bash experiment/phase16/run_stage16_s3r_gridge_repeat_gpu5_f2.sh"}
STARTED_AT=$(date -Is)
CYCLE=0
CYCLE_PID=0
CYCLE_STATUS=""
CYCLE_PROGRESS=""
CYCLE_TELEMETRY_PID=0
CYCLE_STARTED_AT="$STARTED_AT"
TERMINAL_WRITTEN=false
YIELD_REQUESTED=false
FORMAL_PARENT_STATUS=UNKNOWN
FORMAL_PARENT_STATUS_CODE=UNKNOWN

read_formal_parent() {
  local row
  row=$("$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); print("\t".join(map(str,[p["status"],p["status_code"],p["attempt_id"],p["process_alive"]])))' "$FORMAL_STATUS") || return 1
  local attempt alive
  IFS=$'\t' read -r FORMAL_PARENT_STATUS FORMAL_PARENT_STATUS_CODE attempt alive <<< "$row"
  [[ "$attempt" == "$FORMAL_ATTEMPT_ID" && "$alive" == False ]] || return 1
  case "$FORMAL_PARENT_STATUS" in
    COMPLETED|FAILED|BLOCKED|TIMEOUT|KILLED_TARGET_LEAKAGE) return 0 ;;
    *) return 1 ;;
  esac
}

write_queue_status() {
  local state=$1 code=$2 stage=$3 reason=$4 alive=$5 rc=$6
  local temporary="$QUEUE_STATUS.tmp.$$"
  mkdir -p "$QUEUE_ROOT"
  printf '{"experiment_id":"GRAM_PHASE16_S3R_GRIDGE_NONPROMOTIONAL_REPEAT","status":"%s","status_code":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%d,"cycle_pid":%d,"process_alive":%s,"physical_gpu":5,"visible_gpu":0,"gpu_count":1,"minimum_free_mib_per_gpu":13312,"expected_peak_mib_per_gpu":8668,"disk_reservation_mib":32768,"hard_timeout_seconds_per_cycle":604800,"current_cycle":%d,"formal_parent_attempt_id":"%s","formal_parent_status":"%s","formal_parent_status_code":"%s","affects_scientific_results":false,"promotion_eligible":false,"formal_inputs_read_only":true,"planned_independent_repetitions":true,"continue_after_cycle_failure_by_user_authorization":true,"normal_experiment_priority":true,"new_gpu5_process_causes_repeat_cycle_yield":true,"baseline_gpu_pid_path":"%s","automatic_retry":false,"test_read":false,"validation_used":false,"exit_code":%d,"exact_start_command":"%s","queue_root":"%s"}\n' \
    "$state" "$code" "$stage" "$reason" "$STARTED_AT" "$(date -Is)" $$ "$CYCLE_PID" "$alive" "$CYCLE" "$FORMAL_ATTEMPT_ID" "$FORMAL_PARENT_STATUS" "$FORMAL_PARENT_STATUS_CODE" "$BASELINE_GPU_PIDS_REL" "$rc" "$EXACT_COMMAND" "$QUEUE_ROOT_REL" > "$temporary"
  mv "$temporary" "$QUEUE_STATUS"
}

read_progress_field() {
  local field=$1 fallback=$2
  if [[ -n "$CYCLE_PROGRESS" && -s "$CYCLE_PROGRESS" ]]; then
    "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2],sys.argv[3]))' "$CYCLE_PROGRESS" "$field" "$fallback" 2>/dev/null || echo "$fallback"
  else
    echo "$fallback"
  fi
}

write_cycle_status() {
  local state=$1 code=$2 reason=$3 alive=$4 rc=$5
  local output_rel="$QUEUE_ROOT_REL/cycle_$(printf '%04d' "$CYCLE")"
  local output="$ROOT/$output_rel" temporary="$CYCLE_STATUS.tmp.$$"
  local stage current total unit last_progress
  mkdir -p "$output"
  stage=$(read_progress_field stage repeat_cycle)
  current=$(read_progress_field progress_current 0)
  total=$(read_progress_field progress_total 1)
  unit=$(read_progress_field progress_unit cycle_steps)
  last_progress=$(read_progress_field updated_at "$CYCLE_STARTED_AT")
  printf '{"experiment_id":"GRAM_PHASE16_S3R_GRIDGE_NONPROMOTIONAL_REPEAT","attempt_id":"%s_c%04d","status":"%s","status_code":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","last_progress_at":"%s","runner_pid":%d,"workload_pid":%d,"process_alive":%s,"physical_gpu":5,"visible_gpu":0,"gpu_count":1,"minimum_free_mib_per_gpu":13312,"expected_peak_mib_per_gpu":8668,"progress_current":%s,"progress_total":%s,"progress_unit":"%s","hard_timeout_seconds":604800,"cycle":%d,"formal_parent_status":"%s","formal_parent_status_code":"%s","affects_scientific_results":false,"promotion_eligible":false,"formal_inputs_read_only":true,"planned_independent_repetition":true,"normal_experiment_priority":true,"automatic_retry":false,"test_read":false,"validation_used":false,"exit_code":%d,"output_dir":"%s","summary_path":"%s/summary.json"}\n' \
    "$REPEAT_ATTEMPT_PREFIX" "$CYCLE" "$state" "$code" "$stage" "$reason" "$CYCLE_STARTED_AT" "$(date -Is)" "$last_progress" $$ "$CYCLE_PID" "$alive" "$current" "$total" "$unit" "$CYCLE" "$FORMAL_PARENT_STATUS" "$FORMAL_PARENT_STATUS_CODE" "$rc" "$output_rel" "$output_rel" > "$temporary"
  mv "$temporary" "$CYCLE_STATUS"
}

gpu5_compute_pids() {
  local uuid
  uuid=$(nvidia-smi --id="$GPU" --query-gpu=uuid --format=csv,noheader | tr -d ' ')
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' -v uuid="$uuid" '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($1==uuid) print $2}'
}

is_baseline_pid() {
  local pid=$1
  [[ -s "$BASELINE_GPU_PIDS" ]] && grep -qx "$pid" "$BASELINE_GPU_PIDS"
}

is_cycle_descendant() {
  local pid=$1 ancestor=$2 current=$1 parent
  (( ancestor > 0 )) || return 1
  while (( current > 1 )); do
    [[ "$current" == "$ancestor" ]] && return 0
    [[ -r "/proc/$current/status" ]] || return 1
    parent=$(awk '/^PPid:/ {print $2}' "/proc/$current/status")
    [[ -n "$parent" && "$parent" != "$current" ]] || return 1
    current=$parent
  done
  return 1
}

foreign_gpu5_pids() {
  local pid
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    if is_baseline_pid "$pid"; then continue; fi
    if is_cycle_descendant "$pid" "$CYCLE_PID"; then continue; fi
    echo "$pid"
  done < <(gpu5_compute_pids)
}

stop_cycle_telemetry() {
  if (( CYCLE_TELEMETRY_PID <= 0 )); then return; fi
  kill -TERM "$CYCLE_TELEMETRY_PID" 2>/dev/null || true
  wait "$CYCLE_TELEMETRY_PID" 2>/dev/null || true
  CYCLE_TELEMETRY_PID=0
}

terminate_own_cycle() {
  if (( CYCLE_PID > 0 )) && kill -0 "$CYCLE_PID" 2>/dev/null; then
    kill -TERM "$CYCLE_PID" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      if ! kill -0 "$CYCLE_PID" 2>/dev/null; then break; fi
      sleep 1
    done
    if kill -0 "$CYCLE_PID" 2>/dev/null; then
      kill -KILL "$CYCLE_PID" 2>/dev/null || true
    fi
    wait "$CYCLE_PID" 2>/dev/null || true
  fi
  CYCLE_PID=0
}

handle_signal() {
  terminate_own_cycle
  stop_cycle_telemetry
  if [[ -n "$CYCLE_STATUS" ]]; then
    write_cycle_status FAILED INTERRUPTED "Repeat cycle was interrupted; no scientific artifact was affected." false 143
  fi
  write_queue_status FAILED INTERRUPTED finished "Repeat queue was interrupted; formal $FORMAL_ATTEMPT_LABEL and normal experiments remain unchanged." false 143
  TERMINAL_WRITTEN=true
  exit 143
}

handle_exit() {
  local rc=$?
  stop_cycle_telemetry
  if [[ "$TERMINAL_WRITTEN" != true ]]; then
    write_queue_status FAILED QUEUE_EXIT finished "Repeat queue exited unexpectedly; formal $FORMAL_ATTEMPT_LABEL remains unchanged." false "$rc"
  fi
}

cd "$ROOT" || exit 2
if [[ -e "$QUEUE_ROOT" ]]; then
  echo "Refusing existing $FORMAL_ATTEMPT_LABEL repeat root." >&2
  exit 8
fi
mkdir -p "$QUEUE_ROOT"
trap handle_signal TERM INT HUP
trap handle_exit EXIT
if ! read_formal_parent; then
  write_queue_status FAILED FORMAL_PARENT_NOT_TERMINAL finished "Repeat queue requires a sealed terminal $FORMAL_ATTEMPT_LABEL status." false 7
  TERMINAL_WRITTEN=true
  exit 7
fi
if [[ ! -s "$BASELINE_GPU_PIDS" ]]; then
  write_queue_status FAILED BASELINE_GPU_PIDS_MISSING finished "Repeat queue cannot distinguish later priority work without the $FORMAL_ATTEMPT_LABEL GPU baseline." false 7
  TERMINAL_WRITTEN=true
  exit 7
fi
write_queue_status RUNNING PREFLIGHT preflight "Verifying isolated repeat code; no cycle has started." true -1
timeout --signal=TERM --kill-after=10 600 "$PYTHON" -m py_compile \
  experiment/phase16/protocol/gridge_formal_admission.py \
  "$REPEAT_PROTOCOL_PATH" >> "$QUEUE_LOG" 2>&1 || {
    write_queue_status FAILED PREFLIGHT_FAILED finished "Repeat queue syntax preflight failed." false 2
    TERMINAL_WRITTEN=true
    exit 2
  }
timeout --signal=TERM --kill-after=10 600 "$PYTHON" -m unittest discover \
  -s experiment/phase16/tests -p 'test_*.py' -q >> "$QUEUE_LOG" 2>&1 || {
    write_queue_status FAILED PREFLIGHT_FAILED finished "Stage16 regression failed before repeats." false 2
    TERMINAL_WRITTEN=true
    exit 2
  }

while true; do
  CYCLE=$((CYCLE + 1))
  while true; do
    available_disk=$(df -Pm "$QUEUE_ROOT" | awk 'NR==2 {print $4}')
    if (( available_disk < DISK_RESERVATION_MIB )); then
      write_queue_status BLOCKED DISK_RESERVATION_BLOCKED waiting_resources "Repeat queue is paused below the 32768 MiB disk reserve; normal artifacts are protected." true -1
      sleep 60
      continue
    fi
    foreign=$(foreign_gpu5_pids | paste -sd, -)
    if [[ -n "$foreign" ]]; then
      write_queue_status RUNNING WAITING_FOR_PRIORITY_GPU5 waiting_resources "New GPU5 process(es) $foreign have priority; repeat compute is paused." true -1
      sleep 60
      continue
    fi
    available_gpu=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    if (( available_gpu < MINIMUM_FREE )); then
      write_queue_status RUNNING WAITING_FOR_GPU5 waiting_resources "Waiting for GPU5 to meet 13312 MiB free; no existing process is modified." true -1
      sleep 60
      continue
    fi
    break
  done

  CONFIG_REL=$("$PYTHON" -m "$REPEAT_PROTOCOL_MODULE" prepare \
    --cycle "$CYCLE" --queue-root "$QUEUE_ROOT_REL" 2>> "$QUEUE_LOG") || {
      write_queue_status RUNNING CYCLE_PREPARE_FAILED cycle_transition "Cycle preparation failed; a new independent cycle will be attempted after 60 seconds." true 3
      sleep 60
      continue
    }
  CYCLE_OUTPUT_REL="$QUEUE_ROOT_REL/cycle_$(printf '%04d' "$CYCLE")"
  CYCLE_OUTPUT="$ROOT/$CYCLE_OUTPUT_REL"
  CYCLE_STATUS="$CYCLE_OUTPUT/status.json"
  CYCLE_PROGRESS="$CYCLE_OUTPUT/progress.json"
  CYCLE_LOG="$CYCLE_OUTPUT/run.log"
  CYCLE_TELEMETRY="$CYCLE_OUTPUT/gpu_telemetry.csv"
  CYCLE_STARTED_AT=$(date -Is)

  timeout --signal=TERM --kill-after=5 60 "$PYTHON" -m \
    experiment.phase16.protocol.gridge_formal_admission \
    --config "$CONFIG_REL" --capture-identity-only >> "$QUEUE_LOG" 2>&1 || {
      write_cycle_status FAILED IDENTITY_FREEZE_FAILED "Repeat identity freeze failed; this cycle is preserved and not promoted." false 3
      write_queue_status RUNNING CYCLE_FAILED_CONTINUING cycle_transition "A repeat cycle failed before compute; the next independent cycle starts after 60 seconds." true 3
      sleep 60
      continue
    }
  write_cycle_status RUNNING RUNNING "Full non-promotional repeat is starting; formal $FORMAL_ATTEMPT_LABEL cannot be changed." true -1
  write_queue_status RUNNING NONPROMOTIONAL_REPEAT_RUNNING full_reexecution "GPU5 is running an isolated repeat cycle and will yield to any new priority process." true -1
  printf 'timestamp,index,memory_used_mib,memory_free_mib,utilization_gpu_percent\n' > "$CYCLE_TELEMETRY"
  (
    while true; do
      timeout --signal=TERM --kill-after=2 5 nvidia-smi --id="$GPU" --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits >> "$CYCLE_TELEMETRY" 2>/dev/null || true
      sleep 60
    done
  ) &
  CYCLE_TELEMETRY_PID=$!
  env CUDA_VISIBLE_DEVICES="$GPU" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    timeout --signal=TERM --kill-after=20 "$HARD_TIMEOUT" \
    "$PYTHON" -m experiment.phase16.protocol.gridge_formal_admission \
      --config "$CONFIG_REL" \
      --physical-gpu "$GPU" \
      --admission-free-mib "$available_gpu" \
      --expected-peak-mib "$EXPECTED_PEAK" >> "$CYCLE_LOG" 2>&1 &
  CYCLE_PID=$!
  YIELD_REQUESTED=false
  while kill -0 "$CYCLE_PID" 2>/dev/null; do
    sleep 30
    foreign=$(foreign_gpu5_pids | paste -sd, -)
    if [[ -n "$foreign" ]]; then
      YIELD_REQUESTED=true
      write_cycle_status RUNNING YIELDING_TO_PRIORITY "New GPU5 process(es) $foreign appeared; terminating only this repeat cycle." true -1
      terminate_own_cycle
      break
    fi
    write_cycle_status RUNNING RUNNING "Full repeat compute is active; it is non-promotional and normal-experiment priority remains armed." true -1
    write_queue_status RUNNING NONPROMOTIONAL_REPEAT_RUNNING full_reexecution "GPU5 is running an isolated repeat cycle; formal $FORMAL_ATTEMPT_LABEL remains immutable." true -1
  done
  if (( CYCLE_PID > 0 )); then
    wait "$CYCLE_PID"; rc=$?
    CYCLE_PID=0
  else
    rc=143
  fi
  stop_cycle_telemetry
  if [[ "$YIELD_REQUESTED" == true ]]; then
    write_cycle_status BLOCKED YIELDED_TO_PRIORITY_GPU5 "Repeat cycle yielded to a newly detected GPU5 process; its partial artifacts cannot affect formal results." false 143
    write_queue_status RUNNING WAITING_FOR_PRIORITY_GPU5 waiting_resources "Repeat queue yielded its own cycle and is waiting for the priority process to leave." true -1
    continue
  fi
  if (( rc != 0 )); then
    code=CYCLE_FAILED
    [[ "$rc" == 124 ]] && code=CYCLE_TIMEOUT
    write_cycle_status FAILED "$code" "Repeat cycle exited non-zero; it remains non-promotional and the next independent cycle will run." false "$rc"
    write_queue_status RUNNING CYCLE_FAILED_CONTINUING cycle_transition "A repeat cycle failed; by explicit user authorization the queue continues with a new isolated cycle after 60 seconds." true "$rc"
    sleep 60
    continue
  fi
  timeout --signal=TERM --kill-after=20 1800 "$PYTHON" -m \
    "$REPEAT_PROTOCOL_MODULE" finalize \
    --config "$CONFIG_REL" >> "$CYCLE_LOG" 2>&1 || {
      write_cycle_status FAILED CYCLE_ARTIFACT_FAILED "Repeat compute finished but its isolated artifact contract failed; no formal artifact changed." false 3
      write_queue_status RUNNING CYCLE_FAILED_CONTINUING cycle_transition "Repeat artifact validation failed; the next independent cycle starts after 60 seconds." true 3
      sleep 60
      continue
    }
  write_cycle_status COMPLETED NONPROMOTIONAL_REPEAT_COMPUTE_COMPLETE "This full repeat completed and is excluded from scientific promotion." false 0
  write_queue_status RUNNING NONPROMOTIONAL_REPEAT_RUNNING cycle_transition "One full repeat completed; preparing the next independent cycle." true -1
done
