#!/usr/bin/env bash
set -u -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
FORMAL_CONFIG=experiment/phase16/configs/stage16_s3r_gridge_formal_admission_gpu5_f1.json
QUEUE_ROOT_REL=artifacts/phase16/s3_genrecedit/inspired_ridge/stability/toys_seed1502_gpu5
QUEUE_ROOT="$ROOT/$QUEUE_ROOT_REL"
QUEUE_STATUS="$QUEUE_ROOT/status.json"
QUEUE_LOG="$QUEUE_ROOT/run.log"
GPU=5
MINIMUM_FREE=13312
EXPECTED_PEAK=8668
DISK_RESERVATION_MIB=32768
HARD_TIMEOUT=604800
EXACT_COMMAND="bash experiment/phase16/run_stage16_s3r_gridge_stability_gpu5.sh"
STARTED_AT=$(date -Is)
CYCLE=0
CYCLE_PID=0
CYCLE_STATUS=""
CYCLE_PROGRESS=""
CYCLE_TELEMETRY_PID=0
TERMINAL_WRITTEN=false

write_queue_status() {
  local state=$1 code=$2 stage=$3 reason=$4 alive=$5 rc=$6
  local temporary="$QUEUE_STATUS.tmp.$$"
  mkdir -p "$QUEUE_ROOT"
  printf '{"experiment_id":"GRAM_PHASE16_S3R_GRIDGE_STABILITY","status":"%s","status_code":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%d,"cycle_pid":%d,"process_alive":%s,"physical_gpu":5,"visible_gpu":0,"gpu_count":1,"minimum_free_mib_per_gpu":13312,"expected_peak_mib_per_gpu":8668,"disk_reservation_mib":32768,"hard_timeout_seconds_per_cycle":604800,"current_cycle":%d,"authoritative_stage_status":"COMPLETED","authoritative_status_code":"PASS_S16_3R_GRIDGE_CONTRACT_ADMISSION","authoritative_summary_path":"artifacts/phase16/s3_genrecedit/inspired_ridge/admission/toys_seed1502_gpu5_f1/summary.json","affects_scientific_results":false,"promotion_eligible":false,"formal_inputs_read_only":true,"compute_mode":"planned full G-RIDGE reexecution cycles","automatic_retry":false,"stop_on_cycle_failure":true,"test_read":false,"validation_used":false,"exit_code":%d,"exact_start_command":"%s","queue_root":"%s"}\n' \
    "$state" "$code" "$stage" "$reason" "$STARTED_AT" "$(date -Is)" $$ "$CYCLE_PID" "$alive" "$CYCLE" "$rc" "$EXACT_COMMAND" "$QUEUE_ROOT_REL" > "$temporary"
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
  stage=$(read_progress_field stage stability_cycle)
  current=$(read_progress_field progress_current 0)
  total=$(read_progress_field progress_total 1)
  unit=$(read_progress_field progress_unit cycle_steps)
  last_progress=$(read_progress_field updated_at "$STARTED_AT")
  printf '{"experiment_id":"GRAM_PHASE16_S3R_GRIDGE_STABILITY","attempt_id":"s16_s3r_gridge_stability_gpu5_c%04d","status":"%s","status_code":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","last_progress_at":"%s","runner_pid":%d,"workload_pid":%d,"process_alive":%s,"physical_gpu":5,"visible_gpu":0,"gpu_count":1,"minimum_free_mib_per_gpu":13312,"expected_peak_mib_per_gpu":8668,"progress_current":%s,"progress_total":%s,"progress_unit":"%s","hard_timeout_seconds":604800,"cycle":%d,"authoritative_stage_status":"COMPLETED","authoritative_status_code":"PASS_S16_3R_GRIDGE_CONTRACT_ADMISSION","affects_scientific_results":false,"promotion_eligible":false,"formal_inputs_read_only":true,"automatic_retry":false,"planned_repeat_queue":true,"test_read":false,"validation_used":false,"exit_code":%d,"output_dir":"%s","summary_path":"%s/summary.json"}\n' \
    "$CYCLE" "$state" "$code" "$stage" "$reason" "$STARTED_AT" "$(date -Is)" "$last_progress" $$ "$CYCLE_PID" "$alive" "$current" "$total" "$unit" "$CYCLE" "$rc" "$output_rel" "$output_rel" > "$temporary"
  mv "$temporary" "$CYCLE_STATUS"
}

stop_cycle_telemetry() {
  if (( CYCLE_TELEMETRY_PID <= 0 )); then return; fi
  kill -TERM "$CYCLE_TELEMETRY_PID" 2>/dev/null || true
  wait "$CYCLE_TELEMETRY_PID" 2>/dev/null || true
  CYCLE_TELEMETRY_PID=0
}

handle_signal() {
  if (( CYCLE_PID > 0 )) && kill -0 "$CYCLE_PID" 2>/dev/null; then
    kill -TERM "$CYCLE_PID" 2>/dev/null || true
    sleep 10
    kill -KILL "$CYCLE_PID" 2>/dev/null || true
    wait "$CYCLE_PID" 2>/dev/null || true
  fi
  CYCLE_PID=0
  stop_cycle_telemetry
  if [[ -n "$CYCLE_STATUS" ]]; then
    write_cycle_status FAILED INTERRUPTED "Stability cycle was interrupted; no retry was started." false 143
  fi
  write_queue_status FAILED INTERRUPTED finished "Stability queue was interrupted; authoritative S16-3 artifacts remain unchanged." false 143
  TERMINAL_WRITTEN=true
  exit 143
}

handle_exit() {
  local rc=$?
  stop_cycle_telemetry
  if [[ "$TERMINAL_WRITTEN" != true ]]; then
    write_queue_status FAILED QUEUE_EXIT finished "Stability queue exited unexpectedly; authoritative S16-3 artifacts remain unchanged." false "$rc"
  fi
}

cd "$ROOT" || exit 2
if [[ -e "$QUEUE_ROOT" ]]; then
  echo "Refusing existing stability queue root." >&2
  exit 8
fi
mkdir -p "$QUEUE_ROOT"
trap handle_signal TERM INT HUP
trap handle_exit EXIT
write_queue_status RUNNING PREFLIGHT preflight "Verifying authoritative completion and stability queue code; no cycle has started." true -1
timeout --signal=TERM --kill-after=10 600 "$PYTHON" -m py_compile \
  experiment/phase16/protocol/gridge_formal_admission.py \
  experiment/phase16/protocol/gridge_stability_queue.py >> "$QUEUE_LOG" 2>&1 || {
    write_queue_status FAILED PREFLIGHT_FAILED finished "Stability queue syntax preflight failed; no cycle started." false 2
    TERMINAL_WRITTEN=true
    exit 2
  }
timeout --signal=TERM --kill-after=10 600 "$PYTHON" -m unittest discover \
  -s experiment/phase16/tests -p 'test_*.py' -q >> "$QUEUE_LOG" 2>&1 || {
    write_queue_status FAILED PREFLIGHT_FAILED finished "Stage16 regression failed before stability queue; no cycle started." false 2
    TERMINAL_WRITTEN=true
    exit 2
  }

while true; do
  CYCLE=$((CYCLE + 1))
  while true; do
    available_disk=$(df -Pm "$QUEUE_ROOT" | awk 'NR==2 {print $4}')
    if (( available_disk < DISK_RESERVATION_MIB )); then
      write_queue_status BLOCKED DISK_RESERVATION_BLOCKED waiting_resources "Stability queue paused before a new cycle because less than 32768 MiB disk is free; no formal artifact was changed." true -1
      sleep 60
      continue
    fi
    available_gpu=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    if (( available_gpu < MINIMUM_FREE )); then
      write_queue_status RUNNING WAITING_FOR_GPU5 waiting_resources "Waiting for GPU5 to meet the frozen 13312 MiB admission; no existing process is modified." true -1
      sleep 60
      continue
    fi
    break
  done

  CONFIG_REL=$("$PYTHON" -m experiment.phase16.protocol.gridge_stability_queue prepare \
    --cycle "$CYCLE" --queue-root "$QUEUE_ROOT_REL" 2>> "$QUEUE_LOG") || {
      write_queue_status FAILED CYCLE_PREPARE_FAILED finished "Could not prepare the next isolated stability cycle; queue stopped without retry." false 3
      TERMINAL_WRITTEN=true
      exit 3
    }
  CYCLE_OUTPUT_REL="$QUEUE_ROOT_REL/cycle_$(printf '%04d' "$CYCLE")"
  CYCLE_OUTPUT="$ROOT/$CYCLE_OUTPUT_REL"
  CYCLE_STATUS="$CYCLE_OUTPUT/status.json"
  CYCLE_PROGRESS="$CYCLE_OUTPUT/progress.json"
  CYCLE_LOG="$CYCLE_OUTPUT/run.log"
  CYCLE_TELEMETRY="$CYCLE_OUTPUT/gpu_telemetry.csv"

  timeout --signal=TERM --kill-after=5 60 "$PYTHON" -m \
    experiment.phase16.protocol.gridge_formal_admission \
    --config "$CONFIG_REL" --capture-identity-only >> "$QUEUE_LOG" 2>&1 || {
      write_cycle_status FAILED IDENTITY_FREEZE_FAILED "Cycle identity freeze failed; no GPU computation started and no retry was used." false 3
      write_queue_status FAILED CYCLE_FAILED finished "A planned stability cycle failed before compute; queue stopped without retry." false 3
      TERMINAL_WRITTEN=true
      exit 3
    }
  write_cycle_status RUNNING RUNNING "Full stability reexecution is starting; authoritative S16-3 is already completed and unaffected." true -1
  write_queue_status RUNNING PLANNED_STABILITY_QUEUE_RUNNING full_reexecution "Running a planned full G-RIDGE stability cycle on GPU5; authoritative S16-3 remains completed." true -1
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
  while kill -0 "$CYCLE_PID" 2>/dev/null; do
    sleep 30
    write_cycle_status RUNNING RUNNING "Full stability reexecution is actively computing; results cannot alter the authoritative experiment." true -1
    write_queue_status RUNNING PLANNED_STABILITY_QUEUE_RUNNING full_reexecution "GPU5 is executing a full planned stability cycle; this is real model computation, not an idle allocator." true -1
  done
  wait "$CYCLE_PID"; rc=$?
  CYCLE_PID=0
  stop_cycle_telemetry
  if (( rc != 0 )); then
    code=CYCLE_FAILED
    [[ "$rc" == 124 ]] && code=CYCLE_TIMEOUT
    write_cycle_status FAILED "$code" "Stability cycle exited non-zero; it is excluded from authoritative results and was not retried." false "$rc"
    write_queue_status FAILED "$code" finished "A planned stability cycle failed; queue stopped fail-closed and authoritative S16-3 remains completed." false "$rc"
    TERMINAL_WRITTEN=true
    exit "$rc"
  fi
  timeout --signal=TERM --kill-after=20 1800 "$PYTHON" -m \
    experiment.phase16.protocol.gridge_stability_queue finalize \
    --config "$CONFIG_REL" >> "$CYCLE_LOG" 2>&1 || {
      write_cycle_status FAILED CYCLE_ARTIFACT_FAILED "Cycle compute finished but isolated artifact validation failed; queue stopped without retry." false 3
      write_queue_status FAILED CYCLE_ARTIFACT_FAILED finished "Stability cycle artifact validation failed; authoritative S16-3 remains completed." false 3
      TERMINAL_WRITTEN=true
      exit 3
    }
  write_cycle_status COMPLETED STABILITY_CYCLE_COMPUTE_COMPLETE "This full stability cycle finished successfully; it is excluded from scientific promotion and authoritative data." false 0
  write_queue_status RUNNING PLANNED_STABILITY_QUEUE_RUNNING cycle_transition "Completed one full stability cycle; preparing the next independent planned cycle." true -1
done
