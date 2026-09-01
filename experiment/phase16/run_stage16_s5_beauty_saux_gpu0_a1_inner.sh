#!/usr/bin/env bash
set -u -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HOST_ROOT=$(dirname "$(readlink -f "$ROOT/artifacts")")
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
CONFIG=experiment/phase16/configs/stage16_s5_beauty_saux_gpu0_a1.json
TRAIN_CONFIG=experiment/phase16/configs/stage16_s5_beauty_saux_train_gpu0_a1.json
OUTPUT="$HOST_ROOT/artifacts/phase16/s5_beauty_saux/formal/beauty_seed1502_gpu0_a1"
STATUS="$OUTPUT/status.json"
LOG="$OUTPUT/run.log"
TELEMETRY="$OUTPUT/gpu_telemetry.csv"
TRAIN_PROGRESS="$OUTPUT/training/progress.json"
VALIDATION_PROGRESS="$OUTPUT/validation_progress.json"
GPU=0
MINIMUM_FREE=9216
HARD_TIMEOUT=604800
TRAIN_TIMEOUT=172800
VALIDATION_TIMEOUT=345600
HEARTBEAT=30
STARTED_AT=$(date -Is)
START_EPOCH_SECONDS=$(date +%s)
WORKLOAD_PID=0
TELEMETRY_PID=0
ADMISSION_FREE=0
CURRENT_STAGE=preflight
FINAL_WRITTEN=false

export PYTHONDONTWRITEBYTECODE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

progress_triplet() {
  case "$CURRENT_STAGE" in
    training)
      if [[ -s "$TRAIN_PROGRESS" ]]; then
        "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("global_step",0),d.get("maximum_steps",5100),"optimizer_steps")' "$TRAIN_PROGRESS" 2>/dev/null || echo "0 5100 optimizer_steps"
      else
        echo "0 5100 optimizer_steps"
      fi
      ;;
    validation)
      if [[ -s "$VALIDATION_PROGRESS" ]]; then
        "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("progress_current",0),d.get("progress_total",10655),d.get("progress_unit","validation_events"))' "$VALIDATION_PROGRESS" 2>/dev/null || echo "0 10655 validation_events"
      else
        echo "0 10655 validation_events"
      fi
      ;;
    preflight) echo "0 5 pipeline_steps" ;;
    state_freeze) echo "2 5 pipeline_steps" ;;
    finalization) echo "4 5 pipeline_steps" ;;
    finished) echo "5 5 pipeline_steps" ;;
    *) echo "1 5 pipeline_steps" ;;
  esac
}

last_progress_at() {
  local path=""
  [[ "$CURRENT_STAGE" == training ]] && path="$TRAIN_PROGRESS"
  [[ "$CURRENT_STAGE" == validation ]] && path="$VALIDATION_PROGRESS"
  if [[ -n "$path" && -s "$path" ]]; then
    date -Is -r "$path" 2>/dev/null || date -Is
  else
    date -Is
  fi
}

write_status() {
  local state=$1 code=$2 reason=$3 alive=$4 rc=$5 pending=$6
  local current total unit temporary last_progress
  read -r current total unit < <(progress_triplet)
  last_progress=$(last_progress_at)
  temporary="$STATUS.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE16_S5_BEAUTY_SAUX_FROZEN_TRANSFER","attempt_id":"s16_s5_beauty_saux_gpu0_a1","status":"%s","status_code":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","last_progress_at":"%s","runner_pid":%d,"workload_pid":%d,"process_alive":%s,"physical_gpu":0,"visible_gpu":0,"gpu_count":1,"minimum_free_mib_per_gpu":9216,"admission_free_mib_per_gpu":[%d],"expected_training_peak_reserved_mib":5120,"expected_validation_peak_reserved_mib":4096,"progress_current":%s,"progress_total":%s,"progress_unit":"%s","hard_timeout_seconds":604800,"training_hard_timeout_seconds":172800,"validation_hard_timeout_seconds":345600,"disk_reservation_mib":16384,"workload_rc":%d,"exit_code":%d,"exit_code_pending":%s,"test_read":false,"validation_used_for_state_selection_or_tuning":false,"scientific_efficacy_metric_produced":true,"automatic_retry":false,"existing_processes_modified":false,"exact_start_command":"bash experiment/phase16/run_stage16_s5_beauty_saux_gpu0_a1.sh","tmux_session":"phase16_s5_beauty_saux_gpu0_a1","output_dir":"artifacts/phase16/s5_beauty_saux/formal/beauty_seed1502_gpu0_a1","log_path":"artifacts/phase16/s5_beauty_saux/formal/beauty_seed1502_gpu0_a1/run.log","summary_path":"artifacts/phase16/s5_beauty_saux/formal/beauty_seed1502_gpu0_a1/summary.json","isolated_runtime_root":".runtime/phase16_s5_beauty_saux_gpu0_a1_runtime"}\n' \
    "$state" "$code" "$CURRENT_STAGE" "$reason" "$STARTED_AT" "$(date -Is)" "$last_progress" $$ "$WORKLOAD_PID" "$alive" "$ADMISSION_FREE" "$current" "$total" "$unit" "$rc" "$rc" "$pending" > "$temporary"
  mv "$temporary" "$STATUS"
}

terminate_workload() {
  if (( WORKLOAD_PID > 0 )) && kill -0 "$WORKLOAD_PID" 2>/dev/null; then
    kill -TERM "$WORKLOAD_PID" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
      kill -0 "$WORKLOAD_PID" 2>/dev/null || break
      sleep 1
    done
    kill -KILL "$WORKLOAD_PID" 2>/dev/null || true
    wait "$WORKLOAD_PID" 2>/dev/null || true
  fi
  WORKLOAD_PID=0
}

stop_telemetry() {
  if (( TELEMETRY_PID > 0 )); then
    kill "$TELEMETRY_PID" 2>/dev/null || true
    wait "$TELEMETRY_PID" 2>/dev/null || true
    TELEMETRY_PID=0
  fi
}

handle_signal() {
  terminate_workload
  stop_telemetry
  CURRENT_STAGE=finished
  write_status FAILED INTERRUPTED "S16-5 Beauty runner received a termination signal; no automatic retry." false 143 false
  FINAL_WRITTEN=true
  exit 143
}

handle_exit() {
  local rc=$?
  stop_telemetry
  if [[ "$FINAL_WRITTEN" != true ]]; then
    CURRENT_STAGE=finished
    write_status FAILED RUNNER_EXIT "S16-5 Beauty runner exited unexpectedly; no automatic retry." false "$rc" false
  fi
}

admit_gpu() {
  ADMISSION_FREE=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || true)
  [[ "$ADMISSION_FREE" =~ ^[0-9]+$ ]] && (( ADMISSION_FREE >= MINIMUM_FREE ))
}

start_telemetry() {
  printf 'timestamp,physical_gpu,memory_used_mib,memory_free_mib,utilization_gpu_percent\n' > "$TELEMETRY"
  (
    while true; do
      row=$(nvidia-smi --id="$GPU" --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || true)
      [[ -n "$row" ]] && printf '%s,%d,%s\n' "$(date -Is)" "$GPU" "$row" >> "$TELEMETRY"
      sleep "$HEARTBEAT"
    done
  ) &
  TELEMETRY_PID=$!
}

wait_for_stage() {
  local label=$1 last_current=-1 last_change now current total unit
  last_change=$(date +%s)
  while kill -0 "$WORKLOAD_PID" 2>/dev/null; do
    sleep "$HEARTBEAT"
    now=$(date +%s)
    if (( now - START_EPOCH_SECONDS >= HARD_TIMEOUT )); then
      terminate_workload
      CURRENT_STAGE=finished
      write_status TIMEOUT TIMEOUT "S16-5 Beauty exceeded its seven-day hard timeout; no automatic retry." false 124 false
      FINAL_WRITTEN=true
      exit 124
    fi
    read -r current total unit < <(progress_triplet)
    if [[ "$current" != "$last_current" ]]; then
      last_current=$current
      last_change=$now
      write_status RUNNING RUNNING "$label is progressing." true -1 true
    elif (( now - last_change >= 1800 )); then
      write_status RUNNING STALL_SUSPECTED "$label has no recorded progress for at least 1800 seconds; advisory only." true -1 true
    else
      write_status RUNNING RUNNING "$label heartbeat." true -1 true
    fi
  done
  wait "$WORKLOAD_PID"
  local rc=$?
  WORKLOAD_PID=0
  return "$rc"
}

cd "$ROOT" || exit 2
trap handle_signal TERM INT HUP
trap handle_exit EXIT
if [[ "$ROOT" != *"/.runtime/phase16_s5_beauty_saux_gpu0_a1_runtime" ]]; then
  echo "S16-5 inner runner requires the isolated runtime." >&2
  exit 3
fi
if [[ -e "$OUTPUT" || -L "$OUTPUT" ]]; then
  echo "Refusing existing S16-5 output root." >&2
  exit 8
fi
mkdir -p "$OUTPUT"
write_status RUNNING RUNNING "Running frozen-code, syntax, targeted tests, disk, and GPU0 admission preflight without opening Beauty validation." true -1 true

"$PYTHON" -m experiment.phase16.protocol.prepare_stage16_s5_beauty_gpu0_a1_runtime verify --snapshot-root "$ROOT" >> "$LOG" 2>&1 || {
  CURRENT_STAGE=finished; write_status FAILED RUNTIME_IDENTITY_FAILED "S16-5 isolated runtime verification failed." false 3 false; FINAL_WRITTEN=true; exit 3;
}
"$PYTHON" -m py_compile \
  experiment/phase16/protocol/saux_formal_train.py \
  experiment/phase16/protocol/stage16_s4_toys_validation.py \
  experiment/phase16/protocol/stage16_s5_beauty_saux.py \
  experiment/phase16/protocol/prepare_stage16_s5_beauty_gpu0_a1_runtime.py \
  experiment/phase16/tests/test_stage16_s5_beauty_saux.py >> "$LOG" 2>&1 || {
  CURRENT_STAGE=finished; write_status FAILED PREFLIGHT_FAILED "S16-5 Python syntax preflight failed." false 4 false; FINAL_WRITTEN=true; exit 4;
}
bash -n experiment/phase16/run_stage16_s5_beauty_saux_gpu0_a1_inner.sh >> "$LOG" 2>&1 || {
  CURRENT_STAGE=finished; write_status FAILED PREFLIGHT_FAILED "S16-5 runner syntax preflight failed." false 4 false; FINAL_WRITTEN=true; exit 4;
}
CUDA_VISIBLE_DEVICES="" "$PYTHON" -m unittest \
  experiment.phase16.tests.test_stage16_s5_beauty_saux -v >> "$LOG" 2>&1 || {
  CURRENT_STAGE=finished; write_status FAILED PREFLIGHT_FAILED "S16-5 targeted contract tests failed." false 5 false; FINAL_WRITTEN=true; exit 5;
}
"$PYTHON" -m experiment.phase16.protocol.stage16_s5_beauty_saux check-config --config "$CONFIG" >> "$LOG" 2>&1 || {
  CURRENT_STAGE=finished; write_status FAILED PREFLIGHT_FAILED "S16-5 config contract failed." false 5 false; FINAL_WRITTEN=true; exit 5;
}
AVAILABLE_DISK=$(df -Pm "$HOST_ROOT/artifacts" | awk 'NR==2 {print $4}')
if [[ ! "$AVAILABLE_DISK" =~ ^[0-9]+$ ]] || (( AVAILABLE_DISK < 16384 )); then
  CURRENT_STAGE=finished; write_status BLOCKED DISK_ADMISSION_FAILED "S16-5 requires at least 16384 MiB free disk; no workload started." false 10 false; FINAL_WRITTEN=true; exit 10
fi
if ! admit_gpu; then
  CURRENT_STAGE=finished; write_status BLOCKED GPU_ADMISSION_FAILED "GPU0 has less than 9216 MiB free at training admission; no workload started." false 9 false; FINAL_WRITTEN=true; exit 9
fi
cp "$ROOT/runtime_snapshot_manifest.json" "$OUTPUT/runtime_snapshot_manifest.json"
cp "$ROOT/$CONFIG" "$OUTPUT/config.json"
start_telemetry

CURRENT_STAGE=training
write_status RUNNING RUNNING "Training Beauty domain-local official UniSRec S-AUX state on train-only/internal-dev data." true -1 true
timeout --signal=TERM --kill-after=20 "$TRAIN_TIMEOUT" env CUDA_VISIBLE_DEVICES="$GPU" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  "$PYTHON" experiment/phase16/protocol/saux_formal_train.py --config "$TRAIN_CONFIG" >> "$LOG" 2>&1 &
WORKLOAD_PID=$!
wait_for_stage "Beauty S-AUX train-only state construction"; rc=$?
if (( rc == 124 )); then
  CURRENT_STAGE=finished; write_status TIMEOUT TRAINING_TIMEOUT "Beauty S-AUX training exceeded 48 hours; no automatic retry." false 124 false; FINAL_WRITTEN=true; exit 124
fi
if (( rc != 0 )); then
  CURRENT_STAGE=finished; write_status FAILED TRAINING_FAILED "Beauty S-AUX train-only state construction failed; validation was not opened and no retry was started." false "$rc" false; FINAL_WRITTEN=true; exit "$rc"
fi

CURRENT_STAGE=state_freeze
write_status RUNNING RUNNING "Train-only state is frozen; now verifying and reconstructing Beauty F0/portfolio@2 comparators before S-AUX validation." true -1 true
timeout --signal=TERM --kill-after=20 3600 env CUDA_VISIBLE_DEVICES="" \
  "$PYTHON" -m experiment.phase16.protocol.stage16_s5_beauty_saux freeze-comparators --config "$CONFIG" >> "$LOG" 2>&1
rc=$?
if (( rc != 0 )); then
  CURRENT_STAGE=finished; write_status FAILED COMPARATOR_FREEZE_FAILED "Beauty F0/portfolio@2 comparator identity or state-freeze contract failed; S-AUX validation did not start." false "$rc" false; FINAL_WRITTEN=true; exit "$rc"
fi
if ! admit_gpu; then
  CURRENT_STAGE=finished; write_status BLOCKED GPU_READMISSION_FAILED "GPU0 fell below 9216 MiB before frozen validation; no automatic retry." false 9 false; FINAL_WRITTEN=true; exit 9
fi

CURRENT_STAGE=validation
write_status RUNNING RUNNING "Running frozen S-AUX Beauty validation against F0 and unconditional portfolio@2." true -1 true
timeout --signal=TERM --kill-after=20 "$VALIDATION_TIMEOUT" env CUDA_VISIBLE_DEVICES="$GPU" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  "$PYTHON" -m experiment.phase16.protocol.stage16_s5_beauty_saux validate --config "$CONFIG" >> "$LOG" 2>&1 &
WORKLOAD_PID=$!
wait_for_stage "Frozen Beauty S-AUX validation"; rc=$?
if (( rc == 124 )); then
  CURRENT_STAGE=finished; write_status TIMEOUT VALIDATION_TIMEOUT "Frozen Beauty S-AUX validation exceeded 96 hours; no automatic retry." false 124 false; FINAL_WRITTEN=true; exit 124
fi
if (( rc != 0 )); then
  CURRENT_STAGE=finished; write_status FAILED VALIDATION_FAILED "Frozen Beauty S-AUX validation failed; no automatic retry." false "$rc" false; FINAL_WRITTEN=true; exit "$rc"
fi

CURRENT_STAGE=finalization
write_status RUNNING RUNNING "Finalizing paired bootstrap, exact Holm test, item-level diagnostic, and artifact contract." true -1 true
timeout --signal=TERM --kill-after=20 3600 env CUDA_VISIBLE_DEVICES="" \
  "$PYTHON" -m experiment.phase16.protocol.stage16_s5_beauty_saux finalize --config "$CONFIG" >> "$LOG" 2>&1
rc=$?
if (( rc != 0 )); then
  CURRENT_STAGE=finished; write_status FAILED ARTIFACT_CONTRACT_FAILED "S16-5 finalization failed; no result was promoted and no retry was started." false "$rc" false; FINAL_WRITTEN=true; exit "$rc"
fi

VERDICT=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$OUTPUT/summary.json")
stop_telemetry
CURRENT_STAGE=finished
write_status COMPLETED "$VERDICT" "S16-5 Beauty S-AUX frozen transfer completed; inspect summary for Gate and next decision." false 0 false
FINAL_WRITTEN=true
exit 0
