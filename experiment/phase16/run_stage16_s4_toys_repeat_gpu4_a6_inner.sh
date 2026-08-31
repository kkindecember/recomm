#!/usr/bin/env bash
set -u -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HOST_ROOT=$(dirname "$(readlink -f "$ROOT/artifacts")")
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
CONFIG=experiment/phase16/configs/stage16_s4_toys_standalone_gpu4_a6.json
VALIDATOR_MODULE=experiment.phase16.protocol.stage16_s4_toys_validation
FORMAL_STATUS="$HOST_ROOT/artifacts/phase16/s4_toys_standalone/formal/toys_seed1502_gpu4_a6/status.json"
STATUS_ROOT="$HOST_ROOT/.runtime/phase16_s4_gpu4_repeat"
STATUS="$STATUS_ROOT/status.json"
GPU=4
MINIMUM_FREE=19000
PER_ARM_TIMEOUT=172800
HEARTBEAT=30
STARTED_AT=$(date -Is)
WORKLOAD_PID=0
ATTEMPT=0
COMPLETED_CYCLES=0
CURRENT_ARM=none
BASELINE_PIDS=""
TERMINAL_WRITTEN=false
YIELD_REQUESTED=false

export PYTHONDONTWRITEBYTECODE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

gpu4_compute_pids() {
  local uuid
  uuid=$(nvidia-smi --id="$GPU" --query-gpu=uuid --format=csv,noheader | tr -d ' ')
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' -v uuid="$uuid" '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($1==uuid) print $2}'
}

is_baseline_pid() {
  local pid=$1
  [[ " $BASELINE_PIDS " == *" $pid "* ]]
}

is_own_descendant() {
  local pid=$1 current=$1 parent
  (( WORKLOAD_PID > 0 )) || return 1
  while (( current > 1 )); do
    [[ "$current" == "$WORKLOAD_PID" ]] && return 0
    [[ -r "/proc/$current/status" ]] || return 1
    parent=$(awk '/^PPid:/ {print $2}' "/proc/$current/status")
    [[ -n "$parent" && "$parent" != "$current" ]] || return 1
    current=$parent
  done
  return 1
}

foreign_new_pids() {
  local pid
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    if is_own_descendant "$pid"; then continue; fi
    if is_baseline_pid "$pid"; then continue; fi
    echo "$pid"
  done < <(gpu4_compute_pids)
}

write_status() {
  local state=$1 code=$2 stage=$3 reason=$4 alive=$5 rc=$6
  local temporary="$STATUS.tmp.$$"
  mkdir -p "$STATUS_ROOT"
  printf '{"experiment_id":"GRAM_PHASE16_S4_GPU4_DISCARD_ONLY_REPEAT","status":"%s","status_code":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%d,"workload_pid":%d,"process_alive":%s,"physical_gpu":4,"visible_gpu":0,"minimum_free_mib":19000,"memory_only_admission":true,"attempt":%d,"completed_cycles":%d,"current_arm":"%s","discard_output":true,"repeat_artifacts_saved":false,"formal_output_read_only":true,"affects_scientific_results":false,"promotion_eligible":false,"planned_independent_repetitions":true,"automatic_retry_of_formal":false,"yield_to_new_gpu4_process":true,"existing_processes_modified":false,"test_read":false,"exit_code":%d}\n' \
    "$state" "$code" "$stage" "$reason" "$STARTED_AT" "$(date -Is)" $$ "$WORKLOAD_PID" "$alive" "$ATTEMPT" "$COMPLETED_CYCLES" "$CURRENT_ARM" "$rc" > "$temporary"
  mv "$temporary" "$STATUS"
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
  write_status FAILED INTERRUPTED finished "Discard-only repeat queue was interrupted; formal S16-4 data is unchanged." false 143
  TERMINAL_WRITTEN=true
  exit 143
}

handle_exit() {
  local rc=$?
  if [[ "$TERMINAL_WRITTEN" != true ]]; then
    write_status FAILED QUEUE_EXIT finished "Discard-only repeat queue exited unexpectedly; formal S16-4 data is unchanged." false "$rc"
  fi
}

cd "$ROOT" || exit 2
mkdir -p "$STATUS_ROOT"
trap handle_signal TERM INT HUP
trap handle_exit EXIT

if ! "$PYTHON" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["status"]=="COMPLETED" and p["status_code"]=="COMPLETED_S16_4_TOYS_STANDALONE_FROZEN_VALIDATION" and p["process_alive"] is False' "$FORMAL_STATUS" >/dev/null 2>&1; then
  write_status FAILED FORMAL_PARENT_NOT_COMPLETED finished "Repeat queue requires a sealed COMPLETED S16-4 formal parent." false 7
  TERMINAL_WRITTEN=true
  exit 7
fi
if ! "$PYTHON" -m experiment.phase16.protocol.prepare_stage16_s4_gpu4_a6_runtime verify \
  --snapshot-root "$ROOT" >/dev/null 2>&1; then
  write_status FAILED RUNTIME_IDENTITY_FAILED finished "Repeat queue isolated runtime verification failed." false 3
  TERMINAL_WRITTEN=true
  exit 3
fi

BASELINE_PIDS=$(gpu4_compute_pids | paste -sd' ' -)
write_status RUNNING WAITING_GPU4_MEMORY waiting_resources "Discard-only queue waits only for GPU4 free memory; it never writes predictions, metrics, checkpoints, or logs." true -1

while true; do
  while true; do
    foreign=$(foreign_new_pids | paste -sd, -)
    free=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') || free=0
    if [[ -z "$foreign" && "$free" =~ ^[0-9]+$ ]] && (( free >= MINIMUM_FREE )); then
      break
    fi
    write_status RUNNING WAITING_GPU4_MEMORY waiting_resources "Waiting only for GPU4 free memory or a new normal task to leave; utilization is ignored and no external PID is signaled." true -1
    sleep 5
  done

  ATTEMPT=$((ATTEMPT + 1))
  YIELD_REQUESTED=false
  for arm in S-AUX S-PLUS-CTRL S-PLUS G-RIDGE; do
    CURRENT_ARM=$arm
    write_status RUNNING DISCARD_ONLY_RUNNING repeat_compute "Running discard-only $arm occupancy compute; all stdout and computed results are discarded." true -1
    env CUDA_VISIBLE_DEVICES="$GPU" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      timeout --signal=TERM --kill-after=20 "$PER_ARM_TIMEOUT" \
      "$PYTHON" -m "$VALIDATOR_MODULE" --config "$CONFIG" --arm "$arm" --discard-output \
      >/dev/null 2>&1 &
    WORKLOAD_PID=$!
    while kill -0 "$WORKLOAD_PID" 2>/dev/null; do
      sleep "$HEARTBEAT"
      foreign=$(foreign_new_pids | paste -sd, -)
      if [[ -n "$foreign" ]]; then
        YIELD_REQUESTED=true
        write_status RUNNING YIELDING_TO_NORMAL_WORK yielding "A new normal GPU4 task appeared; terminating only this discard workload." true -1
        terminate_own_workload
        break
      fi
      write_status RUNNING DISCARD_ONLY_RUNNING repeat_compute "Discard-only occupancy compute is active; no repeat result is saved." true -1
    done
    if (( WORKLOAD_PID > 0 )); then
      wait "$WORKLOAD_PID"; rc=$?
      WORKLOAD_PID=0
    else
      rc=143
    fi
    if [[ "$YIELD_REQUESTED" == true ]]; then break; fi
    if (( rc != 0 )); then
      write_status FAILED DISCARD_COMPUTE_FAILED finished "A discard-only arm exited nonzero; the queue stopped without retrying it and formal data is unchanged." false "$rc"
      TERMINAL_WRITTEN=true
      exit "$rc"
    fi
  done
  CURRENT_ARM=none
  if [[ "$YIELD_REQUESTED" == true ]]; then
    write_status RUNNING YIELDED_TO_NORMAL_WORK waiting_resources "The discard attempt yielded; waiting to start a fresh independent occupancy attempt." true -1
    continue
  fi
  COMPLETED_CYCLES=$((COMPLETED_CYCLES + 1))
  write_status RUNNING DISCARD_CYCLE_COMPLETED cycle_transition "One independent discard-only cycle completed; no result was saved and the next cycle starts when GPU4 has enough free memory." true 0
done
