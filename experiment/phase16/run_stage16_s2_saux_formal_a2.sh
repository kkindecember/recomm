#!/usr/bin/env bash
set -u -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
CONFIG=experiment/phase16/configs/stage16_s2_saux_formal_toys_a2.json
OUTPUT="$ROOT/artifacts/phase16/s2_saux_formal/toys_seed1502_a2"
STATUS="$OUTPUT/status.json"
PROGRESS="$OUTPUT/progress.json"
LOG="$OUTPUT/run.log"
TELEMETRY="$OUTPUT/gpu_telemetry.csv"
GPU=${1:-5}
MINIMUM_FREE=9216
HARD_TIMEOUT=172800
HEARTBEAT=30
EXACT_COMMAND="bash experiment/phase16/run_stage16_s2_saux_formal_a2.sh $GPU"
STARTED_AT=$(date -Is)
START_EPOCH_SECONDS=$(date +%s)
WORKLOAD_PID=0
ADMISSION_FREE=0

HOLDER_CONTROLLER="$ROOT/tools/gram_ablation_scan.sh"
HOLDER_STATE_ROOT="$ROOT/.runtime/gram_ablation_scan_gpu5"
HOLDER_SESSION=gram_ablation_scan_gpu5
HOLDER_INITIAL_PID=0
HOLDER_RESERVE_MIB=0
HOLDER_RELEASED=false
HOLDER_RESTORED=false
HOLDER_RELEASED_AT=not_released
HOLDER_RESTORED_AT=not_restored
HOLDER_RESTORE_DETAIL=not_required

FINAL_STATE=not_started
FINAL_CODE=NOT_STARTED
FINAL_STAGE=not_started
FINAL_REASON="Not started."
FINAL_RC=-1
FINAL_ALIVE=false
FINAL_PENDING=false

read_progress_field() {
  local field=$1 fallback=$2
  if [[ -f "$PROGRESS" ]]; then
    "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2],sys.argv[3]))' "$PROGRESS" "$field" "$fallback" 2>/dev/null || echo "$fallback"
  else
    echo "$fallback"
  fi
}

write_status() {
  local state=$1 code=$2 stage=$3 reason=$4 rc=$5 alive=$6 pending=$7
  local current last_progress temporary
  FINAL_STATE=$state
  FINAL_CODE=$code
  FINAL_STAGE=$stage
  FINAL_REASON=$reason
  FINAL_RC=$rc
  FINAL_ALIVE=$alive
  FINAL_PENDING=$pending
  current=$(read_progress_field global_step 0)
  last_progress=$(read_progress_field updated_at "$STARTED_AT")
  temporary="$STATUS.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE16_S2_SAUX_FORMAL_TOYS","attempt_id":"s16_s2_saux_toys_a2","status":"%s","status_code":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","last_progress_at":"%s","runner_pid":%d,"workload_pid":%d,"process_alive":%s,"physical_gpu":%d,"visible_gpu":0,"gpu_count":1,"minimum_free_mib_per_gpu":9216,"admission_free_mib_per_gpu":[%d],"expected_peak_reserved_mib_per_gpu":4314,"resource_recalibration_attempt":"s16_s2_saux_batch2048_gpu2_a1","resource_mode":"release_owned_holder_run_then_restore_same_holder","holder_session":"%s","holder_state_root":"%s","holder_initial_pid":%d,"holder_reserve_mib":%d,"holder_released":%s,"holder_released_at":"%s","holder_restored":%s,"holder_restored_at":"%s","holder_restore_detail":"%s","progress_current":%s,"progress_total":4200,"progress_unit":"optimizer_steps","hard_timeout_seconds":172800,"workload_rc":%d,"exit_code":%d,"exit_code_pending":%s,"test_read":false,"automatic_retry":false,"exact_start_command":"%s","output_dir":"artifacts/phase16/s2_saux_formal/toys_seed1502_a2","log_path":"artifacts/phase16/s2_saux_formal/toys_seed1502_a2/run.log","summary_path":"artifacts/phase16/s2_saux_formal/toys_seed1502_a2/summary.json"}\n' \
    "$state" "$code" "$stage" "$reason" "$STARTED_AT" "$(date -Is)" "$last_progress" $$ "$WORKLOAD_PID" "$alive" "$GPU" "$ADMISSION_FREE" \
    "$HOLDER_SESSION" "$HOLDER_STATE_ROOT" "$HOLDER_INITIAL_PID" "$HOLDER_RESERVE_MIB" "$HOLDER_RELEASED" "$HOLDER_RELEASED_AT" "$HOLDER_RESTORED" "$HOLDER_RESTORED_AT" "$HOLDER_RESTORE_DETAIL" \
    "$current" "$rc" "$rc" "$pending" "$EXACT_COMMAND" > "$temporary"
  mv "$temporary" "$STATUS"
}

read_and_validate_holder() {
  local holder_status="$HOLDER_STATE_ROOT/status.json" holder_gpu="$HOLDER_STATE_ROOT/gpu.txt" cmdline=""
  [[ -s "$holder_status" && -s "$holder_gpu" && -x "$HOLDER_CONTROLLER" ]] || return 1
  [[ "$(tr -d '[:space:]' < "$holder_gpu")" == "$GPU" ]] || return 1
  read -r HOLDER_INITIAL_PID HOLDER_RESERVE_MIB < <(
    "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d.get("state")=="running"; print(int(d["pid"]), int(d["reserve_mib"]))' "$holder_status"
  ) || return 1
  [[ "$HOLDER_INITIAL_PID" == "2083287" && "$HOLDER_RESERVE_MIB" == "18263" ]] || return 1
  [[ -r "/proc/$HOLDER_INITIAL_PID/cmdline" ]] || return 1
  cmdline=$(tr '\0' ' ' < "/proc/$HOLDER_INITIAL_PID/cmdline")
  [[ "$cmdline" == *"tools/gram_ablation_scan_worker.py"* \
    && "$cmdline" == *"--state-dir $HOLDER_STATE_ROOT"* \
    && "$cmdline" == *"--reserve-mib $HOLDER_RESERVE_MIB"* ]] || return 1
  tmux has-session -t "$HOLDER_SESSION" 2>/dev/null || return 1
}

release_holder() {
  SESSION="$HOLDER_SESSION" STATE_ROOT="$HOLDER_STATE_ROOT" \
    bash "$HOLDER_CONTROLLER" stop || return 1
  tmux has-session -t "$HOLDER_SESSION" 2>/dev/null && return 1
  [[ ! -e "/proc/$HOLDER_INITIAL_PID" ]] || return 1
  HOLDER_RELEASED=true
  HOLDER_RELEASED_AT=$(date -Is)
  HOLDER_RESTORE_DETAIL=pending_after_experiment
}

restore_holder() {
  local restored_pid=0 restored_reserve=0 restored_used_mib=0 holder_status="$HOLDER_STATE_ROOT/status.json"
  RESERVE_MIB="$HOLDER_RESERVE_MIB" SESSION="$HOLDER_SESSION" STATE_ROOT="$HOLDER_STATE_ROOT" \
    bash "$HOLDER_CONTROLLER" start "$GPU" || return 1
  read -r restored_pid restored_reserve < <(
    "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d.get("state")=="running"; print(int(d["pid"]), int(d["reserve_mib"]))' "$holder_status"
  ) || return 1
  [[ "$restored_pid" =~ ^[1-9][0-9]*$ && "$restored_pid" != "$HOLDER_INITIAL_PID" \
    && "$restored_reserve" == "$HOLDER_RESERVE_MIB" && -r "/proc/$restored_pid/cmdline" ]] || return 1
  restored_used_mib=$(nvidia-smi -i "$GPU" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
    | "$PYTHON" -c 'import sys; pid=sys.argv[1]; rows=[x.strip() for x in sys.stdin if x.strip()]; vals=[int(x.split(",",1)[1].strip()) for x in rows if x.split(",",1)[0].strip()==pid]; print(vals[0] if vals else 0)' "$restored_pid") || return 1
  (( restored_used_mib >= 19000 )) || return 1
  HOLDER_RESTORED=true
  HOLDER_RESTORED_AT=$(date -Is)
  HOLDER_RESTORE_DETAIL="restored_pid_${restored_pid}_used_${restored_used_mib}_mib"
}

restore_holder_on_exit() {
  local shell_rc=$? terminal_state=$FINAL_STATE terminal_code=$FINAL_CODE terminal_stage=$FINAL_STAGE
  local terminal_reason=$FINAL_REASON terminal_rc=$FINAL_RC terminal_alive=$FINAL_ALIVE terminal_pending=$FINAL_PENDING
  trap - EXIT
  if [[ "$HOLDER_RELEASED" == true && "$HOLDER_RESTORED" != true ]]; then
    if restore_holder; then
      write_status "$terminal_state" "$terminal_code" "$terminal_stage" "${terminal_reason} GPU5 holder restored at original reserve size." "$terminal_rc" "$terminal_alive" "$terminal_pending"
    else
      HOLDER_RESTORE_DETAIL=restore_failed_manual_attention_required
      write_status failed HOLDER_RESTORE_FAILED holder_restore_failed "Experiment terminal state was ${terminal_state}; GPU5 holder restoration failed and requires manual attention." 12 false false
      (( shell_rc == 0 )) && shell_rc=12
    fi
  fi
  exit "$shell_rc"
}

terminate_workload() {
  if (( WORKLOAD_PID > 0 )) && kill -0 "$WORKLOAD_PID" 2>/dev/null; then
    kill -TERM "$WORKLOAD_PID" 2>/dev/null || true
    sleep 10
    kill -KILL "$WORKLOAD_PID" 2>/dev/null || true
  fi
}

on_signal() {
  terminate_workload
  write_status failed INTERRUPTED finished "Formal S-AUX a2 runner received a termination signal; no automatic retry." 143 false false
  exit 143
}

trap restore_holder_on_exit EXIT
trap on_signal TERM INT HUP

cd "$ROOT" || exit 2
if [[ "$GPU" != "5" ]]; then
  echo "This frozen attempt is authorized only for physical GPU 5." >&2
  exit 7
fi
if [[ -e "$OUTPUT/summary.json" || -e "$OUTPUT/checkpoints/last_state.pt" ]]; then
  echo "Refusing to overwrite or implicitly resume an existing formal attempt." >&2
  exit 8
fi
mkdir -p "$OUTPUT"

write_status running RUNNING preflight "Running formal S-AUX a2 syntax/tests, recalibration evidence, and owned-holder preflight." -1 true true
timeout --signal=TERM --kill-after=10 300 \
  "$PYTHON" -m py_compile \
    experiment/phase16/protocol/official_specgr_runtime.py \
    experiment/phase16/protocol/specgr_faithful.py \
    experiment/phase16/protocol/saux_formal_train.py \
    experiment/phase16/protocol/finalize_saux_formal.py \
    experiment/phase16/tests/test_saux_formal.py >> "$LOG" 2>&1
rc=$?
if (( rc != 0 )); then
  write_status failed PREFLIGHT_FAILED finished "Formal S-AUX a2 syntax preflight failed; no automatic retry." "$rc" false false
  exit "$rc"
fi
timeout --signal=TERM --kill-after=10 300 \
  "$PYTHON" -m unittest discover -s experiment/phase16/tests -p 'test_saux_formal.py' -v >> "$LOG" 2>&1
rc=$?
if (( rc != 0 )); then
  write_status failed PREFLIGHT_FAILED finished "Formal S-AUX a2 data contract tests failed; no automatic retry." "$rc" false false
  exit "$rc"
fi
"$PYTHON" -c 'import hashlib,json,pathlib,sys; c=json.load(open(sys.argv[1])); p=pathlib.Path(c["resource_recalibration"]["evidence_path"]); h=hashlib.sha256(p.read_bytes()).hexdigest(); assert h == c["resource_recalibration"]["evidence_sha256"]; s=json.load(p.open()); assert s["verdict"] == "PASS_S16_2_SAUX_BATCH2048_MEMORY_SWEEP" and s["recalibration_eligible"] and s["recommended_minimum_free_mib"] == 9216' "$CONFIG" >> "$LOG" 2>&1
rc=$?
if (( rc != 0 )); then
  write_status failed RESOURCE_EVIDENCE_FAILED finished "Formal S-AUX a2 resource recalibration evidence failed; no automatic retry." "$rc" false false
  exit "$rc"
fi
if ! read_and_validate_holder; then
  write_status blocked HOLDER_VALIDATION_FAILED finished "Owned GPU5 holder validation failed; no process was changed." 13 false false
  exit 13
fi
write_status running RUNNING holder_release "Validated owned GPU5 holder; releasing it for formal S-AUX a2." -1 true true
if ! release_holder; then
  write_status blocked HOLDER_RELEASE_FAILED finished "Could not cleanly release the validated GPU5 holder; no workload started." 14 false false
  exit 14
fi
write_status running RUNNING admission "Owned GPU5 holder released; restoration is mandatory on every terminal path." -1 true true
ADMISSION_FREE=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
if (( ADMISSION_FREE < MINIMUM_FREE )); then
  WORKLOAD_PID=0
  write_status failed GPU_ADMISSION_FAILED finished "GPU 5 remained below recalibrated 9216 MiB after holder release; no workload started and no automatic retry." 9 false false
  exit 9
fi

printf 'timestamp,physical_gpu,memory_used_mib,memory_free_mib,utilization_gpu_percent\n' > "$TELEMETRY"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" experiment/phase16/protocol/saux_formal_train.py --config "$CONFIG" >> "$LOG" 2>&1 &
WORKLOAD_PID=$!
write_status running RUNNING training "Official S-AUX formal a2 training on user-selected physical GPU 5." -1 true true
LAST_PROGRESS=0
LAST_CHANGE_SECONDS=$(date +%s)

while kill -0 "$WORKLOAD_PID" 2>/dev/null; do
  now=$(date +%s)
  if (( now - START_EPOCH_SECONDS >= HARD_TIMEOUT )); then
    terminate_workload
    wait "$WORKLOAD_PID" 2>/dev/null || true
    write_status timeout TIMEOUT finished "Formal S-AUX a2 exceeded the 48-hour hard timeout; no automatic retry." 124 false false
    exit 124
  fi
  telemetry=$(nvidia-smi --id="$GPU" --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
  printf '%s,%d,%s\n' "$(date -Is)" "$GPU" "$telemetry" >> "$TELEMETRY"
  CURRENT_PROGRESS=$(read_progress_field global_step 0)
  if [[ "$CURRENT_PROGRESS" != "$LAST_PROGRESS" ]]; then
    LAST_PROGRESS=$CURRENT_PROGRESS
    LAST_CHANGE_SECONDS=$now
    write_status running RUNNING training "Official S-AUX formal a2 training is progressing." -1 true true
  elif (( now - LAST_CHANGE_SECONDS >= 300 )); then
    write_status running STALL_SUSPECTED training "No optimizer-step change for at least 300 seconds; advisory only, workload continues." -1 true true
  else
    write_status running RUNNING training "Official S-AUX formal a2 training heartbeat." -1 true true
  fi
  sleep "$HEARTBEAT"
done

wait "$WORKLOAD_PID"
rc=$?
WORKLOAD_PID=0
if (( rc != 0 )); then
  write_status failed FAILED finished "Formal S-AUX a2 workload exited non-zero; no automatic retry." "$rc" false false
  exit "$rc"
fi
timeout --signal=TERM --kill-after=10 300 \
  "$PYTHON" experiment/phase16/protocol/finalize_saux_formal.py --config "$CONFIG" >> "$LOG" 2>&1
rc=$?
if (( rc != 0 )); then
  write_status failed ARTIFACT_CONTRACT_FAILED finished "Formal S-AUX a2 artifact contract failed; no automatic retry." "$rc" false false
  exit "$rc"
fi
write_status completed COMPLETED finished "PASS_S16_2_SAUX_FAITHFUL_CONTRACT_ADMISSION." 0 false false
