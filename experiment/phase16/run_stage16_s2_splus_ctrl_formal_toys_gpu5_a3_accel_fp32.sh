#!/usr/bin/env bash
set -u -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
CONFIG=experiment/phase16/configs/stage16_s2_splus_ctrl_formal_toys_gpu5_a3_accel_fp32_overlay.json
OUTPUT_REL=artifacts/phase16/s2_splus_ctrl_formal/toys_seed1502_gpu5_a3_accel_fp32
OUTPUT="$ROOT/$OUTPUT_REL"
RESOLVED_CONFIG="$OUTPUT/resolved_config.json"
STATUS="$OUTPUT/status.json"
PROGRESS="$OUTPUT/progress.json"
LOG="$OUTPUT/run.log"
TELEMETRY="$OUTPUT/gpu_telemetry.csv"
GPU=${1:-5}
ATTEMPT_ID=s16_s2_splus_ctrl_formal_toys_gpu5_a3_accel_fp32
MINIMUM_FREE=28672
PER_ARM_HARD_TIMEOUT=1209600
HEARTBEAT=60
STALL_ADVISORY=3600
DISK_RESERVATION_MIB=8192
EXACT_COMMAND="bash experiment/phase16/run_stage16_s2_splus_ctrl_formal_toys_gpu5_a3_accel_fp32.sh 5"
STARTED_AT=$(date -Is)
CURRENT_ARM=preflight
WORKLOAD_PID=0
ADMISSION_FREE=0

HOLDER_CONTROLLER="$ROOT/tools/gram_ablation_scan.sh"
HOLDER_STATE_ROOT="$ROOT/.runtime/gram_ablation_scan_gpu5"
HOLDER_SESSION=gram_ablation_scan_gpu5
HOLDER_RESERVE_MIB=18263
HOLDER_INITIAL_PID=0
HOLDER_RESTORED_PID=0
HOLDER_RELEASED=false
HOLDER_RESTORED=false

FINAL_STATE=running
FINAL_CODE=RUNNING
FINAL_STAGE=preflight
FINAL_REASON="Accelerated paired formal preflight is running."
FINAL_RC=-1
FINAL_ALIVE=false
FINAL_PENDING=true

read_progress_field() {
  local field=$1 fallback=$2
  if [[ -f "$PROGRESS" ]]; then
    "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2],sys.argv[3]))' "$PROGRESS" "$field" "$fallback" 2>/dev/null || echo "$fallback"
  else
    echo "$fallback"
  fi
}

write_status() {
  local state=$1 code=$2 stage=$3 reason=$4 rc=$5 alive=$6 pending=$7 temporary progress last_progress
  FINAL_STATE=$state; FINAL_CODE=$code; FINAL_STAGE=$stage; FINAL_REASON=$reason
  FINAL_RC=$rc; FINAL_ALIVE=$alive; FINAL_PENDING=$pending
  progress=$(read_progress_field paired_optimizer_step 0)
  last_progress=$(read_progress_field updated_at "$STARTED_AT")
  temporary="$STATUS.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE16_S2_SPLUS_CTRL_FORMAL_TOYS","attempt_id":"%s","status":"%s","status_code":"%s","stage":"%s","current_arm":"%s","reason":"%s","started_at":"%s","updated_at":"%s","last_progress_at":"%s","runner_pid":%d,"workload_pid":%d,"process_alive":%s,"physical_gpu":%d,"visible_gpu":0,"gpu_count":1,"minimum_free_mib_per_gpu":%d,"admission_free_mib_per_gpu":[%d],"expected_peak_reserved_mib_per_gpu":17466,"per_arm_hard_timeout_seconds":1209600,"progress_current":%s,"progress_total":25070,"progress_unit":"paired_optimizer_steps","holder_initial_pid":%d,"holder_reserve_mib":%d,"holder_released":%s,"holder_restored":%s,"holder_restored_pid":%d,"workload_rc":%d,"exit_code":%d,"exit_code_pending":%s,"test_read":false,"validation_used":false,"automatic_retry":false,"exact_start_command":"%s","output_dir":"%s","log_path":"%s/run.log","summary_path":"%s/summary.json"}\n' \
    "$ATTEMPT_ID" "$state" "$code" "$stage" "$CURRENT_ARM" "$reason" "$STARTED_AT" "$(date -Is)" "$last_progress" $$ "$WORKLOAD_PID" "$alive" "$GPU" "$MINIMUM_FREE" "$ADMISSION_FREE" "$progress" "$HOLDER_INITIAL_PID" "$HOLDER_RESERVE_MIB" "$HOLDER_RELEASED" "$HOLDER_RESTORED" "$HOLDER_RESTORED_PID" "$rc" "$rc" "$pending" "$EXACT_COMMAND" "$OUTPUT_REL" "$OUTPUT_REL" "$OUTPUT_REL" > "$temporary"
  mv "$temporary" "$STATUS"
}

validate_holder() {
  local holder_status="$HOLDER_STATE_ROOT/status.json" cmdline actual_reserve pid
  [[ -s "$holder_status" && -s "$HOLDER_STATE_ROOT/gpu.txt" ]] || return 1
  [[ "$(tr -d '[:space:]' < "$HOLDER_STATE_ROOT/gpu.txt")" == "$GPU" ]] || return 1
  read -r pid actual_reserve < <(
    "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d.get("state")=="running"; print(int(d["pid"]),int(d["reserve_mib"]))' "$holder_status" 2>/dev/null
  ) || return 1
  [[ "$actual_reserve" == "$HOLDER_RESERVE_MIB" && -r "/proc/$pid/cmdline" ]] || return 1
  cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline")
  [[ "$cmdline" == *"tools/gram_ablation_scan_worker.py"* && "$cmdline" == *"--reserve-mib $HOLDER_RESERVE_MIB"* ]] || return 1
  tmux has-session -t "$HOLDER_SESSION" 2>/dev/null || return 1
  HOLDER_RESTORED_PID=$pid
}

mark_stale_holder_status() {
  local holder_status="$HOLDER_STATE_ROOT/status.json"
  if [[ -f "$holder_status" ]]; then
    "$PYTHON" -c 'import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); d=json.load(p.open()); d["state"]="stopped_before_restore"; q=p.with_name(p.name+".tmp"); q.write_text(json.dumps(d,indent=2)+"\n"); q.replace(p)' "$holder_status"
  fi
}

restore_holder_and_exit() {
  local prior_rc=$? restore_rc=0
  trap - EXIT TERM INT HUP
  if [[ "$HOLDER_RELEASED" == true ]]; then
    if validate_holder; then
      HOLDER_RESTORED=true
    else
      mark_stale_holder_status
      SESSION="$HOLDER_SESSION" STATE_ROOT="$HOLDER_STATE_ROOT" RESERVE_MIB="$HOLDER_RESERVE_MIB" \
        bash "$HOLDER_CONTROLLER" start "$GPU" >> "$LOG" 2>&1 || restore_rc=$?
      if (( restore_rc == 0 )) && validate_holder; then
        HOLDER_RESTORED=true
      else
        HOLDER_RESTORED=false
        FINAL_STATE=blocked; FINAL_CODE=HOLDER_RESTORE_FAILED; FINAL_STAGE=finished
        FINAL_REASON="Formal terminal state reached but the required GPU5 holder could not be restored."
        FINAL_RC=17; FINAL_ALIVE=false; FINAL_PENDING=false
      fi
    fi
  else
    if validate_holder; then HOLDER_RESTORED=true; fi
  fi
  WORKLOAD_PID=0
  write_status "$FINAL_STATE" "$FINAL_CODE" "$FINAL_STAGE" "$FINAL_REASON" "$FINAL_RC" "$FINAL_ALIVE" "$FINAL_PENDING"
  if [[ "$HOLDER_RELEASED" == true && "$HOLDER_RESTORED" != true ]]; then exit 17; fi
  exit "$prior_rc"
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
  FINAL_STATE=failed; FINAL_CODE=INTERRUPTED; FINAL_STAGE=finished
  FINAL_REASON="Accelerated formal runner received a termination signal; holder restoration initiated."
  FINAL_RC=143; FINAL_ALIVE=false; FINAL_PENDING=false
  exit 143
}

fail_and_exit() {
  local code=$1 reason=$2 rc=$3
  FINAL_STATE=failed; FINAL_CODE=$code; FINAL_STAGE=finished; FINAL_REASON=$reason
  FINAL_RC=$rc; FINAL_ALIVE=false; FINAL_PENDING=false
  exit "$rc"
}

run_arm() {
  local arm=$1 arm_started last_progress=0 last_change now telemetry progress rc
  CURRENT_ARM=$arm
  arm_started=$(date +%s)
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" experiment/phase16/protocol/splus_formal_train_accel.py \
    --config "$CONFIG" --arm "$arm" >> "$LOG" 2>&1 &
  WORKLOAD_PID=$!
  write_status running RUNNING training "Formal $arm training is running with GPU5 holder fully released." -1 true true
  last_change=$arm_started
  while kill -0 "$WORKLOAD_PID" 2>/dev/null; do
    now=$(date +%s)
    if (( now - arm_started >= PER_ARM_HARD_TIMEOUT )); then
      terminate_workload
      wait "$WORKLOAD_PID" 2>/dev/null || true
      WORKLOAD_PID=0
      FINAL_STATE=timeout; FINAL_CODE=TIMEOUT; FINAL_STAGE=finished
      FINAL_REASON="Formal $arm exceeded its equal 14-day hard timeout; holder restoration initiated."
      FINAL_RC=124; FINAL_ALIVE=false; FINAL_PENDING=false
      return 124
    fi
    telemetry=$(nvidia-smi --id="$GPU" --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
    printf '%s,%d,%s,%s\n' "$(date -Is)" "$GPU" "$arm" "$telemetry" >> "$TELEMETRY"
    progress=$(read_progress_field paired_optimizer_step 0)
    if [[ "$progress" != "$last_progress" ]]; then
      last_progress=$progress; last_change=$now
      write_status running RUNNING training "Formal $arm accelerated training is progressing." -1 true true
    elif (( now - last_change >= STALL_ADVISORY )); then
      write_status running STALL_SUSPECTED training "No optimizer-step change for at least 3600 seconds; advisory only." -1 true true
    else
      write_status running RUNNING training "Formal $arm accelerated training heartbeat; holder remains released." -1 true true
    fi
    sleep "$HEARTBEAT"
  done
  wait "$WORKLOAD_PID"
  rc=$?
  WORKLOAD_PID=0
  if (( rc != 0 )); then
    FINAL_STATE=failed; FINAL_CODE=FAILED; FINAL_STAGE=finished
    FINAL_REASON="Formal $arm workload exited non-zero; no automatic retry; holder restoration initiated."
    FINAL_RC=$rc; FINAL_ALIVE=false; FINAL_PENDING=false
    return "$rc"
  fi
  write_status running RUNNING arm_completed "Formal $arm completed; preparing next paired stage." -1 false true
  return 0
}

trap restore_holder_and_exit EXIT
trap on_signal TERM INT HUP
cd "$ROOT" || exit 2
if [[ "$GPU" != 5 ]]; then fail_and_exit GPU_SCOPE_FAILED "This formal attempt is authorized only for physical GPU5." 7; fi
if [[ -e "$OUTPUT/summary.json" || -e "$OUTPUT/arms/S-PLUS/checkpoints" || -e "$OUTPUT/arms/S-PLUS-CTRL/checkpoints" ]]; then
  fail_and_exit OUTPUT_EXISTS "Refusing to overwrite or implicitly resume an existing formal attempt." 8
fi
mkdir -p "$OUTPUT"
write_status running RUNNING preflight "Running accelerated formal syntax, tests, resource, disk, and holder preflight." -1 false true
timeout --signal=TERM --kill-after=10 300 "$PYTHON" -m py_compile \
  experiment/phase16/protocol/splus_formal_train_accel.py \
  experiment/phase16/protocol/finalize_splus_formal_accel.py \
  experiment/phase16/tests/test_splus_formal_accel.py >> "$LOG" 2>&1 || fail_and_exit PREFLIGHT_FAILED "Accelerated syntax preflight failed." 2
timeout --signal=TERM --kill-after=10 300 "$PYTHON" -m unittest discover -s experiment/phase16/tests -p 'test_*.py' -q >> "$LOG" 2>&1 || fail_and_exit PREFLIGHT_FAILED "Stage16 tests failed." 2
"$PYTHON" experiment/phase16/protocol/splus_formal_train_accel.py --config "$CONFIG" --resolve-only >> "$LOG" 2>&1 || fail_and_exit CONFIG_RESOLUTION_FAILED "Accelerated overlay resolution failed." 2
"$PYTHON" -c 'import hashlib,json,pathlib,sys;c=json.load(open(sys.argv[1])); s=c["inputs"]["accelerated_batch_sweep_summary"]; p=pathlib.Path(s["path"]); assert hashlib.sha256(p.read_bytes()).hexdigest()==s["sha256"]; d=json.load(p.open()); x=d["selected_candidate"]; assert d["verdict"]=="PASS_S16_2_SPLUS_ACCELERATED_BATCH_SWEEP" and x["embedding_microbatch"]==16 and x["generation_microbatch"]==4 and x["gradient_accumulation"]==64 and d["selected_measurement"]["peak_reserved_mib"]<=28672 and d["selected_measurement"]["checkpoint_unchanged"] and not d["test_read"]' "$RESOLVED_CONFIG" >> "$LOG" 2>&1 || fail_and_exit RESOURCE_EVIDENCE_FAILED "Accelerated batch evidence failed." 3
"$PYTHON" -c 'import hashlib,json,pathlib,sys;c=json.load(open(sys.argv[1]));
for name,expected in c["code_freeze"].items(): assert hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()==expected,name' "$RESOLVED_CONFIG" >> "$LOG" 2>&1 || fail_and_exit CODE_FREEZE_FAILED "Accelerated formal code SHA freeze failed." 3
available_disk=$(df -Pm "$OUTPUT" | awk 'NR==2 {print $4}')
if (( available_disk < DISK_RESERVATION_MIB )); then fail_and_exit DISK_ADMISSION_FAILED "Less than 8192 MiB disk is available." 10; fi
if ! validate_holder; then fail_and_exit HOLDER_VALIDATION_FAILED "GPU5 owned holder identity/state validation failed; no process was changed." 13; fi
HOLDER_INITIAL_PID=$HOLDER_RESTORED_PID
SESSION="$HOLDER_SESSION" STATE_ROOT="$HOLDER_STATE_ROOT" bash "$HOLDER_CONTROLLER" stop >> "$LOG" 2>&1 || fail_and_exit HOLDER_RELEASE_FAILED "GPU5 holder stop command failed." 14
if tmux has-session -t "$HOLDER_SESSION" 2>/dev/null || kill -0 "$HOLDER_INITIAL_PID" 2>/dev/null; then
  fail_and_exit HOLDER_RELEASE_FAILED "GPU5 holder remained alive after release command." 14
fi
HOLDER_RELEASED=true
HOLDER_RESTORED=false
HOLDER_RESTORED_PID=0
ADMISSION_FREE=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
if (( ADMISSION_FREE < MINIMUM_FREE )); then fail_and_exit GPU_ADMISSION_FAILED "GPU5 free memory is below 28672 MiB after holder release." 9; fi
printf 'timestamp,physical_gpu,arm,memory_used_mib,memory_free_mib,utilization_gpu_percent\n' > "$TELEMETRY"

run_arm S-PLUS || exit $?
run_arm S-PLUS-CTRL || exit $?
CURRENT_ARM=finalize
write_status running RUNNING finalize "Both accelerated formal arms completed; validating paired artifacts." -1 false true
timeout --signal=TERM --kill-after=10 600 "$PYTHON" experiment/phase16/protocol/finalize_splus_formal_accel.py \
  --config "$RESOLVED_CONFIG" >> "$LOG" 2>&1 || fail_and_exit ARTIFACT_CONTRACT_FAILED "Accelerated paired artifact contract failed." 3
FINAL_STATE=completed; FINAL_CODE=COMPLETED; FINAL_STAGE=finished
FINAL_REASON="PASS_S16_2_SPLUS_FAITHFUL_CONTRACT_ADMISSION with accelerated matched batching; holder restoration initiated."
FINAL_RC=0; FINAL_ALIVE=false; FINAL_PENDING=false
exit 0
