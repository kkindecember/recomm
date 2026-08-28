#!/usr/bin/env bash
set -u -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
CONFIG=experiment/phase16/configs/stage16_s2_splus_ctrl_formal_toys_gpu7_a4_split_fp32_overlay.json
OUTPUT_REL=artifacts/phase16/s2_splus_ctrl_formal/toys_seed1502_gpu7_a4_ctrl_split_fp32
OUTPUT="$ROOT/$OUTPUT_REL"
RESOLVED_CONFIG="$OUTPUT/resolved_config.json"
STATUS="$OUTPUT/status.json"
PROGRESS="$OUTPUT/progress.json"
LOG="$OUTPUT/run.log"
TELEMETRY="$OUTPUT/gpu_telemetry.csv"
PARENT_OUTPUT_REL=artifacts/phase16/s2_splus_ctrl_formal/toys_seed1502_gpu5_a3_accel_fp32
PARENT_OUTPUT="$ROOT/$PARENT_OUTPUT_REL"
PARENT_RESOLVED="$PARENT_OUTPUT/resolved_config.json"
GPU=${1:-7}
ATTEMPT_ID=s16_s2_splus_ctrl_formal_toys_gpu7_a4_split_fp32
MINIMUM_FREE=28672
HARD_TIMEOUT=1209600
HEARTBEAT=60
STALL_ADVISORY=3600
DISK_RESERVATION_MIB=8192
EXACT_COMMAND="bash experiment/phase16/run_stage16_s2_splus_ctrl_formal_toys_gpu7_a4_split_fp32.sh 7"
STARTED_AT=$(date -Is)
WORKLOAD_PID=0
ADMISSION_FREE=0
FINAL_STATE=running
FINAL_CODE=RUNNING
FINAL_STAGE=preflight
FINAL_REASON="GPU7 isolated S-PLUS-CTRL split preflight is running."
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
  progress=$(read_progress_field arm_optimizer_step 0)
  last_progress=$(read_progress_field updated_at "$STARTED_AT")
  temporary="$STATUS.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE16_S2_SPLUS_CTRL_SPLIT_TOYS","attempt_id":"%s","status":"%s","status_code":"%s","stage":"%s","current_arm":"S-PLUS-CTRL","reason":"%s","started_at":"%s","updated_at":"%s","last_progress_at":"%s","runner_pid":%d,"workload_pid":%d,"process_alive":%s,"physical_gpu":%d,"visible_gpu":0,"gpu_count":1,"minimum_free_mib_per_gpu":%d,"admission_free_mib_per_gpu":[%d],"expected_peak_reserved_mib_per_gpu":17466,"hard_timeout_seconds":1209600,"progress_current":%s,"progress_total":12535,"progress_unit":"ctrl_optimizer_steps","parent_attempt_id":"s16_s2_splus_ctrl_formal_toys_gpu5_a3_accel_fp32","parent_artifacts_modified":false,"workload_rc":%d,"exit_code":%d,"exit_code_pending":%s,"test_read":false,"validation_used":false,"automatic_retry":false,"exact_start_command":"%s","output_dir":"%s","log_path":"%s/run.log","summary_path":"%s/summary.json"}\n' \
    "$ATTEMPT_ID" "$state" "$code" "$stage" "$reason" "$STARTED_AT" "$(date -Is)" "$last_progress" $$ "$WORKLOAD_PID" "$alive" "$GPU" "$MINIMUM_FREE" "$ADMISSION_FREE" "$progress" "$rc" "$rc" "$pending" "$EXACT_COMMAND" "$OUTPUT_REL" "$OUTPUT_REL" "$OUTPUT_REL" > "$temporary"
  mv "$temporary" "$STATUS"
}

terminate_workload() {
  if (( WORKLOAD_PID > 0 )) && kill -0 "$WORKLOAD_PID" 2>/dev/null; then
    kill -TERM "$WORKLOAD_PID" 2>/dev/null || true
    sleep 10
    kill -KILL "$WORKLOAD_PID" 2>/dev/null || true
  fi
}

finish_and_exit() {
  local prior_rc=$?
  trap - EXIT TERM INT HUP
  WORKLOAD_PID=0
  write_status "$FINAL_STATE" "$FINAL_CODE" "$FINAL_STAGE" "$FINAL_REASON" "$FINAL_RC" "$FINAL_ALIVE" "$FINAL_PENDING"
  exit "$prior_rc"
}

on_signal() {
  terminate_workload
  FINAL_STATE=failed; FINAL_CODE=INTERRUPTED; FINAL_STAGE=finished
  FINAL_REASON="GPU7 split CTRL runner received a termination signal; no automatic retry."
  FINAL_RC=143; FINAL_ALIVE=false; FINAL_PENDING=false
  exit 143
}

fail_and_exit() {
  local code=$1 reason=$2 rc=$3
  FINAL_STATE=failed; FINAL_CODE=$code; FINAL_STAGE=finished; FINAL_REASON=$reason
  FINAL_RC=$rc; FINAL_ALIVE=false; FINAL_PENDING=false
  exit "$rc"
}

trap finish_and_exit EXIT
trap on_signal TERM INT HUP
cd "$ROOT" || exit 2
if [[ "$GPU" != 7 ]]; then fail_and_exit GPU_SCOPE_FAILED "This prepared split attempt is authorized only for physical GPU7." 7; fi
if [[ -e "$OUTPUT/summary.json" || -e "$OUTPUT/arms/S-PLUS-CTRL/checkpoints" ]]; then
  fail_and_exit OUTPUT_EXISTS "Refusing to overwrite or implicitly resume an existing GPU7 CTRL split attempt." 8
fi
if [[ ! -s "$PARENT_RESOLVED" ]]; then
  fail_and_exit PARENT_CONFIG_MISSING "The frozen GPU5 a3 resolved config is missing." 11
fi
if [[ -e "$PARENT_OUTPUT/arms/S-PLUS-CTRL/checkpoints" || -e "$PARENT_OUTPUT/arms/S-PLUS-CTRL/summary.json" ]]; then
  fail_and_exit PARENT_CTRL_ALREADY_STARTED "GPU5 a3 has already started or completed its serial CTRL arm; split launch is blocked." 12
fi
mkdir -p "$OUTPUT"
write_status running RUNNING preflight "Running isolated CTRL syntax, tests, parent, config, resource, and disk preflight." -1 false true
timeout --signal=TERM --kill-after=10 300 "$PYTHON" -m py_compile \
  experiment/phase16/protocol/splus_formal_train_accel.py \
  experiment/phase16/protocol/finalize_splus_ctrl_split.py \
  experiment/phase16/tests/test_splus_ctrl_split.py >> "$LOG" 2>&1 || fail_and_exit PREFLIGHT_FAILED "Split syntax preflight failed." 2
timeout --signal=TERM --kill-after=10 300 "$PYTHON" -m unittest discover -s experiment/phase16/tests -p 'test_*.py' -q >> "$LOG" 2>&1 || fail_and_exit PREFLIGHT_FAILED "Stage16 tests failed." 2
"$PYTHON" experiment/phase16/protocol/splus_formal_train_accel.py --config "$CONFIG" --resolve-only >> "$LOG" 2>&1 || fail_and_exit CONFIG_RESOLUTION_FAILED "GPU7 CTRL split overlay resolution failed." 2
"$PYTHON" experiment/phase16/protocol/finalize_splus_ctrl_split.py \
  --mode preflight --config "$RESOLVED_CONFIG" --plus-config "$PARENT_RESOLVED" >> "$LOG" 2>&1 || fail_and_exit SPLIT_IDENTITY_FAILED "GPU7 CTRL is not scientifically identical to GPU5 S-PLUS." 3
available_disk=$(df -Pm "$OUTPUT" | awk 'NR==2 {print $4}')
if (( available_disk < DISK_RESERVATION_MIB )); then fail_and_exit DISK_ADMISSION_FAILED "Less than 8192 MiB disk is available." 10; fi
ADMISSION_FREE=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
if (( ADMISSION_FREE < MINIMUM_FREE )); then fail_and_exit GPU_ADMISSION_FAILED "GPU7 free memory is below the frozen 28672 MiB admission threshold." 9; fi
printf 'timestamp,physical_gpu,arm,memory_used_mib,memory_free_mib,utilization_gpu_percent\n' > "$TELEMETRY"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" experiment/phase16/protocol/splus_formal_train_accel.py \
  --config "$CONFIG" --arm S-PLUS-CTRL >> "$LOG" 2>&1 &
WORKLOAD_PID=$!
write_status running RUNNING training "Formal isolated S-PLUS-CTRL training is running on GPU7." -1 true true
arm_started=$(date +%s)
last_progress=0
last_change=$arm_started
while kill -0 "$WORKLOAD_PID" 2>/dev/null; do
  now=$(date +%s)
  if (( now - arm_started >= HARD_TIMEOUT )); then
    terminate_workload
    wait "$WORKLOAD_PID" 2>/dev/null || true
    WORKLOAD_PID=0
    FINAL_STATE=timeout; FINAL_CODE=TIMEOUT; FINAL_STAGE=finished
    FINAL_REASON="GPU7 split CTRL exceeded its equal 14-day hard timeout."
    FINAL_RC=124; FINAL_ALIVE=false; FINAL_PENDING=false
    exit 124
  fi
  telemetry=$(nvidia-smi --id="$GPU" --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
  printf '%s,%d,S-PLUS-CTRL,%s\n' "$(date -Is)" "$GPU" "$telemetry" >> "$TELEMETRY"
  progress=$(read_progress_field arm_optimizer_step 0)
  if [[ "$progress" != "$last_progress" ]]; then
    last_progress=$progress; last_change=$now
    write_status running RUNNING training "Formal isolated S-PLUS-CTRL training is progressing on GPU7." -1 true true
  elif (( now - last_change >= STALL_ADVISORY )); then
    write_status running STALL_SUSPECTED training "No CTRL optimizer-step change for at least 3600 seconds; advisory only." -1 true true
  else
    write_status running RUNNING training "Formal isolated S-PLUS-CTRL heartbeat on GPU7." -1 true true
  fi
  sleep "$HEARTBEAT"
done
wait "$WORKLOAD_PID"
rc=$?
WORKLOAD_PID=0
if (( rc != 0 )); then
  fail_and_exit FAILED "GPU7 split CTRL workload exited non-zero; no automatic retry." "$rc"
fi
write_status running RUNNING finalize "GPU7 CTRL completed; validating the isolated arm artifact contract." -1 false true
timeout --signal=TERM --kill-after=10 600 "$PYTHON" experiment/phase16/protocol/finalize_splus_ctrl_split.py \
  --mode arm --config "$RESOLVED_CONFIG" >> "$LOG" 2>&1 || fail_and_exit ARTIFACT_CONTRACT_FAILED "GPU7 split CTRL artifact contract failed." 3
FINAL_STATE=completed; FINAL_CODE=COMPLETED_CTRL_ONLY; FINAL_STAGE=finished
FINAL_REASON="PASS_S16_2_S_PLUS_CTRL_SPLIT_FORMAL_EXECUTION; cross-attempt paired finalization remains pending."
FINAL_RC=0; FINAL_ALIVE=false; FINAL_PENDING=false
exit 0
