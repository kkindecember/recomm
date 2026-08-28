#!/usr/bin/env bash
set -u -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
CONFIG=${CONFIG_PATH:-experiment/phase16/configs/stage16_s2_splus_accelerated_batch_sweep_gpu5_a1.json}
OUTPUT_REL=${OUTPUT_REL_PATH:-artifacts/phase16/s2_splus_accelerated_batch_sweep/gpu5_a1}
OUTPUT="$ROOT/$OUTPUT_REL"
STATUS="$OUTPUT/status.json"
LOG="$OUTPUT/run.log"
GPU=${1:-5}
ATTEMPT_ID=${ATTEMPT_ID_OVERRIDE:-s16_s2_splus_accelerated_batch_sweep_gpu5_a1}
CANDIDATES=${CANDIDATE_IDS:-"e64_g16_a16 e32_g8_a32 e16_g4_a64 e8_g2_a128"}
EXACT_COMMAND=${EXACT_COMMAND_OVERRIDE:-"bash experiment/phase16/run_stage16_s2_splus_accelerated_batch_sweep.sh 5"}
HOLDER_CONTROLLER="$ROOT/tools/gram_ablation_scan.sh"
HOLDER_STATE_ROOT="$ROOT/.runtime/gram_ablation_scan_gpu5"
HOLDER_SESSION=gram_ablation_scan_gpu5
HOLDER_RESERVE_MIB=18263
STARTED_AT=$(date -Is)
CURRENT_CANDIDATE=preflight
FINAL_STATE=running
FINAL_CODE=RUNNING
FINAL_REASON="Accelerated-batch sweep is running."
FINAL_RC=-1
HOLDER_RESTORED=false
HOLDER_RESTORED_PID=0

write_status() {
  local state=$1 code=$2 reason=$3 rc=$4 temporary
  FINAL_STATE=$state
  FINAL_CODE=$code
  FINAL_REASON=$reason
  FINAL_RC=$rc
  temporary="$STATUS.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE16_S2_SPLUS_ACCELERATED_BATCH_SWEEP","attempt_id":"%s","status":"%s","status_code":"%s","reason":"%s","current_candidate":"%s","started_at":"%s","updated_at":"%s","runner_pid":%d,"physical_gpu":%d,"holder_released_during_probe":true,"holder_restored":%s,"holder_restored_pid":%d,"exit_code":%d,"scientific_efficacy_metric_produced":false,"validation_used":false,"test_read":false,"automatic_retry":false,"exact_start_command":"%s","output_dir":"%s"}\n' \
    "$ATTEMPT_ID" "$state" "$code" "$reason" "$CURRENT_CANDIDATE" "$STARTED_AT" "$(date -Is)" $$ "$GPU" "$HOLDER_RESTORED" "$HOLDER_RESTORED_PID" "$rc" "$EXACT_COMMAND" "$OUTPUT_REL" > "$temporary"
  mv "$temporary" "$STATUS"
}

validate_restored_holder() {
  local holder_status="$HOLDER_STATE_ROOT/status.json" cmdline
  [[ -s "$holder_status" && -s "$HOLDER_STATE_ROOT/gpu.txt" ]] || return 1
  [[ "$(tr -d '[:space:]' < "$HOLDER_STATE_ROOT/gpu.txt")" == "$GPU" ]] || return 1
  read -r HOLDER_RESTORED_PID actual_reserve < <(
    "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d.get("state")=="running"; print(int(d["pid"]),int(d["reserve_mib"]))' "$holder_status"
  ) || return 1
  [[ "$actual_reserve" == "$HOLDER_RESERVE_MIB" && -r "/proc/$HOLDER_RESTORED_PID/cmdline" ]] || return 1
  cmdline=$(tr '\0' ' ' < "/proc/$HOLDER_RESTORED_PID/cmdline")
  [[ "$cmdline" == *"tools/gram_ablation_scan_worker.py"* && "$cmdline" == *"--reserve-mib $HOLDER_RESERVE_MIB"* ]] || return 1
  tmux has-session -t "$HOLDER_SESSION" 2>/dev/null
}

restore_holder() {
  local prior_rc=$? restore_rc=0
  trap - EXIT TERM INT HUP
  if validate_restored_holder; then
    HOLDER_RESTORED=true
  else
    SESSION="$HOLDER_SESSION" STATE_ROOT="$HOLDER_STATE_ROOT" RESERVE_MIB="$HOLDER_RESERVE_MIB" \
      bash "$HOLDER_CONTROLLER" start "$GPU" >> "$LOG" 2>&1 || restore_rc=$?
    if (( restore_rc == 0 )) && validate_restored_holder; then
      HOLDER_RESTORED=true
    else
      HOLDER_RESTORED=false
      FINAL_STATE=blocked
      FINAL_CODE=HOLDER_RESTORE_FAILED
      FINAL_REASON="Batch probe ended but the required GPU5 holder could not be restored."
      FINAL_RC=17
    fi
  fi
  write_status "$FINAL_STATE" "$FINAL_CODE" "$FINAL_REASON" "$FINAL_RC"
  if [[ "$HOLDER_RESTORED" != true ]]; then exit 17; fi
  exit "$prior_rc"
}

on_signal() {
  FINAL_STATE=failed
  FINAL_CODE=INTERRUPTED
  FINAL_REASON="Batch sweep received a termination signal; holder restoration initiated."
  FINAL_RC=143
  exit 143
}

trap restore_holder EXIT
trap on_signal TERM INT HUP
cd "$ROOT" || exit 2
if [[ "$GPU" != 5 ]]; then
  FINAL_STATE=blocked; FINAL_CODE=GPU_SCOPE_FAILED; FINAL_REASON="This sweep is authorized only for GPU5."; FINAL_RC=7; exit 7
fi
if tmux has-session -t "$HOLDER_SESSION" 2>/dev/null; then
  FINAL_STATE=blocked; FINAL_CODE=HOLDER_NOT_RELEASED; FINAL_REASON="GPU5 holder is unexpectedly active; no probe started."; FINAL_RC=13; exit 13
fi
if [[ -e "$OUTPUT/summary.json" || -e "$OUTPUT/candidates" ]]; then
  FINAL_STATE=blocked; FINAL_CODE=OUTPUT_EXISTS; FINAL_REASON="Refusing to overwrite an existing batch sweep."; FINAL_RC=8; exit 8
fi
mkdir -p "$OUTPUT"
write_status running RUNNING "Running syntax and free-memory admission checks." -1
timeout --signal=TERM --kill-after=10 300 "$PYTHON" -m py_compile experiment/phase16/protocol/splus_accelerated_batch_probe.py >> "$LOG" 2>&1 || {
  FINAL_STATE=failed; FINAL_CODE=PREFLIGHT_FAILED; FINAL_REASON="Batch-probe syntax preflight failed."; FINAL_RC=2; exit 2;
}
admission_free=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
if (( admission_free < 32768 )); then
  FINAL_STATE=blocked; FINAL_CODE=GPU_ADMISSION_FAILED; FINAL_REASON="GPU5 has less than 32768 MiB free after holder release."; FINAL_RC=9; exit 9
fi

for candidate in $CANDIDATES; do
  CURRENT_CANDIDATE=$candidate
  write_status running RUNNING "Running preregistered accelerated-batch candidate." -1
  timeout --signal=TERM --kill-after=10 600 env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
    experiment/phase16/protocol/splus_accelerated_batch_probe.py --config "$CONFIG" --candidate-id "$candidate" >> "$LOG" 2>&1
  rc=$?
  if (( rc == 0 )); then break; fi
  if (( rc != 3 && rc != 4 )); then
    FINAL_STATE=failed; FINAL_CODE=CANDIDATE_ERROR; FINAL_REASON="A batch candidate failed outside the preregistered resource-failure modes."; FINAL_RC=$rc; exit "$rc"
  fi
done

CURRENT_CANDIDATE=aggregate
"$PYTHON" experiment/phase16/protocol/splus_accelerated_batch_probe.py --config "$CONFIG" --aggregate >> "$LOG" 2>&1
rc=$?
if (( rc != 0 )); then
  FINAL_STATE=failed; FINAL_CODE=NO_ELIGIBLE_BATCH; FINAL_REASON="No preregistered accelerated batch passed."; FINAL_RC=$rc; exit "$rc"
fi
FINAL_STATE=completed
FINAL_CODE=COMPLETED
FINAL_REASON="PASS_S16_2_SPLUS_ACCELERATED_BATCH_SWEEP; holder restoration initiated."
FINAL_RC=0
exit 0
