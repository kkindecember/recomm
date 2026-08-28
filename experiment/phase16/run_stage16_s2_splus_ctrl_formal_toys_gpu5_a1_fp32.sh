#!/usr/bin/env bash
set -u -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
CONFIG=${FORMAL_CONFIG:-experiment/phase16/configs/stage16_s2_splus_ctrl_formal_toys_gpu5_a1_fp32.json}
OUTPUT_REL=${FORMAL_OUTPUT_REL:-artifacts/phase16/s2_splus_ctrl_formal/toys_seed1502_gpu5_a1_fp32}
ATTEMPT_ID=${FORMAL_ATTEMPT_ID:-s16_s2_splus_ctrl_formal_toys_gpu5_a1_fp32}
OUTPUT="$ROOT/$OUTPUT_REL"
STATUS="$OUTPUT/status.json"
PROGRESS="$OUTPUT/progress.json"
LOG="$OUTPUT/run.log"
TELEMETRY="$OUTPUT/gpu_telemetry.csv"
GPU=${1:-5}
MINIMUM_FREE=10240
PER_ARM_HARD_TIMEOUT=1209600
HEARTBEAT=60
STALL_ADVISORY=3600
DISK_RESERVATION_MIB=8192
EXACT_COMMAND=${FORMAL_EXACT_COMMAND:-"bash experiment/phase16/run_stage16_s2_splus_ctrl_formal_toys_gpu5_a1_fp32.sh $GPU"}
STARTED_AT=$(date -Is)
WORKLOAD_PID=0
ADMISSION_FREE=0
CURRENT_ARM=preflight
ARM_STARTED_SECONDS=0
HOLDER_PID=0
HOLDER_RESERVE_MIB=0

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
  local progress last_progress temporary
  progress=$(read_progress_field paired_optimizer_step 0)
  last_progress=$(read_progress_field updated_at "$STARTED_AT")
  temporary="$STATUS.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE16_S2_SPLUS_CTRL_FORMAL_TOYS","attempt_id":"%s","status":"%s","status_code":"%s","stage":"%s","current_arm":"%s","reason":"%s","started_at":"%s","updated_at":"%s","last_progress_at":"%s","runner_pid":%d,"workload_pid":%d,"process_alive":%s,"physical_gpu":%d,"visible_gpu":0,"gpu_count":1,"minimum_free_mib_per_gpu":10240,"admission_free_mib_per_gpu":[%d],"expected_peak_reserved_mib_per_gpu":4978,"per_arm_hard_timeout_seconds":1209600,"progress_current":%s,"progress_total":25070,"progress_unit":"paired_optimizer_steps","holder_pid_at_admission":%d,"holder_reserve_mib":%d,"holder_released":false,"workload_rc":%d,"exit_code":%d,"exit_code_pending":%s,"test_read":false,"validation_used":false,"automatic_retry":false,"exact_start_command":"%s","output_dir":"%s","log_path":"%s/run.log","summary_path":"%s/summary.json"}\n' \
    "$ATTEMPT_ID" "$state" "$code" "$stage" "$CURRENT_ARM" "$reason" "$STARTED_AT" "$(date -Is)" "$last_progress" $$ "$WORKLOAD_PID" "$alive" "$GPU" "$ADMISSION_FREE" "$progress" "$HOLDER_PID" "$HOLDER_RESERVE_MIB" "$rc" "$rc" "$pending" "$EXACT_COMMAND" "$OUTPUT_REL" "$OUTPUT_REL" "$OUTPUT_REL" > "$temporary"
  mv "$temporary" "$STATUS"
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
  write_status failed INTERRUPTED finished "Formal paired runner received a termination signal; no automatic retry." 143 false false
  exit 143
}

validate_holder() {
  local state_root="$ROOT/.runtime/gram_ablation_scan_gpu5" status="$ROOT/.runtime/gram_ablation_scan_gpu5/status.json" cmdline=""
  [[ -s "$status" && -s "$state_root/gpu.txt" ]] || return 1
  [[ "$(tr -d '[:space:]' < "$state_root/gpu.txt")" == "$GPU" ]] || return 1
  read -r HOLDER_PID HOLDER_RESERVE_MIB < <(
    "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d.get("state")=="running"; print(int(d["pid"]),int(d["reserve_mib"]))' "$status"
  ) || return 1
  [[ "$HOLDER_PID" == "464054" && "$HOLDER_RESERVE_MIB" == "18263" && -r "/proc/$HOLDER_PID/cmdline" ]] || return 1
  cmdline=$(tr '\0' ' ' < "/proc/$HOLDER_PID/cmdline")
  [[ "$cmdline" == *"tools/gram_ablation_scan_worker.py"* && "$cmdline" == *"--reserve-mib $HOLDER_RESERVE_MIB"* ]] || return 1
  tmux has-session -t gram_ablation_scan_gpu5 2>/dev/null || return 1
}

run_arm() {
  local arm=$1 last_progress=0 last_change now telemetry rc
  CURRENT_ARM=$arm
  ARM_STARTED_SECONDS=$(date +%s)
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" experiment/phase16/protocol/splus_formal_train.py --config "$CONFIG" --arm "$arm" >> "$LOG" 2>&1 &
  WORKLOAD_PID=$!
  write_status running RUNNING training "Formal $arm training is running; holder remains active." -1 true true
  last_change=$ARM_STARTED_SECONDS
  while kill -0 "$WORKLOAD_PID" 2>/dev/null; do
    now=$(date +%s)
    if (( now - ARM_STARTED_SECONDS >= PER_ARM_HARD_TIMEOUT )); then
      terminate_workload
      wait "$WORKLOAD_PID" 2>/dev/null || true
      WORKLOAD_PID=0
      write_status timeout TIMEOUT finished "Formal $arm exceeded its equal 14-day hard timeout; no automatic retry." 124 false false
      return 124
    fi
    telemetry=$(nvidia-smi --id="$GPU" --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
    printf '%s,%d,%s,%s\n' "$(date -Is)" "$GPU" "$arm" "$telemetry" >> "$TELEMETRY"
    progress=$(read_progress_field paired_optimizer_step 0)
    if [[ "$progress" != "$last_progress" ]]; then
      last_progress=$progress
      last_change=$now
      write_status running RUNNING training "Formal $arm training is progressing." -1 true true
    elif (( now - last_change >= STALL_ADVISORY )); then
      write_status running STALL_SUSPECTED training "No optimizer-step change for at least 3600 seconds; advisory only, workload continues." -1 true true
    else
      write_status running RUNNING training "Formal $arm training heartbeat; holder remains active." -1 true true
    fi
    sleep "$HEARTBEAT"
  done
  wait "$WORKLOAD_PID"
  rc=$?
  WORKLOAD_PID=0
  if (( rc != 0 )); then
    write_status failed FAILED finished "Formal $arm workload exited non-zero; no automatic retry." "$rc" false false
    return "$rc"
  fi
  write_status running RUNNING arm_completed "Formal $arm completed; preparing next paired stage." -1 true true
  return 0
}

trap on_signal TERM INT HUP
cd "$ROOT" || exit 2
if [[ "$GPU" != "5" ]]; then
  echo "This frozen formal attempt is authorized only for physical GPU 5." >&2
  exit 7
fi
if [[ -e "$OUTPUT/summary.json" || -e "$OUTPUT/arms/S-PLUS/checkpoints" || -e "$OUTPUT/arms/S-PLUS-CTRL/checkpoints" ]]; then
  echo "Refusing to overwrite or implicitly resume an existing formal attempt." >&2
  exit 8
fi
mkdir -p "$OUTPUT"
write_status running RUNNING preflight "Running paired formal syntax, tests, resource evidence, disk, and holder preflight." -1 true true
timeout --signal=TERM --kill-after=10 300 "$PYTHON" -m py_compile \
  experiment/phase16/protocol/splus_formal_train.py \
  experiment/phase16/protocol/finalize_splus_formal.py \
  experiment/phase16/tests/test_splus_formal.py >> "$LOG" 2>&1
rc=$?
if (( rc != 0 )); then write_status failed PREFLIGHT_FAILED finished "Formal syntax preflight failed; no automatic retry." "$rc" false false; exit "$rc"; fi
timeout --signal=TERM --kill-after=10 300 "$PYTHON" -m unittest discover -s experiment/phase16/tests -p 'test_*.py' -q >> "$LOG" 2>&1
rc=$?
if (( rc != 0 )); then write_status failed PREFLIGHT_FAILED finished "Stage16 tests failed; no automatic retry." "$rc" false false; exit "$rc"; fi
"$PYTHON" -c 'import hashlib,json,pathlib,sys;c=json.load(open(sys.argv[1]));s=c["inputs"]["resource_sweep_summary"];p=pathlib.Path(s["path"]);assert hashlib.sha256(p.read_bytes()).hexdigest()==s["sha256"];d=json.load(p.open());assert d["verdict"]=="PASS_S16_2_SPLUS_OBJECTIVE_RESOURCE_SWEEP" and d["execution_precision"]=="fp32" and d["recommended_minimum_free_mib"]==9216 and d["checkpoint_unchanged"] and not d["test_read"]' "$CONFIG" >> "$LOG" 2>&1
rc=$?
if (( rc != 0 )); then write_status failed RESOURCE_EVIDENCE_FAILED finished "FP32 resource evidence failed; no automatic retry." "$rc" false false; exit "$rc"; fi
"$PYTHON" -c 'import hashlib,json,pathlib,sys;c=json.load(open(sys.argv[1]));
for name,expected in c["code_freeze"].items():
 assert hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()==expected,name' "$CONFIG" >> "$LOG" 2>&1
rc=$?
if (( rc != 0 )); then write_status failed CODE_FREEZE_FAILED finished "Formal code SHA freeze failed; no automatic retry." "$rc" false false; exit "$rc"; fi
available_disk=$(df -Pm "$OUTPUT" | awk 'NR==2 {print $4}')
if (( available_disk < DISK_RESERVATION_MIB )); then write_status blocked DISK_ADMISSION_FAILED finished "Less than 8192 MiB disk is available; no workload started." 10 false false; exit 10; fi
if ! validate_holder; then write_status blocked HOLDER_VALIDATION_FAILED finished "GPU5 owned holder identity/state validation failed; no process was changed." 13 false false; exit 13; fi
ADMISSION_FREE=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
if (( ADMISSION_FREE < MINIMUM_FREE )); then write_status failed GPU_ADMISSION_FAILED finished "GPU5 free memory is below 10240 MiB with holder preserved; no workload started." 9 false false; exit 9; fi
printf 'timestamp,physical_gpu,arm,memory_used_mib,memory_free_mib,utilization_gpu_percent\n' > "$TELEMETRY"

run_arm S-PLUS
rc=$?
if (( rc != 0 )); then exit "$rc"; fi
run_arm S-PLUS-CTRL
rc=$?
if (( rc != 0 )); then exit "$rc"; fi
CURRENT_ARM=finalize
write_status running RUNNING finalize "Both formal arms completed; validating paired artifact and budget contracts." -1 true true
timeout --signal=TERM --kill-after=10 600 "$PYTHON" experiment/phase16/protocol/finalize_splus_formal.py --config "$CONFIG" >> "$LOG" 2>&1
rc=$?
if (( rc != 0 )); then write_status failed ARTIFACT_CONTRACT_FAILED finished "Paired formal artifact contract failed; no automatic retry." "$rc" false false; exit "$rc"; fi
if ! validate_holder; then write_status blocked HOLDER_TERMINAL_OBSERVATION_FAILED finished "Scientific run completed but the untouched GPU5 holder is no longer at its admitted identity." 15 false false; exit 15; fi
write_status completed COMPLETED finished "PASS_S16_2_SPLUS_FAITHFUL_CONTRACT_ADMISSION and matched CTRL formal execution." 0 false false
