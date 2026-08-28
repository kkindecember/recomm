#!/usr/bin/env bash
set -u -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
CONFIG=experiment/phase16/configs/stage16_s2_saux_formal_toys.json
OUTPUT="$ROOT/artifacts/phase16/s2_saux_formal/toys_seed1502_a1"
STATUS="$OUTPUT/status.json"
PROGRESS="$OUTPUT/progress.json"
LOG="$OUTPUT/run.log"
TELEMETRY="$OUTPUT/gpu_telemetry.csv"
GPU=${1:-2}
MINIMUM_FREE=24576
HARD_TIMEOUT=172800
HEARTBEAT=30
TOTAL_STEPS=4200
EXACT_COMMAND="bash experiment/phase16/run_stage16_s2_saux_formal.sh $GPU"
STARTED_AT=$(date -Is)
START_EPOCH_SECONDS=$(date +%s)
WORKLOAD_PID=0

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
  current=$(read_progress_field global_step 0)
  last_progress=$(read_progress_field updated_at "$STARTED_AT")
  temporary="$STATUS.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE16_S2_SAUX_FORMAL_TOYS","attempt_id":"s16_s2_saux_toys_a1","status":"%s","status_code":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","last_progress_at":"%s","runner_pid":%d,"workload_pid":%d,"process_alive":%s,"physical_gpu":%d,"visible_gpu":0,"gpu_count":1,"minimum_free_mib_per_gpu":24576,"admission_free_mib_per_gpu":[%d],"expected_peak_mib_per_gpu":20480,"progress_current":%s,"progress_total":4200,"progress_unit":"optimizer_steps","hard_timeout_seconds":172800,"workload_rc":%d,"exit_code":%d,"exit_code_pending":%s,"test_read":false,"automatic_retry":false,"exact_start_command":"%s","output_dir":"artifacts/phase16/s2_saux_formal/toys_seed1502_a1","log_path":"artifacts/phase16/s2_saux_formal/toys_seed1502_a1/run.log","summary_path":"artifacts/phase16/s2_saux_formal/toys_seed1502_a1/summary.json"}\n' \
    "$state" "$code" "$stage" "$reason" "$STARTED_AT" "$(date -Is)" "$last_progress" $$ "$WORKLOAD_PID" "$alive" "$GPU" "$ADMISSION_FREE" "$current" "$rc" "$rc" "$pending" "$EXACT_COMMAND" > "$temporary"
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
  write_status failed INTERRUPTED finished "Formal S-AUX runner received a termination signal; no automatic retry." 143 false false
  exit 143
}
trap on_signal TERM INT HUP

cd "$ROOT" || exit 2
if [[ "$GPU" != "2" ]]; then
  echo "This frozen attempt is authorized only for physical GPU 2." >&2
  exit 7
fi
if [[ -e "$OUTPUT/summary.json" || -e "$OUTPUT/checkpoints/last_state.pt" ]]; then
  echo "Refusing to overwrite or implicitly resume an existing formal attempt." >&2
  exit 8
fi
mkdir -p "$OUTPUT"
ADMISSION_FREE=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
ADMISSION_UTIL=$(nvidia-smi --id="$GPU" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
if (( ADMISSION_FREE < MINIMUM_FREE )); then
  WORKLOAD_PID=0
  write_status failed GPU_ADMISSION_FAILED admission "GPU 2 free memory fell below 24576 MiB; no workload started and no automatic retry." 9 false false
  exit 9
fi

write_status running RUNNING preflight "Running formal S-AUX syntax/tests before workload start." -1 true true
timeout --signal=TERM --kill-after=10 300 \
  "$PYTHON" -m py_compile \
    experiment/phase16/protocol/official_specgr_runtime.py \
    experiment/phase16/protocol/specgr_faithful.py \
    experiment/phase16/protocol/saux_formal_train.py \
    experiment/phase16/protocol/finalize_saux_formal.py \
    experiment/phase16/tests/test_saux_formal.py >> "$LOG" 2>&1
rc=$?
if (( rc != 0 )); then
  write_status failed FAILED finished "Formal S-AUX syntax preflight failed; no automatic retry." "$rc" false false
  exit "$rc"
fi
timeout --signal=TERM --kill-after=10 300 \
  "$PYTHON" -m unittest discover -s experiment/phase16/tests -p 'test_saux_formal.py' -v >> "$LOG" 2>&1
rc=$?
if (( rc != 0 )); then
  write_status failed FAILED finished "Formal S-AUX data contract tests failed; no automatic retry." "$rc" false false
  exit "$rc"
fi

printf 'timestamp,physical_gpu,memory_used_mib,memory_free_mib,utilization_gpu_percent\n' > "$TELEMETRY"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" experiment/phase16/protocol/saux_formal_train.py --config "$CONFIG" >> "$LOG" 2>&1 &
WORKLOAD_PID=$!
write_status running RUNNING training "Official S-AUX formal training on user-selected physical GPU 2." -1 true true
LAST_PROGRESS=0
LAST_CHANGE_SECONDS=$(date +%s)

while kill -0 "$WORKLOAD_PID" 2>/dev/null; do
  now=$(date +%s)
  if (( now - START_EPOCH_SECONDS >= HARD_TIMEOUT )); then
    terminate_workload
    wait "$WORKLOAD_PID" 2>/dev/null || true
    write_status timeout TIMEOUT finished "Formal S-AUX exceeded the 48-hour hard timeout; no automatic retry." 124 false false
    exit 124
  fi
  telemetry=$(nvidia-smi --id="$GPU" --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
  printf '%s,%d,%s\n' "$(date -Is)" "$GPU" "$telemetry" >> "$TELEMETRY"
  CURRENT_PROGRESS=$(read_progress_field global_step 0)
  if [[ "$CURRENT_PROGRESS" != "$LAST_PROGRESS" ]]; then
    LAST_PROGRESS=$CURRENT_PROGRESS
    LAST_CHANGE_SECONDS=$now
    write_status running RUNNING training "Official S-AUX formal training is progressing." -1 true true
  elif (( now - LAST_CHANGE_SECONDS >= 300 )); then
    write_status running STALL_SUSPECTED training "No optimizer-step change for at least 300 seconds; advisory only, workload continues." -1 true true
  else
    write_status running RUNNING training "Official S-AUX formal training heartbeat." -1 true true
  fi
  sleep "$HEARTBEAT"
done

wait "$WORKLOAD_PID"
rc=$?
if (( rc != 0 )); then
  write_status failed FAILED finished "Formal S-AUX workload exited non-zero; no automatic retry." "$rc" false false
  exit "$rc"
fi
timeout --signal=TERM --kill-after=10 300 \
  "$PYTHON" experiment/phase16/protocol/finalize_saux_formal.py --config "$CONFIG" >> "$LOG" 2>&1
rc=$?
if (( rc != 0 )); then
  write_status failed ARTIFACT_CONTRACT_FAILED finished "Formal S-AUX artifact contract failed; no automatic retry." "$rc" false false
  exit "$rc"
fi
write_status completed COMPLETED finished "PASS_S16_2_SAUX_FAITHFUL_CONTRACT_ADMISSION." 0 false false
