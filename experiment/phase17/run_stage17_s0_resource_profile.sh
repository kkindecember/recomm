#!/usr/bin/env bash
# S17-0 validation-only GRAM resource probe (100 and 1,000 Toys D0 users).
# Usage: bash experiment/phase17/run_stage17_s0_resource_profile.sh {start|status|worker} [gpu|started_at]
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
ACTION=${1:-status}
GPU=${S17_PROFILE_GPU:-5}
ATTEMPT_ID=${S17_PROFILE_ATTEMPT:-attempt_002}
SESSION=s17_s0_gram_profile_toys_d0
TMUX_SESSION_JSON="\"$SESSION\""
if [[ ${S17_PROFILE_DIRECT:-0} == 1 ]]; then
  TMUX_SESSION_JSON=null
fi
OUTPUT="$ROOT/artifacts/phase17/s0_audit/resource_profile"
STATUS="$ROOT/artifacts/phase17/status/s17_s0_gram_profile_toys_d0.status.json"
LOG="$OUTPUT/runner.log"
TELEMETRY="$OUTPUT/gpu_telemetry.csv"
STARTED_AT=""
STAGE=not_started
WORKLOAD_PID=0
TELEMETRY_PID=0

write_status() {
  local scientific_state=$1 execution_state=$2 code=$3 progress_current=$4
  local temporary="${STATUS}.tmp.$$"
  mkdir -p "$(dirname "$STATUS")" "$OUTPUT"
  printf '{"experiment_id":"s17_s0_gram_profile_toys_d0","attempt_id":"%s","step_id":"S17-0","track_id":null,"scientific_state":"%s","execution_state":"%s","status_code":"%s","started_at":"%s","updated_at":"%s","launcher_pid":%d,"workload_pid":%d,"process_alive":%s,"tmux_session":%s,"gpu_ids":[%d],"stage":"%s","progress":{"current":%d,"total":2},"canonical_result_dir":"artifacts/phase17/s0_audit/resource_profile","log_path":"artifacts/phase17/s0_audit/resource_profile/runner.log","test_read":false,"sports_read":false,"result_selection_eligible":false,"occupancy_mode":"none","repeat_iteration":0,"repeat_metrics_ignored":false,"affects_scientific_result":true}\n' \
    "$ATTEMPT_ID" "$scientific_state" "$execution_state" "$code" "$STARTED_AT" "$(date -Is)" "$$" "$WORKLOAD_PID" \
    "$([[ "$scientific_state" == RUNNING ]] && printf true || printf false)" "$TMUX_SESSION_JSON" "$GPU" "$STAGE" "$progress_current" > "$temporary"
  mv "$temporary" "$STATUS"
}

write_attempt() {
  local state=$1 failure_reason=$2 temporary="$OUTPUT/attempts/${ATTEMPT_ID}.json.tmp.$$"
  mkdir -p "$OUTPUT/attempts"
  printf '{"attempt_id":"%s","step_id":"S17-0","track_id":null,"kind":"recovery","started_at":"%s","ended_at":"%s","state":"%s","config_sha256":null,"data_manifest_sha256":null,"source_sha256":null,"scientific_result_eligible":false,"failure_reason":%s,"artifact_dir":"artifacts/phase17/s0_audit/resource_profile"}\n' \
    "$ATTEMPT_ID" "$STARTED_AT" "$(date -Is)" "$state" "$failure_reason" > "$temporary"
  mv "$temporary" "$OUTPUT/attempts/${ATTEMPT_ID}.json"
}

telemetry() {
  printf 'timestamp,index,memory_used_mib,memory_free_mib,utilization_gpu_percent\n' > "$TELEMETRY"
  while true; do
    timeout 10 nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits --id="$GPU" >> "$TELEMETRY" 2>/dev/null || true
    sleep 5
  done
}

run_probe() {
  local size=$1 dataset="Toys_s17_d0_${1}" probe_log="$OUTPUT/probe_${1}.log"
  local started ended rc
  STAGE="gram_profile_${size}"
  write_status RUNNING RUNNING_SCIENTIFIC "S17_0_PROFILE_${size}" "$([[ "$size" == 100 ]] && printf 0 || printf 1)"
  started=$(date +%s)
  cd "$ROOT/GRAM/command" || return 90
  timeout --signal=TERM --kill-after=30 3600 env CUDA_VISIBLE_DEVICES="$GPU" \
    HF_HUB_CACHE="$ROOT/.cache/huggingface" \
    TRANSFORMERS_CACHE="$ROOT/.cache/huggingface/transformers" \
    TRANSFORMERS_OFFLINE=1 "$PYTHON" ../src/main_generative_gram.py \
    --data_path "$ROOT/artifacts/phase17/s0_audit/profile_data" \
    --datasets "$dataset" \
    --distributed 0 --gpu 0 --seed 2023 --train 1 --resource_metrics 1 \
    --log_dir "$OUTPUT/gram_logs_${size}" --prediction_dir "$OUTPUT/predictions_${size}" \
    --item_prompt_max_len 128 --item_prompt all_text \
    --cf_model sasrec --id_linking 1 --max_his 20 \
    --rec_batch_size 16 --gradient_accumulation_steps 8 \
    --rec_lr 1e-3 --rec_epochs 1 --test_epoch_rec 0 --save_rec_epochs 1 \
    --save_predictions 0 --beam_size 50 --top_k_similar_item 5 \
    --item_id_type split --hierarchical_id_type hierarchy_v1_c32_l5_len32768_split \
    --debug_train_100 0 --debug_test_100 0 \
    --cf0_arm A --cf0_phase9 1 --hi_gram_enabled 0 > "$probe_log" 2>&1 &
  WORKLOAD_PID=$!
  write_status RUNNING RUNNING_SCIENTIFIC "S17_0_PROFILE_${size}" "$([[ "$size" == 100 ]] && printf 0 || printf 1)"
  wait "$WORKLOAD_PID"
  rc=$?
  WORKLOAD_PID=0
  ended=$(date +%s)
  printf 'PROFILE_RESULT size=%s rc=%s wall_seconds=%s\n' "$size" "$rc" "$((ended - started))" >> "$probe_log"
  cd "$ROOT" || return 91
  return "$rc"
}

finish() {
  local rc=$?
  trap - EXIT INT TERM HUP
  if (( WORKLOAD_PID > 0 )) && kill -0 "$WORKLOAD_PID" 2>/dev/null; then
    kill -TERM "$WORKLOAD_PID" 2>/dev/null || true
    wait "$WORKLOAD_PID" 2>/dev/null || true
  fi
  if (( TELEMETRY_PID > 0 )); then
    kill "$TELEMETRY_PID" 2>/dev/null || true
    wait "$TELEMETRY_PID" 2>/dev/null || true
  fi
  STAGE=finished
  if (( rc == 0 )); then
    write_status COMPLETED SCIENTIFIC_COMPLETED S17_0_PROFILE_COMPLETE 2
    write_attempt COMPLETED null
  else
    write_status FAILED SCIENTIFIC_FAILED S17_0_PROFILE_FAILED 0
    write_attempt FAILED '"see probe and runner logs; no automatic retry"'
  fi
  exit "$rc"
}

worker() {
  STARTED_AT=${1:?missing start timestamp}
  trap finish EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  trap 'exit 129' HUP
  cd "$ROOT" || exit 2
  mkdir -p "$OUTPUT"
  STAGE=preflight
  write_status RUNNING PREFLIGHT S17_0_PROFILE_PREFLIGHT 0
  "$PYTHON" -m unittest -v experiment.phase17.tests.test_s0_audit > "$OUTPUT/unit_tests.log" 2>&1 || exit 3
  "$PYTHON" -m py_compile experiment/phase17/protocol/s0_audit.py \
    experiment/phase17/protocol/finalize_s0_resource.py || exit 4
  bash -n "$0" || exit 5

  STAGE=gpu_admission
  write_status RUNNING WAITING_FOR_GPU S17_0_PROFILE_GPU_GATE 0
  local gpu_row free_mib util
  local admitted=0
  for _ in $(seq 1 240); do
    gpu_row=$(timeout 15 nvidia-smi --query-gpu=memory.free,utilization.gpu \
      --format=csv,noheader,nounits --id="$GPU" 2>/dev/null | tr -d ' ')
    free_mib=${gpu_row%%,*}
    util=${gpu_row##*,}
    if [[ "$free_mib" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ ]] \
      && (( free_mib >= 27000 && util <= 20 )); then
      admitted=1
      break
    fi
    sleep 30
  done
  (( admitted == 1 )) || exit 7

  telemetry & TELEMETRY_PID=$!
  if [[ ${S17_PROFILE_SKIP_100:-0} == 1 ]]; then
    grep -q 'PROFILE_RESULT size=100 rc=0' "$OUTPUT/probe_100.log" || exit 9
  else
    run_probe 100 || exit $?
  fi
  run_probe 1000 || exit $?
  STAGE=summary
  write_status RUNNING SUMMARIZING S17_0_PROFILE_SUMMARY 2
  S17_PROFILE_GPU="$GPU" "$PYTHON" experiment/phase17/protocol/finalize_s0_resource.py > "$OUTPUT/summary.log" 2>&1 || exit 8
}

case "$ACTION" in
  start)
    [[ ${2:-} =~ ^[0-7]$ ]] && GPU=$2
    mkdir -p "$OUTPUT" "$(dirname "$STATUS")"
    tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists: $SESSION" >&2; exit 1; }
    STARTED_AT=$(date -Is)
    printf -v command 'S17_PROFILE_GPU=%q S17_PROFILE_ATTEMPT=%q bash %q worker %q >> %q 2>&1' "$GPU" "$ATTEMPT_ID" "$0" "$STARTED_AT" "$LOG"
    tmux new-session -d -s "$SESSION" "$command" </dev/null >/dev/null 2>&1
    STAGE=starting
    write_status RUNNING BACKGROUND_STARTED S17_0_PROFILE_STARTING 0
    echo "started $SESSION on GPU$GPU"
    ;;
  worker)
    worker "${2:?missing start timestamp}"
    ;;
  status)
    tmux has-session -t "$SESSION" 2>/dev/null && echo "tmux session: running ($SESSION)" || echo "tmux session: not running ($SESSION)"
    [[ -f "$STATUS" ]] && sed -n '1,120p' "$STATUS" || echo '{"scientific_state":"PENDING"}'
    [[ -f "$LOG" ]] && tail -n 30 "$LOG" || true
    ;;
  *)
    echo "usage: $0 {start|status|worker} [gpu|started_at]" >&2
    exit 2
    ;;
esac
