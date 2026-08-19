#!/usr/bin/env bash
# Collision-safe v1 Toys smoke runner.
#
# Resource policy is deliberately direct: this script never starts/stops a
# holder, protector, or lease sidecar.  The GPU must be assigned explicitly by
# the user for this run.
#
# Usage:
#   bash experiment/phase13/run_v1_collision_safe_smoke.sh start <gpu>
#   bash experiment/phase13/run_v1_collision_safe_smoke.sh status
#   bash experiment/phase13/run_v1_collision_safe_smoke.sh stop
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
ACTION=${1:-status}
GPU=${2:-}

SUB=v1_collision_safe_smoke_toys
SESSION=gram_phase13_${SUB}
OUTPUT="$ROOT/artifacts/phase13/explore/$SUB"
LOG="$OUTPUT/run.log"
STATUS="$OUTPUT/status.json"
TELEMETRY="$OUTPUT/gpu_telemetry.csv"
METRICS="$OUTPUT/metrics_cold_warm.json"
ID_FILE="$ROOT/GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split_v1_mlpcold_collision_safe.txt"
ID_REPORT="$ROOT/artifacts/phase13/explore/v1_collision_safe/toys_id_report.json"
PREDICTIONS="$OUTPUT/predictions"
MIN_FREE_MIB=${MIN_FREE_MIB:-20000}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-7200}

STARTED_AT=""
RUNNER_PID=0
WORKLOAD_PID=0
TELEMETRY_PID=0

write_status() {
  local state=$1 stage=$2 reason=$3 tmp="${STATUS}.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE13_V1_COLLISION_SAFE_SMOKE_TOYS","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%d,"workload_pid":%d,"tmux_session":"%s","physical_gpu":%s,"resource_mode":"user_assigned_direct","log_path":"%s","metrics_path":"%s"}\n' \
    "$state" "$stage" "$reason" "$STARTED_AT" "$(date -Is)" "$RUNNER_PID" "$WORKLOAD_PID" \
    "$SESSION" "${GPU:--1}" "${LOG#$ROOT/}" "${METRICS#$ROOT/}" > "$tmp"
  mv "$tmp" "$STATUS"
}

telemetry() {
  printf 'timestamp,index,memory_used_mib,memory_free_mib,utilization_gpu_percent\n' > "$TELEMETRY"
  while true; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits --id="$GPU" >> "$TELEMETRY" 2>/dev/null || true
    sleep 5
  done
}

cleanup() {
  if (( TELEMETRY_PID > 0 )); then
    kill "$TELEMETRY_PID" >/dev/null 2>&1 || true
    wait "$TELEMETRY_PID" 2>/dev/null || true
    TELEMETRY_PID=0
  fi
}

on_term() {
  if (( WORKLOAD_PID > 0 )) && kill -0 "$WORKLOAD_PID" 2>/dev/null; then
    kill -TERM "$WORKLOAD_PID" 2>/dev/null || true
    wait "$WORKLOAD_PID" 2>/dev/null || true
  fi
  cleanup
  write_status stopped stopped "Stopped by user; no automatic retry."
  exit 143
}

worker() {
  GPU=${1:?missing gpu}
  STARTED_AT=${2:?missing start timestamp}
  RUNNER_PID=$$
  trap on_term TERM INT HUP
  trap cleanup EXIT
  cd "$ROOT"
  mkdir -p "$OUTPUT" "$PREDICTIONS"

  write_status running preflight "Checking code, collision-safe IDs, and assigned GPU."
  "$PYTHON" -m py_compile \
    "$ROOT/GRAM/src/main_generative_gram.py" \
    "$ROOT/experiment/phase13/protocol/make_collision_safe_ids.py" \
    || { write_status failed preflight "Python syntax check failed; no automatic retry."; return 4; }
  "$PYTHON" -m unittest experiment/phase13/tests/test_collision_safe_ids.py \
    || { write_status failed preflight "Collision-safe unit tests failed; no automatic retry."; return 5; }
  [[ -s "$ID_FILE" && -s "$ID_REPORT" ]] \
    || { write_status failed preflight "Collision-safe ID artifact missing; no automatic retry."; return 6; }
  "$PYTHON" -c "import json; r=json.load(open('$ID_REPORT')); assert r['output_collision']['duplicate_excess']==0; assert r['warm_ids_unchanged']; assert r['row_order_unchanged']" \
    || { write_status failed preflight "Collision-safe ID report failed invariants; no automatic retry."; return 7; }

  local free_mib
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GPU" 2>/dev/null | tr -d ' ' || true)
  [[ "$free_mib" =~ ^[0-9]+$ ]] \
    || { write_status failed admission "Could not read GPU${GPU} free memory; no automatic retry."; return 8; }
  (( free_mib >= MIN_FREE_MIB )) \
    || { write_status failed admission "GPU${GPU} has ${free_mib} MiB free; requires ${MIN_FREE_MIB}; no automatic retry."; return 9; }

  telemetry & TELEMETRY_PID=$!
  write_status running training "1 epoch / debug_train_100=1 / debug_test_100=1 on user-assigned GPU${GPU}."
  cd "$ROOT/GRAM/command"
  timeout --signal=TERM "$TIMEOUT_SECONDS" env \
    CUDA_VISIBLE_DEVICES="$GPU" \
    HF_HUB_CACHE="$ROOT/.cache/huggingface" \
    TRANSFORMERS_CACHE="$ROOT/.cache/huggingface/transformers" \
    TRANSFORMERS_OFFLINE=1 \
    "$PYTHON" ../src/main_generative_gram.py \
    --datasets Toys_cold50 \
    --distributed 0 --gpu 0 --seed 2023 --train 1 --resource_metrics 1 \
    --log_dir "$OUTPUT/gram_logs" --prediction_dir "$PREDICTIONS" \
    --item_prompt_max_len 128 --item_prompt all_text \
    --cf_model sasrec --id_linking 1 --max_his 20 \
    --rec_batch_size 16 --gradient_accumulation_steps 8 \
    --rec_lr 1e-3 --rec_epochs 1 \
    --test_epoch_rec 0 --save_rec_epochs 1 \
    --save_predictions 1 --beam_size 50 \
    --top_k_similar_item 5 --item_id_type split \
    --hierarchical_id_type hierarchy_v1_c32_l5_len32768_split_v1_mlpcold_collision_safe \
    --debug_train_100 1 --debug_test_100 1 \
    --cf0_arm A --cf0_phase9 0 &
  WORKLOAD_PID=$!
  wait "$WORKLOAD_PID"
  local workload_rc=$?
  WORKLOAD_PID=0
  if (( workload_rc != 0 )); then
    cleanup
    if (( workload_rc == 124 || workload_rc == 143 )); then
      write_status timed_out training "Hard timeout/termination rc=${workload_rc}; no automatic retry."
    else
      write_status failed training "GRAM smoke exited rc=${workload_rc}; no automatic retry."
    fi
    return "$workload_rc"
  fi

  write_status running postflight "Computing cold/warm metrics from smoke test predictions."
  local pred_tsv
  pred_tsv=$(find "$PREDICTIONS" -maxdepth 1 -type f -name '*_pred_test.tsv' -printf '%T@ %p\n' \
    | sort -nr | head -n 1 | cut -d' ' -f2-)
  [[ -n "$pred_tsv" && -s "$pred_tsv" ]] \
    || { cleanup; write_status failed postflight "No non-empty test prediction TSV; no automatic retry."; return 10; }
  "$PYTHON" "$ROOT/experiment/phase13/protocol/eval_cold_warm.py" \
    --dataset-dir "$ROOT/GRAM/rec_datasets/Toys_cold50" \
    --predictions-tsv "$pred_tsv" \
    --output-json "$METRICS" \
    --version-tag "$SUB" --split-name test \
    || { cleanup; write_status failed postflight "eval_cold_warm failed; no automatic retry."; return 11; }

  cleanup
  write_status completed finished "Smoke completed; prediction and cold/warm metrics are available."
}

case "$ACTION" in
  start)
    [[ "$GPU" =~ ^[0-7]$ ]] || { echo "usage: $0 start <gpu 0-7>" >&2; exit 2; }
    tmux has-session -t "$SESSION" 2>/dev/null \
      && { echo "session already exists: $SESSION" >&2; exit 1; }
    if [[ -e "$STATUS" ]]; then
      echo "refusing to reuse existing output/status: $STATUS" >&2
      exit 3
    fi
    STARTED_AT=$(date -Is)
    mkdir -p "$OUTPUT"
    printf -v launch_cmd 'bash %q worker %q %q >> %q 2>&1' \
      "$0" "$GPU" "$STARTED_AT" "$LOG"
    tmux new-session -d -s "$SESSION" "$launch_cmd"
    write_status starting starting "Background smoke session started on user-assigned GPU${GPU}."
    echo "started $SESSION on GPU${GPU}"
    echo "status: bash $0 status"
    ;;
  worker)
    worker "${2:?missing gpu}" "${3:?missing start timestamp}"
    ;;
  status)
    tmux has-session -t "$SESSION" 2>/dev/null \
      && echo "tmux session: running ($SESSION)" \
      || echo "tmux session: not running ($SESSION)"
    [[ -f "$STATUS" ]] && sed -n '1,100p' "$STATUS" || echo '{"status":"not_started"}'
    [[ -f "$METRICS" ]] && echo "--- cold/warm metrics ---" && sed -n '1,120p' "$METRICS"
    [[ -f "$LOG" ]] && echo "--- last 30 log lines ---" && tail -n 30 "$LOG" || true
    ;;
  stop)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      pane_pid=$(tmux list-panes -t "$SESSION" -F '#{pane_pid}' | head -n 1)
      kill -TERM "$pane_pid"
      echo "TERM sent to $SESSION; status file will record stopped state"
    else
      echo "session not running: $SESSION"
    fi
    ;;
  *)
    echo "usage: $0 {start <gpu>|status|stop}" >&2
    exit 2
    ;;
esac
