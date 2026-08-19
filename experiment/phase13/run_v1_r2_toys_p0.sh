#!/usr/bin/env bash
set -u -o pipefail

# Phase-13 v1-R² Toys validation-only P0.
#
# Usage:
#   bash experiment/phase13/run_v1_r2_toys_p0.sh start <physical_gpu>
#   bash experiment/phase13/run_v1_r2_toys_p0.sh status
#   bash experiment/phase13/run_v1_r2_toys_p0.sh stop

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ACTION=${1:-status}
GPU=${2:-}
OUTPUT="$ROOT/artifacts/phase13/explore/v1_r2_toys_p0"
STATUS="$OUTPUT/status.json"
LOG="$OUTPUT/run.log"
TELEMETRY="$OUTPUT/gpu_telemetry.csv"
SUMMARY="$OUTPUT/summary.json"
SESSION=gram_phase13_v1_r2_toys_p0
PYTHON=${PYTHON_BIN:-/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python}
MIN_FREE_MIB=${MIN_FREE_MIB:-6144}
EXPECTED_INCREMENTAL_MIB=3072
HARD_TIMEOUT_SECONDS=${HARD_TIMEOUT_SECONDS:-3600}

STARTED_AT=""
RUNNER_PID=0
WORKLOAD_PID=0
STAGE=not_started
REASON="not started"

write_status() {
  local state=$1 reason=$2 tmp="$STATUS.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE13_V1_R2_TOYS_P0","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%d,"workload_pid":%d,"tmux_session":"%s","physical_gpu":%s,"minimum_free_mib":%d,"expected_incremental_gpu_mib_upper_bound":%d,"hard_timeout_seconds":%d,"resource_mode":"user_assigned_direct_no_resource_changes","split":"validation","test_predictions_opened":false,"log_path":"%s","status_path":"%s","summary_path":"%s"}\n' \
    "$state" "$STAGE" "$reason" "$STARTED_AT" "$(date -Is)" "$RUNNER_PID" "$WORKLOAD_PID" \
    "$SESSION" "${GPU:--1}" "$MIN_FREE_MIB" "$EXPECTED_INCREMENTAL_MIB" "$HARD_TIMEOUT_SECONDS" \
    "${LOG#$ROOT/}" "${STATUS#$ROOT/}" "${SUMMARY#$ROOT/}" > "$tmp"
  mv "$tmp" "$STATUS"
}

telemetry() {
  printf 'timestamp,index,memory_used_mib,memory_free_mib,utilization_gpu_percent\n' > "$TELEMETRY"
  while true; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits --id="$GPU" >> "$TELEMETRY" 2>/dev/null || true
    sleep 10
  done
}

latest_validation_prediction() {
  find "$ROOT/artifacts/phase13/explore/v0_toys/predictions" -maxdepth 1 -type f \
    -name '*_pred_validation.tsv' -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-
}

worker() {
  GPU=${1:?missing gpu}
  STARTED_AT=${2:?missing start timestamp}
  RUNNER_PID=$$
  local telemetry_pid=0 rc=0 free_mib prediction
  trap '[[ $WORKLOAD_PID -eq 0 ]] || kill -TERM "$WORKLOAD_PID" 2>/dev/null || true; [[ $telemetry_pid -eq 0 ]] || kill "$telemetry_pid" 2>/dev/null || true; STAGE=stopped; write_status stopped "Stopped by signal; no automatic retry."; exit 143' TERM INT HUP

  STAGE=preflight
  write_status running "Checking code, frozen validation inputs, and GPU admission."
  bash -n "$ROOT/experiment/phase13/run_v1_r2_toys_p0.sh" \
    || { write_status failed "Bash syntax check failed; no automatic retry."; return 4; }
  "$PYTHON" -m py_compile \
    "$ROOT/experiment/phase13/protocol/route_resolve.py" \
    "$ROOT/experiment/phase13/tests/test_route_resolve.py" \
    || { write_status failed "Python syntax check failed; no automatic retry."; return 5; }
  cd "$ROOT"
  "$PYTHON" -m pytest -q experiment/phase13/tests/test_route_resolve.py \
    || { write_status failed "R² unit tests failed; no automatic retry."; return 6; }

  prediction=$(latest_validation_prediction)
  [[ -n "$prediction" && -s "$prediction" ]] \
    || { write_status failed "Frozen v0 Toys validation prediction missing."; return 7; }
  for path in \
    "$ROOT/GRAM/rec_datasets/Toys_cold50/user_sequence.txt" \
    "$ROOT/GRAM/rec_datasets/Toys_cold50/cold_split_meta/cold_items.txt" \
    "$ROOT/GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt" \
    "$ROOT/artifacts/phase13/embeddings/Toys_bge_large_en_v1_5_cls_l2.pt"; do
    [[ -s "$path" ]] || { write_status failed "Required input missing: ${path#$ROOT/}"; return 8; }
  done
  [[ ! -e "$SUMMARY" && ! -e "$OUTPUT/config.json" && ! -e "$OUTPUT/resolver.pt" ]] \
    || { write_status failed "Refusing to overwrite an existing R² scientific artifact."; return 9; }

  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GPU" 2>/dev/null | tr -d ' ' || true)
  [[ "$free_mib" =~ ^[0-9]+$ ]] \
    || { write_status blocked "Could not read GPU${GPU} free memory; no resource changes made."; return 10; }
  (( free_mib >= MIN_FREE_MIB )) \
    || { write_status blocked "GPU${GPU} has ${free_mib} MiB free; requires ${MIN_FREE_MIB}; no resource changes made."; return 11; }

  telemetry & telemetry_pid=$!
  STAGE=train_and_validation_eval
  write_status running "Training warm-only resolver and evaluating fixed R² on Toys validation."
  timeout --signal=TERM --kill-after=30 "$HARD_TIMEOUT_SECONDS" \
    env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
      "$ROOT/experiment/phase13/protocol/route_resolve.py" \
      --dataset-dir "$ROOT/GRAM/rec_datasets/Toys_cold50" \
      --item-id-file "$ROOT/GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt" \
      --item-embeddings "$ROOT/artifacts/phase13/embeddings/Toys_bge_large_en_v1_5_cls_l2.pt" \
      --gram-validation-predictions "$prediction" \
      --output-dir "$OUTPUT" --device cuda:0 \
      --route-depth 3 --max-history 20 --epochs 12 --batch-size 256 \
      --hidden-dim 512 --dropout 0.1 --lr 1e-3 --weight-decay 1e-4 \
      --temperature 0.07 --recency-decay 0.85 \
      --global-retrieve-k 200 --top-routes 8 --per-route-k 50 \
      --rrf-k 60 --route-prior-weight 0.25 --seed 12345 &
  WORKLOAD_PID=$!
  write_status running "R² workload active on GPU${GPU}; existing processes were not modified."
  wait "$WORKLOAD_PID" || rc=$?
  WORKLOAD_PID=0
  kill "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
  telemetry_pid=0
  if (( rc != 0 )); then
    STAGE=finished
    write_status failed "R² P0 exited rc=${rc}; no automatic retry."
    return "$rc"
  fi
  [[ -s "$SUMMARY" ]] \
    || { STAGE=finished; write_status failed "R² process exited 0 without summary.json."; return 12; }
  local verdict
  verdict=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$SUMMARY") \
    || { STAGE=finished; write_status failed "Could not read R² verdict."; return 13; }
  STAGE=finished
  write_status completed "$verdict"
  return 0
}

case "$ACTION" in
  start)
    [[ "$GPU" =~ ^[0-7]$ ]] || { echo "usage: $0 start <gpu 0-7>" >&2; exit 2; }
    mkdir -p "$OUTPUT"
    [[ ! -e "$STATUS" ]] || { echo "refusing to reuse existing status: $STATUS" >&2; exit 3; }
    tmux has-session -t "$SESSION" 2>/dev/null \
      && { echo "session already exists: $SESSION" >&2; exit 3; }
    STARTED_AT=$(date -Is)
    STAGE=starting
    write_status starting "Background R² P0 session is starting on GPU${GPU}."
    printf -v launch 'bash %q worker %q %q >> %q 2>&1' "$0" "$GPU" "$STARTED_AT" "$LOG"
    tmux new-session -d -s "$SESSION" "$launch"
    echo "started $SESSION on GPU${GPU}"
    echo "status: $STATUS"
    ;;
  worker)
    worker "${2:?missing gpu}" "${3:?missing start timestamp}"
    ;;
  status)
    [[ -f "$STATUS" ]] && sed -n '1,100p' "$STATUS" || echo '{"status":"not_started"}'
    ;;
  stop)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      tmux send-keys -t "$SESSION" C-c
      echo "stop requested; status file will record the outcome"
    else
      echo "session not running: $SESSION"
    fi
    ;;
  *)
    echo "usage: $0 {start <gpu>|status|stop}" >&2
    exit 2
    ;;
esac
