#!/usr/bin/env bash
set -u -o pipefail

# Phase-13 v1-R² Toys validation-only P6 candidate portfolio.
# Usage: bash experiment/phase13/run_v1_r2_toys_p6_candidate_portfolio.sh {start <gpu>|status|stop}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ACTION=${1:-status}
GPU=${2:-}
OUTPUT="$ROOT/artifacts/phase13/explore/v1_r2_toys_p6_candidate_portfolio"
STATUS="$OUTPUT/status.json"
LOG="$OUTPUT/run.log"
TELEMETRY="$OUTPUT/gpu_telemetry.csv"
SUMMARY="$OUTPUT/summary.json"
SESSION=gram_phase13_v1_r2_toys_p6_candidate_portfolio
TMUX_SOCKET=/tmp/gram_phase13_v1_r2_toys_p6_candidate_portfolio.sock
PYTHON=${PYTHON_BIN:-/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python}
MIN_FREE_MIB=${MIN_FREE_MIB:-6144}
EXPECTED_INCREMENTAL_MIB=4096
HARD_TIMEOUT_SECONDS=${HARD_TIMEOUT_SECONDS:-900}

STARTED_AT=""
RUNNER_PID=0
WORKLOAD_PID=0
STAGE=not_started

write_status() {
  local state=$1 reason=$2 tmp="$STATUS.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE13_V1_R2_TOYS_P6_CANDIDATE_PORTFOLIO","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%d,"workload_pid":%d,"tmux_session":"%s","tmux_socket":"%s","physical_gpu":%s,"minimum_free_mib":%d,"expected_incremental_gpu_mib_upper_bound":%d,"hard_timeout_seconds":%d,"resource_mode":"single_gpu_background_no_resource_changes","split":"validation_outer_5fold_oof","test_predictions_opened":false,"automatic_retry":false,"log_path":"%s","status_path":"%s","summary_path":"%s"}\n' \
    "$state" "$STAGE" "$reason" "$STARTED_AT" "$(date -Is)" "$RUNNER_PID" "$WORKLOAD_PID" \
    "$SESSION" "$TMUX_SOCKET" "${GPU:--1}" "$MIN_FREE_MIB" "$EXPECTED_INCREMENTAL_MIB" "$HARD_TIMEOUT_SECONDS" \
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

worker() {
  GPU=${1:?missing gpu}
  STARTED_AT=${2:?missing start timestamp}
  RUNNER_PID=$$
  local rc=0 free_mib telemetry_pid=0
  trap '[[ $WORKLOAD_PID -eq 0 ]] || kill -TERM "$WORKLOAD_PID" 2>/dev/null || true; [[ $telemetry_pid -eq 0 ]] || kill "$telemetry_pid" 2>/dev/null || true; STAGE=stopped; write_status stopped "Stopped by signal; no automatic retry."; exit 143' TERM INT HUP
  STAGE=preflight
  write_status running "Checking frozen validation inputs, P4/P6 tests, and GPU admission."
  bash -n "$ROOT/experiment/phase13/run_v1_r2_toys_p6_candidate_portfolio.sh" \
    || { write_status failed "Bash syntax check failed; no automatic retry."; return 4; }
  "$PYTHON" -m py_compile \
    "$ROOT/experiment/phase13/protocol/candidate_portfolio.py" \
    "$ROOT/experiment/phase13/tests/test_candidate_portfolio.py" \
    || { write_status failed "Python syntax check failed; no automatic retry."; return 5; }
  cd "$ROOT"
  "$PYTHON" -m pytest -q \
    experiment/phase13/tests/test_counterfactual_slot_router.py \
    experiment/phase13/tests/test_candidate_portfolio.py \
    || { write_status failed "P4/P6 unit tests failed; no automatic retry."; return 6; }
  for path in \
    "$ROOT/GRAM/rec_datasets/Toys_cold50/user_sequence.txt" \
    "$ROOT/GRAM/rec_datasets/Toys_cold50/cold_split_meta/cold_items.txt" \
    "$ROOT/GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt" \
    "$ROOT/artifacts/phase13/embeddings/Toys_bge_large_en_v1_5_cls_l2.pt" \
    "$ROOT/artifacts/phase13/explore/v1_r2_toys_p0/resolver.pt" \
    "$ROOT/artifacts/phase13/explore/v1_r2_toys_p0/predictions_validation.jsonl" \
    "$ROOT/artifacts/phase13/explore/v1_r2_toys_p4_counterfactual_slot_router/counterfactual_slot_router.pt" \
    "$ROOT/artifacts/phase13/explore/v1_r2_toys_p4_counterfactual_slot_router/summary.json"; do
    [[ -s "$path" ]] || { write_status failed "Required input missing: ${path#$ROOT/}"; return 7; }
  done
  [[ ! -e "$SUMMARY" && ! -e "$OUTPUT/config.json" && ! -e "$OUTPUT/policy.json" ]] \
    || { write_status failed "Refusing to overwrite an existing P6 scientific artifact."; return 8; }
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GPU" 2>/dev/null | tr -d ' ' || true)
  [[ "$free_mib" =~ ^[0-9]+$ ]] \
    || { write_status blocked "Could not read GPU${GPU} free memory; no resource changes made."; return 9; }
  (( free_mib >= MIN_FREE_MIB )) \
    || { write_status blocked "GPU${GPU} has ${free_mib} MiB free; requires ${MIN_FREE_MIB}; no resource changes made."; return 10; }

  telemetry & telemetry_pid=$!
  STAGE=validation_outer_fold_portfolio_audit
  write_status running "Evaluating held-fold risk-limited 2/3-candidate portfolios on validation only."
  timeout --signal=TERM --kill-after=20 "$HARD_TIMEOUT_SECONDS" \
    env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
      "$ROOT/experiment/phase13/protocol/candidate_portfolio.py" \
      --dataset-dir "$ROOT/GRAM/rec_datasets/Toys_cold50" \
      --p0-predictions "$ROOT/artifacts/phase13/explore/v1_r2_toys_p0/predictions_validation.jsonl" \
      --item-id-file "$ROOT/GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt" \
      --item-embeddings "$ROOT/artifacts/phase13/embeddings/Toys_bge_large_en_v1_5_cls_l2.pt" \
      --resolver-checkpoint "$ROOT/artifacts/phase13/explore/v1_r2_toys_p0/resolver.pt" \
      --p4-checkpoint "$ROOT/artifacts/phase13/explore/v1_r2_toys_p4_counterfactual_slot_router/counterfactual_slot_router.pt" \
      --p4-summary "$ROOT/artifacts/phase13/explore/v1_r2_toys_p4_counterfactual_slot_router/summary.json" \
      --cold-items "$ROOT/GRAM/rec_datasets/Toys_cold50/cold_split_meta/cold_items.txt" \
      --output-dir "$OUTPUT" --device cuda:0 --folds 5 \
      --train-warm-retention 0.99 --max-history 20 --recency-decay 0.85 --seed 42345 &
  WORKLOAD_PID=$!
  write_status running "P6 workload active on GPU${GPU}; existing processes were not modified."
  wait "$WORKLOAD_PID" || rc=$?
  WORKLOAD_PID=0
  kill "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
  telemetry_pid=0
  if (( rc != 0 )); then
    STAGE=finished
    write_status failed "P6 exited rc=${rc}; no automatic retry."
    return "$rc"
  fi
  [[ -s "$SUMMARY" ]] \
    || { STAGE=finished; write_status failed "P6 exited 0 without summary.json."; return 11; }
  local verdict
  verdict=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$SUMMARY") \
    || { STAGE=finished; write_status failed "Could not read P6 verdict."; return 12; }
  STAGE=finished
  write_status completed "$verdict"
}

case "$ACTION" in
  start)
    [[ "$GPU" =~ ^[0-7]$ ]] || { echo "usage: $0 start <gpu 0-7>" >&2; exit 2; }
    mkdir -p "$OUTPUT"
    [[ ! -e "$STATUS" ]] || { echo "refusing to reuse existing status: $STATUS" >&2; exit 3; }
    tmux -S "$TMUX_SOCKET" has-session -t "$SESSION" 2>/dev/null \
      && { echo "session already exists: $SESSION" >&2; exit 3; }
    STARTED_AT=$(date -Is)
    STAGE=starting
    write_status starting "Background P6 session is starting on GPU${GPU}."
    printf -v launch 'bash %q worker %q %q >> %q 2>&1' "$0" "$GPU" "$STARTED_AT" "$LOG"
    if ! tmux -S "$TMUX_SOCKET" new-session -d -s "$SESSION" "$launch"; then
      STAGE=launch_failed
      write_status failed "Could not create background tmux session; workload was not started."
      exit 4
    fi
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
    if tmux -S "$TMUX_SOCKET" has-session -t "$SESSION" 2>/dev/null; then
      tmux -S "$TMUX_SOCKET" send-keys -t "$SESSION" C-c
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
