#!/usr/bin/env bash
set -u -o pipefail

# Phase-13 v1-R² P4 frozen, one-shot disjoint Toys test confirmation.
# Usage: bash experiment/phase13/run_v1_r2_toys_p4_disjoint_confirmation.sh {start <gpu>|status|stop}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ACTION=${1:-status}
GPU=${2:-}
OUTPUT="$ROOT/artifacts/phase13/explore/v1_r2_toys_p4_disjoint_confirmation"
STATUS="$OUTPUT/status.json"
LOG="$OUTPUT/run.log"
TELEMETRY="$OUTPUT/gpu_telemetry.csv"
SUMMARY="$OUTPUT/summary.json"
TEST_ACCESS="$OUTPUT/test_access.json"
SESSION=gram_phase13_v1_r2_toys_p4_disjoint_confirmation
PYTHON=${PYTHON_BIN:-/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python}
MIN_FREE_MIB=${MIN_FREE_MIB:-3072}
EXPECTED_INCREMENTAL_MIB=2048
HARD_TIMEOUT_SECONDS=${HARD_TIMEOUT_SECONDS:-600}
TEST_PREDICTIONS="$ROOT/artifacts/phase13/explore/v0_toys/predictions/20260809_091709_Toys_cold50_sequential_pred_test.tsv"
PREVIOUS_MANIFEST="$ROOT/artifacts/phase13/explore/v1_r2_toys_p3_fresh_medium_smoke/selection_manifest.json"

STARTED_AT=""
RUNNER_PID=0
WORKLOAD_PID=0
STAGE=not_started

test_opened() {
  [[ -s "$TEST_ACCESS" ]] || { printf false; return; }
  "$PYTHON" -c 'import json,sys; print("true" if json.load(open(sys.argv[1])).get("test_predictions_opened") else "false")' "$TEST_ACCESS" 2>/dev/null || printf false
}

write_status() {
  local state=$1 reason=$2 tmp="$STATUS.tmp.$$" opened
  mkdir -p "$OUTPUT"
  opened=$(test_opened)
  printf '{"experiment_id":"GRAM_PHASE13_V1_R2_TOYS_P4_DISJOINT_CONFIRMATION","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%d,"workload_pid":%d,"tmux_session":"%s","physical_gpu":%s,"minimum_free_mib":%d,"expected_incremental_gpu_mib_upper_bound":%d,"hard_timeout_seconds":%d,"resource_mode":"small_gpu_direct_no_resource_changes","split":"test_hash_tranche_ranks_1001_2000_one_shot","test_predictions_opened":%s,"automatic_retry":false,"log_path":"%s","status_path":"%s","summary_path":"%s"}\n' \
    "$state" "$STAGE" "$reason" "$STARTED_AT" "$(date -Is)" "$RUNNER_PID" "$WORKLOAD_PID" \
    "$SESSION" "${GPU:--1}" "$MIN_FREE_MIB" "$EXPECTED_INCREMENTAL_MIB" "$HARD_TIMEOUT_SECONDS" "$opened" \
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
  write_status running "Checking frozen P4 inputs, disjoint-selection tests, and GPU admission; tranche fields remain unopened."
  bash -n "$ROOT/experiment/phase13/run_v1_r2_toys_p4_disjoint_confirmation.sh" \
    || { write_status failed "Bash syntax check failed; no automatic retry."; return 4; }
  "$PYTHON" -m py_compile \
    "$ROOT/experiment/phase13/protocol/p4_disjoint_confirmation.py" \
    "$ROOT/experiment/phase13/tests/test_p4_disjoint_confirmation.py" \
    || { write_status failed "Python syntax check failed; no automatic retry."; return 5; }
  cd "$ROOT"
  "$PYTHON" -m pytest -q \
    experiment/phase13/tests/test_counterfactual_slot_router.py \
    experiment/phase13/tests/test_fresh_medium_smoke.py \
    experiment/phase13/tests/test_p4_disjoint_confirmation.py \
    || { write_status failed "P4/disjoint unit tests failed; no automatic retry."; return 6; }
  for path in \
    "$ROOT/GRAM/rec_datasets/Toys_cold50/user_sequence.txt" \
    "$ROOT/GRAM/rec_datasets/Toys_cold50/cold_split_meta/cold_items.txt" \
    "$ROOT/GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt" \
    "$ROOT/artifacts/phase13/embeddings/Toys_bge_large_en_v1_5_cls_l2.pt" \
    "$ROOT/artifacts/phase13/explore/v1_r2_toys_p0/resolver.pt" \
    "$ROOT/artifacts/phase13/explore/v1_r2_toys_p4_counterfactual_slot_router/counterfactual_slot_router.pt" \
    "$PREVIOUS_MANIFEST" \
    "$TEST_PREDICTIONS"; do
    [[ -s "$path" ]] || { write_status failed "Required input missing: ${path#$ROOT/}"; return 7; }
  done
  [[ ! -e "$SUMMARY" && ! -e "$OUTPUT/config.json" && ! -e "$OUTPUT/selection_manifest.json" && ! -e "$TEST_ACCESS" ]] \
    || { write_status failed "Refusing to overwrite an existing one-shot scientific artifact."; return 8; }
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GPU" 2>/dev/null | tr -d ' ' || true)
  [[ "$free_mib" =~ ^[0-9]+$ ]] \
    || { write_status blocked "Could not read GPU${GPU} free memory; no resource changes made."; return 9; }
  (( free_mib >= MIN_FREE_MIB )) \
    || { write_status blocked "GPU${GPU} has ${free_mib} MiB free; requires ${MIN_FREE_MIB}; no resource changes made."; return 10; }

  telemetry & telemetry_pid=$!
  STAGE=one_shot_disjoint_test_confirmation
  write_status running "Writing zero-overlap tranche manifest, then performing authorized one-shot P4 confirmation on GPU${GPU}."
  timeout --signal=TERM --kill-after=15 "$HARD_TIMEOUT_SECONDS" \
    env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
      "$ROOT/experiment/phase13/protocol/p4_disjoint_confirmation.py" \
      --dataset-dir "$ROOT/GRAM/rec_datasets/Toys_cold50" \
      --gram-test-predictions "$TEST_PREDICTIONS" \
      --item-id-file "$ROOT/GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt" \
      --item-embeddings "$ROOT/artifacts/phase13/embeddings/Toys_bge_large_en_v1_5_cls_l2.pt" \
      --resolver-checkpoint "$ROOT/artifacts/phase13/explore/v1_r2_toys_p0/resolver.pt" \
      --p4-checkpoint "$ROOT/artifacts/phase13/explore/v1_r2_toys_p4_counterfactual_slot_router/counterfactual_slot_router.pt" \
      --previous-selection-manifest "$PREVIOUS_MANIFEST" \
      --cold-items "$ROOT/GRAM/rec_datasets/Toys_cold50/cold_split_meta/cold_items.txt" \
      --output-dir "$OUTPUT" --device cuda:0 --tranche-start 1000 --sample-size 1000 \
      --frozen-utility-threshold 0.06854262202978134 \
      --max-history 20 --recency-decay 0.85 &
  WORKLOAD_PID=$!
  write_status running "P4 disjoint workload active; existing GPU processes were not modified."
  wait "$WORKLOAD_PID" || rc=$?
  WORKLOAD_PID=0
  kill "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
  telemetry_pid=0
  if (( rc != 0 )); then
    STAGE=finished
    write_status failed "P4 disjoint confirmation exited rc=${rc}; no automatic retry and no test tuning."
    return "$rc"
  fi
  [[ -s "$SUMMARY" ]] \
    || { STAGE=finished; write_status failed "Workload exited 0 without summary.json."; return 11; }
  local verdict
  verdict=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$SUMMARY") \
    || { STAGE=finished; write_status failed "Could not read disjoint-confirmation verdict."; return 12; }
  STAGE=finished
  write_status completed "$verdict"
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
    write_status starting "Background P4 disjoint confirmation is starting on GPU${GPU}; tranche fields remain unopened."
    printf -v launch 'bash %q worker %q %q >> %q 2>&1' "$0" "$GPU" "$STARTED_AT" "$LOG"
    if ! tmux new-session -d -s "$SESSION" "$launch"; then
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
