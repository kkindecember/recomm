#!/usr/bin/env bash
# Phase-13 v1 iter2 P0: BGE prefix-preserving capacity-aware ID assignment.
# Pre-GRAM only. Never changes holders/leases, kills unrelated processes,
# retries, or launches a downstream/formal experiment.
#
# Usage:
#   bash experiment/phase13/run_v1_bge_capacity_assignment_p0.sh start <gpu 0-7>
#   bash experiment/phase13/run_v1_bge_capacity_assignment_p0.sh status
#   bash experiment/phase13/run_v1_bge_capacity_assignment_p0.sh stop
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
ACTION=${1:-status}
GPU=${2:-}

SUB=v1_iter2_bge_capacity_assignment_toys_p0
SESSION=gram_phase13_${SUB}
OUTPUT="$ROOT/artifacts/phase13/explore/$SUB"
LOG="$OUTPUT/run.log"
STATUS="$OUTPUT/status.json"
TELEMETRY="$OUTPUT/gpu_telemetry.csv"
ASSIGN_REPORT="$OUTPUT/assignment_report.json"
SUMMARY="$OUTPUT/screen_summary.json"
EMBED="$ROOT/artifacts/phase13/embeddings/Toys_bge_large_en_v1_5_cls_l2.pt"
MLP_DIR="$ROOT/artifacts/phase13/explore/v1_collision_safe_bge_toys_screen/mlp"
MLP="$MLP_DIR/best.pt"
VOCAB="$MLP_DIR/vocab.json"
RAW_BGE_ID="$ROOT/GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split_v1_bge_mlpcold.txt"
OUTPUT_ID="$ROOT/GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split_v1_bge_capacitycold.txt"
MINILM_ID="$ROOT/GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split_v1_mlpcold.txt"
SOURCE_ID="$ROOT/GRAM/rec_datasets/Toys/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt"
COLD_ITEMS="$ROOT/GRAM/rec_datasets/Toys_cold50/cold_split_meta/cold_items.txt"
MIN_FREE_MIB=4096
EXPECTED_INCREMENTAL_MIB=3072
TIMEOUT_SECONDS=3600

STARTED_AT=""
START_EPOCH=0
RUNNER_PID=0
WORKLOAD_PID=0
TELEMETRY_PID=0
CURRENT_STAGE=not_started
CURRENT_REASON="Not started."

write_status() {
  local state=$1 reason=$2 tmp="${STATUS}.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE13_V1_ITER2_BGE_CAPACITY_ASSIGNMENT_P0","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%d,"workload_pid":%d,"tmux_session":"%s","physical_gpu":%s,"minimum_free_mib":%d,"expected_incremental_gpu_mib_upper_bound":%d,"hard_timeout_seconds":%d,"resource_mode":"user_assigned_direct_no_resource_changes","formal_experiment":false,"gram_training_run":false,"downstream_smoke_run":false,"automatic_next_stage":false,"automatic_retry":false,"prefix_levels_frozen":3,"top_k4":16,"top_k5":16,"log_path":"%s","status_path":"%s","assignment_report_path":"%s","summary_path":"%s","output_id_path":"%s"}\n' \
    "$state" "$CURRENT_STAGE" "$reason" "$STARTED_AT" "$(date -Is)" \
    "$RUNNER_PID" "$WORKLOAD_PID" "$SESSION" "${GPU:--1}" "$MIN_FREE_MIB" \
    "$EXPECTED_INCREMENTAL_MIB" "$TIMEOUT_SECONDS" "${LOG#$ROOT/}" \
    "${STATUS#$ROOT/}" "${ASSIGN_REPORT#$ROOT/}" "${SUMMARY#$ROOT/}" \
    "${OUTPUT_ID#$ROOT/}" > "$tmp"
  mv "$tmp" "$STATUS"
}

telemetry() {
  printf 'timestamp,index,memory_used_mib,memory_free_mib,utilization_gpu_percent\n' > "$TELEMETRY"
  while true; do
    nvidia-smi -i "$GPU" --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits >> "$TELEMETRY" 2>/dev/null || true
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
    WORKLOAD_PID=0
  fi
  cleanup
  CURRENT_STAGE=stopped
  write_status stopped "Stopped by user; no automatic retry or next stage."
  exit 143
}

run_step() {
  local elapsed remaining rc
  elapsed=$(( $(date +%s) - START_EPOCH ))
  remaining=$(( TIMEOUT_SECONDS - elapsed ))
  (( remaining > 0 )) || return 124
  timeout --signal=TERM --kill-after=30 "$remaining" "$@" &
  WORKLOAD_PID=$!
  write_status running "$CURRENT_REASON"
  wait "$WORKLOAD_PID"
  rc=$?
  WORKLOAD_PID=0
  return "$rc"
}

worker() {
  GPU=${1:?missing gpu}
  STARTED_AT=${2:?missing start timestamp}
  START_EPOCH=$(date +%s)
  RUNNER_PID=$$
  trap on_term TERM INT HUP
  trap cleanup EXIT
  cd "$ROOT"

  CURRENT_STAGE=preflight
  CURRENT_REASON="Checking frozen BGE inputs, assignment protocol, isolated outputs, and user-assigned GPU admission."
  write_status running "$CURRENT_REASON"
  "$PYTHON" -m py_compile \
    experiment/phase13/protocol/capacity_aware_assign.py \
    experiment/phase13/protocol/finalize_capacity_assignment_p0.py \
    || { write_status failed "Python syntax check failed; no automatic retry."; return 4; }
  "$PYTHON" -m unittest \
    experiment/phase13/tests/test_capacity_aware_assign.py \
    experiment/phase13/tests/test_semantic_bridge.py \
    experiment/phase13/tests/test_collision_safe_ids.py \
    || { write_status failed "Phase-13 unit tests failed; no automatic retry."; return 5; }
  for path in "$EMBED" "$MLP" "$VOCAB" "$RAW_BGE_ID" "$MINILM_ID" \
    "$SOURCE_ID" "$COLD_ITEMS"; do
    [[ -s "$path" ]] \
      || { write_status failed "Required frozen input missing: ${path#$ROOT/}"; return 6; }
  done
  for path in "$OUTPUT_ID" "$ASSIGN_REPORT" "$SUMMARY"; do
    [[ ! -e "$path" ]] \
      || { write_status failed "Refusing to overwrite existing artifact: ${path#$ROOT/}"; return 7; }
  done
  "$PYTHON" -c "import torch,json; c=torch.load('$MLP',map_location='cpu'); e=torch.load('$EMBED',map_location='cpu'); v=json.load(open('$VOCAB')); assert c['encoder_model']=='BAAI/bge-large-en-v1.5'; assert c['text_dim']==1024; assert c['level_sizes']==[len(x) for x in v['per_level_idx_to_token']]; assert e['model_name']=='BAAI/bge-large-en-v1.5'; assert tuple(e['embeddings'].shape)==(11924,1024)" \
    || { write_status failed "Frozen checkpoint/embedding/vocab compatibility audit failed."; return 8; }
  local free_mib
  free_mib=$(nvidia-smi -i "$GPU" --query-gpu=memory.free \
    --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || true)
  [[ "$free_mib" =~ ^[0-9]+$ ]] \
    || { write_status failed "Could not read GPU${GPU} free memory; no resource changes made."; return 9; }
  (( free_mib >= MIN_FREE_MIB )) \
    || { write_status blocked "GPU${GPU} has ${free_mib} MiB free; requires ${MIN_FREE_MIB}; no resource changes made."; return 10; }

  telemetry & TELEMETRY_PID=$!
  CURRENT_STAGE=capacity_assignment
  CURRENT_REASON="Computing frozen BGE tail logits and exact prefix-3 groupwise unique assignment."
  run_step env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
    experiment/phase13/protocol/capacity_aware_assign.py \
      --embeddings "$EMBED" --mlp "$MLP" --vocab-json "$VOCAB" \
      --raw-assigned-id "$RAW_BGE_ID" --cold-items "$COLD_ITEMS" \
      --output-id "$OUTPUT_ID" --report "$ASSIGN_REPORT" \
      --device cuda:0 --prefix-levels 3 --top-k4 16 --top-k5 16 --batch-size 512 \
    || { cleanup; write_status failed "Capacity-aware assignment failed/timed out; no automatic retry."; return 11; }

  CURRENT_STAGE=p0_gate
  CURRENT_REASON="Applying frozen uniqueness, semantic-retention, and rank-cost P0 gates."
  run_step "$PYTHON" experiment/phase13/protocol/finalize_capacity_assignment_p0.py \
      --assignment-report "$ASSIGN_REPORT" --candidate-id "$OUTPUT_ID" \
      --raw-bge-id "$RAW_BGE_ID" --baseline-minilm-id "$MINILM_ID" \
      --source-id "$SOURCE_ID" --cold-items "$COLD_ITEMS" --output "$SUMMARY" \
    || { cleanup; write_status failed "P0 gate finalization failed; no automatic retry."; return 12; }

  local result
  result=$("$PYTHON" -c "import json,sys; r=json.load(open(sys.argv[1])); c=r['capacity_candidate']; print(f\"{r['verdict']}: prefix4={c['cold_id_metrics']['prefix_accuracy'][3]:.6f}, exact={c['cold_id_metrics']['exact_path_accuracy']:.6f}; no automatic next stage.\")" "$SUMMARY") \
    || { cleanup; write_status failed "Could not read P0 verdict."; return 13; }
  cleanup
  CURRENT_STAGE=finished
  write_status completed "$result"
}

case "$ACTION" in
  start)
    [[ "$GPU" =~ ^[0-7]$ ]] || { echo "usage: $0 start <gpu 0-7>" >&2; exit 2; }
    tmux has-session -t "$SESSION" 2>/dev/null \
      && { echo "session already exists: $SESSION" >&2; exit 1; }
    if [[ -e "$STATUS" ]]; then
      reusable_launch_only=$("$PYTHON" -c "import json,sys; d=json.load(open(sys.argv[1])); print('yes' if d.get('status')=='launch_failed' and int(d.get('runner_pid',0))==0 else 'no')" "$STATUS" 2>/dev/null || echo no)
      [[ "$reusable_launch_only" == yes && ! -e "$OUTPUT_ID" \
        && ! -e "$ASSIGN_REPORT" && ! -e "$SUMMARY" ]] \
        || { echo "refusing to reuse existing output/status: $STATUS" >&2; exit 3; }
    fi
    STARTED_AT=$(date -Is)
    mkdir -p "$OUTPUT"
    printf -v launch_cmd 'bash %q worker %q %q >> %q 2>&1' \
      "$0" "$GPU" "$STARTED_AT" "$LOG"
    write_status starting "Launching background P0 on user-assigned GPU${GPU}; no resource changes or automatic next stage."
    if ! tmux new-session -d -s "$SESSION" "$launch_cmd"; then
      CURRENT_STAGE=launch_failed
      write_status launch_failed "tmux creation failed before worker start; no resource changes were made."
      exit 4
    fi
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
    [[ -f "$STATUS" ]] && sed -n '1,120p' "$STATUS" || echo '{"status":"not_started"}'
    [[ -f "$SUMMARY" ]] && echo "--- P0 summary ---" && sed -n '1,260p' "$SUMMARY"
    [[ -f "$LOG" ]] && echo "--- last 30 log lines ---" && tail -n 30 "$LOG" || true
    ;;
  stop)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      pane_pid=$(tmux list-panes -t "$SESSION" -F '#{pane_pid}' | head -n 1)
      kill -TERM "$pane_pid"
      echo "TERM sent to $SESSION; no automatic retry"
    else
      echo "session not running: $SESSION"
    fi
    ;;
  *)
    echo "usage: $0 {start <gpu 0-7>|status|stop}" >&2
    exit 2
    ;;
esac
