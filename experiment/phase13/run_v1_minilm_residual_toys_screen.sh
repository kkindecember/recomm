#!/usr/bin/env bash
# Pre-GRAM screen for MiniLM + two-layer residual semantic bridge on Toys.
# It never starts/stops protectors, leases, or other users' processes.
#
# Usage:
#   bash experiment/phase13/run_v1_minilm_residual_toys_screen.sh start 6
#   bash experiment/phase13/run_v1_minilm_residual_toys_screen.sh status
#   bash experiment/phase13/run_v1_minilm_residual_toys_screen.sh stop
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
ACTION=${1:-status}
GPU=${2:-}

SUB=v1_collision_safe_minilm_residual_toys_screen
SESSION=gram_phase13_${SUB}
OUTPUT="$ROOT/artifacts/phase13/explore/$SUB"
LOG="$OUTPUT/run.log"
STATUS="$OUTPUT/status.json"
TELEMETRY="$OUTPUT/gpu_telemetry.csv"
SUMMARY="$OUTPUT/screen_summary.json"
EMBED="$ROOT/artifacts/phase13/embeddings/Toys_sbert.pt"
MLP_DIR="$OUTPUT/mlp"
MLP="$MLP_DIR/best.pt"
HISTORY="$MLP_DIR/training_history.json"
ASSIGN_REPORT="$OUTPUT/assign_report.json"
ID_REPORT="$OUTPUT/id_report.json"
SOURCE_ID="$ROOT/GRAM/rec_datasets/Toys/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt"
COLD_ITEMS="$ROOT/GRAM/rec_datasets/Toys_cold50/cold_split_meta/cold_items.txt"
ASSIGNED_ID="$ROOT/GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split_v1_minilm_residual2cold.txt"
SAFE_ID="$ROOT/GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split_v1_minilm_residual2cold_collision_safe.txt"
BASELINE_HISTORY="$ROOT/artifacts/phase13/explore/v1_toys/mlp/training_history.json"
BASELINE_ASSIGNED_ID="$ROOT/GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split_v1_mlpcold.txt"
BASELINE_ID_REPORT="$ROOT/artifacts/phase13/explore/v1_collision_safe/toys_id_report.json"
MIN_FREE_MIB=${MIN_FREE_MIB:-4096}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-3600}

STARTED_AT=""
RUNNER_PID=0
WORKLOAD_PID=0
TELEMETRY_PID=0
CURRENT_STAGE=not_started
CURRENT_REASON="Not started."

write_status() {
  local state=$1 reason=$2 tmp="${STATUS}.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE13_V1_MINILM_RESIDUAL_TOYS_SCREEN","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%d,"workload_pid":%d,"tmux_session":"%s","physical_gpu":%s,"minimum_free_mib":%d,"expected_incremental_gpu_mib_upper_bound":3072,"hard_timeout_seconds":%d,"resource_mode":"user_assigned_direct_no_resource_changes","formal_experiment":false,"gram_training_run":false,"log_path":"%s","status_path":"%s","summary_path":"%s"}\n' \
    "$state" "$CURRENT_STAGE" "$reason" "$STARTED_AT" "$(date -Is)" \
    "$RUNNER_PID" "$WORKLOAD_PID" "$SESSION" "${GPU:--1}" "$MIN_FREE_MIB" \
    "$TIMEOUT_SECONDS" "${LOG#$ROOT/}" "${STATUS#$ROOT/}" "${SUMMARY#$ROOT/}" > "$tmp"
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
  write_status stopped "Stopped by user; no automatic retry."
  exit 143
}

run_step() {
  timeout --signal=TERM "$TIMEOUT_SECONDS" "$@" &
  WORKLOAD_PID=$!
  write_status running "$CURRENT_REASON"
  wait "$WORKLOAD_PID"
  local rc=$?
  WORKLOAD_PID=0
  return "$rc"
}

worker() {
  GPU=${1:?missing gpu}
  STARTED_AT=${2:?missing start timestamp}
  RUNNER_PID=$$
  trap on_term TERM INT HUP
  trap cleanup EXIT
  cd "$ROOT"

  CURRENT_STAGE=preflight
  CURRENT_REASON="Checking residual code, frozen MiniLM inputs, isolated outputs, and GPU admission."
  write_status running "$CURRENT_REASON"
  "$PYTHON" -m py_compile \
    experiment/phase13/protocol/semantic_bridge_residual.py \
    experiment/phase13/protocol/assign_cold_ids.py \
    experiment/phase13/protocol/make_collision_safe_ids.py \
    experiment/phase13/protocol/summarize_residual_mlp_screen.py \
    || { write_status failed "Python syntax check failed; no automatic retry."; return 4; }
  "$PYTHON" -m unittest \
    experiment/phase13/tests/test_semantic_bridge.py \
    experiment/phase13/tests/test_collision_safe_ids.py \
    || { write_status failed "Phase-13 unit tests failed; no automatic retry."; return 5; }
  for path in "$EMBED" "$SOURCE_ID" "$COLD_ITEMS" "$BASELINE_HISTORY" \
    "$BASELINE_ASSIGNED_ID" "$BASELINE_ID_REPORT"; do
    [[ -s "$path" ]] \
      || { write_status failed "Required frozen input missing: ${path#$ROOT/}"; return 6; }
  done
  for path in "$MLP" "$HISTORY" "$ASSIGNED_ID" "$SAFE_ID" "$ASSIGN_REPORT" \
    "$ID_REPORT" "$SUMMARY"; do
    [[ ! -e "$path" ]] \
      || { write_status failed "Refusing to overwrite existing artifact: ${path#$ROOT/}"; return 7; }
  done
  "$PYTHON" -c "import torch; p=torch.load('$EMBED',map_location='cpu'); e=p['embeddings']; assert e.shape==(11924,384); assert 'all-MiniLM-L6-v2' in p['model_name']; assert torch.isfinite(e).all()" \
    || { write_status failed "Frozen MiniLM embedding audit failed; no automatic retry."; return 8; }
  local free_mib
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GPU" 2>/dev/null | tr -d ' ' || true)
  [[ "$free_mib" =~ ^[0-9]+$ ]] \
    || { write_status failed "Could not read GPU${GPU} free memory; no automatic retry."; return 9; }
  (( free_mib >= MIN_FREE_MIB )) \
    || { write_status blocked "GPU${GPU} has ${free_mib} MiB free; requires ${MIN_FREE_MIB}; no resource changes made."; return 10; }

  telemetry & TELEMETRY_PID=$!

  CURRENT_STAGE=residual_bridge_300
  CURRENT_REASON="Training MiniLM 384-768-384 residual bridge from scratch for 300 epochs; no GRAM run."
  run_step env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
    experiment/phase13/protocol/semantic_bridge_residual.py train \
      --embeddings "$EMBED" --id-file "$SOURCE_ID" --cold-items "$COLD_ITEMS" \
      --output-dir "$MLP_DIR" --hidden-dim 768 --epochs 300 --lr 1e-3 \
      --batch-size 512 --device cuda:0 --seed 12345 \
    || { write_status failed "Residual bridge screen failed/timed out; no automatic retry."; return 11; }

  CURRENT_STAGE=id_assignment
  CURRENT_REASON="Assigning cold IDs with the best residual checkpoint."
  run_step env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
    experiment/phase13/protocol/assign_cold_ids.py \
      --embeddings "$EMBED" --mlp "$MLP" --source-id-file "$SOURCE_ID" \
      --cold-items "$COLD_ITEMS" --output-id-file "$ASSIGNED_ID" \
      --report "$ASSIGN_REPORT" --device cuda:0 \
    || { write_status failed "Residual cold-ID assignment failed/timed out; no automatic retry."; return 12; }

  CURRENT_STAGE=collision_audit
  CURRENT_REASON="Computing raw collisions and verifying collision-safe residual IDs."
  run_step "$PYTHON" experiment/phase13/protocol/make_collision_safe_ids.py \
      --input-id-file "$ASSIGNED_ID" --cold-items "$COLD_ITEMS" \
      --output-id-file "$SAFE_ID" --report "$ID_REPORT" \
    || { write_status failed "Residual collision audit failed/timed out; no automatic retry."; return 13; }

  CURRENT_STAGE=decision
  CURRENT_REASON="Applying frozen validation, cold-prefix/path, and collision gates."
  run_step "$PYTHON" experiment/phase13/protocol/summarize_residual_mlp_screen.py \
      --candidate-history "$HISTORY" --candidate-assigned-id "$ASSIGNED_ID" \
      --candidate-id-report "$ID_REPORT" --baseline-history "$BASELINE_HISTORY" \
      --baseline-assigned-id "$BASELINE_ASSIGNED_ID" \
      --baseline-id-report "$BASELINE_ID_REPORT" --source-id "$SOURCE_ID" \
      --cold-items "$COLD_ITEMS" --output "$SUMMARY" \
    || { write_status failed "Residual screen summary failed; no automatic retry."; return 14; }

  local result
  result=$("$PYTHON" -c "import json; r=json.load(open('$SUMMARY')); print(f\"{r['verdict']}: best_val={r['candidate']['best_val_avg_acc']:.6f}, gate={r['gate']['minimum_val_avg_acc']:.6f}\")") \
    || { write_status failed "Could not read completed residual summary."; return 15; }
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
      echo "refusing to reuse existing output/status: $STATUS" >&2
      exit 3
    fi
    STARTED_AT=$(date -Is)
    mkdir -p "$OUTPUT"
    printf -v launch_cmd 'bash %q worker %q %q >> %q 2>&1' \
      "$0" "$GPU" "$STARTED_AT" "$LOG"
    tmux new-session -d -s "$SESSION" "$launch_cmd"
    write_status starting "Background MiniLM residual pre-GRAM screen started on user-assigned GPU${GPU}."
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
