#!/usr/bin/env bash
# Screen whether the already-prepared E5 embedding can train a competitive
# semantic bridge. This does not train/evaluate GRAM and never changes GPU
# protectors, leases, or other users' processes.
#
# Usage:
#   bash experiment/phase13/run_v1_e5_toys_mlp_convergence.sh start 6
#   bash experiment/phase13/run_v1_e5_toys_mlp_convergence.sh status
#   bash experiment/phase13/run_v1_e5_toys_mlp_convergence.sh stop
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
ACTION=${1:-status}
GPU=${2:-}

SUB=v1_collision_safe_e5_toys_mlp400
SESSION=gram_phase13_${SUB}
OUTPUT="$ROOT/artifacts/phase13/explore/$SUB"
LOG="$OUTPUT/run.log"
STATUS="$OUTPUT/status.json"
TELEMETRY="$OUTPUT/gpu_telemetry.csv"
EMBED="$ROOT/artifacts/phase13/embeddings/Toys_e5_large_v2_query_mean_l2.pt"
MLP_DIR="$OUTPUT/mlp"
MLP="$MLP_DIR/best.pt"
HISTORY="$MLP_DIR/training_history.json"
ASSIGN_REPORT="$OUTPUT/assign_report.json"
ID_REPORT="$OUTPUT/id_report.json"
SUMMARY="$OUTPUT/screen_summary.json"
SOURCE_ID="$ROOT/GRAM/rec_datasets/Toys/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt"
COLD_ITEMS="$ROOT/GRAM/rec_datasets/Toys_cold50/cold_split_meta/cold_items.txt"
ASSIGNED_ID="$ROOT/GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split_v1_e5_mlp400cold.txt"
SAFE_ID="$ROOT/GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split_v1_e5_mlp400cold_collision_safe.txt"
MINILM_HISTORY="$ROOT/artifacts/phase13/explore/v1_toys/mlp/training_history.json"
MINILM_ID_REPORT="$ROOT/artifacts/phase13/explore/v1_collision_safe/toys_id_report.json"
PREVIOUS_E5_HISTORY="$ROOT/artifacts/phase13/explore/v1_collision_safe_e5_toys_prep/mlp/training_history.json"
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
  printf '{"experiment_id":"GRAM_PHASE13_V1_E5_TOYS_MLP400_SCREEN","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%d,"workload_pid":%d,"tmux_session":"%s","physical_gpu":%s,"minimum_free_mib":%d,"expected_incremental_gpu_mib_upper_bound":1024,"hard_timeout_seconds":%d,"resource_mode":"user_assigned_direct_no_resource_changes","formal_experiment":false,"gram_training_run":false,"log_path":"%s","status_path":"%s","summary_path":"%s"}\n' \
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
  CURRENT_REASON="Checking code, frozen inputs, isolated outputs, and GPU6 admission."
  write_status running "$CURRENT_REASON"
  "$PYTHON" -m py_compile \
    experiment/phase13/protocol/semantic_bridge.py \
    experiment/phase13/protocol/assign_cold_ids.py \
    experiment/phase13/protocol/make_collision_safe_ids.py \
    experiment/phase13/protocol/summarize_e5_mlp_screen.py \
    || { write_status failed "Python syntax check failed; no automatic retry."; return 4; }
  "$PYTHON" -m unittest \
    experiment/phase13/tests/test_semantic_bridge.py \
    experiment/phase13/tests/test_collision_safe_ids.py \
    || { write_status failed "Phase-13 unit tests failed; no automatic retry."; return 5; }
  for path in "$EMBED" "$SOURCE_ID" "$COLD_ITEMS" "$MINILM_HISTORY" \
    "$MINILM_ID_REPORT" "$PREVIOUS_E5_HISTORY"; do
    [[ -s "$path" ]] \
      || { write_status failed "Required frozen input missing: ${path#$ROOT/}"; return 6; }
  done
  for path in "$MLP" "$HISTORY" "$ASSIGNED_ID" "$SAFE_ID" "$ASSIGN_REPORT" \
    "$ID_REPORT" "$SUMMARY"; do
    [[ ! -e "$path" ]] \
      || { write_status failed "Refusing to overwrite existing artifact: ${path#$ROOT/}"; return 7; }
  done
  "$PYTHON" -c "import torch; p=torch.load('$EMBED',map_location='cpu'); e=p['embeddings']; assert e.shape==(11924,1024); assert p['model_name']=='intfloat/e5-large-v2'; assert p['text_prefix']=='query: '; assert p['l2_normalized'] is True; assert torch.isfinite(e).all()" \
    || { write_status failed "Frozen E5 embedding audit failed; no automatic retry."; return 8; }
  local free_mib
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GPU" 2>/dev/null | tr -d ' ' || true)
  [[ "$free_mib" =~ ^[0-9]+$ ]] \
    || { write_status failed "Could not read GPU${GPU} free memory; no automatic retry."; return 9; }
  (( free_mib >= MIN_FREE_MIB )) \
    || { write_status blocked "GPU${GPU} has ${free_mib} MiB free; requires ${MIN_FREE_MIB}; no resource changes made."; return 10; }

  telemetry & TELEMETRY_PID=$!

  CURRENT_STAGE=semantic_bridge_400
  CURRENT_REASON="Training E5 Semantic Bridge from scratch for 400 epochs; no GRAM run."
  run_step env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
    experiment/phase13/protocol/semantic_bridge.py train \
      --embeddings "$EMBED" --id-file "$SOURCE_ID" --cold-items "$COLD_ITEMS" \
      --output-dir "$MLP_DIR" --epochs 400 --lr 1e-3 --batch-size 512 \
      --device cuda:0 --seed 12345 \
    || { write_status failed "Semantic Bridge screen failed/timed out; no automatic retry."; return 11; }

  CURRENT_STAGE=id_assignment
  CURRENT_REASON="Assigning cold IDs with the best MLP checkpoint for collision diagnostics."
  run_step env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
    experiment/phase13/protocol/assign_cold_ids.py \
      --embeddings "$EMBED" --mlp "$MLP" --source-id-file "$SOURCE_ID" \
      --cold-items "$COLD_ITEMS" --output-id-file "$ASSIGNED_ID" \
      --report "$ASSIGN_REPORT" --device cuda:0 \
    || { write_status failed "Cold-ID assignment failed/timed out; no automatic retry."; return 12; }

  CURRENT_STAGE=collision_audit
  CURRENT_REASON="Computing raw collisions and verifying globally unique collision-safe IDs."
  run_step "$PYTHON" experiment/phase13/protocol/make_collision_safe_ids.py \
      --input-id-file "$ASSIGNED_ID" --cold-items "$COLD_ITEMS" \
      --output-id-file "$SAFE_ID" --report "$ID_REPORT" \
    || { write_status failed "Collision audit failed/timed out; no automatic retry."; return 13; }

  CURRENT_STAGE=decision
  CURRENT_REASON="Applying the frozen 95%-of-MiniLM convergence gate."
  run_step "$PYTHON" experiment/phase13/protocol/summarize_e5_mlp_screen.py \
      --e5-history "$HISTORY" --e5-id-report "$ID_REPORT" \
      --minilm-history "$MINILM_HISTORY" --minilm-id-report "$MINILM_ID_REPORT" \
      --previous-e5-history "$PREVIOUS_E5_HISTORY" --output "$SUMMARY" \
    || { write_status failed "Screen summary/invariant audit failed; no automatic retry."; return 14; }

  local result
  result=$("$PYTHON" -c "import json; r=json.load(open('$SUMMARY')); print(f\"{r['verdict']}: best_val={r['e5_result']['best_val_avg_acc']:.6f}, gate={r['gate']['minimum_val_avg_acc']:.6f}\")") \
    || { write_status failed "Could not read completed screen summary."; return 15; }
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
    write_status starting "Background E5 MLP convergence screen started on user-assigned GPU${GPU}."
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
