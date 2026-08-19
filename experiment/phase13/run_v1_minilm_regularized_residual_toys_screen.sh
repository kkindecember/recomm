#!/usr/bin/env bash
# Three-arm warm-only regularization selection, followed by winner-only cold audit.
# No GRAM run and no protector/lease/process changes.
#
# Usage:
#   bash experiment/phase13/run_v1_minilm_regularized_residual_toys_screen.sh start 6
#   bash experiment/phase13/run_v1_minilm_regularized_residual_toys_screen.sh status
#   bash experiment/phase13/run_v1_minilm_regularized_residual_toys_screen.sh stop
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
ACTION=${1:-status}
GPU=${2:-}

SUB=v1_collision_safe_minilm_regularized_residual_toys_screen
SESSION=gram_phase13_${SUB}
OUTPUT="$ROOT/artifacts/phase13/explore/$SUB"
LOG="$OUTPUT/run.log"
STATUS="$OUTPUT/status.json"
TELEMETRY="$OUTPUT/gpu_telemetry.csv"
SUMMARY="$OUTPUT/screen_summary.json"
EMBED="$ROOT/artifacts/phase13/embeddings/Toys_sbert.pt"
SOURCE_ID="$ROOT/GRAM/rec_datasets/Toys/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt"
COLD_ITEMS="$ROOT/GRAM/rec_datasets/Toys_cold50/cold_split_meta/cold_items.txt"
ARM_A0="$OUTPUT/arms/a0_control"
ARM_A1="$OUTPUT/arms/a1_dropout02"
ARM_A2="$OUTPUT/arms/a2_weight_decay1e3"
ASSIGNED_ID="$ROOT/GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split_v1_minilm_regularized_residualcold.txt"
SAFE_ID="$ROOT/GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split_v1_minilm_regularized_residualcold_collision_safe.txt"
ASSIGN_REPORT="$OUTPUT/winner_assign_report.json"
ID_REPORT="$OUTPUT/winner_id_report.json"
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
  printf '{"experiment_id":"GRAM_PHASE13_V1_MINILM_REGULARIZED_RESIDUAL_TOYS_SCREEN","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%d,"workload_pid":%d,"tmux_session":"%s","physical_gpu":%s,"minimum_free_mib":%d,"expected_incremental_gpu_mib_upper_bound":3072,"hard_timeout_seconds":%d,"resource_mode":"user_assigned_direct_no_resource_changes","formal_experiment":false,"gram_training_run":false,"cold_policy":"warm_select_then_winner_only","log_path":"%s","status_path":"%s","summary_path":"%s"}\n' \
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

train_arm() {
  local arm_name=$1 output_dir=$2 dropout=$3 weight_decay=$4
  CURRENT_STAGE="train_${arm_name}"
  CURRENT_REASON="Training ${arm_name} with warm validation only; cold diagnostic remains uncomputed."
  run_step env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
    experiment/phase13/protocol/semantic_bridge_residual.py train \
      --embeddings "$EMBED" --id-file "$SOURCE_ID" --cold-items "$COLD_ITEMS" \
      --output-dir "$output_dir" --hidden-dim 768 --dropout "$dropout" \
      --weight-decay "$weight_decay" --epochs 200 --lr 1e-3 \
      --batch-size 512 --device cuda:0 --seed 12345
}

worker() {
  GPU=${1:?missing gpu}
  STARTED_AT=${2:?missing start timestamp}
  RUNNER_PID=$$
  trap on_term TERM INT HUP
  trap cleanup EXIT
  cd "$ROOT"

  CURRENT_STAGE=preflight
  CURRENT_REASON="Checking three-arm code, frozen inputs, isolated outputs, and GPU admission."
  write_status running "$CURRENT_REASON"
  "$PYTHON" -m py_compile \
    experiment/phase13/protocol/semantic_bridge_residual.py \
    experiment/phase13/protocol/select_regularized_residual_arm.py \
    experiment/phase13/protocol/finalize_regularized_residual_screen.py \
    experiment/phase13/protocol/assign_cold_ids.py \
    experiment/phase13/protocol/make_collision_safe_ids.py \
    || { write_status failed "Python syntax check failed; no automatic retry."; return 4; }
  "$PYTHON" -m unittest \
    experiment/phase13/tests/test_semantic_bridge.py \
    experiment/phase13/tests/test_collision_safe_ids.py \
    || { write_status failed "Phase-13 unit tests failed; no automatic retry."; return 5; }
  for path in "$EMBED" "$SOURCE_ID" "$COLD_ITEMS" "$BASELINE_ASSIGNED_ID" \
    "$BASELINE_ID_REPORT"; do
    [[ -s "$path" ]] \
      || { write_status failed "Required frozen input missing: ${path#$ROOT/}"; return 6; }
  done
  for path in "$ARM_A0/best.pt" "$ARM_A1/best.pt" "$ARM_A2/best.pt" \
    "$ASSIGNED_ID" "$SAFE_ID" "$ASSIGN_REPORT" "$ID_REPORT" "$SUMMARY"; do
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

  train_arm a0_control "$ARM_A0" 0.0 1e-4 \
    || { write_status failed "A0 control failed/timed out; no automatic retry."; return 11; }
  train_arm a1_dropout02 "$ARM_A1" 0.2 1e-4 \
    || { write_status failed "A1 dropout failed/timed out; no automatic retry."; return 12; }
  train_arm a2_weight_decay1e3 "$ARM_A2" 0.0 1e-3 \
    || { write_status failed "A2 stronger weight decay failed/timed out; no automatic retry."; return 13; }

  CURRENT_STAGE=warm_selection
  CURRENT_REASON="Selecting one arm using warm HScore only; cold diagnostic remains uncomputed."
  run_step "$PYTHON" experiment/phase13/protocol/select_regularized_residual_arm.py \
      --control-history "$ARM_A0/training_history.json" \
      --dropout-history "$ARM_A1/training_history.json" \
      --weight-decay-history "$ARM_A2/training_history.json" \
      --output "$SUMMARY" \
    || { write_status failed "Warm-only arm selection failed; no automatic retry."; return 14; }

  local advance winner_name winner_checkpoint result
  advance=$("$PYTHON" -c "import json; r=json.load(open('$SUMMARY')); print(1 if r['advance_to_cold'] else 0)") \
    || { write_status failed "Could not read warm selection."; return 15; }
  winner_name=$("$PYTHON" -c "import json; print(json.load(open('$SUMMARY'))['winner']['name'])") \
    || { write_status failed "Could not read warm winner name."; return 16; }
  if [[ "$advance" != 1 ]]; then
    cleanup
    CURRENT_STAGE=finished
    write_status completed "FAIL_WARM_GATE: winner=${winner_name}; cold diagnostic was not computed."
    return 0
  fi
  winner_checkpoint=$("$PYTHON" -c "import json; print(json.load(open('$SUMMARY'))['winner']['checkpoint_path'])") \
    || { write_status failed "Could not read warm winner checkpoint."; return 17; }

  CURRENT_STAGE=winner_cold_assignment
  CURRENT_REASON="Warm Gate passed; assigning cold IDs for the single selected winner."
  run_step env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
    experiment/phase13/protocol/assign_cold_ids.py \
      --embeddings "$EMBED" --mlp "$winner_checkpoint" \
      --source-id-file "$SOURCE_ID" --cold-items "$COLD_ITEMS" \
      --output-id-file "$ASSIGNED_ID" --report "$ASSIGN_REPORT" --device cuda:0 \
    || { write_status failed "Winner cold-ID assignment failed/timed out; no automatic retry."; return 18; }

  CURRENT_STAGE=winner_collision_audit
  CURRENT_REASON="Auditing raw and collision-safe IDs for the single warm-selected winner."
  run_step "$PYTHON" experiment/phase13/protocol/make_collision_safe_ids.py \
      --input-id-file "$ASSIGNED_ID" --cold-items "$COLD_ITEMS" \
      --output-id-file "$SAFE_ID" --report "$ID_REPORT" \
    || { write_status failed "Winner collision audit failed/timed out; no automatic retry."; return 19; }

  CURRENT_STAGE=final_decision
  CURRENT_REASON="Applying the frozen winner-only cold prefix/path and collision gates."
  run_step "$PYTHON" experiment/phase13/protocol/finalize_regularized_residual_screen.py \
      --warm-selection "$SUMMARY" --winner-assigned-id "$ASSIGNED_ID" \
      --winner-id-report "$ID_REPORT" --baseline-assigned-id "$BASELINE_ASSIGNED_ID" \
      --baseline-id-report "$BASELINE_ID_REPORT" --source-id "$SOURCE_ID" \
      --cold-items "$COLD_ITEMS" --output "$SUMMARY" \
    || { write_status failed "Final regularized residual Gate failed; no automatic retry."; return 20; }

  result=$("$PYTHON" -c "import json; r=json.load(open('$SUMMARY')); print(f\"{r['verdict']}: winner={r['warm_selection']['winner']['name']}\")") \
    || { write_status failed "Could not read final screen summary."; return 21; }
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
    write_status starting "Background three-arm regularized residual screen started on user-assigned GPU${GPU}."
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
    [[ -f "$LOG" ]] && echo "--- last 40 log lines ---" && tail -n 40 "$LOG" || true
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
