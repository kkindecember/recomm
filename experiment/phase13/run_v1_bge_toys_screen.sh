#!/usr/bin/env bash
# Warm-gated, cold-once pre-GRAM screen for BGE-large-en-v1.5 on Toys.
# This runner never starts/stops protectors, releases leases, kills unrelated
# processes, retries, or launches GRAM.
#
# Usage:
#   bash experiment/phase13/run_v1_bge_toys_screen.sh start 7
#   bash experiment/phase13/run_v1_bge_toys_screen.sh status
#   bash experiment/phase13/run_v1_bge_toys_screen.sh stop
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
ACTION=${1:-status}
GPU=${2:-}
EXPECTED_GPU=7

SUB=v1_collision_safe_bge_toys_screen
SESSION=gram_phase13_${SUB}
OUTPUT="$ROOT/artifacts/phase13/explore/$SUB"
LOG="$OUTPUT/run.log"
STATUS="$OUTPUT/status.json"
TELEMETRY="$OUTPUT/gpu_telemetry.csv"
SUMMARY="$OUTPUT/screen_summary.json"
EMBED="$ROOT/artifacts/phase13/embeddings/Toys_bge_large_en_v1_5_cls_l2.pt"
MLP_DIR="$OUTPUT/mlp"
MLP="$MLP_DIR/best.pt"
HISTORY="$MLP_DIR/training_history.json"
ASSIGN_REPORT="$OUTPUT/assign_report.json"
ID_REPORT="$OUTPUT/id_report.json"
SOURCE_TEXT="$ROOT/GRAM/rec_datasets/Toys/item_plain_text.txt"
SOURCE_ID="$ROOT/GRAM/rec_datasets/Toys/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt"
COLD_ITEMS="$ROOT/GRAM/rec_datasets/Toys_cold50/cold_split_meta/cold_items.txt"
ASSIGNED_ID="$ROOT/GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split_v1_bge_mlpcold.txt"
SAFE_ID="$ROOT/GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split_v1_bge_mlpcold_collision_safe.txt"
BASELINE_HISTORY="$ROOT/artifacts/phase13/explore/v1_toys/mlp/training_history.json"
BASELINE_ASSIGNED_ID="$ROOT/GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split_v1_mlpcold.txt"
BASELINE_ID_REPORT="$ROOT/artifacts/phase13/explore/v1_collision_safe/toys_id_report.json"
MODEL=BAAI/bge-large-en-v1.5
MIN_FREE_MIB=${MIN_FREE_MIB:-4608}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-7200}

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
  printf '{"experiment_id":"GRAM_PHASE13_V1_BGE_TOYS_SCREEN","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%d,"workload_pid":%d,"tmux_session":"%s","physical_gpu":%s,"minimum_free_mib":%d,"expected_incremental_gpu_mib_upper_bound":4096,"hard_timeout_seconds":%d,"resource_mode":"user_assigned_direct_no_resource_changes","formal_experiment":false,"gram_training_run":false,"cold_policy":"warm_gate_then_candidate_only","model_download_may_be_required":true,"log_path":"%s","status_path":"%s","summary_path":"%s","embedding_path":"%s"}\n' \
    "$state" "$CURRENT_STAGE" "$reason" "$STARTED_AT" "$(date -Is)" \
    "$RUNNER_PID" "$WORKLOAD_PID" "$SESSION" "${GPU:--1}" "$MIN_FREE_MIB" \
    "$TIMEOUT_SECONDS" "${LOG#$ROOT/}" "${STATUS#$ROOT/}" \
    "${SUMMARY#$ROOT/}" "${EMBED#$ROOT/}" > "$tmp"
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
  local elapsed remaining
  elapsed=$(( $(date +%s) - START_EPOCH ))
  remaining=$(( TIMEOUT_SECONDS - elapsed ))
  if (( remaining <= 0 )); then
    return 124
  fi
  timeout --signal=TERM "$remaining" "$@" &
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
  [[ "$GPU" == "$EXPECTED_GPU" ]] || return 2
  START_EPOCH=$(date +%s)
  RUNNER_PID=$$
  trap on_term TERM INT HUP
  trap cleanup EXIT
  cd "$ROOT"

  CURRENT_STAGE=preflight
  CURRENT_REASON="Checking BGE protocol, frozen inputs, isolated outputs, and GPU7 admission."
  write_status running "$CURRENT_REASON"
  "$PYTHON" -m py_compile \
    experiment/phase13/protocol/precompute_item_embeddings.py \
    experiment/phase13/protocol/semantic_bridge.py \
    experiment/phase13/protocol/assign_cold_ids.py \
    experiment/phase13/protocol/make_collision_safe_ids.py \
    experiment/phase13/protocol/select_bge_encoder_screen.py \
    experiment/phase13/protocol/finalize_bge_encoder_screen.py \
    || { write_status failed "Python syntax check failed; no automatic retry."; return 4; }
  "$PYTHON" -m unittest \
    experiment/phase13/tests/test_semantic_bridge.py \
    experiment/phase13/tests/test_collision_safe_ids.py \
    || { write_status failed "Phase-13 unit tests failed; no automatic retry."; return 5; }
  for path in "$SOURCE_TEXT" "$SOURCE_ID" "$COLD_ITEMS" "$BASELINE_HISTORY" \
    "$BASELINE_ASSIGNED_ID" "$BASELINE_ID_REPORT"; do
    [[ -s "$path" ]] \
      || { write_status failed "Required frozen input missing: ${path#$ROOT/}"; return 6; }
  done
  for path in "$EMBED" "$MLP" "$HISTORY" "$ASSIGNED_ID" "$SAFE_ID" \
    "$ASSIGN_REPORT" "$ID_REPORT" "$SUMMARY"; do
    [[ ! -e "$path" ]] \
      || { write_status failed "Refusing to overwrite existing artifact: ${path#$ROOT/}"; return 7; }
  done
  local free_mib
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
    --id="$GPU" 2>/dev/null | tr -d ' ' || true)
  [[ "$free_mib" =~ ^[0-9]+$ ]] \
    || { write_status failed "Could not read GPU7 free memory; no automatic retry."; return 8; }
  (( free_mib >= MIN_FREE_MIB )) \
    || { write_status blocked "GPU7 has ${free_mib} MiB free; requires ${MIN_FREE_MIB}; no resource changes made."; return 9; }

  telemetry & TELEMETRY_PID=$!

  CURRENT_STAGE=bge_embedding
  CURRENT_REASON="Downloading BGE if absent, then encoding Toys with CLS pooling and L2 normalization."
  run_step env CUDA_VISIBLE_DEVICES="$GPU" \
    HF_HOME="$ROOT/.cache/huggingface" \
    HF_HUB_CACHE="$ROOT/.cache/huggingface/hub" \
    TRANSFORMERS_CACHE="$ROOT/.cache/huggingface/transformers" \
    "$PYTHON" experiment/phase13/protocol/precompute_item_embeddings.py \
      --item-text "$SOURCE_TEXT" --output "$EMBED" --model "$MODEL" \
      --pooling cls --normalize --device cuda:0 --batch-size 16 --max-seq-len 256 \
    || { write_status failed "BGE embedding failed/timed out; no automatic retry."; return 10; }
  "$PYTHON" -c "import torch; p=torch.load('$EMBED',map_location='cpu'); e=p['embeddings']; assert e.shape==(11924,1024),e.shape; assert p['model_name']=='$MODEL'; assert p['pooling']=='cls'; assert p['text_prefix']==''; assert p['l2_normalized'] is True; assert torch.isfinite(e).all(); assert float((e.norm(p=2,dim=1)-1).abs().max())<1e-4" \
    || { write_status failed "BGE embedding protocol audit failed; no automatic retry."; return 11; }

  CURRENT_STAGE=semantic_bridge_400
  CURRENT_REASON="Training the frozen one-layer BGE semantic bridge for 400 epochs; no cold oracle read."
  run_step env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
    experiment/phase13/protocol/semantic_bridge.py train \
      --embeddings "$EMBED" --id-file "$SOURCE_ID" --cold-items "$COLD_ITEMS" \
      --output-dir "$MLP_DIR" --epochs 400 --lr 1e-3 --batch-size 512 \
      --device cuda:0 --seed 12345 \
    || { write_status failed "BGE semantic bridge failed/timed out; no automatic retry."; return 12; }

  CURRENT_STAGE=warm_gate
  CURRENT_REASON="Applying the frozen warm validation floor before any cold diagnostic."
  run_step "$PYTHON" experiment/phase13/protocol/select_bge_encoder_screen.py \
      --candidate-history "$HISTORY" --baseline-history "$BASELINE_HISTORY" \
      --output "$SUMMARY" --relative-floor 0.995 \
    || { write_status failed "BGE warm-gate selection failed; no automatic retry."; return 13; }

  local advance
  advance=$("$PYTHON" -c "import json,sys; print('yes' if json.load(open(sys.argv[1]))['advance_to_cold'] else 'no')" "$SUMMARY") \
    || { write_status failed "Could not read BGE warm-gate result."; return 14; }
  if [[ "$advance" != yes ]]; then
    local warm_result
    warm_result=$("$PYTHON" -c "import json,sys; r=json.load(open(sys.argv[1])); print(f\"FAIL_WARM_GATE: val={r['candidate']['best_val_avg_acc']:.6f}, gate={r['warm_gate']['minimum_val_avg_acc']:.6f}; cold diagnostic was not computed.\")" "$SUMMARY") \
      || { write_status failed "Could not format BGE warm-gate result."; return 15; }
    cleanup
    CURRENT_STAGE=finished
    write_status completed "$warm_result"
    return 0
  fi

  CURRENT_STAGE=candidate_cold_assignment
  CURRENT_REASON="Warm gate passed; assigning cold IDs once for the BGE candidate."
  run_step env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
    experiment/phase13/protocol/assign_cold_ids.py \
      --embeddings "$EMBED" --mlp "$MLP" --source-id-file "$SOURCE_ID" \
      --cold-items "$COLD_ITEMS" --output-id-file "$ASSIGNED_ID" \
      --report "$ASSIGN_REPORT" --device cuda:0 \
    || { write_status failed "BGE cold-ID assignment failed/timed out; no automatic retry."; return 16; }

  CURRENT_STAGE=collision_audit
  CURRENT_REASON="Computing BGE raw collisions and verifying collision-safe IDs."
  run_step "$PYTHON" experiment/phase13/protocol/make_collision_safe_ids.py \
      --input-id-file "$ASSIGNED_ID" --cold-items "$COLD_ITEMS" \
      --output-id-file "$SAFE_ID" --report "$ID_REPORT" \
    || { write_status failed "BGE collision audit failed/timed out; no automatic retry."; return 17; }

  CURRENT_STAGE=cold_gate
  CURRENT_REASON="Applying the frozen BGE deep-prefix, exact-path, macro, and collision gates."
  run_step "$PYTHON" experiment/phase13/protocol/finalize_bge_encoder_screen.py \
      --warm-selection "$SUMMARY" --candidate-assigned-id "$ASSIGNED_ID" \
      --candidate-id-report "$ID_REPORT" \
      --baseline-assigned-id "$BASELINE_ASSIGNED_ID" \
      --baseline-id-report "$BASELINE_ID_REPORT" --source-id "$SOURCE_ID" \
      --cold-items "$COLD_ITEMS" --output "$SUMMARY" \
    || { write_status failed "BGE cold-gate finalization failed; no automatic retry."; return 18; }

  local result
  result=$("$PYTHON" -c "import json,sys; r=json.load(open(sys.argv[1])); print(f\"{r['verdict']}: prefix3={r['candidate_cold_id_metrics']['prefix_accuracy'][2]:.6f}, exact={r['candidate_cold_id_metrics']['exact_path_accuracy']:.6f}\")" "$SUMMARY") \
    || { write_status failed "Could not read completed BGE summary."; return 19; }
  cleanup
  CURRENT_STAGE=finished
  write_status completed "$result"
}

case "$ACTION" in
  start)
    [[ "$GPU" == "$EXPECTED_GPU" ]] \
      || { echo "usage: $0 start 7 (this experiment is frozen to GPU7)" >&2; exit 2; }
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
    write_status starting "Background BGE pre-GRAM screen started on user-assigned GPU7."
    echo "started $SESSION on GPU7"
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
    echo "usage: $0 {start 7|status|stop}" >&2
    exit 2
    ;;
esac
