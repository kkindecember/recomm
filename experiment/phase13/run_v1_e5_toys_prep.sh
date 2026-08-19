#!/usr/bin/env bash
# Prepare collision-safe E5 v1 iter2 artifacts for Toys on a user-assigned GPU.
#
# This runner never starts/stops a protector or lease and never retries. It is
# backgrounded because the first model download may push total time past 10 min.
#
# Usage:
#   bash experiment/phase13/run_v1_e5_toys_prep.sh start 6
#   bash experiment/phase13/run_v1_e5_toys_prep.sh status
#   bash experiment/phase13/run_v1_e5_toys_prep.sh stop
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
ACTION=${1:-status}
GPU=${2:-}

SUB=v1_collision_safe_e5_toys_prep
SESSION=gram_phase13_${SUB}
OUTPUT="$ROOT/artifacts/phase13/explore/$SUB"
LOG="$OUTPUT/run.log"
STATUS="$OUTPUT/status.json"
TELEMETRY="$OUTPUT/gpu_telemetry.csv"
EMBED="$ROOT/artifacts/phase13/embeddings/Toys_e5_large_v2_query_mean_l2.pt"
MLP_DIR="$OUTPUT/mlp"
MLP="$MLP_DIR/best.pt"
ASSIGN_REPORT="$OUTPUT/assign_report.json"
ID_REPORT="$OUTPUT/id_report.json"
SOURCE_TEXT="$ROOT/GRAM/rec_datasets/Toys/item_plain_text.txt"
SOURCE_ID="$ROOT/GRAM/rec_datasets/Toys/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt"
COLD_ITEMS="$ROOT/GRAM/rec_datasets/Toys_cold50/cold_split_meta/cold_items.txt"
ASSIGNED_ID="$ROOT/GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split_v1_e5_mlpcold.txt"
SAFE_ID="$ROOT/GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split_v1_e5_mlpcold_collision_safe.txt"
MODEL=intfloat/e5-large-v2
MIN_FREE_MIB=${MIN_FREE_MIB:-8192}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-14400}

STARTED_AT=""
RUNNER_PID=0
WORKLOAD_PID=0
TELEMETRY_PID=0
CURRENT_STAGE=not_started

write_status() {
  local state=$1 reason=$2 tmp="${STATUS}.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE13_V1_COLLISION_SAFE_E5_TOYS_PREP","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%d,"workload_pid":%d,"tmux_session":"%s","physical_gpu":%s,"minimum_free_mib":%d,"hard_timeout_seconds":%d,"resource_mode":"user_assigned_direct_no_resource_changes","log_path":"%s","embedding_path":"%s","mlp_path":"%s","id_path":"%s"}\n' \
    "$state" "$CURRENT_STAGE" "$reason" "$STARTED_AT" "$(date -Is)" "$RUNNER_PID" "$WORKLOAD_PID" \
    "$SESSION" "${GPU:--1}" "$MIN_FREE_MIB" "$TIMEOUT_SECONDS" "${LOG#$ROOT/}" \
    "${EMBED#$ROOT/}" "${MLP#$ROOT/}" "${SAFE_ID#$ROOT/}" > "$tmp"
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
  mkdir -p "$OUTPUT" "$ROOT/artifacts/phase13/embeddings"

  CURRENT_STAGE=preflight
  write_status running "Checking scripts, frozen inputs, output isolation, and GPU admission."
  "$PYTHON" -m py_compile \
    experiment/phase13/protocol/precompute_item_embeddings.py \
    experiment/phase13/protocol/semantic_bridge.py \
    experiment/phase13/protocol/assign_cold_ids.py \
    experiment/phase13/protocol/make_collision_safe_ids.py \
    || { write_status failed "Python syntax check failed; no automatic retry."; return 4; }
  "$PYTHON" -m unittest \
    experiment/phase13/tests/test_semantic_bridge.py \
    experiment/phase13/tests/test_collision_safe_ids.py \
    || { write_status failed "Phase-13 unit tests failed; no automatic retry."; return 5; }
  for path in "$SOURCE_TEXT" "$SOURCE_ID" "$COLD_ITEMS"; do
    [[ -s "$path" ]] || { write_status failed "Required frozen input missing: ${path#$ROOT/}"; return 6; }
  done
  for path in "$EMBED" "$MLP" "$ASSIGNED_ID" "$SAFE_ID" "$ID_REPORT"; do
    [[ ! -e "$path" ]] || { write_status failed "Refusing to overwrite existing artifact: ${path#$ROOT/}"; return 7; }
  done
  local free_mib
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GPU" 2>/dev/null | tr -d ' ' || true)
  [[ "$free_mib" =~ ^[0-9]+$ ]] \
    || { write_status failed "Could not read GPU${GPU} free memory; no automatic retry."; return 8; }
  (( free_mib >= MIN_FREE_MIB )) \
    || { write_status blocked "GPU${GPU} has ${free_mib} MiB free; requires ${MIN_FREE_MIB}; no resource changes made."; return 9; }

  telemetry & TELEMETRY_PID=$!

  CURRENT_STAGE=e5_embedding
  write_status running "Encoding all Toys item text with E5 query-prefix/mean-pooling/L2 protocol."
  run_step env CUDA_VISIBLE_DEVICES="$GPU" \
    HF_HOME="$ROOT/.cache/huggingface" \
    HF_HUB_CACHE="$ROOT/.cache/huggingface/hub" \
    TRANSFORMERS_CACHE="$ROOT/.cache/huggingface/transformers" \
    "$PYTHON" experiment/phase13/protocol/precompute_item_embeddings.py \
      --item-text "$SOURCE_TEXT" --output "$EMBED" --model "$MODEL" \
      --text-prefix "query: " --normalize --device cuda:0 \
      --batch-size 32 --max-seq-len 256 \
    || { write_status failed "E5 embedding step failed/timed out; no automatic retry."; return 10; }
  "$PYTHON" -c "import torch; p=torch.load('$EMBED', map_location='cpu'); e=p['embeddings']; assert e.shape==(11924,1024), e.shape; assert p['model_name']=='$MODEL'; assert p['text_prefix']=='query: '; assert p['l2_normalized'] is True; assert torch.isfinite(e).all(); n=e.norm(p=2,dim=1); assert float((n-1).abs().max()) < 1e-4" \
    || { write_status failed "Embedding protocol audit failed; no automatic retry."; return 11; }

  CURRENT_STAGE=semantic_bridge
  write_status running "Training the frozen 1-layer Semantic Bridge on warm Toys items."
  run_step env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
    experiment/phase13/protocol/semantic_bridge.py train \
      --embeddings "$EMBED" --id-file "$SOURCE_ID" --cold-items "$COLD_ITEMS" \
      --output-dir "$MLP_DIR" --epochs 200 --lr 1e-3 --batch-size 512 \
      --device cuda:0 --seed 12345 \
    || { write_status failed "Semantic Bridge training failed/timed out; no automatic retry."; return 12; }

  CURRENT_STAGE=id_assignment
  write_status running "Assigning E5-MLP IDs to cold items while preserving warm IDs."
  run_step env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
    experiment/phase13/protocol/assign_cold_ids.py \
      --embeddings "$EMBED" --mlp "$MLP" --source-id-file "$SOURCE_ID" \
      --cold-items "$COLD_ITEMS" --output-id-file "$ASSIGNED_ID" \
      --report "$ASSIGN_REPORT" --device cuda:0 \
    || { write_status failed "Cold-ID assignment failed/timed out; no automatic retry."; return 13; }

  CURRENT_STAGE=collision_safe_audit
  write_status running "Making E5 IDs globally unique and checking hard invariants."
  run_step "$PYTHON" experiment/phase13/protocol/make_collision_safe_ids.py \
      --input-id-file "$ASSIGNED_ID" --cold-items "$COLD_ITEMS" \
      --output-id-file "$SAFE_ID" --report "$ID_REPORT" \
    || { write_status failed "Collision-safe transformation failed/timed out; no automatic retry."; return 14; }
  "$PYTHON" -c "import json; r=json.load(open('$ID_REPORT')); assert r['n_items']==11924; assert r['n_cold']==5963; assert r['output_collision']['duplicate_excess']==0; assert r['warm_ids_unchanged']; assert r['row_order_unchanged']; assert r['cold_prefixes_unchanged']" \
    || { write_status failed "Collision-safe report invariant audit failed; no automatic retry."; return 15; }

  cleanup
  CURRENT_STAGE=finished
  write_status completed "E5 prep completed; artifacts passed embedding and collision-safe audits. Smoke was not auto-started."
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
    write_status starting "Background E5 prep session started on user-assigned GPU${GPU}."
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
