#!/usr/bin/env bash
set -u -o pipefail

# Beauty validation-only B1 cross-domain confirmation: BGE -> P0 resolver -> B1 analysis.
# Primary candidate is unconditional_portfolio2 with Toys-frozen parameters.
# Usage: bash experiment/phase13/run_v1_r2_beauty_b1_portfolio.sh {start <gpu>|worker <gpu> <timestamp>|status|stop}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ACTION=${1:-status}; GPU=${2:-}
OUTPUT="$ROOT/artifacts/phase13/explore/v1_r2_beauty_b1_portfolio_confirmation"
P0="$ROOT/artifacts/phase13/explore/v1_r2_beauty_p0"
EMB="$ROOT/artifacts/phase13/embeddings/Beauty_bge_large_en_v1_5_cls_l2.pt"
DATA="$ROOT/GRAM/rec_datasets/Beauty_cold50"
IDFILE="$DATA/item_generative_indexing_hierarchy_v1_c128_l7_len32768_split.txt"
COLD="$DATA/cold_split_meta/cold_items.txt"
STATUS="$OUTPUT/status.json"; LOG="$OUTPUT/run.log"; TELEMETRY="$OUTPUT/gpu_telemetry.csv"
SESSION=gram_phase13_v1_r2_beauty_b1_portfolio
PYTHON=${PYTHON_BIN:-/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python}
MIN_FREE_MIB=${MIN_FREE_MIB:-6144}; HARD_TIMEOUT_SECONDS=${HARD_TIMEOUT_SECONDS:-3600}
STARTED_AT=""; RUNNER_PID=0; WORKLOAD_PID=0; STAGE=not_started

write_status() {
  local state=$1 reason=$2 tmp="$STATUS.tmp.$$"; mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE13_V1_R2_BEAUTY_B1_PORTFOLIO_CONFIRMATION","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%d,"workload_pid":%d,"tmux_session":"%s","physical_gpu":%s,"minimum_free_mib":%d,"expected_incremental_gpu_mib_upper_bound":5120,"hard_timeout_seconds":%d,"split":"beauty_validation_only","primary_candidate":"unconditional_portfolio2","test_predictions_opened":false,"automatic_retry":false,"retuning_on_beauty":false,"log_path":"%s","status_path":"%s","summary_path":"%s"}\n' \
    "$state" "$STAGE" "$reason" "$STARTED_AT" "$(date -Is)" "$RUNNER_PID" "$WORKLOAD_PID" "$SESSION" "${GPU:--1}" "$MIN_FREE_MIB" "$HARD_TIMEOUT_SECONDS" \
    "${LOG#$ROOT/}" "${STATUS#$ROOT/}" "${OUTPUT#$ROOT/}/summary.json" > "$tmp"
  mv "$tmp" "$STATUS"
}

telemetry() {
  printf 'timestamp,index,memory_used_mib,memory_free_mib,utilization_gpu_percent\n' > "$TELEMETRY"
  while true; do nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits --id="$GPU" >> "$TELEMETRY" 2>/dev/null || true; sleep 10; done
}

# Frozen: the LAST validation prediction (epoch-30 completed state), matching the
# Toys P0 convention of selecting 20260809_085251_*.
latest_validation_prediction() {
  find "$ROOT/artifacts/phase13/explore/v0_beauty/predictions" -maxdepth 1 -type f \
    -name '*_pred_validation.tsv' -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-
}

run_command() {
  local label=$1; shift; STAGE=$label; write_status running "$label active on GPU${GPU}."
  "$@" & WORKLOAD_PID=$!; write_status running "$label active on GPU${GPU}."
  wait "$WORKLOAD_PID"; local rc=$?; WORKLOAD_PID=0; return "$rc"
}

worker() {
  GPU=${1:?missing gpu}; STARTED_AT=${2:?missing timestamp}; RUNNER_PID=$$
  local telemetry_pid=0 prediction free_mib rc=0
  trap '[[ $WORKLOAD_PID -eq 0 ]] || kill -TERM "$WORKLOAD_PID" 2>/dev/null || true; [[ $telemetry_pid -eq 0 ]] || kill "$telemetry_pid" 2>/dev/null || true; STAGE=stopped; write_status stopped "Stopped by signal; no retry."; exit 143' TERM INT HUP

  STAGE=preflight; write_status running "Checking validation-only inputs, tests, and GPU admission."
  cd "$ROOT"
  bash -n experiment/phase13/run_v1_r2_beauty_b1_portfolio.sh || { write_status failed "Bash syntax failed."; return 4; }
  "$PYTHON" -m pytest -q experiment/phase13/tests/test_route_resolve.py experiment/phase13/tests/test_b1_portfolio_confirmation.py || { write_status failed "Preflight tests failed."; return 5; }

  prediction=$(latest_validation_prediction)
  [[ -n "$prediction" && -s "$prediction" ]] || { write_status failed "Beauty v0 validation prediction missing."; return 6; }
  case "$prediction" in *_pred_test.tsv) write_status failed "Refusing to use a test prediction file."; return 6 ;; esac
  for path in "$DATA/user_sequence.txt" "$DATA/item_plain_text.txt" "$COLD" "$IDFILE"; do
    [[ -s "$path" ]] || { write_status failed "Missing $path"; return 7; }
  done
  [[ ! -e "$OUTPUT/summary.json" ]] || { write_status failed "Refusing to overwrite existing B1 summary."; return 8; }

  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GPU" 2>/dev/null | tr -d ' ' || true)
  [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )) || { write_status blocked "GPU admission failed; requires ${MIN_FREE_MIB} MiB free."; return 9; }

  telemetry & telemetry_pid=$!

  # Stage 1: Beauty domain-local BGE embedding (skipped if already present).
  if [[ ! -s "$EMB" ]]; then
    run_command beauty_bge_embedding timeout 1800 env CUDA_VISIBLE_DEVICES="$GPU" \
      HF_HOME="$ROOT/.cache/huggingface" HF_HUB_CACHE="$ROOT/.cache/huggingface/hub" \
      "$PYTHON" experiment/phase13/protocol/precompute_item_embeddings.py \
        --item-text "$DATA/item_plain_text.txt" --output "$EMB" \
        --model BAAI/bge-large-en-v1.5 --device cuda:0 --batch-size 16 \
        --max-seq-len 256 --pooling cls --normalize || rc=$?
    (( rc == 0 )) || { kill "$telemetry_pid" 2>/dev/null || true; STAGE=finished; write_status failed "Beauty BGE failed rc=${rc}; no retry."; return "$rc"; }
    "$PYTHON" -c "import torch,sys; p=torch.load(sys.argv[1],map_location='cpu'); e=p['embeddings']; assert e.shape[0]==12101 and e.shape[1]==1024, e.shape; assert p['pooling']=='cls'; assert p['l2_normalized'] is True; assert torch.isfinite(e).all()" "$EMB" \
      || { kill "$telemetry_pid" 2>/dev/null || true; STAGE=finished; write_status failed "BGE embedding protocol audit failed."; return 11; }
  fi

  # Stage 2: warm-only P0 resolver, hyperparameters identical to Toys P0.
  mkdir -p "$P0"
  run_command beauty_p0_resolver timeout 1800 env CUDA_VISIBLE_DEVICES="$GPU" \
    "$PYTHON" experiment/phase13/protocol/route_resolve.py \
      --dataset-dir "$DATA" --item-id-file "$IDFILE" --item-embeddings "$EMB" \
      --gram-validation-predictions "$prediction" --output-dir "$P0" --device cuda:0 \
      --route-depth 3 --max-history 20 --epochs 12 --batch-size 256 --hidden-dim 512 \
      --dropout 0.1 --lr 0.001 --weight-decay 0.0001 --temperature 0.07 \
      --recency-decay 0.85 --global-retrieve-k 200 --top-routes 8 --per-route-k 50 \
      --rrf-k 60 --route-prior-weight 0.25 --seed 12345 || rc=$?
  kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true
  (( rc == 0 )) || { STAGE=finished; write_status failed "Beauty P0 resolver failed rc=${rc}; no retry."; return "$rc"; }

  # Stage 3: CPU-only Pareto front + paired bootstrap with the frozen Gate.
  run_command b1_analysis timeout 900 "$PYTHON" experiment/phase13/protocol/b1_portfolio_confirmation.py \
    --p0-predictions "$P0/predictions_validation.jsonl" --cold-items "$COLD" \
    --domain beauty --output-dir "$OUTPUT" || rc=$?
  (( rc == 0 )) || { STAGE=finished; write_status failed "B1 analysis failed rc=${rc}; no retry."; return "$rc"; }

  local verdict; verdict=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$OUTPUT/summary.json") || return 12
  STAGE=finished; write_status completed "$verdict"
}

case "$ACTION" in
  start)
    [[ "$GPU" =~ ^[0-7]$ ]] || { echo "usage: $0 start <gpu>" >&2; exit 2; }
    mkdir -p "$OUTPUT"; [[ ! -e "$STATUS" ]] || { echo "status exists: $STATUS" >&2; exit 3; }
    STARTED_AT=$(date -Is); STAGE=starting; write_status starting "Beauty B1 background pipeline starting on GPU${GPU}."
    printf -v launch 'bash %q worker %q %q >> %q 2>&1' "$0" "$GPU" "$STARTED_AT" "$LOG"
    tmux new-session -d -s "$SESSION" "$launch" || { STAGE=failed; write_status failed "tmux launch failed."; exit 5; }
    echo "started. status: $STATUS" ;;
  worker) worker "${2:?}" "${3:?}" ;;
  status) [[ -f "$STATUS" ]] && sed -n '1,100p' "$STATUS" || echo '{"status":"not_started"}' ;;
  stop) tmux send-keys -t "$SESSION" C-c 2>/dev/null || echo "no session" ;;
  *) echo "usage: $0 {start <gpu>|status|stop}" >&2; exit 2 ;;
esac
