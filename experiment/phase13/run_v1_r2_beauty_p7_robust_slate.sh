#!/usr/bin/env bash
set -u -o pipefail

# Beauty validation-only P7 pipeline: BGE -> P0 -> P4 -> robust slate.
# Usage: bash experiment/phase13/run_v1_r2_beauty_p7_robust_slate.sh {start <gpu>|worker <gpu> <timestamp>|status|stop}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ACTION=${1:-status}; GPU=${2:-}
OUTPUT="$ROOT/artifacts/phase13/explore/v1_r2_beauty_p7_robust_slate"
P0="$ROOT/artifacts/phase13/explore/v1_r2_beauty_p0"
P4="$ROOT/artifacts/phase13/explore/v1_r2_beauty_p4"
EMB="$ROOT/artifacts/phase13/embeddings/Beauty_bge_large_en_v1_5_cls_l2.pt"
STATUS="$OUTPUT/status.json"; LOG="$OUTPUT/run.log"; TELEMETRY="$OUTPUT/gpu_telemetry.csv"
SESSION=gram_phase13_v1_r2_beauty_p7_robust_slate
PYTHON=${PYTHON_BIN:-/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python}
MIN_FREE_MIB=${MIN_FREE_MIB:-8192}; HARD_TIMEOUT_SECONDS=${HARD_TIMEOUT_SECONDS:-7200}
STARTED_AT=""; RUNNER_PID=0; WORKLOAD_PID=0; STAGE=not_started

write_status() {
  local state=$1 reason=$2 tmp="$STATUS.tmp.$$"; mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE13_V1_R2_BEAUTY_P7_ROBUST_SLATE","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%d,"workload_pid":%d,"tmux_session":"%s","physical_gpu":%s,"minimum_free_mib":%d,"expected_incremental_gpu_mib_upper_bound":6144,"hard_timeout_seconds":%d,"split":"beauty_validation_outer_5fold_oof","test_predictions_opened":false,"automatic_retry":false,"log_path":"%s","status_path":"%s","summary_path":"%s"}\n' \
    "$state" "$STAGE" "$reason" "$STARTED_AT" "$(date -Is)" "$RUNNER_PID" "$WORKLOAD_PID" "$SESSION" "${GPU:--1}" "$MIN_FREE_MIB" "$HARD_TIMEOUT_SECONDS" \
    "${LOG#$ROOT/}" "${STATUS#$ROOT/}" "${OUTPUT#$ROOT/}/summary.json" > "$tmp"
  mv "$tmp" "$STATUS"
}

telemetry() {
  printf 'timestamp,index,memory_used_mib,memory_free_mib,utilization_gpu_percent\n' > "$TELEMETRY"
  while true; do nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits --id="$GPU" >> "$TELEMETRY" 2>/dev/null || true; sleep 10; done
}

latest_validation_prediction() {
  find "$ROOT/artifacts/phase13/explore/v0_beauty/predictions" -maxdepth 1 -type f -name '*_pred_validation.tsv' -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-
}

run_command() {
  local label=$1; shift; STAGE=$label; write_status running "$label active on GPU${GPU}."; "$@" & WORKLOAD_PID=$!; write_status running "$label active on GPU${GPU}."; wait "$WORKLOAD_PID"; local rc=$?; WORKLOAD_PID=0; return "$rc"
}

worker() {
  GPU=${1:?missing gpu}; STARTED_AT=${2:?missing timestamp}; RUNNER_PID=$$
  local telemetry_pid=0 prediction free_mib rc=0
  trap '[[ $WORKLOAD_PID -eq 0 ]] || kill -TERM "$WORKLOAD_PID" 2>/dev/null || true; [[ $telemetry_pid -eq 0 ]] || kill "$telemetry_pid" 2>/dev/null || true; STAGE=stopped; write_status stopped "Stopped by signal; no retry."; exit 143' TERM INT HUP
  STAGE=preflight; write_status running "Checking validation-only inputs, tests, and GPU admission."
  cd "$ROOT"
  bash -n experiment/phase13/run_v1_r2_beauty_p7_robust_slate.sh || { write_status failed "Bash syntax failed."; return 4; }
  "$PYTHON" -m pytest -q experiment/phase13/tests/test_route_resolve.py experiment/phase13/tests/test_counterfactual_slot_router.py experiment/phase13/tests/test_candidate_portfolio.py experiment/phase13/tests/test_robust_slate_optimizer.py || { write_status failed "P0/P4/P7 tests failed."; return 5; }
  prediction=$(latest_validation_prediction)
  [[ -n "$prediction" && -s "$prediction" ]] || { write_status failed "Beauty v0 validation prediction missing."; return 6; }
  for path in GRAM/rec_datasets/Beauty_cold50/user_sequence.txt GRAM/rec_datasets/Beauty_cold50/item_plain_text.txt GRAM/rec_datasets/Beauty_cold50/cold_split_meta/cold_items.txt GRAM/rec_datasets/Beauty_cold50/item_generative_indexing_hierarchy_v1_c128_l7_len32768_split.txt; do [[ -s "$path" ]] || { write_status failed "Missing $path"; return 7; }; done
  [[ ! -e "$OUTPUT/summary.json" && ! -e "$P0/summary.json" && ! -e "$P4/summary.json" ]] || { write_status failed "Refusing to overwrite Beauty R2 artifacts."; return 8; }
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GPU" 2>/dev/null | tr -d ' ' || true)
  [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )) || { write_status blocked "GPU admission failed; requires ${MIN_FREE_MIB} MiB free."; return 9; }
  telemetry & telemetry_pid=$!
  if [[ ! -s "$EMB" ]]; then
    run_command beauty_bge_embedding timeout 1800 env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" experiment/phase13/protocol/precompute_item_embeddings.py --item-text GRAM/rec_datasets/Beauty_cold50/item_plain_text.txt --output "$EMB" --model BAAI/bge-large-en-v1.5 --device cuda:0 --batch-size 32 --max-seq-len 256 --pooling cls --normalize --fp16 || rc=$?
    (( rc == 0 )) || { kill "$telemetry_pid" 2>/dev/null || true; STAGE=finished; write_status failed "Beauty BGE failed rc=${rc}; no retry."; return "$rc"; }
  fi
  mkdir -p "$P0" "$P4"
  run_command beauty_p0 timeout 1800 env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" experiment/phase13/protocol/route_resolve.py --dataset-dir GRAM/rec_datasets/Beauty_cold50 --item-id-file GRAM/rec_datasets/Beauty_cold50/item_generative_indexing_hierarchy_v1_c128_l7_len32768_split.txt --item-embeddings "$EMB" --gram-validation-predictions "$prediction" --output-dir "$P0" --device cuda:0 --route-depth 3 --max-history 20 --epochs 12 --batch-size 256 --hidden-dim 512 --dropout 0.1 --lr 0.001 --weight-decay 0.0001 --temperature 0.07 --recency-decay 0.85 --global-retrieve-k 200 --top-routes 8 --per-route-k 50 --rrf-k 60 --route-prior-weight 0.25 --seed 12345 || rc=$?
  (( rc == 0 )) || { kill "$telemetry_pid" 2>/dev/null || true; STAGE=finished; write_status failed "Beauty P0 failed rc=${rc}; no retry."; return "$rc"; }
  run_command beauty_p4 timeout 1800 env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" experiment/phase13/protocol/counterfactual_slot_router.py --dataset-dir GRAM/rec_datasets/Beauty_cold50 --p0-predictions "$P0/predictions_validation.jsonl" --item-id-file GRAM/rec_datasets/Beauty_cold50/item_generative_indexing_hierarchy_v1_c128_l7_len32768_split.txt --item-embeddings "$EMB" --resolver-checkpoint "$P0/resolver.pt" --cold-items GRAM/rec_datasets/Beauty_cold50/cold_split_meta/cold_items.txt --output-dir "$P4" --device cuda:0 --folds 5 --gate-epochs 250 --gate-lr 0.05 --gate-l2 0.01 --train-warm-retention 0.99 --seed 22345 --max-history 20 --recency-decay 0.85 || rc=$?
  (( rc == 0 )) || { kill "$telemetry_pid" 2>/dev/null || true; STAGE=finished; write_status failed "Beauty P4 failed rc=${rc}; no retry."; return "$rc"; }
  run_command beauty_p7 timeout 1800 env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" experiment/phase13/protocol/robust_slate_optimizer.py --dataset-dir GRAM/rec_datasets/Beauty_cold50 --p0-predictions "$P0/predictions_validation.jsonl" --item-id-file GRAM/rec_datasets/Beauty_cold50/item_generative_indexing_hierarchy_v1_c128_l7_len32768_split.txt --item-embeddings "$EMB" --resolver-checkpoint "$P0/resolver.pt" --p4-summary "$P4/summary.json" --cold-items GRAM/rec_datasets/Beauty_cold50/cold_split_meta/cold_items.txt --output-dir "$OUTPUT" --device cuda:0 --folds 5 --bootstrap-models 3 --gate-epochs 250 --gate-lr 0.05 --gate-l2 0.01 --train-warm-retention 0.99 --seed 52345 || rc=$?
  kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true
  (( rc == 0 )) || { STAGE=finished; write_status failed "Beauty P7 failed rc=${rc}; no retry."; return "$rc"; }
  local verdict; verdict=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$OUTPUT/summary.json") || return 12
  STAGE=finished; write_status completed "$verdict"
}

case "$ACTION" in
  start)
    [[ "$GPU" =~ ^[0-7]$ ]] || { echo "usage: $0 start <gpu>" >&2; exit 2; }; mkdir -p "$OUTPUT"; [[ ! -e "$STATUS" ]] || { echo "status exists" >&2; exit 3; }
    STARTED_AT=$(date -Is); STAGE=starting; write_status starting "Beauty P7 background pipeline starting on GPU${GPU}."
    printf -v launch 'bash %q worker %q %q >> %q 2>&1' "$0" "$GPU" "$STARTED_AT" "$LOG"; tmux new-session -d -s "$SESSION" "$launch"; echo "status: $STATUS" ;;
  worker) worker "${2:?}" "${3:?}" ;;
  status) [[ -f "$STATUS" ]] && sed -n '1,100p' "$STATUS" || echo '{"status":"not_started"}' ;;
  stop) tmux send-keys -t "$SESSION" C-c ;;
  *) echo "usage: $0 {start <gpu>|status|stop}" >&2; exit 2 ;;
esac
