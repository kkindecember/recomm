#!/usr/bin/env bash
set -u -o pipefail

# Toys validation-only P7 robust slate.
# Usage: bash experiment/phase13/run_v1_r2_toys_p7_robust_slate.sh {start <gpu>|status|stop}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ACTION=${1:-status}; GPU=${2:-}
OUTPUT="$ROOT/artifacts/phase13/explore/v1_r2_toys_p7_robust_slate"
STATUS="$OUTPUT/status.json"; LOG="$OUTPUT/run.log"; TELEMETRY="$OUTPUT/gpu_telemetry.csv"; SUMMARY="$OUTPUT/summary.json"
SESSION=gram_phase13_v1_r2_toys_p7_robust_slate
PYTHON=${PYTHON_BIN:-/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python}
MIN_FREE_MIB=${MIN_FREE_MIB:-6144}; HARD_TIMEOUT_SECONDS=${HARD_TIMEOUT_SECONDS:-900}
STARTED_AT=""; RUNNER_PID=0; WORKLOAD_PID=0; STAGE=not_started

write_status() {
  local state=$1 reason=$2 tmp="$STATUS.tmp.$$"; mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE13_V1_R2_TOYS_P7_ROBUST_SLATE","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%d,"workload_pid":%d,"tmux_session":"%s","physical_gpu":%s,"minimum_free_mib":%d,"expected_incremental_gpu_mib_upper_bound":4096,"hard_timeout_seconds":%d,"split":"toys_validation_outer_5fold_oof","test_predictions_opened":false,"automatic_retry":false,"log_path":"%s","status_path":"%s","summary_path":"%s"}\n' \
    "$state" "$STAGE" "$reason" "$STARTED_AT" "$(date -Is)" "$RUNNER_PID" "$WORKLOAD_PID" "$SESSION" "${GPU:--1}" "$MIN_FREE_MIB" "$HARD_TIMEOUT_SECONDS" "${LOG#$ROOT/}" "${STATUS#$ROOT/}" "${SUMMARY#$ROOT/}" > "$tmp"
  mv "$tmp" "$STATUS"
}
telemetry() { printf 'timestamp,index,memory_used_mib,memory_free_mib,utilization_gpu_percent\n' > "$TELEMETRY"; while true; do nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits --id="$GPU" >> "$TELEMETRY" 2>/dev/null || true; sleep 10; done; }

worker() {
  GPU=${1:?}; STARTED_AT=${2:?}; RUNNER_PID=$$; local rc=0 free_mib telemetry_pid=0
  trap '[[ $WORKLOAD_PID -eq 0 ]] || kill -TERM "$WORKLOAD_PID" 2>/dev/null || true; [[ $telemetry_pid -eq 0 ]] || kill "$telemetry_pid" 2>/dev/null || true; STAGE=stopped; write_status stopped "Stopped by signal; no retry."; exit 143' TERM INT HUP
  STAGE=preflight; write_status running "Checking frozen P0/P4/P6 inputs, tests, and GPU admission."
  cd "$ROOT"; bash -n experiment/phase13/run_v1_r2_toys_p7_robust_slate.sh || { write_status failed "Bash syntax failed."; return 4; }
  "$PYTHON" -m py_compile experiment/phase13/protocol/robust_slate_optimizer.py || { write_status failed "Python syntax failed."; return 5; }
  "$PYTHON" -m pytest -q experiment/phase13/tests/test_counterfactual_slot_router.py experiment/phase13/tests/test_candidate_portfolio.py experiment/phase13/tests/test_robust_slate_optimizer.py || { write_status failed "P4/P6/P7 tests failed."; return 6; }
  for path in GRAM/rec_datasets/Toys_cold50/user_sequence.txt GRAM/rec_datasets/Toys_cold50/cold_split_meta/cold_items.txt GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt artifacts/phase13/embeddings/Toys_bge_large_en_v1_5_cls_l2.pt artifacts/phase13/explore/v1_r2_toys_p0/resolver.pt artifacts/phase13/explore/v1_r2_toys_p0/predictions_validation.jsonl artifacts/phase13/explore/v1_r2_toys_p4_counterfactual_slot_router/summary.json artifacts/phase13/explore/v1_r2_toys_p6_candidate_portfolio/summary.json; do [[ -s "$path" ]] || { write_status failed "Missing $path"; return 7; }; done
  [[ ! -e "$SUMMARY" && ! -e "$OUTPUT/config.json" && ! -e "$OUTPUT/robust_slate.pt" ]] || { write_status failed "Refusing to overwrite P7 artifacts."; return 8; }
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GPU" 2>/dev/null | tr -d ' ' || true); [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )) || { write_status blocked "GPU admission failed; requires ${MIN_FREE_MIB} MiB."; return 9; }
  telemetry & telemetry_pid=$!; STAGE=toys_p7_outer_fold_robust_slate; write_status running "P7 workload active on GPU${GPU}."
  timeout --signal=TERM --kill-after=20 "$HARD_TIMEOUT_SECONDS" env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" experiment/phase13/protocol/robust_slate_optimizer.py --dataset-dir GRAM/rec_datasets/Toys_cold50 --p0-predictions artifacts/phase13/explore/v1_r2_toys_p0/predictions_validation.jsonl --item-id-file GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt --item-embeddings artifacts/phase13/embeddings/Toys_bge_large_en_v1_5_cls_l2.pt --resolver-checkpoint artifacts/phase13/explore/v1_r2_toys_p0/resolver.pt --p4-summary artifacts/phase13/explore/v1_r2_toys_p4_counterfactual_slot_router/summary.json --p6-summary artifacts/phase13/explore/v1_r2_toys_p6_candidate_portfolio/summary.json --cold-items GRAM/rec_datasets/Toys_cold50/cold_split_meta/cold_items.txt --output-dir "$OUTPUT" --device cuda:0 --folds 5 --bootstrap-models 3 --gate-epochs 250 --gate-lr 0.05 --gate-l2 0.01 --train-warm-retention 0.99 --seed 52345 &
  WORKLOAD_PID=$!; write_status running "P7 workload active on GPU${GPU}."; wait "$WORKLOAD_PID" || rc=$?; WORKLOAD_PID=0; kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true
  (( rc == 0 )) || { STAGE=finished; write_status failed "P7 exited rc=${rc}; no retry."; return "$rc"; }
  local verdict; verdict=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$SUMMARY") || return 10; STAGE=finished; write_status completed "$verdict"
}
case "$ACTION" in
  start) [[ "$GPU" =~ ^[0-7]$ ]] || exit 2; mkdir -p "$OUTPUT"; [[ ! -e "$STATUS" ]] || { echo "status exists" >&2; exit 3; }; STARTED_AT=$(date -Is); STAGE=starting; write_status starting "Toys P7 background session starting on GPU${GPU}."; printf -v launch 'bash %q worker %q %q >> %q 2>&1' "$0" "$GPU" "$STARTED_AT" "$LOG"; tmux new-session -d -s "$SESSION" "$launch"; echo "status: $STATUS" ;;
  worker) worker "${2:?}" "${3:?}" ;;
  status) [[ -f "$STATUS" ]] && sed -n '1,100p' "$STATUS" || echo '{"status":"not_started"}' ;;
  stop) tmux send-keys -t "$SESSION" C-c ;;
  *) echo "usage: $0 {start <gpu>|status|stop}" >&2; exit 2 ;;
esac
