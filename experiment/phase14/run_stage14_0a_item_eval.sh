#!/usr/bin/env bash
set -u -o pipefail

# Phase-14 Stage 14-0A: short CPU evaluator regression (foreground, <10 minutes).
# Usage: bash experiment/phase14/run_stage14_0a_item_eval.sh {run|status}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ACTION=${1:-status}
OUTPUT="$ROOT/artifacts/phase14/diagnostics/item_level_eval"
STATUS="$OUTPUT/status.json"
LOG="$OUTPUT/run.log"
PYTHON=${PYTHON_BIN:-/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python}
STARTED_AT=""
STAGE=not_started

write_status() {
  local state=$1 reason=$2 tmp="$STATUS.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE14_STAGE14_0A_ITEM_LEVEL_EVAL","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%d,"workload_pid":0,"physical_gpu":-1,"resource":"cpu","hard_timeout_seconds":600,"split":"validation","test_predictions_opened":false,"automatic_retry":false,"log_path":"artifacts/phase14/diagnostics/item_level_eval/run.log","status_path":"artifacts/phase14/diagnostics/item_level_eval/status.json"}\n' \
    "$state" "$STAGE" "$reason" "$STARTED_AT" "$(date -Is)" "$$" > "$tmp"
  mv "$tmp" "$STATUS"
}

run_one() {
  local name=$1 dataset=$2 ids=$3 predictions=$4 policy=$5
  "$PYTHON" experiment/phase14/protocol/item_level_eval.py \
    --dataset-dir "$dataset" \
    --item-path-file "$ids" \
    --predictions-tsv "$predictions" \
    --output-dir "$OUTPUT/$name" \
    --invalid-policy "$policy"
}

run_all() {
  local rc=0
  STARTED_AT=$(date -Is)
  STAGE=preflight
  write_status running "Syntax/tests and frozen input checks active."
  cd "$ROOT" || return 2
  "$PYTHON" -m py_compile experiment/phase14/protocol/item_level_eval.py || rc=$?
  (( rc == 0 )) || { STAGE=finished; write_status failed "Python syntax failed; no retry."; return "$rc"; }
  "$PYTHON" -m pytest -q experiment/phase14/tests/test_item_level_eval.py experiment/phase14/tests/test_cold_prefix_support.py || rc=$?
  (( rc == 0 )) || { STAGE=finished; write_status failed "Stage14 evaluator tests failed; no retry."; return "$rc"; }

  STAGE=toys_v0_formal
  write_status running "Toys v0 formal item-level parity active."
  run_one toys_v0_formal \
    GRAM/rec_datasets/Toys_cold50 \
    GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt \
    artifacts/phase13/explore/v0_toys/predictions/20260809_085251_Toys_cold50_sequential_pred_validation.tsv \
    hard_fail || rc=$?
  (( rc == 0 )) || { STAGE=finished; write_status failed "Toys v0 formal parity failed; no retry."; return "$rc"; }

  STAGE=beauty_v0_formal
  write_status running "Beauty v0 formal item-level parity active."
  run_one beauty_v0_formal \
    GRAM/rec_datasets/Beauty_cold50 \
    GRAM/rec_datasets/Beauty_cold50/item_generative_indexing_hierarchy_v1_c128_l7_len32768_split.txt \
    artifacts/phase13/explore/v0_beauty/predictions/20260811_103607_Beauty_cold50_sequential_pred_validation.tsv \
    hard_fail || rc=$?
  (( rc == 0 )) || { STAGE=finished; write_status failed "Beauty v0 formal parity failed; no retry."; return "$rc"; }

  STAGE=toys_raw_v1_legacy_alias_audit
  write_status running "Toys raw-v1 alias audit active; invalid paths recorded, never credited."
  run_one toys_raw_v1_legacy_alias_audit \
    GRAM/rec_datasets/Toys_cold50 \
    GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split_v1_mlpcold.txt \
    artifacts/phase13/explore/v1_toys/predictions/20260810_233903_Toys_cold50_sequential_pred_validation.tsv \
    record || rc=$?
  (( rc == 0 )) || { STAGE=finished; write_status failed "Toys raw-v1 audit failed; no retry."; return "$rc"; }

  STAGE=beauty_raw_v1_legacy_alias_audit
  write_status running "Beauty raw-v1 alias audit active; invalid paths recorded, never credited."
  run_one beauty_raw_v1_legacy_alias_audit \
    GRAM/rec_datasets/Beauty_cold50 \
    GRAM/rec_datasets/Beauty_cold50/item_generative_indexing_hierarchy_v1_c128_l7_len32768_split_v1_mlpcold.txt \
    artifacts/phase13/explore/v1_beauty/predictions/20260812_005851_Beauty_cold50_sequential_pred_validation.tsv \
    record || rc=$?
  (( rc == 0 )) || { STAGE=finished; write_status failed "Beauty raw-v1 audit failed; no retry."; return "$rc"; }

  STAGE=finished
  write_status completed "14-0A complete: dual-domain v0 parity and raw-v1 alias audit finished."
}

case "$ACTION" in
  run)
    mkdir -p "$OUTPUT"
    [[ ! -e "$OUTPUT/toys_v0_formal/summary.json" ]] || { echo "Refusing to overwrite existing Stage14-0A artifacts." >&2; exit 3; }
    run_all >> "$LOG" 2>&1
    ;;
  status)
    [[ -f "$STATUS" ]] && sed -n '1,120p' "$STATUS" || echo '{"status":"not_started"}'
    ;;
  *)
    echo "usage: $0 {run|status}" >&2
    exit 2
    ;;
esac
