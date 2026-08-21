#!/usr/bin/env bash
set -u -o pipefail

# Authorized recovery for the first Stage14-0B Toys 2-user smoke.
# Usage: bash experiment/phase14/run_stage14_0b_toys_smoke_retry1.sh {start <gpu>|worker <gpu> <started_at>|status}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ACTION=${1:-status}
GPU=${2:-}
OUTPUT=${OUTPUT_OVERRIDE:-"$ROOT/artifacts/phase14/diagnostics/oracle_prefix_probe_toys_smoke_retry1"}
EXPERIMENT_ID=${EXPERIMENT_ID:-GRAM_PHASE14_STAGE14_0B_TOYS_SMOKE_RETRY1}
REL_OUTPUT=${OUTPUT#"$ROOT"/}
SESSION=${SESSION_OVERRIDE:-gram_phase14_stage14_0b_toys_smoke}
SCRIPT="$ROOT/experiment/phase14/run_stage14_0b_toys_smoke_retry1.sh"
STATUS="$OUTPUT/status.json"
LOG="$OUTPUT/run.log"
TELEMETRY="$OUTPUT/gpu_telemetry.csv"
SUMMARY="$OUTPUT/summary.json"
PYTHON=${PYTHON_BIN:-/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python}
MIN_FREE_MIB=${MIN_FREE_MIB:-12288}
HARD_TIMEOUT_SECONDS=${HARD_TIMEOUT_SECONDS:-900}
LIMIT_USERS=${LIMIT_USERS:-2}
SMOKE_BATCH_SIZE=${SMOKE_BATCH_SIZE:-2}
SMOKE_TEACHER_BATCH_SIZE=${SMOKE_TEACHER_BATCH_SIZE:-2}
STARTED_AT=""
RUNNER_PID=0
WORKLOAD_PID=0
STAGE=not_started

write_status() {
  local state=$1 reason=$2 workload_rc=${3:--1} tmp="$STATUS.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"%s","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%d,"workload_pid":%d,"workload_rc":%d,"tmux_session":"%s","physical_gpu":%s,"minimum_free_mib":%d,"expected_incremental_gpu_mib_upper_bound":12288,"hard_timeout_seconds":%d,"limit_users":%d,"beam_size":50,"split":"validation","test_predictions_opened":false,"automatic_retry":false,"retry_authorized":true,"log_path":"%s/run.log","status_path":"%s/status.json","summary_path":"%s/summary.json"}\n' \
    "$EXPERIMENT_ID" "$state" "$STAGE" "$reason" "$STARTED_AT" "$(date -Is)" "$RUNNER_PID" "$WORKLOAD_PID" "$workload_rc" "$SESSION" "${GPU:--1}" "$MIN_FREE_MIB" "$HARD_TIMEOUT_SECONDS" "$LIMIT_USERS" "$REL_OUTPUT" "$REL_OUTPUT" "$REL_OUTPUT" > "$tmp"
  mv "$tmp" "$STATUS"
}

telemetry() {
  printf 'timestamp,index,memory_used_mib,memory_free_mib,utilization_gpu_percent\n' > "$TELEMETRY"
  while true; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits --id="$GPU" >> "$TELEMETRY" 2>/dev/null || true
    sleep 10
  done
}

worker() {
  GPU=${1:?}
  STARTED_AT=${2:?}
  RUNNER_PID=$$
  local rc=0 free_mib telemetry_pid=0
  # This worker is intentionally detached.  Re-registering HUP as a terminating
  # trap defeats nohup and caused retry1 to exit rc=143 before CUDA allocation.
  trap '' HUP
  trap '[[ $WORKLOAD_PID -eq 0 ]] || kill -TERM "$WORKLOAD_PID" 2>/dev/null || true; [[ $telemetry_pid -eq 0 ]] || kill "$telemetry_pid" 2>/dev/null || true; STAGE=stopped; write_status stopped "Stopped by TERM/INT; no retry." 143; exit 143' TERM INT
  cd "$ROOT" || exit 2
  STAGE=preflight
  write_status running "Syntax, tests, frozen inputs, offline cache, and GPU admission active."
  bash -n experiment/phase14/run_stage14_0b_toys_smoke_retry1.sh || { STAGE=finished; write_status failed "Bash syntax failed; no retry." 4; exit 4; }
  "$PYTHON" -m py_compile experiment/phase14/protocol/oracle_prefix_probe.py || { STAGE=finished; write_status failed "Python syntax failed; no retry." 5; exit 5; }
  "$PYTHON" -m pytest -q experiment/phase14/tests/test_oracle_prefix_probe.py experiment/phase14/tests/test_item_level_eval.py experiment/phase14/tests/test_cold_prefix_support.py || { STAGE=finished; write_status failed "Phase14 tests failed; no retry." 6; exit 6; }
  for path in \
    artifacts/phase13/explore/v0_toys/gram_logs/Toys_cold50/0_20260808_2256/config.json \
    artifacts/phase13/explore/v0_toys/gram_logs/Toys_cold50/0_20260808_2256/id_0_rec_30/model_rec_phase_1_epoch_30.pt \
    GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt \
    artifacts/phase13/explore/v0_toys/predictions/20260809_085251_Toys_cold50_sequential_pred_validation.tsv \
    artifacts/phase13/embeddings/Toys_bge_large_en_v1_5_cls_l2.pt \
    artifacts/phase13/explore/v1_r2_toys_p0/resolver.pt; do
    [[ -s "$path" ]] || { STAGE=finished; write_status failed "Missing frozen input: $path" 7; exit 7; }
  done
  [[ ! -e "$SUMMARY" ]] || { STAGE=finished; write_status failed "Refusing to overwrite recovery summary." 8; exit 8; }
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GPU" 2>/dev/null | tr -d ' ' || true)
  [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )) || { STAGE=finished; write_status blocked "GPU admission failed; requires ${MIN_FREE_MIB} MiB free." 9; exit 9; }

  telemetry & telemetry_pid=$!
  STAGE=toys_2user_oracle_prefix_probe
  write_status running "Authorized ${LIMIT_USERS}-user recovery workload active on GPU${GPU}."
  timeout --signal=TERM --kill-after=20 "$HARD_TIMEOUT_SECONDS" \
    env CUDA_VISIBLE_DEVICES="$GPU" TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    "$PYTHON" experiment/phase14/protocol/oracle_prefix_probe.py \
      --dataset-dir GRAM/rec_datasets/Toys_cold50 \
      --historical-config artifacts/phase13/explore/v0_toys/gram_logs/Toys_cold50/0_20260808_2256/config.json \
      --checkpoint artifacts/phase13/explore/v0_toys/gram_logs/Toys_cold50/0_20260808_2256/id_0_rec_30/model_rec_phase_1_epoch_30.pt \
      --item-path-file GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt \
      --frozen-predictions artifacts/phase13/explore/v0_toys/predictions/20260809_085251_Toys_cold50_sequential_pred_validation.tsv \
      --item-embeddings artifacts/phase13/embeddings/Toys_bge_large_en_v1_5_cls_l2.pt \
      --resolver-checkpoint artifacts/phase13/explore/v1_r2_toys_p0/resolver.pt \
      --output-dir "$OUTPUT" \
      --device cuda:0 --batch-size "$SMOKE_BATCH_SIZE" --teacher-batch-size "$SMOKE_TEACHER_BATCH_SIZE" --beam-size 50 --limit "$LIMIT_USERS" &
  WORKLOAD_PID=$!
  write_status running "Authorized ${LIMIT_USERS}-user recovery workload active on GPU${GPU}."
  wait "$WORKLOAD_PID" || rc=$?
  WORKLOAD_PID=0
  kill "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
  telemetry_pid=0
  (( rc == 0 )) || { STAGE=finished; write_status failed "Recovery workload exited rc=${rc}; no retry." "$rc"; exit "$rc"; }
  [[ -s "$SUMMARY" ]] || { STAGE=finished; write_status failed "Workload rc=0 but summary missing; no retry." 10; exit 10; }
  STAGE=finished
  write_status completed "Authorized recovery smoke completed; inspect summary and beam parity." 0
}

case "$ACTION" in
  start)
    [[ "$GPU" =~ ^[0-7]$ ]] || { echo "GPU must be 0..7" >&2; exit 2; }
    mkdir -p "$OUTPUT"
    [[ ! -e "$STATUS" ]] || { echo "Recovery status already exists" >&2; exit 3; }
    STARTED_AT=$(date -Is)
    STAGE=starting
    write_status starting "Authorized recovery tmux worker starting on GPU${GPU}."
    tmux has-session -t "$SESSION" 2>/dev/null && { write_status blocked "tmux session already exists: $SESSION" 11; exit 11; }
    printf -v launch 'env OUTPUT_OVERRIDE=%q EXPERIMENT_ID=%q SESSION_OVERRIDE=%q LIMIT_USERS=%q SMOKE_BATCH_SIZE=%q SMOKE_TEACHER_BATCH_SIZE=%q HARD_TIMEOUT_SECONDS=%q bash %q worker %q %q >> %q 2>&1' \
      "$OUTPUT" "$EXPERIMENT_ID" "$SESSION" "$LIMIT_USERS" "$SMOKE_BATCH_SIZE" "$SMOKE_TEACHER_BATCH_SIZE" "$HARD_TIMEOUT_SECONDS" "$SCRIPT" "$GPU" "$STARTED_AT" "$LOG"
    tmux new-session -d -s "$SESSION" "$launch" || { STAGE=finished; write_status failed "tmux launch failed; no retry." 12; exit 12; }
    echo "status: $STATUS"
    ;;
  worker)
    worker "${2:?}" "${3:?}"
    ;;
  status)
    [[ -f "$STATUS" ]] && sed -n '1,120p' "$STATUS" || echo '{"status":"not_started"}'
    ;;
  *)
    echo "usage: $0 {start <gpu>|worker <gpu> <started_at>|status}" >&2
    exit 2
    ;;
esac
