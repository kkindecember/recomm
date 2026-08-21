#!/usr/bin/env bash
set -u -o pipefail

# Stage14-0C validation-only same-backbone verifier control.  The worker uses
# only the remaining memory on the user-assigned GPU and never manages other
# GPU processes.  Formal runs are sequential: Toys, then Beauty.
# Usage: bash experiment/phase14/run_stage14_0c_same_backbone_verifier.sh {start <gpu>|worker <gpu> <started_at>|status}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ACTION=${1:-status}
GPU=${2:-}
OUTPUT=${OUTPUT_OVERRIDE:-"$ROOT/artifacts/phase14/controls/same_backbone_verifier_formal_dual_domain"}
REL_OUTPUT=${OUTPUT#"$ROOT"/}
TOYS_OUTPUT="$OUTPUT/toys"
BEAUTY_OUTPUT="$OUTPUT/beauty"
STATUS="$OUTPUT/status.json"
LOG="$OUTPUT/run.log"
TELEMETRY="$OUTPUT/gpu_telemetry.csv"
SCRIPT="$ROOT/experiment/phase14/run_stage14_0c_same_backbone_verifier.sh"
PYTHON=${PYTHON_BIN:-/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python}
EXPERIMENT_ID=${EXPERIMENT_ID:-GRAM_PHASE14_STAGE14_0C_SAME_BACKBONE_VERIFIER_FORMAL_DUAL_DOMAIN}
MIN_FREE_MIB=${MIN_FREE_MIB:-12288}
DOMAIN_TIMEOUT_SECONDS=${DOMAIN_TIMEOUT_SECONDS:-43200}
FORMAL_BATCH_SIZE=${FORMAL_BATCH_SIZE:-4}
CANDIDATE_CHUNK_SIZE=${CANDIDATE_CHUNK_SIZE:-10}
BOOTSTRAP_RESAMPLES=${BOOTSTRAP_RESAMPLES:-10000}
STARTED_AT=""
RUNNER_PID=0
WORKLOAD_PID=0
PROGRESS_PID=0
STAGE=not_started
CURRENT_DOMAIN=none
TOTAL_USERS=0
COMPLETED_DOMAINS=none

write_status() {
  local state=$1 reason=$2 workload_rc=${3:--1} tmp="$STATUS.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"%s","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%d,"workload_pid":%d,"workload_rc":%d,"physical_gpu":%s,"minimum_free_mib":%d,"domain_timeout_seconds":%d,"batch_size":%d,"candidate_budget":50,"candidate_chunk_size":%d,"bootstrap_resamples":%d,"current_domain":"%s","current_domain_total_users":%d,"completed_domains":"%s","split":"validation","test_predictions_opened":false,"model_training":false,"automatic_retry":false,"resource_mode":"user_assigned_gpu_remaining_memory_only_no_process_changes","estimated_total_wall_clock_hours":"4-8 base; shared-load slowdown recorded from telemetry","log_path":"%s/run.log","status_path":"%s/status.json","toys_summary_path":"%s/toys/summary.json","beauty_summary_path":"%s/beauty/summary.json"}\n' \
    "$EXPERIMENT_ID" "$state" "$STAGE" "$reason" "$STARTED_AT" "$(date -Is)" "$RUNNER_PID" "$WORKLOAD_PID" "$workload_rc" "${GPU:--1}" "$MIN_FREE_MIB" "$DOMAIN_TIMEOUT_SECONDS" "$FORMAL_BATCH_SIZE" "$CANDIDATE_CHUNK_SIZE" "$BOOTSTRAP_RESAMPLES" "$CURRENT_DOMAIN" "$TOTAL_USERS" "$COMPLETED_DOMAINS" "$REL_OUTPUT" "$REL_OUTPUT" "$REL_OUTPUT" "$REL_OUTPUT" > "$tmp"
  mv "$tmp" "$STATUS"
}

telemetry() {
  printf 'timestamp,index,memory_used_mib,memory_free_mib,utilization_gpu_percent\n' > "$TELEMETRY"
  while true; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits --id="$GPU" >> "$TELEMETRY" 2>/dev/null || true
    sleep 60
  done
}

progress_status() {
  local marker=""
  while kill -0 "$WORKLOAD_PID" 2>/dev/null; do
    sleep 60
    kill -0 "$WORKLOAD_PID" 2>/dev/null || break
    marker=$(grep -E '\[verifier\]' "$LOG" 2>/dev/null | tail -n 1 || true)
    [[ -n "$marker" ]] || marker="${CURRENT_DOMAIN} verifier active; awaiting first progress marker."
    write_status running "$marker"
  done
}

stop_owned_children() {
  [[ $PROGRESS_PID -eq 0 ]] || kill "$PROGRESS_PID" 2>/dev/null || true
  [[ $WORKLOAD_PID -eq 0 ]] || kill -TERM "$WORKLOAD_PID" 2>/dev/null || true
}

run_domain() {
  local domain=$1 output=$2 dataset=$3 historical=$4 checkpoint=$5 item_path=$6 r2_predictions=$7
  local rc=0
  CURRENT_DOMAIN=$domain
  if [[ "$domain" == toys ]]; then TOTAL_USERS=8789; else TOTAL_USERS=10655; fi
  STAGE="${domain}_full_validation"
  [[ ! -e "$output/summary.json" ]] || { write_status failed "Refusing to overwrite ${domain} summary." 8; return 8; }
  write_status running "${domain} validation-only verifier starting."
  timeout --signal=TERM --kill-after=30 "$DOMAIN_TIMEOUT_SECONDS" \
    env CUDA_VISIBLE_DEVICES="$GPU" TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    "$PYTHON" experiment/phase14/protocol/same_backbone_verifier.py \
      --dataset-dir "$dataset" \
      --historical-config "$historical" \
      --checkpoint "$checkpoint" \
      --item-path-file "$item_path" \
      --r2-predictions "$r2_predictions" \
      --output-dir "$output" \
      --device cuda:0 \
      --batch-size "$FORMAL_BATCH_SIZE" \
      --candidate-count 50 \
      --candidate-chunk-size "$CANDIDATE_CHUNK_SIZE" \
      --bootstrap-resamples "$BOOTSTRAP_RESAMPLES" \
      --split validation &
  WORKLOAD_PID=$!
  progress_status & PROGRESS_PID=$!
  write_status running "${domain} validation-only verifier active."
  wait "$WORKLOAD_PID" || rc=$?
  WORKLOAD_PID=0
  kill "$PROGRESS_PID" 2>/dev/null || true
  wait "$PROGRESS_PID" 2>/dev/null || true
  PROGRESS_PID=0
  (( rc == 0 )) || { write_status failed "${domain} workload exited rc=${rc}; no automatic retry." "$rc"; return "$rc"; }
  [[ -s "$output/summary.json" ]] || { write_status failed "${domain} rc=0 but summary missing." 10; return 10; }
  if [[ "$domain" == toys ]]; then COMPLETED_DOMAINS=toys; else COMPLETED_DOMAINS=toys,beauty; fi
  write_status running "${domain} completed successfully."
}

worker() {
  GPU=${1:?}
  STARTED_AT=${2:?}
  RUNNER_PID=$$
  local free_mib telemetry_pid=0 rc=0
  trap '' HUP
  trap 'stop_owned_children; [[ $telemetry_pid -eq 0 ]] || kill "$telemetry_pid" 2>/dev/null || true; STAGE=stopped; write_status stopped "Stopped by TERM/INT; no automatic retry." 143; exit 143' TERM INT
  cd "$ROOT" || exit 2
  STAGE=preflight
  write_status running "Syntax, Stage14 tests, frozen inputs, and GPU admission active."
  bash -n experiment/phase14/run_stage14_0c_same_backbone_verifier.sh || { STAGE=finished; write_status failed "Bash syntax failed." 4; exit 4; }
  "$PYTHON" -m py_compile experiment/phase14/protocol/same_backbone_verifier.py || { STAGE=finished; write_status failed "Python syntax failed." 5; exit 5; }
  "$PYTHON" -m pytest -q experiment/phase14/tests || { STAGE=finished; write_status failed "Stage14 tests failed." 6; exit 6; }
  for path in \
    artifacts/phase13/explore/v0_toys/gram_logs/Toys_cold50/0_20260808_2256/config.json \
    artifacts/phase13/explore/v0_toys/gram_logs/Toys_cold50/0_20260808_2256/id_0_rec_30/model_rec_phase_1_epoch_30.pt \
    GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt \
    artifacts/phase13/explore/v1_r2_toys_p0/predictions_validation.jsonl \
    artifacts/phase13/explore/v0_beauty/gram_logs/Beauty_cold50/0_20260810_1004/config.json \
    artifacts/phase13/explore/v0_beauty/gram_logs/Beauty_cold50/0_20260810_1004/id_0_rec_30/model_rec_phase_1_epoch_30.pt \
    GRAM/rec_datasets/Beauty_cold50/item_generative_indexing_hierarchy_v1_c128_l7_len32768_split.txt \
    artifacts/phase13/explore/v1_r2_beauty_p0/predictions_validation.jsonl; do
    [[ -s "$path" ]] || { STAGE=finished; write_status failed "Missing frozen input: $path" 7; exit 7; }
  done
  [[ ! -e "$TOYS_OUTPUT/summary.json" && ! -e "$BEAUTY_OUTPUT/summary.json" ]] || { STAGE=finished; write_status failed "Refusing to overwrite formal summary." 8; exit 8; }
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GPU" 2>/dev/null | tr -d ' ' || true)
  [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )) || { STAGE=finished; write_status blocked "GPU admission failed; requires ${MIN_FREE_MIB} MiB free." 9; exit 9; }
  telemetry & telemetry_pid=$!
  run_domain toys "$TOYS_OUTPUT" \
    GRAM/rec_datasets/Toys_cold50 \
    artifacts/phase13/explore/v0_toys/gram_logs/Toys_cold50/0_20260808_2256/config.json \
    artifacts/phase13/explore/v0_toys/gram_logs/Toys_cold50/0_20260808_2256/id_0_rec_30/model_rec_phase_1_epoch_30.pt \
    GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt \
    artifacts/phase13/explore/v1_r2_toys_p0/predictions_validation.jsonl || rc=$?
  if (( rc == 0 )); then
    run_domain beauty "$BEAUTY_OUTPUT" \
      GRAM/rec_datasets/Beauty_cold50 \
      artifacts/phase13/explore/v0_beauty/gram_logs/Beauty_cold50/0_20260810_1004/config.json \
      artifacts/phase13/explore/v0_beauty/gram_logs/Beauty_cold50/0_20260810_1004/id_0_rec_30/model_rec_phase_1_epoch_30.pt \
      GRAM/rec_datasets/Beauty_cold50/item_generative_indexing_hierarchy_v1_c128_l7_len32768_split.txt \
      artifacts/phase13/explore/v1_r2_beauty_p0/predictions_validation.jsonl || rc=$?
  fi
  kill "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
  telemetry_pid=0
  (( rc == 0 )) || { STAGE=finished; exit "$rc"; }
  CURRENT_DOMAIN=none
  TOTAL_USERS=0
  STAGE=finished
  write_status completed "Formal Toys and Beauty validation-only verifier controls completed." 0
}

case "$ACTION" in
  start)
    [[ "$GPU" =~ ^[0-7]$ ]] || { echo "GPU must be 0..7" >&2; exit 2; }
    mkdir -p "$OUTPUT"
    [[ ! -e "$STATUS" ]] || { echo "Formal status already exists" >&2; exit 3; }
    STARTED_AT=$(date -Is)
    STAGE=starting
    write_status starting "Formal dual-domain background worker starting on GPU${GPU}."
    setsid bash "$SCRIPT" worker "$GPU" "$STARTED_AT" >> "$LOG" 2>&1 < /dev/null &
    child_pid=$!
    disown "$child_pid" 2>/dev/null || true
    echo "runner_pid: $child_pid"
    echo "status: $STATUS"
    ;;
  worker)
    worker "${2:?}" "${3:?}"
    ;;
  status)
    [[ -f "$STATUS" ]] && sed -n '1,160p' "$STATUS" || echo '{"status":"not_started"}'
    ;;
  *)
    echo "usage: $0 {start <gpu>|worker <gpu> <started_at>|status}" >&2
    exit 2
    ;;
esac
