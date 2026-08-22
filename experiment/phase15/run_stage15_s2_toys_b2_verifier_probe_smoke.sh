#!/usr/bin/env bash
set -u -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ACTION=${1:-status}
GPU=${2:-}
OUTPUT=${OUTPUT_OVERRIDE:-"$ROOT/artifacts/phase15/s2_contract_smoke/toys/b2_verifier_probe_smoke"}
STATUS="$OUTPUT/status.json"
LOG="$OUTPUT/run.log"
TELEMETRY="$OUTPUT/gpu_telemetry.csv"
SUMMARY="$OUTPUT/summary.json"
SCRIPT="$ROOT/experiment/phase15/run_stage15_s2_toys_b2_verifier_probe_smoke.sh"
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
SESSION=${SESSION_OVERRIDE:-gram_stage15_s2_toys_b2_verifier_probe}
EXACT_START_COMMAND=${EXACT_START_COMMAND_OVERRIDE:-}
MIN_FREE_MIB=8192
EXPECTED_INCREMENTAL_MIB_UPPER_BOUND=7168
HARD_TIMEOUT_SECONDS=3600
STARTED_AT=""
RUNNER_PID=0
WORKLOAD_PID=0
STAGE=not_started

write_status() {
  local state=$1 reason=$2 rc=${3:--1} tmp="$STATUS.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_STAGE15_S2_TOYS_B2_VERIFIER_PROBE_SMOKE","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%d,"workload_pid":%d,"workload_rc":%d,"physical_gpu":%s,"minimum_free_mib":%d,"expected_incremental_gpu_mib_upper_bound":%d,"hard_timeout_seconds":%d,"verifier_users":16,"candidate_budget":32,"probe_transitions":64,"split":"validation_for_verifier_train_only_for_probe","model_training":false,"original_user_sequence_opened":false,"test_predictions_opened":false,"automatic_retry":false,"resource_mode":"user_assigned_remaining_memory_only_no_process_changes","exact_start_command":"%s","output_dir":"%s","log_path":"%s","summary_path":"%s"}\n' \
    "$state" "$STAGE" "$reason" "$STARTED_AT" "$(date -Is)" "$RUNNER_PID" "$WORKLOAD_PID" "$rc" "${GPU:--1}" "$MIN_FREE_MIB" "$EXPECTED_INCREMENTAL_MIB_UPPER_BOUND" "$HARD_TIMEOUT_SECONDS" "$EXACT_START_COMMAND" "${OUTPUT#$ROOT/}" "${LOG#$ROOT/}" "${SUMMARY#$ROOT/}" > "$tmp"
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
  GPU=${1:?missing gpu}
  STARTED_AT=${2:?missing started_at}
  RUNNER_PID=$$
  local free_mib telemetry_pid=0 rc=0 verdict=""
  trap '' HUP
  trap '[[ $WORKLOAD_PID -eq 0 ]] || kill -TERM "$WORKLOAD_PID" 2>/dev/null || true; [[ $telemetry_pid -eq 0 ]] || kill "$telemetry_pid" 2>/dev/null || true; STAGE=stopped; write_status stopped "Stopped by signal; no automatic retry." 143; exit 143' TERM INT
  cd "$ROOT" || exit 2

  STAGE=preflight
  write_status running "Checking CPU Gate, hook contracts, frozen inputs, and GPU admission."
  bash -n "$SCRIPT" || { STAGE=finished; write_status failed "Bash syntax check failed." 4; exit 4; }
  "$PYTHON" -m py_compile \
    experiment/phase15/protocol/common_adapter.py \
    experiment/phase15/protocol/specgr_gram_adapter.py \
    experiment/phase15/protocol/genrecedit_gram_adapter.py \
    experiment/phase15/protocol/toys_b2_verifier_probe_smoke.py \
    || { STAGE=finished; write_status failed "Python syntax check failed." 5; exit 5; }
  "$PYTHON" -m unittest discover -s experiment/phase15/tests -v \
    || { STAGE=finished; write_status failed "Stage15 contract tests failed." 6; exit 6; }
  for path in \
    artifacts/phase15/s2_dual_domain_preflight/projected_data/Toys_cold50/user_sequence_train_validation.txt \
    artifacts/phase15/s2_contract_smoke/toys/b2_b3_adapter_contract_state/summary.json \
    artifacts/phase15/s2_contract_smoke/toys/b2_b3_adapter_contract_state/specgr_gram/index/manifest.json \
    artifacts/phase15/s2_contract_smoke/toys/b2_b3_adapter_contract_state/genrecedit_gram/edit_requests/position_map.json \
    artifacts/phase13/explore/v0_toys/gram_logs/Toys_cold50/0_20260808_2256/config.json \
    artifacts/phase13/explore/v0_toys/gram_logs/Toys_cold50/0_20260808_2256/id_0_rec_30/model_rec_phase_1_epoch_30.pt \
    GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt \
    GRAM/rec_datasets/Toys_cold50/item_plain_text.txt \
    GRAM/rec_datasets/Toys_cold50/similar_item_sasrec.txt \
    GRAM/rec_datasets/Toys_cold50/cold_split_meta/cold_items.txt \
    GRAM/rec_datasets/Toys_cold50/cold_split_meta/warm_items.txt; do
    [[ -s "$path" ]] || { STAGE=finished; write_status failed "Missing frozen input: $path" 7; exit 7; }
  done
  "$PYTHON" -c 'import json,sys; assert json.load(open(sys.argv[1]))["verdict"] == "PASS_B2_B3_INPUT_CONTRACT"' \
    artifacts/phase15/s2_contract_smoke/toys/b2_b3_adapter_contract_state/summary.json \
    || { STAGE=finished; write_status failed "B2/B3 CPU input Gate is not PASS." 8; exit 8; }
  [[ ! -e "$SUMMARY" && ! -e "$OUTPUT/config.json" && ! -e "$OUTPUT/dataset_view_verifier" && ! -e "$OUTPUT/dataset_view_probe" ]] \
    || { STAGE=finished; write_status failed "Refusing to overwrite existing smoke artifacts." 10; exit 10; }
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GPU" 2>/dev/null | tr -d ' ' || true)
  [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )) \
    || { STAGE=finished; write_status blocked "GPU${GPU} admission failed; requires ${MIN_FREE_MIB} MiB free; no resource changes made." 9; exit 9; }

  telemetry & telemetry_pid=$!
  STAGE=toys_b2_verifier_probe
  write_status running "Frozen GRAM B2 verifier/B3 train-only probe smoke active on GPU${GPU}."
  timeout --signal=TERM --kill-after=30 "$HARD_TIMEOUT_SECONDS" \
    env CUDA_VISIBLE_DEVICES="$GPU" CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    "$PYTHON" experiment/phase15/protocol/toys_b2_verifier_probe_smoke.py \
      --projected-sequences artifacts/phase15/s2_dual_domain_preflight/projected_data/Toys_cold50/user_sequence_train_validation.txt \
      --source-dataset-dir GRAM/rec_datasets/Toys_cold50 \
      --historical-config artifacts/phase13/explore/v0_toys/gram_logs/Toys_cold50/0_20260808_2256/config.json \
      --checkpoint artifacts/phase13/explore/v0_toys/gram_logs/Toys_cold50/0_20260808_2256/id_0_rec_30/model_rec_phase_1_epoch_30.pt \
      --item-path-file GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt \
      --contract-state artifacts/phase15/s2_contract_smoke/toys/b2_b3_adapter_contract_state \
      --output-dir "$OUTPUT" --device cuda:0 \
      --verifier-users 16 --candidates-per-split 16 --candidate-chunk-size 8 \
      --probe-transitions 64 --probe-long-path-minimum 16 --probe-batch-size 4 --seed 1502 &
  WORKLOAD_PID=$!
  write_status running "B2 verifier/B3 probe workload active; no existing process was modified."
  wait "$WORKLOAD_PID" || rc=$?
  WORKLOAD_PID=0
  kill "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
  telemetry_pid=0
  if (( rc != 0 )); then
    STAGE=finished
    write_status failed "B2 verifier/B3 probe workload exited rc=${rc}; no automatic retry." "$rc"
    exit "$rc"
  fi
  [[ -s "$SUMMARY" ]] || { STAGE=finished; write_status failed "Workload exited 0 without summary.json." 11; exit 11; }
  verdict=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$SUMMARY") \
    || { STAGE=finished; write_status failed "Could not read smoke verdict." 12; exit 12; }
  STAGE=finished
  write_status completed "$verdict" 0
}

case "$ACTION" in
  start)
    [[ "$GPU" =~ ^[0-7]$ ]] || { echo "usage: $0 start <gpu 0-7>" >&2; exit 2; }
    [[ -n "$EXACT_START_COMMAND" ]] || EXACT_START_COMMAND="bash experiment/phase15/run_stage15_s2_toys_b2_verifier_probe_smoke.sh start $GPU"
    mkdir -p "$OUTPUT"
    [[ ! -e "$STATUS" ]] || { echo "refusing existing status: $STATUS" >&2; exit 3; }
    tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists: $SESSION" >&2; exit 3; }
    STARTED_AT=$(date -Is)
    STAGE=starting
    write_status starting "Background B2 verifier/B3 probe smoke starting on GPU${GPU}."
    printf -v launch 'env OUTPUT_OVERRIDE=%q SESSION_OVERRIDE=%q EXACT_START_COMMAND_OVERRIDE=%q bash %q worker %q %q >> %q 2>&1' "$OUTPUT" "$SESSION" "$EXACT_START_COMMAND" "$SCRIPT" "$GPU" "$STARTED_AT" "$LOG"
    tmux new-session -d -s "$SESSION" "$launch"
    echo "started $SESSION on GPU${GPU}"
    echo "status: $STATUS"
    ;;
  worker)
    worker "${2:?missing gpu}" "${3:?missing started_at}"
    ;;
  status)
    [[ -f "$STATUS" ]] && sed -n '1,160p' "$STATUS" || echo '{"status":"not_started"}'
    ;;
  *)
    echo "usage: $0 {start <gpu>|worker <gpu> <started_at>|status}" >&2
    exit 2
    ;;
esac
