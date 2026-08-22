#!/usr/bin/env bash
set -u -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ACTION=${1:-status}
GPU=${2:-}
OUTPUT=${OUTPUT_OVERRIDE:-"$ROOT/artifacts/phase15/s2_contract_smoke/toys/b3_edit_state_smoke"}
STATUS="$OUTPUT/status.json"
LOG="$OUTPUT/run.log"
TELEMETRY="$OUTPUT/gpu_telemetry.csv"
SUMMARY="$OUTPUT/summary.json"
SCRIPT="$ROOT/experiment/phase15/run_stage15_s2_toys_b3_edit_state_smoke.sh"
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
SESSION=${SESSION_OVERRIDE:-gram_stage15_s2_toys_b3_edit_state}
MIN_FREE_MIB=8192
EXPECTED_INCREMENTAL_MIB_UPPER_BOUND=6144
HARD_TIMEOUT_SECONDS=3600
EXACT_START_COMMAND=${EXACT_START_COMMAND_OVERRIDE:-}
STARTED_AT=""
RUNNER_PID=0
WORKLOAD_PID=0
STAGE=not_started

write_status() {
  local state=$1 reason=$2 rc=${3:--1} tmp="$STATUS.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_STAGE15_S2_TOYS_B3_EDIT_STATE_SMOKE","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%d,"workload_pid":%d,"workload_rc":%d,"physical_gpu":%s,"minimum_free_mib":%d,"expected_incremental_gpu_mib_upper_bound":%d,"hard_timeout_seconds":%d,"covariance_transitions":256,"positionwise_requests":302400,"delta_smoke_requests":24,"lexical_positions":6,"split":"train_only_for_covariance_and_requests","base_model_training":false,"base_checkpoint_mutation":false,"temporary_delta_application_restored":true,"original_user_sequence_opened":false,"similar_item_sasrec_opened":false,"test_predictions_opened":false,"automatic_retry":false,"resource_mode":"user_assigned_remaining_memory_only_no_process_changes","exact_start_command":"%s","output_dir":"%s","log_path":"%s","summary_path":"%s"}\n' \
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
  EXACT_START_COMMAND=${3:?missing exact command}
  RUNNER_PID=$$
  local free_mib telemetry_pid=0 rc=0 verdict=""
  trap '' HUP
  trap '[[ $WORKLOAD_PID -eq 0 ]] || kill -TERM "$WORKLOAD_PID" 2>/dev/null || true; [[ $telemetry_pid -eq 0 ]] || kill "$telemetry_pid" 2>/dev/null || true; STAGE=stopped; write_status stopped "Stopped by signal; no automatic retry." 143; exit 143' TERM INT
  cd "$ROOT" || exit 2

  STAGE=preflight
  write_status running "Checking B3 contracts, frozen inputs, and GPU admission."
  bash -n "$SCRIPT" || { STAGE=finished; write_status failed "Bash syntax check failed." 4; exit 4; }
  "$PYTHON" -m py_compile \
    experiment/phase15/protocol/common_adapter.py \
    experiment/phase15/protocol/genrecedit_gram_adapter.py \
    experiment/phase15/protocol/toys_b3_edit_state_smoke.py \
    || { STAGE=finished; write_status failed "Python syntax check failed." 5; exit 5; }
  "$PYTHON" -m unittest discover -s experiment/phase15/tests -v \
    || { STAGE=finished; write_status failed "Stage15 contract tests failed." 6; exit 6; }
  for path in \
    artifacts/phase15/s2_dual_domain_preflight/projected_data/Toys_cold50/user_sequence_train_validation.txt \
    artifacts/phase15/s2_contract_smoke/toys/b2_b3_adapter_contract_state/summary.json \
    artifacts/phase15/s2_contract_smoke/toys/b2_b3_adapter_contract_state/genrecedit_gram/edit_requests/pseudo_contexts.jsonl \
    artifacts/phase15/s2_contract_smoke/toys/b2_verifier_probe_smoke_attempt4/summary.json \
    artifacts/phase15/s2_contract_smoke/toys/b2_verifier_probe_smoke_attempt4/genrecedit_gram/probe/layer_probe.json \
    artifacts/phase13/explore/v0_toys/gram_logs/Toys_cold50/0_20260808_2256/config.json \
    artifacts/phase13/explore/v0_toys/gram_logs/Toys_cold50/0_20260808_2256/id_0_rec_30/model_rec_phase_1_epoch_30.pt \
    GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt \
    GRAM/rec_datasets/Toys_cold50/item_plain_text.txt \
    GRAM/rec_datasets/Toys_cold50/cold_split_meta/cold_items.txt \
    GRAM/rec_datasets/Toys_cold50/cold_split_meta/warm_items.txt; do
    [[ -s "$path" ]] || { STAGE=finished; write_status failed "Missing frozen input: $path" 7; exit 7; }
  done
  "$PYTHON" -c 'import json,sys; assert json.load(open(sys.argv[1]))["verdict"] == "PASS_B2_B3_INPUT_CONTRACT"' \
    artifacts/phase15/s2_contract_smoke/toys/b2_b3_adapter_contract_state/summary.json \
    || { STAGE=finished; write_status failed "B2/B3 CPU input Gate is not PASS." 8; exit 8; }
  "$PYTHON" -c 'import json,sys; assert json.load(open(sys.argv[1]))["verdict"] == "PASS_B2_VERIFIER_GPU_HOOK_AND_B3_TRAIN_ONLY_PROBE"' \
    artifacts/phase15/s2_contract_smoke/toys/b2_verifier_probe_smoke_attempt4/summary.json \
    || { STAGE=finished; write_status failed "B3 train-only layer probe Gate is not PASS." 8; exit 8; }
  [[ ! -e "$SUMMARY" && ! -e "$OUTPUT/config.json" && ! -e "$OUTPUT/genrecedit_gram" ]] \
    || { STAGE=finished; write_status failed "Refusing to overwrite existing B3 artifacts." 10; exit 10; }
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GPU" 2>/dev/null | tr -d ' ' || true)
  [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )) \
    || { STAGE=finished; write_status blocked "GPU${GPU} admission failed; requires ${MIN_FREE_MIB} MiB free; no resource changes made." 9; exit 9; }

  telemetry & telemetry_pid=$!
  STAGE=toys_b3_edit_state
  write_status running "Train-only B3 covariance, edit request, deltaW, and trigger smoke active on GPU${GPU}."
  timeout --signal=TERM --kill-after=30 "$HARD_TIMEOUT_SECONDS" \
    env CUDA_VISIBLE_DEVICES="$GPU" CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    "$PYTHON" experiment/phase15/protocol/toys_b3_edit_state_smoke.py \
      --projected-sequences artifacts/phase15/s2_dual_domain_preflight/projected_data/Toys_cold50/user_sequence_train_validation.txt \
      --source-dataset-dir GRAM/rec_datasets/Toys_cold50 \
      --historical-config artifacts/phase13/explore/v0_toys/gram_logs/Toys_cold50/0_20260808_2256/config.json \
      --checkpoint artifacts/phase13/explore/v0_toys/gram_logs/Toys_cold50/0_20260808_2256/id_0_rec_30/model_rec_phase_1_epoch_30.pt \
      --item-path-file GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt \
      --contract-state artifacts/phase15/s2_contract_smoke/toys/b2_b3_adapter_contract_state \
      --probe-state artifacts/phase15/s2_contract_smoke/toys/b2_verifier_probe_smoke_attempt4 \
      --output-dir "$OUTPUT" --device cuda:0 \
      --covariance-transitions 256 --covariance-long-path-minimum 32 --covariance-batch-size 32 \
      --requests-per-position 4 --z-steps 30 --z-learning-rate 0.5 \
      --z-weight-decay 0.2 --z-max-norm 8000 --legal-probability-threshold 0.3 \
      --covariance-ridge 0.01 --preservation-lambda 10000 --seed 1502 &
  WORKLOAD_PID=$!
  write_status running "B3 workload active; frozen GRAM has no optimizer and temporary delta checks must restore exactly."
  wait "$WORKLOAD_PID" || rc=$?
  WORKLOAD_PID=0
  kill "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
  telemetry_pid=0
  if (( rc != 0 )); then
    STAGE=finished
    if (( rc == 124 || rc == 137 )); then
      write_status timeout "B3 workload reached hard timeout; no automatic retry." "$rc"
    else
      write_status failed "B3 workload exited rc=${rc}; no automatic retry." "$rc"
    fi
    exit "$rc"
  fi
  [[ -s "$SUMMARY" ]] || { STAGE=finished; write_status failed "Workload exited 0 without summary.json." 11; exit 11; }
  verdict=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$SUMMARY") \
    || { STAGE=finished; write_status failed "Could not read B3 verdict." 12; exit 12; }
  STAGE=finished
  write_status completed "$verdict" 0
}

case "$ACTION" in
  start)
    [[ "$GPU" =~ ^[0-7]$ ]] || { echo "usage: $0 start <gpu 0-7>" >&2; exit 2; }
    [[ -n "$EXACT_START_COMMAND" ]] || EXACT_START_COMMAND="bash experiment/phase15/run_stage15_s2_toys_b3_edit_state_smoke.sh start $GPU"
    mkdir -p "$OUTPUT"
    [[ ! -e "$STATUS" ]] || { echo "refusing existing status: $STATUS" >&2; exit 3; }
    tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists: $SESSION" >&2; exit 3; }
    STARTED_AT=$(date -Is)
    STAGE=starting
    write_status starting "Background B3 edit state smoke starting on GPU${GPU}."
    printf -v launch 'env OUTPUT_OVERRIDE=%q SESSION_OVERRIDE=%q EXACT_START_COMMAND_OVERRIDE=%q bash %q worker %q %q %q >> %q 2>&1' "$OUTPUT" "$SESSION" "$EXACT_START_COMMAND" "$SCRIPT" "$GPU" "$STARTED_AT" "$EXACT_START_COMMAND" "$LOG"
    tmux new-session -d -s "$SESSION" "$launch"
    echo "started $SESSION on GPU${GPU}"
    echo "status: $STATUS"
    ;;
  worker)
    worker "${2:?missing gpu}" "${3:?missing started_at}" "${4:?missing exact command}"
    ;;
  status)
    [[ -f "$STATUS" ]] && sed -n '1,160p' "$STATUS" || echo '{"status":"not_started"}'
    ;;
  *)
    echo "usage: $0 {start <gpu>|worker <gpu> <started_at> <exact_command>|status}" >&2
    exit 2
    ;;
esac
