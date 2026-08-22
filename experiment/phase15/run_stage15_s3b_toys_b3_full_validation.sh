#!/usr/bin/env bash
set -u -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ACTION=${1:-status}
GPU=${2:-}
OUTPUT=${OUTPUT_OVERRIDE:-"$ROOT/artifacts/phase15/s3_toys/full_validation/b3_branching_seed0"}
STATUS="$OUTPUT/status.json"
LOG="$OUTPUT/run.log"
TELEMETRY="$OUTPUT/gpu_telemetry.csv"
SUMMARY="$OUTPUT/summary.json"
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
SESSION=${SESSION_OVERRIDE:-gram_stage15_s3b_toys_b3_full_validation}
MIN_FREE_MIB=16384
EXPECTED_INCREMENTAL_MIB_UPPER_BOUND=12288
HARD_TIMEOUT_SECONDS=86400
STARTED_AT=""
RUNNER_PID=0
WORKLOAD_PID=0
STAGE=not_started
EXACT_START_COMMAND=""

write_status() {
  local state=$1 reason=$2 rc=${3:--1} tmp="$STATUS.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_STAGE15_S3B_TOYS_B3_FULL_VALIDATION_SEED0","status":"%s","status_code":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%d,"workload_pid":%d,"workload_rc":%d,"exit_code":%d,"exit_code_pending":%s,"physical_gpu":%s,"minimum_free_mib":%d,"expected_incremental_gpu_mib_upper_bound":%d,"hard_timeout_seconds":%d,"events":8789,"train_transitions":4096,"cold_catalog_items":5963,"beam_size":50,"arms":"b0,b1,b3","split":"validation","original_user_sequence_opened":false,"test_opened":false,"test_read":false,"automatic_retry":false,"resource_mode":"user_assigned_remaining_memory_only_no_process_changes","exact_start_command":"%s","output_dir":"%s","log_path":"%s","summary_path":"%s"}\n' \
    "$state" "${state^^}" "$STAGE" "$reason" "$STARTED_AT" "$(date -Is)" "$RUNNER_PID" "$WORKLOAD_PID" "$rc" "$rc" "$([[ "$state" == running || "$state" == starting ]] && echo true || echo false)" "${GPU:--1}" "$MIN_FREE_MIB" "$EXPECTED_INCREMENTAL_MIB_UPPER_BOUND" "$HARD_TIMEOUT_SECONDS" "$EXACT_START_COMMAND" "${OUTPUT#$ROOT/}" "${LOG#$ROOT/}" "${SUMMARY#$ROOT/}" > "$tmp"
  mv "$tmp" "$STATUS"
}

telemetry() {
  printf 'timestamp,index,memory_used_mib,memory_free_mib,utilization_gpu_percent\n' > "$TELEMETRY"
  while true; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits --id="$GPU" >> "$TELEMETRY" 2>/dev/null || true
    sleep 30
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
  write_status running "Checking B3 S15-3A Gate, frozen full-validation inputs, tests, and GPU admission."
  bash -n experiment/phase15/run_stage15_s3b_toys_b3_full_validation.sh \
    || { STAGE=finished; write_status failed "Bash syntax check failed." 4; exit 4; }
  "$PYTHON" -m py_compile \
    experiment/phase15/protocol/genrecedit_gram_adapter.py \
    experiment/phase15/protocol/toys_s3b_b3_full_validation.py \
    experiment/phase15/protocol/refresh_background_status.py \
    || { STAGE=finished; write_status failed "Python syntax check failed." 5; exit 5; }
  "$PYTHON" -m unittest discover -s experiment/phase15/tests -v \
    || { STAGE=finished; write_status failed "Stage15 contract tests failed." 6; exit 6; }
  for path in \
    artifacts/phase15/s2_dual_domain_preflight/projected_data/Toys_cold50/user_sequence_train_validation.txt \
    artifacts/phase15/s3_toys/admission/b3_branching_recovery_attempt5/summary.json \
    artifacts/phase15/s2_contract_smoke/toys/b0_projection_parity_smoke_attempt3/summary.json \
    artifacts/phase13/explore/v1_r2_toys_p0/predictions_validation.jsonl \
    artifacts/phase13/explore/v1_r2_toys_p0/summary.json \
    artifacts/phase13/explore/v1_r2_toys_p0/resolver.pt \
    artifacts/phase13/explore/v0_toys/gram_logs/Toys_cold50/0_20260808_2256/config.json \
    artifacts/phase13/explore/v0_toys/gram_logs/Toys_cold50/0_20260808_2256/id_0_rec_30/model_rec_phase_1_epoch_30.pt \
    artifacts/phase14/m2/pretrained/t5-small/config.json \
    artifacts/phase13/embeddings/Toys_bge_large_en_v1_5_cls_l2.pt \
    GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt \
    GRAM/rec_datasets/Toys_cold50/item_plain_text.txt \
    GRAM/rec_datasets/Toys_cold50/similar_item_sasrec.txt \
    GRAM/rec_datasets/Toys_cold50/cold_split_meta/cold_items.txt \
    GRAM/rec_datasets/Toys_cold50/cold_split_meta/warm_items.txt; do
    [[ -s "$path" ]] || { STAGE=finished; write_status failed "Missing frozen input: $path" 7; exit 7; }
  done
  "$PYTHON" -c 'import json,sys; assert json.load(open(sys.argv[1]))["verdict"] == "PASS_S15_3A_B2_B3_ITEM_DISJOINT_ADMISSION"' \
    artifacts/phase15/s3_toys/admission/b3_branching_recovery_attempt5/summary.json \
    || { STAGE=finished; write_status failed "Exploratory B3 S15-3A Gate is not PASS." 8; exit 8; }
  "$PYTHON" -c 'import json,sys; assert json.load(open(sys.argv[1]))["verdict"] == "PASS_B0_PROJECTION_PARITY"' \
    artifacts/phase15/s2_contract_smoke/toys/b0_projection_parity_smoke_attempt3/summary.json \
    || { STAGE=finished; write_status failed "B0 projection Gate is not PASS." 8; exit 8; }
  [[ ! -e "$SUMMARY" && ! -e "$OUTPUT/config.json" && ! -e "$OUTPUT/b3_genrecedit" ]] \
    || { STAGE=finished; write_status failed "Refusing to overwrite existing B3 full-validation artifacts." 10; exit 10; }
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GPU" 2>/dev/null | tr -d ' ' || true)
  [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )) \
    || { STAGE=finished; write_status blocked "GPU${GPU} admission failed; requires ${MIN_FREE_MIB} MiB free; no process changes made." 9; exit 9; }

  telemetry & telemetry_pid=$!
  STAGE=s3b_toys_b3_full_validation
  write_status running "B0/B1/B3 full 8,789-event validation active on GPU${GPU}."
  timeout --signal=TERM --kill-after=30 "$HARD_TIMEOUT_SECONDS" \
    env CUDA_VISIBLE_DEVICES="$GPU" CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    "$PYTHON" experiment/phase15/protocol/toys_s3b_b3_full_validation.py \
      --projected-sequences artifacts/phase15/s2_dual_domain_preflight/projected_data/Toys_cold50/user_sequence_train_validation.txt \
      --historical-config artifacts/phase13/explore/v0_toys/gram_logs/Toys_cold50/0_20260808_2256/config.json \
      --backbone-path artifacts/phase14/m2/pretrained/t5-small \
      --checkpoint artifacts/phase13/explore/v0_toys/gram_logs/Toys_cold50/0_20260808_2256/id_0_rec_30/model_rec_phase_1_epoch_30.pt \
      --frozen-b0-b1-predictions artifacts/phase13/explore/v1_r2_toys_p0/predictions_validation.jsonl \
      --item-path-file GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt \
      --item-text-file GRAM/rec_datasets/Toys_cold50/item_plain_text.txt \
      --similar-items-file GRAM/rec_datasets/Toys_cold50/similar_item_sasrec.txt \
      --item-embeddings artifacts/phase13/embeddings/Toys_bge_large_en_v1_5_cls_l2.pt \
      --cold-items GRAM/rec_datasets/Toys_cold50/cold_split_meta/cold_items.txt \
      --warm-items GRAM/rec_datasets/Toys_cold50/cold_split_meta/warm_items.txt \
      --s3a-b3-summary artifacts/phase15/s3_toys/admission/b3_branching_recovery_attempt5/summary.json \
      --b0-parity-summary artifacts/phase15/s2_contract_smoke/toys/b0_projection_parity_smoke_attempt3/summary.json \
      --b1-source-summary artifacts/phase13/explore/v1_r2_toys_p0/summary.json \
      --b1-state artifacts/phase13/explore/v1_r2_toys_p0/resolver.pt \
      --output-dir "$OUTPUT" --device cuda:0 \
      --train-transitions 4096 --covariance-transitions 256 --covariance-long-path-minimum 32 \
      --covariance-batch-size 32 --contexts-per-pseudo-cold 10 --requests-per-position 4 --z-steps 30 \
      --beam-size 50 --bootstrap-resamples 10000 --bootstrap-seed 20260822 --seed 1502 &
  WORKLOAD_PID=$!
  write_status running "B3 full-validation workload active; validation is evaluation-only, test remains sealed, and no automatic retry is allowed."
  wait "$WORKLOAD_PID" || rc=$?
  WORKLOAD_PID=0
  kill "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
  telemetry_pid=0
  if (( rc != 0 )); then
    STAGE=finished
    if (( rc == 124 || rc == 137 )); then
      write_status timeout "B3 full validation reached hard timeout; no automatic retry." "$rc"
    else
      write_status failed "B3 full validation exited rc=${rc}; no automatic retry." "$rc"
    fi
    exit "$rc"
  fi
  [[ -s "$SUMMARY" ]] || { STAGE=finished; write_status failed "Workload exited 0 without summary.json." 11; exit 11; }
  verdict=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$SUMMARY") \
    || { STAGE=finished; write_status failed "Could not read B3 full-validation verdict." 12; exit 12; }
  STAGE=finished
  write_status completed "$verdict" 0
}

case "$ACTION" in
  start)
    [[ "$GPU" =~ ^[0-7]$ ]] || { echo "usage: $0 start <gpu 0-7>" >&2; exit 2; }
    mkdir -p "$OUTPUT"
    [[ ! -e "$STATUS" ]] || { echo "refusing existing status: $STATUS" >&2; exit 3; }
    tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists: $SESSION" >&2; exit 3; }
    STARTED_AT=$(date -Is)
    EXACT_START_COMMAND=${EXACT_START_COMMAND_OVERRIDE:-"bash experiment/phase15/run_stage15_s3b_toys_b3_full_validation.sh start $GPU"}
    STAGE=starting
    write_status starting "Background B3 S15-3B full validation starting on GPU${GPU}."
    printf -v launch 'env OUTPUT_OVERRIDE=%q SESSION_OVERRIDE=%q EXACT_START_COMMAND_OVERRIDE=%q bash %q worker %q %q %q >> %q 2>&1' "$OUTPUT" "$SESSION" "$EXACT_START_COMMAND" "$ROOT/experiment/phase15/run_stage15_s3b_toys_b3_full_validation.sh" "$GPU" "$STARTED_AT" "$EXACT_START_COMMAND" "$LOG"
    if ! tmux new-session -d -s "$SESSION" "$launch"; then
      STAGE=finished
      write_status failed "Could not create background tmux session; workload was not started." 13
      exit 13
    fi
    echo "started $SESSION on GPU${GPU}"
    echo "status: $STATUS"
    ;;
  worker)
    worker "${2:?missing gpu}" "${3:?missing started_at}" "${4:?missing exact command}"
    ;;
  status)
    if [[ -f "$STATUS" ]]; then
      "$PYTHON" experiment/phase15/protocol/refresh_background_status.py \
        --status "$STATUS" --log "$LOG" --total 8789 --summary "$SUMMARY"
    else
      echo '{"status":"not_started","status_code":"NOT_STARTED","exit_code":-1,"exit_code_pending":false,"test_read":false,"progress_current":0,"progress_total":8789,"progress_unit":"events"}'
    fi
    ;;
  *)
    echo "usage: $0 {start <gpu>|worker <gpu> <started_at> <exact_command>|status}" >&2
    exit 2
    ;;
esac
