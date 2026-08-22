#!/usr/bin/env bash
set -u -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ACTION=${1:-status}
GPU=${2:-}
OUTPUT=${OUTPUT_OVERRIDE:-"$ROOT/artifacts/phase15/s3_toys/admission/item_disjoint_b2_b3"}
STATUS="$OUTPUT/status.json"
LOG="$OUTPUT/run.log"
TELEMETRY="$OUTPUT/gpu_telemetry.csv"
SUMMARY="$OUTPUT/summary.json"
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
SESSION=${SESSION_OVERRIDE:-gram_stage15_s3a_toys_item_disjoint_admission}
MIN_FREE_MIB=16384
EXPECTED_INCREMENTAL_MIB_UPPER_BOUND=12288
HARD_TIMEOUT_SECONDS=14400
STARTED_AT=""
RUNNER_PID=0
WORKLOAD_PID=0
STAGE=not_started
EXACT_START_COMMAND=""
ADMISSION_ARMS=${ADMISSION_ARMS_OVERRIDE:-b0,b2,b3}

write_status() {
  local state=$1 reason=$2 rc=${3:--1} tmp="$STATUS.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_STAGE15_S3A_TOYS_ITEM_DISJOINT_ADMISSION","status":"%s","status_code":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%d,"workload_pid":%d,"workload_rc":%d,"exit_code":%d,"exit_code_pending":%s,"physical_gpu":%s,"minimum_free_mib":%d,"expected_incremental_gpu_mib_upper_bound":%d,"hard_timeout_seconds":%d,"train_transitions":4096,"eval_events":512,"candidate_budget":50,"beam_size":50,"arms":"%s","split":"stage14_item_disjoint_pseudo_cold_admission","historical_v0_checkpoint_used":false,"held_ground_truth_opened_for_training":false,"test_opened":false,"test_read":false,"automatic_retry":false,"resource_mode":"user_assigned_remaining_memory_only_no_process_changes","exact_start_command":"%s","output_dir":"%s","log_path":"%s","summary_path":"%s"}\n' \
    "$state" "${state^^}" "$STAGE" "$reason" "$STARTED_AT" "$(date -Is)" "$RUNNER_PID" "$WORKLOAD_PID" "$rc" "$rc" "$([[ "$state" == running || "$state" == starting ]] && echo true || echo false)" "${GPU:--1}" "$MIN_FREE_MIB" "$EXPECTED_INCREMENTAL_MIB_UPPER_BOUND" "$HARD_TIMEOUT_SECONDS" "$ADMISSION_ARMS" "$EXACT_START_COMMAND" "${OUTPUT#$ROOT/}" "${LOG#$ROOT/}" "${SUMMARY#$ROOT/}" > "$tmp"
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
  write_status running "Checking S15-2 Gates, Stage14 item-disjoint inputs, tests, and GPU admission."
  bash -n experiment/phase15/run_stage15_s3a_toys_item_disjoint_admission.sh \
    || { STAGE=finished; write_status failed "Bash syntax check failed." 4; exit 4; }
  "$PYTHON" -m py_compile \
    experiment/phase15/protocol/genrecedit_gram_adapter.py \
    experiment/phase15/protocol/toys_s3a_admission.py \
    || { STAGE=finished; write_status failed "Python syntax check failed." 5; exit 5; }
  "$PYTHON" -m unittest discover -s experiment/phase15/tests -v \
    || { STAGE=finished; write_status failed "Stage15 contract tests failed." 6; exit 6; }
  for path in \
    artifacts/phase14/m2/pseudo_cold_screen_toys_formal/summary.json \
    artifacts/phase14/m2/pseudo_cold_screen_toys_formal/clean_base.pt \
    artifacts/phase14/m2/pretrained/t5-small/config.json \
    artifacts/phase14/m2/pseudo_cold_audit_toys_v2/student_readable/filtered_train_sequences.jsonl \
    artifacts/phase14/m2/pseudo_cold_audit_toys_v2/held_ground_truth_DO_NOT_USE_FOR_TRAINING/pseudo_cold_events.jsonl \
    artifacts/phase14/m2/pseudo_cold_audit_toys_v2/pseudo_cold_items.txt \
    artifacts/phase15/s2_contract_smoke/toys/b2_drafter_state_smoke_attempt2/summary.json \
    artifacts/phase15/s2_contract_smoke/toys/b3_edit_state_smoke_attempt2/summary.json \
    artifacts/phase13/embeddings/Toys_bge_large_en_v1_5_cls_l2.pt \
    artifacts/phase13/explore/v0_toys/gram_logs/Toys_cold50/0_20260808_2256/config.json \
    GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt \
    GRAM/rec_datasets/Toys_cold50/item_plain_text.txt \
    GRAM/rec_datasets/Toys_cold50/similar_item_sasrec.txt \
    GRAM/rec_datasets/Toys_cold50/cold_split_meta/cold_items.txt; do
    [[ -s "$path" ]] || { STAGE=finished; write_status failed "Missing frozen input: $path" 7; exit 7; }
  done
  "$PYTHON" -c 'import json,sys; assert json.load(open(sys.argv[1]))["verdict"] == "PASS_B2_TRAIN_ONLY_DRAFTER_STATE_SMOKE"' \
    artifacts/phase15/s2_contract_smoke/toys/b2_drafter_state_smoke_attempt2/summary.json \
    || { STAGE=finished; write_status failed "S15-2 B2 state Gate is not PASS." 8; exit 8; }
  "$PYTHON" -c 'import json,sys; assert json.load(open(sys.argv[1]))["verdict"] == "PASS_B3_TRAIN_ONLY_EDIT_STATE_SMOKE"' \
    artifacts/phase15/s2_contract_smoke/toys/b3_edit_state_smoke_attempt2/summary.json \
    || { STAGE=finished; write_status failed "S15-2 B3 state Gate is not PASS." 8; exit 8; }
  [[ ! -e "$SUMMARY" && ! -e "$OUTPUT/config.json" && ! -e "$OUTPUT/b2_specgr" && ! -e "$OUTPUT/b3_genrecedit" ]] \
    || { STAGE=finished; write_status failed "Refusing to overwrite admission artifacts." 10; exit 10; }
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GPU" 2>/dev/null | tr -d ' ' || true)
  [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )) \
    || { STAGE=finished; write_status blocked "GPU${GPU} admission failed; requires ${MIN_FREE_MIB} MiB free; no resource changes made." 9; exit 9; }

  telemetry & telemetry_pid=$!
  STAGE=s3a_item_disjoint_admission
  write_status running "${ADMISSION_ARMS} 512-event item-disjoint admission active on GPU${GPU}."
  timeout --signal=TERM --kill-after=30 "$HARD_TIMEOUT_SECONDS" \
    env CUDA_VISIBLE_DEVICES="$GPU" CUBLAS_WORKSPACE_CONFIG=:4096:8 TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    "$PYTHON" experiment/phase15/protocol/toys_s3a_admission.py \
      --historical-config artifacts/phase13/explore/v0_toys/gram_logs/Toys_cold50/0_20260808_2256/config.json \
      --backbone-path artifacts/phase14/m2/pretrained/t5-small \
      --clean-base artifacts/phase14/m2/pseudo_cold_screen_toys_formal/clean_base.pt \
      --stage14-summary artifacts/phase14/m2/pseudo_cold_screen_toys_formal/summary.json \
      --train-sequences artifacts/phase14/m2/pseudo_cold_audit_toys_v2/student_readable/filtered_train_sequences.jsonl \
      --held-events artifacts/phase14/m2/pseudo_cold_audit_toys_v2/held_ground_truth_DO_NOT_USE_FOR_TRAINING/pseudo_cold_events.jsonl \
      --pseudo-cold-items artifacts/phase14/m2/pseudo_cold_audit_toys_v2/pseudo_cold_items.txt \
      --real-cold-items GRAM/rec_datasets/Toys_cold50/cold_split_meta/cold_items.txt \
      --item-path-file GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt \
      --item-text-file GRAM/rec_datasets/Toys_cold50/item_plain_text.txt \
      --similar-items-file GRAM/rec_datasets/Toys_cold50/similar_item_sasrec.txt \
      --item-embeddings artifacts/phase13/embeddings/Toys_bge_large_en_v1_5_cls_l2.pt \
      --s2-report-summary artifacts/phase15/s2_contract_smoke/toys/b3_edit_state_smoke_attempt2/summary.json \
      --output-dir "$OUTPUT" --device cuda:0 \
      --train-transitions 4096 --eval-events 512 --drafter-epochs 2 --drafter-batch-size 128 \
      --covariance-transitions 256 --covariance-long-path-minimum 32 --covariance-batch-size 32 \
      --contexts-per-pseudo-cold 10 --requests-per-position 4 --z-steps 30 \
      --beam-size 50 --draft-size 10 --draft-rounds 5 --verifier-threshold -1.6 \
      --candidate-chunk-size 10 --seed 1502 --arms "$ADMISSION_ARMS" &
  WORKLOAD_PID=$!
  write_status running "Admission workload active; held events remain evaluation-only and no automatic retry is allowed."
  wait "$WORKLOAD_PID" || rc=$?
  WORKLOAD_PID=0
  kill "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
  telemetry_pid=0
  if (( rc != 0 )); then
    STAGE=finished
    if (( rc == 124 || rc == 137 )); then
      write_status timeout "Admission reached hard timeout; no automatic retry." "$rc"
    else
      write_status failed "Admission exited rc=${rc}; no automatic retry." "$rc"
    fi
    exit "$rc"
  fi
  [[ -s "$SUMMARY" ]] || { STAGE=finished; write_status failed "Workload exited 0 without summary.json." 11; exit 11; }
  verdict=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$SUMMARY") \
    || { STAGE=finished; write_status failed "Could not read admission verdict." 12; exit 12; }
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
    EXACT_START_COMMAND=${EXACT_START_COMMAND_OVERRIDE:-"bash experiment/phase15/run_stage15_s3a_toys_item_disjoint_admission.sh start $GPU"}
    STAGE=starting
    write_status starting "Background S15-3A admission starting on GPU${GPU}."
    printf -v launch 'env OUTPUT_OVERRIDE=%q SESSION_OVERRIDE=%q EXACT_START_COMMAND_OVERRIDE=%q ADMISSION_ARMS_OVERRIDE=%q bash %q worker %q %q %q >> %q 2>&1' "$OUTPUT" "$SESSION" "$EXACT_START_COMMAND" "$ADMISSION_ARMS" "$ROOT/experiment/phase15/run_stage15_s3a_toys_item_disjoint_admission.sh" "$GPU" "$STARTED_AT" "$EXACT_START_COMMAND" "$LOG"
    tmux new-session -d -s "$SESSION" "$launch"
    echo "started $SESSION on GPU${GPU}"
    echo "status: $STATUS"
    ;;
  worker)
    worker "${2:?missing gpu}" "${3:?missing started_at}" "${4:?missing exact command}"
    ;;
  status)
    if [[ -f "$STATUS" ]]; then
      "$PYTHON" experiment/phase15/protocol/refresh_background_status.py \
        --status "$STATUS" --log "$LOG" --total 512 --summary "$SUMMARY"
    else
      echo '{"status":"not_started","status_code":"NOT_STARTED","exit_code":-1,"exit_code_pending":false,"test_read":false}'
    fi
    ;;
  *)
    echo "usage: $0 {start <gpu>|worker <gpu> <started_at> <exact_command>|status}" >&2
    exit 2
    ;;
esac
