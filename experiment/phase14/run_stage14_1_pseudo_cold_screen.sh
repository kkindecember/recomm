#!/usr/bin/env bash
set -u -o pipefail

# Stage 14-1 formal Toys pseudo-cold screen.
# Usage: bash experiment/phase14/run_stage14_1_pseudo_cold_screen.sh {start <gpu>|worker <gpu> <started_at>|status}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ACTION=${1:-status}
GPU=${2:-}
OUTPUT=${OUTPUT_OVERRIDE:-"$ROOT/artifacts/phase14/m2/pseudo_cold_screen_toys_formal"}
STATUS="$OUTPUT/status.json"
LOG="$OUTPUT/run.log"
TELEMETRY="$OUTPUT/gpu_telemetry.csv"
PYTHON=${PYTHON_BIN:-/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python}
MIN_FREE_MIB=${MIN_FREE_MIB:-24576}
EXPECTED_INCREMENTAL_MIB=23000
HARD_TIMEOUT_SECONDS=${HARD_TIMEOUT_SECONDS:-86400}
HOLDER_CONTROLLER="$ROOT/tools/gram_ablation_scan.sh"
HOLDER_STATE_ROOT=${HOLDER_STATE_ROOT:-"$ROOT/.runtime/gram_ablation_scan_gpu5"}
HOLDER_SESSION=${HOLDER_SESSION:-gram_ablation_scan_gpu5}
STARTED_AT=""
RUNNER_PID=0
WORKLOAD_PID=0
STAGE=not_started
FINAL_STATE=not_started
FINAL_REASON="Not started."
FINAL_RC=-1
HOLDER_INITIAL_PID=0
HOLDER_RESERVE_MIB=0
HOLDER_RELEASED=false
HOLDER_RESTORED=false
HOLDER_RESTORE_DETAIL=not_required

write_status() {
  local state=$1 reason=$2 rc=${3:--1} tmp="$STATUS.tmp.$$"
  FINAL_STATE=$state
  FINAL_REASON=$reason
  FINAL_RC=$rc
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE14_STAGE14_1_PSEUDO_COLD_SCREEN_TOYS","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%d,"workload_pid":%d,"workload_rc":%d,"physical_gpu":%s,"minimum_free_mib":%d,"expected_incremental_mib":%d,"hard_timeout_seconds":%d,"resource_mode":"release_owned_holder_run_then_restore_same_holder","automatic_retry":false,"split":"train_only_pseudo_cold","test_opened":false,"held_ground_truth_opened_for_training":false,"holder_session":"%s","holder_state_root":"%s","holder_initial_pid":%d,"holder_reserve_mib":%d,"holder_released":%s,"holder_restored":%s,"holder_restore_detail":"%s","log_path":"artifacts/phase14/m2/pseudo_cold_screen_toys_formal/run.log","summary_path":"artifacts/phase14/m2/pseudo_cold_screen_toys_formal/summary.json"}\n' \
    "$state" "$STAGE" "$reason" "$STARTED_AT" "$(date -Is)" "$RUNNER_PID" "$WORKLOAD_PID" "$rc" "${GPU:--1}" "$MIN_FREE_MIB" "$EXPECTED_INCREMENTAL_MIB" "$HARD_TIMEOUT_SECONDS" \
    "$HOLDER_SESSION" "$HOLDER_STATE_ROOT" "$HOLDER_INITIAL_PID" "$HOLDER_RESERVE_MIB" "$HOLDER_RELEASED" "$HOLDER_RESTORED" "$HOLDER_RESTORE_DETAIL" > "$tmp"
  mv "$tmp" "$STATUS"
}

read_and_validate_holder() {
  local holder_status="$HOLDER_STATE_ROOT/status.json" holder_gpu="$HOLDER_STATE_ROOT/gpu.txt" cmdline=""
  [[ -s "$holder_status" && -s "$holder_gpu" && -x "$HOLDER_CONTROLLER" ]] || return 1
  [[ "$(tr -d '[:space:]' < "$holder_gpu")" == "$GPU" ]] || return 1
  read -r HOLDER_INITIAL_PID HOLDER_RESERVE_MIB < <(
    "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d.get("state")=="running"; print(int(d["pid"]), int(d["reserve_mib"]))' "$holder_status"
  ) || return 1
  [[ "$HOLDER_INITIAL_PID" =~ ^[1-9][0-9]*$ && "$HOLDER_RESERVE_MIB" =~ ^[1-9][0-9]*$ ]] || return 1
  [[ -r "/proc/$HOLDER_INITIAL_PID/cmdline" ]] || return 1
  cmdline=$(tr '\0' ' ' < "/proc/$HOLDER_INITIAL_PID/cmdline")
  [[ "$cmdline" == *"tools/gram_ablation_scan_worker.py"* \
    && "$cmdline" == *"--state-dir $HOLDER_STATE_ROOT"* \
    && "$cmdline" == *"--reserve-mib $HOLDER_RESERVE_MIB"* ]] || return 1
  tmux has-session -t "$HOLDER_SESSION" 2>/dev/null || return 1
}

release_holder() {
  SESSION="$HOLDER_SESSION" STATE_ROOT="$HOLDER_STATE_ROOT" \
    bash "$HOLDER_CONTROLLER" stop || return 1
  tmux has-session -t "$HOLDER_SESSION" 2>/dev/null && return 1
  [[ ! -e "/proc/$HOLDER_INITIAL_PID" ]] || return 1
  HOLDER_RELEASED=true
  HOLDER_RESTORE_DETAIL=pending_after_experiment
}

restore_holder() {
  local restored_pid=0 restored_reserve=0 restored_used_mib=0 holder_status="$HOLDER_STATE_ROOT/status.json"
  RESERVE_MIB="$HOLDER_RESERVE_MIB" SESSION="$HOLDER_SESSION" STATE_ROOT="$HOLDER_STATE_ROOT" \
    bash "$HOLDER_CONTROLLER" start "$GPU" || return 1
  read -r restored_pid restored_reserve < <(
    "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d.get("state")=="running"; print(int(d["pid"]), int(d["reserve_mib"]))' "$holder_status"
  ) || return 1
  [[ "$restored_pid" =~ ^[1-9][0-9]*$ && "$restored_pid" != "$HOLDER_INITIAL_PID" \
    && "$restored_reserve" == "$HOLDER_RESERVE_MIB" && -r "/proc/$restored_pid/cmdline" ]] || return 1
  restored_used_mib=$(nvidia-smi -i "$GPU" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
    | "$PYTHON" -c 'import sys; pid=sys.argv[1]; rows=[x.strip() for x in sys.stdin if x.strip()]; vals=[int(x.split(",",1)[1].strip()) for x in rows if x.split(",",1)[0].strip()==pid]; print(vals[0] if vals else 0)' "$restored_pid") || return 1
  (( restored_used_mib >= 19000 )) || return 1
  HOLDER_RESTORED=true
  HOLDER_RESTORE_DETAIL="restored_pid_${restored_pid}_used_${restored_used_mib}_mib"
}

restore_holder_on_exit() {
  local shell_rc=$? terminal_state=$FINAL_STATE terminal_reason=$FINAL_REASON terminal_rc=$FINAL_RC
  trap - EXIT
  if [[ "$HOLDER_RELEASED" == true && "$HOLDER_RESTORED" != true ]]; then
    STAGE=holder_restore
    if restore_holder; then
      STAGE=finished
      write_status "$terminal_state" "${terminal_reason}; GPU5 holder restored at the original reserve size." "$terminal_rc"
    else
      HOLDER_RESTORE_DETAIL=restore_failed_manual_attention_required
      STAGE=holder_restore_failed
      write_status failed "Experiment terminal state was ${terminal_state}; GPU5 holder restoration failed and requires manual attention." 12
      (( shell_rc == 0 )) && shell_rc=12
    fi
  fi
  exit "$shell_rc"
}

telemetry() {
  printf 'timestamp,index,memory_used_mib,memory_free_mib,utilization_gpu_percent\n' > "$TELEMETRY"
  while true; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits --id="$GPU" >> "$TELEMETRY" 2>/dev/null || true
    sleep 60
  done
}

worker() {
  GPU=${1:?missing gpu}
  STARTED_AT=${2:?missing started_at}
  RUNNER_PID=$$
  local rc=0 free_mib telemetry_pid=0
  trap '' HUP
  trap restore_holder_on_exit EXIT
  trap '[[ $WORKLOAD_PID -eq 0 ]] || kill -TERM "$WORKLOAD_PID" 2>/dev/null || true; [[ $telemetry_pid -eq 0 ]] || kill "$telemetry_pid" 2>/dev/null || true; STAGE=stopped; write_status stopped "Stopped by signal; no automatic retry." 143; exit 143' TERM INT
  cd "$ROOT" || exit 2
  STAGE=preflight
  write_status running "Syntax, unit tests, frozen inputs, and GPU admission."
  bash -n experiment/phase14/run_stage14_1_pseudo_cold_screen.sh \
    || { STAGE=finished; write_status failed "Runner syntax failed." 4; exit 4; }
  "$PYTHON" -m py_compile \
    experiment/phase14/protocol/r2pd_pseudo_cold_screen.py \
    experiment/phase14/protocol/r2pd_targets.py \
    || { STAGE=finished; write_status failed "Python syntax failed." 5; exit 5; }
  "$PYTHON" -m pytest -q experiment/phase14/tests \
    || { STAGE=finished; write_status failed "Phase14 unit tests failed." 6; exit 6; }
  for path in \
    artifacts/phase14/m2/pretrained/t5-small/pytorch_model.bin \
    artifacts/phase14/m2/pseudo_cold_audit_toys_v2/student_readable/filtered_train_sequences.jsonl \
    artifacts/phase14/m2/pseudo_cold_audit_toys_v2/held_ground_truth_DO_NOT_USE_FOR_TRAINING/pseudo_cold_events.jsonl \
    artifacts/phase14/m2/item_disjoint_r2_teacher_toys/resolver_item_disjoint.pt \
    artifacts/phase14/m2/item_disjoint_r2_teacher_toys/summary.json \
    GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt \
    GRAM/rec_datasets/Toys_cold50/item_plain_text.txt \
    GRAM/rec_datasets/Toys_cold50/similar_item_sasrec.txt; do
    [[ -s "$path" ]] || { STAGE=finished; write_status failed "Missing frozen input: $path" 7; exit 7; }
  done
  [[ ! -e "$OUTPUT/summary.json" && ! -e "$OUTPUT/clean_base.pt" ]] \
    || { STAGE=finished; write_status failed "Refusing to overwrite formal scientific artifacts." 8; exit 8; }
  read_and_validate_holder \
    || { STAGE=finished; write_status blocked "Owned GPU5 holder validation failed; no process was changed." 13; exit 13; }
  STAGE=holder_release
  write_status running "Validated owned GPU5 holder; releasing it for the formal experiment."
  release_holder \
    || { STAGE=finished; write_status blocked "Could not cleanly release the validated GPU5 holder; no workload started." 14; exit 14; }
  write_status running "Owned GPU5 holder released; restoration is mandatory on every terminal path."
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GPU" 2>/dev/null | tr -d ' ' || true)
  [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )) \
    || { STAGE=finished; write_status blocked "GPU admission failed; requires ${MIN_FREE_MIB} MiB free." 9; exit 9; }

  telemetry & telemetry_pid=$!
  STAGE=clean_base_and_a0_a3_screen
  write_status running "Formal 512-transition/512-event A0-A3 screen active on GPU${GPU}."
  timeout --signal=TERM --kill-after=30 "$HARD_TIMEOUT_SECONDS" \
    env CUDA_VISIBLE_DEVICES="$GPU" TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    "$PYTHON" experiment/phase14/protocol/r2pd_pseudo_cold_screen.py \
      --historical-config artifacts/phase13/explore/v0_toys/gram_logs/Toys_cold50/0_20260808_2256/config.json \
      --backbone-path artifacts/phase14/m2/pretrained/t5-small \
      --train-sequences artifacts/phase14/m2/pseudo_cold_audit_toys_v2/student_readable/filtered_train_sequences.jsonl \
      --held-events artifacts/phase14/m2/pseudo_cold_audit_toys_v2/held_ground_truth_DO_NOT_USE_FOR_TRAINING/pseudo_cold_events.jsonl \
      --pseudo-cold-items artifacts/phase14/m2/pseudo_cold_audit_toys_v2/pseudo_cold_items.txt \
      --real-cold-items GRAM/rec_datasets/Toys_cold50/cold_split_meta/cold_items.txt \
      --item-path-file GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt \
      --item-text-file GRAM/rec_datasets/Toys_cold50/item_plain_text.txt \
      --similar-items-file GRAM/rec_datasets/Toys_cold50/similar_item_sasrec.txt \
      --item-embeddings artifacts/phase13/embeddings/Toys_bge_large_en_v1_5_cls_l2.pt \
      --teacher-checkpoint artifacts/phase14/m2/item_disjoint_r2_teacher_toys/resolver_item_disjoint.pt \
      --teacher-summary artifacts/phase14/m2/item_disjoint_r2_teacher_toys/summary.json \
      --output-dir "$OUTPUT" --device cuda:0 \
      --train-examples 512 --eval-events 512 --base-epochs 2 --adapt-epochs 1 \
      --batch-size 2 --synthetic-chunk-size 5 --top-m 50 \
      --learning-rate 0.001 --lambda-cp 1.0 --mu-keep 1.0 \
      --max-history 20 --recency-decay 0.85 --beam-size 50 \
      --bootstrap-resamples 10000 --seed 1401 &
  WORKLOAD_PID=$!
  write_status running "Formal screen workload active; inspect this status file, no automatic retry."
  wait "$WORKLOAD_PID" || rc=$?
  WORKLOAD_PID=0
  kill "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
  telemetry_pid=0
  if (( rc != 0 )); then
    STAGE=finished
    write_status failed "Formal screen exited rc=${rc}; no automatic retry." "$rc"
    exit "$rc"
  fi
  [[ -s "$OUTPUT/summary.json" ]] \
    || { STAGE=finished; write_status failed "Screen exited 0 without summary.json." 10; exit 10; }
  local verdict
  verdict=$(
    "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$OUTPUT/summary.json"
  ) || { STAGE=finished; write_status failed "Could not read formal verdict." 11; exit 11; }
  STAGE=finished
  write_status completed "$verdict" 0
}

case "$ACTION" in
  start)
    [[ "$GPU" =~ ^[0-7]$ ]] || { echo "GPU must be 0..7" >&2; exit 2; }
    mkdir -p "$OUTPUT"
    [[ ! -e "$STATUS" ]] || { echo "Formal status already exists: $STATUS" >&2; exit 3; }
    STARTED_AT=$(date -Is)
    STAGE=starting
    write_status starting "Background worker is starting on GPU${GPU}."
    setsid bash "$0" worker "$GPU" "$STARTED_AT" >> "$LOG" 2>&1 < /dev/null &
    RUNNER_PID=$!
    write_status starting "Background worker launched on GPU${GPU}."
    echo "started pid=$RUNNER_PID"
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
