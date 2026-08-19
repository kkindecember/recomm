#!/usr/bin/env bash
# Matched diagnostic GRAM smoke for BGE collision-safe Toys IDs.
#
# Frozen GPU0 resource lifecycle:
#   scan holder 40239 MiB -> 18000 MiB -> run -> 30000 MiB
# Only the verified gram_ablation_scan_gpu0 holder may be signalled. Existing
# non-holder GPU processes are recorded and never signalled. No workload retry
# and no automatic formal experiment.
#
# Usage:
#   bash experiment/phase13/run_v1_bge_toys_downstream_smoke.sh start 0
#   bash experiment/phase13/run_v1_bge_toys_downstream_smoke.sh status
#   bash experiment/phase13/run_v1_bge_toys_downstream_smoke.sh stop
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
ACTION=${1:-status}
GPU=${2:-}

SUB=smoke_v1_collision_safe_bge_toys
SESSION=gram_phase13_${SUB}
OUTPUT="$ROOT/artifacts/phase13/explore/$SUB"
LOG="$OUTPUT/run.log"
STATUS="$OUTPUT/status.json"
TELEMETRY="$OUTPUT/gpu_telemetry.csv"
METRICS="$OUTPUT/metrics_cold_warm.json"
SUMMARY="$OUTPUT/smoke_summary.json"
PREDICTIONS="$OUTPUT/predictions"
ID_FILE="$ROOT/GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split_v1_bge_mlpcold_collision_safe.txt"
ID_REPORT="$ROOT/artifacts/phase13/explore/v1_collision_safe_bge_toys_screen/id_report.json"
BASELINE_ID_FILE="$ROOT/GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split_v1_mlpcold_collision_safe.txt"
BASELINE_METRICS="$ROOT/artifacts/phase13/explore/smoke_v1_collision_safe_toys/metrics_cold_warm.json"

EXPECTED_GPU=0
MIN_FREE_MIB=22000
EXPECTED_INCREMENTAL_MIB=20480
TIMEOUT_SECONDS=7200
SCAN_SESSION=gram_ablation_scan_gpu0
SCAN_STATE_ROOT="$ROOT/.runtime/gram_ablation_scan_gpu0"
SCAN_TOOL="$ROOT/tools/gram_ablation_scan.sh"
WATCHDOG="$ROOT/experiment/phase13/partial_holder_restore_watchdog.sh"
WATCHDOG_SESSION="gram_phase13_partial_holder_watchdog_${SUB}"
INITIAL_HOLDER_MIB=40239
DURING_HOLDER_MIB=18000
POST_HOLDER_MIB=30000
POST_START_FREE_MIB=$(( POST_HOLDER_MIB + 2500 ))
DURING_START_FREE_MIB=$(( DURING_HOLDER_MIB + 2500 ))
ALLOWED_PIDS_FILE="$OUTPUT/preexisting_gpu_pids.txt"
TRANSITION_MARKER="$OUTPUT/.resource_transition_started"
WATCHDOG_STATUS="$OUTPUT/resource_watchdog_status.json"

STARTED_AT=""
RUNNER_PID=0
WORKLOAD_PID=0
TELEMETRY_PID=0
CURRENT_STAGE=not_started
CURRENT_REASON="Not started."
PREEXISTING_GPU_PIDS=""
RESOURCE_TRANSITION_STARTED=false
RESOURCE_RESTORE_STATE=original_not_verified
FINALIZED=0

write_status() {
  local state=$1 reason=$2 tmp="${STATUS}.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE13_BGE_DOWNSTREAM_DIAGNOSTIC_SMOKE","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%d,"workload_pid":%d,"tmux_session":"%s","physical_gpu":%s,"minimum_free_mib":%d,"expected_incremental_gpu_mib_upper_bound":%d,"hard_timeout_seconds":%d,"resource_mode":"partial_scan_holder_transition","initial_holder_reserve_mib":%d,"during_holder_reserve_mib":%d,"post_holder_reserve_mib":%d,"resource_transition_started":%s,"resource_restore_state":"%s","preexisting_non_holder_gpu_pids":"%s","scan_holder_session":"%s","resource_watchdog_session":"%s","resource_watchdog_status_path":"%s","signals_to_non_holder_processes":0,"formal_experiment":false,"efficacy_gate_consumed":false,"automatic_formal_run":false,"automatic_retry":false,"log_path":"%s","status_path":"%s","metrics_path":"%s","summary_path":"%s"}\n' \
    "$state" "$CURRENT_STAGE" "$reason" "$STARTED_AT" "$(date -Is)" \
    "$RUNNER_PID" "$WORKLOAD_PID" "$SESSION" "${GPU:--1}" "$MIN_FREE_MIB" \
    "$EXPECTED_INCREMENTAL_MIB" "$TIMEOUT_SECONDS" "$INITIAL_HOLDER_MIB" \
    "$DURING_HOLDER_MIB" "$POST_HOLDER_MIB" "$RESOURCE_TRANSITION_STARTED" \
    "$RESOURCE_RESTORE_STATE" "$PREEXISTING_GPU_PIDS" "$SCAN_SESSION" \
    "$WATCHDOG_SESSION" "${WATCHDOG_STATUS#$ROOT/}" \
    "${LOG#$ROOT/}" "${STATUS#$ROOT/}" "${METRICS#$ROOT/}" \
    "${SUMMARY#$ROOT/}" > "$tmp"
  mv "$tmp" "$STATUS"
}

holder_field() {
  local field=$1
  "$PYTHON" -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get(sys.argv[2], 0))" \
    "$SCAN_STATE_ROOT/status.json" "$field" 2>/dev/null || echo 0
}

holder_pid() {
  local state
  state=$(holder_field state)
  [[ "$state" == running ]] || { echo 0; return; }
  holder_field pid
}

pid_on_gpu() {
  local pid=$1
  [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 1
  nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits \
    2>/dev/null | tr -d ' ' | grep -Fxq "$pid"
}

holder_on_gpu() {
  local pid
  pid=$(holder_pid)
  pid_on_gpu "$pid"
}

gpu_pids() {
  nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits \
    2>/dev/null | tr -d ' ' | sed '/^[0-9][0-9]*$/!d' | sort -nu
}

capture_preexisting_non_holder_pids() {
  local holder pid
  holder=$(holder_pid)
  : > "$ALLOWED_PIDS_FILE"
  while read -r pid; do
    [[ -n "$pid" && "$pid" != "$holder" ]] || continue
    printf '%s\n' "$pid" >> "$ALLOWED_PIDS_FILE"
  done < <(gpu_pids)
  PREEXISTING_GPU_PIDS=$(paste -sd, "$ALLOWED_PIDS_FILE" 2>/dev/null || true)
}

has_unknown_gpu_pid() {
  local holder=0 pid
  holder=$(holder_pid)
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    [[ "$pid" == "$holder" ]] && continue
    grep -Fxq "$pid" "$ALLOWED_PIDS_FILE" 2>/dev/null || return 0
  done < <(gpu_pids)
  return 1
}

gpu_free_mib() {
  nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits \
    2>/dev/null | tr -d ' '
}

wait_for_holder() {
  local reserve=$1
  for _ in $(seq 1 20); do
    if holder_on_gpu && [[ "$(holder_field reserve_mib)" == "$reserve" ]]; then
      return 0
    fi
    sleep 1
  done
  return 1
}

stop_verified_holder() {
  local expected=$1 pid
  holder_on_gpu || return 1
  [[ "$(holder_field reserve_mib)" == "$expected" ]] || return 1
  tmux has-session -t "$SCAN_SESSION" 2>/dev/null || return 1
  pid=$(holder_pid)
  env SESSION="$SCAN_SESSION" STATE_ROOT="$SCAN_STATE_ROOT" \
    "$SCAN_TOOL" stop
  for _ in $(seq 1 15); do
    pid_on_gpu "$pid" || return 0
    sleep 1
  done
  return 1
}

start_holder() {
  local reserve=$1
  env RESERVE_MIB="$reserve" SESSION="$SCAN_SESSION" STATE_ROOT="$SCAN_STATE_ROOT" \
    "$SCAN_TOOL" start "$GPU"
  wait_for_holder "$reserve"
}

telemetry() {
  printf 'timestamp,index,memory_used_mib,memory_free_mib,utilization_gpu_percent\n' > "$TELEMETRY"
  while true; do
    nvidia-smi -i "$GPU" --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits >> "$TELEMETRY" 2>/dev/null || true
    sleep 5
  done
}

cleanup_runtime() {
  if (( TELEMETRY_PID > 0 )); then
    kill "$TELEMETRY_PID" >/dev/null 2>&1 || true
    wait "$TELEMETRY_PID" 2>/dev/null || true
    TELEMETRY_PID=0
  fi
}

restore_post_holder() {
  [[ "$RESOURCE_TRANSITION_STARTED" == true ]] || return 0
  RESOURCE_RESTORE_STATE=restoring_post_holder

  if holder_on_gpu && [[ "$(holder_field reserve_mib)" == "$POST_HOLDER_MIB" ]]; then
    RESOURCE_RESTORE_STATE=protected_exact_30000
    return 0
  fi
  if has_unknown_gpu_pid; then
    RESOURCE_RESTORE_STATE=pending_watchdog_unknown_process
    return 1
  fi

  if holder_on_gpu; then
    local active
    active=$(holder_field reserve_mib)
    if [[ "$active" != "$DURING_HOLDER_MIB" && "$active" != "$INITIAL_HOLDER_MIB" ]]; then
      RESOURCE_RESTORE_STATE=pending_watchdog_unexpected_holder
      return 1
    fi
    stop_verified_holder "$active" || {
      RESOURCE_RESTORE_STATE=pending_watchdog_holder_stop_failed
      return 1
    }
  fi

  local free
  free=$(gpu_free_mib || true)
  if [[ "$free" =~ ^[0-9]+$ ]] && (( free >= POST_START_FREE_MIB )); then
    if start_holder "$POST_HOLDER_MIB"; then
      RESOURCE_RESTORE_STATE=protected_exact_30000
      return 0
    fi
    env SESSION="$SCAN_SESSION" STATE_ROOT="$SCAN_STATE_ROOT" \
      "$SCAN_TOOL" stop >/dev/null 2>&1 || true
  fi

  free=$(gpu_free_mib || true)
  if [[ "$free" =~ ^[0-9]+$ ]] && (( free >= DURING_START_FREE_MIB )); then
    start_holder "$DURING_HOLDER_MIB" >/dev/null 2>&1 || true
  fi
  RESOURCE_RESTORE_STATE=pending_watchdog_interim_protection
  return 1
}

finish_run() {
  local state=$1 reason=$2 rc=$3 restore_rc=0
  cleanup_runtime
  restore_post_holder || restore_rc=$?
  if (( restore_rc != 0 )); then
    reason="${reason} Post-run 30000 MiB holder restore is pending in the independent watchdog; no non-holder process was signalled."
  fi
  FINALIZED=1
  write_status "$state" "$reason"
  return "$rc"
}

on_term() {
  if (( WORKLOAD_PID > 0 )) && kill -0 "$WORKLOAD_PID" 2>/dev/null; then
    kill -TERM "$WORKLOAD_PID" 2>/dev/null || true
    wait "$WORKLOAD_PID" 2>/dev/null || true
    WORKLOAD_PID=0
  fi
  CURRENT_STAGE=stopped
  finish_run stopped "Stopped by user; no automatic retry." 143
  exit 143
}

on_exit() {
  local rc=$?
  (( FINALIZED == 0 )) || return
  cleanup_runtime
  restore_post_holder || true
  CURRENT_STAGE=aborted
  write_status failed "Runner exited unexpectedly (rc=${rc}); no automatic retry. Resource restore state is ${RESOURCE_RESTORE_STATE}."
}

worker() {
  GPU=${1:?missing gpu}
  STARTED_AT=${2:?missing start timestamp}
  RUNNER_PID=$$
  trap on_term TERM INT HUP
  trap on_exit EXIT
  cd "$ROOT"

  CURRENT_STAGE=preflight
  CURRENT_REASON="Checking matched protocol, frozen inputs, exact GPU0 holder, and isolated outputs."
  write_status running "$CURRENT_REASON"
  "$PYTHON" -m py_compile \
    GRAM/src/main_generative_gram.py \
    experiment/phase13/protocol/eval_cold_warm.py \
    experiment/phase13/protocol/summarize_bge_downstream_smoke.py \
    || { finish_run failed "Python syntax check failed; no automatic retry." 4; return $?; }
  "$PYTHON" -m unittest \
    experiment/phase13/tests/test_semantic_bridge.py \
    experiment/phase13/tests/test_collision_safe_ids.py \
    || { finish_run failed "Phase-13 unit tests failed; no automatic retry." 5; return $?; }
  for path in "$ID_FILE" "$ID_REPORT" "$BASELINE_ID_FILE" "$BASELINE_METRICS"; do
    [[ -s "$path" ]] \
      || { finish_run failed "Required frozen input missing: ${path#$ROOT/}" 6; return $?; }
  done
  for path in "$METRICS" "$SUMMARY"; do
    [[ ! -e "$path" ]] \
      || { finish_run failed "Refusing to overwrite existing artifact: ${path#$ROOT/}" 7; return $?; }
  done
  "$PYTHON" -c "import json; r=json.load(open('$ID_REPORT')); assert r['output_collision']['duplicate_excess']==0; assert r['warm_ids_unchanged']; assert r['row_order_unchanged']; assert r['cold_prefixes_unchanged']" \
    || { finish_run failed "BGE collision-safe invariant audit failed; no automatic retry." 8; return $?; }
  [[ "$GPU" == "$EXPECTED_GPU" ]] \
    || { finish_run blocked "This frozen runner only permits physical GPU${EXPECTED_GPU}; no resource changes made." 9; return $?; }
  tmux has-session -t "$SCAN_SESSION" 2>/dev/null \
    || { finish_run blocked "Expected scan holder session ${SCAN_SESSION} is absent; no resource changes made." 10; return $?; }
  holder_on_gpu \
    || { finish_run blocked "Could not verify the scan holder PID on GPU0; no resource changes made." 11; return $?; }
  [[ "$(holder_field reserve_mib)" == "$INITIAL_HOLDER_MIB" ]] \
    || { finish_run blocked "GPU0 holder reserve changed from frozen ${INITIAL_HOLDER_MIB} MiB; no resource changes made." 12; return $?; }
  RESOURCE_RESTORE_STATE=original_exact_40239
  capture_preexisting_non_holder_pids
  write_status running "Preflight passed; recorded existing non-holder GPU PIDs and starting independent recovery watchdog."

  bash "$WATCHDOG" start "$SUB" "$GPU" "$INITIAL_HOLDER_MIB" \
    "$DURING_HOLDER_MIB" "$POST_HOLDER_MIB" "$SESSION" "$SCAN_SESSION" \
    "$SCAN_STATE_ROOT" "$ALLOWED_PIDS_FILE" "$TRANSITION_MARKER" "$OUTPUT" \
    || { finish_run blocked "Independent resource watchdog failed to start; original holder left unchanged." 13; return $?; }

  : > "$TRANSITION_MARKER"
  RESOURCE_TRANSITION_STARTED=true
  CURRENT_STAGE=resource_transition
  write_status running "Resizing only the verified GPU0 scan holder from 40239 to 18000 MiB."
  stop_verified_holder "$INITIAL_HOLDER_MIB" \
    || { finish_run failed "Verified initial holder did not stop cleanly; workload was not started." 14; return $?; }
  start_holder "$DURING_HOLDER_MIB" \
    || { finish_run failed "Could not start exact 18000 MiB interim holder; workload was not started." 15; return $?; }
  RESOURCE_RESTORE_STATE=interim_exact_18000

  local free_mib
  free_mib=$(gpu_free_mib || true)
  [[ "$free_mib" =~ ^[0-9]+$ ]] \
    || { finish_run failed "Could not read GPU0 free memory after holder resize; workload was not started." 16; return $?; }
  (( free_mib >= MIN_FREE_MIB )) \
    || { finish_run blocked "GPU0 has ${free_mib} MiB free after exact resize; requires ${MIN_FREE_MIB}; workload was not started." 17; return $?; }

  mkdir -p "$PREDICTIONS"
  telemetry & TELEMETRY_PID=$!
  CURRENT_STAGE=gram_smoke
  CURRENT_REASON="Running matched 1-epoch/100-train/100-test BGE collision-safe GRAM diagnostic smoke."
  write_status running "$CURRENT_REASON"
  cd "$ROOT/GRAM/command"
  timeout --signal=TERM --kill-after=60 "$TIMEOUT_SECONDS" env \
    CUDA_VISIBLE_DEVICES="$GPU" \
    HF_HUB_CACHE="$ROOT/.cache/huggingface/hub" \
    TRANSFORMERS_CACHE="$ROOT/.cache/huggingface/transformers" \
    TRANSFORMERS_OFFLINE=1 \
    "$PYTHON" ../src/main_generative_gram.py \
      --datasets Toys_cold50 \
      --distributed 0 --gpu 0 --seed 2023 --train 1 --resource_metrics 1 \
      --log_dir "$OUTPUT/gram_logs" --prediction_dir "$PREDICTIONS" \
      --item_prompt_max_len 128 --item_prompt all_text \
      --cf_model sasrec --id_linking 1 --max_his 20 \
      --rec_batch_size 16 --gradient_accumulation_steps 8 \
      --rec_lr 1e-3 --rec_epochs 1 \
      --test_epoch_rec 0 --save_rec_epochs 1 \
      --save_predictions 1 --beam_size 50 \
      --top_k_similar_item 5 --item_id_type split \
      --hierarchical_id_type hierarchy_v1_c32_l5_len32768_split_v1_bge_mlpcold_collision_safe \
      --debug_train_100 1 --debug_test_100 1 \
      --cf0_arm A --cf0_phase9 0 &
  WORKLOAD_PID=$!
  write_status running "$CURRENT_REASON"
  wait "$WORKLOAD_PID"
  local workload_rc=$?
  WORKLOAD_PID=0
  if (( workload_rc != 0 )); then
    CURRENT_STAGE=gram_smoke
    if (( workload_rc == 124 || workload_rc == 143 )); then
      finish_run timed_out "GRAM diagnostic smoke timed out/terminated rc=${workload_rc}; no automatic retry." "$workload_rc"
    else
      finish_run failed "GRAM diagnostic smoke exited rc=${workload_rc}; no automatic retry." "$workload_rc"
    fi
    return $?
  fi

  cleanup_runtime
  CURRENT_STAGE=postflight
  CURRENT_REASON="Computing cold/warm metrics and matched directional comparison."
  write_status running "$CURRENT_REASON"
  local pred_tsv
  pred_tsv=$(find "$PREDICTIONS" -maxdepth 1 -type f -name '*_pred_test.tsv' -printf '%T@ %p\n' \
    | sort -nr | head -n 1 | cut -d' ' -f2-)
  [[ -n "$pred_tsv" && -s "$pred_tsv" ]] \
    || { finish_run failed "No non-empty prediction TSV; no automatic retry." 18; return $?; }
  "$PYTHON" "$ROOT/experiment/phase13/protocol/eval_cold_warm.py" \
    --dataset-dir "$ROOT/GRAM/rec_datasets/Toys_cold50" \
    --predictions-tsv "$pred_tsv" --output-json "$METRICS" \
    --version-tag "$SUB" --split-name test \
    || { finish_run failed "eval_cold_warm failed; no automatic retry." 19; return $?; }
  "$PYTHON" "$ROOT/experiment/phase13/protocol/summarize_bge_downstream_smoke.py" \
    --candidate-metrics "$METRICS" --baseline-metrics "$BASELINE_METRICS" \
    --dataset-dir "$ROOT/GRAM/rec_datasets/Toys_cold50" \
    --candidate-safe-id "$ID_FILE" --baseline-safe-id "$BASELINE_ID_FILE" \
    --output "$SUMMARY" \
    || { finish_run failed "Matched smoke summary failed; no automatic retry." 20; return $?; }

  local result
  result=$("$PYTHON" -c "import json,sys; r=json.load(open(sys.argv[1])); print(r['verdict'] + ': diagnostic only; no automatic formal run.')" "$SUMMARY") \
    || { finish_run failed "Could not read smoke verdict." 21; return $?; }
  CURRENT_STAGE=finished
  finish_run completed "$result" 0
}

case "$ACTION" in
  start)
    [[ "$GPU" == "$EXPECTED_GPU" ]] || {
      echo "usage: $0 start 0 (this frozen runner only permits physical GPU0)" >&2
      exit 2
    }
    tmux has-session -t "$SESSION" 2>/dev/null \
      && { echo "session already exists: $SESSION" >&2; exit 1; }
    tmux has-session -t "$WATCHDOG_SESSION" 2>/dev/null \
      && { echo "resource watchdog already exists: $WATCHDOG_SESSION" >&2; exit 1; }
    if [[ -e "$STATUS" ]]; then
      reusable_launch_only=$("$PYTHON" -c "import json,sys; d=json.load(open(sys.argv[1])); print('yes' if d.get('status') in {'starting','launch_failed'} and not d.get('resource_transition_started') and int(d.get('runner_pid',0)) == 0 else 'no')" "$STATUS" 2>/dev/null || echo no)
      if [[ "$reusable_launch_only" != yes || -e "$TRANSITION_MARKER" \
        || -e "$METRICS" || -e "$SUMMARY" ]]; then
        echo "refusing to reuse an output that progressed beyond a launch-only failure: $STATUS" >&2
        exit 3
      fi
      echo "reusing launch-only status after a verified pre-worker tmux failure"
    fi
    STARTED_AT=$(date -Is)
    mkdir -p "$OUTPUT"
    printf -v launch_cmd 'bash %q worker %q %q >> %q 2>&1' \
      "$0" "$GPU" "$STARTED_AT" "$LOG"
    RESOURCE_RESTORE_STATE=original_pending_worker_verification
    write_status starting "Background BGE downstream diagnostic smoke launched; worker must verify exact 40239 MiB holder before any resource change."
    if ! tmux new-session -d -s "$SESSION" "$launch_cmd"; then
      CURRENT_STAGE=launch_failed
      RESOURCE_RESTORE_STATE=original_unchanged_after_tmux_failure
      write_status launch_failed "tmux session creation failed before the worker started; original holder was not changed."
      echo "failed to create tmux session; original holder was not changed" >&2
      exit 4
    fi
    echo "started $SESSION on GPU${GPU}"
    echo "status: bash $0 status"
    ;;
  worker)
    worker "${2:?missing gpu}" "${3:?missing start timestamp}"
    ;;
  status)
    tmux has-session -t "$SESSION" 2>/dev/null \
      && echo "tmux session: running ($SESSION)" \
      || echo "tmux session: not running ($SESSION)"
    [[ -f "$STATUS" ]] && sed -n '1,120p' "$STATUS" || echo '{"status":"not_started"}'
    bash "$WATCHDOG" status "$SUB" "$OUTPUT" 2>/dev/null || true
    [[ -f "$SUMMARY" ]] && echo "--- smoke summary ---" && sed -n '1,240p' "$SUMMARY"
    [[ -f "$LOG" ]] && echo "--- last 30 log lines ---" && tail -n 30 "$LOG" || true
    ;;
  stop)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      pane_pid=$(tmux list-panes -t "$SESSION" -F '#{pane_pid}' | head -n 1)
      kill -TERM "$pane_pid"
      echo "TERM sent to $SESSION; runner/watchdog will restore the 30000 MiB holder"
    else
      echo "session not running: $SESSION"
    fi
    ;;
  *)
    echo "usage: $0 {start 0|status|stop}" >&2
    exit 2
    ;;
esac
