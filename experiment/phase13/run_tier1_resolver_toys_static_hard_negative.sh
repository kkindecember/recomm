#!/usr/bin/env bash
set -u -o pipefail

# Phase-13 Tier-1 Toys static warm-only hard-negative screen.
# Usage: bash experiment/phase13/run_tier1_resolver_toys_static_hard_negative.sh {start <gpu>|status|stop}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ACTION=${1:-status}
GPU=${2:-}
OUTPUT="$ROOT/artifacts/phase13/explore/tier1_resolver_toys_static_warm_hard_negative"
STATUS="$OUTPUT/status.json"
LOG="$OUTPUT/run.log"
TELEMETRY="$OUTPUT/gpu_telemetry.csv"
SUMMARY="$OUTPUT/summary.json"
CONFIG="$ROOT/experiment/phase13/configs/tier1_resolver_toys_static_hard_negative.json"
PYTHON=${PYTHON_BIN:-/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python}
SESSION=gram_phase13_tier1_resolver_toys_static_hard_negative
TMUX_SOCKET=/tmp/gram_phase13_tier1_resolver_toys_static_hard_negative.sock
MIN_FREE_MIB=${MIN_FREE_MIB:-6144}
EXPECTED_INCREMENTAL_MIB=3072
HARD_TIMEOUT_SECONDS=${HARD_TIMEOUT_SECONDS:-3600}

STARTED_AT=""
RUNNER_PID=0
WORKLOAD_PID=0
STAGE=not_started

write_status() {
  local state=$1 reason=$2 tmp="$STATUS.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE13_TIER1_RESOLVER_TOYS_STATIC_WARM_HARD_NEGATIVE","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%d,"workload_pid":%d,"tmux_session":"%s","tmux_socket":"%s","physical_gpu":%s,"minimum_free_mib":%d,"expected_incremental_gpu_mib_upper_bound":%d,"hard_timeout_seconds":%d,"resource_mode":"single_gpu_background_no_resource_changes","split":"toys_validation_development_only","hard_negative_counts":[0,8,16,32],"negative_pool":"warm_only","test_read":false,"beauty_read":false,"sports_read":false,"automatic_next_stage":false,"automatic_retry":false,"log_path":"%s","status_path":"%s","summary_path":"%s"}\n' \
    "$state" "$STAGE" "$reason" "$STARTED_AT" "$(date -Is)" "$RUNNER_PID" "$WORKLOAD_PID" \
    "$SESSION" "$TMUX_SOCKET" "${GPU:--1}" "$MIN_FREE_MIB" "$EXPECTED_INCREMENTAL_MIB" "$HARD_TIMEOUT_SECONDS" \
    "${LOG#$ROOT/}" "${STATUS#$ROOT/}" "${SUMMARY#$ROOT/}" > "$tmp"
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
  STARTED_AT=${2:?missing start timestamp}
  RUNNER_PID=$$
  local rc=0 free_mib compute_mode telemetry_pid=0 verdict
  trap '[[ $WORKLOAD_PID -eq 0 ]] || kill -TERM "$WORKLOAD_PID" 2>/dev/null || true; [[ $telemetry_pid -eq 0 ]] || kill "$telemetry_pid" 2>/dev/null || true; STAGE=stopped; write_status stopped "Stopped by signal; no automatic retry."; exit 143' TERM INT HUP

  if [[ "$GPU" == 0 || "$GPU" == 5 ]]; then
    STAGE=blocked
    write_status blocked "GPU0 and GPU5 are user-protected and forbidden."
    return 3
  fi

  STAGE=preflight
  write_status running "Checking frozen protocol, unit tests, inputs, and GPU admission."
  bash -n "$0" || { write_status failed "Bash syntax check failed; no automatic retry."; return 4; }
  "$PYTHON" -m py_compile \
    "$ROOT/experiment/phase13/protocol/tier1_resolver_static_hard_negative.py" \
    "$ROOT/experiment/phase13/tests/test_tier1_resolver_static_hard_negative.py" \
    || { write_status failed "Python syntax check failed; no automatic retry."; return 5; }
  cd "$ROOT"
  "$PYTHON" -m pytest -q \
    experiment/phase13/tests/test_route_resolve.py \
    experiment/phase13/tests/test_tier1_resolver_checkpoint_trajectory.py \
    experiment/phase13/tests/test_tier1_resolver_static_hard_negative.py \
    || { write_status failed "Resolver hard-negative unit tests failed; no automatic retry."; return 6; }
  for path in \
    "$CONFIG" \
    "$ROOT/GRAM/rec_datasets/Toys_cold50/user_sequence.txt" \
    "$ROOT/GRAM/rec_datasets/Toys_cold50/cold_split_meta/cold_items.txt" \
    "$ROOT/GRAM/rec_datasets/Toys_cold50/cold_split_meta/warm_items.txt" \
    "$ROOT/GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt" \
    "$ROOT/artifacts/phase13/embeddings/Toys_bge_large_en_v1_5_cls_l2.pt" \
    "$ROOT/artifacts/phase13/explore/v0_toys/predictions/20260809_085251_Toys_cold50_sequential_pred_validation.tsv" \
    "$ROOT/artifacts/phase13/explore/v1_r2_toys_p0/predictions_validation.jsonl"; do
    [[ -s "$path" ]] || { write_status failed "Required input missing: ${path#$ROOT/}"; return 7; }
  done
  [[ ! -e "$SUMMARY" && ! -e "$OUTPUT/config.json" ]] \
    || { write_status failed "Refusing to overwrite an existing hard-negative artifact."; return 8; }

  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GPU" 2>/dev/null | tr -d ' ' || true)
  [[ "$free_mib" =~ ^[0-9]+$ ]] \
    || { write_status blocked "Could not read GPU${GPU} free memory; no resource changes made."; return 9; }
  (( free_mib >= MIN_FREE_MIB )) \
    || { write_status blocked "GPU${GPU} has ${free_mib} MiB free; requires ${MIN_FREE_MIB}; no resource changes made."; return 10; }
  compute_mode=$(nvidia-smi --query-gpu=compute_mode --format=csv,noheader,nounits --id="$GPU" 2>/dev/null | tr -d '[:space:]' || true)
  [[ "$compute_mode" == Default ]] \
    || { write_status blocked "GPU${GPU} compute_mode=${compute_mode:-unknown}; requires Default."; return 11; }
  STAGE=cuda_admission
  write_status running "Verifying live CUDA allocation on GPU${GPU}."
  env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -c \
    'import torch; assert torch.cuda.is_available(); x=torch.empty(4*1024*1024,device="cuda",dtype=torch.float32); x.fill_(1); torch.cuda.synchronize(); print(f"cuda_admission_ok device={torch.cuda.get_device_name(0)} allocated={torch.cuda.memory_allocated()}")' \
    || { write_status blocked "GPU${GPU} failed live CUDA allocation admission; no scientific workload started."; return 12; }

  telemetry & telemetry_pid=$!
  STAGE=training
  write_status running "Zero control plus HN=8/16/32 warm-only static hard-negative screen active."
  timeout --signal=TERM --kill-after=30 "$HARD_TIMEOUT_SECONDS" \
    env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
      "$ROOT/experiment/phase13/protocol/tier1_resolver_static_hard_negative.py" \
      --frozen-config "$CONFIG" \
      --output-dir "$OUTPUT" \
      --status-path "$STATUS" \
      --device cuda:0 &
  WORKLOAD_PID=$!
  write_status running "Hard-negative screen active on GPU${GPU}; existing processes were not modified."
  wait "$WORKLOAD_PID" || rc=$?
  WORKLOAD_PID=0
  kill "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
  telemetry_pid=0
  if (( rc != 0 )); then
    STAGE=finished
    write_status failed "Hard-negative screen exited rc=${rc}; no automatic retry."
    return "$rc"
  fi
  [[ -s "$SUMMARY" ]] \
    || { STAGE=finished; write_status failed "Screen exited 0 without summary.json."; return 13; }
  if rg -n 'Traceback|CUDA out of memory|\bNaN\b' "$LOG" >/dev/null 2>&1; then
    STAGE=finished
    write_status failed "Postflight found Traceback/OOM/NaN markers; no automatic retry."
    return 14
  fi
  verdict=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$SUMMARY") \
    || { STAGE=finished; write_status failed "Could not read hard-negative verdict."; return 15; }
  STAGE=finished
  write_status completed "$verdict"
}

case "$ACTION" in
  start)
    [[ "$GPU" =~ ^[0-7]$ ]] || { echo "usage: $0 start <gpu 0-7>" >&2; exit 2; }
    [[ "$GPU" != 0 && "$GPU" != 5 ]] || { echo "GPU0 and GPU5 are user-protected" >&2; exit 3; }
    mkdir -p "$OUTPUT"
    [[ ! -e "$STATUS" ]] || { echo "refusing to reuse existing status: $STATUS" >&2; exit 3; }
    tmux -S "$TMUX_SOCKET" has-session -t "$SESSION" 2>/dev/null \
      && { echo "session already exists: $SESSION" >&2; exit 3; }
    STARTED_AT=$(date -Is)
    STAGE=starting
    write_status starting "Background hard-negative screen is starting on GPU${GPU}."
    printf -v launch 'bash %q worker %q %q >> %q 2>&1' "$0" "$GPU" "$STARTED_AT" "$LOG"
    if ! tmux -S "$TMUX_SOCKET" new-session -d -s "$SESSION" "$launch"; then
      STAGE=launch_failed
      write_status failed "Could not create background tmux session; workload was not started."
      exit 4
    fi
    echo "started $SESSION on GPU${GPU}"
    echo "status: $STATUS"
    ;;
  worker)
    worker "${2:?missing gpu}" "${3:?missing start timestamp}"
    ;;
  status)
    [[ -f "$STATUS" ]] && sed -n '1,180p' "$STATUS" || echo '{"status":"not_started"}'
    ;;
  stop)
    if tmux -S "$TMUX_SOCKET" has-session -t "$SESSION" 2>/dev/null; then
      tmux -S "$TMUX_SOCKET" send-keys -t "$SESSION" C-c
      echo "stop requested; status file will record the outcome"
    else
      echo "session not running: $SESSION"
    fi
    ;;
  *)
    echo "usage: $0 {start <gpu>|status|stop}" >&2
    exit 2
    ;;
esac
