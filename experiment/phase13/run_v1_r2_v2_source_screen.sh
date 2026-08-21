#!/usr/bin/env bash
set -u -o pipefail

# Phase-13 R²-v2 Stage S: Toys+Beauty five-fold OOF source Gate.
# T0 preflight artifacts must already exist. This runner never reads Sports/test.
# Usage: bash experiment/phase13/run_v1_r2_v2_source_screen.sh {start <cpu|gpu-index>|worker <target> <timestamp>|status|stop}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ACTION=${1:-status}; TARGET=${2:-cpu}
OUTPUT="$ROOT/artifacts/phase13/explore/v1_r2_v2_source_screen"
STATUS="$OUTPUT/status.json"; LOG="$OUTPUT/run.log"
SUMMARY="$OUTPUT/summary.json"
SESSION=gram_phase13_r2_v2_source_screen
PYTHON=${PYTHON_BIN:-/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python}
CONFIG=experiment/phase13/configs/r2_v2_cbsa_frozen.json
PROTOCOL=experiment/phase13/protocol/r2_v2_budgeted_slate_allocator.py
MIN_FREE_MIB=${MIN_FREE_MIB:-3072}; HARD_TIMEOUT_SECONDS=${HARD_TIMEOUT_SECONDS:-7200}
STARTED_AT=""; RUNNER_PID=0; WORKLOAD_PID=0; STAGE=not_started; PHYSICAL_GPU=-1

write_status() {
  local state=$1 reason=$2 tmp="$STATUS.tmp.$$"; mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE13_R2_V2_CBSA_SOURCE_OOF","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%d,"workload_pid":%d,"tmux_session":"%s","execution_target":"%s","physical_gpu":%d,"minimum_free_mib":%d,"expected_incremental_gpu_mib_upper_bound":2048,"hard_timeout_seconds":%d,"split":"toys_beauty_validation_5fold_oof","primary_budget":0.97,"sports_read":false,"test_read":false,"automatic_retry":false,"log_path":"%s","status_path":"%s","summary_path":"%s"}\n' \
    "$state" "$STAGE" "$reason" "$STARTED_AT" "$(date -Is)" "$RUNNER_PID" "$WORKLOAD_PID" "$SESSION" "$TARGET" "$PHYSICAL_GPU" "$MIN_FREE_MIB" "$HARD_TIMEOUT_SECONDS" \
    "${LOG#$ROOT/}" "${STATUS#$ROOT/}" "${SUMMARY#$ROOT/}" > "$tmp"
  mv "$tmp" "$STATUS"
}

telemetry() {
  printf 'timestamp,index,memory_used_mib,memory_free_mib,utilization_gpu_percent\n' > "$OUTPUT/gpu_telemetry.csv"
  while true; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits --id="$PHYSICAL_GPU" >> "$OUTPUT/gpu_telemetry.csv" 2>/dev/null || true
    sleep 10
  done
}

worker() {
  TARGET=${1:?missing target}; STARTED_AT=${2:?missing timestamp}; RUNNER_PID=$$
  local device=cpu rc=0 free_mib telemetry_pid=0
  trap '[[ $WORKLOAD_PID -eq 0 ]] || kill -TERM "$WORKLOAD_PID" 2>/dev/null || true; [[ $telemetry_pid -eq 0 ]] || kill "$telemetry_pid" 2>/dev/null || true; STAGE=stopped; write_status stopped "Stopped by signal; no retry."; exit 143' TERM INT HUP
  cd "$ROOT"
  STAGE=preflight; write_status running "Verifying frozen code/config/input hashes and source-only guards."
  bash -n experiment/phase13/run_v1_r2_v2_source_screen.sh || { write_status failed "Bash syntax failed."; return 4; }
  "$PYTHON" -m py_compile "$PROTOCOL" || { write_status failed "Python syntax failed."; return 5; }
  "$PYTHON" -m pytest -q experiment/phase13/tests/test_r2_v2_budgeted_slate_allocator.py || { write_status failed "R2-v2 contract tests failed."; return 6; }
  "$PYTHON" "$PROTOCOL" --mode verify-preflight --project-root "$ROOT" --canonical-config "$CONFIG" --output-dir "$OUTPUT" || { write_status failed "Frozen preflight verification failed."; return 7; }
  [[ ! -e "$SUMMARY" && ! -e "$OUTPUT/predictions_oof.jsonl" ]] || { write_status failed "Refusing to overwrite Stage-S scientific artifacts."; return 8; }

  if [[ "$TARGET" == cpu ]]; then
    PHYSICAL_GPU=-1; device=cpu
  elif [[ "$TARGET" =~ ^[0-7]$ ]]; then
    PHYSICAL_GPU=$TARGET
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$TARGET" 2>/dev/null | tr -d ' ' || true)
    [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )) || { write_status blocked "GPU admission failed; requires ${MIN_FREE_MIB} MiB free."; return 9; }
    device=cuda:0
    telemetry & telemetry_pid=$!
  else
    write_status failed "Execution target must be cpu or GPU index 0-7."; return 2
  fi

  STAGE=source_oof; write_status running "R2-v2 Stage-S source OOF active; Sports/test sealed."
  if [[ "$TARGET" == cpu ]]; then
    timeout --signal=TERM --kill-after=20 "$HARD_TIMEOUT_SECONDS" \
      "$PYTHON" "$PROTOCOL" --mode run-source --project-root "$ROOT" \
      --canonical-config "$CONFIG" --output-dir "$OUTPUT" --device "$device" &
  else
    timeout --signal=TERM --kill-after=20 "$HARD_TIMEOUT_SECONDS" env CUDA_VISIBLE_DEVICES="$TARGET" \
      "$PYTHON" "$PROTOCOL" --mode run-source --project-root "$ROOT" \
      --canonical-config "$CONFIG" --output-dir "$OUTPUT" --device "$device" &
  fi
  WORKLOAD_PID=$!; write_status running "R2-v2 Stage-S source OOF active; Sports/test sealed."
  wait "$WORKLOAD_PID" || rc=$?; WORKLOAD_PID=0
  [[ $telemetry_pid -eq 0 ]] || { kill "$telemetry_pid" 2>/dev/null || true; wait "$telemetry_pid" 2>/dev/null || true; }
  (( rc == 0 )) || { STAGE=finished; write_status failed "Stage-S exited rc=${rc}; no retry."; return "$rc"; }
  local verdict; verdict=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$SUMMARY") || return 10
  STAGE=finished; write_status completed "$verdict"
}

case "$ACTION" in
  start)
    [[ "$TARGET" == cpu || "$TARGET" =~ ^[0-7]$ ]] || { echo "usage: $0 start <cpu|gpu-index>" >&2; exit 2; }
    mkdir -p "$OUTPUT"; [[ ! -e "$STATUS" ]] || { echo "status exists: $STATUS" >&2; exit 3; }
    STARTED_AT=$(date -Is); STAGE=starting; [[ "$TARGET" == cpu ]] || PHYSICAL_GPU=$TARGET
    write_status starting "Stage-S background session starting on ${TARGET}."
    printf -v launch 'bash %q worker %q %q >> %q 2>&1' "$0" "$TARGET" "$STARTED_AT" "$LOG"
    tmux new-session -d -s "$SESSION" "$launch" || { STAGE=failed; write_status failed "tmux launch failed."; exit 5; }
    echo "started. status: $STATUS" ;;
  worker) worker "${2:?}" "${3:?}" ;;
  status) [[ -f "$STATUS" ]] && sed -n '1,100p' "$STATUS" || echo '{"status":"not_started"}' ;;
  stop) tmux send-keys -t "$SESSION" C-c 2>/dev/null || echo "no session" ;;
  *) echo "usage: $0 {start <cpu|gpu-index>|status|stop}" >&2; exit 2 ;;
esac
