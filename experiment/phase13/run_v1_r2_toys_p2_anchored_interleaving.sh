#!/usr/bin/env bash
set -u -o pipefail

# Phase-13 v1-R² Toys validation-only P2 (CPU background runner).
# Usage: bash experiment/phase13/run_v1_r2_toys_p2_anchored_interleaving.sh {start|status|stop}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ACTION=${1:-status}
OUTPUT="$ROOT/artifacts/phase13/explore/v1_r2_toys_p2_anchored_interleaving"
STATUS="$OUTPUT/status.json"
LOG="$OUTPUT/run.log"
SUMMARY="$OUTPUT/summary.json"
SESSION=gram_phase13_v1_r2_toys_p2_anchored_interleaving
PYTHON=${PYTHON_BIN:-/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python}
HARD_TIMEOUT_SECONDS=${HARD_TIMEOUT_SECONDS:-600}

STARTED_AT=""
RUNNER_PID=0
WORKLOAD_PID=0
STAGE=not_started

write_status() {
  local state=$1 reason=$2 tmp="$STATUS.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE13_V1_R2_TOYS_P2_ANCHORED_INTERLEAVING","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%d,"workload_pid":%d,"tmux_session":"%s","resource_mode":"cpu_background_no_gpu","hard_timeout_seconds":%d,"split":"validation_calibration_audit","test_predictions_opened":false,"log_path":"%s","status_path":"%s","summary_path":"%s"}\n' \
    "$state" "$STAGE" "$reason" "$STARTED_AT" "$(date -Is)" "$RUNNER_PID" "$WORKLOAD_PID" \
    "$SESSION" "$HARD_TIMEOUT_SECONDS" "${LOG#$ROOT/}" "${STATUS#$ROOT/}" "${SUMMARY#$ROOT/}" > "$tmp"
  mv "$tmp" "$STATUS"
}

worker() {
  STARTED_AT=${1:?missing start timestamp}
  RUNNER_PID=$$
  local rc=0
  trap '[[ $WORKLOAD_PID -eq 0 ]] || kill -TERM "$WORKLOAD_PID" 2>/dev/null || true; STAGE=stopped; write_status stopped "Stopped by signal; no automatic retry."; exit 143' TERM INT HUP
  STAGE=preflight
  write_status running "Checking frozen inputs and P2 tests."
  bash -n "$ROOT/experiment/phase13/run_v1_r2_toys_p2_anchored_interleaving.sh" \
    || { write_status failed "Bash syntax check failed; no automatic retry."; return 4; }
  "$PYTHON" -m py_compile \
    "$ROOT/experiment/phase13/protocol/anchored_interleaving.py" \
    "$ROOT/experiment/phase13/tests/test_anchored_interleaving.py" \
    || { write_status failed "Python syntax check failed; no automatic retry."; return 5; }
  cd "$ROOT"
  "$PYTHON" -m pytest -q experiment/phase13/tests/test_anchored_interleaving.py \
    || { write_status failed "P2 unit tests failed; no automatic retry."; return 6; }
  for path in \
    "$ROOT/artifacts/phase13/explore/v1_r2_toys_p0/predictions_validation.jsonl" \
    "$ROOT/GRAM/rec_datasets/Toys_cold50/cold_split_meta/cold_items.txt" \
    "$ROOT/GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt"; do
    [[ -s "$path" ]] || { write_status failed "Required input missing: ${path#$ROOT/}"; return 7; }
  done
  [[ ! -e "$SUMMARY" && ! -e "$OUTPUT/config.json" && ! -e "$OUTPUT/calibration_grid.json" ]] \
    || { write_status failed "Refusing to overwrite an existing P2 scientific artifact."; return 8; }

  STAGE=calibration_select_and_audit
  write_status running "Selecting constrained interleaving on calibration, then evaluating audit once."
  timeout --signal=TERM --kill-after=15 "$HARD_TIMEOUT_SECONDS" \
    "$PYTHON" "$ROOT/experiment/phase13/protocol/anchored_interleaving.py" \
      --p0-predictions "$ROOT/artifacts/phase13/explore/v1_r2_toys_p0/predictions_validation.jsonl" \
      --item-id-file "$ROOT/GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt" \
      --cold-items "$ROOT/GRAM/rec_datasets/Toys_cold50/cold_split_meta/cold_items.txt" \
      --output-dir "$OUTPUT" --calibration-warm-retention 0.98 \
      --audit-warm-retention 0.97 --audit-cold-improvement 2.0 &
  WORKLOAD_PID=$!
  write_status running "P2 CPU workload active; no GPU resource was used or changed."
  wait "$WORKLOAD_PID" || rc=$?
  WORKLOAD_PID=0
  if (( rc != 0 )); then
    STAGE=finished
    write_status failed "P2 exited rc=${rc}; no automatic retry."
    return "$rc"
  fi
  [[ -s "$SUMMARY" ]] \
    || { STAGE=finished; write_status failed "P2 exited 0 without summary.json."; return 9; }
  local verdict
  verdict=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$SUMMARY") \
    || { STAGE=finished; write_status failed "Could not read P2 verdict."; return 10; }
  STAGE=finished
  write_status completed "$verdict"
}

case "$ACTION" in
  start)
    mkdir -p "$OUTPUT"
    [[ ! -e "$STATUS" ]] || { echo "refusing to reuse existing status: $STATUS" >&2; exit 3; }
    tmux has-session -t "$SESSION" 2>/dev/null \
      && { echo "session already exists: $SESSION" >&2; exit 3; }
    STARTED_AT=$(date -Is)
    STAGE=starting
    write_status starting "Background P2 CPU session is starting."
    printf -v launch 'bash %q worker %q >> %q 2>&1' "$0" "$STARTED_AT" "$LOG"
    if ! tmux new-session -d -s "$SESSION" "$launch"; then
      STAGE=launch_failed
      write_status failed "Could not create background tmux session; workload was not started."
      exit 4
    fi
    echo "started $SESSION (CPU)"
    echo "status: $STATUS"
    ;;
  worker)
    worker "${2:?missing start timestamp}"
    ;;
  status)
    [[ -f "$STATUS" ]] && sed -n '1,100p' "$STATUS" || echo '{"status":"not_started"}'
    ;;
  stop)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      tmux send-keys -t "$SESSION" C-c
      echo "stop requested; status file will record the outcome"
    else
      echo "session not running: $SESSION"
    fi
    ;;
  *)
    echo "usage: $0 {start|status|stop}" >&2
    exit 2
    ;;
esac
