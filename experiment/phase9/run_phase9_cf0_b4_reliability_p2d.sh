#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="$ROOT/artifacts/phase9/configs/cf0_b4_toys_reliability_p2d_preregistered.json"
OUTPUT="$ROOT/artifacts/phase9/cf0_b4_toys_reliability_p2d"
LOG="$OUTPUT/run.log"
STATUS="$OUTPUT/status.json"
SESSION=gram_phase9_cf0_b4_reliability_p2d
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
EVALUATOR="$ROOT/experiment/phase9/eval_cf0_b4_reliability.py"
TEST="$ROOT/experiment/phase9/test_cf0_b4_reliability.py"
STARTED_AT=""
STAGE=not_started
WORKLOAD_PID=0

write_status() {
  local state=$1 reason=$2 tmp="${STATUS}.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE9_CF0_B4_TOYS_RELIABILITY_FUSION_P2D_V1","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%s,"workload_pid":%s,"tmux_session":"%s","device":"cpu","test_read":false,"sports_read":false}\n' \
    "$state" "$STAGE" "$reason" "$STARTED_AT" "$(date -Is)" "$$" "$WORKLOAD_PID" "$SESSION" > "$tmp"
  mv "$tmp" "$STATUS"
}

verify_config_and_locks() {
  "$PYTHON" -c 'import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); config=json.load(open(sys.argv[2]))
assert config["execution_enabled"] is True
assert config["decision_status"] == "AUTHORIZED_P9_2D_CROSSFIT_RELIABILITY_DEVELOPMENT"
for group in ("inputs_sha256", "code_sha256"):
 for rel,expected in config[group].items():
  path=root/rel
  assert path.is_file(), rel
  assert hashlib.sha256(path.read_bytes()).hexdigest()==expected, rel' "$ROOT" "$CONFIG"
}

finish() {
  local rc=$?
  trap - EXIT INT TERM HUP
  if (( WORKLOAD_PID > 0 )) && kill -0 "$WORKLOAD_PID" 2>/dev/null; then
    kill -TERM "$WORKLOAD_PID" 2>/dev/null || true
    wait "$WORKLOAD_PID" 2>/dev/null || true
  fi
  STAGE=finished
  if (( rc == 0 )); then
    local gate
    gate=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["development_gate"]["status"])' "$OUTPUT/summary.json")
    write_status completed "P9-2D engineering completed; development gate=$gate."
  else
    write_status failed "P9-2D engineering exit=$rc; no automatic retry."
  fi
  exit "$rc"
}

worker() {
  STARTED_AT=${1:?missing start timestamp}
  trap finish EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  trap 'exit 129' HUP
  STAGE=preflight
  write_status preflight "Verifying frozen inputs, code and tests."
  verify_config_and_locks
  "$PYTHON" -m pytest -q "$TEST"
  "$PYTHON" -m py_compile "$EVALUATOR"
  bash -n "$0"
  STAGE=cross_fitted_evaluation
  write_status running "P9-2D five-fold PCRF development experiment running."
  timeout --signal=TERM 1800 "$PYTHON" "$EVALUATOR" \
    --output-dir "$OUTPUT" --fold-seed 2023 --num-folds 5 \
    --bootstrap-replicates 2000 --seed 2023 &
  WORKLOAD_PID=$!
  wait "$WORKLOAD_PID"
  WORKLOAD_PID=0
  [[ -s "$OUTPUT/summary.json" && -s "$OUTPUT/fold_assignments.tsv" && -s "$OUTPUT/per_user_oof.tsv" ]]
}

case "${1:-status}" in
  start)
    mkdir -p "$OUTPUT"
    verify_config_and_locks
    tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists: $SESSION" >&2; exit 1; }
    STARTED_AT=$(date -Is)
    printf -v launch_cmd 'bash %q worker %q >> %q 2>&1' "$0" "$STARTED_AT" "$LOG"
    tmux new-session -d -s "$SESSION" "$launch_cmd"
    STAGE=starting
    write_status starting "Persistent P9-2D CPU session started."
    echo "started $SESSION"
    ;;
  worker) worker "${2:?missing start timestamp}" ;;
  status)
    tmux has-session -t "$SESSION" 2>/dev/null && echo "tmux session: running ($SESSION)" || echo "tmux session: not running ($SESSION)"
    [[ -f "$STATUS" ]] && sed -n '1,100p' "$STATUS" || echo '{"status":"not_started"}'
    [[ -f "$LOG" ]] && tail -n 60 "$LOG" || true
    ;;
  stop)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      pane_pid=$(tmux list-panes -t "$SESSION" -F '#{pane_pid}' | head -n 1)
      kill -TERM "$pane_pid"
      echo "stop requested for $SESSION"
    else
      echo "session not running: $SESSION"
    fi
    ;;
  *) echo "usage: $0 {start|status|stop|worker}" >&2; exit 2 ;;
esac
