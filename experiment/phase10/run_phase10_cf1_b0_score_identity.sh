#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="$ROOT/artifacts/phase10/configs/cf1_b0_toys_score_identity_preregistered.json"
OUTPUT="$ROOT/artifacts/phase10/cf1_b0_toys_score_identity"
LOG="$OUTPUT/run.log"
STATUS="$OUTPUT/status.json"
SESSION=gram_phase10_cf1_b0_score_identity
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
EVALUATOR="$ROOT/experiment/phase10/eval_cf1_b0_score_identity.py"
TEST="$ROOT/experiment/phase10/test_cf1_b0_score_identity.py"
GPU=5
STARTED_AT=""
STAGE=not_started
WORKLOAD_PID=0

write_status() {
  local state=$1 reason=$2 tmp="${STATUS}.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE10_CF1_B0_TOYS_SCORE_IDENTITY_V1","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%s,"workload_pid":%s,"tmux_session":"%s","physical_gpu":5,"test_read":false,"beauty_read":false,"sports_read":false}\n' \
    "$state" "$STAGE" "$reason" "$STARTED_AT" "$(date -Is)" "$$" "$WORKLOAD_PID" "$SESSION" > "$tmp"
  mv "$tmp" "$STATUS"
}

verify_locks() {
  "$PYTHON" -c 'import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); config=json.load(open(sys.argv[2]))
assert config["execution_enabled"] is True
assert config["decision_status"] == "AUTHORIZED_CF1_B0_SCORE_IDENTITY"
for group in ("inputs_sha256", "code_sha256"):
 for rel,expected in config[group].items():
  path=root/rel
  assert path.is_file(), rel
  assert hashlib.sha256(path.read_bytes()).hexdigest()==expected, rel
snapshot=pathlib.Path(config["tokenizer_snapshot"])
for rel,expected in config["tokenizer_sha256"].items():
 path=snapshot/rel
 assert path.is_file(), str(path)
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
    gate=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["scientific_gate"]["status"])' "$OUTPUT/summary.json")
    write_status completed "CF1-B0 engineering completed; scientific gate=$gate."
  else
    write_status failed "CF1-B0 engineering exit=$rc; no automatic retry."
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
  write_status preflight "Verifying frozen B0 inputs, tokenizer, code, tests and GPU lease."
  verify_locks
  "$PYTHON" -m pytest -q "$TEST"
  "$PYTHON" -m py_compile "$EVALUATOR"
  bash -n "$0"
  local free_mib
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GPU" | tr -d ' ')
  (( free_mib >= 12000 )) || { echo "GPU $GPU free memory too low: $free_mib MiB" >&2; exit 1; }
  STAGE=score_identity_64_users
  write_status running "CF1-B0 64-user cached-G50 score identity running."
  CUDA_VISIBLE_DEVICES="$GPU" timeout --signal=TERM 1800 "$PYTHON" "$EVALUATOR" \
    --users 64 --output-dir "$OUTPUT" --candidate-batch-size 10 &
  WORKLOAD_PID=$!
  wait "$WORKLOAD_PID"
  WORKLOAD_PID=0
  [[ -s "$OUTPUT/summary.json" && -s "$OUTPUT/score_pairs.tsv" ]]
}

case "${1:-status}" in
  start)
    mkdir -p "$OUTPUT"
    verify_locks
    [[ ! -e "$OUTPUT/summary.json" ]] || { echo "formal summary already exists; refusing rerun" >&2; exit 1; }
    tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists: $SESSION" >&2; exit 1; }
    STARTED_AT=$(date -Is)
    printf -v launch_cmd 'bash %q worker %q >> %q 2>&1' "$0" "$STARTED_AT" "$LOG"
    tmux new-session -d -s "$SESSION" "$launch_cmd"
    STAGE=starting
    write_status starting "Persistent CF1-B0 GPU5 session started."
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

