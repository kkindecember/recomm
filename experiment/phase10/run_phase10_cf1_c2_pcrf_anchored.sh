#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SMOKE_CONFIG="$ROOT/artifacts/phase10/configs/cf1_c2_toys_pcrf_anchored_preregistered.json"
FORMAL_CONFIG="$ROOT/artifacts/phase10/configs/cf1_c2_toys_pcrf_anchored_formal_preregistered.json"
SMOKE_OUTPUT="$ROOT/artifacts/phase10/cf1_c2_toys_pcrf_anchored_smoke"
FORMAL_OUTPUT="$ROOT/artifacts/phase10/cf1_c2_toys_pcrf_anchored"
LOG="$SMOKE_OUTPUT/run.log"
STATUS="$SMOKE_OUTPUT/status.json"
FORMAL_LOG="$FORMAL_OUTPUT/run.log"
FORMAL_STATUS="$FORMAL_OUTPUT/status.json"
SESSION=gram_phase10_cf1_c2_pcrf_anchored
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
EVALUATOR="$ROOT/experiment/phase10/eval_cf1_c2_pcrf_anchored.py"
TEST="$ROOT/experiment/phase10/test_cf1_c2_pcrf_anchored.py"

write_status() {
  local state=$1 reason=$2 tmp="${STATUS}.tmp.$$"
  mkdir -p "$SMOKE_OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE10_CF1_C2_TOYS_PCRF_ANCHORED_SMOKE_V1","status":"%s","reason":"%s","updated_at":"%s","runner_pid":%s,"test_read":false,"beauty_read":false,"sports_read":false}\n' \
    "$state" "$reason" "$(date -Is)" "$$" > "$tmp"
  mv "$tmp" "$STATUS"
}

verify_smoke_locks() {
  "$PYTHON" -c 'import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); config=json.load(open(sys.argv[2]))
assert config["smoke_enabled"] is True
assert config["formal_execution_enabled"] is False
assert config["decision_status"] == "AUTHORIZED_CF1_C2_IMPLEMENTATION_SMOKE_ONLY"
for group in ("inputs_sha256", "code_sha256"):
 for rel,expected in config[group].items():
  path=root/rel
  assert path.is_file(), rel
  assert hashlib.sha256(path.read_bytes()).hexdigest()==expected, rel' "$ROOT" "$SMOKE_CONFIG"
}

verify_formal_locks() {
  "$PYTHON" -c 'import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); config=json.load(open(sys.argv[2]))
assert config["formal_execution_enabled"] is True
assert config["decision_status"] == "AUTHORIZED_CF1_C2_FORMAL_FIVE_FOLD_OOF"
for group in ("inputs_sha256", "code_sha256"):
 for rel,expected in config[group].items():
  path=root/rel
  assert path.is_file(), rel
  assert hashlib.sha256(path.read_bytes()).hexdigest()==expected, rel' "$ROOT" "$FORMAL_CONFIG"
}

write_formal_status() {
  local state=$1 stage=$2 reason=$3 started_at=$4 tmp="${FORMAL_STATUS}.tmp.$$"
  mkdir -p "$FORMAL_OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE10_CF1_C2_TOYS_PCRF_ANCHORED_V1","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%s,"tmux_session":"%s","test_read":false,"beauty_read":false,"sports_read":false}\n' \
    "$state" "$stage" "$reason" "$started_at" "$(date -Is)" "$$" "$SESSION" > "$tmp"
  mv "$tmp" "$FORMAL_STATUS"
}

formal_worker() {
  local started_at=${1:?missing start timestamp} rc gate
  trap 'write_formal_status failed finished "C2 formal interrupted; no automatic retry." "$started_at"; exit 143' TERM INT HUP
  write_formal_status preflight preflight "Verifying formal C2 locks, tests and evaluator syntax." "$started_at"
  verify_formal_locks
  "$PYTHON" -m pytest -q "$TEST"
  "$PYTHON" -m py_compile "$EVALUATOR"
  write_formal_status running five_fold_oof "Running single full 19,412-user C2 OOF evaluation." "$started_at"
  set +e
  timeout --signal=TERM 3600 "$PYTHON" "$EVALUATOR" --output-dir "$FORMAL_OUTPUT"
  rc=$?
  set -e
  if (( rc == 0 )) && [[ -s "$FORMAL_OUTPUT/summary.json" && -s "$FORMAL_OUTPUT/per_user_oof.tsv" ]]; then
    gate=$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["development_gate"]["status"])' "$FORMAL_OUTPUT/summary.json")
    write_formal_status completed finished "C2 formal completed; development gate=$gate." "$started_at"
  else
    write_formal_status failed finished "C2 formal exit=$rc; no automatic retry." "$started_at"
  fi
  return "$rc"
}

case "${1:-status}" in
  smoke)
    mkdir -p "$SMOKE_OUTPUT"
    [[ ! -e "$SMOKE_OUTPUT/summary.json" ]] || {
      echo "smoke summary already exists; refusing rerun" >&2
      exit 1
    }
    write_status preflight "Verifying C2 locks, unit tests and evaluator syntax."
    verify_smoke_locks
    "$PYTHON" -m pytest -q "$TEST" >> "$LOG" 2>&1
    "$PYTHON" -m py_compile "$EVALUATOR" >> "$LOG" 2>&1
    write_status running "Running deterministic 512-user C2 implementation smoke."
    set +e
    timeout --signal=TERM 900 "$PYTHON" "$EVALUATOR" \
      --users 512 --output-dir "$SMOKE_OUTPUT" >> "$LOG" 2>&1
    rc=$?
    set -e
    if (( rc == 0 )) && [[ -s "$SMOKE_OUTPUT/summary.json" ]]; then
      gate=$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["implementation_gate"]["status"])' "$SMOKE_OUTPUT/summary.json")
      write_status completed "C2 smoke completed; implementation gate=$gate."
    else
      write_status failed "C2 smoke exit=$rc; no automatic retry."
    fi
    exit "$rc"
    ;;
  start)
    verify_formal_locks
    [[ ! -e "$FORMAL_OUTPUT/summary.json" ]] || {
      echo "formal summary already exists; refusing rerun" >&2
      exit 1
    }
    tmux has-session -t "$SESSION" 2>/dev/null && {
      echo "session already exists: $SESSION" >&2
      exit 1
    }
    mkdir -p "$FORMAL_OUTPUT"
    started_at=$(date -Is)
    printf -v launch_cmd 'bash %q worker %q >> %q 2>&1' "$0" "$started_at" "$FORMAL_LOG"
    tmux new-session -d -s "$SESSION" "$launch_cmd"
    write_formal_status starting starting "Persistent formal C2 session started." "$started_at"
    echo "started $SESSION"
    ;;
  worker) formal_worker "${2:?missing start timestamp}" ;;
  status)
    tmux has-session -t "$SESSION" 2>/dev/null && echo "tmux session: running ($SESSION)" || echo "tmux session: not running ($SESSION)"
    [[ -f "$FORMAL_STATUS" ]] && sed -n '1,100p' "$FORMAL_STATUS" || echo '{"status":"not_started"}'
    [[ -f "$FORMAL_LOG" ]] && tail -n 80 "$FORMAL_LOG" || true
    ;;
  smoke-status)
    [[ -f "$STATUS" ]] && sed -n '1,100p' "$STATUS" || echo '{"status":"not_started"}'
    [[ -f "$LOG" ]] && tail -n 80 "$LOG" || true
    ;;
  *) echo "usage: $0 {smoke|start|worker|status|smoke-status}" >&2; exit 2 ;;
esac
