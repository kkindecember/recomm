#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="$ROOT/artifacts/phase10/configs/cf1_b2_toys_full_scores_preregistered.json"
OUTPUT="$ROOT/artifacts/phase10/cf1_b2_toys_full_scores"
LOG="$OUTPUT/run.log"
STATUS="$OUTPUT/status.json"
LEASE_STATUS="$OUTPUT/gpu_lease.json"
SESSION=gram_phase10_cf1_b2_full_scores
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
EVALUATOR="$ROOT/experiment/phase10/eval_cf1_b2_full_scores.py"
TEST="$ROOT/experiment/phase10/test_cf1_b2_full_scores.py"
LEASE_HELPER="$ROOT/experiment/gpu_memory_lease.py"
RESERVER=/home/jiangtangyunzhi/projects/UnitTest/tools/run_codellama.sh
CODELLAMA_HF_HOME=/home/jiangtangyunzhi/hf_cache
GPU=6
TOTAL_LEASE_MIB=30720
EXPECTED_PEAK_MIB=8192
STARTED_AT=""
STAGE=not_started
WORKLOAD_PID=0
LEASE_PID=""
TELEMETRY_PID=""
RESERVATION=codellama_expected_on_gpu6

reserver() {
  env SESSION=codellama HF_HOME="$CODELLAMA_HF_HOME" \
    HF_HUB_CACHE="$CODELLAMA_HF_HOME/hub" \
    TRANSFORMERS_CACHE="$CODELLAMA_HF_HOME/hub" "$RESERVER" "$@"
}

write_status() {
  local state=$1 reason=$2 tmp="${STATUS}.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE10_CF1_B2_TOYS_FULL_SCORE_V1","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%s,"workload_pid":%s,"tmux_session":"%s","physical_gpu":6,"resource_reservation":"%s","test_read":false,"beauty_read":false,"sports_read":false}\n' \
    "$state" "$STAGE" "$reason" "$STARTED_AT" "$(date -Is)" "$$" "$WORKLOAD_PID" "$SESSION" "$RESERVATION" > "$tmp"
  mv "$tmp" "$STATUS"
}

verify_locks() {
  "$PYTHON" -c 'import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); config=json.load(open(sys.argv[2]))
assert config["execution_enabled"] is True
assert config["decision_status"] == "AUTHORIZED_CF1_B2_FULL_VALIDATION_SCORING"
for group in ("inputs_sha256", "code_sha256"):
 for rel,expected in config[group].items():
  path=root/rel
  assert path.is_file(), rel
  assert hashlib.sha256(path.read_bytes()).hexdigest()==expected, rel
for raw,expected in config["external_code_sha256"].items():
 path=pathlib.Path(raw)
 assert path.is_file(), raw
 assert hashlib.sha256(path.read_bytes()).hexdigest()==expected, raw
snapshot=pathlib.Path(config["tokenizer_snapshot"])
for rel,expected in config["tokenizer_sha256"].items():
 path=snapshot/rel
 assert path.is_file(), str(path)
 assert hashlib.sha256(path.read_bytes()).hexdigest()==expected, rel' "$ROOT" "$CONFIG"
}

codellama_on_gpu6() {
  local value
  value=$(reserver status 2>&1 || true)
  [[ "$value" == *"tmux session: running (codellama)"* ]] && [[ "$value" == *"gpu=6"* ]]
}

telemetry() {
  printf 'timestamp,index,memory_used_mib,memory_free_mib,utilization_gpu_percent\n' > "$OUTPUT/gpu_telemetry.csv"
  while true; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits --id="$GPU" >> "$OUTPUT/gpu_telemetry.csv" 2>/dev/null || true
    sleep 5
  done
}

release_lease() {
  [[ -z "$LEASE_PID" ]] || kill "$LEASE_PID" >/dev/null 2>&1 || true
  [[ -z "$LEASE_PID" ]] || wait "$LEASE_PID" 2>/dev/null || true
  LEASE_PID=""
}

restore() {
  if codellama_on_gpu6; then
    RESERVATION=codellama_already_running_on_gpu6
    return 0
  fi
  STAGE=resource_restoration
  RESERVATION=restoring_codellama_to_gpu6
  write_status restoring_resource "CF1-B2 ended; restoring CodeLlama on GPU6."
  reserver start "$GPU" || { RESERVATION=restore_request_failed_on_gpu6; return 1; }
  for _ in $(seq 1 180); do
    if codellama_on_gpu6; then
      RESERVATION=restored_on_gpu6
      return 0
    fi
    sleep 5
  done
  RESERVATION=restore_failed_on_gpu6
  return 1
}

finish() {
  local experiment_rc=$? restore_rc=0
  trap - EXIT INT TERM HUP
  if (( WORKLOAD_PID > 0 )) && kill -0 "$WORKLOAD_PID" 2>/dev/null; then
    kill -TERM "$WORKLOAD_PID" 2>/dev/null || true
    wait "$WORKLOAD_PID" 2>/dev/null || true
  fi
  release_lease
  [[ -z "$TELEMETRY_PID" ]] || kill "$TELEMETRY_PID" >/dev/null 2>&1 || true
  restore || restore_rc=$?
  STAGE=finished
  if (( experiment_rc == 0 && restore_rc == 0 )); then
    local gate
    gate=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["scientific_gate"]["status"])' "$OUTPUT/summary.json")
    write_status completed "CF1-B2 engineering completed; scientific gate=$gate; CodeLlama restored."
  elif (( experiment_rc == 0 )); then
    write_status failed_to_restore_resource "CF1-B2 completed but CodeLlama restoration failed."
  else
    write_status failed "CF1-B2 exit=$experiment_rc; no automatic retry; restoration attempted."
  fi
  (( experiment_rc != 0 )) && exit "$experiment_rc"
  exit "$restore_rc"
}

worker() {
  STARTED_AT=${1:?missing start timestamp}
  trap finish EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  trap 'exit 129' HUP
  STAGE=preflight
  write_status preflight "Verifying frozen B2 inputs, code, tests and CodeLlama ownership."
  verify_locks
  "$PYTHON" -m pytest -q "$TEST"
  "$PYTHON" -m py_compile "$EVALUATOR"
  bash -n "$0"
  codellama_on_gpu6 || { write_status blocked "CodeLlama GPU6 ownership not confirmed."; exit 5; }
  STAGE=resource_release
  write_status releasing_resource "Stopping owned CodeLlama reservation before CF1-B2."
  reserver stop
  RESERVATION=released_for_experiment

  STAGE=gpu_memory_gate
  write_status waiting_for_gpu "Waiting for at least 30720 MiB free on GPU6."
  local free_mib=""
  for _ in $(seq 1 120); do
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GPU" 2>/dev/null | tr -d ' ' || true)
    [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= TOTAL_LEASE_MIB )) && break
    sleep 5
  done
  [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= TOTAL_LEASE_MIB )) || { write_status blocked "GPU6 admission failed: ${free_mib:-unknown} MiB free."; exit 6; }

  STAGE=memory_lease
  "$PYTHON" "$LEASE_HELPER" --gpu "$GPU" --total-lease-mib "$TOTAL_LEASE_MIB" \
    --expected-workload-peak-mib "$EXPECTED_PEAK_MIB" --status-path "$LEASE_STATUS" &
  LEASE_PID=$!
  for _ in $(seq 1 30); do
    [[ -s "$LEASE_STATUS" ]] && grep -q '"state": "holding"' "$LEASE_STATUS" && break
    sleep 1
  done
  [[ -s "$LEASE_STATUS" ]] && grep -q '"state": "holding"' "$LEASE_STATUS" || { write_status blocked "GPU lease sidecar did not hold."; exit 7; }
  telemetry & TELEMETRY_PID=$!

  STAGE=full_validation_scoring
  RESERVATION=gpu6_lease_holding
  write_status running "CF1-B2 all-19412-user candidate scoring running."
  CUDA_VISIBLE_DEVICES="$GPU" timeout --signal=TERM 16200 "$PYTHON" "$EVALUATOR" \
    --users 19412 --output-dir "$OUTPUT" --candidate-batch-size 10 &
  WORKLOAD_PID=$!
  wait "$WORKLOAD_PID"
  WORKLOAD_PID=0
  [[ -s "$OUTPUT/summary.json" && -s "$OUTPUT/candidate_scores.tsv" ]]
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
    write_status starting "Persistent CF1-B2 GPU6 session started."
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

