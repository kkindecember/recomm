#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="$ROOT/artifacts/phase9/configs/cf0_b2_toys_item_p2a_preregistered.json"
OUTPUT="$ROOT/artifacts/phase9/cf0_b2_toys_item_p2a"
SMOKE_OUTPUT="$ROOT/artifacts/phase9/cf0_b2_toys_item_p2a_smoke"
LOG="$OUTPUT/run.log"
STATUS="$OUTPUT/status.json"
LEASE_STATUS="$OUTPUT/gpu_lease.json"
SESSION=gram_phase9_cf0_b2_item_p2a
GPU=6
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
TRAINER="$ROOT/experiment/phase9/train_cf0_b2_item_head.py"
TEST="$ROOT/experiment/phase9/test_cf0_b2_item_head.py"
LEASE_HELPER="$ROOT/experiment/gpu_memory_lease.py"
RESERVER="$ROOT/tools/run_codellama.sh"
CODELLAMA_HF_HOME=/home/jiangtangyunzhi/hf_cache
TOTAL_LEASE_MIB=30720
EXPECTED_PEAK_MIB=8192
HARD_TIMEOUT_SECONDS=14400
WORKLOAD_PID=0
LEASE_PID=""
TELEMETRY_PID=""
STARTED_AT=""
STAGE=not_started
SCI_GATE=not_evaluated
RESERVATION=codellama_expected_on_gpu6

reserver() {
  env SESSION=codellama HF_HOME="$CODELLAMA_HF_HOME" \
    HF_HUB_CACHE="$CODELLAMA_HF_HOME/hub" \
    TRANSFORMERS_CACHE="$CODELLAMA_HF_HOME/hub" "$RESERVER" "$@"
}

write_status() {
  local state=$1 reason=$2 tmp="${STATUS}.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE9_CF0_B2_TOYS_ITEM_P2A_V1","status":"%s","stage":"%s","reason":"%s","scientific_gate":"%s","started_at":"%s","updated_at":"%s","runner_pid":%s,"workload_pid":%s,"tmux_session":"%s","physical_gpu":6,"resource_reservation":"%s","test_read":false,"sports_read":false}\n' \
    "$state" "$STAGE" "$reason" "$SCI_GATE" "$STARTED_AT" "$(date -Is)" "$$" \
    "$WORKLOAD_PID" "$SESSION" "$RESERVATION" > "$tmp"
  mv "$tmp" "$STATUS"
}

config_enabled() {
  "$PYTHON" -c 'import json,sys; c=json.load(open(sys.argv[1])); raise SystemExit(0 if c.get("execution_enabled") is True and c.get("decision_status")=="AUTHORIZED_P9_2A_ISOLATED_ITEM_HEAD" else 1)' "$CONFIG"
}

verify_locks() {
  "$PYTHON" -c 'import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); config=json.load(open(sys.argv[2]))
for rel,expected in config["code_lock"]["files"].items():
 path=root/rel
 assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==expected, rel' "$ROOT" "$CONFIG"
}

codellama_on_target() {
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
  if codellama_on_target; then
    RESERVATION=codellama_already_running_on_gpu6
    return 0
  fi
  STAGE=resource_restoration
  RESERVATION=restoring_codellama_to_gpu6
  write_status restoring_resource "P9-2A ended; restoring CodeLlama on GPU6."
  reserver start "$GPU" || { RESERVATION=restore_request_failed_on_gpu6; return 1; }
  for _ in $(seq 1 180); do
    if codellama_on_target; then
      RESERVATION=restored_on_gpu6
      return 0
    fi
    sleep 5
  done
  RESERVATION=restore_failed_on_gpu6
  return 1
}

finish() {
  local scientific_rc=$? restore_rc=0
  trap - EXIT INT TERM HUP
  if (( WORKLOAD_PID > 0 )) && kill -0 "$WORKLOAD_PID" 2>/dev/null; then
    kill -TERM "$WORKLOAD_PID" 2>/dev/null || true
    wait "$WORKLOAD_PID" 2>/dev/null || true
  fi
  release_lease
  [[ -z "$TELEMETRY_PID" ]] || kill "$TELEMETRY_PID" >/dev/null 2>&1 || true
  restore || restore_rc=$?
  STAGE=finished
  if (( scientific_rc == 0 && restore_rc == 0 )); then
    write_status completed "P9-2A completed; scientific gate=$SCI_GATE."
  elif (( scientific_rc == 0 )); then
    write_status failed_to_restore_resource "P9-2A completed but CodeLlama restoration failed."
  else
    write_status failed "P9-2A engineering exit=$scientific_rc; no automatic retry."
  fi
  (( scientific_rc != 0 )) && exit "$scientific_rc"
  exit "$restore_rc"
}

worker() {
  STARTED_AT=${1:?missing start timestamp}
  trap finish EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  trap 'exit 129' HUP
  mkdir -p "$OUTPUT"
  STAGE=preflight
  write_status preflight "Checking frozen P9-2A code, data, config and tests."
  config_enabled || exit 3
  verify_locks || exit 4
  "$PYTHON" -m pytest -q "$TEST"
  "$PYTHON" -m py_compile "$TRAINER" "$ROOT/experiment/phase9/cf0_diagnostic_metrics.py"
  bash -n "$0"

  STAGE=resource_release
  write_status releasing_resource "Stopping CodeLlama before P9-2A smoke and full run."
  codellama_on_target || { write_status blocked "CodeLlama was not confirmed on GPU6."; exit 5; }
  reserver stop
  RESERVATION=released_for_experiment

  STAGE=gpu_memory_gate
  local free_mib=""
  for _ in $(seq 1 120); do
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GPU" 2>/dev/null | tr -d ' ' || true)
    [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= TOTAL_LEASE_MIB )) && break
    sleep 5
  done
  [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= TOTAL_LEASE_MIB )) || exit 6

  STAGE=memory_lease
  "$PYTHON" "$LEASE_HELPER" --gpu "$GPU" --total-lease-mib "$TOTAL_LEASE_MIB" \
    --expected-workload-peak-mib "$EXPECTED_PEAK_MIB" --status-path "$LEASE_STATUS" &
  LEASE_PID=$!
  for _ in $(seq 1 30); do
    [[ -s "$LEASE_STATUS" ]] && grep -q '"state": "holding"' "$LEASE_STATUS" && break
    sleep 1
  done
  [[ -s "$LEASE_STATUS" ]] && grep -q '"state": "holding"' "$LEASE_STATUS" || exit 7
  telemetry & TELEMETRY_PID=$!

  STAGE=gpu_smoke
  write_status running "P9-2A same-batch GPU smoke running."
  timeout --signal=TERM 600 env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$TRAINER" \
    --device cuda:0 --epochs 1 --batch-size 512 --eval-batch-size 1024 \
    --max-train-samples 1024 --max-validation-samples 512 \
    --output-dir "$SMOKE_OUTPUT"
  "$PYTHON" -c 'import json,sys; s=json.load(open(sys.argv[1])); p=s["resource"]["peak_reserved_mib"]; assert p <= float(sys.argv[2]), (p,sys.argv[2])' \
    "$SMOKE_OUTPUT/summary.json" "$EXPECTED_PEAK_MIB"

  STAGE=full_training
  write_status running "P9-2A isolated item-head full training and validation running."
  timeout --signal=TERM "$HARD_TIMEOUT_SECONDS" env CUDA_VISIBLE_DEVICES="$GPU" \
    "$PYTHON" "$TRAINER" --device cuda:0 --epochs 10 --batch-size 512 \
    --eval-batch-size 1024 --learning-rate 3e-4 --weight-decay 0.01 \
    --warmup-ratio 0.05 --max-history 20 --d-model 512 --num-layers 2 \
    --num-heads 4 --dropout 0.1 --temperature 0.07 \
    --gate-relative-margin 0.20 --nonhead-recall50-min 0.005 \
    --output-dir "$OUTPUT" &
  WORKLOAD_PID=$!
  wait "$WORKLOAD_PID"
  WORKLOAD_PID=0
  [[ -s "$OUTPUT/summary.json" && -s "$OUTPUT/best_item_head.pt" && -s "$OUTPUT/best_validation_ranks.tsv" ]] || exit 8
  SCI_GATE=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["scientific_gate"]["status"])' "$OUTPUT/summary.json")
}

case "${1:-status}" in
  start)
    mkdir -p "$OUTPUT"
    config_enabled || exit 3
    tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists: $SESSION" >&2; exit 1; }
    STARTED_AT=$(date -Is)
    printf -v launch_cmd 'bash %q worker %q >> %q 2>&1' "$0" "$STARTED_AT" "$LOG"
    tmux new-session -d -s "$SESSION" "$launch_cmd"
    STAGE=starting
    write_status starting "Persistent P9-2A session started."
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
