#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="$ROOT/artifacts/phase7/configs/st_gcgd_v2_memory_preregistered.json"
OUTPUT="$ROOT/artifacts/phase7/st_gcgd_v2"
LOG="$OUTPUT/memory_run.log"
STATUS="$OUTPUT/memory_status.json"
SESSION=gram_phase7_st_gcgd_v2_memory
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python3.9
WORKLOAD="$ROOT/experiment/phase7/st_gcgd_v2_memory_pilot.py"
IMPLEMENTATION="$ROOT/experiment/phase7/st_gcgd_v2.py"
TEST="$ROOT/experiment/phase7/test_st_gcgd_v2.py"
RESERVER="$ROOT/tools/run_codellama.sh"
WORKLOAD_PID=0
TELEMETRY_PID=0
CURRENT_DATASET=""
STAGE=not_started
STARTED_AT=""

reserver() {
  env SESSION=codellama HF_HOME=/home/jiangtangyunzhi/hf_cache \
    HF_HUB_CACHE=/home/jiangtangyunzhi/hf_cache/hub TRANSFORMERS_CACHE=/home/jiangtangyunzhi/hf_cache/hub \
    "$RESERVER" "$@"
}

write_status() {
  local state=$1 reason=$2 tmp="${STATUS}.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE7_ST_GCGD_V2_P0_R_MEMORY_V1","status":"%s","stage":"%s","dataset":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%s,"workload_pid":%s,"tmux_session":"%s","physical_gpu":0,"sidecar_active":false,"test_read":false,"sports_read":false}\n' \
    "$state" "$STAGE" "$CURRENT_DATASET" "$reason" "$STARTED_AT" "$(date -Is)" "$$" "$WORKLOAD_PID" "$SESSION" > "$tmp"
  mv "$tmp" "$STATUS"
}

config_ready() {
  "$PYTHON" -c 'import json,sys;c=json.load(open(sys.argv[1]));raise SystemExit(0 if c.get("execution_enabled") is True and c.get("decision_status")=="PREREGISTERED_FROZEN_READY_TO_RUN" else 1)' "$CONFIG"
}

locks_match() {
  "$PYTHON" -c 'import hashlib,json,sys;c=json.load(open(sys.argv[1]))["code_lock"]; paths=sys.argv[2:]; keys=("implementation_sha256","memory_pilot_sha256","test_sha256"); actual=[hashlib.sha256(open(p,"rb").read()).hexdigest() for p in paths]; raise SystemExit(0 if actual==[c[k] for k in keys] else 1)' \
    "$CONFIG" "$IMPLEMENTATION" "$WORKLOAD" "$TEST"
}

codellama_running() {
  local value
  value=$(reserver status 2>&1 || true)
  [[ "$value" == *"tmux session: running (codellama)"* ]] && [[ "$value" == *"gpu=0"* ]]
}

restore() {
  codellama_running && return 0
  reserver start 0 || return $?
  for _ in $(seq 1 60); do
    codellama_running && return 0
    sleep 5
  done
  return 1
}

telemetry() {
  printf 'timestamp,index,memory_used_mib,memory_free_mib,utilization_gpu_percent,dataset\n' > "$OUTPUT/memory_gpu_telemetry.csv"
  while true; do
    local row
    row=$(nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits --id=0 2>/dev/null || true)
    [[ -z "$row" ]] || printf '%s,%s\n' "$row" "$CURRENT_DATASET" >> "$OUTPUT/memory_gpu_telemetry.csv"
    sleep 5
  done
}

finish() {
  local scientific_rc=$? restore_rc=0
  trap - EXIT INT TERM HUP
  (( TELEMETRY_PID == 0 )) || kill "$TELEMETRY_PID" >/dev/null 2>&1 || true
  restore || restore_rc=$?
  STAGE=finished
  if (( scientific_rc == 0 && restore_rc == 0 )); then
    write_status succeeded "P0-R completed; workload-only peaks recorded; CodeLlama restored."
  elif (( scientific_rc == 0 )); then
    write_status failed_to_restore_resource "P0-R completed but CodeLlama restoration failed."
  else
    write_status failed "P0-R exit=$scientific_rc; no automatic retry; CodeLlama restoration requested."
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
  cd "$ROOT"
  mkdir -p "$OUTPUT"
  STAGE=preflight
  config_ready || { write_status blocked "Memory config not frozen/enabled."; exit 3; }
  locks_match || { write_status blocked "Code SHA lock mismatch."; exit 4; }
  "$PYTHON" -m pytest -q "$TEST" || { write_status blocked "P0-R tests failed."; exit 5; }
  codellama_running || { write_status blocked "CodeLlama must be running on GPU0 before release."; exit 6; }
  STAGE=resource_release
  write_status releasing_resource "Stopping CodeLlama for workload-only measurement."
  reserver stop
  local free_mib=""
  for _ in $(seq 1 60); do
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id=0 2>/dev/null | tr -d ' ' || true)
    [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= 30720 )) && break
    sleep 5
  done
  [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= 30720 )) || { write_status blocked "GPU0 admission gate failed."; exit 7; }
  telemetry & TELEMETRY_PID=$!
  for dataset in Toys Beauty; do
    CURRENT_DATASET=$dataset
    STAGE=full_cuda_memory_pilot
    timeout --signal=TERM 21600 env CUDA_VISIBLE_DEVICES=0 HF_HOME="$ROOT/.cache/huggingface" \
      HF_HUB_CACHE="$ROOT/.cache/huggingface" TRANSFORMERS_CACHE="$ROOT/.cache/huggingface" TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
      "$PYTHON" "$WORKLOAD" --config "$CONFIG" --output-root "$OUTPUT" --dataset "$dataset" &
    WORKLOAD_PID=$!
    write_status running "Full train-only CUDA lifecycle memory pilot running."
    wait "$WORKLOAD_PID"
    WORKLOAD_PID=0
  done
}

case "${1:-status}" in
  start)
    mkdir -p "$OUTPUT"
    config_ready || { STAGE=blocked; write_status blocked "Memory config not ready."; exit 3; }
    locks_match || { STAGE=blocked; write_status blocked "Code SHA lock mismatch."; exit 4; }
    tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists: $SESSION" >&2; exit 1; }
    STARTED_AT=$(date -Is)
    printf -v launch_cmd 'bash %q worker %q >> %q 2>&1' "$0" "$STARTED_AT" "$LOG"
    tmux new-session -d -s "$SESSION" "$launch_cmd"
    STAGE=starting
    write_status starting "Persistent P0-R memory pilot started."
    echo "started $SESSION"
    ;;
  worker) worker "${2:?missing start timestamp}" ;;
  status)
    tmux has-session -t "$SESSION" 2>/dev/null && echo "tmux session: running ($SESSION)" || echo "tmux session: not running ($SESSION)"
    [[ -f "$STATUS" ]] && sed -n '1,100p' "$STATUS" || echo '{"status":"not_started"}'
    [[ -f "$LOG" ]] && tail -n 50 "$LOG" || true
    reserver status || true
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits --id=0 || true
    ;;
  *) echo "usage: $0 {start|status|worker}" >&2; exit 2 ;;
esac
