#!/usr/bin/env bash
# Phase-7 GCGD runner scaffold. GPU execution remains fail-closed until frozen.
set -euo pipefail

ROOT=/home/jiangtangyunzhi/projects/recomm
CONFIG="$ROOT/artifacts/phase7/configs/gcgd_v1_design.json"
OUTPUT="$ROOT/artifacts/phase7/gcgd_v1"
LOG="$OUTPUT/run.log"
STATUS="$OUTPUT/status.json"
SESSION=gram_phase7_gcgd_v1
GPU=0
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python3.9
WORKLOAD="$ROOT/experiment/phase7/gcgd_v1.py"
LEASE_HELPER="$ROOT/experiment/gpu_memory_lease.py"
TOTAL_LEASE_MIB=30720
EXPECTED_WORKLOAD_PEAK_MIB=24576
RESERVER=/home/jiangtangyunzhi/projects/UnitTest/tools/run_codellama.sh
CODELLAMA_HF_HOME=/home/jiangtangyunzhi/hf_cache
WORKLOAD_PID=0
LEASE_PID=""
TELEMETRY_PID=""
STARTED_AT=""
STAGE=not_started
RESERVATION=codellama_expected_on_gpu0

reserver() {
  env SESSION=codellama HF_HOME="$CODELLAMA_HF_HOME" \
    HF_HUB_CACHE="$CODELLAMA_HF_HOME/hub" TRANSFORMERS_CACHE="$CODELLAMA_HF_HOME/hub" \
    "$RESERVER" "$@"
}

write_status() {
  local state=$1 reason=$2 tmp="${STATUS}.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE7_GCGD_V1","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%s,"workload_pid":%s,"tmux_session":"%s","physical_gpu":0,"total_gpu_lease_mib":30720,"expected_workload_peak_mib":24576,"log_path":"%s","result_path":"%s","resource_reservation":"%s","test_read":false,"sports_read":false}\n' \
    "$state" "$STAGE" "$reason" "$STARTED_AT" "$(date -Is)" "$$" "$WORKLOAD_PID" "$SESSION" \
    "${LOG#$ROOT/}" "${OUTPUT#$ROOT/}/summary.json" "$RESERVATION" > "$tmp"
  mv "$tmp" "$STATUS"
}

config_is_frozen_and_enabled() {
  "$PYTHON" -c 'import json,sys; c=json.load(open(sys.argv[1])); ready = c.get("execution_enabled") is True and c.get("scientific_workload_implemented") is True and c.get("decision_status") == "PREREGISTERED_FROZEN_READY_TO_RUN"; raise SystemExit(0 if ready else 1)' "$CONFIG"
}

codellama_is_running_on_gpu0() {
  local value
  value=$(reserver status 2>&1 || true)
  [[ "$value" == *"tmux session: running (codellama)"* ]] && [[ "$value" == *"gpu=0"* ]]
}

telemetry() {
  printf 'timestamp,index,memory_used_mib,memory_free_mib,utilization_gpu_percent\n' > "$OUTPUT/gpu_telemetry.csv"
  while true; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits --id="$GPU" >> "$OUTPUT/gpu_telemetry.csv" || true
    sleep 5
  done
}

restore() {
  if codellama_is_running_on_gpu0; then
    RESERVATION=codellama_already_running_on_gpu0
    return 0
  fi
  RESERVATION=restoring_codellama_to_gpu0
  STAGE=resource_restoration
  write_status restoring_resource "Experiment ended; restoring CodeLlama on physical GPU0."
  if reserver start "$GPU"; then
    RESERVATION=codellama_restore_requested_on_gpu0
    return 0
  fi
  RESERVATION=codellama_restore_failed_on_gpu0
  return 1
}

finish() {
  local scientific_rc=$? restore_rc=0
  trap - EXIT INT TERM HUP
  [[ -z "$LEASE_PID" ]] || kill "$LEASE_PID" >/dev/null 2>&1 || true
  [[ -z "$LEASE_PID" ]] || wait "$LEASE_PID" 2>/dev/null || true
  [[ -z "$TELEMETRY_PID" ]] || kill "$TELEMETRY_PID" >/dev/null 2>&1 || true
  restore || restore_rc=$?
  STAGE=finished
  if (( scientific_rc == 0 && restore_rc == 0 )); then
    write_status succeeded "GCGD-v1 completed; results await researcher-requested analysis."
  elif (( scientific_rc == 0 )); then
    write_status failed_to_restore_resource "Scientific workload completed but CodeLlama restoration failed."
  else
    write_status failed "Scientific exit=$scientific_rc; no automatic retry; CodeLlama restoration requested on GPU0."
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
  for required in "$CONFIG" "$WORKLOAD" "$LEASE_HELPER"; do
    [[ -s "$required" ]] || { write_status blocked "Required input missing: $required"; exit 2; }
  done
  config_is_frozen_and_enabled || { write_status blocked "Design is not preregistered and execution-enabled."; exit 3; }
  codellama_is_running_on_gpu0 || { write_status blocked "CodeLlama must be running on physical GPU0 before experiment start."; exit 4; }
  STAGE=resource_release
  write_status releasing_resource "Stopping CodeLlama before GCGD-v1 on physical GPU0."
  reserver stop
  RESERVATION=released_for_experiment
  STAGE=gpu_memory_gate
  local free_mib=""
  for _ in $(seq 1 60); do
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GPU" 2>/dev/null | tr -d ' ' || true)
    [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= TOTAL_LEASE_MIB )) && break
    sleep 60
  done
  [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= TOTAL_LEASE_MIB )) || { write_status blocked "GPU0 free memory below 30720 MiB."; exit 5; }
  telemetry & TELEMETRY_PID=$!
  "$PYTHON" "$LEASE_HELPER" --gpu "$GPU" --total-lease-mib "$TOTAL_LEASE_MIB" \
    --expected-workload-peak-mib "$EXPECTED_WORKLOAD_PEAK_MIB" \
    --status-path "$OUTPUT/gpu_lease_status.json" &
  LEASE_PID=$!
  for _ in $(seq 1 30); do
    [[ -f "$OUTPUT/gpu_lease_status.json" ]] && break
    sleep 1
  done
  [[ -f "$OUTPUT/gpu_lease_status.json" ]] || { write_status blocked "GPU lease sidecar did not become ready."; exit 6; }
  STAGE=scientific_workload
  timeout --signal=TERM 129600 env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$WORKLOAD" \
    --design-check "$CONFIG" &
  WORKLOAD_PID=$!
  write_status running "GCGD-v1 running on physical GPU0 with a 30 GiB total lease."
  wait "$WORKLOAD_PID"
}

case "${1:-status}" in
  start)
    mkdir -p "$OUTPUT"
    if ! config_is_frozen_and_enabled; then
      STAGE=design_not_frozen
      write_status blocked "Start refused: design config is not preregistered and execution-enabled; CodeLlama was not stopped."
      echo "blocked: design is not frozen; CodeLlama unchanged"
      exit 3
    fi
    tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists: $SESSION" >&2; exit 1; }
    STARTED_AT=$(date -Is)
    printf -v launch_cmd 'bash %q worker %q >> %q 2>&1' "$0" "$STARTED_AT" "$LOG"
    tmux new-session -d -s "$SESSION" "$launch_cmd"
    STAGE=starting
    RESERVATION=codellama_expected_on_gpu0
    write_status starting "Persistent GCGD-v1 session started for physical GPU0."
    echo "started $SESSION"
    ;;
  worker) worker "${2:?missing start timestamp}" ;;
  status)
    tmux has-session -t "$SESSION" 2>/dev/null && echo "tmux session: running ($SESSION)" || echo "tmux session: not running ($SESSION)"
    [[ -f "$STATUS" ]] && sed -n '1,100p' "$STATUS" || echo '{"status":"not_started","execution_enabled":false}'
    [[ -f "$LOG" ]] && tail -n 40 "$LOG" || true
    ;;
  *) echo "usage: $0 {start|status|worker}" >&2; exit 2 ;;
esac
