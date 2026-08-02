#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="$ROOT/artifacts/phase7/configs/gcgd_p1_train_smoke_preregistered.json"
OUTPUT="$ROOT/artifacts/phase7/gcgd_p1_train_smoke"
LOG="$OUTPUT/run.log"
STATUS="$OUTPUT/status.json"
SESSION=gram_phase7_gcgd_p1_train_smoke
GPU=0
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python3.9
WORKLOAD="$ROOT/experiment/phase7/gcgd_p1_train_smoke.py"
TEST="$ROOT/experiment/phase7/test_gcgd_p1_train_smoke.py"
LEASE_HELPER="$ROOT/experiment/gpu_memory_lease.py"
RESERVER=/home/jiangtangyunzhi/projects/UnitTest/tools/run_codellama.sh
CODELLAMA_HF_HOME=/home/jiangtangyunzhi/hf_cache
WORKLOAD_HF_HOME="$ROOT/.cache/huggingface"
TOTAL_LEASE_MIB=30720
WORKLOAD_PID=0
LEASE_PID=""
TELEMETRY_PID=""
STARTED_AT=""
STAGE=not_started
CURRENT_DATASET=""
RESERVATION=codellama_expected_on_gpu0

reserver() {
  env SESSION=codellama HF_HOME="$CODELLAMA_HF_HOME" \
    HF_HUB_CACHE="$CODELLAMA_HF_HOME/hub" TRANSFORMERS_CACHE="$CODELLAMA_HF_HOME/hub" \
    "$RESERVER" "$@"
}

write_status() {
  local state=$1 reason=$2 tmp="${STATUS}.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE7_GCGD_P1_TRAIN_ONLY_SMOKE_V1","status":"%s","stage":"%s","dataset":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%s,"workload_pid":%s,"tmux_session":"%s","physical_gpu":0,"total_gpu_lease_mib":30720,"log_path":"%s","resource_reservation":"%s","fresh_validation_read":false,"test_read":false,"sports_read":false}\n' \
    "$state" "$STAGE" "$CURRENT_DATASET" "$reason" "$STARTED_AT" "$(date -Is)" "$$" \
    "$WORKLOAD_PID" "$SESSION" "${LOG#$ROOT/}" "$RESERVATION" > "$tmp"
  mv "$tmp" "$STATUS"
}

config_is_enabled() {
  "$PYTHON" -c 'import json,sys; c=json.load(open(sys.argv[1])); ok=c.get("execution_enabled") is True and c.get("decision_status") == "PREREGISTERED_FROZEN_READY_TO_RUN"; raise SystemExit(0 if ok else 1)' "$CONFIG"
}

code_lock_matches() {
  "$PYTHON" -c 'import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); c=json.load(open(sys.argv[2])); lock=c["code_lock"]
for name in ("implementation","tests","runner"):
 p=root/lock[name]
 if hashlib.sha256(p.read_bytes()).hexdigest()!=lock[name+"_sha256"]: raise SystemExit(1)
for spec in lock.get("dependencies",{}).values():
 p=root/spec["path"]
 if hashlib.sha256(p.read_bytes()).hexdigest()!=spec["sha256"]: raise SystemExit(1)' "$ROOT" "$CONFIG"
}

codellama_is_running_on_gpu0() {
  local value
  value=$(reserver status 2>&1 || true)
  [[ "$value" == *"tmux session: running (codellama)"* ]] && [[ "$value" == *"gpu=0"* ]]
}

telemetry() {
  printf 'timestamp,index,memory_used_mib,memory_free_mib,utilization_gpu_percent,dataset\n' > "$OUTPUT/gpu_telemetry.csv"
  while true; do
    local row dataset
    row=$(nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits --id="$GPU" 2>/dev/null || true)
    dataset=$(sed -n '1p' "$OUTPUT/current_dataset.txt" 2>/dev/null || true)
    [[ -z "$row" ]] || printf '%s,%s\n' "$row" "$dataset" >> "$OUTPUT/gpu_telemetry.csv"
    sleep 5
  done
}

release_lease() {
  [[ -z "$LEASE_PID" ]] || kill "$LEASE_PID" >/dev/null 2>&1 || true
  [[ -z "$LEASE_PID" ]] || wait "$LEASE_PID" 2>/dev/null || true
  LEASE_PID=""
}

restore() {
  if codellama_is_running_on_gpu0; then
    RESERVATION=codellama_already_running_on_gpu0
    return 0
  fi
  RESERVATION=restoring_codellama_to_gpu0
  STAGE=resource_restoration
  write_status restoring_resource "Train-only P1 smoke ended; restoring CodeLlama on GPU0."
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
  release_lease
  [[ -z "$TELEMETRY_PID" ]] || kill "$TELEMETRY_PID" >/dev/null 2>&1 || true
  restore || restore_rc=$?
  STAGE=finished
  if (( scientific_rc == 0 && restore_rc == 0 )); then
    write_status succeeded "P1 train-only smoke completed for both domains; CodeLlama restored."
  elif (( scientific_rc == 0 )); then
    write_status failed_to_restore_resource "Scientific smoke completed but CodeLlama restoration failed."
  else
    write_status failed "Scientific exit=$scientific_rc; no automatic retry; CodeLlama restoration requested."
  fi
  (( scientific_rc != 0 )) && exit "$scientific_rc"
  exit "$restore_rc"
}

start_domain_lease() {
  local dataset=$1 expected_peak=$2
  release_lease
  "$PYTHON" "$LEASE_HELPER" --gpu "$GPU" --total-lease-mib "$TOTAL_LEASE_MIB" \
    --expected-workload-peak-mib "$expected_peak" \
    --status-path "$OUTPUT/gpu_lease_${dataset}.json" &
  LEASE_PID=$!
  for _ in $(seq 1 30); do
    [[ -s "$OUTPUT/gpu_lease_${dataset}.json" ]] && break
    sleep 1
  done
  [[ -s "$OUTPUT/gpu_lease_${dataset}.json" ]] || { write_status blocked "Domain sidecar did not become ready."; exit 6; }
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
  for required in "$CONFIG" "$WORKLOAD" "$TEST" "$LEASE_HELPER"; do
    [[ -s "$required" ]] || { write_status blocked "Required input missing: $required"; exit 2; }
  done
  config_is_enabled || { write_status blocked "Train smoke config is not frozen and enabled."; exit 3; }
  code_lock_matches || { write_status blocked "Train smoke code SHA lock mismatch."; exit 7; }
  env HF_HOME="$WORKLOAD_HF_HOME" HF_HUB_CACHE="$WORKLOAD_HF_HOME" \
    TRANSFORMERS_CACHE="$WORKLOAD_HF_HOME" "$PYTHON" -c \
    'from transformers import AutoTokenizer; AutoTokenizer.from_pretrained("t5-small", local_files_only=True)' \
    || { write_status blocked "Offline t5-small preflight failed before CodeLlama release."; exit 8; }
  codellama_is_running_on_gpu0 || { write_status blocked "CodeLlama must be running on GPU0 before start."; exit 4; }
  STAGE=resource_release
  write_status releasing_resource "Stopping CodeLlama before P1 train-only smoke."
  reserver stop
  RESERVATION=released_for_experiment
  local free_mib=""
  for _ in $(seq 1 60); do
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GPU" 2>/dev/null | tr -d ' ' || true)
    [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= TOTAL_LEASE_MIB )) && break
    sleep 5
  done
  [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= TOTAL_LEASE_MIB )) || { write_status blocked "GPU0 free memory below 30720 MiB."; exit 5; }
  telemetry & TELEMETRY_PID=$!
  for dataset in Toys Beauty; do
    CURRENT_DATASET=$dataset
    printf '%s\n' "$dataset" > "$OUTPUT/current_dataset.txt"
    STAGE=domain_lease
    if [[ "$dataset" == Toys ]]; then start_domain_lease "$dataset" 4608; else start_domain_lease "$dataset" 1792; fi
    STAGE=scientific_workload
    timeout --signal=TERM 7200 env CUDA_VISIBLE_DEVICES="$GPU" HF_HOME="$WORKLOAD_HF_HOME" \
      HF_HUB_CACHE="$WORKLOAD_HF_HOME" TRANSFORMERS_CACHE="$WORKLOAD_HF_HOME" \
      "$PYTHON" "$WORKLOAD" --config "$CONFIG" --output-root "$OUTPUT" --dataset "$dataset" &
    WORKLOAD_PID=$!
    write_status running "P1 train-only smoke running with a domain-specific 30 GiB lease."
    wait "$WORKLOAD_PID"
    WORKLOAD_PID=0
    release_lease
  done
}

case "${1:-status}" in
  start)
    mkdir -p "$OUTPUT"
    config_is_enabled || { STAGE=config_not_enabled; write_status blocked "Start refused: config is not enabled."; exit 3; }
    tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists: $SESSION" >&2; exit 1; }
    STARTED_AT=$(date -Is)
    printf -v launch_cmd 'bash %q worker %q >> %q 2>&1' "$0" "$STARTED_AT" "$LOG"
    tmux new-session -d -s "$SESSION" "$launch_cmd"
    STAGE=starting
    write_status starting "Persistent P1 train-only smoke session started for GPU0."
    echo "started $SESSION"
    ;;
  worker) worker "${2:?missing start timestamp}" ;;
  status)
    tmux has-session -t "$SESSION" 2>/dev/null && echo "tmux session: running ($SESSION)" || echo "tmux session: not running ($SESSION)"
    [[ -f "$STATUS" ]] && sed -n '1,100p' "$STATUS" || echo '{"status":"not_started"}'
    [[ -f "$LOG" ]] && tail -n 50 "$LOG" || true
    ;;
  *) echo "usage: $0 {start|status|worker}" >&2; exit 2 ;;
esac
