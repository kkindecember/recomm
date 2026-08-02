#!/usr/bin/env bash
# One-shot, validation-only recovery for the interrupted GACR-v6 run.
set -euo pipefail

ROOT=/home/jiangtangyunzhi/projects/recomm
CONFIG="$ROOT/artifacts/phase6/configs/gacr_v6_validation_recovery.json"
OUTPUT="$ROOT/artifacts/phase6/gacr_v6_validation_recovery"
LOG="$OUTPUT/run.log"
STATUS="$OUTPUT/status.json"
SESSION=gram_phase6_gacr_v6_validation_recovery
GPU=0
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python3.9
WORKLOAD="$ROOT/experiment/phase6/gacr_v6_recover_validation.py"
LEASE_HELPER="$ROOT/experiment/gpu_memory_lease.py"
# Measured during this recovery: validation candidate construction reached
# 22,820 MiB.  23 GiB leaves a 7 GiB sidecar reservation, for a 30 GiB total
# lease shared by workload and sidecar.
EXPECTED_WORKLOAD_PEAK_MIB=23552
TOTAL_LEASE_MIB=30720
RESERVER=/home/jiangtangyunzhi/projects/UnitTest/tools/run_codellama.sh
CODELLAMA_HF_HOME=/home/jiangtangyunzhi/hf_cache
TELEMETRY_PID=""
WORKLOAD_PID=0
LEASE_PID=""
STARTED_AT=""
STAGE=not_started
RESERVATION=unchanged

# Match the frozen full-fit runner: model loading is deliberately offline and
# must use the repository-local cache that supplied the original run.
export HF_HOME="$ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$ROOT/.cache/huggingface"

reserver() {
  env SESSION=codellama HF_HOME="$CODELLAMA_HF_HOME" \
    HF_HUB_CACHE="$CODELLAMA_HF_HOME/hub" TRANSFORMERS_CACHE="$CODELLAMA_HF_HOME/hub" \
    "$RESERVER" "$@"
}

write_status() {
  local state=$1 reason=$2 tmp="${STATUS}.tmp.$$"
  printf '{"experiment_id":"GRAM_PHASE6_GACR_V6_VALIDATION_ONLY_RECOVERY_V1","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%s,"workload_pid":%s,"tmux_session":"%s","physical_gpu":0,"log_path":"%s","result_path":"%s","resource_reservation":"%s","test_read":false,"sports_read":false}\n' \
    "$state" "$STAGE" "$reason" "$STARTED_AT" "$(date -Is)" "$$" "$WORKLOAD_PID" "$SESSION" \
    "${LOG#$ROOT/}" "${OUTPUT#$ROOT/}/summary.json" "$RESERVATION" > "$tmp"
  mv "$tmp" "$STATUS"
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
  RESERVATION=restoring_to_gpu0; STAGE=resource_restoration
  write_status restoring_resource "Recovery ended; restoring CodeLlama on physical GPU0."
  reserver start "$GPU" || true
  RESERVATION=restore_requested_on_gpu0
}

finish() {
  local rc=$?
  trap - EXIT INT TERM HUP
  [[ -z "$LEASE_PID" ]] || kill "$LEASE_PID" >/dev/null 2>&1 || true
  [[ -z "$LEASE_PID" ]] || wait "$LEASE_PID" 2>/dev/null || true
  [[ -z "$TELEMETRY_PID" ]] || kill "$TELEMETRY_PID" >/dev/null 2>&1 || true
  restore
  STAGE=finished
  if (( rc == 0 )); then
    write_status succeeded "GACR-v6 validation-only recovery completed; results await researcher-requested analysis."
  else
    write_status failed "Recovery exit=$rc; no automatic retry; CodeLlama restoration requested on GPU0."
  fi
  exit "$rc"
}

worker() {
  STARTED_AT=${1:?missing start timestamp}
  trap finish EXIT; trap 'exit 130' INT; trap 'exit 143' TERM; trap 'exit 129' HUP
  cd "$ROOT"; mkdir -p "$OUTPUT"
  STAGE=preflight
  for required in "$CONFIG" "$WORKLOAD" "$LEASE_HELPER" \
    "$ROOT/artifacts/phase6/gacr_v6/Toys/residual_seed2023.pt" \
    "$ROOT/artifacts/phase6/gacr_v6/Toys/residual_seed2024.pt" \
    "$ROOT/artifacts/phase6/gacr_v6/Toys/residual_seed2025.pt" \
    "$ROOT/artifacts/phase6/gacr_v6/Beauty/residual_seed2023.pt" \
    "$ROOT/artifacts/phase6/gacr_v6/Beauty/residual_seed2024.pt" \
    "$ROOT/artifacts/phase6/gacr_v6/Beauty/residual_seed2025.pt"; do
    [[ -s "$required" ]] || { write_status blocked "Required recovery input missing: $required"; exit 2; }
  done
  STAGE=resource_release; write_status releasing_resource "Stopping CodeLlama before validation-only recovery on GPU0."
  reserver stop; RESERVATION=released_for_recovery
  STAGE=gpu_memory_gate
  local free_mib
  free_mib=""
  for _ in $(seq 1 60); do
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GPU" 2>/dev/null | tr -d ' ' || true)
    [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= 30720 )) && break
    sleep 60
  done
  [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= 30720 )) || { write_status blocked "GPU0 free memory below 30720 MiB."; exit 3; }
  STAGE=validation_only_recovery; telemetry & TELEMETRY_PID=$!
  "$PYTHON" "$LEASE_HELPER" --gpu "$GPU" --total-lease-mib "$TOTAL_LEASE_MIB" \
    --expected-workload-peak-mib "$EXPECTED_WORKLOAD_PEAK_MIB" \
    --status-path "$OUTPUT/gpu_lease_status.json" &
  LEASE_PID=$!
  for _ in $(seq 1 30); do
    [[ -f "$OUTPUT/gpu_lease_status.json" ]] && break
    sleep 1
  done
  [[ -f "$OUTPUT/gpu_lease_status.json" ]] || { write_status blocked "GPU lease sidecar did not become ready."; exit 4; }
  timeout --signal=TERM 28800 env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$WORKLOAD" \
    --recovery-config "$CONFIG" --output-root "$OUTPUT" &
  WORKLOAD_PID=$!; write_status running "GACR-v6 validation-only recovery running on GPU0."
  wait "$WORKLOAD_PID"
}

case "${1:-status}" in
  start)
    mkdir -p "$OUTPUT"
    tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists: $SESSION" >&2; exit 1; }
    STARTED_AT=$(date -Is)
    printf -v cmd 'bash %q worker %q >> %q 2>&1' "$0" "$STARTED_AT" "$LOG"
    tmux new-session -d -s "$SESSION" "$cmd"
    STAGE=starting; RESERVATION=scheduled_for_release
    write_status starting "Validation-only recovery session started for GPU0."
    echo "started $SESSION"
    ;;
  worker) worker "${2:?missing start timestamp}" ;;
  status)
    tmux has-session -t "$SESSION" 2>/dev/null && echo "tmux session: running ($SESSION)" || echo "tmux session: not running ($SESSION)"
    [[ -f "$STATUS" ]] && sed -n '1,100p' "$STATUS" || echo '{"status":"not_started"}'
    [[ -f "$LOG" ]] && tail -n 40 "$LOG" || true
    ;;
  *) echo "usage: $0 {start|status|worker}" >&2; exit 2 ;;
esac
