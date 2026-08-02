#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="$ROOT/artifacts/phase5/configs/cet_rank_r1_preregistered.json"
OUTPUT="$ROOT/artifacts/phase5/cet_rank_r1"
LOG="$OUTPUT/run.log"
STATUS="$OUTPUT/status.json"
TELEMETRY="$OUTPUT/gpu_telemetry.csv"
SESSION=gram_phase5_cet_rank_r1
GPU=6
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python3.9
RESERVER="$ROOT/tools/run_codellama.sh"
TELEMETRY_PID=""
STARTED_AT=""
RESERVATION_STATE=unchanged

export HF_HOME="$ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$ROOT/.cache/huggingface"

write_status() {
  local state=$1
  local reason=$2
  local tmp="${STATUS}.tmp.$$"
  printf '{"experiment_id":"GRAM_PHASE5_CET_RANK_R1","status":"%s","reason":"%s","started_at":"%s","updated_at":"%s","resource_reservation":"%s","validation_target_read":false,"test_read":false,"sports_read":false}\n' \
    "$state" "$reason" "$STARTED_AT" "$(date -Is)" "$RESERVATION_STATE" > "$tmp"
  mv "$tmp" "$STATUS"
}

run_stage() {
  local stage=$1
  local dataset=${2:-}
  local args=(--config "$CONFIG" --stage "$stage" --output-root "$OUTPUT")
  if [[ -n "$dataset" ]]; then args+=(--dataset "$dataset"); fi
  timeout --signal=TERM 7200 env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
    "$ROOT/experiment/phase5/cet_rank_r1.py" "${args[@]}"
}

telemetry_worker() {
  printf 'timestamp,index,memory_used_mib,memory_free_mib,utilization_gpu_percent\n' > "$TELEMETRY"
  while true; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits --id="$GPU" >> "$TELEMETRY"
    sleep 5
  done
}

restore_resource() {
  RESERVATION_STATE=restoring
  write_status restoring_resource "Rank-R1 ended; restoring CodeLlama on GPU6."
  for _ in 1 2 3; do
    if "$RESERVER" start "$GPU"; then RESERVATION_STATE=restored; return 0; fi
    sleep 2
  done
  RESERVATION_STATE=restore_failed
  return 1
}

finish() {
  local experiment_rc=$?
  local restore_rc=0
  trap - EXIT INT TERM HUP
  if [[ -n "$TELEMETRY_PID" ]]; then kill "$TELEMETRY_PID" >/dev/null 2>&1 || true; fi
  restore_resource || restore_rc=$?
  if (( experiment_rc == 0 && restore_rc == 0 )); then
    write_status succeeded "CET Rank-R1 completed; inspect summary.json."
  elif (( restore_rc != 0 )); then
    write_status failed_to_restore_resource "Rank-R1 exit=$experiment_rc; restoration failed."
  else
    write_status failed "Rank-R1 exit=$experiment_rc; no automatic retry; resource restored."
  fi
  exit "$experiment_rc"
}

worker() {
  STARTED_AT=${1:?missing start timestamp}
  trap finish EXIT INT TERM HUP
  cd "$ROOT"
  write_status releasing_resource "Stopping CodeLlama before CET Rank-R1."
  "$RESERVER" stop
  RESERVATION_STATE=released_for_experiment
  local free_mib=""
  for _ in $(seq 1 24); do
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GPU" | tr -d ' ')
    [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= 30720 )) && break
    sleep 5
  done
  if [[ ! "$free_mib" =~ ^[0-9]+$ ]] || (( free_mib < 30720 )); then
    write_status blocked "GPU6 free memory ${free_mib:-unknown} MiB below 30720 MiB."
    exit 3
  fi
  write_status running "CET Rank-R1 correctness smoke running."
  telemetry_worker &
  TELEMETRY_PID=$!
  for dataset in Toys Beauty; do run_stage run "$dataset"; done
  run_stage analyze
}

case "${1:-status}" in
  start)
    mkdir -p "$OUTPUT"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "session already exists: $SESSION"
      exit 1
    fi
    if [[ ! -f "$OUTPUT/splits/frozen_manifest.json" ]]; then
      echo "missing frozen split manifest; preflight is incomplete" >&2
      exit 1
    fi
    free_kib=$(df --output=avail "$ROOT" | tail -n 1 | tr -d ' ')
    if (( free_kib < 3145728 )); then
      echo "insufficient disk: $free_kib KiB"
      exit 1
    fi
    STARTED_AT=$(date -Is)
    tmux new-session -d -s "$SESSION" "bash '$0' worker '$STARTED_AT' >> '$LOG' 2>&1"
    RESERVATION_STATE=scheduled_for_release
    write_status starting "Persistent CET Rank-R1 session started."
    echo "started $SESSION"
    ;;
  worker) worker "${2:?missing start timestamp}" ;;
  status)
    if tmux has-session -t "$SESSION" 2>/dev/null; then echo running; else echo not-running; fi
    if [[ -f "$STATUS" ]]; then sed -n '1,80p' "$STATUS"; fi
    if [[ -f "$LOG" ]]; then tail -n 40 "$LOG"; fi
    ;;
  *) echo "usage: $0 {start|status|worker}" >&2; exit 2 ;;
esac
