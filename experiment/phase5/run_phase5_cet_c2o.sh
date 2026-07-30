#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jiangtangyunzhi/projects/recomm
CONFIG="$ROOT/artifacts/phase5/configs/cet_c2o_preregistered.json"
OUTPUT="$ROOT/artifacts/phase5/cet_c2o"
LOG="$OUTPUT/run.log"
STATUS="$OUTPUT/status.json"
TELEMETRY="$OUTPUT/gpu_telemetry.csv"
SESSION=gram_phase5_cet_c2o
GPU=3
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python3.9
RESERVER=/home/jiangtangyunzhi/projects/UnitTest/tools/run_codellama.sh
TELEMETRY_PID=""

export HF_HOME="$ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$ROOT/.cache/huggingface"

write_status() {
  local state=$1
  local reason=$2
  local tmp="${STATUS}.tmp.$$"
  printf '{"experiment_id":"GRAM_PHASE5_CET_C2O","status":"%s","reason":"%s","updated_at":"%s","validation_target_read":false,"test_read":false,"sports_read":false}\n' \
    "$state" "$reason" "$(date -Is)" > "$tmp"
  mv "$tmp" "$STATUS"
}

run_stage() {
  local stage=$1
  local dataset=${2:-}
  local control=${3:-}
  local args=(--config "$CONFIG" --stage "$stage" --output-root "$OUTPUT")
  if [[ -n "$dataset" ]]; then args+=(--dataset "$dataset"); fi
  if [[ -n "$control" ]]; then args+=(--control "$control"); fi
  timeout --signal=TERM 14400 "$PYTHON" \
    "$ROOT/experiment/phase5/cet_c2_optimization_audit.py" "${args[@]}"
}

telemetry_worker() {
  printf 'timestamp,index,memory_used_mib,memory_free_mib,utilization_gpu_percent\n' > "$TELEMETRY"
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits --id="$GPU" >> "$TELEMETRY"
    sleep 5
  done
}

worker() {
  local rc=0
  trap 'rc=$?; if [[ -n "$TELEMETRY_PID" ]]; then kill "$TELEMETRY_PID" >/dev/null 2>&1 || true; fi; "$RESERVER" start "$GPU" >/dev/null 2>&1 || true; if (( rc == 0 )); then write_status succeeded "CET C2-O completed."; else write_status failed "CET C2-O exit=$rc; no automatic retry; resource restored."; fi' EXIT
  export CUDA_VISIBLE_DEVICES=$GPU
  cd "$ROOT"
  write_status running "CET C2-O frozen-checkpoint optimization audit running."
  "$RESERVER" stop
  local free_mib=""
  for _ in $(seq 1 24); do
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GPU" | tr -d ' ')
    [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= 30720 )) && break
    sleep 5
  done
  if [[ ! "$free_mib" =~ ^[0-9]+$ ]] || (( free_mib < 30720 )); then
    echo "insufficient GPU memory after resource release: ${free_mib:-unknown} MiB"
    exit 3
  fi
  telemetry_worker &
  TELEMETRY_PID=$!
  for dataset in Toys Beauty; do
    for control in C0 C1 C2; do
      run_stage audit "$dataset" "$control"
    done
  done
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
      run_stage make-splits
    fi
    free_kib=$(df --output=avail "$ROOT" | tail -n 1 | tr -d ' ')
    if (( free_kib < 3145728 )); then
      echo "insufficient disk: $free_kib KiB"
      exit 1
    fi
    tmux new-session -d -s "$SESSION" "bash '$0' worker >> '$LOG' 2>&1"
    write_status starting "Persistent CET C2-O session started."
    echo "started $SESSION"
    ;;
  worker)
    worker
    ;;
  status)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "running"
    else
      echo "not-running"
    fi
    if [[ -f "$STATUS" ]]; then sed -n '1,80p' "$STATUS"; fi
    if [[ -f "$LOG" ]]; then tail -n 40 "$LOG"; fi
    ;;
  *)
    echo "usage: $0 {start|status|worker}" >&2
    exit 2
    ;;
esac
