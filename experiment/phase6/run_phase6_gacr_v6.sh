#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jiangtangyunzhi/projects/recomm
CONFIG="$ROOT/artifacts/phase6/configs/gacr_v6_preregistered.json"
OUTPUT="$ROOT/artifacts/phase6/gacr_v6"
LOG="$OUTPUT/run.log"
STATUS="$OUTPUT/status.json"
TELEMETRY="$OUTPUT/gpu_telemetry.csv"
SESSION=gram_phase6_gacr_v6
GPU=0
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python3.9
WORKLOAD="$ROOT/experiment/phase6/gacr_v6.py"
RESERVER=/home/jiangtangyunzhi/projects/UnitTest/tools/run_codellama.sh
RESERVER_SESSION=codellama
CODELLAMA_HF_HOME=/home/jiangtangyunzhi/hf_cache
MIN_FREE_MIB=30720
GPU_GATE_POLLS=720
GPU_GATE_POLL_SECONDS=60
HARD_TIMEOUT_SECONDS=129600
RESTORE_ATTEMPTS=3
RESTORE_POLLS=180
RESTORE_POLL_SECONDS=5
TELEMETRY_PID=""
WORKLOAD_PID=0
STARTED_AT=""
CURRENT_STAGE=not_started
RESERVATION_STATE=unchanged

export HF_HOME="$ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$ROOT/.cache/huggingface"

reserver() {
  env SESSION="$RESERVER_SESSION" \
    HF_HOME="$CODELLAMA_HF_HOME" \
    HF_HUB_CACHE="$CODELLAMA_HF_HOME/hub" \
    TRANSFORMERS_CACHE="$CODELLAMA_HF_HOME/hub" \
    "$RESERVER" "$@"
}

write_status() {
  local state=$1 reason=$2 tmp="${STATUS}.tmp.$$"
  printf '{"experiment_id":"GRAM_PHASE6_GACR_V6_FULL_FIT","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%s,"workload_pid":%s,"tmux_session":"%s","physical_gpu":0,"log_path":"%s","result_path":"%s","resource_reservation":"%s","codellama_restore_gpu":0,"test_read":false,"sports_read":false}\n' \
    "$state" "$CURRENT_STAGE" "$reason" "$STARTED_AT" "$(date -Is)" \
    "$$" "$WORKLOAD_PID" "$SESSION" "${LOG#$ROOT/}" \
    "${OUTPUT#$ROOT/}/summary.json" "$RESERVATION_STATE" > "$tmp"
  mv "$tmp" "$STATUS"
}

telemetry_worker() {
  printf 'timestamp,index,memory_used_mib,memory_free_mib,utilization_gpu_percent\n' > "$TELEMETRY"
  while true; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits --id="$GPU" >> "$TELEMETRY" || true
    sleep 5
  done
}

reservation_status_is_running() {
  local value=${1:-}
  [[ "$value" == *"tmux session: running (codellama)"* ]] \
    && [[ "$value" == *"state=running"* ]] \
    && [[ "$value" == *"gpu=$GPU"* ]]
}

restore_resource() {
  RESERVATION_STATE=restoring_to_gpu0
  CURRENT_STAGE=resource_restoration
  write_status restoring_resource "Experiment ended; restoring CodeLlama on physical GPU0."
  local attempt poll value
  for attempt in $(seq 1 "$RESTORE_ATTEMPTS"); do
    value=$(reserver status 2>&1 || true)
    if reservation_status_is_running "$value"; then
      RESERVATION_STATE=restored_on_gpu0
      return 0
    fi
    reserver start "$GPU" || true
    for poll in $(seq 1 "$RESTORE_POLLS"); do
      value=$(reserver status 2>&1 || true)
      if reservation_status_is_running "$value"; then
        RESERVATION_STATE=restored_on_gpu0
        return 0
      fi
      sleep "$RESTORE_POLL_SECONDS"
    done
  done
  RESERVATION_STATE=restore_failed_on_gpu0
  return 1
}

finish() {
  local experiment_rc=$? restore_rc=0
  trap - EXIT INT TERM HUP
  if [[ -n "$TELEMETRY_PID" ]]; then
    kill "$TELEMETRY_PID" >/dev/null 2>&1 || true
  fi
  restore_resource || restore_rc=$?
  CURRENT_STAGE=finished
  if (( experiment_rc == 0 && restore_rc == 0 )); then
    write_status succeeded "GACR-v6 full-fit completed; results await researcher-requested analysis."
  elif (( restore_rc != 0 )); then
    write_status failed_to_restore_resource "Experiment exit=$experiment_rc; CodeLlama GPU0 restoration failed."
  else
    write_status failed "Experiment exit=$experiment_rc; no automatic retry; CodeLlama restored on GPU0."
  fi
  if (( experiment_rc != 0 )); then exit "$experiment_rc"; fi
  exit "$restore_rc"
}

worker() {
  STARTED_AT=${1:?missing start timestamp}
  trap finish EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  trap 'exit 129' HUP
  cd "$ROOT"
  CURRENT_STAGE=preflight
  local required
  for required in "$CONFIG" "$WORKLOAD" \
    "$ROOT/artifacts/phase4/gcdh_p0/Toys/C1/model.pt" \
    "$ROOT/artifacts/phase4/gcdh_p0/Beauty/C1/model.pt" \
    "$ROOT/artifacts/phase6/gacr_v2/Toys/residual_seed2023.pt" \
    "$ROOT/artifacts/phase6/gacr_v2/Toys/residual_seed2024.pt" \
    "$ROOT/artifacts/phase6/gacr_v2/Toys/residual_seed2025.pt" \
    "$ROOT/artifacts/phase6/gacr_v2/Beauty/residual_seed2023.pt" \
    "$ROOT/artifacts/phase6/gacr_v2/Beauty/residual_seed2024.pt" \
    "$ROOT/artifacts/phase6/gacr_v2/Beauty/residual_seed2025.pt"; do
    if [[ ! -s "$required" ]]; then
      write_status blocked "Required locked material missing: $required"
      exit 2
    fi
  done
  CURRENT_STAGE=resource_release
  write_status releasing_resource "Stopping CodeLlama before GACR-v6 on physical GPU0."
  reserver stop
  RESERVATION_STATE=released_for_experiment
  CURRENT_STAGE=gpu_memory_gate
  write_status waiting_for_gpu "Waiting for physical GPU0 free memory >= ${MIN_FREE_MIB} MiB."
  local free_mib=""
  for _ in $(seq 1 "$GPU_GATE_POLLS"); do
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
      --id="$GPU" 2>/dev/null | tr -d ' ' || true)
    if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )); then break; fi
    sleep "$GPU_GATE_POLL_SECONDS"
  done
  if [[ ! "$free_mib" =~ ^[0-9]+$ ]] || (( free_mib < MIN_FREE_MIB )); then
    write_status blocked "GPU0 free memory ${free_mib:-unknown} MiB below ${MIN_FREE_MIB} MiB after gate timeout."
    exit 3
  fi
  CURRENT_STAGE=gacr_v6_full_fit
  telemetry_worker &
  TELEMETRY_PID=$!
  timeout --signal=TERM "$HARD_TIMEOUT_SECONDS" env CUDA_VISIBLE_DEVICES="$GPU" \
    "$PYTHON" "$WORKLOAD" --config "$CONFIG" --output-root "$OUTPUT" &
  WORKLOAD_PID=$!
  write_status running "GACR-v6 full-fit running on physical GPU0."
  wait "$WORKLOAD_PID"
}

main() {
  case "${1:-status}" in
    start)
      mkdir -p "$OUTPUT"
      if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "session already exists: $SESSION" >&2
        exit 1
      fi
      local free_kib
      free_kib=$(df --output=avail "$ROOT" | tail -n 1 | tr -d ' ')
      if (( free_kib < 5242880 )); then
        echo "insufficient disk: $free_kib KiB" >&2
        exit 1
      fi
      STARTED_AT=$(date -Is)
      local launch_cmd
      printf -v launch_cmd 'bash %q worker %q >> %q 2>&1' "$0" "$STARTED_AT" "$LOG"
      tmux new-session -d -s "$SESSION" "$launch_cmd"
      RESERVATION_STATE=scheduled_for_release
      CURRENT_STAGE=starting
      write_status starting "Persistent GACR-v6 full-fit session started for physical GPU0."
      echo "started $SESSION"
      ;;
    worker) worker "${2:?missing start timestamp}" ;;
    status)
      if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "tmux session: running ($SESSION)"
      else
        echo "tmux session: not running ($SESSION)"
      fi
      if [[ -f "$STATUS" ]]; then sed -n '1,100p' "$STATUS"; else echo '{"status":"not_started"}'; fi
      if [[ -f "$LOG" ]]; then tail -n 40 "$LOG"; fi
      ;;
    *) echo "usage: bash experiment/phase6/run_phase6_gacr_v6.sh {start|status|worker}" >&2; exit 2 ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi
