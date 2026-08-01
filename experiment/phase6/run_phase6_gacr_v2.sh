#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jiangtangyunzhi/projects/recomm
CONFIG=${CONFIG:-$ROOT/artifacts/phase6/configs/gacr_v2_preregistered.json}
OUTPUT=${OUTPUT:-$ROOT/artifacts/phase6/gacr_v2}
LOG="$OUTPUT/run.log"
STATUS="$OUTPUT/status.json"
TELEMETRY="$OUTPUT/gpu_telemetry.csv"
SESSION=${SESSION:-gram_phase6_gacr_v2}
GPU=6
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python3.9
EXPERIMENT_ID=${EXPERIMENT_ID:-GRAM_PHASE6_GACR_V2}
RUN_LABEL=${RUN_LABEL:-GACR-v2}
WORKLOAD_SCRIPT=${WORKLOAD_SCRIPT:-$ROOT/experiment/phase6/gacr_v2.py}
WORKLOAD_STAGE=${WORKLOAD_STAGE:-gacr_v2_growth_pilot}
RESERVER=${RESERVER:-/home/jiangtangyunzhi/projects/UnitTest/tools/run_codellama.sh}
RESERVER_SESSION=${RESERVER_SESSION:-codellama}
RESTORE_ATTEMPTS=${RESTORE_ATTEMPTS:-3}
RESTORE_POLLS=${RESTORE_POLLS:-180}
RESTORE_POLL_SECONDS=${RESTORE_POLL_SECONDS:-5}
TELEMETRY_PID=""
WORKLOAD_PID=0
STARTED_AT=""
RESERVATION_STATE=unchanged
CURRENT_STAGE=not_started
GCDH_CONFIG="$ROOT/artifacts/phase4/configs/gcdh_p0_preregistered.json"
CHECKPOINT_ROOT="$ROOT/artifacts/phase4/gcdh_p0"
EXPECTED_TOYS_SHA=1307ab9d3aa5e56af97fad7276d63cb276260efd3d314b199e350a611c798af6
EXPECTED_BEAUTY_SHA=5842f45998325cfee47427fd5d323ffdde23fda373016ca86e5164d3d908d2f2

export HF_HOME="$ROOT/.cache/huggingface"
export TRANSFORMERS_CACHE="$ROOT/.cache/huggingface"

write_status() {
  local state=$1
  local reason=$2
  local tmp="${STATUS}.tmp.$$"
  printf '{"experiment_id":"%s","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%s,"workload_pid":%s,"tmux_session":"%s","physical_gpu":6,"log_path":"%s","result_path":"%s","resource_reservation":"%s","test_read":false,"sports_read":false}\n' \
    "$EXPERIMENT_ID" \
    "$state" "$CURRENT_STAGE" "$reason" "$STARTED_AT" "$(date -Is)" "$$" "$WORKLOAD_PID" "$SESSION" \
    "${LOG#$ROOT/}" "${OUTPUT#$ROOT/}/summary.json" "$RESERVATION_STATE" > "$tmp"
  mv "$tmp" "$STATUS"
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
  CURRENT_STAGE=resource_restoration
  write_status restoring_resource "Experiment ended; restoring CodeLlama on physical GPU6."
  local attempt poll reservation_status
  for attempt in $(seq 1 "$RESTORE_ATTEMPTS"); do
    reservation_status=$(reserver status 2>&1 || true)
    if reservation_status_is_running "$reservation_status"; then
      RESERVATION_STATE=restored
      return 0
    fi
    reserver start "$GPU" || true
    for poll in $(seq 1 "$RESTORE_POLLS"); do
      reservation_status=$(reserver status 2>&1 || true)
      if reservation_status_is_running "$reservation_status"; then
        RESERVATION_STATE=restored
        return 0
      fi
      sleep "$RESTORE_POLL_SECONDS"
    done
    sleep 2
  done
  RESERVATION_STATE=restore_failed
  return 1
}

reserver() {
  env SESSION="$RESERVER_SESSION" "$RESERVER" "$@"
}

reservation_status_is_running() {
  local reservation_status=${1:-}
  [[ "$reservation_status" == *"tmux session: running (codellama)"* ]] \
    && [[ "$reservation_status" == *"state=running"* ]] \
    && [[ "$reservation_status" == *"gpu=$GPU"* ]]
}

finish() {
  local experiment_rc=$?
  local restore_rc=0
  trap - EXIT INT TERM HUP
  if [[ -n "$TELEMETRY_PID" ]]; then
    kill "$TELEMETRY_PID" >/dev/null 2>&1 || true
  fi
  restore_resource || restore_rc=$?
  CURRENT_STAGE=finished
  if (( experiment_rc == 0 && restore_rc == 0 )); then
    write_status succeeded "$RUN_LABEL completed; results await researcher-requested analysis."
  elif (( restore_rc != 0 )); then
    write_status failed_to_restore_resource "Experiment exit=$experiment_rc; CodeLlama restoration failed."
  else
    write_status failed "Experiment exit=$experiment_rc; no automatic retry; CodeLlama restored."
  fi
  if (( experiment_rc != 0 )); then
    exit "$experiment_rc"
  fi
  exit "$restore_rc"
}

expected_sha() {
  case "$1" in
    Toys) printf '%s\n' "$EXPECTED_TOYS_SHA" ;;
    Beauty) printf '%s\n' "$EXPECTED_BEAUTY_SHA" ;;
    *) return 2 ;;
  esac
}

verify_checkpoint() {
  local dataset=$1
  local checkpoint="$CHECKPOINT_ROOT/$dataset/C1/model.pt"
  local expected actual
  expected=$(expected_sha "$dataset")
  [[ -s "$checkpoint" ]] || return 1
  actual=$(sha256sum "$checkpoint")
  actual=${actual%% *}
  [[ "$actual" == "$expected" ]]
}

run_reconstruction() {
  local dataset=$1
  CURRENT_STAGE="reconstruct_gcdh_p0_c1_${dataset}"
  timeout --signal=TERM 43200 env CUDA_VISIBLE_DEVICES="$GPU" \
    "$PYTHON" "$ROOT/experiment/phase4/gcdh_p0.py" \
    --config "$GCDH_CONFIG" --stage train --dataset "$dataset" \
    --control C1 --output-root "$CHECKPOINT_ROOT" &
  WORKLOAD_PID=$!
  write_status running "Reconstructing locked GCDH-P0 C1 checkpoint for $dataset."
  wait "$WORKLOAD_PID"
  WORKLOAD_PID=0
  if ! verify_checkpoint "$dataset"; then
    write_status blocked "Reconstructed $dataset C1 checkpoint does not match historical SHA256."
    return 5
  fi
  write_status running "Reconstructed $dataset C1 checkpoint matches historical SHA256."
}

worker() {
  STARTED_AT=${1:?missing start timestamp}
  trap finish EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  trap 'exit 129' HUP
  cd "$ROOT"
  local required
  for required in "$CONFIG" "$GCDH_CONFIG" \
    "$ROOT/GRAM/log/Toys/1_20260720_1830/id_0_rec_30/model_rec_phase_1_epoch_30.pt" \
    "$ROOT/GRAM/log/Beauty/4_20260718_2153/id_0_rec_30/model_rec_phase_1_epoch_25.pt" \
    "$ROOT/artifacts/phase4/gcdh_p0_splits/Toys/manifest.json" \
    "$ROOT/artifacts/phase4/gcdh_p0_splits/Beauty/manifest.json"; do
    if [[ ! -s "$required" ]]; then
      CURRENT_STAGE=preflight
      write_status blocked "Required locked material missing: $required"
      exit 2
    fi
  done
  for dataset in Toys Beauty; do
    checkpoint="$CHECKPOINT_ROOT/$dataset/C1/model.pt"
    if [[ -e "$checkpoint" ]] && ! verify_checkpoint "$dataset"; then
      CURRENT_STAGE=preflight
      write_status blocked "Existing $dataset C1 checkpoint has the wrong SHA256; refusing overwrite."
      exit 6
    fi
  done
  CURRENT_STAGE=resource_release
  write_status releasing_resource "Stopping CodeLlama before $RUN_LABEL."
  reserver stop
  RESERVATION_STATE=released_for_experiment
  CURRENT_STAGE=gpu_memory_gate
  write_status released_for_experiment "CodeLlama stopped; waiting for GPU6 memory gate."
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
  CURRENT_STAGE=$WORKLOAD_STAGE
  telemetry_worker &
  TELEMETRY_PID=$!
  for dataset in Toys Beauty; do
    if ! verify_checkpoint "$dataset"; then
      run_reconstruction "$dataset"
    fi
  done
  timeout --signal=TERM 21600 env CUDA_VISIBLE_DEVICES="$GPU" \
    "$PYTHON" "$WORKLOAD_SCRIPT" \
    --config "$CONFIG" --output-root "$OUTPUT" &
  WORKLOAD_PID=$!
  write_status running "$RUN_LABEL pilot running."
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
    if [[ ! -f "$CONFIG" ]]; then
      echo "missing preregistered config: $CONFIG" >&2
      exit 1
    fi
    free_kib=$(df --output=avail "$ROOT" | tail -n 1 | tr -d ' ')
    if (( free_kib < 3145728 )); then
      echo "insufficient disk: $free_kib KiB" >&2
      exit 1
    fi
    STARTED_AT=$(date -Is)
    printf -v launch_cmd \
      'env CONFIG=%q OUTPUT=%q SESSION=%q EXPERIMENT_ID=%q RUN_LABEL=%q WORKLOAD_SCRIPT=%q WORKLOAD_STAGE=%q RESERVER=%q RESERVER_SESSION=%q RESTORE_ATTEMPTS=%q RESTORE_POLLS=%q RESTORE_POLL_SECONDS=%q bash %q worker %q >> %q 2>&1' \
      "$CONFIG" "$OUTPUT" "$SESSION" "$EXPERIMENT_ID" "$RUN_LABEL" \
      "$WORKLOAD_SCRIPT" "$WORKLOAD_STAGE" "$RESERVER" "$RESERVER_SESSION" "$RESTORE_ATTEMPTS" \
      "$RESTORE_POLLS" "$RESTORE_POLL_SECONDS" "$0" "$STARTED_AT" "$LOG"
    tmux new-session -d -s "$SESSION" "$launch_cmd"
    RESERVATION_STATE=scheduled_for_release
    CURRENT_STAGE=starting
    write_status starting "Persistent $RUN_LABEL session started."
    echo "started $SESSION"
    ;;
  worker)
    worker "${2:?missing start timestamp}"
    ;;
  status)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "tmux session: running ($SESSION)"
    else
      echo "tmux session: not running ($SESSION)"
    fi
    if [[ -f "$STATUS" ]]; then sed -n '1,100p' "$STATUS"; else echo '{"status":"not_started"}'; fi
    if [[ -f "$LOG" ]]; then tail -n 40 "$LOG"; fi
    ;;
  *)
    echo "usage: bash experiment/phase6/run_phase6_gacr_v2.sh {start|status|worker}" >&2
    exit 2
    ;;
esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
