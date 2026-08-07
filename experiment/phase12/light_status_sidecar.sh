#!/usr/bin/env bash
# Sidecar that watches a "light-protocol" HI-GRAM tmux training run and writes
# a status.json + telemetry CSV that mirrors the full-protocol runner's layout.
#
# Why: the light-protocol path (see README/current_run memory) skips the 30 GiB
# lease + CodeLlama placement so we can co-tenant a shared GPU. It also means
# there is no runner supervising the workload, so status.json isn't created
# automatically. This sidecar recreates the observable status file without
# taking a lease.
#
# Usage:
#   bash light_status_sidecar.sh <sub> <gpu> <tmux_session> <output_dir>
# Example:
#   bash light_status_sidecar.sh toys_v1_light 0 \
#       gram_phase12_hi_gram_toys_v1_light \
#       /path/to/artifacts/phase12/hi_gram/toys_v1_light
#
# Sidecar itself should be launched in a dedicated tmux session so it survives
# terminal exit. It exits automatically once the training tmux session is gone.
set -uo pipefail

SUB=${1:?sub name required}
GPU=${2:?gpu index required}
SESSION=${3:?training tmux session required}
OUTPUT=${4:?output dir required}

STATUS="$OUTPUT/status.json"
TELEMETRY="$OUTPUT/gpu_telemetry.csv"
LOG="$OUTPUT/run.log"
mkdir -p "$OUTPUT"

STARTED_AT=$(date -Is)

# Initialize telemetry header if missing
if [[ ! -s "$TELEMETRY" ]]; then
  printf 'timestamp,index,memory_used_mib,memory_free_mib,utilization_gpu_percent\n' > "$TELEMETRY"
fi

json_escape() {
  # minimal escape for JSON string values we write (log path, reason, etc.)
  local s=${1//\\/\\\\}
  s=${s//\"/\\\"}
  printf '%s' "$s"
}

write_status() {
  local state=$1 stage=$2 reason=$3 workload_pid=$4 latest_epoch=$5 latest_loss=$6 latest_val=$7
  local now
  now=$(date -Is)
  local tmp="${STATUS}.tmp.$$"
  printf '{"experiment_id":"GRAM_PHASE12_HI_GRAM_%s_V1","sub":"%s","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%s,"workload_pid":%s,"tmux_session":"%s","physical_gpu":%d,"total_gpu_lease_mib":0,"resource_reservation":"none_light_mode","log_path":"%s","test_read":false,"sports_read":false,"yelp_read":false,"hi_gram_enabled":true,"latest_epoch":"%s","latest_train_loss":"%s","latest_validation":"%s"}\n' \
    "${SUB^^}" "$SUB" "$state" "$stage" "$(json_escape "$reason")" "$STARTED_AT" "$now" \
    "$$" "$workload_pid" "$SESSION" "$GPU" "$(json_escape "${LOG#/}")" \
    "$(json_escape "$latest_epoch")" "$(json_escape "$latest_loss")" "$(json_escape "$latest_val")" \
    > "$tmp"
  mv "$tmp" "$STATUS"
}

find_workload_pid() {
  # Find OUR python training process. Disambiguate by the --log_dir path
  # segment (unique per sub) so we don't pick up sibling experiments; and
  # require comm=python so we return the actual training PID rather than the
  # bash wrapper whose argv also contains the python command string.
  ps -eo pid,comm,args --no-headers 2>/dev/null | awk -v pat="main_generative_gram\\.py.*/${SUB}/gram_logs" \
    '$2 == "python" && $0 ~ pat {print $1; exit}'
}

parse_log() {
  # Emits: latest_epoch<TAB>latest_loss<TAB>latest_val
  local epoch loss val
  epoch=$(grep -E "Start training recommender for phase 1, epoch [0-9]+" "$LOG" 2>/dev/null \
    | tail -1 | grep -oE 'epoch [0-9]+' | awk '{print $2}')
  loss=$(grep -E "average training loss for rec phase 1 epoch [0-9]+ is" "$LOG" 2>/dev/null \
    | tail -1 | sed -E 's/.*epoch ([0-9]+) is (.*)/e\1=\2/')
  val=$(grep -E "validation (hit|ndcg)@10:" "$LOG" 2>/dev/null | tail -2 \
    | awk '{printf "%s%s ",$2,$3}' | sed 's/ $//')
  printf '%s\t%s\t%s' "${epoch:-0}" "${loss:-none}" "${val:-none}"
}

# Main loop
INTERVAL=30
while true; do
  # Training tmux session gone → decide status by exit condition and stop.
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    IFS=$'\t' read -r epoch loss val <<<"$(parse_log)"
    if [[ "$loss" == *"nan"* ]]; then
      write_status failed finished "training exited; last loss is NaN" 0 "$epoch" "$loss" "$val"
    elif grep -qE "Traceback|OutOfMemory|CUDA error" "$LOG" 2>/dev/null; then
      write_status failed finished "training exited with python traceback" 0 "$epoch" "$loss" "$val"
    else
      write_status succeeded finished "training tmux session ended cleanly" 0 "$epoch" "$loss" "$val"
    fi
    break
  fi

  workload_pid=$(find_workload_pid || true)
  workload_pid=${workload_pid:-0}

  IFS=$'\t' read -r epoch loss val <<<"$(parse_log)"

  if [[ "$loss" == *"nan"* ]]; then
    write_status running training_nan_detected "training loss is NaN; consider stopping" \
      "$workload_pid" "$epoch" "$loss" "$val"
  else
    write_status running "hi_gram_training_${SUB}" \
      "HI-GRAM $SUB training on GPU${GPU} (light protocol)." \
      "$workload_pid" "$epoch" "$loss" "$val"
  fi

  # Append one telemetry sample
  nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader,nounits --id="$GPU" >> "$TELEMETRY" 2>/dev/null || true

  sleep "$INTERVAL"
done
