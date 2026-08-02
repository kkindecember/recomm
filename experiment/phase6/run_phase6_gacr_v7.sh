#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="$ROOT/artifacts/phase6/configs/gacr_v7_preregistered.json"
OUTPUT="$ROOT/artifacts/phase6/gacr_v7"
LOG="$OUTPUT/run.log"
STATUS="$OUTPUT/status.json"
TELEMETRY="$OUTPUT/gpu_telemetry.csv"
SESSION=gram_phase6_gacr_v7
GPU=0
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python3.9
WORKLOAD="$ROOT/experiment/phase6/gacr_v7.py"
TEST_FILE="$ROOT/experiment/phase6/test_gacr_v7.py"
PLAN="$ROOT/plan/第六阶段/GRAM_第六阶段_GACR-v7全量指标对齐残差训练实验计划.md"
RESERVER="$ROOT/tools/run_codellama.sh"
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
  printf '{"experiment_id":"GRAM_PHASE6_GACR_V7_METRIC_ALIGNED_FULL_FIT","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%s,"workload_pid":%s,"tmux_session":"%s","physical_gpu":0,"log_path":"%s","result_path":"%s","resource_reservation":"%s","codellama_restore_gpu":0,"test_read":false,"sports_read":false}\n' \
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
  [[ "$value" == *"tmux session: running ($RESERVER_SESSION)"* ]] \
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
  local experiment_rc=$? restore_rc=0 result_status=""
  trap - EXIT INT TERM HUP
  if [[ -n "$TELEMETRY_PID" ]]; then
    kill "$TELEMETRY_PID" >/dev/null 2>&1 || true
  fi
  if [[ -s "$OUTPUT/summary.json" ]]; then
    result_status=$("$PYTHON" -c \
      'import json,sys; print(json.load(open(sys.argv[1])).get("result_status", ""))' \
      "$OUTPUT/summary.json" 2>/dev/null || true)
  fi
  restore_resource || restore_rc=$?
  CURRENT_STAGE=finished
  if (( experiment_rc == 0 && restore_rc == 0 )); then
    if [[ "$result_status" == STOPPED_BEFORE_FRESH_VALIDATION_CALIBRATION_GATE_FAILED ]]; then
      write_status completed_without_validation "Calibration noninferiority failed; fresh validation was not read; CodeLlama restored."
    else
      write_status succeeded "GACR-v7 completed; results await researcher-requested analysis."
    fi
  elif (( restore_rc != 0 )); then
    write_status failed_to_restore_resource "Experiment exit=$experiment_rc; CodeLlama GPU0 restoration failed."
  else
    write_status failed "Experiment exit=$experiment_rc; no automatic retry; CodeLlama restored on GPU0."
  fi
  if (( experiment_rc != 0 )); then exit "$experiment_rc"; fi
  exit "$restore_rc"
}

verify_frozen_materials() {
  "$PYTHON" - "$CONFIG" "$WORKLOAD" "$TEST_FILE" "$0" "$PLAN" <<'PY'
import hashlib
import json
import pathlib
import sys

config_path, implementation, tests, runner, plan = map(pathlib.Path, sys.argv[1:])
config = json.loads(config_path.read_text())
if config.get("decision_status_before_run") != "PREREGISTERED_FROZEN_READY_TO_RUN":
    raise SystemExit("v7 config is not execution-enabled")
expected = config["implementation_lock"]
for key, path in (
    ("implementation_sha256", implementation),
    ("test_sha256", tests),
    ("runner_sha256", runner),
    ("plan_sha256", plan),
):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected[key]:
        raise SystemExit(f"frozen material mismatch {path}: expected={expected[key]} actual={actual}")
PY
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
  for required in "$CONFIG" "$WORKLOAD" "$TEST_FILE" "$PLAN" \
    "$ROOT/artifacts/phase4/gcdh_p0/Toys/C1/model.pt" \
    "$ROOT/artifacts/phase4/gcdh_p0/Beauty/C1/model.pt" \
    "$ROOT/artifacts/phase6/gacr_v2/Toys/residual_seed2023.pt" \
    "$ROOT/artifacts/phase6/gacr_v2/Toys/residual_seed2024.pt" \
    "$ROOT/artifacts/phase6/gacr_v2/Toys/residual_seed2025.pt" \
    "$ROOT/artifacts/phase6/gacr_v2/Beauty/residual_seed2023.pt" \
    "$ROOT/artifacts/phase6/gacr_v2/Beauty/residual_seed2024.pt" \
    "$ROOT/artifacts/phase6/gacr_v2/Beauty/residual_seed2025.pt" \
    "$ROOT/artifacts/phase6/gacr_v6/Toys/residual_seed2023.pt" \
    "$ROOT/artifacts/phase6/gacr_v6/Toys/residual_seed2024.pt" \
    "$ROOT/artifacts/phase6/gacr_v6/Toys/residual_seed2025.pt" \
    "$ROOT/artifacts/phase6/gacr_v6/Beauty/residual_seed2023.pt" \
    "$ROOT/artifacts/phase6/gacr_v6/Beauty/residual_seed2024.pt" \
    "$ROOT/artifacts/phase6/gacr_v6/Beauty/residual_seed2025.pt"; do
    if [[ ! -s "$required" ]]; then
      write_status blocked "Required locked material missing: $required"
      exit 2
    fi
  done
  verify_frozen_materials
  CURRENT_STAGE=resource_release
  write_status releasing_resource "Stopping CodeLlama before GACR-v7 on physical GPU0."
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
  CURRENT_STAGE=gacr_v7_metric_aligned_full_fit
  telemetry_worker &
  TELEMETRY_PID=$!
  timeout --signal=TERM "$HARD_TIMEOUT_SECONDS" env CUDA_VISIBLE_DEVICES="$GPU" \
    "$PYTHON" "$WORKLOAD" --config "$CONFIG" --output-root "$OUTPUT" &
  WORKLOAD_PID=$!
  write_status running "GACR-v7 metric-aligned full-fit running on physical GPU0."
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
      write_status starting "Persistent GACR-v7 session started for physical GPU0."
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
    *) echo "usage: bash experiment/phase6/run_phase6_gacr_v7.sh {start|status|worker}" >&2; exit 2 ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi
