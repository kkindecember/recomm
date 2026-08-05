#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="$ROOT/artifacts/phase11/configs/bw3_p1_admission_recovery_preregistered.json"
OUT="$ROOT/artifacts/phase11/bw3_p1_admission_recovery"
LOG="$OUT/run.log"
STATUS="$OUT/status.json"
LEASE_STATUS="$OUT/gpu_lease.json"
TELEMETRY="$OUT/gpu_telemetry.csv"
SESSION=gram_phase11_bw3_p1_admission_recovery
GPU=6
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
GENERATOR="$ROOT/experiment/phase11/generate_bw3_pseudofuture_beams.py"
TRAINER="$ROOT/experiment/phase11/train_bw3_admission_gate.py"
PLAN="$ROOT/plan/第十一阶段/GRAM_第十一阶段_BW3训练前缀扩展准入计划.md"
LEASE_HELPER="$ROOT/experiment/gpu_memory_lease.py"
RESERVER=/home/jiangtangyunzhi/projects/UnitTest/tools/run_codellama.sh
CODELLAMA_HF_HOME=/home/jiangtangyunzhi/hf_cache
WORKLOAD_CACHE="$ROOT/.cache/huggingface"
TOTAL_LEASE_MIB=30720
EXPECTED_PEAK_MIB=26881
WORKLOAD_PID=0
LEASE_PID=""
TELEMETRY_PID=""
STARTED_AT=""
STAGE=not_started
RESERVATION=codellama_expected_on_gpu6
FINAL_STATUS=succeeded
FINAL_REASON="BW3-P1 recovery completed; results await researcher analysis."

reserver() {
  env SESSION=codellama HF_HOME="$CODELLAMA_HF_HOME" \
    HF_HUB_CACHE="$CODELLAMA_HF_HOME/hub" \
    TRANSFORMERS_CACHE="$CODELLAMA_HF_HOME/hub" "$RESERVER" "$@"
}

write_status() {
  local state=$1 reason=$2 tmp="${STATUS}.tmp.$$"
  mkdir -p "$OUT"
  printf '{"experiment_id":"GRAM_PHASE11_BW3_P1_TRAIN_PREFIX_ADMISSION_RECOVERY_V1","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%s,"workload_pid":%s,"tmux_session":"%s","physical_gpu":6,"total_gpu_lease_mib":30720,"expected_workload_peak_mib":26881,"resource_reservation":"%s","output_reuse":false,"validation_target_read":false,"test_read":false,"sports_read":false,"log_path":"%s"}\n' \
    "$state" "$STAGE" "$reason" "$STARTED_AT" "$(date -Is)" "$$" "$WORKLOAD_PID" \
    "$SESSION" "$RESERVATION" "${LOG#$ROOT/}" > "$tmp"
  mv "$tmp" "$STATUS"
}

config_enabled() {
  "$PYTHON" -c 'import json,sys; c=json.load(open(sys.argv[1])); raise SystemExit(0 if c.get("execution_enabled") is True and c.get("decision_status")=="AUTHORIZED_BW3_P1_RECOVERY" else 1)' "$CONFIG"
}

verify_locks() {
  "$PYTHON" -c 'import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); config=json.load(open(sys.argv[2]))
for rel,expected in config["code_lock"]["files"].items():
 path=root/rel
 assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==expected, rel
for entry in config["input_lock"]:
 path=root/entry["path"]
 assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==entry["sha256"], entry["path"]' "$ROOT" "$CONFIG"
}

codellama_on_target() {
  local value
  value=$(reserver status 2>&1 || true)
  [[ "$value" == *"tmux session: running (codellama)"* ]] && [[ "$value" == *"gpu=6"* ]]
}

ensure_codellama_on_target() {
  if codellama_on_target; then
    RESERVATION=codellama_confirmed_on_gpu6
    return 0
  fi
  reserver stop >/dev/null 2>&1 || true
  reserver start "$GPU"
  for _ in $(seq 1 180); do
    if codellama_on_target; then
      RESERVATION=codellama_confirmed_on_gpu6
      return 0
    fi
    sleep 5
  done
  RESERVATION=codellama_prepare_failed_on_gpu6
  return 1
}

telemetry() {
  printf 'timestamp,index,memory_used_mib,memory_free_mib,utilization_gpu_percent\n' > "$TELEMETRY"
  while true; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits --id="$GPU" >> "$TELEMETRY" 2>/dev/null || true
    sleep 5
  done
}

release_lease() {
  [[ -z "$LEASE_PID" ]] || kill "$LEASE_PID" >/dev/null 2>&1 || true
  [[ -z "$LEASE_PID" ]] || wait "$LEASE_PID" 2>/dev/null || true
  LEASE_PID=""
}

restore() {
  if codellama_on_target; then
    RESERVATION=codellama_already_running_on_gpu6
    return 0
  fi
  STAGE=resource_restoration
  RESERVATION=restoring_codellama_to_gpu6
  write_status restoring_resource "BW3-P1 ended; restoring CodeLlama on GPU6."
  reserver start "$GPU" || { RESERVATION=restore_request_failed_on_gpu6; return 1; }
  for _ in $(seq 1 180); do
    if codellama_on_target; then
      RESERVATION=restored_on_gpu6
      return 0
    fi
    sleep 5
  done
  RESERVATION=restore_failed_on_gpu6
  return 1
}

finish() {
  local scientific_rc=$? restore_rc=0
  trap - EXIT INT TERM HUP
  if (( WORKLOAD_PID > 0 )) && kill -0 "$WORKLOAD_PID" 2>/dev/null; then
    kill -TERM "$WORKLOAD_PID" 2>/dev/null || true
    wait "$WORKLOAD_PID" 2>/dev/null || true
  fi
  release_lease
  [[ -z "$TELEMETRY_PID" ]] || kill "$TELEMETRY_PID" >/dev/null 2>&1 || true
  if (( scientific_rc != 0 )); then
    FINAL_STATUS=failed
    FINAL_REASON="Scientific exit=${scientific_rc}; no automatic retry."
  fi
  restore || restore_rc=$?
  STAGE=finished
  if (( restore_rc != 0 )); then
    write_status failed_to_restore_resource "$FINAL_REASON CodeLlama restoration failed."
  else
    write_status "$FINAL_STATUS" "$FINAL_REASON"
  fi
  (( scientific_rc != 0 )) && exit "$scientific_rc"
  exit "$restore_rc"
}

run_unit() {
  local dataset=$1 split=$2 offset=$3 users=$4
  local unit="$OUT/$dataset/$split"
  STAGE="${dataset}_${split}_beams"
  write_status running "Generating fresh offset-${offset} beam50/200 for ${users} users."
  timeout --signal=TERM 7200 env CUDA_VISIBLE_DEVICES="$GPU" PYTHONHASHSEED=2023 \
    HF_HUB_CACHE="$WORKLOAD_CACHE" TRANSFORMERS_CACHE="$WORKLOAD_CACHE/transformers" \
    TRANSFORMERS_OFFLINE=1 "$PYTHON" "$GENERATOR" --dataset "$dataset" \
    --offset "$offset" --users "$users" --device cuda:0 --output-dir "$unit" &
  WORKLOAD_PID=$!
  write_status running "Generating fresh offset-${offset} beam50/200 for ${users} users."
  wait "$WORKLOAD_PID"
  WORKLOAD_PID=0
  local gate
  gate=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["gate"]["status"])' "$unit/summary.json")
  if [[ "$gate" != passed ]]; then
    FINAL_STATUS=stopped
    FINAL_REASON="${dataset} ${split} pseudo-future integrity failed; validation remains unread."
    return 10
  fi
}

worker() {
  STARTED_AT=${1:?missing start timestamp}
  trap finish EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  trap 'exit 129' HUP
  cd "$ROOT"
  mkdir -p "$OUT"
  STAGE=preflight
  write_status preflight "Checking frozen code, inputs, offline cache and tests."
  for required in "$CONFIG" "$GENERATOR" "$TRAINER" "$PLAN" "$LEASE_HELPER" "$RESERVER"; do
    [[ -s "$required" ]] || { write_status blocked "Missing required file: $required"; exit 2; }
  done
  config_enabled || { write_status blocked "Config is not enabled."; exit 3; }
  verify_locks || { write_status blocked "SHA256 lock mismatch."; exit 4; }
  PYTHONPATH="$ROOT/experiment/phase11" "$PYTHON" -m pytest -q \
    "$ROOT/experiment/phase11/test_generate_bw3.py" \
    "$ROOT/experiment/phase11/test_bw3_admission_gate.py" \
    "$ROOT/experiment/phase11/test_bw3_pseudofuture.py"
  "$PYTHON" -m py_compile "$GENERATOR" "$TRAINER"
  bash -n "$0"
  env HF_HUB_CACHE="$WORKLOAD_CACHE" TRANSFORMERS_CACHE="$WORKLOAD_CACHE/transformers" \
    TRANSFORMERS_OFFLINE=1 "$PYTHON" -c \
    'from transformers import AutoTokenizer; AutoTokenizer.from_pretrained("t5-small",local_files_only=True)'

  STAGE=codellama_pre_reservation
  write_status preparing_resource "Ensuring CodeLlama occupies GPU6 before controlled release."
  ensure_codellama_on_target || { write_status blocked "Could not establish CodeLlama pre-reservation on GPU6."; exit 5; }
  STAGE=resource_release
  write_status releasing_resource "Stopping CodeLlama before BW3-P1."
  reserver stop
  RESERVATION=released_for_experiment

  STAGE=gpu_memory_gate
  write_status waiting_for_gpu "Waiting for at least 30720 MiB free on GPU6."
  local free_mib=""
  for _ in $(seq 1 120); do
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GPU" 2>/dev/null | tr -d ' ' || true)
    [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= TOTAL_LEASE_MIB )) && break
    sleep 5
  done
  [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= TOTAL_LEASE_MIB )) || { write_status blocked "GPU6 admission failed: ${free_mib:-unknown} MiB free."; exit 6; }

  STAGE=memory_lease
  "$PYTHON" "$LEASE_HELPER" --gpu "$GPU" --total-lease-mib "$TOTAL_LEASE_MIB" \
    --expected-workload-peak-mib "$EXPECTED_PEAK_MIB" --status-path "$LEASE_STATUS" &
  LEASE_PID=$!
  for _ in $(seq 1 30); do
    [[ -s "$LEASE_STATUS" ]] && grep -q '"state": "holding"' "$LEASE_STATUS" && break
    sleep 1
  done
  [[ -s "$LEASE_STATUS" ]] && grep -q '"state": "holding"' "$LEASE_STATUS" || { write_status blocked "GPU lease sidecar did not hold."; exit 7; }
  telemetry & TELEMETRY_PID=$!

  run_unit Toys fit 4 1024 || { [[ "$FINAL_STATUS" == stopped ]] && return 0; return 1; }
  run_unit Toys calibration 3 512 || { [[ "$FINAL_STATUS" == stopped ]] && return 0; return 1; }
  run_unit Beauty fit 4 1024 || { [[ "$FINAL_STATUS" == stopped ]] && return 0; return 1; }
  run_unit Beauty calibration 3 512 || { [[ "$FINAL_STATUS" == stopped ]] && return 0; return 1; }

  STAGE=fit_calibration
  write_status running "Fitting gates and selecting margins on offset-3 only."
  "$PYTHON" "$TRAINER" --root "$OUT" --output-dir "$OUT/admission" &
  WORKLOAD_PID=$!
  write_status running "Fitting gates and selecting margins on offset-3 only."
  wait "$WORKLOAD_PID"
  WORKLOAD_PID=0
  local p1_gate
  p1_gate=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["p1_gate"]["status"])' "$OUT/admission/summary.json")
  if [[ "$p1_gate" == passed ]]; then
    FINAL_STATUS=succeeded
    FINAL_REASON="BW3-P1 passed; one-shot validation P2 is eligible but was not started."
  else
    FINAL_STATUS=stopped
    FINAL_REASON="BW3-P1 calibration failed; validation remains unread."
  fi
}

case "${1:-status}" in
  start)
    mkdir -p "$OUT"
    config_enabled || { STAGE=config_not_enabled; write_status blocked "Start refused: config is not enabled."; exit 3; }
    tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists: $SESSION" >&2; exit 1; }
    [[ ! -e "$OUT/Toys" && ! -e "$OUT/Beauty" && ! -e "$OUT/admission" ]] || { echo "recovery output already contains scientific products; refusing reuse" >&2; exit 8; }
    STARTED_AT=$(date -Is)
    printf -v launch_cmd 'bash %q worker %q >> %q 2>&1' "$0" "$STARTED_AT" "$LOG"
    tmux new-session -d -s "$SESSION" "$launch_cmd"
    STAGE=starting
    write_status starting "Persistent BW3-P1 recovery session started from empty outputs."
    echo "started $SESSION"
    ;;
  worker) worker "${2:?missing start timestamp}" ;;
  status)
    tmux has-session -t "$SESSION" 2>/dev/null && echo "tmux session: running ($SESSION)" || echo "tmux session: not running ($SESSION)"
    [[ -f "$STATUS" ]] && sed -n '1,100p' "$STATUS" || echo '{"status":"not_started"}'
    [[ -f "$LEASE_STATUS" ]] && sed -n '1,100p' "$LEASE_STATUS" || true
    [[ -f "$TELEMETRY" ]] && tail -n 4 "$TELEMETRY" || true
    [[ -f "$LOG" ]] && tail -n 30 "$LOG" || true
    ;;
  stop)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      pane_pid=$(tmux list-panes -t "$SESSION" -F '#{pane_pid}' | head -n 1)
      kill -TERM "$pane_pid"
      echo "stop requested for $SESSION; cleanup will restore CodeLlama on GPU6"
    else
      echo "session not running: $SESSION"
    fi
    ;;
  *) echo "usage: $0 {start|status|stop|worker}" >&2; exit 2 ;;
esac
