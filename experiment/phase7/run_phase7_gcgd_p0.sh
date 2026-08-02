#!/usr/bin/env bash
# CPU-only P0 lineage audit. CodeLlama remains resident on GPU0 throughout.
set -euo pipefail

ROOT=/home/jiangtangyunzhi/projects/recomm
CONFIG="$ROOT/artifacts/phase7/configs/gcgd_p0_preregistered.json"
OUTPUT="$ROOT/artifacts/phase7/gcgd_p0"
LOG="$OUTPUT/run.log"
STATUS="$OUTPUT/status.json"
SESSION=gram_phase7_gcgd_p0
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python3.9
WORKLOAD="$ROOT/experiment/phase7/gcgd_p0.py"
RESERVER=/home/jiangtangyunzhi/projects/UnitTest/tools/run_codellama.sh
CODELLAMA_HF_HOME=/home/jiangtangyunzhi/hf_cache
WORKLOAD_PID=0
STARTED_AT=""
STAGE=not_started
RESERVATION=codellama_expected_running_on_gpu0

reserver() {
  env SESSION=codellama HF_HOME="$CODELLAMA_HF_HOME" \
    HF_HUB_CACHE="$CODELLAMA_HF_HOME/hub" TRANSFORMERS_CACHE="$CODELLAMA_HF_HOME/hub" \
    "$RESERVER" "$@"
}

write_status() {
  local state=$1 reason=$2 tmp="${STATUS}.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE7_GCGD_P0_LINEAGE_V1","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%s,"workload_pid":%s,"tmux_session":"%s","mode":"cpu_only","physical_gpu_reserved_by_codellama":0,"codellama_reservation_mib":30720,"log_path":"%s","result_path":"%s","resource_reservation":"%s","test_read":false,"sports_read":false}\n' \
    "$state" "$STAGE" "$reason" "$STARTED_AT" "$(date -Is)" "$$" "$WORKLOAD_PID" "$SESSION" \
    "${LOG#$ROOT/}" "${OUTPUT#$ROOT/}/summary.json" "$RESERVATION" > "$tmp"
  mv "$tmp" "$STATUS"
}

config_is_ready() {
  "$PYTHON" -c 'import json,sys; c=json.load(open(sys.argv[1])); ready=c.get("execution_enabled") is True and c.get("decision_status")=="PREREGISTERED_FROZEN_READY_TO_RUN" and c.get("execution",{}).get("mode")=="cpu_only"; raise SystemExit(0 if ready else 1)' "$CONFIG"
}

locked_materials_match() {
  "$PYTHON" -c 'import hashlib,json,sys; c=json.load(open(sys.argv[1]))["implementation_lock"]; paths=sys.argv[2:]; keys=("implementation_sha256","test_sha256","runner_sha256"); actual=[hashlib.sha256(open(path,"rb").read()).hexdigest() for path in paths]; expected=[c[key] for key in keys]; raise SystemExit(0 if actual==expected else 1)' \
    "$CONFIG" "$WORKLOAD" "$ROOT/experiment/phase7/test_gcgd_p0.py" "$0"
}

codellama_is_running_on_gpu0() {
  local value
  value=$(reserver status 2>&1 || true)
  [[ "$value" == *"tmux session: running (codellama)"* ]] \
    && [[ "$value" == *"state=running session=codellama gpu=0"* ]]
}

ensure_codellama_gpu0() {
  if codellama_is_running_on_gpu0; then
    RESERVATION=codellama_running_on_gpu0
    return 0
  fi
  RESERVATION=restoring_codellama_to_gpu0
  reserver start 0 || true
  for _ in $(seq 1 60); do
    codellama_is_running_on_gpu0 && { RESERVATION=codellama_running_on_gpu0; return 0; }
    sleep 5
  done
  RESERVATION=codellama_restore_failed_on_gpu0
  return 1
}

finish() {
  local scientific_rc=$? reservation_rc=0
  trap - EXIT INT TERM HUP
  ensure_codellama_gpu0 || reservation_rc=$?
  STAGE=finished
  if (( scientific_rc == 0 && reservation_rc == 0 )); then
    write_status succeeded "P0 lineage audit completed; CodeLlama remained/restored on GPU0."
  elif (( scientific_rc == 0 )); then
    write_status failed_to_restore_resource "P0 completed but CodeLlama GPU0 reservation is not running."
  else
    write_status failed "P0 exit=$scientific_rc; no automatic retry; CodeLlama GPU0 reservation checked."
  fi
  (( scientific_rc != 0 )) && exit "$scientific_rc"
  exit "$reservation_rc"
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
  for required in "$CONFIG" "$WORKLOAD"; do
    [[ -s "$required" ]] || { write_status blocked "Required input missing: $required"; exit 2; }
  done
  config_is_ready || { write_status blocked "P0 config is not frozen and execution-enabled."; exit 3; }
  locked_materials_match || { write_status blocked "P0 implementation/test/runner SHA mismatch."; exit 4; }
  ensure_codellama_gpu0 || { write_status blocked "CodeLlama could not be confirmed on physical GPU0."; exit 5; }
  STAGE=cpu_lineage_audit
  timeout --signal=TERM 1800 env CUDA_VISIBLE_DEVICES="" HF_HOME="$ROOT/.cache/huggingface" \
    TRANSFORMERS_CACHE="$ROOT/.cache/huggingface" TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    "$PYTHON" "$WORKLOAD" --config "$CONFIG" --output-root "$OUTPUT" &
  WORKLOAD_PID=$!
  write_status running "CPU-only P0 audit running in background; CodeLlama holds GPU0."
  wait "$WORKLOAD_PID"
}

case "${1:-status}" in
  start)
    mkdir -p "$OUTPUT"
    config_is_ready || { STAGE=blocked; write_status blocked "P0 config is not ready."; exit 3; }
    locked_materials_match || { STAGE=blocked; write_status blocked "P0 locked material SHA mismatch."; exit 4; }
    tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists: $SESSION" >&2; exit 1; }
    STARTED_AT=$(date -Is)
    printf -v launch_cmd 'bash %q worker %q >> %q 2>&1' "$0" "$STARTED_AT" "$LOG"
    tmux new-session -d -s "$SESSION" "$launch_cmd"
    STAGE=starting
    write_status starting "Persistent CPU-only P0 session started; CodeLlama remains on GPU0."
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
