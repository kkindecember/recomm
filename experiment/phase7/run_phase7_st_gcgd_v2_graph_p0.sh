#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jiangtangyunzhi/projects/recomm
CONFIG="$ROOT/artifacts/phase7/configs/st_gcgd_v2_graph_p0_preregistered.json"
OUTPUT="$ROOT/artifacts/phase7/st_gcgd_v2"
LOG="$OUTPUT/graph_p0_run.log"
STATUS="$OUTPUT/graph_p0_status.json"
SESSION=gram_phase7_st_gcgd_v2_graph_p0
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python3.9
WORKLOAD="$ROOT/experiment/phase7/st_gcgd_v2.py"
TEST="$ROOT/experiment/phase7/test_st_gcgd_v2.py"
LEASE_HELPER="$ROOT/experiment/gpu_memory_lease.py"
RESERVER=/home/jiangtangyunzhi/projects/UnitTest/tools/run_codellama.sh
WORKLOAD_PID=0
LEASE_PID=0
TELEMETRY_PID=0
CURRENT_DATASET=""
STAGE=not_started
STARTED_AT=""

reserver() {
  env SESSION=codellama HF_HOME=/home/jiangtangyunzhi/hf_cache \
    HF_HUB_CACHE=/home/jiangtangyunzhi/hf_cache/hub TRANSFORMERS_CACHE=/home/jiangtangyunzhi/hf_cache/hub \
    "$RESERVER" "$@"
}

write_status() {
  local state=$1 reason=$2 tmp="${STATUS}.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE7_ST_GCGD_V2_P0_G_V1","status":"%s","stage":"%s","dataset":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%s,"workload_pid":%s,"tmux_session":"%s","physical_gpu":0,"total_gpu_lease_mib":30720,"test_read":false,"sports_read":false}\n' \
    "$state" "$STAGE" "$CURRENT_DATASET" "$reason" "$STARTED_AT" "$(date -Is)" "$$" "$WORKLOAD_PID" "$SESSION" > "$tmp"
  mv "$tmp" "$STATUS"
}

config_ready() {
  "$PYTHON" -c 'import json,sys;c=json.load(open(sys.argv[1]));raise SystemExit(0 if c.get("execution_enabled") is True and c.get("decision_status")=="PREREGISTERED_FROZEN_READY_TO_RUN" else 1)' "$CONFIG"
}

lineage_and_locks_match() {
  "$PYTHON" -c 'import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); c=json.load(open(sys.argv[2])); lock=c["code_lock"]
for key,path_key in (("implementation_sha256","implementation"),("test_sha256","test")):
 p=root/lock[path_key];
 if hashlib.sha256(p.read_bytes()).hexdigest()!=lock[key]: raise SystemExit(1)
for dataset in c["datasets"]:
 spec=c["p0_r_lineage"]["datasets"][dataset]
 for name in ("memory_summary","hard_negative_cache"):
  p=root/spec[name]
  if hashlib.sha256(p.read_bytes()).hexdigest()!=spec[name+"_sha256"]: raise SystemExit(1)' "$ROOT" "$CONFIG"
}

codellama_running() {
  local value
  value=$(reserver status 2>&1 || true)
  [[ "$value" == *"tmux session: running (codellama)"* ]] && [[ "$value" == *"gpu=0"* ]]
}

release_lease() {
  (( LEASE_PID == 0 )) || kill "$LEASE_PID" >/dev/null 2>&1 || true
  (( LEASE_PID == 0 )) || wait "$LEASE_PID" 2>/dev/null || true
  LEASE_PID=0
}

restore() {
  codellama_running && return 0
  reserver start 0 || return $?
  for _ in $(seq 1 60); do codellama_running && return 0; sleep 5; done
  return 1
}

telemetry() {
  printf 'timestamp,index,memory_used_mib,memory_free_mib,utilization_gpu_percent,dataset\n' > "$OUTPUT/graph_p0_gpu_telemetry.csv"
  while true; do
    local row
    row=$(nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits --id=0 2>/dev/null || true)
    [[ -z "$row" ]] || printf '%s,%s\n' "$row" "$CURRENT_DATASET" >> "$OUTPUT/graph_p0_gpu_telemetry.csv"
    sleep 5
  done
}

finish() {
  local scientific_rc=$? restore_rc=0
  trap - EXIT INT TERM HUP
  release_lease
  (( TELEMETRY_PID == 0 )) || kill "$TELEMETRY_PID" >/dev/null 2>&1 || true
  restore || restore_rc=$?
  STAGE=finished
  if (( scientific_rc == 0 && restore_rc == 0 )); then
    write_status succeeded "P0-G completed; results await qualification decision; CodeLlama restored."
  elif (( scientific_rc == 0 )); then
    write_status failed_to_restore_resource "P0-G completed but CodeLlama restoration failed."
  else
    write_status failed "P0-G exit=$scientific_rc; no automatic retry; CodeLlama restoration requested."
  fi
  (( scientific_rc != 0 )) && exit "$scientific_rc"
  exit "$restore_rc"
}

start_lease() {
  local dataset=$1 budget
  budget=$("$PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1]))["execution"]["domain_gpu_lease_mib"][sys.argv[2]]["workload_budget_mib"])' "$CONFIG" "$dataset")
  "$PYTHON" "$LEASE_HELPER" --gpu 0 --total-lease-mib 30720 --expected-workload-peak-mib "$budget" --status-path "$OUTPUT/graph_p0_gpu_lease_${dataset}.json" &
  LEASE_PID=$!
  for _ in $(seq 1 30); do [[ -s "$OUTPUT/graph_p0_gpu_lease_${dataset}.json" ]] && return 0; sleep 1; done
  return 1
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
  config_ready || { write_status blocked "P0-G config not frozen/enabled."; exit 3; }
  lineage_and_locks_match || { write_status blocked "P0-R lineage or code SHA mismatch."; exit 4; }
  "$PYTHON" -m pytest -q "$TEST" || { write_status blocked "P0-G tests failed."; exit 5; }
  codellama_running || { write_status blocked "CodeLlama must be running on GPU0 before release."; exit 6; }
  STAGE=resource_release
  reserver stop
  local free_mib=""
  for _ in $(seq 1 60); do
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id=0 2>/dev/null | tr -d ' ' || true)
    [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= 30720 )) && break
    sleep 5
  done
  [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= 30720 )) || { write_status blocked "GPU0 admission gate failed."; exit 7; }
  telemetry & TELEMETRY_PID=$!
  for dataset in Toys Beauty; do
    CURRENT_DATASET=$dataset
    STAGE=domain_lease
    start_lease "$dataset" || { write_status blocked "Sidecar lease failed."; exit 8; }
    STAGE=train_only_graph_qualification
    timeout --signal=TERM 21600 env CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$WORKLOAD" --mode p0-g --config "$CONFIG" --output-root "$OUTPUT" --dataset "$dataset" &
    WORKLOAD_PID=$!
    write_status running "P0-G four-arm train-only qualification running."
    wait "$WORKLOAD_PID"
    WORKLOAD_PID=0
    release_lease
  done
}

case "${1:-status}" in
  start)
    mkdir -p "$OUTPUT"
    config_ready || { STAGE=blocked; write_status blocked "P0-G config not ready."; exit 3; }
    lineage_and_locks_match || { STAGE=blocked; write_status blocked "P0-R lineage or code SHA mismatch."; exit 4; }
    tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists: $SESSION" >&2; exit 1; }
    STARTED_AT=$(date -Is)
    printf -v launch_cmd 'bash %q worker %q >> %q 2>&1' "$0" "$STARTED_AT" "$LOG"
    tmux new-session -d -s "$SESSION" "$launch_cmd"
    STAGE=starting
    write_status starting "Persistent P0-G session started."
    echo "started $SESSION"
    ;;
  worker) worker "${2:?missing start timestamp}" ;;
  status)
    tmux has-session -t "$SESSION" 2>/dev/null && echo "tmux session: running ($SESSION)" || echo "tmux session: not running ($SESSION)"
    [[ -f "$STATUS" ]] && sed -n '1,100p' "$STATUS" || echo '{"status":"not_started"}'
    [[ -f "$LOG" ]] && tail -n 50 "$LOG" || true
    reserver status || true
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits --id=0 || true
    ;;
  *) echo "usage: $0 {start|status|worker}" >&2; exit 2 ;;
esac
