#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jiangtangyunzhi/projects/recomm
CONFIG="$ROOT/artifacts/phase7/configs/st_gcgd_v21_graph_p0g2_preregistered.json"
OUTPUT="$ROOT/artifacts/phase7/st_gcgd_v21_p0g2"
LOG="$OUTPUT/run.log"
STATUS="$OUTPUT/status.json"
CURRENT_FILE="$OUTPUT/current_dataset.txt"
SESSION=gram_phase7_st_gcgd_v21_p0g2
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python3.9
WORKLOAD="$ROOT/experiment/phase7/st_gcgd_v21_p0g2.py"
TEST="$ROOT/experiment/phase7/test_st_gcgd_v21.py"
LEASE_HELPER="$ROOT/experiment/gpu_memory_lease.py"
RESERVER=/home/jiangtangyunzhi/projects/UnitTest/tools/run_codellama.sh
WORKLOAD_PID=0
LEASE_PID=0
TELEMETRY_PID=0
CURRENT_DATASET=""
STAGE=not_started
STARTED_AT=""

reserver() {
  env SESSION=codellama HF_HOME=/home/jiangtangyunzhi/hf_cache HF_HUB_CACHE=/home/jiangtangyunzhi/hf_cache/hub \
    TRANSFORMERS_CACHE=/home/jiangtangyunzhi/hf_cache/hub "$RESERVER" "$@"
}

write_status() {
  local state=$1 reason=$2 tmp="${STATUS}.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE7_ST_GCGD_V21_P0_G2_V1","status":"%s","stage":"%s","dataset":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%s,"workload_pid":%s,"tmux_session":"%s","physical_gpu":0,"total_gpu_lease_mib":30720,"test_read":false,"sports_read":false}\n' \
    "$state" "$STAGE" "$CURRENT_DATASET" "$reason" "$STARTED_AT" "$(date -Is)" "$$" "$WORKLOAD_PID" "$SESSION" > "$tmp"
  mv "$tmp" "$STATUS"
}

config_ready() {
  "$PYTHON" -c 'import json,sys;c=json.load(open(sys.argv[1]));raise SystemExit(0 if c.get("execution_enabled") is True and c.get("decision_status")=="PREREGISTERED_FROZEN_READY_TO_RUN" else 1)' "$CONFIG"
}

locks_match() {
  "$PYTHON" -c 'import hashlib,json,pathlib,sys
c=json.load(open(sys.argv[1])); root=pathlib.Path(sys.argv[2]); lock=c["code_lock"]
for key in ("implementation","entrypoint","test","runner"):
 p=root/lock[key]["path"]
 if hashlib.sha256(p.read_bytes()).hexdigest()!=lock[key]["sha256"]: raise SystemExit(1)
for dataset in c["datasets"]:
 p=root/c["p0_r2_lineage"][dataset]["memory_summary"]
 if hashlib.sha256(p.read_bytes()).hexdigest()!=c["p0_r2_lineage"][dataset]["memory_summary_sha256"]: raise SystemExit(1)
 p=root/c["hard_negative_bank"]["datasets"][dataset]["path"]
 if hashlib.sha256(p.read_bytes()).hexdigest()!=c["hard_negative_bank"]["datasets"][dataset]["sha256"]: raise SystemExit(1)' "$CONFIG" "$ROOT"
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
  printf 'timestamp,index,memory_used_mib,memory_free_mib,utilization_gpu_percent,dataset\n' > "$OUTPUT/gpu_telemetry.csv"
  while true; do
    local row dataset
    row=$(nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits --id=0 2>/dev/null || true)
    dataset=$(sed -n '1p' "$CURRENT_FILE" 2>/dev/null || true)
    [[ -z "$row" ]] || printf '%s,%s\n' "$row" "$dataset" >> "$OUTPUT/gpu_telemetry.csv"
    sleep 5
  done
}

start_lease() {
  local dataset=$1 budget
  budget=$("$PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1]))["execution"]["domain_gpu_lease_mib"][sys.argv[2]]["workload_budget_mib"])' "$CONFIG" "$dataset")
  "$PYTHON" "$LEASE_HELPER" --gpu 0 --total-lease-mib 30720 --expected-workload-peak-mib "$budget" \
    --status-path "$OUTPUT/gpu_lease_${dataset}.json" &
  LEASE_PID=$!
  for _ in $(seq 1 30); do [[ -s "$OUTPUT/gpu_lease_${dataset}.json" ]] && return 0; sleep 1; done
  return 1
}

aggregate() {
  "$PYTHON" -c 'import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); data={d:json.load(open(root/d/"summary.json")) for d in ("Toys","Beauty")}
overall=all(v["qualification"]["passed"] and v["resource_audit"]["passed"] for v in data.values())
out={"experiment_id":"GRAM_PHASE7_ST_GCGD_V21_P0_G2_V1","status":"P0_G2_QUALIFIED_BOTH_DOMAINS" if overall else "P0_G2_FAIL_CLOSED","datasets":data,"p1_automatically_enabled":False,"test_read":False,"sports_read":False}
(root/"summary.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")' "$OUTPUT"
}

finish() {
  local scientific_rc=$? restore_rc=0
  trap - EXIT INT TERM HUP
  release_lease
  (( TELEMETRY_PID == 0 )) || kill "$TELEMETRY_PID" >/dev/null 2>&1 || true
  restore || restore_rc=$?
  STAGE=finished
  if (( scientific_rc == 0 && restore_rc == 0 )); then
    write_status succeeded "P0-G2 completed; qualification recorded; CodeLlama restored."
  elif (( scientific_rc == 0 )); then
    write_status failed_to_restore_resource "P0-G2 completed but CodeLlama restoration failed."
  else
    write_status failed "P0-G2 exit=$scientific_rc; no automatic retry; CodeLlama restoration requested."
  fi
  (( scientific_rc != 0 )) && exit "$scientific_rc"
  exit "$restore_rc"
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
  config_ready || { write_status blocked "P0-G2 config not ready."; exit 3; }
  locks_match || { write_status blocked "P0-G2 code/input SHA mismatch."; exit 4; }
  "$PYTHON" -m pytest -q "$TEST" "$ROOT/experiment/phase7/test_st_gcgd_v2.py" || { write_status blocked "P0-G2 tests failed."; exit 5; }
  codellama_running || { write_status blocked "CodeLlama must run on GPU0 before release."; exit 6; }
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
    printf '%s\n' "$dataset" > "$CURRENT_FILE"
    STAGE=domain_lease
    start_lease "$dataset" || { write_status blocked "Sidecar lease failed."; exit 8; }
    STAGE=train_only_graph_qualification
    timeout --signal=TERM 43200 env CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$WORKLOAD" --config "$CONFIG" --output-root "$OUTPUT" --dataset "$dataset" &
    WORKLOAD_PID=$!
    write_status running "Deep four-arm P0-G2 running under exact 30 GiB lease."
    wait "$WORKLOAD_PID"
    WORKLOAD_PID=0
    release_lease
  done
  STAGE=aggregation
  aggregate
}

case "${1:-status}" in
  start)
    mkdir -p "$OUTPUT"
    config_ready || { STAGE=blocked; write_status blocked "P0-G2 config not ready."; exit 3; }
    locks_match || { STAGE=blocked; write_status blocked "P0-G2 code/input SHA mismatch."; exit 4; }
    tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists: $SESSION" >&2; exit 1; }
    STARTED_AT=$(date -Is)
    printf -v launch_cmd 'bash %q worker %q >> %q 2>&1' "$0" "$STARTED_AT" "$LOG"
    tmux new-session -d -s "$SESSION" "$launch_cmd"
    STAGE=starting
    write_status starting "Persistent P0-G2 session started."
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
