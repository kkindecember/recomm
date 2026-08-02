#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="$ROOT/artifacts/phase7/configs/st_gcgd_v21_p0g2_beauty_recovery.json"
OUTPUT="$ROOT/artifacts/phase7/st_gcgd_v21_p0g2_recovery"
LOG="$OUTPUT/run.log"
STATUS="$OUTPUT/status.json"
SESSION=gram_phase7_st_gcgd_v21_p0g2_beauty_recovery
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python3.9
WORKLOAD="$ROOT/experiment/phase7/st_gcgd_v21_p0g2.py"
LEASE_HELPER="$ROOT/experiment/gpu_memory_lease.py"
RESERVER=/home/jiangtangyunzhi/projects/UnitTest/tools/run_codellama.sh
WORKLOAD_PID=0
LEASE_PID=0
TELEMETRY_PID=0
STAGE=not_started
STARTED_AT=""

reserver() {
  env SESSION=codellama HF_HOME=/home/jiangtangyunzhi/hf_cache HF_HUB_CACHE=/home/jiangtangyunzhi/hf_cache/hub \
    TRANSFORMERS_CACHE=/home/jiangtangyunzhi/hf_cache/hub "$RESERVER" "$@"
}

write_status() {
  local state=$1 reason=$2 tmp="${STATUS}.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE7_ST_GCGD_V21_P0_G2_BEAUTY_RECOVERY_V1","status":"%s","stage":"%s","dataset":"Beauty","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%s,"workload_pid":%s,"tmux_session":"%s","physical_gpu":0,"resource_override":"USER_AUTHORIZED_OVERSHOOT","test_read":false,"sports_read":false}\n' \
    "$state" "$STAGE" "$reason" "$STARTED_AT" "$(date -Is)" "$$" "$WORKLOAD_PID" "$SESSION" > "$tmp"
  mv "$tmp" "$STATUS"
}

preflight() {
  "$PYTHON" -c 'import hashlib,json,pathlib,sys
c=json.load(open(sys.argv[1])); root=pathlib.Path(sys.argv[2]); lock=c["code_lock"]
if not (c.get("execution_enabled") is True and c.get("recovery",{}).get("dataset")=="Beauty"): raise SystemExit(1)
for spec in lock.values():
 p=root/spec["path"]
 if hashlib.sha256(p.read_bytes()).hexdigest()!=spec["sha256"]: raise SystemExit(1)
p=root/c["recovery"]["parent_toys_summary"]
if hashlib.sha256(p.read_bytes()).hexdigest()!=c["recovery"]["parent_toys_summary_sha256"]: raise SystemExit(1)
p=root/c["hard_negative_bank"]["datasets"]["Beauty"]["path"]
raise SystemExit(0 if hashlib.sha256(p.read_bytes()).hexdigest()==c["hard_negative_bank"]["datasets"]["Beauty"]["sha256"] else 1)' "$CONFIG" "$ROOT"
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
    local row
    row=$(nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits --id=0 2>/dev/null || true)
    [[ -z "$row" ]] || printf '%s,Beauty\n' "$row" >> "$OUTPUT/gpu_telemetry.csv"
    sleep 5
  done
}

aggregate() {
  "$PYTHON" -c 'import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); repo=pathlib.Path(sys.argv[2]); c=json.load(open(sys.argv[3]))
toys=json.load(open(repo/c["recovery"]["parent_toys_summary"])); beauty=json.load(open(root/"Beauty"/"summary.json"))
qualified=toys["qualification"]["passed"] and beauty["qualification"]["passed"]
out={"experiment_id":"GRAM_PHASE7_ST_GCGD_V21_P0_G2_RECOVERED_V1","status":"P0_G2_QUALIFIED_BOTH_DOMAINS" if qualified else "P0_G2_FAIL_CLOSED","datasets":{"Toys":toys,"Beauty":beauty},"resource_policy":"USER_AUTHORIZED_OVERSHOOT_ABOVE_30_GIB","scientific_configuration_changed":False,"p1_automatically_enabled":False,"test_read":False,"sports_read":False}
(root/"summary.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")' "$OUTPUT" "$ROOT" "$CONFIG"
}

finish() {
  local scientific_rc=$? restore_rc=0
  trap - EXIT INT TERM HUP
  release_lease
  (( TELEMETRY_PID == 0 )) || kill "$TELEMETRY_PID" >/dev/null 2>&1 || true
  restore || restore_rc=$?
  STAGE=finished
  if (( scientific_rc == 0 && restore_rc == 0 )); then
    write_status succeeded "Beauty recovery completed; CodeLlama restored."
  elif (( scientific_rc == 0 )); then
    write_status failed_to_restore_resource "Beauty completed but CodeLlama restoration failed."
  else
    write_status failed "Beauty recovery exit=$scientific_rc; no automatic retry; CodeLlama restoration requested."
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
  preflight || { write_status blocked "Recovery config/code/input SHA mismatch."; exit 4; }
  "$PYTHON" -m pytest -q experiment/phase7/test_st_gcgd_v21.py experiment/phase7/test_st_gcgd_v2.py || { write_status blocked "Tests failed."; exit 5; }
  codellama_running || { write_status blocked "CodeLlama must run before release."; exit 6; }
  STAGE=resource_release
  reserver stop
  local free_mib=""
  for _ in $(seq 1 60); do
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id=0 2>/dev/null | tr -d ' ' || true)
    [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= 30720 )) && break
    sleep 5
  done
  [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= 30720 )) || { write_status blocked "GPU0 admission failed."; exit 7; }
  telemetry & TELEMETRY_PID=$!
  local budget
  budget=$("$PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1]))["execution"]["domain_gpu_lease_mib"]["Beauty"]["workload_budget_mib"])' "$CONFIG")
  "$PYTHON" "$LEASE_HELPER" --gpu 0 --total-lease-mib 30720 --expected-workload-peak-mib "$budget" --status-path "$OUTPUT/gpu_lease_Beauty.json" &
  LEASE_PID=$!
  for _ in $(seq 1 30); do [[ -s "$OUTPUT/gpu_lease_Beauty.json" ]] && break; sleep 1; done
  [[ -s "$OUTPUT/gpu_lease_Beauty.json" ]] || { write_status blocked "Sidecar failed."; exit 8; }
  STAGE=train_only_beauty_recovery
  timeout --signal=TERM 43200 env CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$WORKLOAD" --config "$CONFIG" --output-root "$OUTPUT" --dataset Beauty &
  WORKLOAD_PID=$!
  write_status running "Beauty P0-G2 recovery running under user-relaxed resource tolerance."
  wait "$WORKLOAD_PID"
  WORKLOAD_PID=0
  release_lease
  STAGE=aggregation
  aggregate
}

case "${1:-status}" in
  start)
    mkdir -p "$OUTPUT"
    preflight || { STAGE=blocked; write_status blocked "Recovery preflight failed."; exit 4; }
    tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists: $SESSION" >&2; exit 1; }
    STARTED_AT=$(date -Is)
    printf -v launch_cmd 'bash %q worker %q >> %q 2>&1' "$0" "$STARTED_AT" "$LOG"
    tmux new-session -d -s "$SESSION" "$launch_cmd"
    STAGE=starting
    write_status starting "Persistent Beauty recovery session started."
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
