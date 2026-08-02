#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="$ROOT/artifacts/phase7/configs/st_gcgd_v21_p1_preregistered.json"
OUTPUT="$ROOT/artifacts/phase7/st_gcgd_v21_p1"
LOG="$OUTPUT/run.log"
STATUS="$OUTPUT/status.json"
SESSION=gram_phase7_st_gcgd_v21_p1
GPU=0
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python3.9
WORKLOAD="$ROOT/experiment/phase7/st_gcgd_v21_p1.py"
TEST="$ROOT/experiment/phase7/test_st_gcgd_v21_p1.py"
RESERVER=/home/jiangtangyunzhi/projects/UnitTest/tools/run_codellama.sh
WORKLOAD_HF_HOME="$ROOT/.cache/huggingface"
CODELLAMA_HF_HOME=/home/jiangtangyunzhi/hf_cache
WORKLOAD_PID=0
TELEMETRY_PID=""
STARTED_AT=""
STAGE=not_started
CURRENT_DATASET=""
RESERVATION=codellama_expected_on_gpu0

reserver() {
  env SESSION=codellama HF_HOME="$CODELLAMA_HF_HOME" HF_HUB_CACHE="$CODELLAMA_HF_HOME/hub" TRANSFORMERS_CACHE="$CODELLAMA_HF_HOME/hub" "$RESERVER" "$@"
}

write_status() {
  local state=$1 reason=$2 tmp="${STATUS}.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE7_ST_GCGD_V21_P1_V1","status":"%s","stage":"%s","dataset":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%s,"workload_pid":%s,"tmux_session":"%s","physical_gpu":0,"resource_policy":"USER_AUTHORIZED_NO_HARD_30G_CAP","log_path":"%s","result_path":"%s","resource_reservation":"%s","test_read":false,"sports_read":false}\n' \
    "$state" "$STAGE" "$CURRENT_DATASET" "$reason" "$STARTED_AT" "$(date -Is)" "$$" "$WORKLOAD_PID" "$SESSION" "${LOG#$ROOT/}" "${OUTPUT#$ROOT/}/summary.json" "$RESERVATION" > "$tmp"
  mv "$tmp" "$STATUS"
}

config_is_enabled() {
  "$PYTHON" -c 'import json,sys;c=json.load(open(sys.argv[1]));raise SystemExit(0 if c.get("execution_enabled") is True and c.get("decision_status")=="PREREGISTERED_FROZEN_READY_TO_RUN" else 1)' "$CONFIG"
}

locks_match() {
  "$PYTHON" -c 'import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);c=json.load(open(sys.argv[2]))
for spec in c["code_lock"].values():
 p=root/spec["path"]
 if hashlib.sha256(p.read_bytes()).hexdigest()!=spec["sha256"]: raise SystemExit(1)
for spec in c["lineage_lock"].values():
 p=root/spec["path"]
 if hashlib.sha256(p.read_bytes()).hexdigest()!=spec["sha256"]: raise SystemExit(1)
for d in c["datasets"]:
 i=c["inputs"]
 if hashlib.sha256((root/i["checkpoint_root"]/d/"C1/model.pt").read_bytes()).hexdigest()!=i["expected_parent_checkpoint_sha256"][d]: raise SystemExit(1)
 if hashlib.sha256((root/i["gacr_v3_residual_root"]/d/"residual_seed2023.pt").read_bytes()).hexdigest()!=i["expected_gacr_v3_seed2023_residual_sha256"][d]: raise SystemExit(1)' "$ROOT" "$CONFIG"
}

codellama_is_running_on_gpu0() {
  local value
  value=$(reserver status 2>&1 || true)
  [[ "$value" == *"tmux session: running (codellama)"* ]] && [[ "$value" == *"gpu=0"* ]]
}

telemetry() {
  printf 'timestamp,index,memory_used_mib,memory_free_mib,utilization_gpu_percent,dataset\n' > "$OUTPUT/gpu_telemetry.csv"
  while true; do
    local row dataset
    row=$(nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits --id="$GPU" 2>/dev/null || true)
    dataset=$(sed -n '1p' "$OUTPUT/current_dataset.txt" 2>/dev/null || true)
    [[ -z "$row" ]] || printf '%s,%s\n' "$row" "$dataset" >> "$OUTPUT/gpu_telemetry.csv"
    sleep 5
  done
}

restore() {
  if codellama_is_running_on_gpu0; then RESERVATION=codellama_already_running_on_gpu0; return 0; fi
  RESERVATION=restoring_codellama_to_gpu0; STAGE=resource_restoration
  write_status restoring_resource "P1 ended; restoring CodeLlama on GPU0."
  if reserver start "$GPU"; then RESERVATION=codellama_restore_requested_on_gpu0; return 0; fi
  RESERVATION=codellama_restore_failed_on_gpu0; return 1
}

finish() {
  local scientific_rc=$? restore_rc=0
  trap - EXIT INT TERM HUP
  [[ -z "$TELEMETRY_PID" ]] || kill "$TELEMETRY_PID" >/dev/null 2>&1 || true
  restore || restore_rc=$?
  STAGE=finished
  if (( scientific_rc == 0 && restore_rc == 0 )); then
    write_status succeeded "Five-arm P1 completed for both domains; CodeLlama restored."
  elif (( scientific_rc == 0 )); then
    write_status failed_to_restore_resource "Scientific P1 completed but CodeLlama restoration failed."
  else
    write_status failed "Scientific exit=$scientific_rc; no automatic retry; CodeLlama restoration requested."
  fi
  (( scientific_rc != 0 )) && exit "$scientific_rc"
  exit "$restore_rc"
}

worker() {
  STARTED_AT=${1:?missing start timestamp}; trap finish EXIT; trap 'exit 130' INT; trap 'exit 143' TERM; trap 'exit 129' HUP
  cd "$ROOT"; mkdir -p "$OUTPUT"; STAGE=preflight
  for required in "$CONFIG" "$WORKLOAD" "$TEST"; do [[ -s "$required" ]] || { write_status blocked "Required input missing: $required"; exit 2; }; done
  config_is_enabled || { write_status blocked "P1 config is not frozen/enabled."; exit 3; }
  locks_match || { write_status blocked "P1 code/input SHA lock mismatch."; exit 7; }
  env HF_HOME="$WORKLOAD_HF_HOME" HF_HUB_CACHE="$WORKLOAD_HF_HOME" TRANSFORMERS_CACHE="$WORKLOAD_HF_HOME" "$PYTHON" -c 'from transformers import AutoTokenizer;AutoTokenizer.from_pretrained("t5-small",local_files_only=True)' || { write_status blocked "Offline t5-small preflight failed."; exit 8; }
  "$PYTHON" -m pytest -q "$TEST"
  codellama_is_running_on_gpu0 || { write_status blocked "CodeLlama must be running on GPU0 before start."; exit 4; }
  STAGE=resource_release; write_status releasing_resource "Stopping CodeLlama; user authorized workload above 30720+256 MiB."
  reserver stop; RESERVATION=released_for_experiment
  local free_mib=""
  for _ in $(seq 1 60); do
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GPU" 2>/dev/null | tr -d ' ' || true)
    [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= 40000 )) && break
    sleep 5
  done
  [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= 40000 )) || { write_status blocked "GPU0 free memory below 40000 MiB after release."; exit 5; }
  telemetry & TELEMETRY_PID=$!
  for dataset in Toys Beauty; do
    CURRENT_DATASET=$dataset; printf '%s\n' "$dataset" > "$OUTPUT/current_dataset.txt"; STAGE=scientific_workload
    timeout --signal=TERM 86400 env CUDA_VISIBLE_DEVICES="$GPU" HF_HOME="$WORKLOAD_HF_HOME" HF_HUB_CACHE="$WORKLOAD_HF_HOME" TRANSFORMERS_CACHE="$WORKLOAD_HF_HOME" "$PYTHON" "$WORKLOAD" --config "$CONFIG" --output-root "$OUTPUT" --dataset "$dataset" &
    WORKLOAD_PID=$!; write_status running "Five-arm P1 running without artificial 30 GiB sidecar cap; actual peak is audited."; wait "$WORKLOAD_PID"; WORKLOAD_PID=0
  done
  STAGE=aggregation
  "$PYTHON" -c 'import hashlib,json,pathlib,sys
r=pathlib.Path(sys.argv[1]);ds={d:json.load(open(r/d/"summary.json")) for d in ("Toys","Beauty")}
s={"experiment_id":"GRAM_PHASE7_ST_GCGD_V21_P1_V1","status":"PASS","datasets":ds,"resource_policy":"USER_AUTHORIZED_NO_HARD_30G_CAP","gpu_telemetry_sha256":hashlib.sha256((r/"gpu_telemetry.csv").read_bytes()).hexdigest(),"test_read":False,"sports_read":False,"effect_decision_automatic":False}
(r/"summary.json").write_text(json.dumps(s,ensure_ascii=False,indent=2)+"\n")' "$OUTPUT"
}

case "${1:-status}" in
  start)
    mkdir -p "$OUTPUT"; config_is_enabled || { STAGE=config_not_enabled; write_status blocked "Start refused: config not enabled."; exit 3; }
    tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists: $SESSION" >&2; exit 1; }
    STARTED_AT=$(date -Is); printf -v launch_cmd 'bash %q worker %q >> %q 2>&1' "$0" "$STARTED_AT" "$LOG"; tmux new-session -d -s "$SESSION" "$launch_cmd"
    STAGE=starting; write_status starting "Persistent ST-GCGD-v2.1 P1 session started on GPU0."; echo "started $SESSION" ;;
  worker) worker "${2:?missing start timestamp}" ;;
  status)
    tmux has-session -t "$SESSION" 2>/dev/null && echo "tmux session: running ($SESSION)" || echo "tmux session: not running ($SESSION)"
    [[ -f "$STATUS" ]] && sed -n '1,100p' "$STATUS" || echo '{"status":"not_started"}'
    [[ -f "$LOG" ]] && tail -n 50 "$LOG" || true ;;
  *) echo "usage: $0 {start|status|worker}" >&2; exit 2 ;;
esac
