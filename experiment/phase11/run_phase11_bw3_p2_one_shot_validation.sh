#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="$ROOT/artifacts/phase11/configs/bw3_p2_one_shot_validation_preregistered.json"
OUT="$ROOT/artifacts/phase11/bw3_p2_one_shot_validation"
SCIENTIFIC_OUT="$OUT/scientific"
STAGING_OUT="$OUT/.scientific_complete_unrevealed"
LOG="$OUT/run.log"
STATUS="$OUT/status.json"
GPU_TELEMETRY="$OUT/gpu_telemetry.csv"
CPU_TELEMETRY="$OUT/cpu_telemetry.csv"
SESSION=gram_phase11_bw3_p2_one_shot_validation
GPU=6
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
EVALUATOR="$ROOT/experiment/phase11/eval_bw3_p2_one_shot_validation.py"
TEST="$ROOT/experiment/phase11/test_bw3_p2_one_shot_validation.py"
PLAN="$ROOT/plan/第十一阶段/GRAM_第十一阶段_BW3-P2_Listwise扩展准入独立验证计划.md"
RESERVER=/home/jiangtangyunzhi/projects/UnitTest/tools/run_codellama.sh
CODELLAMA_ROOT=/home/jiangtangyunzhi/projects/UnitTest/experiments/codellama
CODELLAMA_STATUS="$CODELLAMA_ROOT/status.txt"
CODELLAMA_LATEST="$CODELLAMA_ROOT/latest_cycle.json"
CODELLAMA_GPU_FILE="$CODELLAMA_ROOT/gpu.txt"
CODELLAMA_PID_FILE="$CODELLAMA_ROOT/runner.pid"
CODELLAMA_HF_HOME=/home/jiangtangyunzhi/hf_cache
HARD_TIMEOUT_SECONDS=1800
WORKLOAD_PID=0
TELEMETRY_PID=0
STARTED_AT=""
STAGE=not_started
SCIENTIFIC_STATUS=not_started
SCIENTIFIC_REASON="P2-1 has not started."
RESOURCE_STATUS=codellama_not_checked
RESOURCE_AUDIT=not_evaluated
VALIDATION_ACCESS_STARTED=false
VALIDATION_CONSUMED=false
RESULTS_REVEALED=false
VALIDATION_USERS_PROCESSED=0

reserver() {
  env SESSION=codellama HF_HOME="$CODELLAMA_HF_HOME" \
    HF_HUB_CACHE="$CODELLAMA_HF_HOME/hub" TRANSFORMERS_CACHE="$CODELLAMA_HF_HOME/hub" \
    "$RESERVER" "$@"
}

sync_validation_state() {
  [[ -f "$STATUS" ]] || return 0
  VALIDATION_ACCESS_STARTED=$("$PYTHON" -c 'import json,sys; print(str(json.load(open(sys.argv[1])).get("validation_access_started",False)).lower())' "$STATUS" 2>/dev/null || echo false)
  VALIDATION_CONSUMED=$("$PYTHON" -c 'import json,sys; print(str(json.load(open(sys.argv[1])).get("validation_consumed",False)).lower())' "$STATUS" 2>/dev/null || echo false)
  RESULTS_REVEALED=$("$PYTHON" -c 'import json,sys; print(str(json.load(open(sys.argv[1])).get("results_revealed",False)).lower())' "$STATUS" 2>/dev/null || echo false)
  VALIDATION_USERS_PROCESSED=$("$PYTHON" -c 'import json,sys; print(int(json.load(open(sys.argv[1])).get("validation_users_processed",0)))' "$STATUS" 2>/dev/null || echo 0)
}

write_status() {
  local state=$1 reason=$2 tmp="${STATUS}.tmp.$$"
  mkdir -p "$OUT"
  "$PYTHON" -c 'import json,pathlib,sys
path=pathlib.Path(sys.argv[1]); payload={
"experiment_id":"GRAM_PHASE11_BW3_P2_LISTWISE_ADMISSION_ONE_SHOT_VALIDATION_V1",
"status":sys.argv[2],"stage":sys.argv[3],"reason":sys.argv[4],
"scientific_status":sys.argv[5],"scientific_reason":sys.argv[6],
"resource_status":sys.argv[7],"resource_audit":sys.argv[8],"started_at":sys.argv[9],"updated_at":sys.argv[10],
"runner_pid":int(sys.argv[11]),"workload_pid":int(sys.argv[12]),"tmux_session":sys.argv[13],
"compute_device":"cpu","physical_gpu_observed":6,"codellama_required_reserve_mib":30720,
"sidecar_started":False,"validation_access_started":sys.argv[14]=="true",
"validation_consumed":sys.argv[15]=="true","results_revealed":sys.argv[16]=="true",
"validation_users_expected":1024,"validation_users_processed":int(sys.argv[17]),
"test_read":False,"sports_read":False,"automatic_retry":False,
"log_path":"artifacts/phase11/bw3_p2_one_shot_validation/run.log"}
path.write_text(json.dumps(payload,ensure_ascii=False)+"\n")' \
    "$tmp" "$state" "$STAGE" "$reason" "$SCIENTIFIC_STATUS" "$SCIENTIFIC_REASON" \
    "$RESOURCE_STATUS" "$RESOURCE_AUDIT" "$STARTED_AT" "$(date -Is)" "$$" "$WORKLOAD_PID" "$SESSION" \
    "$VALIDATION_ACCESS_STARTED" "$VALIDATION_CONSUMED" "$RESULTS_REVEALED" "$VALIDATION_USERS_PROCESSED"
  mv "$tmp" "$STATUS"
}

config_enabled() {
  "$PYTHON" -c 'import json,sys
c=json.load(open(sys.argv[1])); raise SystemExit(0 if c.get("execution_enabled") is True and c.get("decision_status")=="AUTHORIZED_P2_ONE_SHOT_FORMAL_RUN" else 1)' "$CONFIG"
}

verify_locks() {
  "$PYTHON" -c 'import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); config=json.load(open(sys.argv[2]))
for section in ("code_lock","input_lock"):
 for rel,expected in config[section]["files"].items():
  path=root/rel
  if not path.is_file(): raise SystemExit(f"missing locked file: {rel}")
  actual=hashlib.sha256(path.read_bytes()).hexdigest()
  if actual!=expected: raise SystemExit(f"SHA mismatch: {rel}: {actual} != {expected}")' "$ROOT" "$CONFIG"
}

codellama_actual_reserved() {
  "$PYTHON" -c 'import json,pathlib,sys
p=pathlib.Path(sys.argv[1])
try: print(json.load(p.open()).get("gpu",{}).get("memory_reserved_mib",0))
except Exception: print(0)' "$CODELLAMA_LATEST"
}

codellama_live_used() {
  local pid
  [[ -s "$CODELLAMA_PID_FILE" ]] || { echo 0; return; }
  pid=$(tr -d '[:space:]' < "$CODELLAMA_PID_FILE")
  [[ "$pid" =~ ^[0-9]+$ ]] || { echo 0; return; }
  { nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits --id="$GPU" 2>/dev/null || true; } \
    | awk -F, -v wanted="$pid" '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($1==wanted) {print $2; found=1}} END {if (!found) print 0}'
}

codellama_on_target() {
  local gpu state reserve actual live_used
  tmux has-session -t codellama 2>/dev/null || return 1
  [[ -s "$CODELLAMA_GPU_FILE" && -s "$CODELLAMA_STATUS" ]] || return 1
  gpu=$(tr -d '[:space:]' < "$CODELLAMA_GPU_FILE")
  state=$(sed -n '1p' "$CODELLAMA_STATUS")
  reserve=$(sed -n 's/.* reserve_mib=\([0-9][0-9]*\).*/\1/p' "$CODELLAMA_STATUS")
  actual=$(codellama_actual_reserved)
  live_used=$(codellama_live_used)
  [[ "$gpu" == "$GPU" && "$state" == *" state=running "* ]] || return 1
  [[ "$reserve" =~ ^[0-9]+$ ]] && (( reserve >= 30720 )) || return 1
  "$PYTHON" -c 'import sys; raise SystemExit(0 if float(sys.argv[1])>=30720 and float(sys.argv[2])>=30720 else 1)' "$actual" "$live_used"
}

ensure_codellama() {
  if codellama_on_target; then
    RESOURCE_STATUS=codellama_preserved_running_on_gpu6
    return 0
  fi
  if tmux has-session -t codellama 2>/dev/null; then
    RESOURCE_STATUS=blocked_codellama_running_but_target_or_reserve_mismatch
    return 1
  fi
  RESOURCE_STATUS=starting_codellama_on_gpu6
  reserver start "$GPU"
  for _ in $(seq 1 180); do
    if codellama_on_target; then
      RESOURCE_STATUS=codellama_started_running_on_gpu6
      return 0
    fi
    sleep 5
  done
  RESOURCE_STATUS=blocked_codellama_not_ready
  return 1
}

telemetry() {
  printf 'timestamp,index,memory_used_mib,memory_free_mib,utilization_gpu_percent,codellama_tmux,codellama_state,codellama_reserved_mib,codellama_live_used_mib,experiment_gpu_pid_observed,stage\n' > "$GPU_TELEMETRY"
  printf 'timestamp,workload_pid,workload_alive,rss_kib,elapsed_seconds,stage\n' > "$CPU_TELEMETRY"
  local begin now gpu_row tmux_alive controller_state reserved live_used gpu_pids observed alive rss child_pid observed_pid current_stage
  begin=$(date +%s)
  while true; do
    now=$(date -Is)
    current_stage=$("$PYTHON" -c 'import json,sys
try: print(json.load(open(sys.argv[1])).get("stage","unknown"))
except Exception: print("unknown")' "$STATUS")
    gpu_row=$(nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits --id="$GPU" 2>/dev/null | tr -d ' ' || true)
    tmux_alive=0; tmux has-session -t codellama 2>/dev/null && tmux_alive=1
    controller_state=$(sed -n 's/.* state=\([^ ]*\).*/\1/p' "$CODELLAMA_STATUS" 2>/dev/null || true)
    reserved=$(codellama_actual_reserved); live_used=$(codellama_live_used)
    gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits --id="$GPU" 2>/dev/null | tr '\n' ' ' || true)
    child_pid=$(pgrep -P "$WORKLOAD_PID" 2>/dev/null | head -n 1 || true); observed_pid=${child_pid:-$WORKLOAD_PID}
    observed=0; [[ "$observed_pid" -gt 0 && " $gpu_pids " == *" $observed_pid "* ]] && observed=1
    if [[ "$gpu_row" == *,*,*,* ]]; then
      printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "$now" "$gpu_row" "$tmux_alive" "${controller_state:-unknown}" "$reserved" "$live_used" "$observed" "$current_stage" >> "$GPU_TELEMETRY"
    else
      printf '%s,%s,,,,%s,%s,%s,%s,%s,%s\n' "$now" "$GPU" "$tmux_alive" "${controller_state:-unknown}" "$reserved" "$live_used" "$observed" "$current_stage" >> "$GPU_TELEMETRY"
    fi
    alive=0; rss=0
    if (( observed_pid > 0 )) && kill -0 "$observed_pid" 2>/dev/null; then
      alive=1; rss=$(ps -o rss= -p "$observed_pid" 2>/dev/null | tr -d ' ' || true)
    fi
    printf '%s,%s,%s,%s,%s,%s\n' "$now" "$observed_pid" "$alive" "${rss:-0}" "$(( $(date +%s) - begin ))" "$current_stage" >> "$CPU_TELEMETRY"
    sleep 5
  done
}

audit_telemetry() {
  "$PYTHON" -c 'import csv,sys
rows=list(csv.DictReader(open(sys.argv[1]))); ok=bool(rows)
for row in rows:
 ok &= row["codellama_tmux"]=="1" and row["codellama_state"]=="running"
 try: ok &= float(row["codellama_reserved_mib"])>=30720 and float(row["codellama_live_used_mib"])>=30720
 except Exception: ok=False
 ok &= row["experiment_gpu_pid_observed"]=="0"
raise SystemExit(0 if ok else 1)' "$GPU_TELEMETRY"
}

restore_if_needed() {
  if codellama_on_target; then RESOURCE_STATUS=preserved_running; return 0; fi
  if tmux has-session -t codellama 2>/dev/null; then RESOURCE_STATUS=failed_to_restore_resource_mismatched_live_session; return 1; fi
  RESOURCE_STATUS=restoring_codellama_to_gpu6
  reserver start "$GPU" || { RESOURCE_STATUS=failed_to_restore_resource_start_request; return 1; }
  for _ in $(seq 1 180); do
    if codellama_on_target; then RESOURCE_STATUS=restored; return 0; fi
    sleep 5
  done
  RESOURCE_STATUS=failed_to_restore_resource_timeout
  return 1
}

finish() {
  local scientific_rc=$? resource_rc=0
  trap - EXIT INT TERM HUP
  if (( WORKLOAD_PID > 0 )) && kill -0 "$WORKLOAD_PID" 2>/dev/null; then
    kill -TERM "$WORKLOAD_PID" 2>/dev/null || true; wait "$WORKLOAD_PID" 2>/dev/null || true
  fi
  if (( TELEMETRY_PID > 0 )); then kill "$TELEMETRY_PID" >/dev/null 2>&1 || true; wait "$TELEMETRY_PID" 2>/dev/null || true; fi
  sync_validation_state
  if (( scientific_rc != 0 )); then
    SCIENTIFIC_STATUS=failed
    if [[ "$VALIDATION_CONSUMED" == true ]]; then
      SCIENTIFIC_REASON="P2 exited ${scientific_rc} after validation consumption; strict confirmatory retry is forbidden."
    else
      SCIENTIFIC_REASON="P2 preflight exited ${scientific_rc} before validation consumption; no automatic retry."
    fi
  fi
  restore_if_needed || resource_rc=$?
  STAGE=finished
  if (( resource_rc != 0 )); then
    write_status failed_to_restore_resource "$SCIENTIFIC_REASON Resource restoration failed independently."
  elif [[ "$SCIENTIFIC_STATUS" == passed* || "$SCIENTIFIC_STATUS" == failed_p2_gate ]]; then
    write_status completed "$SCIENTIFIC_REASON"
  elif [[ "$SCIENTIFIC_STATUS" == blocked* ]]; then
    write_status blocked "$SCIENTIFIC_REASON"
  else
    write_status failed "$SCIENTIFIC_REASON"
  fi
  (( scientific_rc != 0 )) && exit "$scientific_rc"
  exit "$resource_rc"
}

worker() {
  STARTED_AT=${1:?missing start timestamp}
  trap finish EXIT; trap 'exit 130' INT; trap 'exit 143' TERM; trap 'exit 129' HUP
  cd "$ROOT"
  STAGE=preflight
  write_status preflight "Checking frozen P2 code, inputs, CPU path and synthetic tests without semantic validation access."
  for required in "$CONFIG" "$EVALUATOR" "$TEST" "$PLAN" "$RESERVER"; do
    [[ -s "$required" ]] || { SCIENTIFIC_REASON="Missing required file: $required"; exit 2; }
  done
  config_enabled || { SCIENTIFIC_STATUS=blocked_not_authorized; SCIENTIFIC_REASON="Frozen config is not authorized."; exit 3; }
  verify_locks || { SCIENTIFIC_STATUS=blocked_input_lock_mismatch; SCIENTIFIC_REASON="Frozen SHA lock verification failed before validation access."; exit 4; }
  CUDA_VISIBLE_DEVICES='' "$PYTHON" -m pytest -q "$TEST"
  CUDA_VISIBLE_DEVICES='' "$PYTHON" -m py_compile "$EVALUATOR"
  bash -n "$0"

  STAGE=codellama_precheck
  write_status preparing_resource "Confirming CodeLlama 30 GiB reservation remains on GPU6 before validation access."
  ensure_codellama || { SCIENTIFIC_STATUS=blocked_codellama_not_ready; SCIENTIFIC_REASON="CodeLlama pre-reservation was not ready before validation access."; exit 5; }

  STAGE=starting_one_shot_validation
  SCIENTIFIC_STATUS=running
  SCIENTIFIC_REASON="Starting CPU-only two-domain atomic P2; evaluator will mark validation consumed before semantic access."
  write_status running "$SCIENTIFIC_REASON"
  timeout --signal=TERM --kill-after=10 "$HARD_TIMEOUT_SECONDS" \
    env CUDA_VISIBLE_DEVICES='' PYTHONHASHSEED=2023 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "$PYTHON" "$EVALUATOR" --config "$CONFIG" --output-dir "$STAGING_OUT" --status-path "$STATUS" &
  WORKLOAD_PID=$!
  telemetry & TELEMETRY_PID=$!
  wait "$WORKLOAD_PID"
  WORKLOAD_PID=0
  kill "$TELEMETRY_PID" >/dev/null 2>&1 || true; wait "$TELEMETRY_PID" 2>/dev/null || true; TELEMETRY_PID=0
  sync_validation_state
  if audit_telemetry; then
    RESOURCE_AUDIT=passed
  else
    RESOURCE_AUDIT=failed
    SCIENTIFIC_STATUS=failed_resource_protocol
    SCIENTIFIC_REASON="Two-domain P2 evaluation completed, but the full-run resource audit failed; results remain unrevealed and P2 is ineligible."
    exit 9
  fi
  local gate
  gate=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["p2_gate"]["status"])' "$STAGING_OUT/summary.json")
  "$PYTHON" -c 'import json,os,pathlib,sys
p=pathlib.Path(sys.argv[1]); d=json.load(p.open()); scientific=d["p2_gate"]["status"]
d["resource_audit"]="passed"; d["results_revealed"]=True
d["p2_gate"]["resource_and_access_protocol_passed"]=True
d["p2_gate"]["status"]="passed_independent_validation_eligible_for_next_plan_discussion" if scientific=="passed_scientific_gate_awaiting_resource_audit" else "failed"
t=p.with_name(f".{p.name}.tmp.{os.getpid()}"); t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n"); os.replace(t,p)' "$STAGING_OUT/summary.json"
  mv "$STAGING_OUT" "$SCIENTIFIC_OUT"
  RESULTS_REVEALED=true
  VALIDATION_ACCESS_STARTED=true
  VALIDATION_CONSUMED=true
  VALIDATION_USERS_PROCESSED=1024
  STAGE=results_atomically_revealed
  if [[ "$gate" == passed_scientific_gate_awaiting_resource_audit ]]; then
    SCIENTIFIC_STATUS=passed_independent_validation
    SCIENTIFIC_REASON="P2 scientific gate passed; no successor is authorized or started."
  else
    SCIENTIFIC_STATUS=failed_p2_gate
    SCIENTIFIC_REASON="P2 completed but did not pass the preregistered gate; no retry or successor is authorized."
  fi
  write_status finalizing "$SCIENTIFIC_REASON"
}

case "${1:-status}" in
  start)
    mkdir -p "$OUT"
    config_enabled || { STAGE=config_not_enabled; SCIENTIFIC_STATUS=blocked_not_authorized; SCIENTIFIC_REASON="Start refused: P2 config is disabled and not authorized."; write_status blocked "$SCIENTIFIC_REASON"; exit 3; }
    tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists: $SESSION" >&2; exit 1; }
    [[ ! -e "$SCIENTIFIC_OUT" && ! -e "$STAGING_OUT" ]] || { echo "scientific or unrevealed staging output already exists; refusing overwrite" >&2; exit 8; }
    STARTED_AT=$(date -Is)
    printf -v launch_cmd 'bash %q worker %q >> %q 2>&1' "$0" "$STARTED_AT" "$LOG"
    tmux new-session -d -s "$SESSION" "$launch_cmd"
    STAGE=starting; SCIENTIFIC_STATUS=starting; SCIENTIFIC_REASON="Persistent P2 session started."
    write_status starting "$SCIENTIFIC_REASON"
    echo "started $SESSION"
    ;;
  worker) worker "${2:?missing start timestamp}" ;;
  status)
    tmux has-session -t "$SESSION" 2>/dev/null && echo "tmux session: running ($SESSION)" || echo "tmux session: not running ($SESSION)"
    [[ -f "$STATUS" ]] && sed -n '1,100p' "$STATUS" || echo '{"status":"not_started","execution_enabled":false,"validation_consumed":false}'
    echo "CodeLlama controller:"
    reserver status 2>&1 | sed -n '1,8p' || true
    [[ -f "$GPU_TELEMETRY" ]] && { echo "GPU telemetry tail:"; tail -n 4 "$GPU_TELEMETRY"; }
    [[ -f "$CPU_TELEMETRY" ]] && { echo "CPU telemetry tail:"; tail -n 4 "$CPU_TELEMETRY"; }
    [[ -f "$LOG" ]] && { echo "log tail:"; tail -n 20 "$LOG"; }
    true
    ;;
  stop)
    if [[ -f "$STATUS" ]]; then
      runner_pid=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("runner_pid",0))' "$STATUS" 2>/dev/null || echo 0)
      if [[ "$runner_pid" =~ ^[0-9]+$ ]] && (( runner_pid > 0 )) && kill -0 "$runner_pid" 2>/dev/null; then
        kill -TERM "$runner_pid"
        echo "TERM sent to P2 runner $runner_pid; validation may already be consumed; CodeLlama was not stopped."
        exit 0
      fi
    fi
    echo "P2 runner is not active; CodeLlama was not changed."
    ;;
  *) echo "usage: $0 {start|status|stop}" >&2; exit 2 ;;
esac
