#!/usr/bin/env bash
set -u -o pipefail

# Phase-13 Tier-1 Toys resolver single-trajectory convergence audit.
# Usage: bash experiment/phase13/run_tier1_resolver_toys_checkpoint_trajectory.sh {start <gpu>|status|stop}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RAW_ACTION=${1:-status}
ACTION=$RAW_ACTION
RECOVERY_MODE=false
case "$RAW_ACTION" in
  start-recovery) ACTION=start; RECOVERY_MODE=true ;;
  worker-recovery) ACTION=worker; RECOVERY_MODE=true ;;
  status-recovery) ACTION=status; RECOVERY_MODE=true ;;
  stop-recovery) ACTION=stop; RECOVERY_MODE=true ;;
esac
GPU=${2:-}
if [[ "$RECOVERY_MODE" == true ]]; then
  OUTPUT="$ROOT/artifacts/phase13/explore/tier1_resolver_toys_checkpoint_trajectory_recovery_cuda_admission"
  EXPERIMENT_ID=GRAM_PHASE13_TIER1_RESOLVER_TOYS_CHECKPOINT_TRAJECTORY_RECOVERY
  SESSION=gram_phase13_tier1_resolver_toys_checkpoint_trajectory_recovery
  TMUX_SOCKET=/tmp/gram_phase13_tier1_resolver_toys_checkpoint_trajectory_recovery.sock
  RECOVERY_JSON=true
  PARENT_ARTIFACT=artifacts/phase13/explore/tier1_resolver_toys_checkpoint_trajectory
else
  OUTPUT="$ROOT/artifacts/phase13/explore/tier1_resolver_toys_checkpoint_trajectory"
  EXPERIMENT_ID=GRAM_PHASE13_TIER1_RESOLVER_TOYS_CHECKPOINT_TRAJECTORY
  SESSION=gram_phase13_tier1_resolver_toys_checkpoint_trajectory
  TMUX_SOCKET=/tmp/gram_phase13_tier1_resolver_toys_checkpoint_trajectory.sock
  RECOVERY_JSON=false
  PARENT_ARTIFACT=""
fi
STATUS="$OUTPUT/status.json"
LOG="$OUTPUT/run.log"
TELEMETRY="$OUTPUT/gpu_telemetry.csv"
SUMMARY="$OUTPUT/summary.json"
CONFIG="$ROOT/experiment/phase13/configs/tier1_resolver_toys_checkpoint_trajectory.json"
PYTHON=${PYTHON_BIN:-/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python}
MIN_FREE_MIB=${MIN_FREE_MIB:-6144}
EXPECTED_INCREMENTAL_MIB=3072
HARD_TIMEOUT_SECONDS=${HARD_TIMEOUT_SECONDS:-5400}

STARTED_AT=""
RUNNER_PID=0
WORKLOAD_PID=0
STAGE=not_started

write_status() {
  local state=$1 reason=$2 tmp="$STATUS.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"%s","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%d,"workload_pid":%d,"tmux_session":"%s","tmux_socket":"%s","physical_gpu":%s,"minimum_free_mib":%d,"expected_incremental_gpu_mib_upper_bound":%d,"hard_timeout_seconds":%d,"resource_mode":"single_gpu_background_no_resource_changes","split":"toys_validation_development_only","checkpoint_epochs":[12,30,60,100,150],"recovery":%s,"parent_artifact":"%s","permitted_change":"CUDA admission checks and physical GPU only","test_read":false,"beauty_read":false,"sports_read":false,"automatic_next_stage":false,"automatic_retry":false,"log_path":"%s","status_path":"%s","summary_path":"%s"}\n' \
    "$EXPERIMENT_ID" "$state" "$STAGE" "$reason" "$STARTED_AT" "$(date -Is)" "$RUNNER_PID" "$WORKLOAD_PID" \
    "$SESSION" "$TMUX_SOCKET" "${GPU:--1}" "$MIN_FREE_MIB" "$EXPECTED_INCREMENTAL_MIB" "$HARD_TIMEOUT_SECONDS" \
    "$RECOVERY_JSON" "$PARENT_ARTIFACT" \
    "${LOG#$ROOT/}" "${STATUS#$ROOT/}" "${SUMMARY#$ROOT/}" > "$tmp"
  mv "$tmp" "$STATUS"
}

telemetry() {
  printf 'timestamp,index,memory_used_mib,memory_free_mib,utilization_gpu_percent\n' > "$TELEMETRY"
  while true; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits --id="$GPU" >> "$TELEMETRY" 2>/dev/null || true
    sleep 10
  done
}

worker() {
  GPU=${1:?missing gpu}
  STARTED_AT=${2:?missing start timestamp}
  RUNNER_PID=$$
  local rc=0 free_mib compute_mode telemetry_pid=0 verdict
  trap '[[ $WORKLOAD_PID -eq 0 ]] || kill -TERM "$WORKLOAD_PID" 2>/dev/null || true; [[ $telemetry_pid -eq 0 ]] || kill "$telemetry_pid" 2>/dev/null || true; STAGE=stopped; write_status stopped "Stopped by signal; no automatic retry."; exit 143' TERM INT HUP

  if [[ "$GPU" == 0 || "$GPU" == 5 ]]; then
    STAGE=blocked
    write_status blocked "GPU0 and GPU5 are user-protected and forbidden for this experiment."
    return 3
  fi

  STAGE=preflight
  write_status running "Checking frozen config, resolver tests, inputs, and GPU admission."
  bash -n "$ROOT/experiment/phase13/run_tier1_resolver_toys_checkpoint_trajectory.sh" \
    || { write_status failed "Bash syntax check failed; no automatic retry."; return 4; }
  "$PYTHON" -m py_compile \
    "$ROOT/experiment/phase13/protocol/tier1_resolver_checkpoint_trajectory.py" \
    "$ROOT/experiment/phase13/tests/test_tier1_resolver_checkpoint_trajectory.py" \
    || { write_status failed "Python syntax check failed; no automatic retry."; return 5; }
  cd "$ROOT"
  "$PYTHON" -m pytest -q \
    experiment/phase13/tests/test_route_resolve.py \
    experiment/phase13/tests/test_tier1_resolver_checkpoint_trajectory.py \
    || { write_status failed "Resolver trajectory unit tests failed; no automatic retry."; return 6; }
  for path in \
    "$CONFIG" \
    "$ROOT/GRAM/rec_datasets/Toys_cold50/user_sequence.txt" \
    "$ROOT/GRAM/rec_datasets/Toys_cold50/cold_split_meta/cold_items.txt" \
    "$ROOT/GRAM/rec_datasets/Toys_cold50/cold_split_meta/warm_items.txt" \
    "$ROOT/GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt" \
    "$ROOT/artifacts/phase13/embeddings/Toys_bge_large_en_v1_5_cls_l2.pt" \
    "$ROOT/artifacts/phase13/explore/v0_toys/predictions/20260809_085251_Toys_cold50_sequential_pred_validation.tsv" \
    "$ROOT/artifacts/phase13/explore/v1_r2_toys_p0/predictions_validation.jsonl"; do
    [[ -s "$path" ]] || { write_status failed "Required input missing: ${path#$ROOT/}"; return 7; }
  done
  [[ ! -e "$SUMMARY" && ! -e "$OUTPUT/config.json" ]] \
    || { write_status failed "Refusing to overwrite an existing trajectory artifact."; return 8; }

  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GPU" 2>/dev/null | tr -d ' ' || true)
  [[ "$free_mib" =~ ^[0-9]+$ ]] \
    || { write_status blocked "Could not read GPU${GPU} free memory; no resource changes made."; return 9; }
  (( free_mib >= MIN_FREE_MIB )) \
    || { write_status blocked "GPU${GPU} has ${free_mib} MiB free; requires ${MIN_FREE_MIB}; no resource changes made."; return 10; }
  compute_mode=$(nvidia-smi --query-gpu=compute_mode --format=csv,noheader,nounits --id="$GPU" 2>/dev/null | tr -d '[:space:]' || true)
  [[ "$compute_mode" == Default ]] \
    || { write_status blocked "GPU${GPU} compute_mode=${compute_mode:-unknown}; requires Default; no resource changes made."; return 14; }
  STAGE=cuda_admission
  write_status running "Verifying that GPU${GPU} can create a CUDA context and allocate memory."
  env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -c \
    'import torch; assert torch.cuda.is_available(); x=torch.empty(4*1024*1024,device="cuda",dtype=torch.float32); x.fill_(1); torch.cuda.synchronize(); print(f"cuda_admission_ok device={torch.cuda.get_device_name(0)} allocated={torch.cuda.memory_allocated()}")' \
    || { write_status blocked "GPU${GPU} failed live CUDA allocation admission; no scientific workload started."; return 15; }

  telemetry & telemetry_pid=$!
  STAGE=training
  write_status running "Single 150-epoch trajectory active; checkpoint audits at 12/30/60/100/150."
  timeout --signal=TERM --kill-after=30 "$HARD_TIMEOUT_SECONDS" \
    env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
      "$ROOT/experiment/phase13/protocol/tier1_resolver_checkpoint_trajectory.py" \
      --frozen-config "$CONFIG" \
      --output-dir "$OUTPUT" \
      --status-path "$STATUS" \
      --device cuda:0 &
  WORKLOAD_PID=$!
  write_status running "Trajectory workload active on GPU${GPU}; existing processes were not modified."
  wait "$WORKLOAD_PID" || rc=$?
  WORKLOAD_PID=0
  kill "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
  telemetry_pid=0
  if (( rc != 0 )); then
    STAGE=finished
    write_status failed "Resolver trajectory exited rc=${rc}; no automatic retry."
    return "$rc"
  fi
  [[ -s "$SUMMARY" ]] \
    || { STAGE=finished; write_status failed "Trajectory exited 0 without summary.json."; return 11; }
  if rg -n 'Traceback|CUDA out of memory|\bNaN\b' "$LOG" >/dev/null 2>&1; then
    STAGE=finished
    write_status failed "Postflight found Traceback/OOM/NaN markers; no automatic retry."
    return 12
  fi
  verdict=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$SUMMARY") \
    || { STAGE=finished; write_status failed "Could not read trajectory verdict."; return 13; }
  STAGE=finished
  write_status completed "$verdict"
}

case "$ACTION" in
  start)
    [[ "$GPU" =~ ^[0-7]$ ]] || { echo "usage: $0 start <gpu 0-7>" >&2; exit 2; }
    [[ "$GPU" != 0 && "$GPU" != 5 ]] || { echo "GPU0 and GPU5 are user-protected" >&2; exit 3; }
    mkdir -p "$OUTPUT"
    [[ ! -e "$STATUS" ]] || { echo "refusing to reuse existing status: $STATUS" >&2; exit 3; }
    tmux -S "$TMUX_SOCKET" has-session -t "$SESSION" 2>/dev/null \
      && { echo "session already exists: $SESSION" >&2; exit 3; }
    STARTED_AT=$(date -Is)
    STAGE=starting
    write_status starting "Background resolver trajectory session is starting on GPU${GPU}."
    worker_action=worker
    [[ "$RECOVERY_MODE" == false ]] || worker_action=worker-recovery
    printf -v launch 'bash %q %q %q %q >> %q 2>&1' "$0" "$worker_action" "$GPU" "$STARTED_AT" "$LOG"
    if ! tmux -S "$TMUX_SOCKET" new-session -d -s "$SESSION" "$launch"; then
      STAGE=launch_failed
      write_status failed "Could not create background tmux session; workload was not started."
      exit 4
    fi
    echo "started $SESSION on GPU${GPU}"
    echo "status: $STATUS"
    ;;
  worker)
    worker "${2:?missing gpu}" "${3:?missing start timestamp}"
    ;;
  status)
    [[ -f "$STATUS" ]] && sed -n '1,160p' "$STATUS" || echo '{"status":"not_started"}'
    ;;
  stop)
    if tmux -S "$TMUX_SOCKET" has-session -t "$SESSION" 2>/dev/null; then
      tmux -S "$TMUX_SOCKET" send-keys -t "$SESSION" C-c
      echo "stop requested; status file will record the outcome"
    else
      echo "session not running: $SESSION"
    fi
    ;;
  *)
    echo "usage: $0 {start <gpu>|status|stop|start-recovery <gpu>|status-recovery|stop-recovery}" >&2
    exit 2
    ;;
esac
