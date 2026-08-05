#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="$ROOT/artifacts/phase9/configs/cf0_b_toys_p1_preregistered.json"
OUTPUT="$ROOT/artifacts/phase9/cf0_b_toys_p1"
LOG="$OUTPUT/run.log"
STATUS="$OUTPUT/status.json"
LEASE_STATUS="$OUTPUT/gpu_lease.json"
SESSION=gram_phase9_cf0_b_toys_p1
GPU=6
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
TEST="$ROOT/experiment/phase9/test_gram_cf0.py"
PLAN="$ROOT/plan/第九阶段/GRAM_第九阶段_CF0协同序列增强效果验证计划.md"
LEASE_HELPER="$ROOT/experiment/gpu_memory_lease.py"
RESERVER="$ROOT/tools/run_codellama.sh"
CODELLAMA_HF_HOME=/home/jiangtangyunzhi/hf_cache
WORKLOAD_CACHE="$ROOT/.cache/huggingface"
TOTAL_LEASE_MIB=30720
EXPECTED_PEAK_MIB=12288
WORKLOAD_PID=0
LEASE_PID=""
TELEMETRY_PID=""
STARTED_AT=""
STAGE=not_started
RESERVATION=codellama_expected_on_gpu6

reserver() {
  env SESSION=codellama HF_HOME="$CODELLAMA_HF_HOME" \
    HF_HUB_CACHE="$CODELLAMA_HF_HOME/hub" \
    TRANSFORMERS_CACHE="$CODELLAMA_HF_HOME/hub" "$RESERVER" "$@"
}

write_status() {
  local state=$1 reason=$2 tmp="${STATUS}.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE9_CF0_B_TOYS_P1_FULL_V1","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%s,"workload_pid":%s,"tmux_session":"%s","physical_gpu":6,"total_gpu_lease_mib":30720,"resource_reservation":"%s","log_path":"%s","test_read":false,"sports_read":false}\n' \
    "$state" "$STAGE" "$reason" "$STARTED_AT" "$(date -Is)" "$$" "$WORKLOAD_PID" \
    "$SESSION" "$RESERVATION" "${LOG#$ROOT/}" > "$tmp"
  mv "$tmp" "$STATUS"
}

config_enabled() {
  "$PYTHON" -c 'import json,sys; c=json.load(open(sys.argv[1])); raise SystemExit(0 if c.get("execution_enabled") is True and c.get("decision_status")=="AUTHORIZED_P9_1_FULL" else 1)' "$CONFIG"
}

verify_locks() {
  "$PYTHON" -c 'import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); config=json.load(open(sys.argv[2]))
parent=config["parent_checkpoint"]; p=root/parent["path"]
assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==parent["sha256"]
for rel,expected in config["code_lock"]["files"].items():
 path=root/rel
 assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==expected, rel' "$ROOT" "$CONFIG"
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
  printf 'timestamp,index,memory_used_mib,memory_free_mib,utilization_gpu_percent\n' > "$OUTPUT/gpu_telemetry.csv"
  while true; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits --id="$GPU" >> "$OUTPUT/gpu_telemetry.csv" 2>/dev/null || true
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
  write_status restoring_resource "CF0 P9-1 ended; restoring CodeLlama on GPU6."
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
  restore || restore_rc=$?
  STAGE=finished
  if (( scientific_rc == 0 && restore_rc == 0 )); then
    write_status succeeded "CF0-B Toys P9-1 full run completed; results await researcher analysis."
  elif (( scientific_rc == 0 )); then
    write_status failed_to_restore_resource "P9-1 completed but CodeLlama restoration failed."
  else
    write_status failed "Scientific exit=$scientific_rc; no automatic retry."
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
  write_status preflight "Checking frozen code, checkpoint, offline cache and tests."
  for required in "$CONFIG" "$TEST" "$PLAN" "$LEASE_HELPER"; do
    [[ -s "$required" ]] || { write_status blocked "Missing required file: $required"; exit 2; }
  done
  config_enabled || { write_status blocked "Config is not enabled."; exit 3; }
  verify_locks || { write_status blocked "SHA256 lock mismatch."; exit 4; }
  "$PYTHON" -m pytest -q "$TEST"
  "$PYTHON" -m py_compile \
    "$ROOT/GRAM/src/main_generative_gram.py" "$ROOT/GRAM/src/model/gram.py" \
    "$ROOT/GRAM/src/runner/single_runner_gram.py"
  bash -n "$0"
  env HF_HUB_CACHE="$WORKLOAD_CACHE" TRANSFORMERS_CACHE="$WORKLOAD_CACHE/transformers" \
    TRANSFORMERS_OFFLINE=1 "$PYTHON" -c \
    'from transformers import AutoTokenizer,AutoModelForSeq2SeqLM; AutoTokenizer.from_pretrained("t5-small",local_files_only=True); AutoModelForSeq2SeqLM.from_pretrained("t5-small",local_files_only=True)'

  STAGE=codellama_pre_reservation
  write_status preparing_resource "Ensuring CodeLlama occupies GPU6 before P9-1 release."
  ensure_codellama_on_target || { write_status blocked "Could not establish CodeLlama pre-reservation on GPU6."; exit 5; }
  STAGE=resource_release
  write_status releasing_resource "Stopping CodeLlama before CF0 P9-1."
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

  STAGE=cf0_b_toys_p1_training
  cd "$ROOT/GRAM/command"
  timeout --signal=TERM 86400 env CUDA_VISIBLE_DEVICES="$GPU" \
    HF_HUB_CACHE="$WORKLOAD_CACHE" TRANSFORMERS_CACHE="$WORKLOAD_CACHE/transformers" \
    TRANSFORMERS_OFFLINE=1 "$PYTHON" ../src/main_generative_gram.py \
    --datasets Toys --distributed 0 --gpu 0 --seed 2023 --train 1 --resource_metrics 1 \
    --log_dir "$OUTPUT/gram_logs" --prediction_dir "$OUTPUT/predictions" \
    --item_prompt_max_len 128 --item_prompt all_text \
    --cf_model sasrec --id_linking 1 --max_his 20 --rec_batch_size 16 \
    --gradient_accumulation_steps 8 --rec_lr 1e-4 --rec_epochs 5 \
    --test_epoch_rec 0 --save_rec_epochs 5 --save_predictions 1 --beam_size 50 \
    --top_k_similar_item 5 --item_id_type split \
    --hierarchical_id_type hierarchy_v1_c32_l5_len32768_split \
    --rec_model_path "$ROOT/GRAM/log/Toys/1_20260720_1830/id_0_rec_30/model_rec_phase_1_epoch_30.pt" \
    --debug_train_100 0 --debug_test_100 0 --cf0_phase9 1 --cf0_arm B \
    --cf0_num_layers 2 --cf0_num_heads 4 --cf0_dropout 0.1 --cf0_loss_weight 0.1 \
    --cf0_lr 1e-3 --cf0_injection_scale 0.1 --cf0_pretrain_epochs 1 \
    --cf0_unfreeze_top_layers 2 &
  WORKLOAD_PID=$!
  write_status running "CF0-B Toys full P9-1 training/validation running."
  wait "$WORKLOAD_PID"
  WORKLOAD_PID=0
}

case "${1:-status}" in
  start)
    mkdir -p "$OUTPUT"
    config_enabled || { STAGE=config_not_enabled; write_status blocked "Start refused: config is not enabled."; exit 3; }
    tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists: $SESSION" >&2; exit 1; }
    STARTED_AT=$(date -Is)
    printf -v launch_cmd 'bash %q worker %q >> %q 2>&1' "$0" "$STARTED_AT" "$LOG"
    tmux new-session -d -s "$SESSION" "$launch_cmd"
    STAGE=starting
    write_status starting "Persistent CF0-B Toys full P9-1 session started."
    echo "started $SESSION"
    ;;
  worker) worker "${2:?missing start timestamp}" ;;
  status)
    tmux has-session -t "$SESSION" 2>/dev/null && echo "tmux session: running ($SESSION)" || echo "tmux session: not running ($SESSION)"
    [[ -f "$STATUS" ]] && sed -n '1,100p' "$STATUS" || echo '{"status":"not_started"}'
    [[ -f "$LOG" ]] && tail -n 50 "$LOG" || true
    ;;
  stop)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      pane_pid=$(tmux list-panes -t "$SESSION" -F '#{pane_pid}' | head -n 1)
      kill -TERM "$pane_pid"
      echo "stop requested for $SESSION"
    else
      echo "session not running: $SESSION"
    fi
    ;;
  *) echo "usage: $0 {start|status|stop|worker}" >&2; exit 2 ;;
esac
