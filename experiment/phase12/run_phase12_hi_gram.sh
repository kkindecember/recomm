#!/usr/bin/env bash
# Phase-12 HI-GRAM runner (exploration mode, GPU7 single-card sequential).
#
# Sub-experiments (each has its own tmux session + output dir):
#   smoke_beauty      Beauty 1 epoch smoke, from t5-small cold start
#   beauty_v1         Beauty full 30 epochs, HI-GRAM v1 default config
#   toys_v1           Toys   full 30 epochs, HI-GRAM v1 default config
#
# Usage:
#   bash experiment/phase12/run_phase12_hi_gram.sh start <sub>
#   bash experiment/phase12/run_phase12_hi_gram.sh status <sub>
#   bash experiment/phase12/run_phase12_hi_gram.sh stop <sub>
#
# Protocol (inherited from phase 9):
#   - CodeLlama must be occupying target GPU before start; runner stops it,
#     runs workload, then restores it.
#   - 30 GiB total lease held via gpu_memory_lease.py sidecar.
#   - No auto-retry. Any failure blocks; researcher must diagnose.
#   - Test set never read; validation only.
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
LEASE_HELPER="$ROOT/experiment/gpu_memory_lease.py"

# CodeLlama tool lives in the recomm project so its status is visible in
# /mnt/18T/.../recomm/.runtime/codellama/status.txt (user directive 2026-08-05,
# supersedes an earlier note that pointed at the UnitTest path).
RESERVER=/mnt/18T/jiangtangyunzhi/projects/recomm/tools/run_codellama.sh
CODELLAMA_HF_HOME=/home/jiangtangyunzhi/hf_cache
WORKLOAD_CACHE="$ROOT/.cache/huggingface"

GPU=6
TOTAL_LEASE_MIB=30720
EXPECTED_PEAK_MIB=14336   # HI-GRAM ~8M extra params; expected peak <14 GiB
HI_GRAM_TEST="$ROOT/experiment/phase12/test_hi_gram_encoder.py"

ACTION=${1:-status}
# SUB is passed as $2 for start/status/stop; the worker path receives SUB via env
# HI_GRAM_SUB (set at start time when spawning the tmux command) so as not to
# confuse it with the STARTED_AT timestamp positional argument.
if [[ "$ACTION" == "worker" ]]; then
  SUB="${HI_GRAM_SUB:-}"
else
  SUB=${2:-}
fi

if [[ -z "$SUB" && "$ACTION" != "help" && "$ACTION" != "--help" && "$ACTION" != "-h" ]]; then
  echo "usage: $0 {start|status|stop} <sub>" >&2
  echo "  sub ∈ {smoke_beauty, beauty_v1, toys_v1}" >&2
  exit 2
fi

# --- per-sub configuration ---
DATASET=""
EPOCHS=""
CLUSTER=""
ID_LEN=""
NUM_CF=""
BEAM_SIZE=""
DEBUG_TRAIN_100=""
DEBUG_TEST_100=""
TEST_EPOCH_REC=""
SAVE_REC_EPOCHS=""
case "${SUB:-}" in
  smoke_beauty)
    DATASET=Beauty; EPOCHS=1; CLUSTER=128; ID_LEN=7; NUM_CF=10; BEAM_SIZE=50
    DEBUG_TRAIN_100=1; DEBUG_TEST_100=1; TEST_EPOCH_REC=0; SAVE_REC_EPOCHS=1
    ;;
  beauty_v1)
    DATASET=Beauty; EPOCHS=30; CLUSTER=128; ID_LEN=7; NUM_CF=10; BEAM_SIZE=50
    DEBUG_TRAIN_100=0; DEBUG_TEST_100=0; TEST_EPOCH_REC=5; SAVE_REC_EPOCHS=5
    ;;
  toys_v1)
    DATASET=Toys; EPOCHS=30; CLUSTER=32; ID_LEN=5; NUM_CF=5; BEAM_SIZE=50
    DEBUG_TRAIN_100=0; DEBUG_TEST_100=0; TEST_EPOCH_REC=5; SAVE_REC_EPOCHS=5
    ;;
  "")
    :  # allowed for worker/help; error handled above
    ;;
  *)
    echo "unknown sub: $SUB" >&2; exit 2 ;;
esac

SESSION="gram_phase12_hi_gram_${SUB}"
OUTPUT="$ROOT/artifacts/phase12/hi_gram/${SUB}"
LOG="$OUTPUT/run.log"
STATUS="$OUTPUT/status.json"
LEASE_STATUS="$OUTPUT/gpu_lease.json"
TELEMETRY_CSV="$OUTPUT/gpu_telemetry.csv"

WORKLOAD_PID=0
LEASE_PID=""
TELEMETRY_PID=""
STARTED_AT=""
STAGE=not_started
RESERVATION=codellama_expected_on_gpu${GPU}

reserver() {
  env SESSION=codellama HF_HOME="$CODELLAMA_HF_HOME" \
    HF_HUB_CACHE="$CODELLAMA_HF_HOME/hub" \
    TRANSFORMERS_CACHE="$CODELLAMA_HF_HOME/hub" "$RESERVER" "$@"
}

write_status() {
  local state=$1 reason=$2 tmp="${STATUS}.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE12_HI_GRAM_%s_V1","sub":"%s","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%s,"workload_pid":%s,"tmux_session":"%s","physical_gpu":%d,"total_gpu_lease_mib":%d,"resource_reservation":"%s","log_path":"%s","test_read":false,"sports_read":false,"yelp_read":false,"hi_gram_enabled":true}\n' \
    "${SUB^^}" "$SUB" "$state" "$STAGE" "$reason" "$STARTED_AT" "$(date -Is)" "$$" "$WORKLOAD_PID" \
    "$SESSION" "$GPU" "$TOTAL_LEASE_MIB" "$RESERVATION" "${LOG#$ROOT/}" > "$tmp"
  mv "$tmp" "$STATUS"
}

codellama_on_target() {
  local value
  value=$(reserver status 2>&1 || true)
  # The UnitTest tool prints "tmux session: running (codellama)".
  # We additionally verify via nvidia-smi that CodeLlama actually holds GPU7.
  [[ "$value" == *"tmux session: running (codellama)"* ]] || return 1
  # Check that a python process from unittest-transformers env is on GPU7
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader --id="$GPU" 2>/dev/null | \
    grep -q "unittest-transformers" || return 1
}

ensure_codellama_on_target() {
  if codellama_on_target; then
    RESERVATION=codellama_confirmed_on_gpu${GPU}
    return 0
  fi
  # Try to start CodeLlama on GPU7 (will fail if it's already running elsewhere;
  # user should manually migrate in that case).
  reserver start "$GPU" >/dev/null 2>&1 || true
  for _ in $(seq 1 180); do
    if codellama_on_target; then
      RESERVATION=codellama_confirmed_on_gpu${GPU}
      return 0
    fi
    sleep 5
  done
  RESERVATION=codellama_prepare_failed_on_gpu${GPU}
  return 1
}

telemetry() {
  printf 'timestamp,index,memory_used_mib,memory_free_mib,utilization_gpu_percent\n' > "$TELEMETRY_CSV"
  while true; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits --id="$GPU" >> "$TELEMETRY_CSV" 2>/dev/null || true
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
    RESERVATION=codellama_already_running_on_gpu${GPU}
    return 0
  fi
  STAGE=resource_restoration
  RESERVATION=restoring_codellama_to_gpu${GPU}
  write_status restoring_resource "Experiment ended; restoring CodeLlama on GPU${GPU}."
  reserver start "$GPU" >/dev/null 2>&1 || { RESERVATION=restore_request_failed_on_gpu${GPU}; return 1; }
  for _ in $(seq 1 180); do
    if codellama_on_target; then
      RESERVATION=restored_on_gpu${GPU}
      return 0
    fi
    sleep 5
  done
  RESERVATION=restore_failed_on_gpu${GPU}
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
    write_status succeeded "HI-GRAM $SUB completed; results await researcher analysis."
  elif (( scientific_rc == 0 )); then
    write_status failed_to_restore_resource "$SUB completed but CodeLlama restoration failed."
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
  write_status preflight "Running CPU unit tests and syntax checks."
  "$PYTHON" -m unittest "$HI_GRAM_TEST" -v 2>&1 | tee "$OUTPUT/cpu_unit_tests.log"
  local unit_rc=${PIPESTATUS[0]}
  (( unit_rc == 0 )) || { write_status blocked "HI-GRAM CPU unit tests failed (rc=$unit_rc)."; exit 3; }
  "$PYTHON" -m py_compile \
    "$ROOT/GRAM/src/main_generative_gram.py" \
    "$ROOT/GRAM/src/arguments.py" \
    "$ROOT/GRAM/src/model/gram.py" || { write_status blocked "Python syntax check failed."; exit 4; }
  bash -n "$0" || { write_status blocked "Runner shell syntax invalid."; exit 5; }

  STAGE=codellama_pre_reservation
  write_status preparing_resource "Ensuring CodeLlama occupies GPU${GPU}."
  ensure_codellama_on_target || { write_status blocked "Could not confirm CodeLlama on GPU${GPU}."; exit 6; }

  STAGE=resource_release
  write_status releasing_resource "Stopping CodeLlama before HI-GRAM $SUB."
  reserver stop >/dev/null 2>&1 || true
  RESERVATION=released_for_experiment

  STAGE=gpu_memory_gate
  write_status waiting_for_gpu "Waiting for ≥ ${TOTAL_LEASE_MIB} MiB free on GPU${GPU}."
  local free_mib=""
  for _ in $(seq 1 120); do
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GPU" 2>/dev/null | tr -d ' ' || true)
    [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= TOTAL_LEASE_MIB )) && break
    sleep 5
  done
  [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= TOTAL_LEASE_MIB )) || { write_status blocked "GPU${GPU} admission failed: ${free_mib:-unknown} MiB free."; exit 7; }

  STAGE=memory_lease
  write_status leasing "Starting 30 GiB total-lease sidecar (expected workload peak ${EXPECTED_PEAK_MIB} MiB)."
  "$PYTHON" "$LEASE_HELPER" --gpu "$GPU" --total-lease-mib "$TOTAL_LEASE_MIB" \
    --expected-workload-peak-mib "$EXPECTED_PEAK_MIB" --status-path "$LEASE_STATUS" &
  LEASE_PID=$!
  for _ in $(seq 1 30); do
    [[ -s "$LEASE_STATUS" ]] && grep -q '"state": "holding"' "$LEASE_STATUS" && break
    sleep 1
  done
  [[ -s "$LEASE_STATUS" ]] && grep -q '"state": "holding"' "$LEASE_STATUS" || { write_status blocked "GPU lease sidecar did not hold."; exit 8; }

  STAGE=telemetry
  telemetry & TELEMETRY_PID=$!

  STAGE=hi_gram_training_${SUB}
  write_status running "HI-GRAM $SUB training on GPU${GPU}."
  local item_id="hierarchy_v1_c${CLUSTER}_l${ID_LEN}_len32768_split"
  cd "$ROOT/GRAM/command"
  timeout --signal=TERM 86400 env CUDA_VISIBLE_DEVICES="$GPU" \
    HF_HUB_CACHE="$WORKLOAD_CACHE" TRANSFORMERS_CACHE="$WORKLOAD_CACHE/transformers" \
    TRANSFORMERS_OFFLINE=1 "$PYTHON" ../src/main_generative_gram.py \
    --datasets "$DATASET" \
    --distributed 0 --gpu 0 --seed 2023 --train 1 --resource_metrics 1 \
    --log_dir "$OUTPUT/gram_logs" --prediction_dir "$OUTPUT/predictions" \
    --item_prompt_max_len 128 --item_prompt all_text \
    --cf_model sasrec --id_linking 1 --max_his 20 \
    --rec_batch_size 16 --gradient_accumulation_steps 8 \
    --rec_lr 1e-3 --rec_epochs "$EPOCHS" \
    --test_epoch_rec "$TEST_EPOCH_REC" --save_rec_epochs "$SAVE_REC_EPOCHS" \
    --save_predictions 1 --beam_size "$BEAM_SIZE" \
    --top_k_similar_item "$NUM_CF" --item_id_type split \
    --hierarchical_id_type "$item_id" \
    --debug_train_100 "$DEBUG_TRAIN_100" --debug_test_100 "$DEBUG_TEST_100" \
    --cf0_arm A --cf0_phase9 0 \
    --hi_gram_enabled 1 \
    --hi_gram_local_window 5 \
    --hi_gram_local_layers 2 --hi_gram_global_layers 2 \
    --hi_gram_num_heads 4 --hi_gram_dropout 0.1 \
    --hi_gram_fusion_scale_init 0.1 \
    --hi_gram_include_user_prompt 0 &
  WORKLOAD_PID=$!
  wait "$WORKLOAD_PID"
  WORKLOAD_PID=0
}

case "$ACTION" in
  start)
    mkdir -p "$OUTPUT"
    tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists: $SESSION" >&2; exit 1; }
    STARTED_AT=$(date -Is)
    printf -v launch_cmd 'HI_GRAM_SUB=%q bash %q worker %q >> %q 2>&1' "$SUB" "$0" "$STARTED_AT" "$LOG"
    tmux new-session -d -s "$SESSION" "$launch_cmd"
    STAGE=starting
    write_status starting "Persistent HI-GRAM $SUB session started; runner will drive CodeLlama and lease."
    echo "started $SESSION"
    ;;
  worker)
    # Called from tmux — receives STARTED_AT as $2 (the second positional after "worker").
    # SUB is provided via HI_GRAM_SUB env var (set in launch_cmd above).
    worker "${2:?missing start timestamp}"
    ;;
  status)
    tmux has-session -t "$SESSION" 2>/dev/null && echo "tmux session: running ($SESSION)" || echo "tmux session: not running ($SESSION)"
    [[ -f "$STATUS" ]] && sed -n '1,100p' "$STATUS" || echo '{"status":"not_started"}'
    [[ -f "$LEASE_STATUS" ]] && echo "--- lease ---" && sed -n '1,20p' "$LEASE_STATUS"
    [[ -f "$LOG" ]] && echo "--- last 30 log lines ---" && tail -n 30 "$LOG" || true
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
  help|--help|-h)
    grep -E '^# ' "$0" | head -30
    ;;
  *) echo "usage: $0 {start|status|stop} <sub>" >&2; exit 2 ;;
esac
