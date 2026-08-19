#!/usr/bin/env bash
# Phase-13 CANARD exploratory runner (single-card sequential).
#
# Wraps GRAM training under the phase12-style protection protocol:
#   1. Preflight (unit tests, syntax checks)
#   2. Ensure GPU protector is holding the target card (CodeLlama on GPU6 or
#      ablation-scan holder on other cards)
#   3. Stop protector, verify >= 30 GiB free
#   4. Start gpu_memory_lease sidecar holding (30G - workload peak)
#   5. Run GRAM training + eval
#   6. Postflight: run eval_cold_warm.py to split test predictions by
#      cold/warm target; write metrics_cold_warm.json
#   7. Restart the protector (exit trap guarantees this even on crash)
#
# Sub-experiments (add more as you build v1, v2, ...):
#   smoke_v0_beauty      Beauty_cold50, 1 epoch smoke, from t5-small cold start
#   v0_beauty            Beauty_cold50, 30 epochs, vanilla GRAM baseline
#   v0_beauty_eta80      Beauty_cold80, aggressive cold hedge for v0 iteration
#   v0_toys              Toys_cold50, 30 epochs, second-domain vanilla baseline
#   v1_beauty            Beauty_cold50 + MLP-predicted cold ids (c128/l7)
#   v1_toys              Toys_cold50 + MLP-predicted cold ids (c32/l5)
#   smoke_v1_collision_safe_beauty  Beauty collision-safe IDs, 1 epoch / 100 samples
#   smoke_v1_collision_safe_toys    Toys collision-safe IDs, 1 epoch / 100 samples
#   v1_collision_safe_beauty        Beauty collision-safe IDs, full 30 epochs
#   v1_collision_safe_toys          Toys collision-safe IDs, full 30 epochs
#   v2_toys              Toys_cold50 + MLP v2 with LLM prior (c32/l5)
#
# Usage:
#   bash experiment/phase13/run_phase13_explore.sh start <sub> [gpu]
#   bash experiment/phase13/run_phase13_explore.sh status <sub>
#   bash experiment/phase13/run_phase13_explore.sh stop <sub>
#
# Protector selection (env var, defaults inferred from GPU):
#   PROTECTOR_TOOL=codellama       Use tools/run_codellama.sh (default for GPU6)
#   PROTECTOR_TOOL=ablation_scan   Use tools/gram_ablation_scan.sh (default for others)
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
LEASE_HELPER="$ROOT/experiment/gpu_memory_lease.py"

# Protector tools
CODELLAMA_TOOL="$ROOT/tools/run_codellama.sh"
ABLATION_SCAN_TOOL="$ROOT/tools/gram_ablation_scan.sh"
CODELLAMA_HF_HOME=/home/jiangtangyunzhi/hf_cache
WORKLOAD_CACHE="$ROOT/.cache/huggingface"

COLD_SPLIT_PY="$ROOT/experiment/phase13/protocol/cold_split.py"
EVAL_COLD_WARM_PY="$ROOT/experiment/phase13/protocol/eval_cold_warm.py"

ACTION=${1:-status}
if [[ "$ACTION" == "worker" ]]; then
  SUB="${PHASE13_SUB:-}"
  GPU="${PHASE13_GPU:-6}"
else
  SUB=${2:-}
  GPU=${3:-6}
fi

if [[ -z "$SUB" && "$ACTION" != "help" && "$ACTION" != "--help" && "$ACTION" != "-h" ]]; then
  echo "usage: $0 {start|status|stop} <sub> [gpu]" >&2
  echo "  sub ∈ {smoke_v0_beauty, v0_beauty, v0_beauty_eta80, v0_toys, v1_beauty, v1_toys, v2_toys}" >&2
  exit 2
fi

# Auto-pick protector if not set explicitly
if [[ -z "${PROTECTOR_TOOL:-}" ]]; then
  if [[ "$GPU" == "6" ]]; then
    PROTECTOR_TOOL=codellama
  else
    PROTECTOR_TOOL=ablation_scan
  fi
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
COLD_ETA=""
COLD_SEED=""
COLD_MIN_WARM=""
COLD_BUCKETS=""
V1_MLPCOLD=0  # 1 = require MLP-cold artifacts and use v1_mlpcold id suffix
V1_COLLISION_SAFE=0  # 1 = require collision-safe v1 IDs and audit report
V2_MLPCOLD_LLMPRIOR=0  # 1 = require MLP v2 + LLM prior artifacts, use v2_mlpcold_llmprior suffix
V2ITER2_MLPCOLD_LLMPRIOR=0  # 1 = v2_iter2 (vocab-constrained + OOV-masked), use v2iter2_mlpcold_llmprior suffix
RESTORE_IMMEDIATELY=0  # collision-safe reruns return the card to scan holder at exit

case "${SUB:-}" in
  smoke_v0_beauty)
    DATASET=Beauty_cold50; EPOCHS=1; CLUSTER=128; ID_LEN=7; NUM_CF=10; BEAM_SIZE=50
    DEBUG_TRAIN_100=1; DEBUG_TEST_100=1; TEST_EPOCH_REC=0; SAVE_REC_EPOCHS=1
    COLD_ETA=0.5; COLD_SEED=12345; COLD_MIN_WARM=3; COLD_BUCKETS=10
    ;;
  v0_beauty)
    DATASET=Beauty_cold50; EPOCHS=30; CLUSTER=128; ID_LEN=7; NUM_CF=10; BEAM_SIZE=50
    DEBUG_TRAIN_100=0; DEBUG_TEST_100=0; TEST_EPOCH_REC=5; SAVE_REC_EPOCHS=5
    COLD_ETA=0.5; COLD_SEED=12345; COLD_MIN_WARM=3; COLD_BUCKETS=10
    ;;
  v0_beauty_eta80)
    # Aggressive cold hedge — plan v0 iteration option "如果 gate 失败,可能是
    # split 不够激进,提高到 η=0.8 再试". Pre-run in parallel with v0_beauty on
    # a separate GPU so we don't lose 12h if η=0.5 turns out to be too weak.
    DATASET=Beauty_cold80; EPOCHS=30; CLUSTER=128; ID_LEN=7; NUM_CF=10; BEAM_SIZE=50
    DEBUG_TRAIN_100=0; DEBUG_TEST_100=0; TEST_EPOCH_REC=5; SAVE_REC_EPOCHS=5
    COLD_ETA=0.8; COLD_SEED=12345; COLD_MIN_WARM=3; COLD_BUCKETS=10
    ;;
  v0_toys)
    # Second-domain v0 baseline. Same cold-split params as v0_beauty for
    # comparability; runs in parallel on a separate GPU. Gets us a Toys data
    # point that publication phase will need anyway.
    DATASET=Toys_cold50; EPOCHS=30; CLUSTER=32; ID_LEN=5; NUM_CF=5; BEAM_SIZE=50
    DEBUG_TRAIN_100=0; DEBUG_TEST_100=0; TEST_EPOCH_REC=5; SAVE_REC_EPOCHS=5
    COLD_ETA=0.5; COLD_SEED=12345; COLD_MIN_WARM=3; COLD_BUCKETS=10
    ;;
  v1_beauty)
    # v1 Semantic Bridge on Beauty_cold50: uses MLP-predicted hierarchical ids
    # for cold items (warm items keep their original ids). Training pipeline
    # of GRAM itself is identical to v0_beauty; the only change is the
    # hierarchical_id_type pointing at a merged id file.
    #
    # Requires v1_prep artifacts (see experiment/phase13/tests/v1_prep_recipe.md
    # or the top-level README's "Phase 13 v1 pipeline" section):
    #   1. artifacts/phase13/embeddings/beauty_sbert.pt   (precompute_item_embeddings.py)
    #   2. artifacts/phase13/explore/v1_beauty/mlp/best.pt (semantic_bridge.py train)
    #   3. GRAM/rec_datasets/Beauty_cold50/item_generative_indexing_hierarchy_v1_c128_l7_len32768_split_v1_mlpcold.txt
    #      (assign_cold_ids.py)
    # Preflight refuses to launch if these are missing.
    DATASET=Beauty_cold50; EPOCHS=30; CLUSTER=128; ID_LEN=7; NUM_CF=10; BEAM_SIZE=50
    DEBUG_TRAIN_100=0; DEBUG_TEST_100=0; TEST_EPOCH_REC=5; SAVE_REC_EPOCHS=5
    COLD_ETA=0.5; COLD_SEED=12345; COLD_MIN_WARM=3; COLD_BUCKETS=10
    V1_MLPCOLD=1
    ;;
  v1_toys)
    # v1 Semantic Bridge on Toys_cold50 (second-domain mirror of v1_beauty).
    # Same MLP-predicted-cold-id pipeline; only difference is Toys uses the
    # smaller c32/l5 hierarchical id space (matches v0_toys for apples-to-apples).
    #
    # Requires v1_prep artifacts:
    #   1. artifacts/phase13/embeddings/toys_sbert.pt
    #   2. artifacts/phase13/explore/v1_toys/mlp/best.pt
    #   3. GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split_v1_mlpcold.txt
    DATASET=Toys_cold50; EPOCHS=30; CLUSTER=32; ID_LEN=5; NUM_CF=5; BEAM_SIZE=50
    DEBUG_TRAIN_100=0; DEBUG_TEST_100=0; TEST_EPOCH_REC=5; SAVE_REC_EPOCHS=5
    COLD_ETA=0.5; COLD_SEED=12345; COLD_MIN_WARM=3; COLD_BUCKETS=10
    V1_MLPCOLD=1
    ;;
  smoke_v1_collision_safe_beauty)
    DATASET=Beauty_cold50; EPOCHS=1; CLUSTER=128; ID_LEN=7; NUM_CF=10; BEAM_SIZE=50
    DEBUG_TRAIN_100=1; DEBUG_TEST_100=1; TEST_EPOCH_REC=0; SAVE_REC_EPOCHS=1
    COLD_ETA=0.5; COLD_SEED=12345; COLD_MIN_WARM=3; COLD_BUCKETS=10
    V1_COLLISION_SAFE=1; RESTORE_IMMEDIATELY=1
    ;;
  smoke_v1_collision_safe_toys)
    DATASET=Toys_cold50; EPOCHS=1; CLUSTER=32; ID_LEN=5; NUM_CF=5; BEAM_SIZE=50
    DEBUG_TRAIN_100=1; DEBUG_TEST_100=1; TEST_EPOCH_REC=0; SAVE_REC_EPOCHS=1
    COLD_ETA=0.5; COLD_SEED=12345; COLD_MIN_WARM=3; COLD_BUCKETS=10
    V1_COLLISION_SAFE=1; RESTORE_IMMEDIATELY=1
    ;;
  v1_collision_safe_beauty)
    DATASET=Beauty_cold50; EPOCHS=30; CLUSTER=128; ID_LEN=7; NUM_CF=10; BEAM_SIZE=50
    DEBUG_TRAIN_100=0; DEBUG_TEST_100=0; TEST_EPOCH_REC=5; SAVE_REC_EPOCHS=5
    COLD_ETA=0.5; COLD_SEED=12345; COLD_MIN_WARM=3; COLD_BUCKETS=10
    V1_COLLISION_SAFE=1; RESTORE_IMMEDIATELY=1
    ;;
  v1_collision_safe_toys)
    DATASET=Toys_cold50; EPOCHS=30; CLUSTER=32; ID_LEN=5; NUM_CF=5; BEAM_SIZE=50
    DEBUG_TRAIN_100=0; DEBUG_TEST_100=0; TEST_EPOCH_REC=5; SAVE_REC_EPOCHS=5
    COLD_ETA=0.5; COLD_SEED=12345; COLD_MIN_WARM=3; COLD_BUCKETS=10
    V1_COLLISION_SAFE=1; RESTORE_IMMEDIATELY=1
    ;;
  v2_toys)
    # v2 LLM Prior on Toys_cold50 (extends v1 with LLM prior regularization).
    # MLP v2 trained with KL(MLP || LLM prior) on warm items, using DeepSeek API.
    # Hierarchical ID suffix: v2_mlpcold_llmprior
    #
    # Requires v2_prep artifacts:
    #   1. artifacts/phase13/embeddings/Toys_sbert.pt
    #   2. artifacts/phase13/explore/v2_toys/mlp/best.pt (trained with λ_llm=0.5)
    #   3. artifacts/phase13/explore/v2_toys/llm_priors_all.jsonl (warm+cold priors)
    #   4. GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split_v2_mlpcold_llmprior.txt
    DATASET=Toys_cold50; EPOCHS=30; CLUSTER=32; ID_LEN=5; NUM_CF=5; BEAM_SIZE=50
    DEBUG_TRAIN_100=0; DEBUG_TEST_100=0; TEST_EPOCH_REC=5; SAVE_REC_EPOCHS=5
    COLD_ETA=0.5; COLD_SEED=12345; COLD_MIN_WARM=3; COLD_BUCKETS=10
    V2_MLPCOLD_LLMPRIOR=1
    ;;
  v2_beauty)
    # v2 LLM Prior on Beauty_cold50 (cross-domain validation of v2 approach).
    DATASET=Beauty_cold50; EPOCHS=30; CLUSTER=128; ID_LEN=7; NUM_CF=10; BEAM_SIZE=50
    DEBUG_TRAIN_100=0; DEBUG_TEST_100=0; TEST_EPOCH_REC=5; SAVE_REC_EPOCHS=5
    COLD_ETA=0.5; COLD_SEED=12345; COLD_MIN_WARM=3; COLD_BUCKETS=10
    V2_MLPCOLD_LLMPRIOR=1
    ;;
  v2_toys_iter2)
    # v2_iter2: vocab-constrained LLM prior + OOV mask (fix from iter1 -48% regression)
    # Suffix: v2iter2_mlpcold_llmprior (from artifacts/phase13/explore/v2_toys_iter2/)
    DATASET=Toys_cold50; EPOCHS=30; CLUSTER=32; ID_LEN=5; NUM_CF=5; BEAM_SIZE=50
    DEBUG_TRAIN_100=0; DEBUG_TEST_100=0; TEST_EPOCH_REC=5; SAVE_REC_EPOCHS=5
    COLD_ETA=0.5; COLD_SEED=12345; COLD_MIN_WARM=3; COLD_BUCKETS=10
    V2ITER2_MLPCOLD_LLMPRIOR=1
    ;;
  v2_beauty_iter2)
    DATASET=Beauty_cold50; EPOCHS=30; CLUSTER=128; ID_LEN=7; NUM_CF=10; BEAM_SIZE=50
    DEBUG_TRAIN_100=0; DEBUG_TEST_100=0; TEST_EPOCH_REC=5; SAVE_REC_EPOCHS=5
    COLD_ETA=0.5; COLD_SEED=12345; COLD_MIN_WARM=3; COLD_BUCKETS=10
    V2ITER2_MLPCOLD_LLMPRIOR=1
    ;;
  "")
    :  # allowed for worker/help; error handled above
    ;;
  *)
    echo "unknown sub: $SUB" >&2; exit 2 ;;
esac

SESSION="gram_phase13_explore_${SUB}"
OUTPUT="$ROOT/artifacts/phase13/explore/${SUB}"
LOG="$OUTPUT/run.log"
STATUS="$OUTPUT/status.json"
LEASE_STATUS="$OUTPUT/gpu_lease.json"
TELEMETRY_CSV="$OUTPUT/gpu_telemetry.csv"
METRICS_COLD_WARM="$OUTPUT/metrics_cold_warm.json"

TOTAL_LEASE_MIB=${LEASE_MIB_OVERRIDE:-30720}
# v1_toys measured peak_reserved=19290; use 21000 for safety margin
EXPECTED_PEAK_MIB=${EXPECTED_PEAK_MIB_OVERRIDE:-21000}
HOLD_TIMEOUT_HOURS=${HOLD_TIMEOUT_HOURS:-6}
HARD_TIMEOUT_SECONDS=${HARD_TIMEOUT_SECONDS:-259200}  # 72h; validation-heavy Beauty needs >30h
SCAN_SESSION=${SCAN_SESSION:-gram_ablation_scan_gpu${GPU}}
SCAN_STATE_ROOT=${SCAN_STATE_ROOT:-$ROOT/.runtime/gram_ablation_scan_gpu${GPU}}
RESOURCE_WATCHDOG="$ROOT/experiment/phase13/resource_watchdog.sh"

[[ "$HARD_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
  echo "HARD_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 2
}

WORKLOAD_PID=0
LEASE_PID=""
TELEMETRY_PID=""
POSTFLIGHT_RC=0
TERMINATION_SIGNAL=""
STARTED_AT=""
STAGE=not_started

# Protector abstraction ------------------------------------------------------
# The two tools share a start <gpu> / stop / status interface. Their process
# names and NVML footprints differ, so we detect "is present on target GPU"
# differently for each.

case "$PROTECTOR_TOOL" in
  codellama)
    PROTECTOR_TOOL_PATH="$CODELLAMA_TOOL"
    PROTECTOR_PROCESS_MATCH="unittest-transformers"
    RESERVATION=codellama_expected_on_gpu${GPU}
    ;;
  ablation_scan)
    PROTECTOR_TOOL_PATH="$ABLATION_SCAN_TOOL"
    PROTECTOR_PROCESS_MATCH="gram-repro"
    RESERVATION=ablation_scan_expected_on_gpu${GPU}
    ;;
  *)
    echo "unknown PROTECTOR_TOOL: $PROTECTOR_TOOL (expected codellama|ablation_scan)" >&2
    exit 2 ;;
esac

protector() {
  # $1 = start|stop|status ; for start we also pass $GPU
  if [[ "$PROTECTOR_TOOL" == "codellama" ]]; then
    env SESSION=codellama HF_HOME="$CODELLAMA_HF_HOME" \
      HF_HUB_CACHE="$CODELLAMA_HF_HOME/hub" \
      TRANSFORMERS_CACHE="$CODELLAMA_HF_HOME/hub" "$PROTECTOR_TOOL_PATH" "$@"
  else
    env RESERVE_MIB="${HOLDER_RESERVE_MIB_OVERRIDE:-$(adaptive_reserve_mib)}" \
      SESSION="$SCAN_SESSION" STATE_ROOT="$SCAN_STATE_ROOT" \
      "$PROTECTOR_TOOL_PATH" "$@"
  fi
}

# Largest holder that currently fits on $GPU. A fixed RESERVE_MIB OOMs on a
# fragmented card: v2_toys_iter2 asked for 22 GiB with 21.78 GiB free, the
# protector died, and GPU0 sat unprotected for hours.
CUDA_CTX_MIB=${CUDA_CTX_MIB:-2200}   # CUDA context costs ~2 GiB on top of the tensor
MIN_WORTH_HOLDING_MIB=${MIN_WORTH_HOLDING_MIB:-2000}
adaptive_reserve_mib() {
  local free_mib
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
    --id="$GPU" 2>/dev/null | tr -d ' ')
  [[ "$free_mib" =~ ^[0-9]+$ ]] || { echo 0; return; }
  local want=$(( free_mib - CUDA_CTX_MIB ))
  # Never round *up* to a floor — that is what guarantees the OOM we are avoiding.
  (( want < MIN_WORTH_HOLDING_MIB )) && want=0
  echo "$want"
}

protector_on_target() {
  # Verify the protector process is holding memory on the target GPU
  if [[ "$PROTECTOR_TOOL" == "ablation_scan" ]]; then
    local holder_pid
    holder_pid=$("$PYTHON" -c \
      "import json; print(json.load(open('$SCAN_STATE_ROOT/status.json')).get('pid', 0))" \
      2>/dev/null || echo 0)
    [[ "$holder_pid" =~ ^[0-9]+$ ]] && (( holder_pid > 0 )) || return 1
    nvidia-smi --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader,nounits --id="$GPU" 2>/dev/null | \
      grep -Eq "^[[:space:]]*${holder_pid},.*gram-repro" || return 1
  else
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
      --format=csv,noheader --id="$GPU" 2>/dev/null | \
      grep -q "$PROTECTOR_PROCESS_MATCH" || return 1
  fi
}

protector_at_requested_size() {
  protector_on_target || return 1
  if [[ "$PROTECTOR_TOOL" == "ablation_scan" && -n "${HOLDER_RESERVE_MIB_OVERRIDE:-}" ]]; then
    local actual_reserve
    actual_reserve=$("$PYTHON" -c \
      "import json; print(json.load(open('$SCAN_STATE_ROOT/status.json')).get('reserve_mib', 0))" \
      2>/dev/null || echo 0)
    [[ "$actual_reserve" == "$HOLDER_RESERVE_MIB_OVERRIDE" ]] || return 1
  fi
}

ensure_protector_on_target() {
  if protector_on_target; then
    RESERVATION=${PROTECTOR_TOOL}_confirmed_on_gpu${GPU}
    return 0
  fi
  protector start "$GPU" >/dev/null 2>&1 || true
  for _ in $(seq 1 180); do
    if protector_on_target; then
      RESERVATION=${PROTECTOR_TOOL}_confirmed_on_gpu${GPU}
      return 0
    fi
    sleep 5
  done
  RESERVATION=${PROTECTOR_TOOL}_prepare_failed_on_gpu${GPU}
  return 1
}

restore() {
  if protector_at_requested_size; then
    RESERVATION=${PROTECTOR_TOOL}_exact_holder_on_gpu${GPU}
    return 0
  fi
  STAGE=resource_restoration
  RESERVATION=restoring_${PROTECTOR_TOOL}_to_gpu${GPU}
  write_status restoring_resource "Experiment ended; restoring $PROTECTOR_TOOL on GPU${GPU}."
  protector start "$GPU" >/dev/null 2>&1 || true
  for _ in $(seq 1 30); do
    if protector_at_requested_size; then
      RESERVATION=exact_holder_restored_on_gpu${GPU}
      return 0
    fi
    sleep 1
  done

  # If the original-sized scan holder no longer fits, retain as much of the
  # still-free card as is safe. This is explicitly reported as degraded, never
  # mislabeled as an exact restore. The independent watchdog keeps the same
  # recovery policy if this shell is killed before reaching here.
  if [[ "$PROTECTOR_TOOL" == "ablation_scan" ]]; then
    protector stop >/dev/null 2>&1 || true
    local adaptive_mib
    adaptive_mib=$(adaptive_reserve_mib)
    if (( adaptive_mib >= MIN_WORTH_HOLDING_MIB )); then
      env RESERVE_MIB="$adaptive_mib" SESSION="$SCAN_SESSION" STATE_ROOT="$SCAN_STATE_ROOT" \
        "$ABLATION_SCAN_TOOL" start "$GPU" >/dev/null 2>&1 || true
      for _ in $(seq 1 30); do
        if protector_on_target; then
          RESERVATION=degraded_scan_${adaptive_mib}mib_on_gpu${GPU}
          return 1
        fi
        sleep 1
      done
    fi
  fi
  RESERVATION=resource_restore_unprotected_on_gpu${GPU}
  return 2
}

# ---------------------------------------------------------------------------

write_status() {
  local state=$1 reason=$2 tmp="${STATUS}.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE13_%s_V1","sub":"%s","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%s,"workload_pid":%s,"tmux_session":"%s","physical_gpu":%d,"total_gpu_lease_mib":%d,"hard_timeout_seconds":%d,"resource_reservation":"%s","protector_tool":"%s","log_path":"%s","dataset":"%s"}\n' \
    "${SUB^^}" "$SUB" "$state" "$STAGE" "$reason" "$STARTED_AT" "$(date -Is)" "$$" "$WORKLOAD_PID" \
    "$SESSION" "$GPU" "$TOTAL_LEASE_MIB" "$HARD_TIMEOUT_SECONDS" "$RESERVATION" "$PROTECTOR_TOOL" "${LOG#$ROOT/}" "$DATASET" > "$tmp"
  mv "$tmp" "$STATUS"
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
  # The pane closing can deliver HUP to the sidecar before its Python signal
  # handler writes state=released. The process is already gone after wait, so
  # make the durable state agree with the verified cleanup outcome here.
  printf '{"gpu": %d, "state": "released"}\n' "$GPU" > "$LEASE_STATUS"
}

# Postflight: partition test predictions by warm/cold target
run_postflight_eval() {
  STAGE=postflight_eval
  write_status postflight_eval "Running eval_cold_warm.py on latest test predictions."
  local pred_tsv
  pred_tsv=$(ls -t "$OUTPUT/predictions"/*_test.tsv 2>/dev/null | head -n 1 || true)
  if [[ -z "$pred_tsv" ]]; then
    echo "[postflight] no *_test.tsv found under $OUTPUT/predictions — training did not reach test inference" >&2
    return 2
  fi
  "$PYTHON" "$EVAL_COLD_WARM_PY" \
    --dataset-dir "$ROOT/GRAM/rec_datasets/$DATASET" \
    --predictions-tsv "$pred_tsv" \
    --output-json "$METRICS_COLD_WARM" \
    --version-tag "$SUB" \
    --split-name test || {
      echo "[postflight] eval_cold_warm.py failed" >&2
      return 1
    }
}

finish() {
  local shell_rc=$?
  trap - EXIT INT TERM HUP
  local scientific_rc=${WORKLOAD_RC-$shell_rc}
  local postflight_rc=${POSTFLIGHT_RC:-0}
  local restore_rc=0
  if (( WORKLOAD_PID > 0 )) && kill -0 "$WORKLOAD_PID" 2>/dev/null; then
    kill -TERM "$WORKLOAD_PID" 2>/dev/null || true
    wait "$WORKLOAD_PID" 2>/dev/null || true
  fi
  WORKLOAD_PID=0
  [[ -z "$TELEMETRY_PID" ]] || kill "$TELEMETRY_PID" >/dev/null 2>&1 || true

  STAGE=finished
  if (( scientific_rc == 0 && postflight_rc != 0 )); then
    release_lease
    restore || restore_rc=$?
    write_status failed "Training succeeded but postflight failed (rc=$postflight_rc); resource=$RESERVATION; no automatic retry."
    exit "$postflight_rc"
  elif (( scientific_rc == 0 && RESTORE_IMMEDIATELY == 1 )); then
    release_lease
    restore || restore_rc=$?
    printf '{"gpu": %d, "state": "released"}\n' "$GPU" > "$LEASE_STATUS"
    if (( restore_rc == 0 )); then
      write_status succeeded "Training and postflight succeeded; exact holder restored on GPU${GPU}."
    else
      write_status succeeded_resource_degraded "Training succeeded; resource protection is $RESERVATION (restore_rc=$restore_rc)."
    fi
    exit 0
  elif (( scientific_rc == 0 )); then
    write_status holding_post_training "Training done. Lease held on GPU${GPU}. Run 'stop $SUB' to release and restore protector. Auto-releases after ${HOLD_TIMEOUT_HOURS}h."
    echo "[finish] training done; lease still held on GPU${GPU}. Use 'stop $SUB' to release." >&2
    echo "[finish] auto-release after ${HOLD_TIMEOUT_HOURS}h if nobody stops it." >&2
    # Bounded hold: v2_toys_iter2 sat here for 32h unattended, squatting a card
    # nobody was using.
    hold_deadline=$(( $(date +%s) + HOLD_TIMEOUT_HOURS * 3600 ))
    while (( $(date +%s) < hold_deadline )); do sleep 60; done
    echo "[finish] hold timeout (${HOLD_TIMEOUT_HOURS}h) reached; releasing lease and restoring protector." >&2
    release_lease
    restore || restore_rc=$?
    write_status hold_timeout_released "Held ${HOLD_TIMEOUT_HOURS}h; resource=$RESERVATION (restore_rc=$restore_rc)."
    exit 0
  else
    # Training failed — release and try to restore
    release_lease
    restore || restore_rc=$?
    if [[ -n "$TERMINATION_SIGNAL" ]]; then
      write_status stopped "Stopped by $TERMINATION_SIGNAL (rc=$scientific_rc); resource=$RESERVATION (restore_rc=$restore_rc)."
    elif (( scientific_rc == 124 || scientific_rc == 137 )); then
      write_status timed_out "Hard timeout after ${HARD_TIMEOUT_SECONDS}s (rc=$scientific_rc); resource=$RESERVATION (restore_rc=$restore_rc)."
    else
      write_status failed "Scientific exit=$scientific_rc; resource=$RESERVATION (restore_rc=$restore_rc); no automatic retry."
    fi
    exit "$scientific_rc"
  fi
}

on_signal() {
  TERMINATION_SIGNAL=$1
  exit "$2"
}

ensure_cold_split_ready() {
  local target_dir="$ROOT/GRAM/rec_datasets/$DATASET"
  local cfg="$target_dir/cold_split_meta/config.json"
  if [[ -f "$cfg" ]]; then
    # Verify config matches expected params (defensive)
    local actual_eta actual_seed actual_min actual_buckets
    actual_eta=$("$PYTHON" -c "import json; print(json.load(open('$cfg'))['eta'])")
    actual_seed=$("$PYTHON" -c "import json; print(json.load(open('$cfg'))['seed'])")
    actual_min=$("$PYTHON" -c "import json; print(json.load(open('$cfg'))['min_warm_history'])")
    actual_buckets=$("$PYTHON" -c "import json; print(json.load(open('$cfg'))['buckets'])")
    if [[ "$actual_eta" == "$COLD_ETA" ]] && [[ "$actual_seed" == "$COLD_SEED" ]] \
       && [[ "$actual_min" == "$COLD_MIN_WARM" ]] && [[ "$actual_buckets" == "$COLD_BUCKETS" ]]; then
      return 0
    fi
    echo "[cold-split] config drift detected — regenerating" >&2
  fi
  # Extract source from DATASET name: Beauty_cold50 -> Beauty
  local src_dataset="${DATASET%_cold*}"
  "$PYTHON" "$COLD_SPLIT_PY" \
    --source-dir "$ROOT/GRAM/rec_datasets/$src_dataset" \
    --output-dir "$target_dir" \
    --eta "$COLD_ETA" --buckets "$COLD_BUCKETS" \
    --seed "$COLD_SEED" --min-warm-history "$COLD_MIN_WARM" --force
}

worker() {
  STARTED_AT=${1:?missing start timestamp}
  trap finish EXIT
  trap 'on_signal INT 130' INT
  trap 'on_signal TERM 143' TERM
  trap 'on_signal HUP 129' HUP
  cd "$ROOT"
  mkdir -p "$OUTPUT"

  STAGE=preflight
  write_status preflight "Running syntax checks and cold-split reconciliation."
  "$PYTHON" -m py_compile \
    "$ROOT/GRAM/src/main_generative_gram.py" \
    "$ROOT/GRAM/src/arguments.py" \
    "$COLD_SPLIT_PY" "$EVAL_COLD_WARM_PY" \
    || { write_status blocked "Python syntax check failed."; exit 4; }
  bash -n "$0" || { write_status blocked "Runner shell syntax invalid."; exit 5; }

  STAGE=cold_split_reconciliation
  write_status preparing_data "Ensuring $DATASET is present with cold-split config."
  ensure_cold_split_ready || { write_status blocked "cold_split.py failed."; exit 6; }

  if (( V1_MLPCOLD == 1 )); then
    STAGE=v1_mlpcold_preflight
    write_status preparing_data "Verifying v1 MLP-cold artifacts (embeddings, MLP, merged id file)."
    local dataset_dir="$ROOT/GRAM/rec_datasets/$DATASET"
    local embed_pt="$ROOT/artifacts/phase13/embeddings/${DATASET%_cold*}_sbert.pt"
    local mlp_pt="$ROOT/artifacts/phase13/explore/${SUB}/mlp/best.pt"
    local hier_type="hierarchy_v1_c${CLUSTER}_l${ID_LEN}_len32768_split_v1_mlpcold"
    local mlpcold_id="$dataset_dir/item_generative_indexing_${hier_type}.txt"
    for p in "$embed_pt" "$mlp_pt" "$mlpcold_id"; do
      if [[ ! -f "$p" ]]; then
        write_status blocked "v1 MLP-cold artifact missing: $p"
        cat <<EOF
[v1-preflight] MISSING: $p

Run the v1 prep pipeline first (all steps must be done in order):

# 1. Precompute sentence-BERT embeddings for ALL Beauty items
bash -c "$PYTHON $ROOT/experiment/phase13/protocol/precompute_item_embeddings.py \\
  --item-text $ROOT/GRAM/rec_datasets/${DATASET%_cold*}/item_plain_text.txt \\
  --output $embed_pt \\
  --model sentence-transformers/all-MiniLM-L6-v2 \\
  --device cuda:0"

# 2. Train the semantic bridge MLP on warm items
bash -c "$PYTHON $ROOT/experiment/phase13/protocol/semantic_bridge.py train \\
  --embeddings $embed_pt \\
  --id-file $ROOT/GRAM/rec_datasets/${DATASET%_cold*}/item_generative_indexing_hierarchy_v1_c${CLUSTER}_l${ID_LEN}_len32768_split.txt \\
  --cold-items $dataset_dir/cold_split_meta/cold_items.txt \\
  --output-dir $(dirname $mlp_pt) \\
  --epochs 200 --lr 1e-3 --device cuda:0"

# 3. Assign MLP-predicted ids to cold items, merge with warm ids
bash -c "$PYTHON $ROOT/experiment/phase13/protocol/assign_cold_ids.py \\
  --embeddings $embed_pt \\
  --mlp $mlp_pt \\
  --source-id-file $ROOT/GRAM/rec_datasets/${DATASET%_cold*}/item_generative_indexing_hierarchy_v1_c${CLUSTER}_l${ID_LEN}_len32768_split.txt \\
  --cold-items $dataset_dir/cold_split_meta/cold_items.txt \\
  --output-id-file $mlpcold_id \\
  --report $(dirname $mlp_pt)/assign_report.json --device cuda:0"

Then re-run this runner.
EOF
        exit 10
      fi
    done
  fi

  if (( V1_COLLISION_SAFE == 1 )); then
    STAGE=v1_collision_safe_preflight
    write_status preparing_data "Verifying globally unique v1 collision-safe IDs and audit invariants."
    local dataset_dir="$ROOT/GRAM/rec_datasets/$DATASET"
    local domain_lower="${DATASET%_cold*}"
    domain_lower="${domain_lower,,}"
    local hier_type="hierarchy_v1_c${CLUSTER}_l${ID_LEN}_len32768_split_v1_mlpcold_collision_safe"
    local collision_safe_id="$dataset_dir/item_generative_indexing_${hier_type}.txt"
    local collision_report="$ROOT/artifacts/phase13/explore/v1_collision_safe/${domain_lower}_id_report.json"
    for p in "$collision_safe_id" "$collision_report"; do
      if [[ ! -s "$p" ]]; then
        write_status blocked "Collision-safe artifact missing or empty: $p"
        exit 12
      fi
    done
    "$PYTHON" -m unittest experiment/phase13/tests/test_collision_safe_ids.py \
      || { write_status blocked "Collision-safe unit tests failed."; exit 13; }
    "$PYTHON" -c \
      "import json; r=json.load(open('$collision_report')); assert r['output_collision']['duplicate_excess']==0; assert r['warm_ids_unchanged']; assert r['row_order_unchanged']; assert r['cold_prefixes_unchanged']" \
      || { write_status blocked "Collision-safe report invariants failed."; exit 14; }
  fi

  if (( V2_MLPCOLD_LLMPRIOR == 1 )); then
    STAGE=v2_mlpcold_llmprior_preflight
    write_status preparing_data "Verifying v2 MLP + LLM prior artifacts."
    local dataset_dir="$ROOT/GRAM/rec_datasets/$DATASET"
    local embed_pt="$ROOT/artifacts/phase13/embeddings/${DATASET%_cold*}_sbert.pt"
    local mlp_pt="$ROOT/artifacts/phase13/explore/${SUB}/mlp/best.pt"
    local llm_priors="$ROOT/artifacts/phase13/explore/${SUB}/llm_priors_all.jsonl"
    local hier_type="hierarchy_v1_c${CLUSTER}_l${ID_LEN}_len32768_split_v2_mlpcold_llmprior"
    local v2_id="$dataset_dir/item_generative_indexing_${hier_type}.txt"
    for p in "$embed_pt" "$mlp_pt" "$llm_priors" "$v2_id"; do
      if [[ ! -f "$p" ]]; then
        write_status blocked "v2 artifact missing: $p"
        echo "[v2-preflight] MISSING: $p"
        echo ""
        echo "Run the v2 prep pipeline (see artifacts/phase13/explore/v2_toys/run_v2_retrain_and_prepare.sh)"
        exit 11
      fi
    done
  fi

  if (( V2ITER2_MLPCOLD_LLMPRIOR == 1 )); then
    STAGE=v2iter2_mlpcold_llmprior_preflight
    write_status preparing_data "Verifying v2_iter2 MLP + LLM prior artifacts."
    local dataset_dir="$ROOT/GRAM/rec_datasets/$DATASET"
    local embed_pt="$ROOT/artifacts/phase13/embeddings/${DATASET%_cold*}_sbert.pt"
    local mlp_pt="$ROOT/artifacts/phase13/explore/${SUB}/mlp/best.pt"
    local llm_priors="$ROOT/artifacts/phase13/explore/${SUB}/llm_priors_all.jsonl"
    local hier_type="hierarchy_v1_c${CLUSTER}_l${ID_LEN}_len32768_split_v2iter2_mlpcold_llmprior"
    local v2_id="$dataset_dir/item_generative_indexing_${hier_type}.txt"
    for p in "$embed_pt" "$mlp_pt" "$llm_priors" "$v2_id"; do
      if [[ ! -f "$p" ]]; then
        write_status blocked "v2_iter2 artifact missing: $p"
        echo "[v2iter2-preflight] MISSING: $p"
        echo "Run: bash experiment/phase13/prep_v2iter2.sh <toys|beauty>"
        exit 11
      fi
    done
  fi

  if [[ "${SKIP_PROTECTOR_CHECK:-0}" == "1" ]]; then
    RESERVATION=released_for_experiment
    protector stop >/dev/null 2>&1 || true
  else
    STAGE=${PROTECTOR_TOOL}_pre_reservation
    write_status preparing_resource "Ensuring $PROTECTOR_TOOL occupies GPU${GPU}."
    ensure_protector_on_target || { write_status blocked "Could not confirm $PROTECTOR_TOOL on GPU${GPU}."; exit 7; }

    STAGE=resource_release
    write_status releasing_resource "Stopping $PROTECTOR_TOOL before Phase-13 $SUB."
    protector stop >/dev/null 2>&1 || true
    RESERVATION=released_for_experiment
  fi

  STAGE=gpu_memory_gate
  write_status waiting_for_gpu "Waiting for ≥ ${TOTAL_LEASE_MIB} MiB free on GPU${GPU}."
  local free_mib=""
  for _ in $(seq 1 120); do
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GPU" 2>/dev/null | tr -d ' ' || true)
    [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= TOTAL_LEASE_MIB )) && break
    sleep 5
  done
  [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= TOTAL_LEASE_MIB )) || { write_status blocked "GPU${GPU} admission failed: ${free_mib:-unknown} MiB free."; exit 8; }

  if [[ "${NO_LEASE_SIDECAR:-0}" == "1" ]]; then
    STAGE=ready_no_lease
    write_status ready "GPU${GPU} ready (${free_mib} MiB free). Training without lease sidecar."
    echo '{"gpu": '$GPU', "state": "no_sidecar"}' > "$LEASE_STATUS"
  else
    STAGE=memory_lease
    write_status leasing "Starting ${TOTAL_LEASE_MIB} MiB total-lease sidecar (expected workload peak ${EXPECTED_PEAK_MIB} MiB)."
    "$PYTHON" "$LEASE_HELPER" --gpu "$GPU" --total-lease-mib "$TOTAL_LEASE_MIB" \
      --expected-workload-peak-mib "$EXPECTED_PEAK_MIB" --status-path "$LEASE_STATUS" &
    LEASE_PID=$!
    for _ in $(seq 1 30); do
      [[ -s "$LEASE_STATUS" ]] && grep -q '"state": "holding"' "$LEASE_STATUS" && break
      sleep 1
    done
    [[ -s "$LEASE_STATUS" ]] && grep -q '"state": "holding"' "$LEASE_STATUS" || { write_status blocked "GPU lease sidecar did not hold."; exit 9; }
  fi

  STAGE=telemetry
  telemetry & TELEMETRY_PID=$!

  STAGE=gram_training_${SUB}
  write_status running "Phase-13 $SUB training on GPU${GPU} (dataset=$DATASET)."
  local item_id="hierarchy_v1_c${CLUSTER}_l${ID_LEN}_len32768_split"
  if (( V1_MLPCOLD == 1 )); then
    item_id="${item_id}_v1_mlpcold"
  fi
  if (( V1_COLLISION_SAFE == 1 )); then
    item_id="${item_id}_v1_mlpcold_collision_safe"
  fi
  if (( V2_MLPCOLD_LLMPRIOR == 1 )); then
    item_id="${item_id}_v2_mlpcold_llmprior"
  fi
  if (( V2ITER2_MLPCOLD_LLMPRIOR == 1 )); then
    item_id="${item_id}_v2iter2_mlpcold_llmprior"
  fi
  cd "$ROOT/GRAM/command"
  timeout --signal=TERM --kill-after=60 "$HARD_TIMEOUT_SECONDS" env CUDA_VISIBLE_DEVICES="$GPU" \
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
    --cf0_arm A --cf0_phase9 0 &
  WORKLOAD_PID=$!
  write_status running "Phase-13 $SUB training on GPU${GPU} (dataset=$DATASET, workload_pid=$WORKLOAD_PID)."
  wait "$WORKLOAD_PID"
  WORKLOAD_RC=$?
  WORKLOAD_PID=0
  if (( WORKLOAD_RC == 0 )); then
    run_postflight_eval || POSTFLIGHT_RC=$?
  fi
  return "$WORKLOAD_RC"
}

case "$ACTION" in
  start)
    mkdir -p "$OUTPUT"
    tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists: $SESSION" >&2; exit 1; }
    STARTED_AT=$(date -Is)
    printf -v launch_cmd 'PHASE13_SUB=%q PHASE13_GPU=%q PROTECTOR_TOOL=%q LEASE_MIB_OVERRIDE=%q EXPECTED_PEAK_MIB_OVERRIDE=%q HOLDER_RESERVE_MIB_OVERRIDE=%q SKIP_PROTECTOR_CHECK=%q NO_LEASE_SIDECAR=%q HOLD_TIMEOUT_HOURS=%q HARD_TIMEOUT_SECONDS=%q SCAN_SESSION=%q SCAN_STATE_ROOT=%q bash %q worker %q >> %q 2>&1' \
      "$SUB" "$GPU" "$PROTECTOR_TOOL" "${LEASE_MIB_OVERRIDE:-30720}" "${EXPECTED_PEAK_MIB_OVERRIDE:-21000}" "${HOLDER_RESERVE_MIB_OVERRIDE:-}" "${SKIP_PROTECTOR_CHECK:-0}" "${NO_LEASE_SIDECAR:-0}" "${HOLD_TIMEOUT_HOURS:-6}" "$HARD_TIMEOUT_SECONDS" "$SCAN_SESSION" "$SCAN_STATE_ROOT" "$0" "$STARTED_AT" "$LOG"
    STAGE=starting
    write_status starting "Persistent Phase-13 $SUB session is starting; runner will drive $PROTECTOR_TOOL and lease."
    tmux new-session -d -s "$SESSION" "$launch_cmd"
    if [[ "$PROTECTOR_TOOL" == "ablation_scan" && -n "${HOLDER_RESERVE_MIB_OVERRIDE:-}" ]]; then
      bash "$RESOURCE_WATCHDOG" start "$SUB" "$GPU" "$HOLDER_RESERVE_MIB_OVERRIDE" \
        "$SESSION" "$SCAN_SESSION" "$SCAN_STATE_ROOT" || {
          echo "WARNING: independent resource watchdog did not start for $SUB" >&2
        }
    fi
    echo "started $SESSION (gpu=$GPU protector=$PROTECTOR_TOOL)"
    ;;
  worker)
    worker "${2:?missing start timestamp}"
    ;;
  status)
    tmux has-session -t "$SESSION" 2>/dev/null && echo "tmux session: running ($SESSION)" || echo "tmux session: not running ($SESSION)"
    [[ -f "$STATUS" ]] && sed -n '1,100p' "$STATUS" || echo '{"status":"not_started"}'
    [[ -f "$LEASE_STATUS" ]] && echo "--- lease ---" && sed -n '1,20p' "$LEASE_STATUS"
    [[ -f "$METRICS_COLD_WARM" ]] && echo "--- cold/warm metrics ---" && sed -n '1,40p' "$METRICS_COLD_WARM"
    [[ -f "$LOG" ]] && echo "--- last 30 log lines ---" && tail -n 30 "$LOG" || true
    [[ -x "$RESOURCE_WATCHDOG" ]] && bash "$RESOURCE_WATCHDOG" status "$SUB" || true
    ;;
  stop)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      pane_pid=$(tmux list-panes -t "$SESSION" -F '#{pane_pid}' | head -n 1)
      kill -TERM "$pane_pid"
      for _ in $(seq 1 180); do
        tmux has-session -t "$SESSION" 2>/dev/null || break
        sleep 1
      done
      if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "WARNING: $SESSION did not exit in 180s; left running so its cleanup trap is not bypassed" >&2
        exit 1
      fi
      echo "stopped $SESSION after cleanup; inspect status for exact/degraded resource result"
    else
      echo "session not running: $SESSION"
    fi
    ;;
  help|--help|-h)
    grep -E '^# ' "$0" | head -40
    ;;
  *) echo "usage: $0 {start|status|stop} <sub> [gpu]" >&2; exit 2 ;;
esac
