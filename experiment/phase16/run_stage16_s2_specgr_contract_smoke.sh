#!/usr/bin/env bash
set -u -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUTPUT="$ROOT/artifacts/phase16/s2_specgr_contract_smoke"
STATUS="$OUTPUT/status.json"
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
CONFIG=experiment/phase16/configs/stage16_s2_specgr_contract_smoke.json
EXACT_COMMAND="bash experiment/phase16/run_stage16_s2_specgr_contract_smoke.sh"
STARTED_AT=$(date -Is)
STAGE=preflight
PROGRESS=0
TOTAL=3
SELECTED_GPU=null
ADMISSION_FREE=0

write_status() {
  local state=$1 code=$2 reason=$3 rc=$4 alive=$5 pending=$6
  local temporary="$STATUS.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE16_S2_SPECGR_CONTRACT_SMOKE","attempt_id":"s16_s2_contract_a1","status":"%s","status_code":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","last_progress_at":"%s","runner_pid":%d,"workload_pid":%d,"process_alive":%s,"physical_gpu":%s,"visible_gpu":%s,"gpu_count":%d,"minimum_free_mib_per_gpu":10240,"admission_free_mib_per_gpu":[%d],"expected_peak_mib_per_gpu":8192,"progress_current":%d,"progress_total":%d,"progress_unit":"contract_steps","hard_timeout_seconds":600,"workload_rc":%d,"exit_code":%d,"exit_code_pending":%s,"test_read":false,"automatic_retry":false,"exact_start_command":"%s","output_dir":"artifacts/phase16/s2_specgr_contract_smoke","log_path":null,"summary_path":"artifacts/phase16/s2_specgr_contract_smoke/summary.json"}\n' \
    "$state" "$code" "$STAGE" "$reason" "$STARTED_AT" "$(date -Is)" "$(date -Is)" $$ $$ "$alive" "$SELECTED_GPU" "$SELECTED_GPU" "$([[ "$SELECTED_GPU" == null ]] && echo 0 || echo 1)" "$ADMISSION_FREE" "$PROGRESS" "$TOTAL" "$rc" "$rc" "$pending" "$EXACT_COMMAND" > "$temporary"
  mv "$temporary" "$STATUS"
}

cd "$ROOT" || exit 2
if [[ -e "$OUTPUT/summary.json" || -e "$OUTPUT/smoke_summary.json" ]]; then
  echo "Refusing to overwrite existing S16-2 contract-smoke artifacts." >&2
  exit 8
fi

write_status running RUNNING "S16-2 official-runtime syntax and contract tests." -1 true true
timeout --signal=TERM --kill-after=10 300 \
  "$PYTHON" -m py_compile \
    experiment/phase16/protocol/official_specgr_runtime.py \
    experiment/phase16/protocol/specgr_faithful.py \
    experiment/phase16/protocol/specgr_contract_smoke.py \
    experiment/phase16/protocol/finalize_s2_contract_smoke.py \
    experiment/phase16/tests/test_specgr_faithful.py
rc=$?
if (( rc != 0 )); then
  STAGE=finished
  write_status failed FAILED "S16-2 syntax preflight failed; no automatic retry." "$rc" false false
  exit "$rc"
fi

timeout --signal=TERM --kill-after=10 300 \
  "$PYTHON" -m unittest discover -s experiment/phase16/tests -v
rc=$?
if (( rc != 0 )); then
  STAGE=finished
  write_status failed FAILED "S16-2 contract tests failed; no automatic retry." "$rc" false false
  exit "$rc"
fi
PROGRESS=1
STAGE=gpu_admission
write_status running RUNNING "Selecting one GPU with at least 10240 MiB free for the bounded smoke." -1 true true

BEST_UTIL=101
BEST_FREE=-1
SELECTED=-1
while IFS=',' read -r index free util; do
  index=${index//[[:space:]]/}
  free=${free//[[:space:]]/}
  util=${util//[[:space:]]/}
  if (( free >= 10240 )) && (( util < BEST_UTIL || (util == BEST_UTIL && free > BEST_FREE) )); then
    SELECTED=$index
    BEST_FREE=$free
    BEST_UTIL=$util
  fi
done < <(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits)

if (( SELECTED < 0 )); then
  STAGE=finished
  write_status failed GPU_ADMISSION_FAILED "No GPU had 10240 MiB free; no automatic retry." 9 false false
  exit 9
fi
SELECTED_GPU=$SELECTED
ADMISSION_FREE=$BEST_FREE
PROGRESS=2
STAGE=bounded_train_only_smoke
write_status running RUNNING "Running official UniSRec and real-GRAM one-step train-only contract smoke." -1 true true

CUDA_VISIBLE_DEVICES="$SELECTED" timeout --signal=TERM --kill-after=15 600 \
  "$PYTHON" experiment/phase16/protocol/specgr_contract_smoke.py \
    --config "$CONFIG" \
    --physical-gpu "$SELECTED" \
    --admission-free-mib "$BEST_FREE" \
    --admission-util-percent "$BEST_UTIL"
rc=$?
if (( rc != 0 )); then
  STAGE=finished
  if (( rc == 124 )); then
    write_status timeout TIMEOUT "S16-2 small smoke exceeded 600 seconds; no automatic retry." "$rc" false false
  else
    write_status failed FAILED "S16-2 small smoke failed; no automatic retry and no formal gate promotion." "$rc" false false
  fi
  exit "$rc"
fi

STAGE=artifact_contract
write_status running RUNNING "Finalizing S16-2 implementation/contract smoke without promoting formal gates." -1 true true
timeout --signal=TERM --kill-after=10 300 \
  "$PYTHON" experiment/phase16/protocol/finalize_s2_contract_smoke.py --config "$CONFIG"
rc=$?
if (( rc != 0 )); then
  STAGE=finished
  write_status failed FAILED "S16-2 contract-smoke artifact finalization failed; no automatic retry." "$rc" false false
  exit "$rc"
fi

PROGRESS=3
STAGE=finished
write_status completed COMPLETED "PASS_S16_2_SPECGR_IMPLEMENTATION_CONTRACT_SMALL_SMOKE; formal training/admission remains pending user GPU." 0 false false
echo "PASS_S16_2_SPECGR_IMPLEMENTATION_CONTRACT_SMALL_SMOKE"
