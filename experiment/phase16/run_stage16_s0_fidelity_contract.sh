#!/usr/bin/env bash
set -u -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUTPUT="$ROOT/artifacts/phase16/s0_fidelity_contract"
STATUS="$OUTPUT/status.json"
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
HARD_TIMEOUT_SECONDS=300
EXACT_COMMAND="bash experiment/phase16/run_stage16_s0_fidelity_contract.sh"
STARTED_AT=$(date -Is)
STAGE=preflight
PROGRESS=0
TOTAL=3

write_status() {
  local state=$1 code=$2 reason=$3 rc=$4 alive=$5 pending=$6
  local temporary="$STATUS.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE16_S0_FIDELITY_CONTRACT","attempt_id":"s16_s0_a1","status":"%s","status_code":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","last_progress_at":"%s","runner_pid":%d,"workload_pid":%d,"process_alive":%s,"physical_gpu":null,"visible_gpu":null,"gpu_count":0,"minimum_free_mib_per_gpu":0,"admission_free_mib_per_gpu":[],"expected_peak_mib_per_gpu":0,"progress_current":%d,"progress_total":%d,"progress_unit":"audit_steps","hard_timeout_seconds":%d,"workload_rc":%d,"exit_code":%d,"exit_code_pending":%s,"test_read":false,"automatic_retry":false,"exact_start_command":"%s","output_dir":"artifacts/phase16/s0_fidelity_contract","log_path":null,"summary_path":"artifacts/phase16/s0_fidelity_contract/summary.json"}\n' \
    "$state" "$code" "$STAGE" "$reason" "$STARTED_AT" "$(date -Is)" "$(date -Is)" $$ $$ "$alive" "$PROGRESS" "$TOTAL" "$HARD_TIMEOUT_SECONDS" "$rc" "$rc" "$pending" "$EXACT_COMMAND" > "$temporary"
  mv "$temporary" "$STATUS"
}

cd "$ROOT" || exit 2
if [[ -e "$OUTPUT/summary.json" ]]; then
  echo "Refusing to overwrite completed S16-0 artifacts." >&2
  exit 8
fi

write_status running RUNNING "S16-0 CPU fidelity audit preflight." -1 true true

timeout --signal=TERM --kill-after=10 "$HARD_TIMEOUT_SECONDS" \
  "$PYTHON" -m py_compile \
    experiment/phase16/protocol/fidelity_bridge.py \
    experiment/phase16/protocol/fidelity_contract_audit.py \
    experiment/phase16/tests/test_fidelity_bridge.py
rc=$?
if (( rc != 0 )); then
  STAGE=finished
  write_status failed FAILED "S16-0 Python syntax preflight failed; no automatic retry." "$rc" false false
  exit "$rc"
fi
PROGRESS=1
STAGE=bridge_unit_tests
write_status running RUNNING "Running fixed-width and variable-lexical bridge tests." -1 true true

timeout --signal=TERM --kill-after=10 "$HARD_TIMEOUT_SECONDS" \
  "$PYTHON" -m unittest discover -s experiment/phase16/tests -v
rc=$?
if (( rc != 0 )); then
  STAGE=finished
  write_status failed FAILED "S16-0 bridge tests failed; no automatic retry." "$rc" false false
  exit "$rc"
fi
PROGRESS=2
STAGE=function_level_audit
write_status running RUNNING "Verifying pinned source evidence, defaults, and F0/F1 mappings." -1 true true

timeout --signal=TERM --kill-after=10 "$HARD_TIMEOUT_SECONDS" \
  "$PYTHON" experiment/phase16/protocol/fidelity_contract_audit.py \
    --config experiment/phase16/configs/stage16_s0_fidelity_contract.json
rc=$?
if (( rc != 0 )); then
  STAGE=finished
  if (( rc == 124 )); then
    write_status timeout TIMEOUT "S16-0 exceeded the 300-second hard timeout; no automatic retry." "$rc" false false
  else
    write_status failed FAILED "S16-0 function-level audit failed; no automatic retry." "$rc" false false
  fi
  exit "$rc"
fi

PROGRESS=3
STAGE=finished
write_status completed COMPLETED "PASS_S16_0_FIDELITY_CONTRACT; test remained sealed and no GPU/network was used." 0 false false
echo "PASS_S16_0_FIDELITY_CONTRACT"
