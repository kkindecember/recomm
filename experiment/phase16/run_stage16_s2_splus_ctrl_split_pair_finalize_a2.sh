#!/usr/bin/env bash
set -u -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
CONFIG=experiment/phase16/configs/stage16_s2_splus_ctrl_split_pair_gpu5_a3_gpu7_a4_a2.json
OUTPUT_REL=artifacts/phase16/s2_splus_ctrl_formal/toys_seed1502_gpu5_a3_gpu7_a4_split_pair_a2
OUTPUT="$ROOT/$OUTPUT_REL"
STATUS="$OUTPUT/status.json"
LOG="$OUTPUT/run.log"
EXACT_COMMAND="bash experiment/phase16/run_stage16_s2_splus_ctrl_split_pair_finalize_a2.sh"
PAIR_FINALIZER_TIMEOUT_SECONDS=1800
STARTED_AT=$(date -Is)

write_status() {
  local state=$1 code=$2 stage=$3 reason=$4 rc=$5 pending=$6 temporary
  temporary="$STATUS.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE16_S2_SPLUS_CTRL_SPLIT_PAIR_TOYS","attempt_id":"s16_s2_splus_ctrl_split_pair_gpu5_a3_gpu7_a4_a2","status":"%s","status_code":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","runner_pid":%d,"workload_pid":0,"process_alive":false,"physical_gpus":[5,7],"progress_current":%d,"progress_total":1,"progress_unit":"pair_contract","pair_finalizer_timeout_seconds":%d,"parent_attempt_id":"s16_s2_splus_ctrl_split_pair_gpu5_a3_gpu7_a4","recovery_change":"cpu_pair_finalizer_timeout_600_to_1800_seconds_only","exit_code":%d,"exit_code_pending":%s,"test_read":false,"validation_used":false,"automatic_retry":false,"source_artifacts_modified":false,"scientific_configuration_modified":false,"exact_start_command":"%s","output_dir":"%s","log_path":"%s/run.log","summary_path":"%s/summary.json"}\n' \
    "$state" "$code" "$stage" "$reason" "$STARTED_AT" "$(date -Is)" $$ "$([[ "$state" == completed ]] && echo 1 || echo 0)" "$PAIR_FINALIZER_TIMEOUT_SECONDS" "$rc" "$pending" "$EXACT_COMMAND" "$OUTPUT_REL" "$OUTPUT_REL" "$OUTPUT_REL" > "$temporary"
  mv "$temporary" "$STATUS"
}

cd "$ROOT" || exit 2
if [[ -e "$OUTPUT/summary.json" || -e "$OUTPUT/artifact_contract.json" ]]; then
  mkdir -p "$OUTPUT"
  write_status failed OUTPUT_EXISTS finished "Refusing to overwrite an existing a2 split-pair finalization." 8 false
  exit 8
fi
mkdir -p "$OUTPUT"
write_status running RUNNING finalize "Revalidating the immutable GPU5 S-PLUS and GPU7 CTRL source arms with a widened CPU hash timeout only." -1 true
timeout --signal=TERM --kill-after=10 300 "$PYTHON" -m py_compile \
  experiment/phase16/protocol/finalize_splus_ctrl_split.py \
  experiment/phase16/tests/test_splus_ctrl_split.py \
  experiment/phase16/tests/test_splus_ctrl_split_pair_recovery.py >> "$LOG" 2>&1 || {
    write_status failed PREFLIGHT_FAILED finished "Split-pair a2 syntax preflight failed." 2 false
    exit 2
  }
timeout --signal=TERM --kill-after=10 300 "$PYTHON" -m unittest \
  experiment.phase16.tests.test_splus_ctrl_split \
  experiment.phase16.tests.test_splus_ctrl_split_pair_recovery -q >> "$LOG" 2>&1 || {
    write_status failed PREFLIGHT_FAILED finished "Split-pair a2 regression tests failed." 2 false
    exit 2
  }
timeout --signal=TERM --kill-after=10 "$PAIR_FINALIZER_TIMEOUT_SECONDS" \
  "$PYTHON" experiment/phase16/protocol/finalize_splus_ctrl_split.py \
  --mode pair --config "$CONFIG" >> "$LOG" 2>&1
FINALIZE_RC=$?
if [[ "$FINALIZE_RC" -ne 0 ]]; then
  if [[ "$FINALIZE_RC" -eq 124 || "$FINALIZE_RC" -eq 137 ]]; then
    write_status blocked PAIR_FINALIZER_TIMEOUT finished "CPU checkpoint hashing exceeded the widened a2 timeout; no automatic retry." "$FINALIZE_RC" false
  else
    write_status blocked SOURCE_OR_PAIR_CONTRACT_FAILED finished "A source arm is pending or the split-pair contract failed; no automatic retry." "$FINALIZE_RC" false
  fi
  exit "$FINALIZE_RC"
fi
write_status completed COMPLETED finished "PASS_S16_2_SPLUS_FAITHFUL_CONTRACT_ADMISSION from immutable isolated GPU5/GPU7 source arms." 0 false
exit 0
