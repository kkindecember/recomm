#!/usr/bin/env bash
set -u -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export OUTPUT_OVERRIDE=${OUTPUT_OVERRIDE:-"$ROOT/artifacts/phase15/s3_toys/admission/b3_branching_recovery"}
export SESSION_OVERRIDE=${SESSION_OVERRIDE:-gram_stage15_s3a_toys_b3_branching_recovery}
export EXACT_START_COMMAND_OVERRIDE=${EXACT_START_COMMAND_OVERRIDE:-"bash experiment/phase15/run_stage15_s3a_toys_b3_branching_recovery.sh ${1:-status} ${2:-}"}
export ADMISSION_ARMS_OVERRIDE=b0,b2,b3

exec bash "$ROOT/experiment/phase15/run_stage15_s3a_toys_item_disjoint_admission.sh" "$@"
