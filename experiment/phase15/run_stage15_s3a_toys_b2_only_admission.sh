#!/usr/bin/env bash
set -u -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export OUTPUT_OVERRIDE="$ROOT/artifacts/phase15/s3_toys/admission/b2_item_disjoint"
export SESSION_OVERRIDE=gram_stage15_s3a_toys_b2_item_disjoint
export ADMISSION_ARMS_OVERRIDE=b0,b2
export EXACT_START_COMMAND_OVERRIDE="bash experiment/phase15/run_stage15_s3a_toys_b2_only_admission.sh start ${2:-7}"
exec bash "$ROOT/experiment/phase15/run_stage15_s3a_toys_item_disjoint_admission.sh" "$@"
