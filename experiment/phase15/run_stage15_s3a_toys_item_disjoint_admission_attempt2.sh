#!/usr/bin/env bash
set -u -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BASE="$ROOT/experiment/phase15/run_stage15_s3a_toys_item_disjoint_admission.sh"

# Attempt-2 is intentionally not started automatically.  It preserves the
# failed attempt-1 and runs the clean-base train-only layer probe before B3.
export OUTPUT_OVERRIDE="$ROOT/artifacts/phase15/s3_toys/admission/item_disjoint_b2_b3_attempt2"
export SESSION_OVERRIDE=gram_stage15_s3a_toys_item_disjoint_admission_attempt2
export EXACT_START_COMMAND_OVERRIDE="bash experiment/phase15/run_stage15_s3a_toys_item_disjoint_admission_attempt2.sh start ${2:-7}"
exec bash "$BASE" "$@"
