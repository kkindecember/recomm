#!/usr/bin/env bash
set -u -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export OUTPUT_OVERRIDE="$ROOT/artifacts/phase15/s2_contract_smoke/toys/b2_drafter_state_smoke_attempt2"
export SESSION_OVERRIDE=gram_stage15_s2_toys_b2_drafter_state_attempt2
export EXACT_START_COMMAND_OVERRIDE="bash experiment/phase15/run_stage15_s2_toys_b2_drafter_state_smoke_attempt2.sh start 1"
exec bash "$ROOT/experiment/phase15/run_stage15_s2_toys_b2_drafter_state_smoke.sh" "$@"
