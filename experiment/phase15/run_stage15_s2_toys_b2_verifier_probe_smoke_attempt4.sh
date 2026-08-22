#!/usr/bin/env bash
set -u -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export OUTPUT_OVERRIDE="$ROOT/artifacts/phase15/s2_contract_smoke/toys/b2_verifier_probe_smoke_attempt4"
export SESSION_OVERRIDE=gram_stage15_s2_toys_b2_verifier_probe_attempt4
export EXACT_START_COMMAND_OVERRIDE="bash experiment/phase15/run_stage15_s2_toys_b2_verifier_probe_smoke_attempt4.sh start 5"
exec bash "$ROOT/experiment/phase15/run_stage15_s2_toys_b2_verifier_probe_smoke.sh" "$@"
