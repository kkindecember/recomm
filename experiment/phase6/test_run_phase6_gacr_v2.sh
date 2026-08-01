#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_phase6_gacr_v2.sh"

GPU=6

reservation_status_is_running $'tmux session: running (codellama)\n2026-07-31 state=running session=codellama gpu=6'

if reservation_status_is_running $'tmux session: not running (codellama)\n2026-07-31 state=running session=codellama gpu=6'; then
  echo "stale status must not pass" >&2
  exit 1
fi

if reservation_status_is_running $'tmux session: running (codellama)\n2026-07-31 state=waiting_for_gpu session=codellama gpu=6'; then
  echo "waiting reservation must not pass" >&2
  exit 1
fi

if reservation_status_is_running $'tmux session: running (codellama)\n2026-07-31 state=running session=codellama gpu=4'; then
  echo "wrong GPU must not pass" >&2
  exit 1
fi

echo "resource restoration status checks passed"

TEST_OUTPUT=$(mktemp -d)
OUTPUT=$TEST_OUTPUT
LOG="$OUTPUT/run.log"
STATUS="$OUTPUT/status.json"
TELEMETRY="$OUTPUT/gpu_telemetry.csv"
EXPERIMENT_ID=GRAM_PHASE6_RUNNER_TEST
SESSION=gram_phase6_runner_test
STARTED_AT=2026-07-31T00:00:00+08:00
CURRENT_STAGE=starting
RESERVATION_STATE=scheduled_for_release
write_status starting "Runner JSON format test."
python3 -m json.tool "$STATUS" >/dev/null
grep -q '"log_path":"' "$STATUS"
grep -q '"result_path":"' "$STATUS"
echo "generic status JSON checks passed"
