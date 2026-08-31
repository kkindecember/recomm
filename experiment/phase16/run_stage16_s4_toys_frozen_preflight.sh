#!/usr/bin/env bash
set -u -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
CONFIG=experiment/phase16/configs/stage16_s4_toys_frozen_preflight.json
OUTPUT="$ROOT/artifacts/phase16/s4_toys_standalone/preflight/a1"

cd "$ROOT" || exit 2
if [[ -e "$OUTPUT" ]]; then
  echo "Refusing existing S16-4 preflight artifact root: $OUTPUT" >&2
  exit 8
fi

bash -n experiment/phase16/run_stage16_s4_toys_frozen_preflight.sh || exit 4
"$PYTHON" -m py_compile \
  experiment/phase16/protocol/stage16_s4_toys_frozen_preflight.py \
  experiment/phase16/tests/test_stage16_s4_toys_frozen_preflight.py || exit 5
"$PYTHON" -m unittest \
  experiment.phase16.tests.test_stage16_s4_toys_frozen_preflight -v || exit 6

timeout --signal=TERM --kill-after=10 1800 \
  "$PYTHON" experiment/phase16/protocol/stage16_s4_toys_frozen_preflight.py \
    --config "$CONFIG"
