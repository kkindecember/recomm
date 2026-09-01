#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
CONFIG=experiment/phase16/configs/stage16_s4_portfolio2_correction_a9_cpu.json
OUTPUT=artifacts/phase16/s4_toys_standalone/correction/toys_seed1502_portfolio2_a9_cpu

cd "$ROOT"
if [[ -e "$OUTPUT" ]]; then
  echo "Refusing existing S16-4 a9 correction root; no automatic retry." >&2
  exit 8
fi

timeout --signal=TERM --kill-after=5 30 \
  "$PYTHON" -m py_compile \
    experiment/phase16/protocol/finalize_stage16_s4_toys.py \
    experiment/phase16/protocol/finalize_stage16_s4_toys_recovery.py \
    experiment/phase16/protocol/finalize_stage16_s4_portfolio2_correction.py \
    experiment/phase16/tests/test_stage16_s4_toys_validation.py \
    experiment/phase16/tests/test_stage16_s4_toys_recovery.py \
    experiment/phase16/tests/test_stage16_s4_portfolio2_correction.py

timeout --signal=TERM --kill-after=5 180 \
  "$PYTHON" -m unittest \
    experiment.phase16.tests.test_stage16_s4_toys_validation \
    experiment.phase16.tests.test_stage16_s4_toys_recovery \
    experiment.phase16.tests.test_stage16_s4_portfolio2_correction -q

env CUDA_VISIBLE_DEVICES="" \
  timeout --signal=TERM --kill-after=5 600 \
  "$PYTHON" -m \
    experiment.phase16.protocol.finalize_stage16_s4_portfolio2_correction \
    --config "$CONFIG"
