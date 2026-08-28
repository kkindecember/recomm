#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
CONFIG=experiment/phase16/configs/stage16_s2_splus_ctrl_duplicate_guard_gpu5_a3_gpu7_a4.json
OUTPUT=artifacts/phase16/s2_splus_ctrl_formal/toys_seed1502_gpu5_a3_gpu7_a4_duplicate_guard

cd "$ROOT"
mkdir -p "$OUTPUT"
"$PYTHON" -m py_compile \
  experiment/phase16/protocol/splus_ctrl_duplicate_guard.py \
  experiment/phase16/tests/test_splus_ctrl_duplicate_guard.py >> "$OUTPUT/preflight.log" 2>&1
"$PYTHON" -m unittest experiment.phase16.tests.test_splus_ctrl_duplicate_guard -q >> "$OUTPUT/preflight.log" 2>&1
exec "$PYTHON" experiment/phase16/protocol/splus_ctrl_duplicate_guard.py \
  --config "$CONFIG" --mode watch --armed
