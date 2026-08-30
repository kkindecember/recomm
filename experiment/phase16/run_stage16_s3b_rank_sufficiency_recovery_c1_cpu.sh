#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
CONFIG=experiment/phase16/configs/stage16_s3b_rank_sufficiency_recovery_c1_cpu.json
OUTPUT=artifacts/phase16/s3_genrecedit/rank_sufficiency_recovery/toys_seed1502_b1_recovery_c1_cpu

cd "$ROOT"
if [[ -e "$OUTPUT" ]]; then
  echo "Refusing existing S16-3B recovery attempt root; no automatic retry." >&2
  exit 8
fi

timeout --signal=TERM --kill-after=5 30 \
  "$PYTHON" -m py_compile \
    experiment/phase16/protocol/finalize_s3b_rank_sufficiency_recovery.py \
    experiment/phase16/tests/test_gfull_rank_sufficiency_recovery.py

timeout --signal=TERM --kill-after=5 60 \
  "$PYTHON" -m unittest \
    experiment.phase16.tests.test_gfull_rank_sufficiency_recovery -q

env CUDA_VISIBLE_DEVICES="" \
  timeout --signal=TERM --kill-after=5 120 \
  "$PYTHON" -m \
    experiment.phase16.protocol.finalize_s3b_rank_sufficiency_recovery \
    --config "$CONFIG"
