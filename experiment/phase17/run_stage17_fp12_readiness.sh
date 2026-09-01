#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec /home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python \
  "${PROJECT_ROOT}/experiment/phase17/protocol/s17_fp12_readiness_runtime.py" \
  --root "${PROJECT_ROOT}"
