#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export S17_REPOSITORY_ROOT="${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec /home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python \
  "${PROJECT_ROOT}/experiment/phase17/protocol/s3_formal_parent_runtime.py" \
  launch --root "${PROJECT_ROOT}"
