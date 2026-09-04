#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

GRAM_PYTHON="/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python"
RUNTIME="${PROJECT_ROOT}/experiment/phase17/protocol/s17_fp12_external_d0_recovery_runtime.py"

if [[ $# -eq 0 ]]; then
  set -- inspect
fi

exec "${GRAM_PYTHON}" "${RUNTIME}" "$@" --root "${PROJECT_ROOT}"
