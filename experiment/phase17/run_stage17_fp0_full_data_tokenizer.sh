#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

ACTION="${1:-prepare}"
if [[ "${ACTION}" != "prepare" && "${ACTION}" != "launch" ]]; then
  printf 'usage: %s [prepare|launch]\n' "$0" >&2
  exit 2
fi

exec /home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python \
  "${PROJECT_ROOT}/experiment/phase17/protocol/s17_fp0_full_data_tokenizer_runtime.py" \
  "${ACTION}" --root "${PROJECT_ROOT}"
