#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_PATH="${PROJECT_ROOT}/artifacts/phase17/fullport/envs/latte_05e4e6d98322_torch_2_7_1_cu126/bin/python"
RUNTIME="${PROJECT_ROOT}/experiment/phase17/protocol/s17_fp3_external_d0_runtime.py"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

ACTION="${1:-inspect}"
case "${ACTION}" in
  prepare|launch|inspect)
    exec "${PYTHON_PATH}" "${RUNTIME}" "${ACTION}" --root "${PROJECT_ROOT}"
    ;;
  authorize)
    if [[ $# -ne 3 ]]; then
      printf 'usage: %s authorize PHYSICAL_GPU RESEARCHER_DIRECTION\n' "$0" >&2
      exit 2
    fi
    exec "${PYTHON_PATH}" "${RUNTIME}" authorize --root "${PROJECT_ROOT}" \
      --gpu "$2" --researcher-direction "$3"
    ;;
  *)
    printf 'usage: %s [prepare|authorize PHYSICAL_GPU RESEARCHER_DIRECTION|launch|inspect]\n' "$0" >&2
    exit 2
    ;;
esac
