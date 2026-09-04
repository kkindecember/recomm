#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_PATH="${PROJECT_ROOT}/artifacts/phase17/fullport/envs/latte_05e4e6d98322_torch_2_7_1_cu126/bin/python"
RUNTIME="${PROJECT_ROOT}/experiment/phase17/protocol/s17_fp3_setrec_formal_runtime.py"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

ACTION="${1:-inspect-all}"
case "${ACTION}" in
  prepare-all|launch-all|inspect-all)
    exec "${PYTHON_PATH}" "${RUNTIME}" "${ACTION}" --root "${PROJECT_ROOT}"
    ;;
  authorize-all)
    if [[ $# -ne 2 ]]; then
      printf 'usage: %s authorize-all RESEARCHER_DIRECTION\n' "$0" >&2
      exit 2
    fi
    exec "${PYTHON_PATH}" "${RUNTIME}" authorize-all --root "${PROJECT_ROOT}" \
      --researcher-direction "$2"
    ;;
  inspect)
    if [[ $# -ne 2 ]]; then
      printf 'usage: %s inspect ARM\n' "$0" >&2
      exit 2
    fi
    exec "${PYTHON_PATH}" "${RUNTIME}" inspect --root "${PROJECT_ROOT}" --arm "$2"
    ;;
  *)
    printf 'usage: %s [prepare-all|authorize-all RESEARCHER_DIRECTION|launch-all|inspect-all|inspect ARM]\n' "$0" >&2
    exit 2
    ;;
esac
