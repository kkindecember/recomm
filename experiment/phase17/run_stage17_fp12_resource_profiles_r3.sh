#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

ACTION="${1:-inspect-all}"
ARM="${2:-}"
GRAM_PYTHON="/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python"
NATIVE_PYTHON="${PROJECT_ROOT}/artifacts/phase17/fullport/envs/latte_05e4e6d98322_torch_2_7_1_cu126/bin/python"
RUNTIME="${PROJECT_ROOT}/experiment/phase17/protocol/s17_fp12_resource_profile_r3_runtime.py"

run_one() {
  local action="$1"
  local arm="$2"
  local python_path="${GRAM_PYTHON}"
  if [[ "${arm}" == N* ]]; then
    python_path="${NATIVE_PYTHON}"
  fi
  "${python_path}" "${RUNTIME}" "${action}" --arm "${arm}" --root "${PROJECT_ROOT}"
}

ARMS=(
  G0_GRAM_B0_FRESH
  G1_GRAM_PSID_FULL
  G2_GRAM_LATTE_FULL
  N0_NATIVE_PSID
  N1_NATIVE_LATTE
)

case "${ACTION}" in
  prepare-all|inspect-all)
    one_action="${ACTION%-all}"
    for arm in "${ARMS[@]}"; do
      run_one "${one_action}" "${arm}"
    done
    ;;
  prepare|authorize|launch|inspect)
    if [[ -z "${ARM}" ]]; then
      printf 'usage: %s %s ARM\n' "$0" "${ACTION}" >&2
      exit 2
    fi
    run_one "${ACTION}" "${ARM}"
    ;;
  *)
    printf 'usage: %s [prepare-all|inspect-all|prepare ARM|authorize ARM|launch ARM|inspect ARM]\n' "$0" >&2
    exit 2
    ;;
esac
