#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_PATH="/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python"
RUNTIME="${PROJECT_ROOT}/experiment/phase17/protocol/s17_fp12_g1_runtime_guard_v3.py"
HOST_MIGRATION="${PROJECT_ROOT}/experiment/phase17/protocol/s17_fp12_g1_guard_v3_host_migration.py"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

ACTION="${1:-inspect}"
case "${ACTION}" in
  prepare|launch|stop-v2|migrate-v2|inspect)
    exec "${PYTHON_PATH}" "${RUNTIME}" "${ACTION}" --root "${PROJECT_ROOT}"
    ;;
  validate-host-v2)
    exec "${PYTHON_PATH}" "${HOST_MIGRATION}" validate --root "${PROJECT_ROOT}"
    ;;
  resume-migration)
    exec "${PYTHON_PATH}" "${HOST_MIGRATION}" resume-migration --root "${PROJECT_ROOT}"
    ;;
  *)
    printf 'usage: %s [prepare|launch|stop-v2|migrate-v2|validate-host-v2|resume-migration|inspect]\n' "$0" >&2
    exit 2
    ;;
esac
