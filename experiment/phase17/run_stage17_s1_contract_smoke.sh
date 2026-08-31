#!/usr/bin/env bash
# S17-1 bounded foreground smoke. The scientific worker executes from a frozen snapshot.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
ACTION=${1:-status}
GPU=${2:--1}
EXPERIMENT_ID=s17_s1_public_framework
ATTEMPT_ID=attempt_001
SNAPSHOT_WORKER="$ROOT/artifacts/phase17/snapshots/$EXPERIMENT_ID/$ATTEMPT_ID/src/000_s1_contract_runtime.py"

case "$ACTION" in
  run)
    S17_REPOSITORY_ROOT="$ROOT" "$PYTHON" \
      "$ROOT/experiment/phase17/protocol/s1_contract_runtime.py" \
      prepare --root "$ROOT" --gpu "$GPU"
    S17_REPOSITORY_ROOT="$ROOT" "$PYTHON" "$SNAPSHOT_WORKER" worker --root "$ROOT"
    ;;
  status)
    S17_REPOSITORY_ROOT="$ROOT" "$PYTHON" \
      "$ROOT/experiment/phase17/protocol/s1_contract_runtime.py" status --root "$ROOT"
    ;;
  *)
    echo "usage: $0 {run|status} [physical_gpu]" >&2
    exit 2
    ;;
esac
