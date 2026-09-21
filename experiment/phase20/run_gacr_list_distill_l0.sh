#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
L0_PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
L0_OUTPUT=artifacts/phase20/gacr_list_distill_toys_l0_v1
case "${1:-status}" in
  start)
    test -f "$L0_OUTPUT/config.json"
    if [ -e "$L0_OUTPUT/controller_started.json" ]; then
      echo 'Existing L0 attempt; refusing overwrite or implicit retry.'
      exit 1
    fi
    tmux new-session -d -s s20_gacr_list_distill_l0_v1 "$L0_PYTHON -u -m experiment.phase20.gacr_list_distill_l0 > $L0_OUTPUT/controller.log 2>&1"
    echo 'Started L0 two-arm controller; fixed32 steps, profile gate, bounded runtime.'
    ;;
  status) cat "$L0_OUTPUT/status.json" ;;
  *) echo 'Usage: run_gacr_list_distill_l0.sh start|status'; exit 2 ;;
esac
