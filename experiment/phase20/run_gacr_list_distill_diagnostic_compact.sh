#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
Q0_COMPACT_PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
Q0_COMPACT_OUTPUT=artifacts/phase20/gacr_list_distill_toys_q0_compact_v2
Q0_COMPACT_SESSION=s20_gacr_list_distill_q0_compact_v2
case "${1:-status}" in
  start)
    test -f "$Q0_COMPACT_OUTPUT/config.json"
    if [ -e "$Q0_COMPACT_OUTPUT/status.json" ]; then
      echo 'Existing compact Q0 attempt; refusing overwrite or automatic retry.'; exit 1
    fi
    tmux new-session -d -s "$Q0_COMPACT_SESSION" "CUDA_VISIBLE_DEVICES=GPU-7876ca7a-7bb3-eb10-f22b-2ec3b17b9253 $Q0_COMPACT_PYTHON -u -m experiment.phase20.gacr_list_distill_diagnostic_compact > $Q0_COMPACT_OUTPUT/run.log 2>&1"
    echo 'Started read-only compact Q0 on GPU7; no optimizer; 90 minute compute timeout.'
    ;;
  status) cat "$Q0_COMPACT_OUTPUT/status.json" ;;
  *) echo 'Usage: run_gacr_list_distill_diagnostic_compact.sh start|status'; exit 2 ;;
esac
