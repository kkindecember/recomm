#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
Q0_PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
Q0_OUTPUT=artifacts/phase20/gacr_list_distill_toys_q0_v1
Q0_SESSION=s20_gacr_list_distill_q0_v1
case "${1:-status}" in
  start)
    test -f "$Q0_OUTPUT/config.json"
    if [ -e "$Q0_OUTPUT/status.json" ]; then
      echo 'Existing Q0 run; refusing overwrite or automatic retry.'; exit 1
    fi
    tmux new-session -d -s "$Q0_SESSION" "CUDA_VISIBLE_DEVICES=GPU-07374397-b213-09c1-2988-718acb6ed231 $Q0_PYTHON -u -m experiment.phase20.gacr_list_distill_diagnostic > $Q0_OUTPUT/run.log 2>&1"
    echo 'Started read-only Q0 on GPU5; zero optimizer steps; 90 minute compute timeout.'
    ;;
  status) cat "$Q0_OUTPUT/status.json" ;;
  *) echo 'Usage: run_gacr_list_distill_diagnostic.sh start|status'; exit 2 ;;
esac
