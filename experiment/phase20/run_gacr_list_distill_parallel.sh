#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
S20_PARALLEL_PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
S20_PARALLEL_SESSION=s20_gacr_list_distill_parallel_v2
case "${1:-status}" in
  start)
    "$S20_PARALLEL_PYTHON" -m experiment.phase20.gacr_list_distill_parallel --preflight
    if [ -f artifacts/phase20/gacr_list_distill_toys_gpu1_v2/parallel_v2/status.json ]; then
      echo 'Existing parallel attempt; refusing restart.'; exit 1
    fi
    tmux new-session -d -s "$S20_PARALLEL_SESSION" "$S20_PARALLEL_PYTHON -u -m experiment.phase20.gacr_list_distill_parallel"
    echo 'Started parallel controller; status: artifacts/phase20/gacr_list_distill_toys_gpu1_v2/status.json'
    ;;
  status)
    "$S20_PARALLEL_PYTHON" -c 'from pathlib import Path; print(Path("artifacts/phase20/gacr_list_distill_toys_gpu1_v2/status.json").read_text())'
    ;;
  stop)
    "$S20_PARALLEL_PYTHON" -m experiment.phase20.gacr_list_distill_parallel --stop
    ;;
  *) echo 'Usage: run_gacr_list_distill_parallel.sh {start|status|stop}' >&2; exit 2 ;;
esac
