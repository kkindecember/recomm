#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
S20_RECOVERY_PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
S20_RECOVERY_SESSION=s20_gacr_list_distill_recovery_v3
case "${1:-status}" in
  start)
    test -f experiment/phase20/gacr_list_distill_toys_recovery_v3.json
    if [ -f artifacts/phase20/gacr_list_distill_toys_recovery_v3/status.json ]; then
      echo 'Existing recovery attempt; refusing overwrite/retry.'; exit 1
    fi
    tmux new-session -d -s "$S20_RECOVERY_SESSION" "$S20_RECOVERY_PYTHON -u -m experiment.phase20.gacr_list_distill_recovery > artifacts/phase20/gacr_list_distill_toys_recovery_v3/controller.log 2>&1"
    echo 'Started recovery; artifacts/phase20/gacr_list_distill_toys_recovery_v3/status.json'
    ;;
  status)
    cat artifacts/phase20/gacr_list_distill_toys_recovery_v3/status.json
    ;;
  stop)
    "$S20_RECOVERY_PYTHON" -m experiment.phase20.gacr_list_distill_recovery --stop
    ;;
  *) echo 'Usage: run_gacr_list_distill_recovery.sh {start|status|stop}'; exit 2 ;;
esac
