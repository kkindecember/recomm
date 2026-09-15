#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
S20_CM_PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
S20_CM_SESSION=s20_gacr_controls_multiseed_v1
case "${1:-status}" in
  start)
    if tmux has-session -t "$S20_CM_SESSION" 2>/dev/null; then
      echo 'Controls/multiseed session already exists.'
      exit 1
    fi
    if [ -f artifacts/phase20/gacr_controls_multiseed_v1/status.json ]; then
      echo 'Existing attempt found. Refusing restart or overwrite.'
      exit 1
    fi
    tmux new-session -d -s "$S20_CM_SESSION" "$S20_CM_PYTHON -u -m experiment.phase20.gacr_controls_multiseed_queue --config experiment/phase20/gacr_controls_multiseed_v1.json"
    echo 'Started background queue; status: artifacts/phase20/gacr_controls_multiseed_v1/status.json'
    ;;
  status)
    "$S20_CM_PYTHON" -c 'from pathlib import Path; p=Path("artifacts/phase20/gacr_controls_multiseed_v1/status.json"); print(p.read_text() if p.exists() else "Not started")'
    ;;
  stop)
    "$S20_CM_PYTHON" -c 'import json,os,signal; from pathlib import Path; d=json.loads(Path("artifacts/phase20/gacr_controls_multiseed_v1/status.json").read_text()); assert d["state"] in ["STARTING","RUNNING","WAITING_FOR_RESOURCE","STOPPING"]; pid=d["pid"]; assert b"experiment.phase20.gacr_controls_multiseed_queue" in Path(f"/proc/{pid}/cmdline").read_bytes(); assert Path(f"/proc/{pid}/stat").read_text().split()[21]==d["pid_start_ticks"]; os.kill(pid,signal.SIGTERM); print("Stop requested for verified queue",pid)'
    ;;
  *) echo 'Usage: run_gacr_controls_multiseed.sh {start|status|stop}' >&2; exit 2 ;;
esac
