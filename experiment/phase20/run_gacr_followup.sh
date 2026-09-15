#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
S20_GACR_PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
S20_GACR_SESSION=s20_gacr_beauty_transfer_v1
case "${1:-status}" in
  start)
    if tmux has-session -t "$S20_GACR_SESSION" 2>/dev/null; then
      echo 'Beauty transfer session already exists.'
      exit 1
    fi
    tmux new-session -d -s "$S20_GACR_SESSION" "$S20_GACR_PYTHON -u -m experiment.phase20.gacr_transfer_queue --config experiment/phase20/gacr_v6_pcrf_beauty_v1.json"
    echo "Started $S20_GACR_SESSION; see report/第二十阶段/Stage20_后续固定对照与Beauty迁移进展报告.md"
    ;;
  status)
    "$S20_GACR_PYTHON" -c 'from pathlib import Path; p=Path("artifacts/phase20/gacr_beauty_transfer_queue_v1/status.json"); print(p.read_text() if p.exists() else "Not started")'
    ;;
  stop)
    "$S20_GACR_PYTHON" -c 'import json,os,signal; from pathlib import Path; p=Path("artifacts/phase20/gacr_beauty_transfer_queue_v1/status.json"); d=json.loads(p.read_text()); pid=d["pid"]; assert d["state"] in ["RUNNING","WAITING_FOR_RESOURCE","STOPPING"]; cmd=Path(f"/proc/{pid}/cmdline").read_bytes(); assert b"experiment.phase20.gacr_transfer_queue" in cmd; os.kill(pid,signal.SIGTERM); print("Stop requested for verified Beauty supervisor",pid)'
    ;;
  *) echo 'Usage: run_gacr_followup.sh {start|status|stop}' >&2; exit 2 ;;
esac
