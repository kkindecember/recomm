#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
S20_LIST_PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
S20_LIST_SESSION=s20_gacr_list_distill_toys_gpu1_v2
case "${1:-status}" in
  start)
    if tmux has-session -t "$S20_LIST_SESSION" 2>/dev/null; then
      echo 'List distillation session already exists.'; exit 1
    fi
    if [ -f artifacts/phase20/gacr_list_distill_toys_gpu1_v2/status.json ]; then
      echo 'Existing attempt found. Refusing restart or overwrite.'; exit 1
    fi
    tmux new-session -d -s "$S20_LIST_SESSION" "$S20_LIST_PYTHON -u -m experiment.phase20.gacr_list_distill_queue --config experiment/phase20/gacr_list_distill_toys_gpu1_v2.json"
    echo 'Started background queue; status: artifacts/phase20/gacr_list_distill_toys_gpu1_v2/status.json'
    ;;
  status)
    "$S20_LIST_PYTHON" -c 'from pathlib import Path; p=Path("artifacts/phase20/gacr_list_distill_toys_gpu1_v2/status.json"); print(p.read_text() if p.exists() else "Not started")'
    ;;
  stop)
    "$S20_LIST_PYTHON" -c 'import json,os,signal; from pathlib import Path; d=json.loads(Path("artifacts/phase20/gacr_list_distill_toys_gpu1_v2/status.json").read_text()); assert d["state"] in ["STARTING","RUNNING","WAITING_FOR_RESOURCE","STOPPING"]; pid=d["pid"]; assert b"experiment.phase20.gacr_list_distill_queue" in Path(f"/proc/{pid}/cmdline").read_bytes(); assert Path(f"/proc/{pid}/stat").read_text().split()[21]==d["pid_start_ticks"]; os.kill(pid,signal.SIGTERM); print("Stop requested for verified queue",pid)'
    ;;
  *) echo 'Usage: run_gacr_list_distill_gpu1.sh {start|status|stop}' >&2; exit 2 ;;
esac
