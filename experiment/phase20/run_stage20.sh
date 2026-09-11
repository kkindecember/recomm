#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
S20_PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
S20_SESSION=s20_sprec_gram_v1
case "${1:-status}" in
  start)
    if tmux has-session -t "$S20_SESSION" 2>/dev/null; then
      echo "Stage20 session already exists."
      exit 1
    fi
    tmux new-session -d -s "$S20_SESSION" "$S20_PYTHON -u -m experiment.phase20.supervise"
    echo "Started $S20_SESSION; inspect artifacts/phase20/status.json."
    ;;
  status)
    "$S20_PYTHON" -c 'from pathlib import Path; p=Path("artifacts/phase20/status.json"); print(p.read_text() if p.exists() else "Stage20 has not started")'
    ;;
  stop)
    "$S20_PYTHON" - <<'PY'
import json, os, signal
from pathlib import Path
d=json.loads(Path('artifacts/phase20/status.json').read_text())
if d['state'] not in ('RUNNING','STOPPING'):
    raise SystemExit('Stage20 is not running.')
pid=d['supervisor_pid']
cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ')
if b'experiment.phase20.supervise' not in cmd:
    raise SystemExit('PID no longer identifies the Stage20 supervisor; no signal sent.')
os.kill(pid,signal.SIGTERM)
print('Requested Stage20 stop; training saves at the next optimizer boundary.')
PY
    ;;
  *) echo 'Usage: run_stage20.sh {start|status|stop}' >&2; exit 2 ;;
esac
