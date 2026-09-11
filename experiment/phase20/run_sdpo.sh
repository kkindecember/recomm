#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
S20_SDPO_PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
case "${1:-status}" in
  start)
    "$S20_SDPO_PYTHON" -c 'import json; from pathlib import Path; c=json.loads(Path("experiment/phase20/sdpo_toys_v1.json").read_text()); assert json.loads((Path(c["smoke_output"])/"smoke.json").read_text())["state"]=="PASSED"; assert not (Path(c["output"])/"manifest.json").exists()'
    if tmux has-session -t s20_sdpo_gram_toys_v1 2>/dev/null; then
      echo 'S-DPO session already exists.'
      exit 1
    fi
    mkdir -p artifacts/phase20/sdpo_gram_toys_v1
    tmux new-session -d -s s20_sdpo_gram_toys_v1 "exec env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $S20_SDPO_PYTHON -u -m experiment.phase20.run_sdpo --mode run >> artifacts/phase20/sdpo_gram_toys_v1/run.log 2>&1"
    echo 'Started s20_sdpo_gram_toys_v1.'
    ;;
  overview)
    if ! tmux has-session -t s20_overview 2>/dev/null; then
      tmux new-session -d -s s20_overview "$S20_SDPO_PYTHON -u -m experiment.phase20.overview"
    fi
    echo 'Overview: artifacts/phase20/overview/status.json'
    ;;
  status)
    "$S20_SDPO_PYTHON" -c 'from pathlib import Path; p=Path("artifacts/phase20/sdpo_gram_toys_v1/status.json"); print(p.read_text() if p.exists() else "S-DPO has not started")'
    ;;
  stop)
    "$S20_SDPO_PYTHON" - <<'PY'
from pathlib import Path
import json,os,signal
d=json.loads(Path('artifacts/phase20/sdpo_gram_toys_v1/status.json').read_text())
if d['state'] != 'RUNNING': raise SystemExit('S-DPO is not running.')
pid=d['pid']; cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ')
if b'experiment.phase20.run_sdpo' not in cmd: raise SystemExit('PID no longer matches; no signal sent.')
os.kill(pid,signal.SIGTERM)
print('Requested S-DPO stop at next safe boundary.')
PY
    ;;
  *) echo 'Usage: run_sdpo.sh {start|overview|status|stop}' >&2; exit 2 ;;
esac
