#!/usr/bin/env bash
set -euo pipefail
cd /mnt/18T/jiangtangyunzhi/projects/recomm
if [[ "${1:-launch}" == "tie-audit" ]]; then
  exec /home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python -B \
    -m experiment.phase20.source_competition_tie_audit
fi
if [[ "${1:-launch}" == "preflight" ]]; then
  exec /home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python -B \
    -m experiment.phase20.source_competition_preflight
fi
exec /home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python -B \
  -m experiment.phase20.source_competition_screen "${1:-launch}"
