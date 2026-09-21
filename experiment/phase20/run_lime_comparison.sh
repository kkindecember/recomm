#!/usr/bin/env bash
set -euo pipefail
cd /mnt/18T/jiangtangyunzhi/projects/recomm
if [[ "${1:-launch}" == "recover" ]]; then
  exec /home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python -B \
    -m experiment.phase20.lime_comparison_recovery launch
fi
exec /home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python -B \
  -m experiment.phase20.lime_comparison_job "${1:-launch}"
