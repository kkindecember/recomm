#!/usr/bin/env bash
set -euo pipefail
cd /mnt/18T/jiangtangyunzhi/projects/recomm
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
exec /home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python -m experiment.phase20.lime_partial_state "${@:-launch}"
