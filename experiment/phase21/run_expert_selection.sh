#!/usr/bin/env bash
set -eu
cd /mnt/18T/jiangtangyunzhi/projects/recomm
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
exec /home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python -u -m experiment.phase21.expert_selection supervise > artifacts/phase21/expert_selection_20261014_v1/supervisor.log 2>&1
