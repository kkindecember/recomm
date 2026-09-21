#!/usr/bin/env bash
set -euo pipefail
cd /mnt/18T/jiangtangyunzhi/projects/recomm
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=''
export PYTHONUNBUFFERED=1
exec timeout --signal=TERM --kill-after=10s 1300 \
 /home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python \
 -m experiment.phase20.unseen_interaction_probe "${1:?prepare, run or audit}"
