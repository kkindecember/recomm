#!/usr/bin/env bash
set -euo pipefail
cd /mnt/18T/jiangtangyunzhi/projects/recomm
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
exec /home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python -m experiment.phase20.lime_expert_ablation
