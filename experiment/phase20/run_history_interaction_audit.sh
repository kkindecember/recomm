#!/usr/bin/env bash
set -euo pipefail
cd /mnt/18T/jiangtangyunzhi/projects/recomm
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=GPU-7876ca7a-7bb3-eb10-f22b-2ec3b17b9253
exec timeout --signal=TERM --kill-after=10s 1300 \
  /home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python \
  -m experiment.phase20.audit_history_interaction_numeric_followup
