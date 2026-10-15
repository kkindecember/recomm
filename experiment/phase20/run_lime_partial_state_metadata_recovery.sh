#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
cd /mnt/18T/jiangtangyunzhi/projects/recomm
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=''
exec /home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python -u -m experiment.phase20.recover_lime_partial_state_metadata \
  > artifacts/phase20/lime_partial_state_20260922_v1/metadata_recovery_supervisor.log 2>&1
