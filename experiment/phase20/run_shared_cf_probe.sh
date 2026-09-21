#!/usr/bin/env bash
set -euo pipefail
cd /mnt/18T/jiangtangyunzhi/projects/recomm
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
case "${1:-}" in
  check|prepare|verify) export CUDA_VISIBLE_DEVICES=''; limit=900 ;;
  run) export CUDA_VISIBLE_DEVICES=GPU-7876ca7a-7bb3-eb10-f22b-2ec3b17b9253; limit=3700 ;;
  *) echo 'Usage: run_shared_cf_probe.sh check|prepare|run|verify' >&2; exit 2 ;;
esac
exec timeout --signal=TERM --kill-after=10s "$limit" \
  /home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python -u \
  -m experiment.phase20.shared_cf_probe "$1"
