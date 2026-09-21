#!/usr/bin/env bash
set -euo pipefail
cd /mnt/18T/jiangtangyunzhi/projects/recomm
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
case "${1:-}" in
  prepare) limit=900 ;;
  profile) limit=300 ;;
  train) limit=7200 ;;
  evaluate) limit=1800 ;;
  *) echo 'Usage: run_cf_exposure.sh prepare|profile|train|evaluate' >&2; exit 2 ;;
esac
if [[ "$1" == profile || "$1" == train ]]; then
  export CUDA_VISIBLE_DEVICES="${CF_EXPOSURE_GPU:-GPU-7876ca7a-7bb3-eb10-f22b-2ec3b17b9253}"
else
  export CUDA_VISIBLE_DEVICES=''
fi
exec timeout --signal=TERM --kill-after=10s "$limit" \
  /home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python \
  -m experiment.phase20.cf_exposure_screen --mode "$1"
