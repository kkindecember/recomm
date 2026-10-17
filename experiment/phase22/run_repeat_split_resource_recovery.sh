#!/usr/bin/env bash
set -euo pipefail
cd /mnt/18T/jiangtangyunzhi/projects/recomm
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
exec /home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python -u -m experiment.phase22.repeat_split_resource_recovery "${@:-supervise}"
