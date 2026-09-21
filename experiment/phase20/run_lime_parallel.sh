#!/usr/bin/env bash
set -euo pipefail
cd /mnt/18T/jiangtangyunzhi/projects/recomm
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ "${1:-}" == train ]]; then
    shift
    exec /home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python -B -m experiment.phase20.lime_parallel_train "$@"
fi
exec /home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python -B -m experiment.phase20.lime_parallel_job "$@"
