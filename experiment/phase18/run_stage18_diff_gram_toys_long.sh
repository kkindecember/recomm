#!/usr/bin/env bash
set -euo pipefail
cd /mnt/18T/jiangtangyunzhi/projects/recomm
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=GPU-4e97077a-5bab-99a7-2fdc-598df6be74cc
exec /home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python -u -m \
    experiment.phase18.protocol.s18_diff_gram_toys_long \
    --config experiment/phase18/config/s18_diff_gram_toys_long_fixed.json "$@"
