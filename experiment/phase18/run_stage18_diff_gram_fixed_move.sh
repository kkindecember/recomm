#!/usr/bin/env bash
set -euo pipefail
cd /mnt/18T/jiangtangyunzhi/projects/recomm
export CUDA_VISIBLE_DEVICES=GPU-07374397-b213-09c1-2988-718acb6ed231
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec /home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python -u -m \
    experiment.phase18.protocol.s18_diff_gram_fixed_move \
    --config experiment/phase18/config/s18_diff_gram_beauty_gpu5_move.json
