#!/usr/bin/env bash
set -euo pipefail
cd /mnt/18T/jiangtangyunzhi/projects/recomm
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=GPU-4cc17bbe-a85f-bdab-44ef-bbe29d1afc09
exec /home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python -u -m \
    experiment.phase18.protocol.s18_diff_gram_fixed \
    --config experiment/phase18/config/s18_diff_gram_beauty_fixed.json "$@"
