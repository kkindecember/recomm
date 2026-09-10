#!/usr/bin/env bash
set -euo pipefail
cd /mnt/18T/jiangtangyunzhi/projects/recomm
task_probe_gpu="$1"
shift
export CUDA_VISIBLE_DEVICES="$task_probe_gpu"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec /home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python -u -m \
    experiment.phase18.analysis.s18_diff_gram_speed_probe --gpu "$task_probe_gpu" "$@"
