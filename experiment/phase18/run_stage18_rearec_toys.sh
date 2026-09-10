#!/usr/bin/env bash
set -euo pipefail
cd /mnt/18T/jiangtangyunzhi/projects/recomm
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=GPU-4e97077a-5bab-99a7-2fdc-598df6be74cc
exec artifacts/phase17/fullport/envs/latte_05e4e6d98322_torch_2_7_1_cu126/bin/python -u -m experiment.phase18.protocol.s18_rearec_toys \
  --config experiment/phase18/config/s18_rearec_toys.json "$@"
