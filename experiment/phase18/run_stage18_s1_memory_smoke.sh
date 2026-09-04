#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/18T/jiangtangyunzhi/projects/recomm
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python

cd "$ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT"

physical_gpu=${1:?physical GPU is required}
max_users=${2:?max users is required}
generation_cache=${3:?generation cache mode on/off is required}
cross_attention_cache=${4:?cross-attention cache mode on/off is required}
secondary_physical_gpu=${5:-}
if [[ -n "$secondary_physical_gpu" ]]; then
  export CUDA_VISIBLE_DEVICES="$physical_gpu,$secondary_physical_gpu"
  secondary_args=(--secondary-physical-gpu "$secondary_physical_gpu")
else
  export CUDA_VISIBLE_DEVICES="$physical_gpu"
  secondary_args=()
fi

exec "$PYTHON" experiment/phase18/protocol/s18_s1_memory_smoke.py \
  --physical-gpu "$physical_gpu" \
  --max-users "$max_users" \
  --generation-cache "$generation_cache" \
  --cross-attention-cache "$cross_attention_cache" \
  "${secondary_args[@]}"
