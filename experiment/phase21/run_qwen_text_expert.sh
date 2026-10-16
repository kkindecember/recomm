#!/usr/bin/env bash
set -euo pipefail
cd /mnt/18T/jiangtangyunzhi/projects/recomm
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export HF_HOME=/mnt/18T/jiangtangyunzhi/projects/recomm/.cache/qwen_text_expert
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=60
.runtime/qwen_embedding_20261015/env/bin/python -u -m experiment.phase21.qwen_text_encoder download
exec /home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python -u -m experiment.phase21.qwen_text_expert supervise
