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

exec "$PYTHON" experiment/phase18/protocol/s18_s1_resource_recovery.py "$@"
