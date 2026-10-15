#!/usr/bin/env bash
set -euo pipefail
cd /mnt/18T/jiangtangyunzhi/projects/recomm
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export RECOMM_ATTRIBUTE_OUT=artifacts/phase21/attribute_retention_beauty_recovery_20260923_v1
export RECOMM_ATTRIBUTE_REPORT=report/第二十一阶段/属性保留Beauty资源恢复_20260923.md
export RECOMM_ATTRIBUTE_DOMAINS=Beauty
exec /home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python -m experiment.phase21.attribute_retention supervise
