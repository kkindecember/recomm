#!/usr/bin/env bash
set -euo pipefail
cd /mnt/18T/jiangtangyunzhi/projects/recomm
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
exec /home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python -u -m experiment.phase21.qwen_switch_controls supervise > artifacts/phase21/qwen_switch_controls_1024_20261017_v1/supervisor.log 2>&1
