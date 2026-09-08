#!/usr/bin/env bash
set -euo pipefail
cd /mnt/18T/jiangtangyunzhi/projects/recomm
task_family="$1"
export CUDA_VISIBLE_DEVICES="$2"
shift 2
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
if [[ "$task_family" == diff_gram ]]; then
    task_python=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
elif [[ "$task_family" == diffgrm ]]; then
    task_python=artifacts/phase17/fullport/envs/latte_05e4e6d98322_torch_2_7_1_cu126/bin/python
else
    exit 2
fi
exec "$task_python" -u -m experiment.phase18.protocol.s18_full_budget_resume "$@"
