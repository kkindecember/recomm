#!/usr/bin/env bash
set -euo pipefail
cd /mnt/18T/jiangtangyunzhi/projects/recomm
exec /home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python -m experiment.phase20.guided_decoding_job "${1:-launch}"
