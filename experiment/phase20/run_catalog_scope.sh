#!/usr/bin/env bash
set -euo pipefail
cd /mnt/18T/jiangtangyunzhi/projects/recomm
exec /home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python -B \
  -m experiment.phase20.catalog_scope_job "${1:-launch}"
