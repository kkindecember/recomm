#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/18T/jiangtangyunzhi/projects/recomm"
PYTHON="/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python"

cd "$ROOT"
exec "$PYTHON" -m experiment.phase18.protocol.s18_s1_prepare "$@"
