#!/usr/bin/env bash
set -euo pipefail
cd /mnt/18T/jiangtangyunzhi/projects/recomm
exec /home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python -m experiment.phase20.lime_prefix_expert "${@:-supervise}"
