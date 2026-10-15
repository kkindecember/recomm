#!/usr/bin/env bash
set -euo pipefail
cd /mnt/18T/jiangtangyunzhi/projects/recomm
exec /home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python -m experiment.phase20.lime_eligible_path_screen "${@:-supervise}"
