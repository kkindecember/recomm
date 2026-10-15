#!/usr/bin/env bash
set -euo pipefail
cd /mnt/18T/jiangtangyunzhi/projects/recomm
run_dir=artifacts/phase20/lime_early_correction_20260922_v1
python_bin=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
mkdir -p "$run_dir"
timeout 600 "$python_bin" -u -m experiment.phase20.lime_early_correction 2>&1 | tee "$run_dir/run.log"
timeout 600 "$python_bin" -u -m experiment.phase20.review_lime_early_correction 2>&1 | tee "$run_dir/verification.log"
