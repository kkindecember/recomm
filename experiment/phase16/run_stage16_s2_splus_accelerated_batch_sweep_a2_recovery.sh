#!/usr/bin/env bash
set -u -o pipefail
CONFIG_PATH=experiment/phase16/configs/stage16_s2_splus_accelerated_batch_sweep_gpu5_a2_recovery.json \
OUTPUT_REL_PATH=artifacts/phase16/s2_splus_accelerated_batch_sweep/gpu5_a2_recovery \
ATTEMPT_ID_OVERRIDE=s16_s2_splus_accelerated_batch_sweep_gpu5_a2_recovery \
CANDIDATE_IDS="e32_g8_a32 e16_g4_a64 e8_g2_a128" \
EXACT_COMMAND_OVERRIDE="bash experiment/phase16/run_stage16_s2_splus_accelerated_batch_sweep_a2_recovery.sh 5" \
bash experiment/phase16/run_stage16_s2_splus_accelerated_batch_sweep.sh "${1:-5}"
