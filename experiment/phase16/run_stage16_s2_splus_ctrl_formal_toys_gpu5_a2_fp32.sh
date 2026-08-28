#!/usr/bin/env bash
set -u -o pipefail

export FORMAL_CONFIG=experiment/phase16/configs/stage16_s2_splus_ctrl_formal_toys_gpu5_a2_fp32.json
export FORMAL_OUTPUT_REL=artifacts/phase16/s2_splus_ctrl_formal/toys_seed1502_gpu5_a2_fp32
export FORMAL_ATTEMPT_ID=s16_s2_splus_ctrl_formal_toys_gpu5_a2_fp32
export FORMAL_EXACT_COMMAND="bash experiment/phase16/run_stage16_s2_splus_ctrl_formal_toys_gpu5_a2_fp32.sh ${1:-5}"

exec bash experiment/phase16/run_stage16_s2_splus_ctrl_formal_toys_gpu5_a1_fp32.sh "$@"
