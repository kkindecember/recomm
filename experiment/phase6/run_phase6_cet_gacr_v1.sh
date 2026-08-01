#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jiangtangyunzhi/projects/recomm
export CONFIG="$ROOT/artifacts/phase6/configs/cet_gacr_v1_preregistered.json"
export OUTPUT="$ROOT/artifacts/phase6/cet_gacr_v1"
export SESSION=gram_phase6_cet_gacr_v1
export EXPERIMENT_ID=GRAM_PHASE6_CET_GACR_V1
export RUN_LABEL=CET-v1-x-GACR-v3
export WORKLOAD_SCRIPT="$ROOT/experiment/phase6/cet_gacr_v1.py"
export WORKLOAD_STAGE=cet_v1_x_frozen_gacr_v3_four_arm_validation

exec bash "$ROOT/experiment/phase6/run_phase6_gacr_v2.sh" "$@"
