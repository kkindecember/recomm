#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jiangtangyunzhi/projects/recomm
export CONFIG="$ROOT/artifacts/phase6/configs/gacr_v3_preregistered.json"
export OUTPUT="$ROOT/artifacts/phase6/gacr_v3"
export SESSION=gram_phase6_gacr_v3
export EXPERIMENT_ID=GRAM_PHASE6_GACR_V3
export RUN_LABEL=GACR-v3
export WORKLOAD_SCRIPT="$ROOT/experiment/phase6/gacr_v3.py"
export WORKLOAD_STAGE=gacr_v3_target_free_safety_pilot

exec bash "$ROOT/experiment/phase6/run_phase6_gacr_v2.sh" "$@"
