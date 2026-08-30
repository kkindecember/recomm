#!/usr/bin/env bash
set -u -o pipefail

export S16_S3_CONFIG=experiment/phase16/configs/stage16_s3_gfull_objective_resource_sweep_a2.json
export S16_S3_ATTEMPT_ID=s16_s3_gfull_resource_a2
export S16_S3_OUTPUT_REL=artifacts/phase16/s3_genrecedit/resource_sweep/toys_seed1502_a2
export S16_S3_EXACT_COMMAND="bash experiment/phase16/run_stage16_s3_gfull_objective_resource_sweep_a2.sh"

exec bash experiment/phase16/run_stage16_s3_gfull_objective_resource_sweep.sh
