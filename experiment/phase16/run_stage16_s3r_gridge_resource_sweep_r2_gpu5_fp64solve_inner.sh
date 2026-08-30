#!/usr/bin/env bash
set -u -o pipefail

export S16_S3_CONFIG=experiment/phase16/configs/stage16_s3r_gridge_resource_sweep_r2_gpu5_fp64solve.json
export S16_S3_EXPERIMENT_ID=GRAM_PHASE16_S3R_GRIDGE_OBJECTIVE_RESOURCE_SWEEP
export S16_S3_ATTEMPT_ID=s16_s3r_gridge_resource_r2_gpu5_fp64solve
export S16_S3_OUTPUT_REL=artifacts/phase16/s3_genrecedit/inspired_ridge/resource_sweep/toys_seed1502_r2_gpu5_fp64solve
export S16_S3_EXACT_COMMAND="bash experiment/phase16/run_stage16_s3r_gridge_resource_sweep_r2_gpu5_fp64solve.sh"
export S16_S3_FIXED_GPU=5
export S16_S3_EXCLUDED_GPUS=0,4,7
export S16_S3_MINIMUM_FREE=18432
export S16_S3_EXPECTED_PEAK=12288
export S16_S3_HARD_TIMEOUT=900
export S16_S3_SUCCESS_CODE=PASS_S16_3R_GRIDGE_OBJECTIVE_RESOURCE_SWEEP
export S16_S3_SUCCESS_REASON="S16-3R G-RIDGE resource sweep completed after the isolated FP64-solve engineering correction; inspired formal Gate remains pending explicit execution confirmation."
export S16_S3_LINEAR_BLOCK_CODE=RESOURCE_BLOCKED_INSPIRED_RIDGE_LINEAR_SYSTEM
export S16_S3_LINEAR_BLOCK_REASON="The preregistered G-RIDGE solve/condition/residual contract failed; no pinv, jitter, resampling, fallback, or automatic retry was used."
export S16_S3_VALID_Z_BLOCK_CODE=RESOURCE_BLOCKED_INSPIRED_VALID_Z
export S16_S3_VALID_Z_BLOCK_REASON="At least one fixed G-RIDGE position subset produced no valid z; no outcome resampling or automatic retry was used."

exec bash experiment/phase16/run_stage16_s3_gfull_objective_resource_sweep.sh
