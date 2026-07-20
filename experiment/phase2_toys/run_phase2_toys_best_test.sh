#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
GPU=${1:?usage: run_phase2_toys_best_test.sh PHYSICAL_GPU BEST_CHECKPOINT}
BEST_CHECKPOINT=${2:?usage: run_phase2_toys_best_test.sh PHYSICAL_GPU BEST_CHECKPOINT}
exec "$ROOT/experiment/phase2_toys/run_phase2_toys_job.sh" best_test "$GPU" "$BEST_CHECKPOINT"
