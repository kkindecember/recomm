#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
exec "$ROOT/experiment/phase2_toys/run_phase2_toys_job.sh" smoke "${1:?usage: run_phase2_toys_smoke.sh PHYSICAL_GPU}"
