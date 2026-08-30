#!/usr/bin/env bash
set -u -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SESSION=phase16_s3r_gridge_resource_r2_gpu5_fp64solve
OUTPUT="$ROOT/artifacts/phase16/s3_genrecedit/inspired_ridge/resource_sweep/toys_seed1502_r2_gpu5_fp64solve"
INNER=experiment/phase16/run_stage16_s3r_gridge_resource_sweep_r2_gpu5_fp64solve_inner.sh

cd "$ROOT" || exit 2
if [[ -e "$OUTPUT" ]]; then
  echo "Refusing existing S16-3R G-RIDGE r2 artifact root; retries require a new attempt." >&2
  exit 8
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Refusing existing tmux session $SESSION." >&2
  exit 8
fi

tmux new-session -d -s "$SESSION" "cd '$ROOT' && exec bash '$INNER'"
for _ in 1 2 3 4 5; do
  if [[ -s "$OUTPUT/status.json" ]]; then
    echo "STARTED $SESSION"
    exit 0
  fi
  sleep 1
done

echo "tmux session started but status.json was not visible within five seconds; inspect the isolated session/artifact." >&2
exit 3
