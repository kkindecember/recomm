#!/usr/bin/env bash
set -u -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SESSION=phase16_s3r_gridge_repeat_gpu5_f2
QUEUE_ROOT="$ROOT/artifacts/phase16/s3_genrecedit/inspired_ridge/stability/toys_seed1502_gpu5_f2"
FORMAL_STATUS="$ROOT/artifacts/phase16/s3_genrecedit/inspired_ridge/admission/toys_seed1502_gpu5_f2/status.json"
INNER=experiment/phase16/run_stage16_s3r_gridge_repeat_gpu5_f2_inner.sh

cd "$ROOT" || exit 2
if [[ ! -s "$FORMAL_STATUS" ]]; then
  echo "Refusing repeat queue before f2 reaches a recorded terminal state." >&2
  exit 7
fi
if [[ -e "$QUEUE_ROOT" ]]; then
  echo "Refusing existing f2 repeat root; cycles are never overwritten." >&2
  exit 8
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Refusing existing tmux session $SESSION." >&2
  exit 8
fi

tmux new-session -d -s "$SESSION" "cd '$ROOT' && exec bash '$INNER'"
for _ in 1 2 3 4 5 6 7 8 9 10; do
  if [[ -s "$QUEUE_ROOT/status.json" ]]; then
    echo "STARTED $SESSION"
    exit 0
  fi
  sleep 1
done
echo "f2 repeat session started but status.json was not visible within ten seconds." >&2
exit 3
