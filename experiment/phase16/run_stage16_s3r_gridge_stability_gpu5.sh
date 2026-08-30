#!/usr/bin/env bash
set -u -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SESSION=phase16_s3r_gridge_stability_gpu5
QUEUE_ROOT="$ROOT/artifacts/phase16/s3_genrecedit/inspired_ridge/stability/toys_seed1502_gpu5"
FORMAL_COMPLETION="$ROOT/artifacts/phase16/s3_genrecedit/inspired_ridge/admission/toys_seed1502_gpu5_f1/authoritative_completion.json"
INNER=experiment/phase16/run_stage16_s3r_gridge_stability_gpu5_inner.sh

cd "$ROOT" || exit 2
if [[ ! -s "$FORMAL_COMPLETION" ]]; then
  echo "Refusing stability queue before authoritative S16-3 completion." >&2
  exit 7
fi
if [[ -e "$QUEUE_ROOT" ]]; then
  echo "Refusing existing stability queue root; no implicit restart." >&2
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
echo "Stability queue session started but status.json was not visible within ten seconds." >&2
exit 3
