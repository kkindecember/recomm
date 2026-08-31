#!/usr/bin/env bash
set -u -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
SNAPSHOT="$ROOT/.runtime/phase16_s4_toys_gpu0_a3_runtime"
SESSION=phase16_s4_toys_standalone_gpu0_a3
OUTPUT="$ROOT/artifacts/phase16/s4_toys_standalone/formal/toys_seed1502_gpu0_a3"
INNER=experiment/phase16/run_stage16_s4_toys_standalone_gpu0_a3_inner.sh

cd "$ROOT" || exit 2
if [[ -e "$OUTPUT" || -L "$OUTPUT" ]]; then
  echo "Refusing existing S16-4 GPU0 a3 artifact root; formal data cannot be overwritten." >&2
  exit 8
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Refusing existing tmux session $SESSION." >&2
  exit 8
fi

"$PYTHON" -m experiment.phase16.protocol.prepare_stage16_s4_gpu0_a3_runtime \
  prepare --snapshot-root "$SNAPSHOT" || exit 3
"$PYTHON" -m experiment.phase16.protocol.prepare_stage16_s4_gpu0_a3_runtime \
  verify --snapshot-root "$SNAPSHOT" || exit 3
if ! cmp -s "$ROOT/$INNER" "$SNAPSHOT/$INNER"; then
  echo "Refusing S16-4 because the inner runner differs from the isolated snapshot." >&2
  exit 3
fi

if ! tmux new-session -d -s "$SESSION" "cd '$SNAPSHOT'; exec bash '$INNER'"; then
  echo "Failed to create the S16-4 GPU0 tmux session; no formal output was created." >&2
  exit 3
fi
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
  if [[ -s "$OUTPUT/status.json" ]]; then
    echo "STARTED $SESSION"
    exit 0
  fi
  sleep 1
done

echo "S16-4 GPU0 tmux session started but status.json was not visible within twenty seconds." >&2
exit 3
