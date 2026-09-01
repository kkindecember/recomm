#!/usr/bin/env bash
set -u -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
SNAPSHOT="$ROOT/.runtime/phase16_s5_beauty_saux_gpu0_a1_runtime"
SESSION=phase16_s5_beauty_saux_gpu0_a1
OUTPUT="$ROOT/artifacts/phase16/s5_beauty_saux/formal/beauty_seed1502_gpu0_a1"
INNER=experiment/phase16/run_stage16_s5_beauty_saux_gpu0_a1_inner.sh

cd "$ROOT" || exit 2
if [[ -e "$OUTPUT" || -L "$OUTPUT" ]]; then
  echo "Refusing existing S16-5 Beauty artifact root; formal data cannot be overwritten." >&2
  exit 8
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Refusing existing tmux session $SESSION." >&2
  exit 8
fi

"$PYTHON" -m experiment.phase16.protocol.prepare_stage16_s5_beauty_gpu0_a1_runtime \
  prepare --snapshot-root "$SNAPSHOT" || exit 3
"$PYTHON" -m experiment.phase16.protocol.prepare_stage16_s5_beauty_gpu0_a1_runtime \
  verify --snapshot-root "$SNAPSHOT" || exit 3
if ! cmp -s "$ROOT/$INNER" "$SNAPSHOT/$INNER"; then
  echo "Refusing S16-5 because the inner runner differs from the isolated snapshot." >&2
  exit 3
fi

if ! tmux new-session -d -s "$SESSION" "cd '$SNAPSHOT'; exec bash '$INNER'"; then
  echo "Failed to create the S16-5 Beauty tmux session; no formal output was created." >&2
  exit 3
fi
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
  if [[ -s "$OUTPUT/status.json" ]]; then
    echo "STARTED $SESSION"
    echo "STATUS ${OUTPUT#$ROOT/}/status.json"
    exit 0
  fi
  sleep 1
done

echo "S16-5 Beauty tmux session started but status.json was not visible within twenty seconds." >&2
exit 3
