#!/usr/bin/env bash
set -u -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
SNAPSHOT="$ROOT/.runtime/phase16_s3r_gridge_f3_runtime"
SESSION=phase16_s3r_gridge_formal_gpu5_f3
OUTPUT="$ROOT/artifacts/phase16/s3_genrecedit/inspired_ridge/admission/toys_seed1502_gpu5_f3"
INNER=experiment/phase16/run_stage16_s3r_gridge_formal_admission_gpu5_f3_inner.sh

cd "$ROOT" || exit 2
if [[ -e "$OUTPUT" ]]; then
  echo "Refusing existing f3 artifact root; f3 cannot be overwritten." >&2
  exit 8
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Refusing existing tmux session $SESSION." >&2
  exit 8
fi

"$PYTHON" -m experiment.phase16.protocol.prepare_s3r_gridge_f3_runtime \
  prepare --snapshot-root "$SNAPSHOT" || exit 3
"$PYTHON" -m experiment.phase16.protocol.prepare_s3r_gridge_f3_runtime \
  verify --snapshot-root "$SNAPSHOT" || exit 3
if ! cmp -s "$ROOT/$INNER" "$SNAPSHOT/$INNER"; then
  echo "Refusing f3 because the launch wrapper differs from its isolated snapshot." >&2
  exit 3
fi

tmux new-session -d -s "$SESSION" "cd '$SNAPSHOT'; exec bash '$INNER'"
for _ in 1 2 3 4 5 6 7 8 9 10; do
  if [[ -s "$OUTPUT/status.json" ]]; then
    echo "STARTED $SESSION"
    exit 0
  fi
  sleep 1
done

echo "f3 tmux session started but status.json was not visible within ten seconds; inspect the isolated session/artifact." >&2
exit 3
