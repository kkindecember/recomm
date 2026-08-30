#!/usr/bin/env bash
set -u -o pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SESSION=phase16_s3r_gridge_formal_gpu5_f1
OUTPUT="$ROOT/artifacts/phase16/s3_genrecedit/inspired_ridge/admission/toys_seed1502_gpu5_f1"
INNER=experiment/phase16/run_stage16_s3r_gridge_formal_admission_gpu5_f1_inner.sh

cd "$ROOT" || exit 2
if [[ -e "$OUTPUT" ]]; then
  echo "Refusing existing formal S16-3R artifact root; retry/resume requires explicit review and a separately authorized command." >&2
  exit 8
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Refusing existing tmux session $SESSION." >&2
  exit 8
fi

tmux new-session -d -s "$SESSION" "cd '$ROOT' && exec bash '$INNER'"
for _ in 1 2 3 4 5 6 7 8 9 10; do
  if [[ -s "$OUTPUT/status.json" ]]; then
    echo "STARTED $SESSION"
    exit 0
  fi
  sleep 1
done

echo "Formal tmux session started but status.json was not visible within ten seconds; inspect the isolated session/artifact." >&2
exit 3
