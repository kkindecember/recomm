#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
S20_REFERENCE_PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
S20_REFERENCE_SESSION=s20_conditional_reference_toys_v1
case "${1:-status}" in
  start)
    if tmux has-session -t "$S20_REFERENCE_SESSION" 2>/dev/null; then
      echo "Conditional-reference session already exists."
      exit 1
    fi
    tmux new-session -d -s "$S20_REFERENCE_SESSION" "env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 $S20_REFERENCE_PYTHON -u -m experiment.phase20.conditional_reference_screen --mode supervise"
    echo "Started $S20_REFERENCE_SESSION"
    ;;
  status)
    "$S20_REFERENCE_PYTHON" -c 'from pathlib import Path; p=Path("artifacts/phase20/conditional_reference_toys_v1"); [print(str(p/n)+"\n"+(p/n).read_text()) for n in ["queue_status.json", "run/status.json"] if (p/n).exists()]'
    ;;
  *) echo 'Usage: run_conditional_reference.sh {start|status}' >&2; exit 2 ;;
esac
