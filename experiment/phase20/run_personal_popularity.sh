#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
S20_PP_PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
S20_PP_SESSION=s20_personal_popularity_v1
case "${1:-status}" in
  start)
    if tmux has-session -t "$S20_PP_SESSION" 2>/dev/null; then
      echo "Personal-popularity session already exists."
      exit 1
    fi
    tmux new-session -d -s "$S20_PP_SESSION" "env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 $S20_PP_PYTHON -u -m experiment.phase20.run_personal_popularity --mode supervise"
    echo "Started $S20_PP_SESSION; see report/第二十阶段/EAGER_PPAC实验进展报告.md"
    ;;
  status)
    "$S20_PP_PYTHON" -c 'from pathlib import Path; p=Path("artifacts/phase20/personal_popularity_probe_v1/queue_status.json"); print(p.read_text() if p.exists() else "Not started")'
    ;;
  *) echo 'Usage: run_personal_popularity.sh {start|status}' >&2; exit 2 ;;
esac
