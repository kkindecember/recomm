#!/usr/bin/env bash
set -euo pipefail
cd /mnt/18T/jiangtangyunzhi/projects/recomm
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
action="${1:-status}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$action" == "native" ]]; then
  export RECOMM_CLEAN_GRAM_CONFIG=experiment/phase20/lime_clean_gram_20260920_native_v2.json
  action="${1:-status}"
  if [[ $# -gt 0 ]]; then shift; fi
fi
case "$action" in
  launch|supervise|status)
    exec /home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python -B \
      -m experiment.phase20.lime_clean_gram_job "$action" "$@"
    ;;
  prepare|cpu_check)
    export CUDA_VISIBLE_DEVICES=''
    exec /home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python -B \
      -m experiment.phase20.lime_clean_gram_worker "$action" "$@"
    ;;
  *) echo "Usage: $0 {prepare|cpu_check --domain Toys|launch|status}" >&2; exit 2 ;;
esac
