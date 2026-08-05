#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_OUT="$ROOT/artifacts/phase9/p9x_beauty_item_head"
VAL_OUT="$ROOT/artifacts/phase9/p9x_beauty_validation"
SMOKE_OUT="$ROOT/artifacts/phase9/p9x_beauty_item_head_smoke"
LOG="$VAL_OUT/run.log"
STATUS="$VAL_OUT/status.json"
SESSION=gram_phase9_p9x_beauty_validation
GPU=3
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
TRAINER="$ROOT/experiment/phase9/train_cf0_b2_item_head.py"
EVALUATOR="$ROOT/experiment/phase9/eval_p9x_fixed_pcrf.py"
TEST1="$ROOT/experiment/phase9/test_cf0_b2_item_head.py"
TEST2="$ROOT/experiment/phase9/test_p9x_fixed_pcrf.py"
DATA="$ROOT/GRAM/rec_datasets/Beauty"
INDEX=item_generative_indexing_hierarchy_v1_c128_l7_len32768_split.txt
PREDICTIONS="$ROOT/GRAM/preds/20260722_125916_Beauty_sequential_pred_validation.tsv"
STARTED_AT=""

write_status() {
  local state=$1 stage=$2 reason=$3 tmp="${STATUS}.tmp.$$"
  mkdir -p "$VAL_OUT"
  printf '{"experiment_id":"GRAM_PHASE9_P9X_BEAUTY_VALIDATION_V1","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","tmux_session":"%s","physical_gpu":%s,"test_read":false,"sports_read":false}\n' \
    "$state" "$stage" "$reason" "$STARTED_AT" "$(date -Is)" "$SESSION" "$GPU" > "$tmp"
  mv "$tmp" "$STATUS"
}

worker() {
  STARTED_AT=${1:?missing start timestamp}
  trap 'write_status failed interrupted "Interrupted; no automatic retry."; exit 130' INT TERM HUP
  cd "$ROOT"
  mkdir -p "$TRAIN_OUT" "$VAL_OUT"
  write_status preflight preflight "Checking code, Beauty train/validation inputs, tests and GPU admission."
  for required in "$TRAINER" "$EVALUATOR" "$TEST1" "$TEST2" "$DATA/$INDEX" "$DATA/user_sequence.txt" "$PREDICTIONS"; do
    [[ -s "$required" ]] || { write_status blocked preflight "Missing required input: $required"; exit 2; }
  done
  "$PYTHON" -m pytest -q "$TEST1" "$TEST2"
  "$PYTHON" -m py_compile "$TRAINER" "$EVALUATOR"
  local free_mib
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GPU" | tr -d ' ')
  [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= 12000 )) || { write_status blocked gpu_admission "GPU${GPU} has ${free_mib:-unknown} MiB free; require 12000 MiB."; exit 3; }

  write_status running gpu_smoke "Beauty item-head 1024-train/512-validation smoke running."
  timeout --signal=TERM 600 env CUDA_VISIBLE_DEVICES="$GPU" PYTHONHASHSEED=2023 "$PYTHON" "$TRAINER" \
    --data-dir "$DATA" --item-index-name "$INDEX" --dataset-name Beauty \
    --experiment-id GRAM_PHASE9_P9X_BEAUTY_ITEM_HEAD_SMOKE_V1 --device cuda:0 --epochs 1 \
    --batch-size 512 --eval-batch-size 1024 --max-train-samples 1024 \
    --max-validation-samples 512 --output-dir "$SMOKE_OUT"

  write_status running item_head_training "Beauty isolated item-head 10-epoch training/validation running."
  timeout --signal=TERM 14400 env CUDA_VISIBLE_DEVICES="$GPU" PYTHONHASHSEED=2023 "$PYTHON" "$TRAINER" \
    --data-dir "$DATA" --item-index-name "$INDEX" --dataset-name Beauty \
    --experiment-id GRAM_PHASE9_P9X_BEAUTY_ITEM_HEAD_V1 --device cuda:0 --epochs 10 \
    --batch-size 512 --eval-batch-size 1024 --learning-rate 3e-4 --weight-decay 0.01 \
    --warmup-ratio 0.05 --max-history 20 --d-model 512 --num-layers 2 --num-heads 4 \
    --dropout 0.1 --temperature 0.07 --gate-relative-margin 0.20 \
    --nonhead-recall50-min 0.005 --output-dir "$TRAIN_OUT"
  local item_gate
  item_gate=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["scientific_gate"]["status"])' "$TRAIN_OUT/summary.json")
  [[ "$item_gate" == passed ]] || { write_status stopped item_head_gate "Beauty item-head gate failed; PCRF validation and test not read."; exit 0; }

  write_status running pcrf_validation "Frozen PCRF Beauty validation admission running."
  "$PYTHON" "$EVALUATOR" --dataset Beauty --data-dir "$DATA" --item-index-name "$INDEX" \
    --predictions "$PREDICTIONS" --item-head "$TRAIN_OUT/best_item_head.pt" \
    --output-dir "$VAL_OUT" --mode validation --lambda-weight 1.0 --beta 0.5 --gamma 1.0 \
    --bootstrap-replicates 2000 --seed 2023
  local gate
  gate=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["validation_admission"]["status"])' "$VAL_OUT/summary.json")
  if [[ "$gate" == passed ]]; then
    write_status succeeded finished "Beauty validation admission passed; one-shot test is now eligible but was not read."
  else
    write_status stopped finished "Beauty validation admission failed; test remains unread."
  fi
}

case "${1:-status}" in
  start)
    mkdir -p "$VAL_OUT"
    tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists: $SESSION" >&2; exit 1; }
    STARTED_AT=$(date -Is)
    printf -v launch_cmd 'bash %q worker %q >> %q 2>&1' "$0" "$STARTED_AT" "$LOG"
    tmux new-session -d -s "$SESSION" "$launch_cmd"
    write_status starting starting "Persistent Beauty train/validation session started."
    echo "started $SESSION"
    ;;
  worker) worker "${2:?missing start timestamp}" ;;
  status)
    tmux has-session -t "$SESSION" 2>/dev/null && echo "tmux session: running ($SESSION)" || echo "tmux session: not running ($SESSION)"
    [[ -f "$STATUS" ]] && sed -n '1,100p' "$STATUS" || echo '{"status":"not_started"}'
    [[ -f "$LOG" ]] && tail -n 60 "$LOG" || true
    ;;
  stop)
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      pane_pid=$(tmux list-panes -t "$SESSION" -F '#{pane_pid}' | head -n 1)
      kill -TERM "$pane_pid"
      echo "stop requested for $SESSION"
    else
      echo "session not running: $SESSION"
    fi
    ;;
  *) echo "usage: $0 {start|status|stop|worker}" >&2; exit 2 ;;
esac
