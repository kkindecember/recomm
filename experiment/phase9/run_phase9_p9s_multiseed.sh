#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/artifacts/phase9/p9s_multiseed"
LOG="$OUT/run.log"
STATUS="$OUT/status.json"
SESSION=gram_phase9_p9s_multiseed
GPU=0
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
TRAINER="$ROOT/experiment/phase9/train_cf0_b2_item_head.py"
EVALUATOR="$ROOT/experiment/phase9/eval_p9x_fixed_pcrf.py"
SUMMARIZER="$ROOT/experiment/phase9/summarize_p9s_multiseed.py"
STARTED_AT=""

write_status() {
  local state=$1 stage=$2 reason=$3 tmp="${STATUS}.tmp.$$"
  mkdir -p "$OUT"
  printf '{"experiment_id":"GRAM_PHASE9_P9S_MULTISEED_VALIDATION_V1","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","physical_gpu":%s,"test_read":false,"sports_read":false}\n' \
    "$state" "$stage" "$reason" "$STARTED_AT" "$(date -Is)" "$GPU" > "$tmp"
  mv "$tmp" "$STATUS"
}

dataset_values() {
  local dataset=$1
  if [[ "$dataset" == Toys ]]; then
    DATA="$ROOT/GRAM/rec_datasets/Toys"
    INDEX=item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt
    PREDICTIONS="$ROOT/GRAM/preds/20260722_020042_Toys_sequential_pred_validation.tsv"
    EXISTING="$ROOT/artifacts/phase9/cf0_b2_toys_item_p2a/best_item_head.pt"
  else
    DATA="$ROOT/GRAM/rec_datasets/Beauty"
    INDEX=item_generative_indexing_hierarchy_v1_c128_l7_len32768_split.txt
    PREDICTIONS="$ROOT/GRAM/preds/20260722_125916_Beauty_sequential_pred_validation.tsv"
    EXISTING="$ROOT/artifacts/phase9/p9x_beauty_item_head/best_item_head.pt"
  fi
}

evaluate_seed() {
  local dataset=$1
  local seed=$2
  local checkpoint=$3
  local base="$OUT/$dataset/seed$seed"
  mkdir -p "$base/validation"
  "$PYTHON" "$EVALUATOR" --dataset "$dataset" --data-dir "$DATA" --item-index-name "$INDEX" \
    --predictions "$PREDICTIONS" --item-head "$checkpoint" --output-dir "$base/validation" \
    --mode validation --lambda-weight 1.0 --beta 0.5 --gamma 1.0 \
    --bootstrap-replicates 2000 --seed 2023
}

worker() {
  STARTED_AT=${1:?missing start timestamp}
  finish() {
    local rc=$?
    trap - EXIT INT TERM HUP
    if (( rc != 0 )); then
      write_status failed failed "Engineering exit=${rc}; no automatic retry."
    fi
    exit "$rc"
  }
  trap finish EXIT
  trap 'write_status failed interrupted "Interrupted; no automatic retry."; exit 130' INT TERM HUP
  cd "$ROOT"
  mkdir -p "$OUT"
  write_status preflight preflight "Checking validation-only inputs, tests, and GPU admission."
  "$PYTHON" -m pytest -q "$ROOT/experiment/phase9/test_cf0_b2_item_head.py" \
    "$ROOT/experiment/phase9/test_p9x_fixed_pcrf.py" "$ROOT/experiment/phase9/test_p9s_multiseed.py"
  "$PYTHON" -m py_compile "$TRAINER" "$EVALUATOR" "$SUMMARIZER"
  local free_mib
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GPU" | tr -d ' ')
  [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= 12000 )) || { write_status blocked gpu_admission "GPU${GPU} has ${free_mib:-unknown} MiB free; require 12000 MiB."; exit 3; }

  for dataset in Toys Beauty; do
    dataset_values "$dataset"
    write_status running "${dataset}_seed2023_validation" "Evaluating frozen seed 2023 checkpoint."
    evaluate_seed "$dataset" 2023 "$EXISTING"
    for seed in 2024 2025; do
      local base="$OUT/$dataset/seed$seed"
      local checkpoint="$base/item_head/best_item_head.pt"
      mkdir -p "$base/item_head"
      write_status running "${dataset}_seed${seed}_training" "Training isolated item-head seed ${seed}."
      timeout --signal=TERM 14400 env CUDA_VISIBLE_DEVICES="$GPU" PYTHONHASHSEED="$seed" "$PYTHON" "$TRAINER" \
        --data-dir "$DATA" --item-index-name "$INDEX" --dataset-name "$dataset" \
        --experiment-id "GRAM_PHASE9_P9S_${dataset^^}_ITEM_HEAD_SEED${seed}_V1" \
        --device cuda:0 --seed "$seed" --epochs 10 --batch-size 512 --eval-batch-size 1024 \
        --learning-rate 3e-4 --weight-decay 0.01 --warmup-ratio 0.05 --max-history 20 \
        --d-model 512 --num-layers 2 --num-heads 4 --dropout 0.1 --temperature 0.07 \
        --gate-relative-margin 0.20 --nonhead-recall50-min 0.005 --output-dir "$base/item_head"
      local gate
      gate=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["scientific_gate"]["status"])' "$base/item_head/summary.json")
      [[ "$gate" == passed ]] || { write_status stopped "${dataset}_seed${seed}_item_gate" "New item-head gate failed; remaining runs not attempted."; exit 0; }
      write_status running "${dataset}_seed${seed}_validation" "Evaluating frozen PCRF validation seed ${seed}."
      evaluate_seed "$dataset" "$seed" "$checkpoint"
    done
  done
  write_status running aggregate "Aggregating six validation units."
  "$PYTHON" "$SUMMARIZER" --root "$OUT" --output "$OUT"
  local gate
  gate=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["robustness_gate"]["status"])' "$OUT/summary.json")
  if [[ "$gate" == passed ]]; then
    write_status succeeded finished "P9-S robustness gate passed; Phase 11 beam-width pilot is eligible."
  else
    write_status stopped finished "P9-S robustness gate failed; stop before Phase 11."
  fi
}

case "${1:-status}" in
  start)
    mkdir -p "$OUT"
    tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists: $SESSION" >&2; exit 1; }
    STARTED_AT=$(date -Is)
    printf -v launch_cmd 'bash %q worker %q >> %q 2>&1' "$0" "$STARTED_AT" "$LOG"
    tmux new-session -d -s "$SESSION" "$launch_cmd"
    write_status starting starting "Persistent P9-S multiseed validation session started."
    echo "started $SESSION"
    ;;
  worker) worker "${2:?missing start timestamp}" ;;
  status)
    tmux has-session -t "$SESSION" 2>/dev/null && echo "tmux session: running ($SESSION)" || echo "tmux session: not running ($SESSION)"
    [[ -f "$STATUS" ]] && sed -n '1,100p' "$STATUS" || echo '{"status":"not_started"}'
    [[ -f "$LOG" ]] && tail -n 50 "$LOG" || true
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
