#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="$ROOT/artifacts/phase9/p9r_toys_fresh_beam_512"
LOG="$OUTPUT/run.log"
STATUS="$OUTPUT/status.json"
SESSION=gram_phase9_p9r_toys_fresh_beam_512
GPU=3
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
SCRIPT="$ROOT/experiment/phase9/eval_p9r_fresh_beam.py"
TEST="$ROOT/experiment/phase9/test_p9r_fresh_beam.py"
PLAN="$ROOT/plan/第九阶段/GRAM_第九阶段_P9-R原Checkpoint新鲜Beam复现计划.md"
CHECKPOINT="$ROOT/GRAM/log/Toys/1_20260720_1830/id_0_rec_30/model_rec_phase_1_epoch_30.pt"
CACHE="$ROOT/GRAM/preds/20260722_020042_Toys_sequential_pred_validation.tsv"
ITEM_HEAD="$ROOT/artifacts/phase9/cf0_b2_toys_item_p2a/best_item_head.pt"
STARTED_AT=""

write_status() {
  local state=$1 stage=$2 reason=$3 tmp="${STATUS}.tmp.$$"
  mkdir -p "$OUTPUT"
  printf '{"experiment_id":"GRAM_PHASE9_P9R_TOYS_FRESH_BEAM_512_V1","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","tmux_session":"%s","physical_gpu":%s,"test_read":false,"beauty_read":false,"sports_read":false,"log_path":"%s"}\n' \
    "$state" "$stage" "$reason" "$STARTED_AT" "$(date -Is)" "$SESSION" "$GPU" "${LOG#$ROOT/}" > "$tmp"
  mv "$tmp" "$STATUS"
}

worker() {
  STARTED_AT=${1:?missing start timestamp}
  trap 'write_status failed interrupted "P9-R interrupted; no automatic retry."; exit 130' INT TERM HUP
  cd "$ROOT"
  mkdir -p "$OUTPUT"
  write_status preflight preflight "Checking frozen inputs, offline model cache, tests, and GPU admission."
  for required in "$SCRIPT" "$TEST" "$PLAN" "$CHECKPOINT" "$CACHE" "$ITEM_HEAD"; do
    [[ -s "$required" ]] || { write_status blocked preflight "Missing required input: $required"; exit 2; }
  done
  "$PYTHON" -m pytest -q "$TEST"
  "$PYTHON" -m py_compile "$SCRIPT"
  env HF_HUB_CACHE="$ROOT/.cache/huggingface" TRANSFORMERS_CACHE="$ROOT/.cache/huggingface/transformers" TRANSFORMERS_OFFLINE=1 \
    "$PYTHON" -c 'from transformers import AutoTokenizer,T5Config; AutoTokenizer.from_pretrained("t5-small",local_files_only=True); T5Config.from_pretrained("t5-small",local_files_only=True)'
  local free_mib
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GPU" | tr -d ' ')
  [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= 16000 )) || { write_status blocked gpu_admission "GPU${GPU} has ${free_mib:-unknown} MiB free; require 16000 MiB."; exit 3; }
  write_status running fresh_decode "512-user fresh constrained decoding and frozen PCRF evaluation running."
  timeout --signal=TERM 21600 env CUDA_VISIBLE_DEVICES="$GPU" PYTHONHASHSEED=2023 \
    HF_HUB_CACHE="$ROOT/.cache/huggingface" TRANSFORMERS_CACHE="$ROOT/.cache/huggingface/transformers" TRANSFORMERS_OFFLINE=1 \
    "$PYTHON" "$SCRIPT" --users 512 --device cuda:0 --seed 2023 --output-dir "$OUTPUT"
  write_status succeeded finished "P9-R completed; inspect summary.json scientific_gate."
}

case "${1:-status}" in
  start)
    mkdir -p "$OUTPUT"
    tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists: $SESSION" >&2; exit 1; }
    STARTED_AT=$(date -Is)
    printf -v launch_cmd 'bash %q worker %q >> %q 2>&1' "$0" "$STARTED_AT" "$LOG"
    tmux new-session -d -s "$SESSION" "$launch_cmd"
    write_status starting starting "Persistent P9-R session started."
    echo "started $SESSION"
    ;;
  worker) worker "${2:?missing start timestamp}" ;;
  status)
    tmux has-session -t "$SESSION" 2>/dev/null && echo "tmux session: running ($SESSION)" || echo "tmux session: not running ($SESSION)"
    [[ -f "$STATUS" ]] && sed -n '1,100p' "$STATUS" || echo '{"status":"not_started"}'
    [[ -f "$LOG" ]] && tail -n 40 "$LOG" || true
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
