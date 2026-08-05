#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/artifacts/phase11/bw1_candidate_ceiling"
LOG="$OUT/run.log"
STATUS="$OUT/status.json"
SESSION=gram_phase11_bw1_candidate_ceiling
GPU=0
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
EVALUATOR="$ROOT/experiment/phase11/eval_bw1_candidate_ceiling.py"
SUMMARIZER="$ROOT/experiment/phase11/summarize_bw1_candidate_ceiling.py"
STARTED_AT=""

write_status() {
  local state=$1
  local stage=$2
  local reason=$3
  local tmp="${STATUS}.tmp.$$"
  mkdir -p "$OUT"
  printf '{"experiment_id":"GRAM_PHASE11_BW1_CANDIDATE_CEILING_VALIDATION_V1","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","physical_gpu":%s,"test_read":false,"sports_read":false}\n' \
    "$state" "$stage" "$reason" "$STARTED_AT" "$(date -Is)" "$GPU" > "$tmp"
  mv "$tmp" "$STATUS"
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
  write_status preflight preflight "Checking frozen inputs, tests, and GPU admission."
  PYTHONPATH="$ROOT/experiment/phase11" "$PYTHON" -m pytest -q \
    "$ROOT/experiment/phase11/test_bw1_candidate_ceiling.py" \
    "$ROOT/experiment/phase11/test_summarize_bw1.py" \
    "$ROOT/experiment/phase9/test_p9r_fresh_beam.py"
  "$PYTHON" -m py_compile "$EVALUATOR" "$SUMMARIZER"
  local free_mib
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GPU" | tr -d ' ')
  [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= 16000 )) || { write_status blocked gpu_admission "GPU${GPU} has ${free_mib:-unknown} MiB free; require 16000 MiB."; exit 3; }

  for dataset in Toys Beauty; do
    write_status running "${dataset}_fresh_beams" "Decoding independent validation beams at widths 50,100,200."
    timeout --signal=TERM 21600 env CUDA_VISIBLE_DEVICES="$GPU" PYTHONHASHSEED=2023 "$PYTHON" "$EVALUATOR" \
      --dataset "$dataset" --users 512 --widths 50 100 200 --device cuda:0 \
      --output-dir "$OUT/$dataset"
    local gate
    gate=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["integrity_gate"]["status"])' "$OUT/$dataset/summary.json")
    [[ "$gate" == passed ]] || { write_status stopped "${dataset}_integrity" "BW1 integrity gate failed; remaining work not attempted."; exit 0; }
  done
  write_status running aggregate "Aggregating cross-dataset candidate-ceiling decision."
  "$PYTHON" "$SUMMARIZER" --root "$OUT" --output "$OUT"
  local decision
  decision=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' "$OUT/summary.json")
  write_status succeeded finished "BW1 completed with preregistered decision=${decision}."
}

case "${1:-status}" in
  start)
    mkdir -p "$OUT"
    tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists: $SESSION" >&2; exit 1; }
    STARTED_AT=$(date -Is)
    printf -v launch_cmd 'bash %q worker %q >> %q 2>&1' "$0" "$STARTED_AT" "$LOG"
    tmux new-session -d -s "$SESSION" "$launch_cmd"
    write_status starting starting "Persistent Phase 11 BW1 validation session started."
    echo "started $SESSION"
    ;;
  worker) worker "${2:?missing start timestamp}" ;;
  status)
    tmux has-session -t "$SESSION" 2>/dev/null && echo "tmux session: running ($SESSION)" || echo "tmux session: not running ($SESSION)"
    [[ -f "$STATUS" ]] && sed -n '1,100p' "$STATUS" || echo '{"status":"not_started"}'
    [[ -f "$LOG" ]] && tail -n 30 "$LOG" || true
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
