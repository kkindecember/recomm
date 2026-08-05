#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/artifacts/phase11/bw3_p1_admission"
LOG="$OUT/run.log"
STATUS="$OUT/status.json"
SESSION=gram_phase11_bw3_p1_admission
GPU=4
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
GENERATOR="$ROOT/experiment/phase11/generate_bw3_pseudofuture_beams.py"
TRAINER="$ROOT/experiment/phase11/train_bw3_admission_gate.py"
STARTED_AT=""

write_status() {
  local state=$1
  local stage=$2
  local reason=$3
  local tmp="${STATUS}.tmp.$$"
  mkdir -p "$OUT"
  printf '{"experiment_id":"GRAM_PHASE11_BW3_P1_TRAIN_PREFIX_ADMISSION_V1","status":"%s","stage":"%s","reason":"%s","started_at":"%s","updated_at":"%s","physical_gpu":%s,"validation_target_read":false,"test_read":false,"sports_read":false}\n' \
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
  write_status preflight preflight "Checking code, tests, frozen inputs, and GPU admission."
  PYTHONPATH="$ROOT/experiment/phase11" "$PYTHON" -m pytest -q \
    "$ROOT/experiment/phase11/test_generate_bw3.py" \
    "$ROOT/experiment/phase11/test_bw3_admission_gate.py" \
    "$ROOT/experiment/phase11/test_bw3_pseudofuture.py"
  "$PYTHON" -m py_compile "$GENERATOR" "$TRAINER"
  local free_mib
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GPU" | tr -d ' ')
  [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= 30000 )) || { write_status blocked gpu_admission "GPU${GPU} has ${free_mib:-unknown} MiB free; require 30000 MiB."; exit 3; }

  for dataset in Toys Beauty; do
    for spec in "fit 4 1024" "calibration 3 512"; do
      read -r split offset users <<< "$spec"
      local unit="$OUT/$dataset/$split"
      write_status running "${dataset}_${split}_beams" "Generating offset-${offset} beam50/200 for ${users} users."
      timeout --signal=TERM 7200 env CUDA_VISIBLE_DEVICES="$GPU" PYTHONHASHSEED=2023 "$PYTHON" "$GENERATOR" \
        --dataset "$dataset" --offset "$offset" --users "$users" --device cuda:0 --output-dir "$unit"
      local gate
      gate=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["gate"]["status"])' "$unit/summary.json")
      [[ "$gate" == passed ]] || { write_status stopped "${dataset}_${split}_integrity" "Pseudo-future beam integrity failed; remaining work not attempted."; exit 0; }
    done
  done

  write_status running fit_calibration "Fitting linear admission gates and selecting margins on offset-3 only."
  "$PYTHON" "$TRAINER" --root "$OUT" --output-dir "$OUT/admission"
  local p1_gate
  p1_gate=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["p1_gate"]["status"])' "$OUT/admission/summary.json")
  if [[ "$p1_gate" == passed ]]; then
    write_status succeeded finished "BW3-P1 passed; one-shot validation P2 is eligible."
  else
    write_status stopped finished "BW3-P1 calibration failed; validation remains unread."
  fi
}

case "${1:-status}" in
  start)
    mkdir -p "$OUT"
    tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists: $SESSION" >&2; exit 1; }
    STARTED_AT=$(date -Is)
    printf -v launch_cmd 'bash %q worker %q >> %q 2>&1' "$0" "$STARTED_AT" "$LOG"
    tmux new-session -d -s "$SESSION" "$launch_cmd"
    write_status starting starting "Persistent BW3-P1 session started."
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
