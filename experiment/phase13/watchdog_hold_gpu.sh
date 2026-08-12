#!/usr/bin/env bash
# Watchdog: monitors a running experiment tmux session.
# When it exits, immediately grabs GPU memory via protector or lease fallback.
# Usage: bash experiment/phase13/watchdog_hold_gpu.sh <session_name> <gpu> <protector_tool>
#   e.g. bash experiment/phase13/watchdog_hold_gpu.sh gram_phase13_explore_v2_toys_iter2 0 ablation_scan
set -uo pipefail

SESSION=${1:?usage: $0 <session> <gpu> <protector_tool>}
GPU=${2:?}
PROTECTOR=${3:?}
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
LEASE_HELPER="$ROOT/experiment/gpu_memory_lease.py"

echo "[watchdog] monitoring $SESSION on GPU${GPU}, protector=$PROTECTOR"

# Wait for session to die
while tmux has-session -t "$SESSION" 2>/dev/null; do
  sleep 10
done

echo "[watchdog] $SESSION exited, grabbing GPU${GPU} memory..."
sleep 2

free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id="$GPU" 2>/dev/null | tr -d ' ')
echo "[watchdog] GPU${GPU} free: ${free_mib} MiB"

if [[ ! "$free_mib" =~ ^[0-9]+$ ]] || (( free_mib < 2048 )); then
  echo "[watchdog] too little free memory, cannot hold"
  exit 1
fi

hold_mib=$(( free_mib - 512 ))

# Try protector first
if [[ "$PROTECTOR" == "codellama" ]]; then
  HOLDER_RESERVE_MIB_OVERRIDE="$hold_mib" "$ROOT/tools/run_codellama.sh" start "$GPU" && {
    echo "[watchdog] codellama restored (${hold_mib} MiB) on GPU${GPU}"
    exit 0
  }
else
  RESERVE_MIB="$hold_mib" "$ROOT/tools/gram_ablation_scan.sh" start "$GPU" && {
    echo "[watchdog] scan restored (${hold_mib} MiB) on GPU${GPU}"
    exit 0
  }
fi

# Fallback: raw lease
echo "[watchdog] protector failed, using raw lease fallback"
"$PYTHON" "$LEASE_HELPER" --gpu "$GPU" --total-lease-mib "$hold_mib" \
  --expected-workload-peak-mib 1 \
  --status-path "$ROOT/artifacts/phase13/explore/watchdog_lease_gpu${GPU}.json"
