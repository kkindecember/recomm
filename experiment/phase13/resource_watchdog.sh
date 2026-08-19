#!/usr/bin/env bash
# Independent per-experiment GPU-holder recovery watchdog.
#
# It never kills a process. While the experiment tmux session exists it only
# monitors. After that session disappears it verifies the exact scanholder. If
# none exists, it waits for orphan GPU processes to clear, then restores either
# the requested holder or a clearly reported adaptive (degraded) holder.
#
# Usage:
#   resource_watchdog.sh start  SUB GPU RESERVE_MIB RUNNER_SESSION SCAN_SESSION SCAN_STATE_ROOT
#   resource_watchdog.sh status SUB
#   resource_watchdog.sh stop   SUB
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
SCAN_TOOL="$ROOT/tools/gram_ablation_scan.sh"
ACTION=${1:-status}
SUB=${2:-}

[[ -n "$SUB" ]] || { echo "missing SUB" >&2; exit 2; }

WATCH_SESSION="gram_phase13_watchdog_${SUB}"
WATCH_ROOT="$ROOT/.runtime/phase13_watchdog_${SUB}"
WATCH_STATUS="$WATCH_ROOT/status.json"
WATCH_LOG="$WATCH_ROOT/run.log"

write_status() {
  local state=$1 reason=$2 gpu=${3:--1} reserve=${4:-0} tmp="${WATCH_STATUS}.tmp.$$"
  mkdir -p "$WATCH_ROOT"
  printf '{"sub":"%s","state":"%s","reason":"%s","updated_at":"%s","watchdog_pid":%d,"physical_gpu":%d,"requested_reserve_mib":%d,"tmux_session":"%s"}\n' \
    "$SUB" "$state" "$reason" "$(date -Is)" "$$" "$gpu" "$reserve" "$WATCH_SESSION" > "$tmp"
  mv "$tmp" "$WATCH_STATUS"
}

holder_pid() {
  local state_root=$1
  "$PYTHON" -c "import json; d=json.load(open('$state_root/status.json')); print(d.get('pid', 0) if d.get('state') == 'running' else 0)" \
    2>/dev/null || echo 0
}

holder_reserve() {
  local state_root=$1
  "$PYTHON" -c "import json; d=json.load(open('$state_root/status.json')); print(d.get('reserve_mib', 0))" \
    2>/dev/null || echo 0
}

holder_on_gpu() {
  local gpu=$1 state_root=$2 pid
  pid=$(holder_pid "$state_root")
  [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 1
  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits --id="$gpu" 2>/dev/null | \
    grep -Eq "^[[:space:]]*${pid},.*gram-repro" || return 1
}

gpu_has_compute_processes() {
  local gpu=$1
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    --id="$gpu" 2>/dev/null | grep -Eq '[0-9]'
}

adaptive_reserve_mib() {
  local gpu=$1 free_mib
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
    --id="$gpu" 2>/dev/null | tr -d ' ' || true)
  [[ "$free_mib" =~ ^[0-9]+$ ]] || { echo 0; return; }
  local want=$(( free_mib - 2200 ))
  (( want >= 2000 )) || want=0
  echo "$want"
}

worker() {
  local gpu=${1:?missing GPU}
  local requested=${2:?missing reserve MiB}
  local runner_session=${3:?missing runner session}
  local scan_session=${4:?missing scan session}
  local scan_state_root=${5:?missing scan state root}
  [[ "$gpu" =~ ^[0-9]+$ && "$requested" =~ ^[1-9][0-9]*$ ]] || {
    write_status invalid_configuration "GPU and reserve must be positive integers" "$gpu" "${requested:-0}"
    return 2
  }
  trap 'write_status stopped "watchdog stopped by signal" "$gpu" "$requested"; exit 143' TERM INT HUP

  while tmux has-session -t "$runner_session" 2>/dev/null; do
    write_status monitoring "runner session is active" "$gpu" "$requested"
    sleep 10
  done

  write_status recovering "runner session exited; verifying resource holder" "$gpu" "$requested"
  while true; do
    if holder_on_gpu "$gpu" "$scan_state_root"; then
      local actual
      actual=$(holder_reserve "$scan_state_root")
      if [[ "$actual" == "$requested" ]]; then
        write_status protected_exact "exact scanholder is active" "$gpu" "$requested"
      else
        write_status protected_degraded "scanholder is active at ${actual} MiB" "$gpu" "$requested"
      fi
      return 0
    fi

    # A surviving workload/lease or another user's process may still own the
    # card. Do not OOM it and do not kill it; its presence already prevents the
    # memory from becoming completely idle. Retry until the card is clear.
    if gpu_has_compute_processes "$gpu"; then
      write_status waiting_for_orphan "GPU has compute processes but no verified holder; no process was killed" "$gpu" "$requested"
      sleep 10
      continue
    fi

    write_status restoring_exact "starting requested ${requested} MiB scanholder" "$gpu" "$requested"
    env RESERVE_MIB="$requested" SESSION="$scan_session" STATE_ROOT="$scan_state_root" \
      "$SCAN_TOOL" start "$gpu" >> "$WATCH_LOG" 2>&1 || true
    for _ in $(seq 1 20); do
      if holder_on_gpu "$gpu" "$scan_state_root" && [[ "$(holder_reserve "$scan_state_root")" == "$requested" ]]; then
        write_status protected_exact "watchdog restored exact scanholder" "$gpu" "$requested"
        return 0
      fi
      sleep 1
    done

    env SESSION="$scan_session" STATE_ROOT="$scan_state_root" "$SCAN_TOOL" stop >> "$WATCH_LOG" 2>&1 || true
    local adaptive
    adaptive=$(adaptive_reserve_mib "$gpu")
    if (( adaptive >= 2000 )); then
      write_status restoring_degraded "exact holder failed; starting adaptive ${adaptive} MiB scanholder" "$gpu" "$requested"
      env RESERVE_MIB="$adaptive" SESSION="$scan_session" STATE_ROOT="$scan_state_root" \
        "$SCAN_TOOL" start "$gpu" >> "$WATCH_LOG" 2>&1 || true
      for _ in $(seq 1 20); do
        if holder_on_gpu "$gpu" "$scan_state_root"; then
          write_status protected_degraded "watchdog restored adaptive ${adaptive} MiB scanholder" "$gpu" "$requested"
          return 0
        fi
        sleep 1
      done
    fi
    write_status retrying "holder restore failed; retrying without killing any process" "$gpu" "$requested"
    sleep 10
  done
}

case "$ACTION" in
  start)
    GPU=${3:?missing GPU}
    REQUESTED=${4:?missing reserve MiB}
    RUNNER_SESSION=${5:?missing runner session}
    SCAN_SESSION=${6:?missing scan session}
    SCAN_STATE_ROOT=${7:?missing scan state root}
    tmux has-session -t "$WATCH_SESSION" 2>/dev/null && {
      echo "watchdog already running: $WATCH_SESSION"
      exit 0
    }
    mkdir -p "$WATCH_ROOT"
    : > "$WATCH_LOG"
    write_status starting "launching independent resource watchdog" "$GPU" "$REQUESTED"
    printf -v launch_cmd 'bash %q worker %q %q %q %q %q %q >> %q 2>&1' \
      "$0" "$SUB" "$GPU" "$REQUESTED" "$RUNNER_SESSION" "$SCAN_SESSION" "$SCAN_STATE_ROOT" "$WATCH_LOG"
    tmux new-session -d -s "$WATCH_SESSION" "$launch_cmd"
    for _ in $(seq 1 10); do
      tmux has-session -t "$WATCH_SESSION" 2>/dev/null && {
        echo "started $WATCH_SESSION"
        exit 0
      }
      sleep 1
    done
    echo "watchdog session failed to start: $WATCH_SESSION" >&2
    exit 1
    ;;
  worker)
    worker "${3:?missing GPU}" "${4:?missing reserve MiB}" "${5:?missing runner session}" \
      "${6:?missing scan session}" "${7:?missing scan state root}"
    ;;
  status)
    tmux has-session -t "$WATCH_SESSION" 2>/dev/null \
      && echo "--- watchdog: running ($WATCH_SESSION) ---" \
      || echo "--- watchdog: not running ($WATCH_SESSION) ---"
    [[ -f "$WATCH_STATUS" ]] && sed -n '1,30p' "$WATCH_STATUS" || echo '{"state":"not_started"}'
    ;;
  stop)
    if tmux has-session -t "$WATCH_SESSION" 2>/dev/null; then
      tmux send-keys -t "$WATCH_SESSION" C-c
      echo "stop signal sent to $WATCH_SESSION"
    else
      echo "watchdog not running: $WATCH_SESSION"
    fi
    ;;
  help|--help|-h)
    sed -n '1,14p' "$0"
    ;;
  *)
    echo "usage: $0 {start|status|stop} SUB ..." >&2
    exit 2
    ;;
esac
