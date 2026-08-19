#!/usr/bin/env bash
# Independent exact-size recovery watchdog for a deliberately resized scan holder.
#
# The runner creates TRANSITION_MARKER immediately before changing the holder.
# If the runner tmux disappears after that point, this watchdog restores the
# requested POST_RESERVE_MIB. It never signals non-holder GPU processes. When a
# new/unrecognised process is present, it leaves the interim holder in place and
# waits instead of risking an OOM.
#
# Usage:
#   partial_holder_restore_watchdog.sh start SUB GPU INITIAL DURING POST \
#     RUNNER_SESSION SCAN_SESSION SCAN_STATE_ROOT ALLOWED_PIDS_FILE \
#     TRANSITION_MARKER OUTPUT
#   partial_holder_restore_watchdog.sh status SUB OUTPUT
#   partial_holder_restore_watchdog.sh stop SUB OUTPUT
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python
SCAN_TOOL="$ROOT/tools/gram_ablation_scan.sh"
ACTION=${1:-status}
SUB=${2:-}

[[ -n "$SUB" ]] || { echo "missing SUB" >&2; exit 2; }

WATCH_SESSION="gram_phase13_partial_holder_watchdog_${SUB}"

holder_field() {
  local state_root=$1 field=$2
  "$PYTHON" -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get(sys.argv[2], 0))" \
    "$state_root/status.json" "$field" 2>/dev/null || echo 0
}

holder_pid() {
  local state_root=$1 state
  state=$(holder_field "$state_root" state)
  [[ "$state" == running ]] || { echo 0; return; }
  holder_field "$state_root" pid
}

holder_on_gpu() {
  local gpu=$1 state_root=$2 pid
  pid=$(holder_pid "$state_root")
  [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 1
  nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits \
    2>/dev/null | tr -d ' ' | grep -Fxq "$pid"
}

gpu_pids() {
  local gpu=$1
  nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits \
    2>/dev/null | tr -d ' ' | sed '/^[0-9][0-9]*$/!d' | sort -nu
}

has_unknown_gpu_pid() {
  local gpu=$1 state_root=$2 allowed_file=$3 holder=0 pid
  holder=$(holder_pid "$state_root")
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    [[ "$pid" == "$holder" ]] && continue
    grep -Fxq "$pid" "$allowed_file" 2>/dev/null || return 0
  done < <(gpu_pids "$gpu")
  return 1
}

free_mib() {
  local gpu=$1
  nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits \
    2>/dev/null | tr -d ' '
}

wait_for_holder() {
  local gpu=$1 state_root=$2 reserve=$3
  for _ in $(seq 1 20); do
    if holder_on_gpu "$gpu" "$state_root" \
      && [[ "$(holder_field "$state_root" reserve_mib)" == "$reserve" ]]; then
      return 0
    fi
    sleep 1
  done
  return 1
}

worker() {
  local gpu=${1:?missing GPU}
  local initial=${2:?missing initial reserve}
  local during=${3:?missing during reserve}
  local post=${4:?missing post reserve}
  local runner_session=${5:?missing runner session}
  local scan_session=${6:?missing scan session}
  local scan_state_root=${7:?missing scan state root}
  local allowed_file=${8:?missing allowed PID file}
  local marker=${9:?missing transition marker}
  local output=${10:?missing output directory}
  local watch_status="$output/resource_watchdog_status.json"
  local watch_log="$output/resource_watchdog.log"
  local post_start_free=$(( post + 2500 ))
  local during_start_free=$(( during + 2500 ))

  write_status() {
    local state=$1 reason=$2 active=${3:-0} tmp="${watch_status}.tmp.$$"
    mkdir -p "$output"
    printf '{"state":"%s","reason":"%s","updated_at":"%s","watchdog_pid":%d,"physical_gpu":%d,"initial_reserve_mib":%d,"during_reserve_mib":%d,"post_reserve_mib":%d,"active_reserve_mib":%d,"tmux_session":"%s","signals_to_non_holder_processes":0}\n' \
      "$state" "$reason" "$(date -Is)" "$$" "$gpu" "$initial" "$during" "$post" "$active" "$WATCH_SESSION" > "$tmp"
    mv "$tmp" "$watch_status"
  }

  trap 'write_status stopped "watchdog stopped by signal" "$(holder_field "$scan_state_root" reserve_mib)"; exit 143' TERM INT HUP

  while tmux has-session -t "$runner_session" 2>/dev/null; do
    write_status monitoring "runner session is active; no holder action"
    sleep 10
  done

  if [[ ! -e "$marker" ]]; then
    write_status not_armed "runner exited before the holder transition; original holder was left unchanged" \
      "$(holder_field "$scan_state_root" reserve_mib)"
    return 0
  fi

  while true; do
    local current=0 holder=0 free=0
    if holder_on_gpu "$gpu" "$scan_state_root"; then
      current=$(holder_field "$scan_state_root" reserve_mib)
      holder=$(holder_pid "$scan_state_root")
      if [[ "$current" == "$post" ]]; then
        write_status protected_exact "exact post-run holder is active" "$current"
        return 0
      fi
      if [[ "$current" != "$during" && "$current" != "$initial" ]]; then
        write_status waiting_unexpected_holder "holder has an unrecognised reserve; no action taken" "$current"
        sleep 10
        continue
      fi
    fi

    if has_unknown_gpu_pid "$gpu" "$scan_state_root" "$allowed_file"; then
      write_status waiting_for_unknown_process "new/unrecognised GPU process present; interim holder retained and no process signalled" "$current"
      sleep 10
      continue
    fi

    if (( holder > 0 )); then
      write_status resizing "stopping only the verified scan holder before exact post-run restore" "$current"
      env SESSION="$scan_session" STATE_ROOT="$scan_state_root" \
        "$SCAN_TOOL" stop >> "$watch_log" 2>&1 || true
      for _ in $(seq 1 15); do
        holder_on_gpu "$gpu" "$scan_state_root" || break
        sleep 1
      done
      if holder_on_gpu "$gpu" "$scan_state_root"; then
        write_status retrying "verified holder did not stop; retrying without touching other processes" "$current"
        sleep 10
        continue
      fi
    fi

    free=$(free_mib "$gpu" || true)
    if [[ "$free" =~ ^[0-9]+$ ]] && (( free >= post_start_free )); then
      write_status restoring_exact "starting exact ${post} MiB post-run holder"
      env RESERVE_MIB="$post" SESSION="$scan_session" STATE_ROOT="$scan_state_root" \
        "$SCAN_TOOL" start "$gpu" >> "$watch_log" 2>&1 || true
      if wait_for_holder "$gpu" "$scan_state_root" "$post"; then
        write_status protected_exact "watchdog restored exact post-run holder" "$post"
        return 0
      fi
      env SESSION="$scan_session" STATE_ROOT="$scan_state_root" \
        "$SCAN_TOOL" stop >> "$watch_log" 2>&1 || true
    fi

    free=$(free_mib "$gpu" || true)
    if [[ "$free" =~ ^[0-9]+$ ]] && (( free >= during_start_free )); then
      write_status restoring_interim "exact post-run holder cannot currently start; restoring ${during} MiB interim protection"
      env RESERVE_MIB="$during" SESSION="$scan_session" STATE_ROOT="$scan_state_root" \
        "$SCAN_TOOL" start "$gpu" >> "$watch_log" 2>&1 || true
      wait_for_holder "$gpu" "$scan_state_root" "$during" || true
    fi
    write_status retrying "post-run holder not yet exact; no non-holder process was signalled" \
      "$(holder_field "$scan_state_root" reserve_mib)"
    sleep 10
  done
}

case "$ACTION" in
  start)
    GPU=${3:?missing GPU}
    INITIAL=${4:?missing initial reserve}
    DURING=${5:?missing during reserve}
    POST=${6:?missing post reserve}
    RUNNER_SESSION=${7:?missing runner session}
    SCAN_SESSION=${8:?missing scan session}
    SCAN_STATE_ROOT=${9:?missing scan state root}
    ALLOWED_FILE=${10:?missing allowed PID file}
    MARKER=${11:?missing transition marker}
    OUTPUT=${12:?missing output directory}
    tmux has-session -t "$WATCH_SESSION" 2>/dev/null && {
      echo "watchdog already running: $WATCH_SESSION" >&2
      exit 1
    }
    mkdir -p "$OUTPUT"
    : > "$OUTPUT/resource_watchdog.log"
    printf -v launch_cmd 'bash %q worker %q %q %q %q %q %q %q %q %q %q %q >> %q 2>&1' \
      "$0" "$SUB" "$GPU" "$INITIAL" "$DURING" "$POST" "$RUNNER_SESSION" \
      "$SCAN_SESSION" "$SCAN_STATE_ROOT" "$ALLOWED_FILE" "$MARKER" "$OUTPUT" \
      "$OUTPUT/resource_watchdog.log"
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
    worker "${3:?}" "${4:?}" "${5:?}" "${6:?}" "${7:?}" "${8:?}" \
      "${9:?}" "${10:?}" "${11:?}" "${12:?}"
    ;;
  status)
    OUTPUT=${3:?missing output directory}
    tmux has-session -t "$WATCH_SESSION" 2>/dev/null \
      && echo "watchdog session: running ($WATCH_SESSION)" \
      || echo "watchdog session: not running ($WATCH_SESSION)"
    [[ -f "$OUTPUT/resource_watchdog_status.json" ]] \
      && sed -n '1,60p' "$OUTPUT/resource_watchdog_status.json" \
      || echo '{"state":"not_started"}'
    ;;
  stop)
    OUTPUT=${3:?missing output directory}
    if tmux has-session -t "$WATCH_SESSION" 2>/dev/null; then
      tmux send-keys -t "$WATCH_SESSION" C-c
      echo "stop signal sent to $WATCH_SESSION"
    else
      echo "watchdog not running: $WATCH_SESSION"
    fi
    ;;
  *)
    echo "usage: $0 {start|status|stop} SUB ..." >&2
    exit 2
    ;;
esac
