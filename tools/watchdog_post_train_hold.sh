#!/usr/bin/env bash
# Watchdog: after a Phase-13 experiment finishes (status.status == holding_post_training,
# succeeded, failed, or timed_out), stop the runner and start the protector on that GPU
# to keep the memory slot occupied.
#
# Runs in the background; poll interval = 30s. Idempotent — once it has handed off
# a given (sub,started_at) pair to the protector, it won't fire again.
#
# Usage:
#   nohup bash tools/watchdog_post_train_hold.sh v2_toys_iter2:0:ablation_scan:25000 v2_beauty_iter2:6:codellama:25000 &
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

INTERVAL=30
STATE_DIR="$ROOT/.runtime/watchdog_post_train_hold"
mkdir -p "$STATE_DIR"
LOG="$STATE_DIR/watchdog.log"

log() {
    echo "[$(date -Is)] $*" | tee -a "$LOG"
}

if (( $# == 0 )); then
    echo "usage: $0 <sub>:<gpu>:<protector>:<reserve_mib> ..." >&2
    echo "  protector ∈ {ablation_scan, codellama}" >&2
    exit 2
fi

declare -A HANDLED_STARTED  # keyed by sub — value: last started_at we handled

log "watchdog up, targets: $*, interval=${INTERVAL}s"

while true; do
    for target in "$@"; do
        IFS=':' read -r SUB GPU PROTECTOR RESERVE_MIB <<<"$target"
        STATUS="$ROOT/artifacts/phase13/explore/$SUB/status.json"
        [[ -f "$STATUS" ]] || continue

        # Parse status JSON with python for robustness
        read -r STATE STARTED <<<$(python3 -c "
import json, sys
try:
    d = json.load(open('$STATUS'))
    print(d.get('status', ''), d.get('started_at', ''))
except Exception as e:
    print('', '')
")
        [[ -n "$STATE" && -n "$STARTED" ]] || continue

        # Terminal states we care about
        case "$STATE" in
            holding_post_training|succeeded|failed|timed_out) ;;
            *) continue ;;
        esac

        # Handle once per (sub, started_at) pair
        if [[ "${HANDLED_STARTED[$SUB]:-}" == "$STARTED" ]]; then
            continue
        fi

        log "[$SUB] entered state=$STATE (started=$STARTED); protector=$PROTECTOR gpu=$GPU reserve=${RESERVE_MIB}MiB"

        # 1) Stop the runner via runner script (releases sidecar, invokes restore)
        bash "$ROOT/experiment/phase13/run_phase13_explore.sh" stop "$SUB" >>"$LOG" 2>&1 || true
        sleep 3

        # 2) Ensure protector is on the target GPU with the requested reserve size.
        #    stop then start (idempotent — kill any current holder then relaunch at correct size)
        case "$PROTECTOR" in
            ablation_scan)
                bash "$ROOT/tools/gram_ablation_scan.sh" stop >>"$LOG" 2>&1 || true
                sleep 3
                RESERVE_MIB="$RESERVE_MIB" bash "$ROOT/tools/gram_ablation_scan.sh" start "$GPU" >>"$LOG" 2>&1 || \
                    log "[$SUB] WARNING: ablation_scan restart failed"
                ;;
            codellama)
                bash "$ROOT/tools/run_codellama.sh" stop >>"$LOG" 2>&1 || true
                sleep 3
                HOLDER_RESERVE_MIB_OVERRIDE="$RESERVE_MIB" bash "$ROOT/tools/run_codellama.sh" start "$GPU" >>"$LOG" 2>&1 || \
                    log "[$SUB] WARNING: codellama restart failed"
                ;;
            *)
                log "[$SUB] ERROR: unknown protector $PROTECTOR"
                ;;
        esac

        HANDLED_STARTED[$SUB]="$STARTED"
        log "[$SUB] handoff complete"
    done
    sleep "$INTERVAL"
done
