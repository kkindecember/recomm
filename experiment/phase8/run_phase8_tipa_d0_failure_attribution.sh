#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/artifacts/phase8/tipa_d0_failure_attribution"
PY=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python3.9
SESSION=gram_phase8_tipa_d0_failure_attribution
SCRIPT="$ROOT/experiment/phase8/tipa_d0_failure_attribution.py"
TEST="$ROOT/experiment/phase8/test_tipa_d0_failure_attribution.py"
CFG="$ROOT/artifacts/phase8/configs/tipa_d0_failure_attribution_preregistered.json"
REPORT="$ROOT/report/第八阶段/GRAM_第八阶段_TIPA-D0路径对齐失败归因审计报告.md"

status() {
    mkdir -p "$OUT"
    printf '{"experiment_id":"GRAM_PHASE8_TIPA_D0_FAILURE_ATTRIBUTION_V1","status":"%s","stage":"%s","updated_at":"%s","runner_pid":%s,"cpu_only":true,"gpu_count":0,"optimizer_steps":0,"test_read":false,"sports_read":false,"tipa_p1_unlocked":false}\n' \
        "$1" "$2" "$(date -Is)" "$$" > "$OUT/status.json"
}

verify_outputs() {
    "$PY" - "$OUT" "$REPORT" <<'PY'
import csv,json,pathlib,sys
out=pathlib.Path(sys.argv[1]); report=pathlib.Path(sys.argv[2])
required=[out/n for n in ('summary.json','paired_effects.csv','strata.csv','prefix_census.csv','bootstrap_intervals.csv','integrity.json','manifest.json')] + [report]
missing=[str(p) for p in required if not p.is_file() or p.stat().st_size==0]
if missing: raise SystemExit(f'missing/empty outputs: {missing}')
summary=json.load(open(out/'summary.json')); integrity=json.load(open(out/'integrity.json'))
if summary['status']!='ANALYZED' or summary['tipa_p1_unlocked'] is not False: raise SystemExit('summary seal failed')
if any((summary['sports_read'],summary['test_read'],summary['external_development_read'])): raise SystemExit('forbidden read flag')
if summary['optimizer_steps']!=0 or summary['gpu_count']!=0: raise SystemExit('compute boundary failed')
if integrity['status']!='PASS': raise SystemExit('integrity failed')
expected={'paired_effects.csv':512,'strata.csv':96,'prefix_census.csv':20,'bootstrap_intervals.csv':6}
for name,n in expected.items():
    with open(out/name,newline='') as f: rows=list(csv.DictReader(f))
    if len(rows)!=n: raise SystemExit(f'{name}: {len(rows)} != {n}')
PY
}

verify_lock() {
    "$PY" - "$ROOT" "$CFG" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); cfg=json.load(open(sys.argv[2]))
if cfg.get('execution_enabled') is not True or cfg.get('decision_status')!='PREREGISTERED_FROZEN_READY_TO_RUN':
    raise SystemExit('D0 config is not frozen/enabled')
for name,spec in cfg['implementation_lock'].items():
    actual=hashlib.sha256((root/spec['path']).read_bytes()).hexdigest()
    if actual!=spec['sha256']: raise SystemExit(f'implementation drift: {name}: {actual}')
for relative,expected in cfg['parent_input_hashes'].items():
    actual=hashlib.sha256((root/relative).read_bytes()).hexdigest()
    if actual!=expected: raise SystemExit(f'BLOCKED_PARENT_ARTIFACT_DRIFT: {relative}: {actual}')
PY
}

worker() {
    trap 'rc=$?; if [[ $rc == 0 ]]; then status succeeded finished; else status failed execution_invalid; fi; exit $rc' EXIT
    cd "$ROOT"
    status running preflight
    verify_lock
    "$PY" -m pytest -q "$TEST"
    status running analysis
    timeout --signal=TERM 600 "$PY" "$SCRIPT" --output "$OUT" --report "$REPORT"
    status running verification
    verify_outputs
}

case "${1:-status}" in
    start)
        mkdir -p "$OUT"
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            echo "TIPA-D0 already running in $SESSION" >&2
            exit 1
        fi
        : > "$OUT/run.log"
        tmux new-session -d -s "$SESSION" "bash '$0' worker >> '$OUT/run.log' 2>&1"
        status starting scheduled
        echo "TIPA-D0 started in tmux session $SESSION"
        ;;
    worker) worker ;;
    status)
        tmux has-session -t "$SESSION" 2>/dev/null && echo "tmux: running" || echo "tmux: not running"
        [[ -f "$OUT/status.json" ]] && cat "$OUT/status.json" || true
        [[ -f "$OUT/run.log" ]] && tail -n 50 "$OUT/run.log" || true
        ;;
    *) echo "usage: $0 {start|status|worker}" >&2; exit 2 ;;
esac
