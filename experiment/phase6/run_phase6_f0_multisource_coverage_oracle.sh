#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"; OUT="$ROOT/artifacts/phase6/f0_multisource_coverage_oracle"; CFG="$ROOT/artifacts/phase6/configs/f0_multisource_coverage_oracle_preregistered.json"; PY=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python3.9; SESSION=gram_phase6_f0t; PID=0; LEASE=0; export HF_HOME="$ROOT/.cache/huggingface" TRANSFORMERS_CACHE="$ROOT/.cache/huggingface"
status(){ mkdir -p "$OUT"; printf '{"status":"%s","stage":"%s","updated_at":"%s","test_read":false,"sports_read":false}\n' "$1" "$2" "$(date -Is)" > "$OUT/status.json"; }
restore(){ env SESSION=codellama /home/jiangtangyunzhi/projects/UnitTest/tools/run_codellama.sh start 6 || true; }
finish(){ rc=$?; trap - EXIT; [[ $PID != 0 ]] && kill "$PID" 2>/dev/null || true; [[ $LEASE != 0 ]] && kill "$LEASE" 2>/dev/null || true; restore; status "$([[ $rc == 0 ]] && echo succeeded || echo failed)" finished; exit "$rc"; }
worker(){ trap finish EXIT; cd "$ROOT"; status running preflight; "$PY" - "$CFG" experiment/phase6/f0_multisource_coverage_oracle.py experiment/phase6/test_f0_multisource_coverage_oracle.py "$0" 'plan/第六阶段/GRAM_第六阶段_F0多源候选覆盖与Oracle审计实验计划.md' <<'PY'
import hashlib,json,pathlib,sys
c=json.load(open(sys.argv[1])); keys=('implementation_sha256','test_sha256','runner_sha256','plan_sha256');
for k,p in zip(keys,sys.argv[2:]):
 if hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()!=c['implementation_lock'][k]: raise SystemExit('frozen material mismatch: '+p)
PY
 "$PY" -m pytest -q experiment/phase6/test_f0_multisource_coverage_oracle.py; env SESSION=codellama /home/jiangtangyunzhi/projects/UnitTest/tools/run_codellama.sh stop; status waiting_for_gpu admission; free=""; for _ in $(seq 1 720); do free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id=6|tr -d ' '); [[ "$free" =~ ^[0-9]+$ ]] && ((free>=30720)) && break; sleep 60; done; ((free>=30720)); "$PY" experiment/gpu_memory_lease.py --gpu 6 --total-lease-mib 30720 --expected-workload-peak-mib 24576 --status-path "$OUT/gpu_lease.json" & LEASE=$!; sleep 5; status running f0t_gpu6; timeout --signal=TERM 172800 env CUDA_VISIBLE_DEVICES=6 "$PY" experiment/phase6/f0_multisource_coverage_oracle.py --config "$CFG" --output-root "$OUT" & PID=$!; wait "$PID"; }
case "${1:-status}" in start) mkdir -p "$OUT"; tmux has-session -t "$SESSION" 2>/dev/null && exit 1; tmux new-session -d -s "$SESSION" "bash $0 worker >> '$OUT/run.log' 2>&1"; status starting scheduled; echo "started $SESSION";; worker) worker;; status) tmux has-session -t "$SESSION" 2>/dev/null && echo "tmux: running" || echo "tmux: not running"; test -f "$OUT/status.json" && cat "$OUT/status.json"; test -f "$OUT/run.log" && tail -n 40 "$OUT/run.log";; esac
