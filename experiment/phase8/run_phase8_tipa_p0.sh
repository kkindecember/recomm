#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"; OUT="$ROOT/artifacts/phase8/tipa_p0"; CFG="$ROOT/artifacts/phase8/configs/tipa_p0_preregistered.json"; PLAN="$ROOT/plan/第八阶段/GRAM_第八阶段_TIPA-P0商品到词法路径对齐审计实验计划.md"; PY=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python3.9; SESSION=gram_phase8_tipa_p0; GPU=6; PID=0; LEASE=0; TELEMETRY=0; STAGE=not_started
export HF_HOME="$ROOT/.cache/huggingface" TRANSFORMERS_CACHE="$ROOT/.cache/huggingface"
status(){ mkdir -p "$OUT"; printf '{"experiment_id":"GRAM_PHASE8_TIPA_P0A_V1","status":"%s","stage":"%s","updated_at":"%s","physical_gpu":6,"test_read":false,"sports_read":false}\n' "$1" "$2" "$(date -Is)" > "$OUT/status.json"; }
codellama(){ env SESSION=codellama HF_HOME=/home/jiangtangyunzhi/hf_cache HF_HUB_CACHE=/home/jiangtangyunzhi/hf_cache/hub TRANSFORMERS_CACHE=/home/jiangtangyunzhi/hf_cache/hub /home/jiangtangyunzhi/projects/UnitTest/tools/run_codellama.sh "$@"; }
restore(){ codellama start 6 || true; }
finish(){ rc=$?; trap - EXIT; [[ $PID != 0 ]] && kill "$PID" 2>/dev/null || true; [[ $LEASE != 0 ]] && kill "$LEASE" 2>/dev/null || true; [[ $TELEMETRY != 0 ]] && kill "$TELEMETRY" 2>/dev/null || true; restore; status "$([[ $rc == 0 ]] && echo succeeded || echo failed)" finished; exit "$rc"; }
telemetry(){ printf 'timestamp,index,memory_used_mib,memory_free_mib,utilization_gpu_percent\n' > "$OUT/gpu_telemetry.csv"; while true; do nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits --id=6 >> "$OUT/gpu_telemetry.csv" 2>/dev/null || true; sleep 5; done; }
verify(){ "$PY" - "$ROOT" "$CFG" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); c=json.load(open(sys.argv[2]))
if c.get('execution_enabled') is not True or c.get('decision_status')!='PREREGISTERED_FROZEN_READY_TO_RUN': raise SystemExit('config not frozen/enabled')
for key,spec in c['implementation_lock'].items():
 p=root/spec['path']; actual=hashlib.sha256(p.read_bytes()).hexdigest()
 if actual!=spec['sha256']: raise SystemExit(f'lock mismatch {key}: {p}')
for d,spec in c['teacher']['checkpoints'].items():
 p=root/spec['path'];
 if hashlib.sha256(p.read_bytes()).hexdigest()!=spec['sha256']: raise SystemExit(f'teacher mismatch: {d}')
for d,spec in c['lineage_lock'].items():
 temporal=c['teacher']['temporal_lineage']['datasets'][d]
 for field,key in (('user_sequence','user_sequence_sha256'),('item_index','item_index_sha256')):
  p=root/temporal[field]
  if hashlib.sha256(p.read_bytes()).hexdigest()!=spec[key]: raise SystemExit(f'lineage mismatch: {d}/{field}')
 parent=root/c['inputs']['checkpoint_root']/d/'C1/model.pt'
 if hashlib.sha256(parent.read_bytes()).hexdigest()!=c['inputs']['parent_checkpoint_sha256'][d]: raise SystemExit(f'parent mismatch: {d}')
PY
}
worker(){ trap finish EXIT; cd "$ROOT"; mkdir -p "$OUT"; STAGE=preflight; status running "$STAGE"; verify; "$PY" -m pytest -q experiment/phase8/test_tipa_p0.py; codellama stop; STAGE=waiting_for_gpu; status running "$STAGE"; free=0; for _ in $(seq 1 120); do free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id=6 2>/dev/null|tr -d ' '||true); [[ "$free" =~ ^[0-9]+$ ]] && ((free>=40000)) && break; sleep 5; done; [[ "$free" =~ ^[0-9]+$ ]] && ((free>=40000)); "$PY" experiment/gpu_memory_lease.py --gpu 6 --total-lease-mib 30720 --expected-workload-peak-mib 24576 --status-path "$OUT/gpu_lease.json" & LEASE=$!; sleep 5; telemetry & TELEMETRY=$!; for dataset in Toys Beauty; do STAGE="tipa_${dataset}"; status running "$STAGE"; timeout --signal=TERM 86400 env CUDA_VISIBLE_DEVICES=6 "$PY" experiment/phase8/tipa_p0.py --config "$CFG" --output-root "$OUT" --dataset "$dataset" & PID=$!; wait "$PID"; PID=0; done; "$PY" - "$OUT" <<'PY'
import hashlib,json,pathlib,sys
r=pathlib.Path(sys.argv[1]); ds={d:json.load(open(r/d/'summary.json')) for d in ('Toys','Beauty')}; passed=all(v['mechanism']['passed'] for v in ds.values())
out={'experiment_id':'GRAM_PHASE8_TIPA_P0A_V1','status':'PASS','decision':'TIPA_P1_DESIGN_ALLOWED' if passed else 'STOP_TIPA_NO_PATH_REALIZATION','datasets':ds,'telemetry_sha256':hashlib.sha256((r/'gpu_telemetry.csv').read_bytes()).hexdigest(),'test_read':False,'sports_read':False}
(r/'summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
PY
}
case "${1:-status}" in start) mkdir -p "$OUT"; tmux has-session -t "$SESSION" 2>/dev/null && exit 1; tmux new-session -d -s "$SESSION" "bash '$0' worker >> '$OUT/run.log' 2>&1"; status starting scheduled; echo "started $SESSION";; worker) worker;; status) tmux has-session -t "$SESSION" 2>/dev/null && echo 'tmux: running' || echo 'tmux: not running'; [[ -f "$OUT/status.json" ]] && cat "$OUT/status.json" || true; [[ -f "$OUT/run.log" ]] && tail -n 40 "$OUT/run.log" || true;; *) echo "usage: $0 {start|status|worker}" >&2; exit 2;; esac
