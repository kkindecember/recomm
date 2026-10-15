"""Isolated, explicitly authorized B-only generation after the C screen stopped."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
from experiment.phase21 import text_adaptation as original
from experiment.phase21 import text_contribution as runner
from experiment.phase21.night_run import ContinueGeneration
from experiment.phase21.night_resume_core import durable_file, process_identity, identity_alive
from experiment.phase20.catalog_scope_core import metrics, contrast

ROOT = runner.ROOT
PRIOR = original.OUT
OUT = ROOT / 'artifacts/phase21/text_b_followup_20260923_v1'
PLAN = ROOT / 'plan/第二十一阶段/21.4D_B组真实生成与融合诊断_20260923.md'
REPORT = ROOT / 'report/第二十一阶段/B组真实生成与融合诊断_20260923.md'
MODULE = 'experiment.phase21.text_b_followup'
DOMAINS = runner.DOMAINS
read, sha, stamp = runner.read, runner.sha, runner.stamp


def write(path, data):
    runner.write(path, data); durable_file(path)


def status(path, state, **kw):
    write(path / 'status.json', dict(state=state, updated_at=stamp(), pid=os.getpid(), test_read=False, **kw))


def folder(job):
    return OUT / 'diagnostic' if job == 'diagnostic' else OUT / 'B' / job


def complete(job):
    p = folder(job)
    if read(p / 'status.json').get('state') != 'COMPLETED' or not (p / 'summary.json').exists(): return False
    s = read(p / 'summary.json')
    if job != 'diagnostic':
        assert s['users'] == 512 and s['model_unchanged'] and s['verified_by_sort_and_count']
        assert (p / 'ranks.npz').is_file()
    return True


def verify_manifest(progress=False):
    manifest = read(OUT / 'manifest.json')['sha256']
    for i, (rel, digest) in enumerate(manifest.items(), 1):
        assert sha(ROOT / rel) == digest, rel
        if progress and i % 10 == 0: status(OUT, 'VERIFYING_INPUTS', verified=i, total=len(manifest))


def prepare():
    OUT.mkdir(exist_ok=False); status(OUT, 'PREPARING')
    assert read(PRIOR / 'status.json')['state'] == 'COMPLETED_SCREEN_ONLY'
    cfg = dict(read(PRIOR / 'config.json'), max_parallel=2, gpu_pool=original.GPU_POOL,
               auto_start_after_reboot=False, selection='B_selected_after_development_screen',
               no_new_fitting=True)
    write(OUT / 'config.json', cfg); shutil.copy2(PLAN, OUT / 'protocol.md')
    (OUT / 'A').mkdir(); (OUT / 'B').mkdir(); (OUT / 'diagnostic').mkdir()
    write(OUT / 'B' / 'config.json', cfg)
    (OUT / 'shared').symlink_to(PRIOR / 'shared', target_is_directory=True)
    frozen = dict(read(PRIOR / 'manifest.json')['sha256'])
    for domain in DOMAINS:
        (OUT / 'A' / domain).symlink_to((PRIOR / 'A' / domain).resolve(), target_is_directory=True)
        p = OUT / 'B' / domain; p.mkdir()
        prior = PRIOR / 'B' / domain
        assert read(prior / 'status.json')['state'] == 'CACHE_READY'
        assert read(prior / 'fit_checks.json')['state'] == 'PASSED'
        for name in ('samples.json', 'catalog.json', 'data_config.json', 'semantic.npy', 'fused.npy',
                     'gate.pt', 'gate_prediction.npz', 'fit_checks.json', 'cache_manifest.json'):
            (p / name).symlink_to((prior / name).resolve())
            expected = read(prior / 'cache_manifest.json')['sha256'].get(name)
            frozen[str((prior / name).relative_to(ROOT))] = expected or sha(prior / name)
        for arm in ('A', 'B'):
            for name in ('gate_prediction.npz', 'fused.npy'):
                src = PRIOR / arm / domain / name
                frozen.setdefault(str(src.relative_to(ROOT)), sha(src))
            src = PRIOR / 'cache_main' / domain / f'{arm}_ranks.npz'
            frozen[str(src.relative_to(ROOT))] = sha(src)
        # Make the reused baseline generation verifiable, not merely linked.
        for name in ('summary.json', 'ranks.npz', 'predictions.jsonl'):
            src = (PRIOR / 'A' / domain).resolve() / name
            frozen[str(src.relative_to(ROOT))] = sha(src)
        status(p, 'QUEUED', domain=domain, arm='B')
    status(OUT / 'diagnostic', 'QUEUED')
    paths = [Path(__file__), ROOT / 'experiment/phase21/text_b_fusion_diagnostic.py',
             ROOT / 'experiment/phase21/test_text_b_followup.py', ROOT / 'experiment/phase21/run_text_b_followup.sh',
             ROOT / 'experiment/phase21/night_run.py', ROOT / 'experiment/phase21/night_resume_core.py',
             PLAN, OUT / 'config.json', OUT / 'B' / 'config.json']
    for p in paths: frozen[str(p.relative_to(ROOT))] = sha(p)
    for rel in frozen:
        src = ROOT / rel
        if src.suffix in ('.py', '.sh', '.md'):
            dest = OUT / 'sources' / rel; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dest)
    write(OUT / 'manifest.json', dict(sha256=frozen, created_at=stamp(), prior_screen_unchanged=True))
    status(OUT, 'PREPARED'); write_report()


class FollowupWorker(ContinueGeneration, original.GenerationWorker):
    # No fallback to fitting if a cache is missing. Its hash must match first.
    fit_signals = original.GenerationWorker.fit_signals


def worker(job):
    p = folder(job); stop = threading.Event()
    def heartbeat():
        while not stop.is_set():
            write(p / 'heartbeat.json', dict(pid=os.getpid(), updated_at=stamp(), liveness_only=True))
            stop.wait(30)
    thread = threading.Thread(target=heartbeat, daemon=True); thread.start()
    try:
        status(p, 'RUNNING', stage='verify_inputs')
        manifest = read(OUT / 'manifest.json')['sha256']
        for rel, digest in manifest.items():
            if rel.startswith(('experiment/', 'GRAM/src/', 'third_party/')): assert sha(ROOT / rel) == digest, rel
        if job == 'diagnostic':
            from experiment.phase21.text_b_fusion_diagnostic import run
            run(OUT, PRIOR, lambda **kw: status(p, 'RUNNING', **kw))
            status(p, 'COMPLETED'); return
        for name, digest in read(p / 'cache_manifest.json')['sha256'].items(): assert sha(p / name) == digest, name
        cfg = read(p / 'data_config.json')
        assert sha(ROOT / cfg['parent_checkpoint']) == cfg['parent_sha256']
        original.OUT = OUT; original.bind()
        FollowupWorker('B', job).run()
    except BaseException as error:
        (p / f'failure_{os.getpid()}.txt').write_text(traceback.format_exc())
        status(p, 'RESOURCE_TIME_LIMIT' if isinstance(error, TimeoutError) else 'FAILED', error=repr(error))
        raise
    finally:
        stop.set(); thread.join(timeout=2)


def summarize():
    result = {}; support = {}
    for domain in DOMAINS:
        rr = {a: runner.base.arrays(OUT / a / domain / 'ranks.npz') for a in ('A', 'B')}
        np.testing.assert_array_equal(rr['A']['users'], rr['B']['users'])
        np.testing.assert_array_equal(rr['A']['GRAM'], rr['B']['GRAM'])
        m = {a: {k: metrics(rr[a][k]) for k in ('GRAM', 'LIME', 'late', 'guided')} for a in ('A', 'B')}
        c = {k: contrast(rr['B'][k], rr['A'][k], repeats=2000, seed=20260923) for k in ('LIME', 'late', 'guided')}
        support[domain] = bool(c['guided']['delta_N10'] > 0 and c['guided']['delta_H10'] >= 0)
        coverage = {a: dict(outside_original50_to_guided10=int(((rr[a]['GRAM'] > 50) & (rr[a]['guided'] <= 10)).sum()),
                            original50_lost=int(((rr[a]['GRAM'] <= 50) & (rr[a]['guided'] > 50)).sum())) for a in ('A', 'B')}
        cache = {a: runner.base.arrays(PRIOR / 'cache_main' / domain / f'{a}_ranks.npz') for a in ('A', 'B')}
        for a in ('A', 'B'):
            np.testing.assert_array_equal(cache[a]['users'], rr[a]['users'])
            np.testing.assert_array_equal(cache[a]['fusion_catalog'], rr[a]['LIME'])
        masks = dict(B_text_gained=(cache['B']['text'] <= 10) & (cache['A']['text'] > 10),
                     B_text_lost=(cache['A']['text'] <= 10) & (cache['B']['text'] > 10))
        cohorts = {k: dict(users=int(mask.sum()), A_guided_hits=int(((rr['A']['guided'] <= 10) & mask).sum()),
                           B_guided_hits=int(((rr['B']['guided'] <= 10) & mask).sum())) for k, mask in masks.items()}
        result[domain] = dict(metrics=m, B_minus_A=c, coverage=coverage, text_change_cohorts=cohorts,
            B_guided_minus_B_late=contrast(rr['B']['guided'], rr['B']['late'], repeats=2000, seed=20260923))
    summary = dict(state='COMPLETED', updated_at=stamp(), domains=result, directional_support=support,
                   dual_domain_direction=all(support.values()), test_read=False,
                   selection='Post-screen exploratory followup; not independent confirmation', new_optimizer_steps=0)
    write(OUT / 'summary.json', summary)
    return summary


def write_report():
    lines = ['# B组真实生成与融合诊断', '', 'Origin Skill：academic-research-suite，内联执行。', '',
             '看过缓存结果后的后续探索；原C筛查未通过记录保留。A复用旧完整组合；B复用已完成评分与融合器，不重新训练。', '']
    dpath = OUT / 'diagnostic' / 'summary.json'
    if dpath.exists():
        d = read(dpath)
        lines += ['融合诊断已完成。AA=原分数/原系数，AB=新文本/原系数，BA=原分数/新系数，BB=新文本/新系数。',
                  '系数是已保存的每用户专家权重与历史惩罚；交叉格只作代数诊断，不是新方法或因果归因。', '',
                  '| 域 | AA全目录N10 | AB | BA | BB | B文本新增前十人数 | 其中BB命中 |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for domain, v in d['domains'].items():
            c = v['cohorts']['B_text_gained']; m = v['metrics']
            lines.append(f"| {domain} | {m['AA']['N10']:.6f} | {m['AB']['N10']:.6f} | {m['BA']['N10']:.6f} | {m['BB']['N10']:.6f} | {c['users']} | {c['top10']['BB']} |")
        lines += ['', '所有512人都进入总指标；上述目标定义子组仅用于事后描述，不能作线上选择策略。']
    else: lines += ['融合诊断尚未完成。']
    if (OUT / 'summary.json').exists():
        s = read(OUT / 'summary.json')
        lines += ['', '真实生成状态：COMPLETED；两域各512人及排名复核完成。', '',
                  '| 域 | 组 | 引导N10 | H10 | H50 | 同规则后重排N10 |', '|---|---|---:|---:|---:|---:|']
        for domain, v in s['domains'].items():
            for a in ('A', 'B'):
                m = v['metrics'][a]; g = m['guided']
                lines.append(f"| {domain} | {a} | {g['N10']:.6f} | {g['H10']:.6f} | {g['H50']:.6f} | {m['late']['N10']:.6f} |")
        lines += ['', 'B减A的生成差值与描述性区间：', '', '| 域 | ΔN10 | 描述性95%区间 | 前十新增/丢失 |', '|---|---:|---|---|']
        for domain, v in s['domains'].items():
            c = v['B_minus_A']['guided']; lo, hi = c['descriptive_95pct_N10']
            lines.append(f"| {domain} | {c['delta_N10']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {c['top10_gained']}/{c['top10_lost']} |")
        lines += ['', '预定两域方向条件：'+str(s['directional_support'])+'。它不是显著性或最终方法结论。']
    else:
        lines += ['', '真实生成尚未收齐；最新进度见status.json。尚无B真实生成收益结论。']
    lines += ['', '单上游种子、重复开发集、看过缓存结果后选择B。区间不包含训练随机性与研究选择；未读取test。',
              '不自动调参或追加实验。', '',
              '[实时状态](../../artifacts/phase21/text_b_followup_20260923_v1/status.json)；'
              '[诊断明细](../../artifacts/phase21/text_b_followup_20260923_v1/diagnostic/summary.json)；'
              '[执行协议](../../plan/第二十一阶段/21.4D_B组真实生成与融合诊断_20260923.md)。']
    REPORT.write_text('\n'.join(lines) + '\n')


def discover():
    active = {}
    for path in OUT.glob('launch_*.json'):
        for r in read(path)['launches']:
            if not identity_alive(r['identity']): continue
            cmd = Path(f"/proc/{r['identity']['pid']}/cmdline").read_bytes().decode().split('\0')
            assert MODULE in cmd and 'worker' in cmd and r['job'] in cmd
            assert r['job'] not in active, 'Duplicate worker'
            active[r['job']] = r
    return active


def supervise():
    lock = (OUT / 'controller.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify_manifest(progress=True)
    cfg = read(OUT / 'config.json'); active = discover(); children = {}; blocked = {}; started = time.monotonic()
    jobs = ['diagnostic', *DOMAINS]; attempt = f'{int(time.time())}_{os.getpid()}'; launches = []
    for j in jobs:
        if j not in active and read(folder(j) / 'status.json')['state'] in ('FAILED', 'RESOURCE_TIME_LIMIT'):
            blocked[j] = read(folder(j) / 'status.json')
    reported_diagnostic = False
    while True:
        now = time.time()
        if time.monotonic() - started > cfg['supervisor_timeout_seconds']:
            status(OUT, 'RESOURCE_TIME_LIMIT', active=active, reason='48h controller limit; workers left recoverable'); return
        for j, rec in list(active.items()):
            if j in children: children[j].poll()
            if identity_alive(rec['identity']) and now - rec['started_wall'] > cfg['worker_timeout_seconds']:
                status(folder(j), 'RESOURCE_TIME_LIMIT', reason='Actual 4h worker limit')
                os.killpg(rec['identity']['pid'], signal.SIGKILL if rec.get('terminating') else signal.SIGTERM)
                rec['terminating'] = True
            if not identity_alive(rec['identity']):
                del active[j]
                if not complete(j):
                    s = read(folder(j) / 'status.json')
                    if s['state'] not in ('FAILED', 'RESOURCE_TIME_LIMIT'):
                        status(folder(j), 'INTERRUPTED', previous=s); s = read(folder(j) / 'status.json')
                    blocked[j] = s
        pending = [j for j in jobs if j not in active and j not in blocked and not complete(j)]
        query_error = None; free = {}
        if any(j != 'diagnostic' for j in pending) and sum(j != 'diagnostic' for j in active) < cfg['max_parallel']:
            try:
                output = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True, timeout=30)
                free = {x.split(',')[0].strip(): int(x.split(',')[1]) for x in output.splitlines()}
            except (subprocess.SubprocessError, ValueError) as e: query_error = repr(e)
        busy = {r['gpu'] for r in active.values() if r['gpu']}
        available = sorted([g for g in cfg['gpu_pool'] if g not in busy and free.get(g, 0) >= cfg['minimum_free_mib']], key=lambda g: -free[g])
        for j in pending:
            gpu = None
            if j != 'diagnostic':
                if not available or sum(x != 'diagnostic' for x in active) >= cfg['max_parallel']: continue
                gpu = available.pop(0)
            env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=gpu or '', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
            status(folder(j), 'STARTING', gpu=gpu)
            with (folder(j) / f'worker_{attempt}.log').open('x') as log:
                proc = subprocess.Popen([sys.executable, '-u', '-m', MODULE, 'worker', '--job', j], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            children[j] = proc; identity = process_identity(proc.pid); assert identity
            rec = dict(job=j, gpu=gpu, identity=identity, started_wall=time.time())
            active[j] = rec; launches.append(rec)
            write(OUT / f'launch_{attempt}.json', dict(launches=launches, supervisor_pid=os.getpid()))
        if complete('diagnostic') and not reported_diagnostic:
            write_report(); reported_diagnostic = True
        pending = [j for j in jobs if j not in active and j not in blocked and not complete(j)]
        terminal = not active and not pending
        if all(complete(d) for d in DOMAINS) and not (OUT / 'summary.json').exists(): summarize(); write_report()
        state = ('INCOMPLETE' if blocked else 'COMPLETED') if terminal else ('RUNNING' if active else 'WAITING_FOR_RESOURCES')
        status(OUT, state, active={j: dict(**r, worker=read(folder(j) / 'status.json')) for j, r in active.items()},
               pending=pending, blocked=blocked, resource_query_error=query_error, elapsed_seconds=time.monotonic() - started,
               resume_command='bash experiment/phase21/run_text_b_followup.sh resume', auto_start_after_reboot=False)
        if terminal: write_report(); return
        time.sleep(15)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['start', 'resume', 'worker'])
    parser.add_argument('--job', choices=['diagnostic', *DOMAINS]); args = parser.parse_args()
    if args.action == 'worker':
        assert args.job; worker(args.job); return
    if args.action == 'start': prepare()
    try: supervise()
    except BlockingIOError: print('Controller already running.', flush=True)
    except BaseException as e:
        (OUT / f'controller_failure_{os.getpid()}.txt').write_text(traceback.format_exc())
        status(OUT, 'CONTROLLER_ERROR', error=repr(e), worker_policy='Leave workers alive; resume adopts them')
        raise


if __name__ == '__main__': main()
