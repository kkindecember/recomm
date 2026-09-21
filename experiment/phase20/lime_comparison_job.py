"""Durable two-domain LIME supervisor, including public asset download."""
import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
from experiment.phase20.guided_decoding_job import ROOT, read, write, sha, now, gpu_candidates

CONFIG = ROOT/'experiment/phase20/lime_comparison_20260920_v1.json'


def stop_child(child):
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try: child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def finish(out, states):
    summaries = {d: read(out/d/'summary.json') for d, s in states.items() if s['state'] == 'COMPLETED'}
    write(out/'summary.json', dict(domains=summaries, states=states, test_read=False))
    labels = [('LIME', 'LIME完整融合'), ('LIME_repeat_allowed', 'LIME不屏蔽历史'),
              ('common_G50', '生成50＋共同评分'), ('common_LIME50', 'LIME50＋共同评分'),
              ('common_SASRec50', 'SASRec50＋共同评分'), ('common_BGE50', 'BGE50＋共同评分'),
              ('LIME_scores_G50', 'LIME评分限制生成50')]
    lines = ['# LIME完整方法与候选替换对照', '',
        '本地开发集适配。三个独立SASRec种子；主融合器5折交叉拟合。没有读取test。', '',
        '| 域 | 方法 | NDCG@10（三seed均值） | Hit@10 |', '|---|---|---:|---:|']
    for d, s in summaries.items():
        results = [s['results'][f'{seed}_crossfit'] for seed in (0, 1, 2)]
        old = results[0]['A_head_seeds'].values()
        lines.append(f'| {d} | A当前完整方案（三小头均值） | {sum(x["N10"] for x in old)/3:.8f} | {sum(x["H10"] for x in old)/3:.8f} |')
        for key, label in labels:
            values = [r['metrics'][key]['all'] for r in results]
            lines.append(f'| {d} | {label} | {sum(x["N10"] for x in values)/3:.8f} | {sum(x["H10"] for x in values)/3:.8f} |')
    lines += ['', 'A的小头随机性与LIME整套SASRec训练随机性不同。双方监督来源、历史长度差异详见protocol.md。',
        '完整系统准确率与候选来源替换是两个比较；不以缓存吞吐推断端到端时延。',
        '逐seed、低频、校准2048敏感性、配对区间、训练成本见各域summary.json。失败域不删除。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def supervise(c, out):
    start = time.monotonic()
    jobs = {}
    states = {d: dict(state='QUEUED', stage='等待GPU显存') for d in c['domains']}
    download = None
    logs = []
    def status(state, **extra):
        asset = read(out/'assets/status.json') if (out/'assets/status.json').exists() else {}
        write(out/'status.json', dict(state=state, updated_at=now(), supervisor_pid=os.getpid(),
            elapsed_seconds=round(time.monotonic()-start, 1), domains=states, assets=asset,
            active_workers=len(jobs), test_read=False, **extra))
    def interrupted(signum, frame): raise InterruptedError(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        status('PREPARING', stage='冻结配置、代码及作者版本')
        assert subprocess.check_output(['git', '-C', str(ROOT/'third_party/LIME-Rec'), 'rev-parse', 'HEAD'], text=True).strip() == c['author_commit']
        assert not subprocess.check_output(['git', '-C', str(ROOT/'third_party/LIME-Rec'), 'status', '--porcelain'], text=True).strip()
        shutil.copy2(CONFIG, out/'config.json')
        shutil.copy2(ROOT/c['contract'], out/'protocol.md')
        manifest = {}
        sources = list(ROOT.glob('experiment/phase20/*lime_comparison*'))
        sources += list((ROOT/'third_party/LIME-Rec/lime_rec').rglob('*.py'))
        sources += [ROOT/'third_party/LIME-Rec/scripts/training/train_sasrec.py',
            ROOT/'third_party/LIME-Rec/scripts/evaluation/run_repeat_aware_gate.py',
            ROOT/'third_party/LIME-Rec/scripts/data/download_gram_data.py',
            ROOT/'experiment/phase20/catalog_scope_core.py', ROOT/'experiment/phase20/guided_decoding_job.py']
        for path in sources:
            if not path.is_file(): continue
            key = str(path.relative_to(ROOT)); manifest[key] = sha(path)
            copy = out/'sources'/key; copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, copy)
        write(out/'source_manifest.json', manifest)
        asset_log = (out/'assets/host_supervisor.log').open('x'); logs.append(asset_log)
        download = subprocess.Popen([sys.executable, '-B', '-m', 'experiment.phase20.lime_comparison_assets'],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=asset_log, stderr=subprocess.STDOUT, start_new_session=True)
        write(out/'assets/host_launch.json', dict(pid=download.pid, created_at=now()))
        while True:
            for d, job in list(jobs.items()):
                path = out/d/'status.json'
                detail = read(path) if path.exists() else {}
                elapsed = time.monotonic()-job['started']
                states[d].update(worker=detail, worker_elapsed_seconds=round(elapsed, 1), stage=detail.get('stage', '导入模型'))
                code = job['child'].poll()
                if code is None and elapsed > c['domain_timeout_seconds']:
                    stop_child(job['child']); code = job['child'].returncode
                    states[d]['error'] = '12小时硬超时'
                if code is None: continue
                states[d]['worker_alive'] = False
                states[d]['state'] = 'COMPLETED' if code == 0 and detail.get('state') == 'COMPLETED' else 'FAILED'
                if states[d]['state'] == 'FAILED': states[d]['error'] = detail.get('error', f'worker exit {code}')
                job['log'].close(); del jobs[d]
            if download.poll() is not None and download.returncode != 0:
                # Training is useful independently and continues; workers fail explicitly before semantic scoring.
                asset_file = out/'assets/status.json'
                if not asset_file.exists() or read(asset_file).get('state') != 'FAILED':
                    write(asset_file, dict(state='FAILED', updated_at=now(), error=f'download exit {download.returncode}'))
            pending = [d for d, s in states.items() if s['state'] == 'QUEUED']
            available = gpu_candidates() if pending else []
            used = {j['gpu']['uuid'] for j in jobs.values()}
            available = [g for g in available if g['free_mib'] >= c['minimum_free_mib'] and g['uuid'] not in used]
            for d in pending:
                if not available or len(jobs) >= c['max_parallel_workers']: break
                gpu = available.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['uuid'],
                    OMP_NUM_THREADS=str(c['cpu_threads']), OPENBLAS_NUM_THREADS=str(c['cpu_threads']),
                    MKL_NUM_THREADS=str(c['cpu_threads']), PYTHONUNBUFFERED='1', PYTHONPATH=str(ROOT),
                    TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                    CUBLAS_WORKSPACE_CONFIG=':4096:8')
                log = (out/f'{d}.worker.log').open('x'); logs.append(log)
                child = subprocess.Popen([sys.executable, '-B', '-m', 'experiment.phase20.lime_comparison_worker', '--domain', d],
                    cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                jobs[d] = dict(child=child, gpu=gpu, log=log, started=time.monotonic())
                states[d].update(state='RUNNING', stage='核对输入后开始训练', worker_pid=child.pid, worker_alive=True, gpu=gpu)
            for d, s in states.items():
                if s['state'] == 'QUEUED' and time.monotonic()-start > c['resource_wait_seconds']:
                    s.update(state='FAILED', error='GPU显存等待超过1小时')
            if not jobs and all(s['state'] != 'QUEUED' for s in states.values()): break
            status('RUNNING' if jobs else 'WAITING_GPU', stage='两域独立训练与候选对照')
            time.sleep(10)
        if download.poll() is None: stop_child(download)
        finish(out, states)
        for path, digest in manifest.items(): assert sha(ROOT/path) == digest, f'Source changed during run: {path}'
        status('COMPLETED' if all(s['state'] == 'COMPLETED' for s in states.values()) else 'FINISHED_WITH_FAILURES', stage='查看report.md与各域日志')
    except BaseException as e:
        for d, job in jobs.items():
            stop_child(job['child']); states[d].update(state='STOPPED', worker_alive=False)
        if download is not None: stop_child(download)
        (out/'failure.txt').write_text(traceback.format_exc())
        status('FAILED', stage='调度异常', error=repr(e))
        raise
    finally:
        for log in logs: log.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['launch', 'supervise', 'smoke'])
    args = p.parse_args()
    c = read(CONFIG); out = ROOT/c['output']
    if args.action == 'smoke':
        candidates = [g for g in gpu_candidates() if g['free_mib'] >= c['minimum_free_mib']]
        assert len(candidates), 'No 6GiB GPU for smoke'
        for d, gpu in zip(c['domains'], (candidates*2)[:2]):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['uuid'], OMP_NUM_THREADS='4',
                OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
            subprocess.run([sys.executable, '-B', '-m', 'experiment.phase20.lime_comparison_worker',
                '--domain', d, '--smoke'], cwd=ROOT, env=env, check=True, timeout=600)
    elif args.action == 'launch':
        assert not (out/'launch.json').exists(), 'Formal attempt already exists'
        for d in c['domains']: assert read(out/(d+'_smoke')/'status.json')['state'] == 'PRECHECK_PASSED'
        assert read(out/'tests.json')['state'] == 'PASSED'
        with (out/'supervisor.log').open('x') as log:
            child = subprocess.Popen([sys.executable, '-B', '-m', 'experiment.phase20.lime_comparison_job', 'supervise'],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(out/'launch.json', dict(pid=child.pid, created_at=now()))
        print(f'PID {child.pid}; status {out}/status.json', flush=True)
    else:
        supervise(c, out)


if __name__ == '__main__':
    main()
