"""Durable launcher and independent two-domain supervision."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from experiment.phase20.guided_decoding_job import ROOT, read, write, sha, now, gpu_candidates

CONFIG = ROOT/'experiment/phase20/catalog_scope_20260919_v1.json'


def finish_report(out, states):
    summaries = {d: read(out/d/'summary.json') for d, s in states.items() if s['state'] == 'COMPLETED'}
    write(out/'summary.json', dict(domains=summaries, domain_states=states,
        new_optimizer_steps=0, new_beam_decodes=0, test_read=False,
        caveat='Reused development validation, calibration-selected scalar; not an independent test.'))
    labels = {'A': '当前完整方案', 'B': '全目录普通融合', 'C': '同分数限制生成50候选',
              'directory50': '同分数限制目录头50候选', 'CF50': '同分数限制CF50候选'}
    lines = ['# 生成候选作用：双域结果', '',
        'B、C逐商品分数完全相同；三臂统一过滤输入历史最后20件。A沿用旧分数再过滤，原始未过滤A另存各域summary。', '',
        '| 域 | 方法 | 评价人数 | NDCG@10 | Hit@10 | Hit@50 |', '|---|---|---:|---:|---:|---:|']
    for d, summary in summaries.items():
        for arm, title in labels.items():
            m = summary['metrics'][arm]['all']
            lines.append(f'| {d} | {title} | {summary["evaluation_users"]} | {m["N10"]:.8f} | {m["H10"]:.8f} | {m["H50"]:.8f} |')
    lines += ['', '每域2048名校准用户选择普通融合权重，主权重按B校准N10选择，C共用。另保存C校准选权重和等权敏感性，不能替换主表。', '',
        '这是历史开发validation上的机制筛查，不是新的独立测试。B仍使用C1编码器和目录头，只跳过自回归解码，不是完全去掉生成模型训练。', '',
        '各域summary.json包含配对区间、候选覆盖损益、三小头参照与过滤影响；verification.json包含缓存复核。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def supervise(config, out):
    start = time.monotonic()
    active = {}
    states = {d: dict(state='QUEUED', stage='等待分配GPU') for d in config['domains']}
    def status(state, **extra):
        write(out/'status.json', dict(state=state, updated_at=now(), supervisor_pid=os.getpid(),
            elapsed_seconds=round(time.monotonic()-start, 1), domains=states,
            active_workers=len(active), new_optimizer_steps=0, new_beam_decodes=0,
            minimum_free_mib=config['minimum_free_mib'], test_read=False, automatic_followup=False, **extra))
    def terminate(job):
        process = job['process']
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    def handler(signum, frame):
        raise InterruptedError(f'Job interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    try:
        status('PREPARING', stage='保存协议与代码快照')
        shutil.copy2(CONFIG, out/'config.json')
        shutil.copy2(ROOT/config['contract'], out/'protocol.md')
        source_hashes = {}
        for path in sorted(ROOT.glob('experiment/phase20/*catalog_scope*')):
            if path.is_file():
                source_hashes[str(path.relative_to(ROOT))] = sha(path)
                target = out/'sources'/path.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        write(out/'source_manifest.json', source_hashes)
        for d in states: (out/d).mkdir()
        while True:
            for d, job in list(active.items()):
                path = out/d/'status.json'
                detail = read(path) if path.exists() else dict(stage='导入模型模块')
                elapsed = time.monotonic()-job['started']
                states[d].update(worker=detail, stage=detail.get('stage'), worker_elapsed_seconds=round(elapsed, 1))
                code = job['process'].poll()
                if code is None and elapsed > config['domain_timeout_seconds']:
                    terminate(job)
                    code = job['process'].returncode
                    states[d]['error'] = '本域超过预定2小时硬超时'
                if code is None: continue
                job['log'].close()
                del active[d]
                states[d]['worker_alive'] = False
                if code == 0 and detail.get('state') == 'COMPLETED':
                    states[d].update(state='COMPLETED', stage='本域完成，排名复核通过')
                else:
                    states[d].update(state='FAILED', stage='本域失败，另一域继续',
                        error=states[d].get('error', detail.get('error', f'exit {code}; see worker.log')))
            pending = [d for d, s in states.items() if s['state'] == 'QUEUED']
            gpus = gpu_candidates() if pending else []
            occupied = {job['gpu']['uuid'] for job in active.values()}
            available = [g for g in gpus if g['free_mib'] >= config['minimum_free_mib'] and g['uuid'] not in occupied]
            for d in pending:
                if len(active) >= config['max_parallel_workers'] or not available: break
                gpu = available.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['uuid'],
                    OMP_NUM_THREADS=str(config['cpu_threads']), MKL_NUM_THREADS=str(config['cpu_threads']),
                    OPENBLAS_NUM_THREADS=str(config['cpu_threads']), TOKENIZERS_PARALLELISM='false',
                    PYTHONUNBUFFERED='1', PYTHONPATH=str(ROOT), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                    CUBLAS_WORKSPACE_CONFIG=':4096:8')
                log = (out/d/'worker.log').open('x')
                try:
                    child = subprocess.Popen([sys.executable, '-B', '-m', 'experiment.phase20.catalog_scope_worker', '--domain', d],
                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL, start_new_session=True)
                except BaseException:
                    log.close()
                    raise
                active[d] = dict(process=child, gpu=gpu, started=time.monotonic(), log=log)
                states[d].update(state='RUNNING', stage='启动缓存核对与目录评分', worker_pid=child.pid, worker_alive=True, gpu=gpu)
            for d, s in states.items():
                if s['state'] == 'QUEUED':
                    s['waiting_seconds'] = round(time.monotonic()-start, 1)
                    if time.monotonic()-start > config['resource_wait_seconds']:
                        s.update(state='FAILED', stage='资源等待超时', error='No eligible GPU within one hour')
            if not active and all(s['state'] != 'QUEUED' for s in states.values()): break
            status('RUNNING' if active else 'WAITING_GPU', stage='双域独立后台运行', gpu_candidates=gpus)
            time.sleep(10)
        finish_report(out, states)
        success = all(s['state'] == 'COMPLETED' for s in states.values())
        status('COMPLETED' if success else 'FINISHED_WITH_FAILURES', stage='结果已汇总，查看report.md')
    except BaseException as error:
        for d, job in active.items():
            terminate(job)
            job['log'].close()
            states[d].update(state='STOPPED', stage='本次调度停止')
        (out/'failure.txt').write_text(traceback.format_exc())
        status('FAILED', stage='调度异常', error=repr(error))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('launch', 'supervise', 'stop'))
    args = parser.parse_args()
    config = read(CONFIG)
    out = ROOT/config['output']
    if args.action == 'launch':
        out.mkdir(parents=True, exist_ok=False)
        write(out/'status.json', dict(state='STARTING', updated_at=now(), stage='启动双域后台调度'))
        with (out/'supervisor.log').open('x') as log:
            child = subprocess.Popen([sys.executable, '-B', '-m', 'experiment.phase20.catalog_scope_job', 'supervise'],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, PYTHONUNBUFFERED='1'), start_new_session=True)
        write(out/'launch.json', dict(pid=child.pid, created_at=now(), status=str(out/'status.json')))
        print(json.dumps(dict(pid=child.pid, status=str(out/'status.json')), ensure_ascii=False))
    elif args.action == 'supervise':
        supervise(config, out)
    else:
        receipt = read(out/'launch.json')
        process = Path(f'/proc/{receipt["pid"]}')
        if process.exists():
            command = (process/'cmdline').read_bytes().split(b'\0')
            assert b'experiment.phase20.catalog_scope_job' in command and b'supervise' in command
            assert (process/'cwd').resolve() == ROOT
            os.kill(receipt['pid'], signal.SIGTERM)
        print(json.dumps(dict(stop_requested_pid=receipt['pid'])))


if __name__ == '__main__':
    main()
