"""Independent per-domain GPU jobs; one failure does not cancel the other domain."""
import os
import signal
import subprocess
import sys
import time
import traceback

from experiment.phase20.guided_decoding_job import ROOT, read, write, now, prepare, verify, gpu_candidates


def write_report(out, states):
    summaries={d:read(out/d/'summary.json') for d,s in states.items() if s['state']=='COMPLETED'}
    write(out/'summary.json',dict(domains=summaries,domain_states=states,test_read=False,new_training_steps=0,
        caveat='Exploratory reused-validation screen; intervals are descriptive; no automatic followup.'))
    lines=['# 冻结协同引导生成：结果','','两个数据域独立执行；未完成域不计为零分。', '',
        '| 域 | 搜索 | 完整后处理 N10 | H10 | 相对原搜索 ΔN10 |','|---|---|---:|---:|---:|']
    for domain,summary in summaries.items():
        for arm in ('vanilla','real_cf','shuffled_cf','popularity'):
            m=summary['metrics'][arm+'.softmax']; delta=m['N10']-summary['metrics']['vanilla.softmax']['N10']
            lines.append(f'| {domain} | {arm} | {m["N10"]:.6f} | {m["H10"]:.6f} | {delta:+.6f} |')
    lines += ['', '配对区间、候选进出、late-only对照和辅助头见各域summary.json；verification.json记录缓存重放。',
        '这是已使用validation上的探索性筛查，不构成独立测试或算法创新性证明。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def supervise_parallel(config,out):
    started=time.monotonic(); active={}; completed=[]
    states={d:dict(state='QUEUED',mode='smoke',stage='等待GPU检查',queued_at_seconds=0.) for d in config['domains']}
    def status(state,**kwargs):
        write(out/'status.json',dict(state=state,updated_at=now(),supervisor_pid=os.getpid(),
            elapsed_seconds=round(time.monotonic()-started,1),completed=completed,domains=states,
            parallel_workers=len(active),max_parallel_workers=config['max_parallel_workers'],
            minimum_free_mib=config['minimum_free_mib'],test_read=False,optimizer_steps=0,automatic_followup=False,**kwargs))
    def interrupted(sig,frame): raise InterruptedError(f'Supervisor signal {sig}')
    old_handlers={s:signal.signal(s,interrupted) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        status('PREPARING',stage='冻结文件与用户清单')
        prepare(config,out)
        status('VERIFYING',stage='核对冻结输入，准备两域独立运行')
        verify(out)
        while True:
            for domain,job in list(active.items()):
                process=job['process']; mode=job['mode']; elapsed=time.monotonic()-job['started']
                path=out/domain/f'{mode}_status.json'
                detail=read(path) if path.exists() else dict(stage='加载模型模块')
                states[domain].update(worker=detail,stage=detail.get('stage'),worker_elapsed_seconds=round(elapsed,1))
                code=process.poll()
                if code is None and elapsed>job['budget']:
                    os.killpg(process.pid,signal.SIGTERM)
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGKILL); process.wait()
                    states[domain]['error']=f'{mode} exceeded hard timeout of {job["budget"]} seconds'
                    code=process.returncode
                if code is None:
                    states[domain]['state']='WAITING_GPU' if detail.get('state')=='WAITING_GPU' else 'RUNNING'
                    continue
                job['log'].close(); del active[domain]
                states[domain]['worker_alive']=False
                if code:
                    states[domain].update(state='FAILED',stage='本域失败；其他域继续',
                        error=states[domain].get('error',f'{domain}/{mode} exit {code}; see {domain}/{mode}.log'))
                    continue
                completed.append(f'{domain}/{mode}')
                if mode=='run':
                    states[domain].update(state='COMPLETED',stage='本域结果和重放检查完成')
                else:
                    smoke=read(out/domain/'smoke.json')
                    states[domain]['profile']=smoke
                    if smoke['projected_domain_seconds']>config['max_projected_domain_seconds']:
                        states[domain].update(state='STOPPED_RESOURCE_BUDGET',stage='实测单域超过11小时预算，保留速度数据')
                    else:
                        states[domain].update(state='QUEUED',mode='run',stage='GPU检查通过，等待正式任务空位',
                            queued_at_seconds=time.monotonic()-started)
            pending=[d for d,s in states.items() if s['state']=='QUEUED']
            available=gpu_candidates() if pending else []
            occupied={j['gpu']['uuid'] for j in active.values()}
            slots=[g for g in available if g['free_mib']>=config['minimum_free_mib'] and g['uuid'] not in occupied]
            for domain in pending:
                if len(active)>=config['max_parallel_workers'] or not slots: break
                gpu=slots.pop(0); mode=states[domain]['mode']
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],OMP_NUM_THREADS=str(config['cpu_threads']),
                    MKL_NUM_THREADS=str(config['cpu_threads']),TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',
                    PYTHONPATH=str(ROOT),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
                command=[sys.executable,'-m','experiment.phase20.guided_decoding_worker','--domain',domain,'--mode',mode]
                logpath=out/domain/f'{mode}.log'; log=logpath.open('x')
                try:
                    child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,start_new_session=True)
                except BaseException:
                    log.close(); raise
                budget=config['smoke_timeout_seconds' if mode=='smoke' else 'domain_timeout_seconds']
                active[domain]=dict(process=child,mode=mode,gpu=gpu,log=log,started=time.monotonic(),budget=budget)
                states[domain].update(state='RUNNING',stage='启动GPU检查' if mode=='smoke' else '启动正式筛查',
                    worker_pid=child.pid,worker_alive=True,gpu=gpu,log=str(logpath.relative_to(ROOT)),timeout_seconds=budget)
                states[domain].pop('worker',None)
            for domain,s in states.items():
                if s['state']=='QUEUED':
                    wait=time.monotonic()-started-s['queued_at_seconds']
                    s['waiting_seconds']=round(wait,1)
                    if wait>config['resource_wait_seconds']:
                        s.update(state='FAILED',error='GPU resource wait budget exhausted',stage='资源等待超时')
            pending=[d for d,s in states.items() if s['state']=='QUEUED']
            if not active and not pending: break
            running=any(s['state']=='RUNNING' for s in states.values())
            status('RUNNING' if running else 'WAITING_GPU',stage='两域独立调度',gpu_candidates=available)
            time.sleep(10)
        write_report(out,states)
        done=all(s['state']=='COMPLETED' for s in states.values())
        status('COMPLETED' if done else 'FINISHED_WITH_FAILURES',stage='两域任务结束，详情见各域状态',
            summary=str((out/'summary.json').relative_to(ROOT)),report=str((out/'report.md').relative_to(ROOT)))
    except BaseException as exc:
        for job in active.values():
            process=job['process']
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGKILL); process.wait()
            job['log'].close()
        (out/'failure.txt').write_text(traceback.format_exc())
        status('FAILED',stage='调度异常，保留现场',error=str(exc))
        raise
    finally:
        for sig,handler in old_handlers.items(): signal.signal(sig,handler)
