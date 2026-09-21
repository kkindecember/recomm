"""Runtime-only recovery: reuse identical cross-attention caches across beams."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback

import numpy as np

from experiment.phase20 import lime_decode_screen as base
from experiment.phase18.core.diff_generation_cache import install_cross_cache_reuse

OLD=base.OUT
base.OUT=OLD/'cache_reuse_v2'
OUT=base.OUT
MODULE='experiment.phase20.lime_decode_screen_fast'


def prepare():
    assert all(base.read(OLD/d/'status.json')['state']=='FAILED' for d in base.DOMAINS)
    assert not any((OLD/d/'tuning.jsonl').exists() for d in base.DOMAINS)
    OUT.mkdir(exist_ok=False)
    config=base.read(OLD/'config.json'); config['cross_cache_reorder_reuse']=True
    base.write(OUT/'config.json',config)
    for d in base.DOMAINS:
        (OUT/d).mkdir()
        for name in ('signals.npz','samples.json','data_config.json','preflight.json'):
            # Read-only identical data, no recomputation, no cohort/outcome selection.
            os.link(OLD/d/name,OUT/d/name)
    manifest=base.read(OLD/'manifest.json')
    manifest['runtime_recovery']='Avoid copying identical cross-attention KV caches during beam ancestry reorder; validate exact paths/scores before results.'
    for p in [base.ROOT/'experiment/phase20/lime_decode_screen_fast.py',
              base.ROOT/'experiment/phase18/core/diff_generation_cache.py',OUT/'config.json']:
        manifest['sha256'][str(p.relative_to(base.ROOT))]=base.sha(p)
    base.write(OUT/'manifest.json',manifest)
    base.write(OUT/'status.json',dict(state='PREPARED',updated_at=base.stamp(),test_read=False))
    (OUT/'protocol.md').write_text((OLD/'protocol.md').read_text()+'\n\n## 运行时恢复 v2\n\n原始短测全部一致性检查通过，但成本超预算，未查看tuning/evaluation效果。保持全部用户、模型、公式、alpha和停止条件；仅复用同一用户各beam完全相同的cross-attention缓存，不重复按beam祖先拷贝。每域3例逐项比对原路径与分数一致后才运行；成本按优化后的decode计，原未优化对照不计入优化吞吐。所有旧输出保留。\n')
    print(json.dumps(dict(prepared=str(OUT))),flush=True)


class FastWorker(base.Worker):
    def __init__(self,domain):
        super().__init__(domain)
        install_cross_cache_reuse(self.model,50)

    def smoke(self):
        import torch
        self.stage='cache_reuse_smoke';self.status();checks=[];fast=[]
        for j in range(3):
            ctx=self.context(j)
            self.model._diff_cross_cache_reuse=False
            native=self.decode(ctx,native_entry=True,record=False)
            self.model._diff_cross_cache_reuse=True
            zero=self.decode(ctx);fast.append(self.seconds[-1])
            assert zero['paths']==native['paths']
            np.testing.assert_array_equal(zero['scores'],native['scores'])
            q=self.index.conditional_logs(self.signals['lime'][j])
            guided=self.decode(ctx,q,.3);fast.append(self.seconds[-1])
            zero2=self.decode(ctx,q,0.);fast.append(self.seconds[-1])
            assert zero2['paths']==zero['paths']
            np.testing.assert_array_equal(zero2['scores'],zero['scores'])
            checks.append(dict(user_id=self.samples[j]['user_id'],cross_cache_exact=True,
                encoder_cache_exact=True,zero_strength_exact=True,path_scores_reconstructed=True))
            self.status(completed_smoke_users=j+1,optimized_decode_seconds=fast)
            del ctx,native,zero,zero2,guided
        longest=max(range(len(self.data)),key=lambda i:len(self.data.records[i]['history']))
        ctx=self.context(0,longest);self.decode(ctx);fast.append(self.seconds[-1]);del ctx
        assert self.digest()==self.initial_digest
        estimate=max(fast)*2176
        base.write(self.folder/'smoke.json',dict(checks=checks,max_history_checked=True,
            projected_formal_seconds=estimate,optimized_decode_seconds=fast,all_decode_seconds=self.seconds,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,initial_digest=self.initial_digest))
        if estimate>self.config['worker_timeout_seconds']-300:raise TimeoutError(f'Optimized profile exceeds budget: {estimate}')
        self.status(projected_formal_seconds=estimate)


def supervise():
    config=base.read(OUT/'config.json');active={};done={};pending=set(base.DOMAINS);started=time.monotonic()
    while pending or active:
        for d in list(pending):
            if base.gpu_free(base.GPU[d])>=config['minimum_free_mib']:
                with (OUT/d/'worker.log').open('x') as log:
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=base.GPU[d],PYTHONUNBUFFERED='1',
                        OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
                    p=subprocess.Popen([sys.executable,'-m',MODULE,'worker','--domain',d],cwd=base.ROOT,
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                active[d]=(p,time.monotonic());pending.remove(d)
            elif time.monotonic()-started>config['resource_wait_seconds']:
                done[d]='RESOURCE_TIMEOUT';pending.remove(d)
        for d,(p,begin) in list(active.items()):
            rc=p.poll()
            if rc is not None:done[d]=rc;del active[d]
            elif time.monotonic()-begin>config['worker_timeout_seconds']+180:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
                done[d]='TIMEOUT';del active[d]
        base.write(OUT/'status.json',dict(state='RUNNING' if active or pending else
            'COMPLETED' if all(v==0 for v in done.values()) else 'INCOMPLETE',updated_at=base.stamp(),
            supervisor_pid=os.getpid(),active={d:p.pid for d,(p,_) in active.items()},pending=sorted(pending),completed=done,
            elapsed_seconds=time.monotonic()-started,test_read=False))
        if pending or active:time.sleep(5)
    if all(v==0 for v in done.values()):base.verify()


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','launch','supervise','worker','verify']);p.add_argument('--domain',choices=base.DOMAINS)
    args=p.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='launch':
        assert base.read(OUT/'status.json')['state']=='PREPARED' and not (OUT/'launch.json').exists()
        with (OUT/'supervisor.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-m',MODULE,'supervise'],cwd=base.ROOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        base.write(OUT/'launch.json',dict(pid=child.pid,started_at=base.stamp()))
        print(json.dumps(dict(pid=child.pid,status=str(OUT/'status.json'))),flush=True)
    elif args.action=='supervise':supervise()
    elif args.action=='verify':base.verify()
    else:
        try:FastWorker(args.domain).run()
        except Exception:
            detail=traceback.format_exc();print(detail,flush=True)
            base.write(OUT/args.domain/'failure.json',dict(at=base.stamp(),error=detail))
            previous=base.read(OUT/args.domain/'status.json') if (OUT/args.domain/'status.json').exists() else {}
            base.write(OUT/args.domain/'status.json',dict(previous,state='FAILED',updated_at=base.stamp(),error=detail))
            raise


if __name__=='__main__':main()
