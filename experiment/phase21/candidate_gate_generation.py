"""Conditional true-generation transfer screen for the matched candidate head.

prepare -> worker Toys / worker Beauty (separate CUDA_VISIBLE_DEVICES) -> summarize.
Uses the established GRAM worker, prefix mass aggregation and AGRec-inspired
probability mixture unchanged. New training is prohibited here.
"""
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

import numpy as np

from experiment.phase20 import lime_clean_benefit as clean
from experiment.phase20 import lime_decode_screen as base
from experiment.phase21.candidate_gate_core import Gate,DU,DC,WIDTH

ROOT=base.ROOT
SCREEN=ROOT/'artifacts/phase21/candidate_gate_20261010_v1'
FULL=ROOT/'artifacts/phase21/full_validation_20260924_v1'
OUT=ROOT/'artifacts/phase21/candidate_gate_generation_20261010_v1'
PLAN=ROOT/'plan/第二十一阶段/21.19B_候选门控真实生成迁移筛查_20261010.md'
read,write,sha=base.read,base.write,base.sha


def prepare():
    import torch
    torch.set_num_threads(2)
    assert read(SCREEN/'status.json')['state']=='COMPLETE'
    assert read(SCREEN/'verification.json')['state']=='VERIFIED'
    summary=read(SCREEN/'summary.json')
    for d in ('Toys','Beauty'):
        r=summary[d]['results']['primary']['comparisons']['score_vs_user']
        assert r['ci95'][0]>0 and min(r['seed_deltas'])>0,('Screen admission failed',d,r)
    OUT.mkdir(exist_ok=False)
    cfg=dict(evaluation_users=2048,alpha=.3,eta=.1,head_seed=0,beam_size=50,length_penalty=1.,
             worker_timeout_seconds=7200,expected_workload_peak_mib=12288,total_lease_mib=30720,
             minimum_free_mib=32768,allocation_limit_gib=11,test_read=False,optimizer_steps=0,
             gpus=dict(Toys='GPU-07374397-b213-09c1-2988-718acb6ed231',Beauty='GPU-e2e37406-aaaa-da68-cf74-1a487b1c93e2'))
    write(OUT/'config.json',cfg);shutil.copy2(PLAN,OUT/'protocol.md')
    manifest={}
    def use(p):manifest[str(p.relative_to(ROOT))]=sha(p)
    for d in ('Toys','Beauty'):
        folder=OUT/d;folder.mkdir()
        samples=read(FULL/d/'samples.json')
        order=[i for i,s in enumerate(samples) if not s['previously_evaluated'] and s['training_holdout_role']=='none']
        import hashlib
        order.sort(key=lambda i:hashlib.sha256(f'candidate-gate-generation-20261010|{d}|{samples[i]["user_id"]}'.encode()).digest())
        ix=np.array(order[:cfg['evaluation_users']]);assert len(ix)==cfg['evaluation_users']
        chosen=[dict(samples[k],full_validation_index=int(k),validation_index=j) for j,k in enumerate(ix)]
        write(folder/'samples.json',chosen)
        write(folder/'data_config.json',read(FULL/d/'data_config.json'))
        shutil.copy2(FULL/d/'catalog.json',folder/'catalog.json')
        features=SCREEN/d/'validation'
        u=np.load(features/'user.npy',mmap_mode='r')[ix]
        c=np.load(features/'candidate.npy',mmap_mode='r')[ix]
        mask=np.load(features/'mask.npy',mmap_mode='r')[ix]
        ids=np.load(features/'ids.npy',mmap_mode='r')[ix]
        e=np.load(features/'scores.npy',mmap_mode='r')[ix]
        original=np.load(FULL/d/'fused.npy',mmap_mode='r')[ix]
        w=np.load(FULL/d/'gate_prediction.npz')['weights'][ix]
        signals={'original':original}
        for arm in ('user','score'):
            path=SCREEN/d/(arm+'_seed0.pt');checkpoint=torch.load(path,map_location='cpu');use(path)
            assert set(checkpoint['fit_users']).isdisjoint(s['user_id'] for s in chosen)
            model=Gate();model.load_state_dict(checkpoint['state_dict']);model.eval()
            m=1 if arm=='user' else 200
            x=np.zeros((len(ix),m,WIDTH),np.float32);x[:,:,:DU]=u[:,None,:]
            if arm=='score':x[:,:,DU:DU+DC]=c
            x=np.clip((x-checkpoint['mean'])/checkpoint['std'],-10,10)
            with torch.no_grad():
                dw=model(torch.from_numpy(x),torch.from_numpy(w[:,None,:]),cfg['eta']).numpy()
            delta=(dw*e).sum(2);f=original.copy()
            for j in range(len(ix)):f[j,ids[j,mask[j]]]+=delta[j,mask[j]]
            assert float(np.max(np.abs(f-original)))<=cfg['eta']+1e-6
            expected_ranks=np.load(SCREEN/d/'ranks.npz')[arm+'_0'][ix]
            for j,s in enumerate(chosen):
                masked=f[j].copy();masked[np.asarray(s['history'])-1]=-np.inf
                target=s['target']-1
                order=np.lexsort((np.arange(len(masked)),-masked))
                actual=int(np.flatnonzero(order==target)[0])+1
                assert actual==expected_ranks[j],('Prepared score parity',d,arm,j,actual,int(expected_ranks[j]))
            signals[arm]=f
        np.savez(folder/'signals.npz',**signals)
        users={s['user_id'] for s in chosen};references=[]
        with (FULL/d/'predictions.jsonl').open() as f:
            for line in f:
                r=json.loads(line)
                if r['user_id'] in users:references.append(r)
        assert len(references)==len(chosen);write(folder/'references.json',references)
        for name in ('samples.json','catalog.json','data_config.json','signals.npz','references.json'):use(folder/name)
        cp=ROOT/read(folder/'data_config.json')['parent_checkpoint'];use(cp)
        assert manifest[str(cp.relative_to(ROOT))]==read(folder/'data_config.json')['parent_sha256']
        write(folder/'status.json',dict(state='PREPARED',users_total=len(chosen)))
    for p in (PLAN,Path(__file__),ROOT/'experiment/phase21/candidate_gate_core.py',ROOT/'experiment/gpu_memory_lease.py',
              ROOT/'experiment/GPU_MEMORY_LEASE_POLICY.md',OUT/'config.json',SCREEN/'summary.json',SCREEN/'verification.json',
              ROOT/'experiment/phase20/lime_clean_benefit.py',ROOT/'experiment/phase20/lime_decode_screen.py',
              ROOT/'experiment/phase20/lime_decode_screen_core.py'):
        use(p)
        if p.suffix in ('.py','.md'):
            dest=OUT/'sources'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    write(OUT/'manifest.json',dict(sha256=manifest));write(OUT/'status.json',dict(state='PREPARED'))
    print(json.dumps(cfg),flush=True)


class Worker(clean.Worker):
    def __init__(self,domain,lease):
        self.lease=lease;clean.OUT=OUT
        super().__init__(domain)
        self.signals=base.arrays(self.folder/'signals.npz')
        self.reference={r['user_id']:r for r in read(self.folder/'references.json')}

    def context(self,j,override_index=None):
        return super().context(j,override_index=j)

    def status(self,state='RUNNING',**kw):
        assert self.lease.poll() is None,'GPU lease exited'
        super().status(state,**kw)

    def run(self):
        self.stage='native_and_prior_parity';checks=[]
        for j in (0,len(self.samples)//2,len(self.samples)-1):
            ctx=self.context(j);native=self.decode(ctx);q=self.index.conditional_logs(self.signals['original'][j])
            zero=self.decode(ctx,q,0.);old=self.decode(ctx,q,.3)
            assert native['paths']==zero['paths'];np.testing.assert_array_equal(native['scores'],zero['scores'])
            ref=self.reference[self.samples[j]['user_id']]
            for value,key in ((native,'native'),(old,'guided')):
                assert value['ids']==ref[key]['ids'],(j,key)
                np.testing.assert_allclose(value['scores'],ref[key]['scores'],atol=2e-6,rtol=1e-5)
            checks.append(dict(user=self.samples[j]['user_id'],alpha0_exact=True,old_outputs_reproduced=True))
        write(self.folder/'smoke.json',dict(state='PASSED',checks=checks))
        self.stage='matched_new_generation';self.status()
        with (self.folder/'predictions.jsonl').open('x') as f:
            for j,s in enumerate(self.samples):
                ctx=self.context(j);native=self.decode(ctx)
                record=dict(user_id=s['user_id'],target=s['target'],history=s['history'],
                    native=dict(ids=native['ids'],scores=native['scores'].tolist()),
                    original=self.reference[s['user_id']]['guided'])
                # All native paths are freshly verified, not just smoke users.
                ref=self.reference[s['user_id']]['native']
                assert native['ids']==ref['ids']
                np.testing.assert_allclose(native['scores'],ref['scores'],atol=2e-6,rtol=1e-5)
                for arm in ('user','score'):
                    q=self.index.conditional_logs(self.signals[arm][j])
                    late=native['guide'].rescore(native['paths'],q,.3)
                    guided=self.decode(ctx,q,.3)
                    record[arm+'_late']=dict(ids=native['ids'],scores=late.tolist())
                    record[arm+'_guided']=dict(ids=guided['ids'],scores=guided['scores'].tolist())
                f.write(json.dumps(record)+'\n');self.users_completed=j+1
                if (j+1)%16==0 or j+1==len(self.samples):
                    f.flush();os.fsync(f.fileno());self.status()
                del ctx,native,guided
        assert self.digest()==self.initial_digest
        self.stage='complete';self.status('COMPLETED',model_digest_unchanged=True)


def worker(domain):
    import torch
    cfg=read(OUT/'config.json');folder=OUT/domain
    assert os.environ['CUDA_VISIBLE_DEVICES']==cfg['gpus'][domain]
    free=subprocess.check_output(['nvidia-smi','--id='+cfg['gpus'][domain],
        '--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    assert int(free.strip())>=cfg['minimum_free_mib'],'Insufficient GPU admission memory'
    for rel,digest in read(OUT/'manifest.json')['sha256'].items():assert sha(ROOT/rel)==digest,rel
    lease=None
    def interrupt(signum,frame):raise InterruptedError('Worker interrupted')
    signal.signal(signal.SIGTERM,interrupt)
    try:
        with (folder/'lease.log').open('x') as log:
            lease_path=folder/'lease_status.json'
            lease=subprocess.Popen([sys.executable,str(ROOT/'experiment/gpu_memory_lease.py'),'--gpu','0',
                '--total-lease-mib',str(cfg['total_lease_mib']),'--expected-workload-peak-mib',str(cfg['expected_workload_peak_mib']),
                '--status-path',str(lease_path)],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
            begin=time.monotonic()
            while not lease_path.exists() or read(lease_path).get('state')!='holding':
                assert lease.poll() is None,'GPU lease failed';assert time.monotonic()-begin<180
                time.sleep(1)
            Worker(domain,lease).run()
    except BaseException as e:
        write(folder/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()))
        previous=read(folder/'status.json');write(folder/'status.json',dict(state='FAILED',previous=previous,error=repr(e)))
        raise
    finally:
        if lease is not None and lease.poll() is None:
            lease.terminate()
            try:lease.wait(timeout=10)
            except subprocess.TimeoutExpired:lease.kill();lease.wait()


def summarize():
    from experiment.phase20.catalog_scope_core import metrics,contrast
    result={}
    for d in ('Toys','Beauty'):
        p=OUT/d;assert read(p/'status.json')['state']=='COMPLETED'
        samples=read(p/'samples.json');rr={};records=[]
        with (p/'predictions.jsonl').open() as f:records=[json.loads(line) for line in f]
        assert len(records)==len(samples)
        for row,s in zip(records,samples):
            assert row['user_id']==s['user_id']
            for key in ('native','original','user_late','user_guided','score_late','score_guided'):
                v=row[key];assert len(v['ids'])==len(set(v['ids']))==50
                ordered=sorted([(a,b) for a,b in zip(v['ids'],v['scores']) if a not in s['history']],key=lambda x:(-x[1],x[0]))
                rank=next((i+1 for i,(a,b) in enumerate(ordered) if a==s['target']),2**30)
                rr.setdefault(key,[]).append(rank)
        rr={k:np.array(v) for k,v in rr.items()};np.savez_compressed(p/'ranks.npz',**rr)
        pairs=[('score_guided','original'),('score_guided','user_guided'),('score_guided','score_late'),
               ('score_guided','native'),('user_guided','original')]
        result[d]=dict(users=len(samples),metrics={k:metrics(v) for k,v in rr.items()},
            comparisons={a+'_vs_'+b:contrast(rr[a],rr[b],repeats=2000,seed=20261010) for a,b in pairs})
    write(OUT/'summary.json',result);write(OUT/'status.json',dict(state='COMPLETE',test_read=False))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','worker','summarize']);parser.add_argument('domain',nargs='?')
    args=parser.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='worker':worker(args.domain)
    else:summarize()
