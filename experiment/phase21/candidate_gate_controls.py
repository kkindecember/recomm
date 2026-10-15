"""Two fixed architecture controls using read-only features from stage 21.19."""
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch

from experiment.phase21 import candidate_gate_screen as S
from experiment.phase21.candidate_gate_core import Gate,WIDTH,compressed_loss,utilities

ROOT=S.ROOT
SOURCE_OUT=S.OUT
OUT=ROOT/'artifacts/phase21/candidate_gate_controls_20261010_v1'
PLAN=ROOT/'plan/第二十一阶段/21.19C_候选校正结构对照_20261010.md'
START=time.monotonic()


class Scalar(torch.nn.Module):
    def __init__(self):
        super().__init__();self.input=torch.nn.Linear(WIDTH,32);self.output=torch.nn.Linear(32,1)
        torch.nn.init.zeros_(self.output.weight);torch.nn.init.zeros_(self.output.bias)

    def forward(self,x,w0,eta):
        return eta*torch.tanh(self.output(torch.relu(self.input(x))))


class CachedData(S.Data):
    def __init__(self,domain,training,direct):
        self.domain,self.training,self.direct=domain,training,direct
        folder=SOURCE_OUT/domain/('train' if training else 'validation')
        self.a={p.stem:np.load(p,mmap_mode='r') for p in folder.glob('*.npy')}
        rows=json.loads((folder/'users.json').read_text());self.n=len(rows)
        origin=(S.SOURCE if training else S.FULL)/domain
        all_samples=json.loads((origin/'samples.json').read_text())
        self.source_ix=np.array([r['source_index'] for r in rows])
        self.samples=[all_samples[i] for i in self.source_ix]
        self.targets=np.array([r['target'] for r in rows],np.int64)
        self.folds=np.array([r['fold'] for r in rows]);self.cohorts=np.array([r['cohort'] for r in rows])
        self.experts=np.load(origin/'experts.npy',mmap_mode='r')
        self.fused=np.load(origin/('lime.npy' if training else 'fused.npy'),mmap_mode='r')
        with np.load(origin/'gate_prediction.npz') as g:self.w=g['weights'][self.source_ix];self.penalty=g['penalty'][self.source_ix]

    def forward(self,model,ix,arm,norm):
        if not self.direct:return super().forward(model,ix,arm,norm)
        x=np.clip((self.inputs(ix,arm)-norm[0])/norm[1],-10,10)
        delta=model(torch.from_numpy(x),None,S.CFG['eta']).squeeze(-1)
        args=[torch.from_numpy(np.asarray(self.a[k][ix])) for k in ('base','mask','outside','target_base','target_pos')]
        loss,ce,kl=compressed_loss(args[0],delta,*args[1:],lam=S.CFG['kl'])
        return loss,ce,kl,delta


def evaluate(data,model,norm):
    if not data.direct:return S.evaluate(data,model,norm,'score')
    ranks=np.empty(data.n,np.int64);maximum=0.
    with torch.no_grad():
        for start in range(0,data.n,128):
            ix=np.arange(start,min(start+128,data.n))
            x=np.clip((data.inputs(ix,'score')-norm[0])/norm[1],-10,10)
            delta=model(torch.from_numpy(x),None,S.CFG['eta']).squeeze(-1).numpy()
            f=np.array(data.fused[data.source_ix[ix]],copy=True)
            for j,k in enumerate(ix):
                m=data.a['mask'][k];ci=data.a['ids'][k,m];f[j,ci]+=delta[j,m]
                maximum=max(maximum,float(np.abs(delta[j,m]).max()))
                f[j,np.array(data.samples[k]['history'])-1]=-np.inf
            y=data.targets[ix];ts=f[np.arange(len(ix)),y]
            r=1+(f>ts[:,None]).sum(1)+((f==ts[:,None])&(np.arange(f.shape[1])[None,:]<y[:,None])).sum(1)
            r[~np.isfinite(ts)]=f.shape[1]+1;ranks[ix]=r
            if start==0:
                for j in (0,len(ix)-1):
                    order=np.lexsort((np.arange(f.shape[1]),-f[j]))
                    assert r[j]==int(np.flatnonzero(order==y[j])[0])+1
            if start%4096==0:S.status('direct_evaluate',domain=data.domain,completed=start,total=data.n)
    assert maximum<=.1+1e-6
    return ranks,dict(maximum_score_change=maximum)


def main():
    assert json.loads((SOURCE_OUT/'verification.json').read_text())['state']=='VERIFIED'
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    OUT.mkdir(exist_ok=False);shutil.copy2(PLAN,OUT/'protocol.md')
    S.write(OUT/'manifest.json',dict(sha256={str(p.relative_to(ROOT)):S.sha(p) for p in (
        PLAN,Path(__file__),ROOT/'experiment/phase21/candidate_gate_screen.py',ROOT/'experiment/phase21/candidate_gate_core.py',
        SOURCE_OUT/'manifest.json',SOURCE_OUT/'verification.json',SOURCE_OUT/'summary.json')}))
    all_results={};S.CFG['timeout_seconds']=1800
    for control in ('unbounded_gate','direct_residual'):
        direct=control=='direct_residual';S.OUT=OUT/control;S.OUT.mkdir()
        S.CFG['eta']=.1 if direct else 1.;S.Gate=Scalar if direct else Gate
        S.write(S.OUT/'config.json',dict(S.CFG,control=control))
        for d in ('Toys','Beauty'):
            (S.OUT/d).mkdir();train=CachedData(d,True,direct);valid=CachedData(d,False,direct)
            predictions={};fits={};diagnostics={}
            for seed in S.CFG['seeds']:
                model,norm,info=S.fit(train,'score',seed)
                fits[str(seed)]=info;predictions[str(seed)],diagnostics[str(seed)]=evaluate(valid,model,norm)
            np.savez_compressed(S.OUT/d/'ranks.npz',**predictions)
            old=np.load(SOURCE_OUT/d/'ranks.npz');cohorts={}
            for name in ('primary','all'):
                mask=np.ones(valid.n,bool) if name=='all' else valid.cohorts==name
                ranks={**{f'control_{s}':predictions[str(s)][mask] for s in S.CFG['seeds']},
                       **{f'candidate_{s}':old[f'score_{s}'][mask] for s in S.CFG['seeds']},'original':old['original'][mask]}
                metrics={k:dict(N10=float(utilities(r).mean()),H10=float((r<=10).mean()),H50=float((r<=50).mean())) for k,r in ranks.items()}
                cu=np.mean([utilities(ranks[f'control_{s}']) for s in S.CFG['seeds']],axis=0)
                ca=np.mean([utilities(ranks[f'candidate_{s}']) for s in S.CFG['seeds']],axis=0)
                diffs=np.stack([cu-ca,cu-utilities(ranks['original'])],axis=1);rng=np.random.default_rng(20261010);boots=[]
                for start in range(0,2000,50):boots.append(diffs[rng.integers(0,len(cu),size=(50,len(cu)))].mean(1))
                ci=np.quantile(np.concatenate(boots),[.025,.975],axis=0)
                cohorts[name]=dict(users=len(cu),metrics=metrics,control_mean_N10=float(cu.mean()),
                    candidate_mean_N10=float(ca.mean()),delta_vs_candidate=float((cu-ca).mean()),ci95_vs_candidate=ci[:,0].tolist(),
                    delta_vs_original=float(diffs[:,1].mean()),ci95_vs_original=ci[:,1].tolist())
            result=dict(control=control,domain=d,fits=fits,diagnostics=diagnostics,cohorts=cohorts)
            S.write(S.OUT/d/'summary.json',result);all_results.setdefault(control,{})[d]=result
            S.write(OUT/'summary.json',all_results)
            S.write(OUT/'status.json',dict(state='RUNNING',control=control,domain=d,elapsed_seconds=time.monotonic()-START))
    S.write(OUT/'status.json',dict(state='COMPLETE',elapsed_seconds=time.monotonic()-START,test_read=False))


if __name__=='__main__':
    try:main()
    except BaseException as e:
        if OUT.exists():S.write(OUT/'status.json',dict(state='FAILED',error=repr(e),traceback=traceback.format_exc()))
        raise
