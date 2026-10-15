"""Exploratory single-head fusion; no original LIME gate or penalty inputs.

Both features and candidate proposals use ONLY the retained experts. The
three-expert and SASRec+text models are trained from scratch on ranker_fit.
Outside the correction set the retained-expert mean is the fixed fallback.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import time
import traceback

import numpy as np
import torch

from experiment.phase21.candidate_gate_core import ranking, fold, cohort, compressed_loss, utilities
from experiment.phase21.candidate_gate_screen import ROOT, SOURCE, FULL, write, sha

OUT = ROOT/'artifacts/phase21/compact_candidate_gate_20261014_v1'
SETS = {'three':[0,1,2], 'without_itemcf':[0,2]}
CFG = dict(seed=0, epochs=20, checkpoints=[0,4,8,12,20], batch_size=64, lr=.002, weight_decay=.001,
           kl=0., temperature=20., threads=2, timeout_seconds=21600, user_selection='ranker_fit',
           evaluation='exploratory_validation', test_read=False, old_gate_used=False,
           history_penalty=False, selection_criterion='guided NDCG@10 above original GRAM in both domains')
START = time.monotonic()


def status(stage, **kwargs):
    write(OUT/'status.json',dict(state='RUNNING',stage=stage,pid=os.getpid(),elapsed_seconds=time.monotonic()-START,**kwargs))
    print(stage,kwargs,flush=True)
    if time.monotonic()-START > CFG['timeout_seconds']: raise TimeoutError('Compact exploration time budget')


def features(retained, history, counts):
    """No label, old fused score, dropped-expert score, or old weight argument."""
    e = np.asarray(retained,dtype=np.float32)
    k,v = e.shape
    f = e.mean(0)
    eligible = np.ones(v,bool); eligible[np.asarray(history[-20:])-1] = False
    orders,ranks = zip(*(ranking(x,eligible) for x in [*e,f]))
    ci = np.unique(np.concatenate([o[:50] for o in orders]))
    ref = orders[-1][9]
    scores = e[:,ci].T
    er = np.stack([r[ci] for r in ranks[:-1]],axis=1)
    cx = np.column_stack([scores,scores-e[:,ref],1./er,er<=10,f[ci],f[ci]-f[ref],1./ranks[-1][ci],
                          np.full(len(ci),f[ref]-f[orders[-1][10]]),np.full(len(ci),f[orders[-1][0]]-f[ref]),
                          np.log1p(counts[ci+1])]).astype(np.float32)
    hist = np.zeros(v,bool);hist[np.asarray(history)-1]=True
    # Stable descending score / ascending item-ID tie rule on train AND validation.
    fullorders = [np.lexsort((np.arange(v),-x)) for x in e]
    top = np.asarray([e[j,o[:2]] for j,o in enumerate(fullorders)])
    uf = np.concatenate([[np.log1p(len(history))/5],top[:,0]-e.mean(1),top[:,0]-top[:,1],
                         e.std(1,ddof=1),[hist[o[:5]].mean() for o in fullorders],[hist.mean()]]).astype(np.float32)
    assert len(ci) <= 50*(k+1) and len(uf)==4*k+2 and cx.shape==(len(ci),4*k+6)
    return ci, uf, cx, f, ranks[-1]


class Head(torch.nn.Module):
    def __init__(self, width, experts):
        super().__init__()
        self.input = torch.nn.Linear(width,32)
        self.output = torch.nn.Linear(32,experts)
        torch.nn.init.zeros_(self.output.weight);torch.nn.init.zeros_(self.output.bias)

    def forward(self,x):
        return self.output(torch.relu(self.input(x))).softmax(-1)


class Data:
    def __init__(self,domain,name,training,manifest):
        self.domain,self.name,self.training = domain,name,training
        self.keep = SETS[name];self.k=len(self.keep);self.du=4*self.k+2;self.dc=4*self.k+6
        self.width=self.du+self.dc;self.m=50*(self.k+1)
        origin=(SOURCE if training else FULL)/domain
        rows=json.loads((origin/'samples.json').read_text())
        self.ix=np.array([i for i,s in enumerate(rows) if not training or s['role']=='ranker_fit'])
        self.samples=[rows[i] for i in self.ix];self.n=len(self.samples)
        self.folds=np.array([fold(s['user_id'],domain) for s in self.samples])
        self.cohorts=np.array(['train' if training else cohort(s) for s in self.samples])
        self.y=np.array([s['target']-1 for s in self.samples])
        self.experts=np.load(origin/'experts.npy',mmap_mode='r')
        counts=np.load(SOURCE/domain/'frequencies.npy')
        self.folder=OUT/name/domain/('train' if training else 'validation');self.folder.mkdir(parents=True)
        self.a={}
        def arr(key,shape,dtype='float32'):
            self.a[key]=np.lib.format.open_memmap(self.folder/(key+'.npy'),mode='w+',dtype=dtype,shape=shape)
            return self.a[key]
        u=arr('user',(self.n,self.du));c=arr('candidate',(self.n,self.m,self.dc))
        ids=arr('ids',(self.n,self.m),'int64');mask=arr('mask',(self.n,self.m),'bool')
        scores=arr('scores',(self.n,self.m,self.k));b=arr('base',(self.n,self.m))
        out=arr('outside',(self.n,));tb=arr('target_base',(self.n,));tp=arr('target_pos',(self.n,),'int64')
        ranks=arr('mean_rank',(self.n,),'int64')
        for fname in ('samples.json','experts.npy'):
            path=origin/fname;rel=str(path.relative_to(ROOT))
            if rel not in manifest: manifest[rel]=sha(path)
        for j,src in enumerate(self.ix):
            s=self.samples[j];e=np.array(self.experts[src,self.keep,:],copy=True)
            ci,uf,cx,f,rr=features(e,s['full_history'],counts);n=len(ci)
            u[j]=uf;c[j,:n]=cx;ids[j,:n]=ci;mask[j,:n]=True;scores[j,:n]=e[:,ci].T
            z=f.astype(np.float64)*20;z-=z.max();b[j,:n]=z[ci]
            outside=np.ones(len(f),bool);outside[ci]=False
            out[j]=np.log(np.exp(z[outside]).sum());tb[j]=z[self.y[j]]
            match=np.flatnonzero(ci==self.y[j]);tp[j]=int(match[0]) if len(match) else -1
            ranks[j]=rr[self.y[j]]
            if (j+1)%2048==0:status('features',domain=domain,experts=name,training=training,completed=j+1,total=self.n)
        for a in self.a.values(): a.flush()
        write(self.folder/'users.json',[dict(user_id=s['user_id'],source_index=int(src),target=int(self.y[j]),cohort=str(self.cohorts[j])) for j,(s,src) in enumerate(zip(self.samples,self.ix))])
        write(self.folder/'metadata.json',dict(users=self.n,expert_indices=self.keep,width=self.width,max_candidates=self.m,target_in_candidates=int((tp>=0).sum()),mean_candidates=float(mask.sum(1).mean()),original_gate_accessed=False))

    def inputs(self,ix,arm):
        m=1 if arm=='user' else self.m
        x=np.zeros((len(ix),m,self.width),np.float32);x[:,:,:self.du]=self.a['user'][ix,None,:]
        if arm=='candidate':x[:,:,self.du:]=self.a['candidate'][ix]
        return x

    def norm(self,ix,arm):
        total=np.zeros(self.width);sq=total.copy();n=0
        for start in range(0,len(ix),128):
            jj=ix[start:start+128];x=self.inputs(jj,arm)
            mask=np.ones(x.shape[:2],bool) if arm=='user' else self.a['mask'][jj]
            v=x[mask].astype(np.float64);total+=v.sum(0);sq+=(v*v).sum(0);n+=len(v)
        mean=total/n;std=np.sqrt(np.maximum(sq/n-mean*mean,0));std[std<1e-5]=1.
        return mean.astype(np.float32),std.astype(np.float32)

    def delta(self,model,ix,arm,norm):
        x=np.clip((self.inputs(ix,arm)-norm[0])/norm[1],-10,10)
        w=model(torch.from_numpy(x))
        es=torch.from_numpy(np.array(self.a['scores'][ix],copy=True))
        return (w*es).sum(2)-es.mean(2),w

    def objective(self,model,ix,arm,norm):
        delta,_=self.delta(model,ix,arm,norm)
        args=[torch.from_numpy(np.array(self.a[k][ix],copy=True)) for k in ('base','mask','outside','target_base','target_pos')]
        return compressed_loss(args[0],delta,*args[1:],lam=CFG['kl'])[0]


def fit(data,arm):
    fi=np.flatnonzero(data.folds!=0);hi=np.flatnonzero(data.folds==0);norm=data.norm(fi,arm)
    def init():torch.manual_seed(CFG['seed']);return Head(data.width,data.k)
    model=init();opt=torch.optim.AdamW(model.parameters(),lr=CFG['lr'],weight_decay=CFG['weight_decay'])
    rng=np.random.default_rng(CFG['seed']);curve=[]
    def heldout():
        with torch.no_grad():return sum(float(data.objective(model,hi[s:s+64],arm,norm))*len(hi[s:s+64]) for s in range(0,len(hi),64))/len(hi)
    curve.append(dict(epoch=0,loss=heldout()))
    for epoch in range(1,CFG['epochs']+1):
        order=rng.permutation(fi)
        for start in range(0,len(order),64):
            opt.zero_grad();loss=data.objective(model,order[start:start+64],arm,norm)
            assert torch.isfinite(loss);loss.backward();opt.step()
        if epoch in CFG['checkpoints']:
            curve.append(dict(epoch=epoch,loss=heldout()));status('selection',domain=data.domain,experts=data.name,arm=arm,epoch=epoch,heldout=curve[-1]['loss'])
    selected=min(curve,key=lambda x:(x['loss'],x['epoch']))['epoch']
    norm=data.norm(np.arange(data.n),arm);model=init()
    opt=torch.optim.AdamW(model.parameters(),lr=CFG['lr'],weight_decay=CFG['weight_decay']);rng=np.random.default_rng(CFG['seed'])
    for epoch in range(selected):
        order=rng.permutation(data.n)
        for start in range(0,len(order),64):
            opt.zero_grad();loss=data.objective(model,order[start:start+64],arm,norm);assert torch.isfinite(loss);loss.backward();opt.step()
    ck=dict(state_dict=model.state_dict(),mean=norm[0],std=norm[1],experts=data.keep,arm=arm,selected_epochs=selected,
            fit_users=[s['user_id'] for s in data.samples],config=CFG,curve=curve,width=data.width,parameters=sum(p.numel() for p in model.parameters()))
    torch.save(ck,OUT/data.name/data.domain/(arm+'.pt'))
    return model,norm,dict(selected_epochs=selected,curve=curve,parameters=ck['parameters'])


def evaluate(data,model,norm,arm):
    ranks=np.empty(data.n,np.int64);checks=[];weight_spread=[]
    gen=json.loads((ROOT/'artifacts/phase21/candidate_gate_generation_20261010_v1'/data.domain/'samples.json').read_text())
    chosen={s['user_id']:j for j,s in enumerate(gen)}
    signals=np.lib.format.open_memmap(OUT/data.name/data.domain/(arm+'_generation_signals.npy'),mode='w+',dtype='float32',shape=(len(gen),data.experts.shape[2]))
    with torch.no_grad():
        for start in range(0,data.n,128):
            ix=np.arange(start,min(start+128,data.n));delta,w=data.delta(model,ix,arm,norm);delta=delta.numpy();w=w.numpy()
            for j,k in enumerate(ix):
                f=np.array(data.experts[data.ix[k],data.keep,:],copy=True).mean(0)
                mask=data.a['mask'][k];ids=data.a['ids'][k,mask];f[ids]+=delta[j,mask]
                s=data.samples[k]
                if s['user_id'] in chosen:signals[chosen[s['user_id']]]=f
                if arm=='candidate':weight_spread.append(float(w[j,mask].std(0).mean()))
                if k in (0,data.n//2,data.n-1):
                    # Independent NumPy MLP + expert mixture verifies all candidate scores.
                    x=np.clip((data.inputs(np.array([k]),arm)-norm[0])/norm[1],-10,10)
                    sd=model.state_dict();z=np.maximum(x@sd['input.weight'].numpy().T+sd['input.bias'].numpy(),0)
                    z=z@sd['output.weight'].numpy().T+sd['output.bias'].numpy();p=np.exp(z-z.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
                    expected=(p[0]*data.a['scores'][k]).sum(-1)[mask]
                    np.testing.assert_allclose(f[ids],expected,atol=2e-6,rtol=1e-5)
                    checks.append(int(k))
                f[np.asarray(s['history'])-1]=-np.inf;y=data.y[k];t=f[y]
                ranks[k]=1+np.count_nonzero(f>t)+np.count_nonzero((f==t)&(np.arange(len(f))<y)) if np.isfinite(t) else len(f)+1
                if k in checks:
                    expected_rank=int(np.flatnonzero(np.lexsort((np.arange(len(f)),-f))==y)[0])+1
                    if np.isfinite(t):assert ranks[k]==expected_rank
            if start%4096==0:status('evaluate',domain=data.domain,experts=data.name,arm=arm,completed=start,total=data.n)
    signals.flush()
    return ranks,dict(numpy_forward_checked=checks,mean_candidate_weight_std=float(np.mean(weight_spread)) if weight_spread else 0.)


def summarize(domain,data,ranks):
    old=np.load(ROOT/'artifacts/phase21/candidate_gate_20261010_v1'/domain/'ranks.npz')
    results={}
    for name in ('primary','previous512','all'):
        mask=np.ones(data.n,bool) if name=='all' else data.cohorts==name
        rr={k:v[mask] for k,v in ranks.items()}
        rr['old_fusion']=old['original'][mask];rr['old_candidate_eta01']=old['score_0'][mask]
        result={k:dict(N10=float(utilities(v).mean()),H10=float((v<=10).mean()),H50=float((v<=50).mean())) for k,v in rr.items()}
        delta=utilities(rr['candidate'])-utilities(rr['user'])
        rng=np.random.default_rng(20261014)
        boot=np.concatenate([delta[rng.integers(0,len(delta),size=(50,len(delta)))].mean(1) for _ in range(20)])
        results[name]=dict(users=int(mask.sum()),metrics=result,candidate_minus_user=dict(delta_N10=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist()))
    return results


def tests():
    rng=np.random.default_rng(5);all_e=rng.random((3,213)).astype(np.float32)
    a=features(all_e[[0,2]],[1,3,7],np.ones(214))
    all_e[1]=rng.normal(100,50,213)
    b=features(all_e[[0,2]],[1,3,7],np.ones(214))
    for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
    h=Head(24,2);w=h(torch.randn(2,5,24));torch.testing.assert_close(w,torch.full_like(w,.5))
    # Existing compressed objective is mathematically exact, including outside targets.
    from experiment.phase21.candidate_gate_core import tests as core_tests
    core_tests()
    return dict(dropped_expert_perturbation_invariant=True,uniform_initialization=True,compressed_objective_dense_equivalent=True)


def main():
    torch.set_num_threads(CFG['threads']);torch.set_num_interop_threads(1)
    OUT.mkdir(exist_ok=False);write(OUT/'config.json',CFG);write(OUT/'tests.json',tests())
    shutil.copy2(__file__,OUT/Path(__file__).name)
    paths=[Path(__file__),ROOT/'experiment/phase21/candidate_gate_core.py',ROOT/'experiment/phase21/candidate_gate_screen.py']
    paths += [SOURCE/d/'frequencies.npy' for d in ('Toys','Beauty')]
    paths += [ROOT/'artifacts/phase21/candidate_gate_generation_20261010_v1'/d/'samples.json' for d in ('Toys','Beauty')]
    manifest={str(p.relative_to(ROOT)):sha(p) for p in paths};results={}
    for name in SETS:
        for domain in ('Toys','Beauty'):
            train=Data(domain,name,True,manifest);valid=Data(domain,name,False,manifest)
            assert {s['user_id'] for s in train.samples}.isdisjoint(s['user_id'] for s in valid.samples if cohort(s)=='primary')
            ranks={'mean':np.array(valid.a['mean_rank'])};fits={};checks={}
            for arm in ('user','candidate'):
                model,norm,info=fit(train,arm);fits[arm]=info
                ranks[arm],checks[arm]=evaluate(valid,model,norm,arm)
                np.save(OUT/name/domain/(arm+'_ranks.npy'),ranks[arm])
            np.savez_compressed(OUT/name/domain/'ranks.npz',**ranks)
            result=dict(cohorts=summarize(domain,valid,ranks),fits=fits,checks=checks)
            write(OUT/name/domain/'summary.json',result);results.setdefault(name,{})[domain]=result
            write(OUT/'summary.json',results);write(OUT/'manifest.json',dict(sha256=manifest))
    write(OUT/'status.json',dict(state='COMPLETE',elapsed_seconds=time.monotonic()-START,test_read=False))


if __name__=='__main__':
    def interrupt(signum,frame):raise TimeoutError('Compact head interrupted')
    signal.signal(signal.SIGTERM,interrupt)
    try:main()
    except BaseException as e:
        if OUT.exists():write(OUT/'status.json',dict(state='FAILED',error=repr(e),traceback=traceback.format_exc()))
        raise
