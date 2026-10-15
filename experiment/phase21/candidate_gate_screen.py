"""Matched candidate-gate experiment, frozen caches and CPU only.

Run: OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
     <gram-repro-python> -m experiment.phase21.candidate_gate_screen
"""
import json
import hashlib
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch

from experiment.phase21.candidate_gate_core import (ARMS,USER_NAMES,CAND_NAMES,EVID_NAMES,DU,DC,DE,WIDTH,
    Evidence,Gate,candidates,user_features,fold,cohort,donor_indices,compressed_loss,ranking,utilities,tests)

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'artifacts/phase20/lime_clean_benefit_20260922_v1'
FULL=ROOT/'artifacts/phase21/full_validation_20260924_v1'
OUT=ROOT/'artifacts/phase21/candidate_gate_20261010_v1'
PLAN=ROOT/'plan/第二十一阶段/21.19_候选级权重与历史依据匹配试验_20261010.md'
CFG=dict(seeds=[0,1,2],epochs=20,checkpoints=[0,4,8,12,20],batch_size=64,lr=.002,weight_decay=.001,
         eta=.1,kl=.2,threads=4,timeout_seconds=14400,bootstrap_repeats=2000,bootstrap_seed=20261010,
         train_labels='ranker_fit_only',test_read=False,upstream_optimizer_steps=0)
START=time.monotonic()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*2**20),b''):h.update(b)
    return h.hexdigest()


def status(stage,**kw):
    elapsed=time.monotonic()-START
    write(OUT/'status.json',dict(state='RUNNING',stage=stage,pid=os.getpid(),elapsed_seconds=elapsed,**kw))
    if elapsed>CFG['timeout_seconds']:raise TimeoutError('Fixed experiment time budget exceeded')
    print(stage,kw,round(elapsed,1),flush=True)


class Data:
    def __init__(self,domain,training,engine,manifest):
        self.domain,self.training=domain,training
        folder=(SOURCE if training else FULL)/domain
        name='train' if training else 'validation'
        self.out=OUT/domain/name;self.out.mkdir(parents=True)
        rows=json.loads((folder/'samples.json').read_text())
        selected=[i for i,s in enumerate(rows) if not training or s['role']=='ranker_fit']
        self.source_ix=np.array(selected);self.samples=[rows[i] for i in selected]
        self.n=len(selected)
        self.experts=np.load(folder/'experts.npy',mmap_mode='r')
        self.fused=np.load(folder/('lime.npy' if training else 'fused.npy'),mmap_mode='r')
        with np.load(folder/'gate_prediction.npz') as g:
            self.w=g['weights'][selected];self.penalty=g['penalty'][selected]
            original_f=g['features'][selected] if 'features' in g else None
        self.targets=np.array([s['target']-1 for s in self.samples],np.int64)
        self.folds=np.array([fold(s['user_id'],domain) for s in self.samples])
        self.cohorts=np.array(['train' if training else cohort(s) for s in self.samples])
        ds=donor_indices(self.samples,domain,training)
        counts=np.load(SOURCE/domain/'frequencies.npy')
        self.a={}
        def arr(key,shape,dtype='float32'):
            self.a[key]=np.lib.format.open_memmap(self.out/(key+'.npy'),mode='w+',dtype=dtype,shape=shape)
            return self.a[key]
        u=arr('user',(self.n,DU));c=arr('candidate',(self.n,200,DC))
        ev=arr('evidence',(self.n,200,DE));sh=arr('shuffled',(self.n,200,DE))
        ids=arr('ids',(self.n,200),'int64');mask=arr('mask',(self.n,200),'bool')
        es=arr('scores',(self.n,200,3));base=arr('base',(self.n,200));outside=arr('outside',(self.n,))
        target_base=arr('target_base',(self.n,));target_pos=arr('target_pos',(self.n,),'int64')
        original_rank=arr('original_rank',(self.n,),'int64');baseline_ce=[];source_stats={}
        max_reconstruction_error=0.;gate_feature_diffs=[]
        for fname in ('samples.json','experts.npy','lime.npy' if training else 'fused.npy','gate_prediction.npz'):
            path=folder/fname;source_stats[str(path)]=(path.stat().st_size,path.stat().st_mtime_ns)
            manifest[str(path.relative_to(ROOT))]=sha(path)
        if not training:
            expected=json.loads((folder/'score_cache_manifest.json').read_text())['sha256']
            for fname,digest in expected.items():
                key=str((folder/fname).relative_to(ROOT))
                if key in manifest:assert manifest[key]==digest,(domain,fname)
        for j,(src,s) in enumerate(zip(selected,self.samples)):
            e=np.asarray(self.experts[src]);f=np.asarray(self.fused[src]);h=s['full_history']
            ci,ref,cx,rr=candidates(e,f,h,counts);m=len(ci)
            ids[j,:m]=ci;mask[j,:m]=True;c[j,:m]=cx
            u[j]=user_features(e,h,self.w[j],self.penalty[j])
            # Use the same CPU feature definition for both splits. Record any
            # topk tie/numerical difference from the original cached gate input.
            if original_f is not None:
                gate_feature_diffs.append(float(np.max(np.abs(u[j,:14]-original_f[j]))))
            es[j,:m]=e[:,ci].T
            ev[j,:m]=engine.features(ci,ref,h)
            sh[j,:m]=engine.features(ci,ref,self.samples[int(ds[j])]['full_history'])
            scaled=f.astype(np.float64)*20;scaled-=scaled.max()
            base[j,:m]=scaled[ci]
            outside_mask=np.ones(len(f),bool);outside_mask[ci]=False
            outside[j]=np.log(np.exp(scaled[outside_mask]).sum())
            target_base[j]=scaled[self.targets[j]]
            match=np.flatnonzero(ci==self.targets[j]);target_pos[j]=int(match[0]) if len(match) else -1
            original_rank[j]=rr[self.targets[j]]
            baseline_ce.append(float(np.log(np.exp(scaled).sum())-scaled[self.targets[j]]))
            if j in (0,self.n//2,self.n-1):
                engine.check(ci,ref,h)
                hist=np.zeros(len(f),bool);hist[np.asarray(h)-1]=True
                expected=(e*self.w[j,:,None]).sum(0)-self.penalty[j]*hist
                err=float(np.max(np.abs(f-expected)));max_reconstruction_error=max(err,max_reconstruction_error)
                assert err<5e-7
                again=candidates(e,f,h,counts)
                np.testing.assert_array_equal(ci,again[0]);np.testing.assert_array_equal(cx,again[2])
            if (j+1)%1024==0:status('features',domain=domain,split=name,completed=j+1,total=self.n)
        for a in self.a.values():a.flush()
        for path,stat in source_stats.items():
            assert (Path(path).stat().st_size,Path(path).stat().st_mtime_ns)==stat
        self.info=dict(users=self.n,unique_users=len({s['user_id'] for s in self.samples}),
            target_in_candidates=int((target_pos>=0).sum()),mean_candidates=float(mask.sum(1).mean()),
            cohorts={v:int((self.cohorts==v).sum()) for v in sorted(set(self.cohorts))},
            max_fused_reconstruction_error=max_reconstruction_error,baseline_ce=float(np.mean(baseline_ce)),
            max_train_cpu_vs_cached_feature_difference=max(gate_feature_diffs,default=0),
            changed_donor_users=self.n,source_unchanged=True)
        write(self.out/'metadata.json',self.info)
        write(self.out/'users.json',[dict(user_id=s['user_id'],source_index=int(src),target=int(self.targets[i]),
              fold=int(self.folds[i]),cohort=str(self.cohorts[i]),donor_user=self.samples[int(ds[i])]['user_id'])
              for i,(src,s) in enumerate(zip(selected,self.samples))])
        status('features_complete',domain=domain,split=name,**self.info)

    def inputs(self,ix,arm):
        global_arm=arm in ('user','global')
        m=1 if global_arm else 200
        x=np.zeros((len(ix),m,WIDTH),np.float32)
        x[:,:,:DU]=self.a['user'][ix,None,:]
        if not global_arm:x[:,:,DU:DU+DC]=self.a['candidate'][ix]
        if arm in ('evidence','shuffled'):x[:,:,DU+DC:]=self.a[arm][ix]
        return x

    def norm(self,ix,arm):
        total=np.zeros(WIDTH);square=total.copy();n=0
        for start in range(0,len(ix),128):
            jj=ix[start:start+128];x=self.inputs(jj,arm)
            mask=np.ones(x.shape[:2],bool) if x.shape[1]==1 else self.a['mask'][jj]
            v=x[mask].astype(np.float64);total+=v.sum(0);square+=(v*v).sum(0);n+=len(v)
        mean=total/n;std=np.sqrt(np.maximum(square/n-mean**2,0))
        std[std<1e-5]=1.
        return mean.astype(np.float32),std.astype(np.float32)

    def forward(self,model,ix,arm,norm):
        x=self.inputs(ix,arm);x=(x-norm[0])/norm[1]
        # Extreme out-of-training values are bounded identically for every arm.
        x=np.clip(x,-10,10)
        dw=model(torch.from_numpy(x),torch.from_numpy(self.w[ix,None,:]),CFG['eta'])
        if arm=='global':
            e=torch.from_numpy(np.array(self.experts[self.source_ix[ix]],copy=True))
            delta=(e*dw.transpose(1,2)).sum(1)
            base=torch.from_numpy(np.array(self.fused[self.source_ix[ix]],copy=True))*20
            z=base+20*delta;target=torch.from_numpy(self.targets[ix])
            ce=torch.nn.functional.cross_entropy(z,target)
            kl=(base.softmax(1)*(base.log_softmax(1)-z.log_softmax(1))).sum(1).mean()
            return ce+CFG['kl']*kl,ce,kl,dw
        delta=(dw*torch.from_numpy(self.a['scores'][ix])).sum(2)
        args=[torch.from_numpy(np.asarray(self.a[k][ix])) for k in ('base','mask','outside','target_base','target_pos')]
        loss,ce,kl=compressed_loss(args[0],delta,*args[1:],lam=CFG['kl'])
        return loss,ce,kl,dw


def fit(data,arm,seed):
    fit_ix=np.flatnonzero(data.folds!=0);held_ix=np.flatnonzero(data.folds==0)
    assert not {data.samples[i]['user_id'] for i in fit_ix}&{data.samples[i]['user_id'] for i in held_ix}
    norm=data.norm(fit_ix,arm)
    def initialize():
        torch.manual_seed(seed);return Gate()
    model=initialize();opt=torch.optim.AdamW(model.parameters(),lr=CFG['lr'],weight_decay=CFG['weight_decay'])
    rng=np.random.default_rng(seed);curve=[]
    def objective():
        vals=[]
        with torch.no_grad():
            for start in range(0,len(held_ix),CFG['batch_size']):
                ix=held_ix[start:start+CFG['batch_size']];out=data.forward(model,ix,arm,norm)
                vals.append([len(ix)]+[float(z) for z in out[:3]])
        v=np.asarray(vals);return np.average(v[:,1:],axis=0,weights=v[:,0]).tolist()
    curve.append(dict(epoch=0,heldout=objective()))
    for epoch in range(1,CFG['epochs']+1):
        order=rng.permutation(fit_ix);losses=[]
        for start in range(0,len(order),CFG['batch_size']):
            ix=order[start:start+CFG['batch_size']];opt.zero_grad()
            loss,*_=data.forward(model,ix,arm,norm)
            assert torch.isfinite(loss);loss.backward();opt.step();losses.append(float(loss.detach()))
        if epoch in CFG['checkpoints']:
            curve.append(dict(epoch=epoch,heldout=objective(),train_loss=float(np.mean(losses))))
            status('fit_selection',domain=data.domain,arm=arm,seed=seed,epoch=epoch,heldout=curve[-1]['heldout'])
    selected=min(curve,key=lambda r:(r['heldout'][0],r['epoch']))['epoch']
    norm=data.norm(np.arange(data.n),arm);model=initialize()
    opt=torch.optim.AdamW(model.parameters(),lr=CFG['lr'],weight_decay=CFG['weight_decay'])
    rng=np.random.default_rng(seed)
    for epoch in range(1,selected+1):
        order=rng.permutation(data.n)
        for start in range(0,len(order),CFG['batch_size']):
            ix=order[start:start+CFG['batch_size']];opt.zero_grad()
            loss,*_=data.forward(model,ix,arm,norm);assert torch.isfinite(loss);loss.backward();opt.step()
        if epoch%4==0:status('refit',domain=data.domain,arm=arm,seed=seed,epoch=epoch,total=selected)
    dest=OUT/data.domain/(arm+'_seed'+str(seed)+'.pt')
    torch.save(dict(state_dict=model.state_dict(),mean=norm[0],std=norm[1],config=CFG,
                    arm=arm,seed=seed,selected_epochs=selected,fit_users=[s['user_id'] for s in data.samples]),dest)
    active=DU if arm in ('user','global') else DU+DC if arm=='score' else WIDTH
    info=dict(seed=seed,selected_epochs=selected,curve=curve,parameters=sum(p.numel() for p in model.parameters()),
              active_input_dimensions=active,fit_users=len(fit_ix),internal_holdout_users=len(held_ix))
    write(dest.with_suffix('.json'),info)
    return model,norm,info


def evaluate(data,model,norm,arm):
    ranks=np.empty(data.n,np.int64); maxdelta=0.;avg_weight_std=[];changed=0
    with torch.no_grad():
        for start in range(0,data.n,128):
            ix=np.arange(start,min(start+128,data.n));x=data.inputs(ix,arm)
            dw=model(torch.from_numpy(np.clip((x-norm[0])/norm[1],-10,10)),
                     torch.from_numpy(data.w[ix,None,:]),CFG['eta']).numpy()
            f=np.array(data.fused[data.source_ix[ix]],copy=True)
            if arm=='global':
                d=(np.asarray(data.experts[data.source_ix[ix]])*dw.transpose(0,2,1)).sum(1);f+=d
                maxdelta=max(maxdelta,float(np.abs(d).max()))
            else:
                d=(dw*data.a['scores'][ix]).sum(2)
                for j,k in enumerate(ix):
                    m=data.a['mask'][k];ci=data.a['ids'][k,m];f[j,ci]+=d[j,m]
                    maxdelta=max(maxdelta,float(np.abs(d[j,m]).max()))
                    if dw.shape[1]>1:avg_weight_std.append(float(dw[j,m].std(0).mean()))
            for j,k in enumerate(ix):
                f[j,np.asarray(data.samples[k]['full_history'][-20:])-1]=-np.inf
            y=data.targets[ix];ts=f[np.arange(len(ix)),y]
            r=1+(f>ts[:,None]).sum(1)+((f==ts[:,None])&(np.arange(f.shape[1])[None,:]<y[:,None])).sum(1)
            r[~np.isfinite(ts)]=f.shape[1]+1;ranks[ix]=r
            if start==0:
                for j in (0,len(ix)-1):
                    order=np.lexsort((np.arange(f.shape[1]),-f[j]));actual=int(np.flatnonzero(order==y[j])[0])+1
                    if np.isfinite(ts[j]):assert r[j]==actual
            changed+=int((r!=data.a['original_rank'][ix]).sum())
            if start%4096==0:status('evaluate',domain=data.domain,arm=arm,completed=start,total=data.n)
    assert maxdelta<=CFG['eta']+1e-6
    return ranks,dict(max_score_change=maxdelta,mean_candidate_weight_std=float(np.mean(avg_weight_std)) if avg_weight_std else 0.,
                       target_rank_changed_users=changed)


def summarize(data,ranks):
    result={};keys=list(ranks)
    for name in ('primary','previous512','seen_fit_user_later_event','all'):
        ix=np.arange(data.n) if name=='all' else np.flatnonzero(data.cohorts==name)
        rr={k:v[ix] for k,v in ranks.items()}
        m={k:dict(N10=float(utilities(v).mean()),H10=float((v<=10).mean()),H50=float((v<=50).mean())) for k,v in rr.items()}
        # Average per-user utilities across head seeds; seeds are not users.
        u={'original':utilities(rr['original'])}
        for a in ARMS:u[a]=np.mean([utilities(rr[f'{a}_{s}']) for s in CFG['seeds']],axis=0)
        contrasts=[('score','user'),('score','global'),('score','original'),('evidence','score'),
                   ('evidence','shuffled'),('evidence','original'),('user','original'),('global','original')]
        diffs=np.stack([u[a]-u[b] for a,b in contrasts],1)
        rng=np.random.default_rng(CFG['bootstrap_seed']);boots=[]
        for start in range(0,CFG['bootstrap_repeats'],50):
            pick=rng.integers(0,len(ix),size=(min(50,CFG['bootstrap_repeats']-start),len(ix)))
            boots.append(diffs[pick].mean(1))
        boots=np.concatenate(boots)
        ci=np.quantile(boots,[.025,.975],axis=0)
        cs={a+'_vs_'+b:dict(delta_N10=float(diffs[:,j].mean()),ci95=ci[:,j].tolist(),
             seed_deltas=[m[f'{a}_{s}']['N10']-(m['original']['N10'] if b=='original' else m[f'{b}_{s}']['N10']) for s in CFG['seeds']])
             for j,(a,b) in enumerate(contrasts)}
        result[name]=dict(users=len(ix),metrics=m,comparisons=cs,
            seed_mean_N10={k:float(v.mean()) for k,v in u.items()})
    return result


def main():
    torch.set_num_threads(CFG['threads']);torch.set_num_interop_threads(1)
    OUT.mkdir(exist_ok=False);write(OUT/'config.json',dict(CFG,user_names=USER_NAMES,candidate_names=CAND_NAMES,evidence_names=EVID_NAMES))
    shutil.copy2(PLAN,OUT/'protocol.md');manifest={str(PLAN.relative_to(ROOT)):sha(PLAN)}
    for name in ('candidate_gate_core.py','candidate_gate_screen.py'):
        path=ROOT/'experiment/phase21'/name;shutil.copy2(path,OUT/name);manifest[str(path.relative_to(ROOT))]=sha(path)
    check=tests();write(OUT/'correctness_checks.json',check);status('checks_passed',checks=check)
    summary={}
    for domain in ('Toys','Beauty'):
        source=SOURCE/domain;cfg=json.loads((source/'data_config.json').read_text())
        catalog=json.loads((source/'catalog.json').read_text())
        for path in (source/'data_config.json',source/'catalog.json',source/'frequencies.npy',ROOT/cfg['embedding'],ROOT/cfg['itemcf']):
            manifest[str(path.relative_to(ROOT))]=sha(path)
        with np.load(ROOT/cfg['embedding']) as a:
            assert a['item_ids'].tolist()==catalog['items'][1:];emb=a['embeddings']
        engine=Evidence(catalog['items'][1:],emb,json.loads((ROOT/cfg['itemcf']).read_text()))
        train=Data(domain,True,engine,manifest)
        # Fit all models before reading any validation outcome in this run.
        models={};fitinfo={}
        for arm in ARMS:
            for seed in CFG['seeds']:
                model,norm,info=fit(train,arm,seed);key=f'{arm}_{seed}'
                models[key]=(model,norm,arm);fitinfo[key]=info
        valid=Data(domain,False,engine,manifest)
        train_users={s['user_id'] for s in train.samples}
        assert not train_users&{s['user_id'] for s,c in zip(valid.samples,valid.cohorts) if c=='primary'}
        assert valid.info['cohorts']['primary']==(15406 if domain=='Toys' else 17766)
        ranks={'original':np.asarray(valid.a['original_rank']).copy()};diagnostics={}
        for key,(model,norm,arm) in models.items():
            ranks[key],diagnostics[key]=evaluate(valid,model,norm,arm)
            np.savez_compressed(OUT/domain/'ranks.npz',**ranks)
        summary[domain]=dict(train=train.info,validation=valid.info,fits=fitinfo,
                             diagnostics=diagnostics,results=summarize(valid,ranks))
        write(OUT/domain/'summary.json',summary[domain]);write(OUT/'summary.json',summary)
        write(OUT/'manifest.json',dict(sha256=manifest))
        status('domain_complete',domain=domain,primary=summary[domain]['results']['primary']['seed_mean_N10'])
        del train,valid,engine,models
    write(OUT/'status.json',dict(state='COMPLETE',pid=os.getpid(),elapsed_seconds=time.monotonic()-START,test_read=False))


if __name__=='__main__':
    try:main()
    except BaseException as e:
        if OUT.exists():write(OUT/'status.json',dict(state='FAILED',error=repr(e),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-START))
        raise
