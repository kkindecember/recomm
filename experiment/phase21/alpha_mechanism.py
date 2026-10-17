"""Bounded alpha exploration, fixed equal SASRec/BGE, independent of expert screen."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
import torch

from experiment.phase21 import anchored_alpha as OLD
from experiment.phase21 import alpha_mechanism_core as C
from experiment.phase21 import joint_fusion_core as J
from experiment.phase21 import qwen4_alpha_rank_core as A
from experiment.phase21 import old10_expert_selection as B
from experiment.phase21 import fixed_expert_screen_4096 as FIX
from experiment.phase21.candidate_gate_core import fold
from experiment.phase20 import lime_clean_benefit as CLEAN
from experiment.phase20.catalog_scope_core import metrics, contrast

ROOT = B.ROOT
OUT = ROOT/'artifacts/phase21/alpha_mechanism_20261017_v1'
PLAN = ROOT/'plan/第二十一阶段/21.50_SASRec_BGE干预强度的锚定位置与历史证据_20261017.md'
REPORT = ROOT/'report/第二十一阶段/SASRec_BGE干预强度的锚定位置与历史证据_20261017.md'
MODULE = 'experiment.phase21.alpha_mechanism'
DOMAINS = ('Toys', 'Beauty')
LEARNED = dict(global_nll=(True,False,0.), user_nll=(False,False,0.),
    user_kl001=(False,False,.001),user_kl01=(False,False,.01),
    user_kl1=(False,False,.1),evidence_kl01=(False,True,.01))
ARMS = ('fixed03', *list(LEARNED)[:-1], 'mass_raw', 'mass_mean', 'mass_matched', 'evidence_kl01')
CFG = dict(seed=0, epochs=20, checkpoints=[0,4,8,12,20], batch_size=32, lr=.002,
    weight_decay=.001, score_scale=20., alpha_initial=.3, beam_size=50,length_penalty=1.,
    expert_pair=['SASRec','BGE'],beta=[.5,.5],beta_training=False,
    arms=list(ARMS),learned={k:list(v) for k,v in LEARNED.items()},lambda_grid=[.001,.01,.1],
    evaluation_users=512,input_validation_slice=[512,1024],history_filter=False,
    history_penalty=False,text_self_exclusion=False,test_read=False,
    expert_training=False,gram_optimizer_steps=0,
    total_lease_mib=16384,expected_workload_peak_mib=12288,minimum_free_mib=18432,
    allocation_limit_gib=11,max_parallel=2,worker_timeout_seconds=14400,
    resource_wait_seconds=86400,scientific_auto_retry=False,
    stop_after='ten arms x fixed512 per domain; no automatic next experiment')
read, sha, stamp = B.read, B.sha, B.stamp
write = FIX.write


def progress(domain, stage, **extra):
    value = dict(state='RUNNING',stage=stage,pid=os.getpid(),updated_at=stamp(),
        gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),test_read=False,**extra)
    write(OUT/domain/'status.json',value)
    print(json.dumps(dict(domain=domain,**value)),flush=True)


def prepare():
    checks = C.tests()
    OUT.mkdir(exist_ok=False)
    write(OUT/'status.json',dict(state='PREPARING',updated_at=stamp(),test_read=False))
    write(OUT/'config.json',CFG);write(OUT/'checks.json',checks)
    shutil.copy2(PLAN,OUT/'protocol.md')
    manifest,domain_inputs = {},{}
    def use(path, expected=None):
        rel=str(path.relative_to(ROOT));digest=manifest.get(rel) or sha(path)
        assert expected is None or digest==expected,rel
        manifest[rel]=digest;return rel
    prior=read(OLD.OUT/'manifest.json')['sha256']
    common=[]
    # Snapshot the current corrected implementation rather than requiring the
    # failed historical driver's source digest to match a later logging fix.
    for rel in prior:
        if rel.startswith(('experiment/','GRAM/src/','.cache/huggingface/')):
            common.append(use(ROOT/rel))
    for path in (Path(__file__),ROOT/'experiment/phase21/alpha_mechanism_core.py',
                 ROOT/'experiment/phase21/fixed_expert_screen_4096.py',
                 ROOT/'experiment/gpu_memory_lease.py',PLAN,OUT/'config.json'):
        rel=use(path)
        if rel not in common:common.append(rel)
    for d in DOMAINS:
        p=OUT/d;p.mkdir();files=[]
        for name in ('catalog.json','data_config.json','fit_samples.json','teacher.npz',
            'teacher_checks.json','pool.npz','parent_full.pt','parent_fold.pt',
            'train_user.npy','dev_user.npy','dev_experts.npy','samples.json','references.json'):
            src=OLD.OUT/d/name
            files.append(use(src,prior.get(str(src.relative_to(ROOT)))))
            shutil.copy2(src,p/name);files.append(use(p/name))
        fit=read(p/'fit_samples.json');dev=read(p/'samples.json')
        assert len(dev)==512 and len({s['user_id'] for s in dev})==512
        assert {s['user_id'] for s in dev}.isdisjoint(s['user_id'] for s in fit)
        assert np.load(p/'teacher.npz')['users'].tolist()==[s['user_id'] for s in fit]
        assert sha(p/'teacher.npz')==read(p/'teacher_checks.json')['sha256']
        train_meta=read(OLD.BASE/d/'train/users.json')
        assert [s['user_id'] for s in train_meta]==[s['user_id'] for s in fit]
        for name in ('item.npy','user.npy','users.json','metadata.json'):
            files.append(use(OLD.BASE/d/'train'/name))
        cfg=read(p/'data_config.json')
        files += [use(ROOT/cfg['parent_checkpoint'],cfg['parent_sha256']),use(ROOT/cfg['embedding'])]
        full=torch.load(p/'parent_full.pt',map_location='cpu')
        internal=torch.load(p/'parent_fold.pt',map_location='cpu')
        assert full['fit_users']==[s['user_id'] for s in fit]
        assert internal['fit_users']==[s['user_id'] for s in fit if fold(s['user_id'],d)!=0]
        refs=read(p/'references.json')
        assert all(r['user_id']==s['user_id'] and r['target']==s['target'] for r,s in zip(refs,dev))
        domain_inputs[d]=common+files
        write(p/'status.json',dict(state='PREPARED',fit_users=len(fit),evaluation_users=len(dev),updated_at=stamp()))
    for rel in manifest:
        path=ROOT/rel
        if path.suffix in ('.py','.sh','.md'):
            dest=OUT/'sources'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    write(OUT/'manifest.json',dict(sha256=manifest,domain_inputs=domain_inputs))
    write(OUT/'status.json',dict(state='PREPARED',updated_at=stamp(),test_read=False))
    REPORT.write_text('# SASRec+BGE干预强度探索\n\n状态：PREPARED，尚无本轮真实生成结论。\n')
    print(json.dumps(dict(state='PREPARED',checks=checks,files=len(manifest))),flush=True)


def cache(domain):
    # Reuse the batched frozen-GRAM path teacher, with the corrected flexible
    # progress writer. Missing competitor paths require frozen inference only.
    OLD.OUT=OUT;OLD.progress=progress
    OLD.cache(domain)


def hidden(parent, user, device):
    x=J.normalize(torch.as_tensor(np.asarray(user),device=device),
        *[torch.as_tensor(parent[k][:10],device=device) for k in ('mean','std')])
    sd=parent['state_dict']
    with torch.no_grad():
        return torch.relu(torch.nn.functional.linear(x,sd['input.weight'][:,:10].to(device),
            sd['input.bias'].to(device)))


def cache_expert_evidence(domain,tree,items,device):
    p=OUT/domain;train=read(p/'fit_samples.json');dev=read(p/'samples.json')
    cfg=read(p/'data_config.json');emb=np.load(ROOT/cfg['embedding'])
    assert emb['item_ids'].tolist()==read(p/'catalog.json')['items'][1:]
    embeddings=torch.as_tensor(emb['embeddings'],device=device)
    sources={'train':np.load(OLD.BASE/domain/'train/item.npy',mmap_mode='r'),
             'dev':np.load(p/'dev_experts.npy',mmap_mode='r')}
    q=torch.zeros((*items.shape,tree.valid.shape[1]),device=device)
    for begin in range(0,len(train),32):
        end=min(begin+32,len(train))
        e=sources['train'][begin:end,:,:2]
        signal=torch.as_tensor(.5*e[:,:,0]+.5*e[:,:,1],device=device)
        q[begin:end],_=A.pool_logs(tree,signal,items[begin:end])
    ev=np.empty((*q.shape,3),np.float32);maximum_error=0.
    dev_nodes=np.lib.format.open_memmap(p/'dev_prefix_evidence.npy',mode='w+',dtype='float32',
        shape=(len(dev),len(tree.index.nodes),3))
    for part,samples in (('train',train),('dev',dev)):
        for j,s in enumerate(samples):
            with torch.no_grad():features,raw=C.evidence(embeddings,s['full_history'])
            rr=raw.cpu().numpy();norm=(rr-rr.min())/max(float(rr.max()-rr.min()),1e-8)
            expected=np.asarray(sources[part][j,:,1])
            err=float(np.max(np.abs(norm-expected)));maximum_error=max(maximum_error,err)
            np.testing.assert_allclose(norm,expected,atol=8e-6,rtol=2e-5)
            e=sources[part][j,:,:2];signal=.5*e[:,0]+.5*e[:,1]
            values=C.prefix_evidence(tree.index,features.cpu().numpy(),signal)
            if part=='train':
                ancestors=tree.ancestors[:-1,items[j]].T.cpu().numpy()
                ev[j]=values[ancestors]
            else:dev_nodes[j]=values
            if j==0 or (j+1)%128==0 or j+1==len(samples):
                progress(domain,'bge_evidence_and_equal_expert_mass',part=part,users_completed=j+1,
                    users_total=len(samples),max_reconstruction_error=maximum_error)
    dev_nodes.flush();np.save(p/'train_prefix_evidence.npy',ev)
    np.save(p/'expert_path_logs.npy',q.cpu().numpy())
    write(p/'evidence_checks.json',dict(state='PASSED',max_cached_bge_reconstruction_error=maximum_error,
        bge_full_catalog_scores_reconstructed=True,sasrec_attribution=False,
        evidence='positive signed-contribution shares; top1/top3/weighted relative position',
        target_argument_used=False,prefix_aggregation='fixed-expert leaf-mass conditional expectation'))
    return q,torch.as_tensor(ev,device=device)


def mass_calibration(tree,items,valid,legal):
    count=torch.as_tensor([len(c) for c in tree.index.children],device=items.device)
    branch=(count[tree.ancestors[:-1,items].permute(1,2,0)]>1)&valid
    # Deduplicate prefixes per user so repeated pool paths do not reweight m.
    values=[]
    ancestors=tree.ancestors[:-1,items].permute(1,2,0).cpu().numpy()
    ll=legal.cpu().numpy();bm=branch.cpu().numpy()
    for j in range(len(items)):
        _,ix=np.unique(ancestors[j][bm[j]],return_index=True)
        values.extend(ll[j][bm[j]][ix].tolist())
    masses=np.exp(np.asarray(values,dtype=np.float64)).clip(0,1);deficit=1-masses
    lo,hi=0.,1.
    while np.minimum(.99,hi*deficit).mean()<.3:
        hi*=2;assert hi<1e15,'Cannot match strength: too few non-unit masses'
    for _ in range(70):
        mid=(lo+hi)/2
        if np.minimum(.99,mid*deficit).mean()<.3:lo=mid
        else:hi=mid
    return dict(raw_mean=float(.3*deficit.mean()),matched_scale=float((lo+hi)/2),
        matched_training_mean=float(np.minimum(.99,hi*deficit).mean()),
        branching_prefixes=len(values),m_quantiles=np.quantile(masses,[0,.1,.5,.9,1]).tolist(),
        calibration_distribution='unique fit-user pool branching prefixes; no dev/test'),branch


def fit(domain):
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(11*2**30/torch.cuda.get_device_properties(0).total_memory)
    device=torch.device('cuda:0');p=OUT/domain
    samples=read(p/'fit_samples.json');pool=np.load(p/'pool.npz')
    items=torch.as_tensor(pool['ids']-1,device=device);cmask=torch.as_tensor(pool['valid'],device=device)
    teacher=np.load(p/'path_teacher.npz')
    assert sha(p/'path_teacher.npz')==read(p/'path_teacher_checks.json')['sha256']
    native,legal=[torch.as_tensor(teacher[k],device=device) for k in ('native','legal')]
    tree=J.PathMass(read(p/'catalog.json')['decoder_paths'],device);valid=tree.valid[items]
    q,ev=cache_expert_evidence(domain,tree,items,device)
    calibration,branch=mass_calibration(tree,items,valid&cmask[:,:,None],legal)
    write(p/'mass_calibration.json',calibration)
    native_scores=C.path_scores(native,legal,q,torch.zeros(len(items),device=device),valid)
    fi=np.asarray([i for i,s in enumerate(samples) if fold(s['user_id'],domain)!=0])
    hi=np.asarray([i for i,s in enumerate(samples) if fold(s['user_id'],domain)==0])
    users=np.load(p/'train_user.npy');dev=np.load(p/'dev_user.npy')
    parents={name:torch.load(p/('parent_'+name+'.pt'),map_location='cpu') for name in ('fold','full')}
    hs={name:hidden(parent,users,device) for name,parent in parents.items()}
    dev_hidden=hidden(parents['full'],dev,device)
    norm={}
    for name,ix in (('fold',fi),('full',np.arange(len(items)))):
        selected=ev[ix][(valid&cmask[:,:,None])[ix]]
        mean=selected.mean(0);std=selected.std(0,unbiased=False).clamp_min(.01)
        norm[name]=(mean,std)
    sensitivity=[]
    for aa in (.0,.1,.3,.45,.6,.75,.9,.99):
        scores=C.path_scores(native,legal,q,torch.full((len(items),),aa,device=device),valid)
        sensitivity.append(dict(alpha=aa,target_nll=-float(scores[:,0].mean()),
            pool_kl=float(C.pool_kl(native_scores,scores,cmask).mean())))
    diagnostic=[]
    for d in range(valid.shape[-1]):
        mask=branch[:,0,d]
        if bool(mask.any()):
            m=legal[:,0,d][mask].exp()
            advantage=(q[:,0,d]-(native[:,0,d]-legal[:,0,d]))[mask]
            diagnostic.append(dict(depth=d,branching_targets=int(mask.sum()),
                mass_quantiles=torch.quantile(m,torch.tensor([.1,.5,.9],device=device)).cpu().tolist(),
                expert_conditional_target_better_fraction=float((advantage>0).float().mean())))
    write(p/'training_signal_audit.json',dict(fixed_alpha_grid=sensitivity,target_branch_depths=diagnostic,
        inference_performed=False,note='Teacher-forced training paths; not beam search or dev quality.'))
    info={};alpha={'fixed03':np.full(len(dev),.3),'mass_mean':np.full(len(dev),calibration['raw_mean'])}
    for arm,(global_only,use_ev,lam) in LEARNED.items():
        def run(parent_name,ix,epochs,select):
            torch.manual_seed(0);model=C.Head(global_only,use_ev).to(device)
            opt=torch.optim.AdamW(model.parameters(),lr=CFG['lr'],weight_decay=CFG['weight_decay'])
            rng=np.random.default_rng(0);curve=[]
            en=((ev-norm[parent_name][0])/norm[parent_name][1]).clamp(-10,10) if use_ev else None
            def loss(rows):
                aa=model(hs[parent_name][rows],en[rows] if use_ev else None)
                score=C.path_scores(native[rows],legal[rows],q[rows],aa,valid[rows])
                nll=-score[:,0].mean()
                kl=C.pool_kl(native_scores[rows],score,cmask[rows]).mean()
                return nll+lam*kl,nll,kl
            for epoch in range(epochs+1):
                if epoch:
                    order=rng.permutation(ix)
                    for begin in range(0,len(order),32):
                        opt.zero_grad();value,_,_=loss(order[begin:begin+32])
                        assert torch.isfinite(value);value.backward()
                        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                        opt.step()
                if select and epoch in CFG['checkpoints']:
                    total=np.zeros(3)
                    with torch.no_grad():
                        for begin in range(0,len(hi),32):
                            rows=hi[begin:begin+32]
                            total+=np.asarray([float(x) for x in loss(rows)])*len(rows)
                    total/=len(hi);curve.append(dict(epoch=epoch,loss=total[0],nll=total[1],kl=total[2]))
                if epoch in CFG['checkpoints']:
                    progress(domain,'fit_alpha',arm=arm,phase='selection' if select else 'refit',epoch=epoch)
            return model,curve
        selection,curve=run('fold',fi,20,True)
        epoch=min(curve,key=lambda r:(r['loss'],r['epoch']))['epoch'];del selection
        model,_=run('full',np.arange(len(items)),epoch,False)
        sd={k:v.detach().cpu() for k,v in model.state_dict().items()}
        if not use_ev:
            with torch.no_grad():alpha[arm]=model(dev_hidden).cpu().double().numpy()
        torch.save(dict(state_dict=sd,global_only=global_only,use_evidence=use_ev,
            evidence_mean=norm['full'][0].cpu(),evidence_std=norm['full'][1].cpu(),
            fit_users=[s['user_id'] for s in samples],selected_epochs=epoch,lambda_value=lam),p/(arm+'.pt'))
        info[arm]=dict(selected_epochs=epoch,selection_curve=curve,lambda_value=lam,
            trainable_parameters=sum(p.numel() for p in model.parameters()),
            evidence_coefficients=sd['weight'][32:].tolist() if use_ev else None,
            alpha_quantiles=np.quantile(alpha[arm],[0,.1,.5,.9,1]).tolist() if not use_ev else None)
    np.savez(p/'alphas.npz',**alpha);np.save(p/'dev_hidden.npy',dev_hidden.cpu().numpy())
    e=np.load(p/'dev_experts.npy');np.save(p/'signals.npy',.5*e[:,:,0]+.5*e[:,:,1])
    write(p/'training.json',dict(fits=info,internal_fit_users=len(fi),internal_heldout_users=len(hi),
        full_fit_users=len(samples),beta_fixed=[.5,.5],frozen_hidden_parameters_unchanged=True,
        evaluation_used_for_selection=False,test_read=False))
    files=[f for f in p.iterdir() if f.suffix in ('.pt','.npy','.npz','.json') and f.name not in
        ('status.json','lease_status.json','lease_process.json','training_manifest.json')]
    write(p/'training_manifest.json',dict(sha256={str(f.relative_to(ROOT)):sha(f) for f in files}))


class Generation(CLEAN.Worker):
    def __init__(self,domain,lease_pid):
        self.lease=B.S.LeaseHandle(lease_pid);self.row_seconds=[];CLEAN.OUT=OUT
        super().__init__(domain)
        self.truth=self.samples
        from experiment.phase18.core.diff_data import DiffDataset
        self.samples=[dict(s,target=1) for s in self.truth];self.data=DiffDataset(self.samples,self.catalog)
        self.refs=read(self.folder/'references.json');self.alpha=dict(np.load(self.folder/'alphas.npz'))
        self.signals=np.load(self.folder/'signals.npy');self.mass=read(self.folder/'mass_calibration.json')
        self.ev=np.load(self.folder/'dev_prefix_evidence.npy',mmap_mode='r')
        saved=torch.load(self.folder/'evidence_kl01.pt',map_location='cpu')
        self.en_mean=saved['evidence_mean'].numpy();self.en_std=saved['evidence_std'].numpy()
        self.ew=saved['state_dict']['weight'].numpy();self.eb=float(saved['state_dict']['bias'])
        self.hidden=np.load(self.folder/'dev_hidden.npy')
        self.stats={arm:[] for arm in ARMS}

    def context(self,j,override_index=None):return super().context(j,override_index=j)

    def status(self,state='RUNNING',**extra):
        assert self.lease.poll() is None,'GPU lease exited'
        done=getattr(self,'users_completed',0)
        rate=float(np.median(self.row_seconds[-32:])) if self.row_seconds else None
        write(self.folder/'status.json',dict(state=state,stage=self.stage,pid=os.getpid(),updated_at=stamp(),
            gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),users_completed=done,users_total=512,
            completed_decodes=self.completed,elapsed_seconds=time.monotonic()-self.started,
            median_seconds_per_user=rate,remaining_generation_seconds_estimate=rate*(512-done) if rate else None,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,test_read=False,**extra))

    def policy(self,j,arm):
        if arm in self.alpha:return lambda p,m:float(self.alpha[arm][j])
        if arm=='mass_raw':return lambda p,m:.3*max(0.,1-math.exp(min(0.,m)))
        if arm=='mass_matched':return lambda p,m:min(.99,self.mass['matched_scale']*max(0.,1-math.exp(min(0.,m))))
        assert arm=='evidence_kl01'
        base=float(self.hidden[j]@self.ew[:32]+self.eb)
        def value(prefix,m):
            features=np.clip((self.ev[j,self.index.lookup[prefix]]-self.en_mean)/self.en_std,-10,10)
            z=float(base+features@self.ew[32:])
            return float(np.clip(1/(1+np.exp(-np.clip(z,-60,60))),1e-6,1-1e-6))
        return value

    def variable_decode(self,ctx,q,policy):
        from transformers import LogitsProcessorList
        from model.gram import GRAM
        from model.gram_t5_outputs import BaseModelOutputWithPastAndCrossAttentions
        guide=C.PrefixGuide(self.index,q,policy)
        with torch.no_grad():
            output=super(GRAM,self.model).generate(input_ids=ctx['ids'].reshape(1,-1),
                attention_mask=ctx['mask'].reshape(1,-1),
                encoder_outputs=BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=ctx['hidden']),
                max_length=self.max_length,prefix_allowed_tokens_fn=self.prefix_fn,
                num_beams=50,num_return_sequences=50,return_dict_in_generate=True,output_scores=True,
                length_penalty=1.,logits_processor=LogitsProcessorList([guide]))
        paths=[p[:p.index(1)+1] for p in output.sequences.cpu().tolist()]
        ids=[self.index.path_to_item[tuple(p)] for p in paths]
        scores=output.sequences_scores.detach().cpu().double().numpy()
        assert len(set(ids))==50 and np.isfinite(scores).all()
        np.testing.assert_allclose(guide.rescore(paths),scores,atol=2e-6,rtol=1e-5)
        self.completed+=1
        branches=[prefix for prefix in guide.trace if len(self.index.children[self.index.lookup[prefix]])>1]
        aa=np.asarray([guide.alpha_trace[p] for p in branches]);mm=np.asarray([math.exp(guide.trace[p][2]) for p in branches])
        summary=dict(branches=len(branches),alpha_mean=float(aa.mean()),
            alpha_quantiles=np.quantile(aa,[0,.1,.5,.9,1]).tolist(),m_mean=float(mm.mean()),
            mass_conservation_log_error=guide.max_mass_error)
        return dict(ids=ids,scores=scores.tolist()),summary

    def run(self):
        self.stage='native_zero_dynamic_smoke';self.status();checks=[]
        assert self.initial_digest==read(self.folder/'teacher_checks.json')['model_digest']
        for j in (0,256,511):
            ctx=self.context(j);native=self.decode(ctx);ref=self.refs[j]['native']
            B.Generation.equal(dict(ids=native['ids'],scores=native['scores'].tolist()),ref)
            q=self.index.conditional_logs(self.signals[j])
            zero,_=self.variable_decode(ctx,q,lambda p,m:0.)
            B.Generation.equal(zero,ref)
            original=self.decode(ctx,q,.3)
            dynamic,_=self.variable_decode(ctx,q,lambda p,m:.3)
            B.Generation.equal(dynamic,dict(ids=original['ids'],scores=original['scores'].tolist()))
            # Dynamic policy must satisfy fresh-beam scoring invariants too.
            self.variable_decode(ctx,q,self.policy(j,'mass_raw'))
            self.variable_decode(ctx,q,self.policy(j,'evidence_kl01'))
            self.samples[j]['target']=2;changed=self.context(j);self.samples[j]['target']=1
            torch.testing.assert_close(ctx['ids'],changed['ids'],atol=0,rtol=0)
            torch.testing.assert_close(ctx['mask'],changed['mask'],atol=0,rtol=0)
            checks.append(dict(row=j,native_reproduced=True,zero_exact=True,constant_matches_original=True,
                target_placeholder_invariant=True,dynamic_path_rescore_and_mass_checks=True))
        write(self.folder/'smoke.json',dict(state='PASSED',checks=checks))
        self.stage='ten_arm_generation';self.status()
        with (self.folder/'predictions.jsonl').open('x') as stream:
            for j,s in enumerate(self.truth):
                began=time.monotonic();ctx=self.context(j);q=self.index.conditional_logs(self.signals[j])
                row=dict(user_id=s['user_id'],target=s['target'],history=s['history'],native=self.refs[j]['native'])
                detail={}
                for arm in ARMS:
                    row[arm],detail[arm]=self.variable_decode(ctx,q,self.policy(j,arm))
                    self.stats[arm].append(detail[arm])
                row['guidance_statistics']=detail;stream.write(json.dumps(row)+'\n');stream.flush()
                self.users_completed=j+1;self.row_seconds.append(time.monotonic()-began)
                if j==0 or (j+1)%16==0 or j+1==len(self.truth):self.status()
        assert self.digest()==self.initial_digest
        write(self.folder/'guidance_statistics.json',self.stats)
        write(self.folder/'runtime_checks.json',dict(state='PASSED',gram_model_digest_unchanged=True,
            gram_optimizer_steps=0,beta_fixed=[.5,.5],history_filter=False,all_users_and_arms=True,
            dynamic_path_scores_verified=True,legal_mass_preserved=True))
        self.stage='complete';self.status('COMPLETED')


def summarize():
    result={};lines=['# SASRec+BGE干预强度的锚定、位置与历史证据','','## Material Passport','',
        '- Origin Skill: academic-research-suite / experiment-agent，内联执行','- Origin Mode: validate',
        '- Origin Date: 2026-10-17','- Verification Status: ANALYZED；开发集、单种子、真实生成',
        '- Version Label: alpha_mechanism_20261017_v1','',
        '固定等权SASRec+BGE，只改α。所有十组保留；不读取test，不自动确定最终方法。','']
    for d in DOMAINS:
        p=OUT/d;samples=read(p/'samples.json');keys=('native',*ARMS);ranks={k:[] for k in keys}
        with (p/'predictions.jsonl').open() as stream:
            for j,line in enumerate(stream):
                row=json.loads(line);s=samples[j]
                assert all(row[k]==s[k] for k in ('user_id','target','history'))
                for k in keys:ranks[k].append(B.rank(row[k],s['target']))
        assert j+1==len(samples)==512
        ranks={k:np.asarray(v) for k,v in ranks.items()}
        measured={k:metrics(v) for k,v in ranks.items()}
        pairs=[(a,b) for a in ARMS for b in ('native','fixed03') if a!=b]
        pairs += [('mass_raw','mass_mean'),('evidence_kl01','user_kl01'),('mass_matched','fixed03')]
        comparisons={a+'_vs_'+b:contrast(ranks[a],ranks[b],seed=20261017) for a,b in pairs}
        result[d]=dict(users=512,metrics=measured,comparisons=comparisons,
            training=read(p/'training.json'),mass_calibration=read(p/'mass_calibration.json'))
        np.savez_compressed(p/'ranks.npz',**ranks);write(p/'summary.json',result[d])
        lines += [f'## {d}','','| α方案 | NDCG@10 | 对GRAM相对变化 | Hit@10 |','|---|---:|---:|---:|']
        for k in keys:
            m=measured[k];rel=100*(m['N10']/measured['native']['N10']-1)
            lines.append(f'| {k} | {m["N10"]:.6f} | {rel:+.2f}% | {m["H10"]:.6f} |')
        files=[p/n for n in ('predictions.jsonl','ranks.npz','summary.json','smoke.json','runtime_checks.json')]
        write(p/'output_manifest.json',dict(sha256={str(f.relative_to(ROOT)):sha(f) for f in files}))
    review=dict(coverage='11/11',simpson='Domains separated.',ecological='Per-user heterogeneity retained.',
        berkson='Restricted processed validation cohort.',collider='No outcome-based selection.',
        base_rate='All512 prescribed users kept.',regression_to_mean='Reused development sample, no independent confirmation.',
        survivorship='All ten arms required.',look_elsewhere='Unadjusted descriptive intervals for all prescribed contrasts.',
        forking_paths='Frozen arms, exploratory method selection, seed and test confirmation deferred.',
        causation='Matched equal beta; evidence adds three weights; no universal causal claim.',
        reverse_causality='No targets as inference features; supervised labels only for fitting and metrics.')
    write(OUT/'summary.json',dict(domains=result,statistical_review=review,test_read=False,automatic_final_selection=False))
    lines += ['', '配对区间为固定训练模型下的描述性bootstrap，未做多重比较校正；不包括训练种子和选择不确定性。',
        '候选池KL及目标路径NLL是代理；m保留规则不修复父节点的合法总质量塌陷。',
        '证据只分解BGE正支持，SASRec不做线性历史归因。质量提升与机制解释分开判断。']
    REPORT.write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('test','prepare','supervise','worker','cache','fit','generate'))
    parser.add_argument('--domain',choices=DOMAINS);parser.add_argument('--lease-pid',type=int);args=parser.parse_args()
    try:
        if args.mode=='test':print(json.dumps(C.tests()),flush=True)
        elif args.mode=='prepare':prepare()
        elif args.mode=='supervise':
            B.OUT,B.CFG,B.MODULE,B.REPORT,B.summarize=OUT,CFG,MODULE,REPORT,summarize
            B.supervise()
        elif args.mode=='worker':
            OLD.OUT,OLD.CFG,OLD.MODULE,OLD.progress=OUT,CFG,MODULE,progress
            return OLD.worker(args.domain)
        elif args.mode=='cache':cache(args.domain)
        elif args.mode=='fit':fit(args.domain)
        else:Generation(args.domain,args.lease_pid).run()
        return 0
    except BaseException as exc:
        if OUT.exists():
            p=OUT/args.domain if args.domain else OUT
            write(p/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),updated_at=stamp()))
            write(p/'status.json',dict(state='FAILED',error=repr(exc),updated_at=stamp(),test_read=False))
        raise


if __name__=='__main__':sys.exit(main())
