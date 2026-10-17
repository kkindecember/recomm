"""One-parameter global alpha with the same KL loss as the user-alpha arm."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
import torch

from experiment.phase21 import alpha_mechanism as M
from experiment.phase21 import alpha_mechanism_core as C
from experiment.phase21 import old10_expert_selection as B
from experiment.phase21 import anchored_alpha as OLD
from experiment.phase21.candidate_gate_core import fold
from experiment.phase21.review_alpha_mechanism import independent_rank, utility, decomposition

ROOT=M.ROOT
SOURCE=M.OUT
OUT=ROOT/'artifacts/phase21/global_kl_control_20261017_v1'
PLAN=ROOT/'plan/第二十一阶段/21.51_全局alpha同KL损失的匹配对照_20261017.md'
REPORT=ROOT/'report/第二十一阶段/全局alpha同KL损失匹配对照_20261017.md'
MODULE='experiment.phase21.global_kl_control'
DOMAINS=M.DOMAINS
ARM='global_kl1'
REFERENCES=('native','fixed03','global_nll','user_nll','user_kl1')
CFG=dict(M.CFG,arms=[ARM],learned={ARM:[True,False,.1]},lambda_grid=[.1],
    trainable_parameters=1,training_backend='CPU float32; historical global_nll replay guard',
    baseline_replay_tolerance=2e-5,worker_timeout_seconds=1800,
    stop_after='one additional arm x same512 in each domain; no automatic expansion')
read,write,sha,stamp=M.read,M.write,M.sha,M.stamp


def progress(domain,stage,**extra):
    value=dict(state='RUNNING',stage=stage,pid=os.getpid(),updated_at=stamp(),test_read=False,**extra)
    write(OUT/domain/'status.json',value)
    print(json.dumps(dict(domain=domain,**value)),flush=True)


def tensors(domain):
    p=OUT/domain;samples=read(p/'fit_samples.json');pool=np.load(p/'pool.npz')
    targets=np.asarray([s['target']-1 for s in samples])
    np.testing.assert_array_equal(np.load(p/'teacher.npz')['targets'],targets)
    np.testing.assert_array_equal(pool['ids'][:,0]-1,targets)
    assert pool['users'].tolist()==[s['user_id'] for s in samples]
    cache=np.load(p/'path_teacher.npz')
    native,legal=[torch.from_numpy(cache[k]) for k in ('native','legal')]
    q=torch.from_numpy(np.load(p/'expert_path_logs.npy'))
    paths=read(p/'catalog.json')['decoder_paths']
    lengths=np.asarray([len(path)-1 for path in paths])
    valid=torch.from_numpy(np.arange(native.shape[-1])[None,None,:]<lengths[pool['ids']-1][:,:,None])
    cmask=torch.from_numpy(pool['valid'])
    assert native.shape==legal.shape==q.shape==valid.shape
    assert bool(cmask[:,0].all())
    native_scores=C.path_scores(native,legal,q,torch.zeros(len(samples)),valid)
    assert float(C.pool_kl(native_scores,native_scores,cmask).abs().max())==0.
    baseline=C.path_scores(native,legal,q,torch.full((len(samples),),.3),valid)
    expected=next(r['target_nll'] for r in read(p/'training_signal_audit.json')['fixed_alpha_grid'] if r['alpha']==.3)
    assert abs(-float(baseline[:,0].mean())-expected)<2e-5
    return samples,native,legal,q,valid,cmask,native_scores


def tests():
    checks=C.tests()
    head=C.Head(global_only=True);x=torch.randn(4,32)
    torch.testing.assert_close(head(x),head(x*100),rtol=0,atol=0)
    assert sum(p.numel() for p in head.parameters())==1
    checks.update(global_head_input_independent=True,global_parameters=1)
    return checks


def prepare():
    assert read(SOURCE/'status.json')['state']=='COMPLETED'
    checks=tests();OUT.mkdir(exist_ok=False)
    write(OUT/'config.json',CFG);write(OUT/'checks.json',checks);shutil.copy2(PLAN,OUT/'protocol.md')
    manifest,domain_inputs={},{}
    def use(path,expected=None):
        rel=str(path.relative_to(ROOT));digest=manifest.get(rel) or sha(path)
        assert expected is None or digest==expected,rel
        manifest[rel]=digest;return rel
    common=[]
    for rel,digest in read(SOURCE/'manifest.json')['sha256'].items():
        if rel.startswith(('experiment/','GRAM/src/','.cache/huggingface/')):
            common.append(use(ROOT/rel,digest))
    for f in (Path(__file__),ROOT/'experiment/phase21/review_alpha_mechanism.py',
              ROOT/'experiment/phase21/run_global_kl_control.sh',PLAN,OUT/'config.json'):
        common.append(use(f))
    for d in DOMAINS:
        p=OUT/d;p.mkdir();files=[]
        prior=read(SOURCE/d/'training_manifest.json')['sha256']
        for name in ('catalog.json','data_config.json','fit_samples.json','samples.json','teacher.npz',
            'teacher_checks.json','pool.npz','path_teacher.npz','path_teacher_checks.json','expert_path_logs.npy',
            'signals.npy','alphas.npz','global_nll.pt','user_kl1.pt','training.json','training_signal_audit.json'):
            src=SOURCE/d/name;files.append(use(src,prior[str(src.relative_to(ROOT))]))
            dest=p/({'training.json':'source_training.json','alphas.npz':'source_alphas.npz'}.get(name,name))
            shutil.copy2(src,dest);files.append(use(dest))
        output_manifest=read(SOURCE/d/'output_manifest.json')['sha256']
        source_predictions=SOURCE/d/'predictions.jsonl'
        files.append(use(source_predictions,output_manifest[str(source_predictions.relative_to(ROOT))]))
        refs=[];samples=read(p/'samples.json');fit=read(p/'fit_samples.json')
        for j,line in enumerate(source_predictions.read_text().splitlines()):
            row=json.loads(line);s=samples[j]
            assert all(row[k]==s[k] for k in ('user_id','target','history'))
            refs.append({k:row[k] for k in ('user_id','target','history',*REFERENCES)})
        assert len(refs)==len(samples)==len({s['user_id'] for s in samples})==512
        assert {s['user_id'] for s in samples}.isdisjoint(s['user_id'] for s in fit)
        for model_name in ('global_nll','user_kl1'):
            model=torch.load(p/(model_name+'.pt'),map_location='cpu')
            assert model['fit_users']==[s['user_id'] for s in fit]
        write(p/'references.json',refs);files.append(use(p/'references.json'))
        cfg=read(p/'data_config.json');files.append(use(ROOT/cfg['parent_checkpoint'],cfg['parent_sha256']))
        domain_inputs[d]=common+files
        tensors(d)
        write(p/'status.json',dict(state='PREPARED',updated_at=stamp(),fit_users=len(fit),evaluation_users=512))
    for rel in manifest:
        f=ROOT/rel
        if f.suffix in ('.py','.sh','.md'):
            dest=OUT/'sources'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
    write(OUT/'manifest.json',dict(sha256=manifest,domain_inputs=domain_inputs))
    write(OUT/'status.json',dict(state='PREPARED',updated_at=stamp(),test_read=False))
    REPORT.write_text('# 全局α＋同KL损失匹配对照\n\n已准备；尚无新增生成结果。\n')
    print(json.dumps(dict(state='PREPARED',checks=checks,files=len(manifest))),flush=True)


def train(domain):
    p=OUT/domain;assert not (p/'training_manifest.json').exists()
    samples,native,legal,q,valid,cmask,native_scores=tensors(domain)
    n=len(samples);hidden=torch.zeros(n,32)
    fi=np.asarray([j for j,s in enumerate(samples) if fold(s['user_id'],domain)!=0])
    hi=np.asarray([j for j,s in enumerate(samples) if fold(s['user_id'],domain)==0])
    def run(lam,indices,epochs,select,label):
        torch.manual_seed(0);model=C.Head(global_only=True)
        optimizer=torch.optim.AdamW(model.parameters(),lr=CFG['lr'],weight_decay=CFG['weight_decay'])
        rng=np.random.default_rng(0);curve=[]
        def loss(rows):
            aa=model(hidden[rows])
            score=C.path_scores(native[rows],legal[rows],q[rows],aa,valid[rows])
            nll=-score[:,0].mean();kl=C.pool_kl(native_scores[rows],score,cmask[rows]).mean()
            return nll+lam*kl,nll,kl
        for epoch in range(epochs+1):
            if epoch:
                order=rng.permutation(indices)
                for start in range(0,len(order),32):
                    optimizer.zero_grad();value,_,_=loss(order[start:start+32])
                    assert torch.isfinite(value);value.backward()
                    assert all(t.grad is None or torch.isfinite(t.grad).all() for t in model.parameters())
                    optimizer.step()
            if select and epoch in CFG['checkpoints']:
                total=np.zeros(3)
                with torch.no_grad():
                    for start in range(0,len(hi),32):
                        rows=hi[start:start+32];total+=np.asarray([float(x) for x in loss(rows)])*len(rows)
                total/=len(hi);curve.append(dict(epoch=epoch,loss=total[0],nll=total[1],kl=total[2]))
            if epoch in CFG['checkpoints']:
                progress(domain,'fit_global_alpha_cpu',check=label,phase='selection' if select else 'refit',epoch=epoch)
        return model,curve
    def complete_fit(lam,label):
        model,curve=run(lam,fi,20,True,label)
        epoch=min(curve,key=lambda r:(r['loss'],r['epoch']))['epoch'];del model
        model,_=run(lam,np.arange(n),epoch,False,label)
        return model,curve,epoch
    reference=torch.load(p/'global_nll.pt',map_location='cpu')
    replay,replay_curve,replay_epoch=complete_fit(0.,'historical_global_nll_replay')
    expected=read(p/'source_training.json')['fits']['global_nll']
    assert replay_epoch==reference['selected_epochs']==expected['selected_epochs']
    with torch.no_grad():replay_alpha=float(replay(hidden[:1])[0])
    reference_model=C.Head(global_only=True);reference_model.load_state_dict(reference['state_dict'])
    with torch.no_grad():reference_alpha=float(reference_model(hidden[:1])[0])
    delta=abs(replay_alpha-reference_alpha)
    curve_delta=max(abs(a['nll']-b['nll']) for a,b in zip(replay_curve,expected['selection_curve']))
    assert delta<=CFG['baseline_replay_tolerance'] and curve_delta<=CFG['baseline_replay_tolerance']
    write(p/'replay_checks.json',dict(state='PASSED',cpu_alpha=replay_alpha,historical_gpu_alpha=reference_alpha,
        alpha_abs_difference=delta,max_holdout_nll_difference=curve_delta,selected_epoch=replay_epoch,
        same_optimizer_and_folds=True,tolerance=CFG['baseline_replay_tolerance']))
    model,curve,epoch=complete_fit(.1,ARM)
    with torch.no_grad():alpha=float(model(hidden[:1])[0])
    checkpoint=dict(state_dict=model.state_dict(),alpha=alpha,global_only=True,use_evidence=False,
        selected_epochs=epoch,lambda_value=.1,fit_users=[s['user_id'] for s in samples],config=CFG)
    torch.save(checkpoint,p/(ARM+'.pt'))
    np.save(p/'alpha.npy',np.full(512,alpha))
    write(p/'training.json',dict(alpha=alpha,selected_epochs=epoch,selection_curve=curve,lambda_value=.1,
        trainable_parameters=1,internal_fit_users=len(fi),internal_heldout_users=len(hi),full_fit_users=n,
        training_backend='CPU float32',historical_gpu_replay_passed=True,evaluation_used_for_selection=False,
        expert_training_steps=0,gram_optimizer_steps=0,test_read=False))
    files=[p/name for name in (ARM+'.pt','alpha.npy','training.json','replay_checks.json')]
    write(p/'training_manifest.json',dict(sha256={str(f.relative_to(ROOT)):sha(f) for f in files}))
    write(p/'status.json',dict(state='TRAINED_WAITING_FOR_GPU',alpha=alpha,updated_at=stamp(),test_read=False))
    print(json.dumps(dict(domain=domain,state='TRAINED',alpha=alpha,replay_alpha_error=delta)),flush=True)


def verify_training(domain):
    for rel,digest in read(OUT/domain/'training_manifest.json')['sha256'].items():
        assert sha(ROOT/rel)==digest,rel


class Generation(M.CLEAN.Worker):
    def __init__(self,domain,lease_pid):
        self.lease=B.S.LeaseHandle(lease_pid);self.row_seconds=[];M.CLEAN.OUT=OUT
        super().__init__(domain)
        self.truth=self.samples
        from experiment.phase18.core.diff_data import DiffDataset
        self.samples=[dict(s,target=1) for s in self.truth];self.data=DiffDataset(self.samples,self.catalog)
        self.refs=read(self.folder/'references.json');self.signals=np.load(self.folder/'signals.npy')
        verify_training(domain);self.alpha=float(read(self.folder/'training.json')['alpha'])
        self.user_alpha=np.load(self.folder/'source_alphas.npz')['user_kl1']

    def context(self,j,override_index=None):return super().context(j,override_index=j)

    def status(self,state='RUNNING',**extra):
        assert self.lease.poll() is None,'GPU lease exited'
        done=getattr(self,'users_completed',0);rate=float(np.median(self.row_seconds[-32:])) if self.row_seconds else None
        write(self.folder/'status.json',dict(state=state,stage=self.stage,pid=os.getpid(),updated_at=stamp(),
            gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),users_completed=done,users_total=512,
            completed_decodes=self.completed,elapsed_seconds=time.monotonic()-self.started,
            median_seconds_per_user=rate,remaining_generation_seconds_estimate=rate*(512-done) if rate else None,
            alpha=getattr(self,'alpha',None),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            test_read=False,**extra))

    def run(self):
        self.stage='matched_reference_smoke';self.status();checks=[]
        assert self.initial_digest==read(self.folder/'teacher_checks.json')['model_digest']
        def entry(value):return dict(ids=value['ids'],scores=value['scores'].tolist())
        for j in (0,256,511):
            ctx=self.context(j);q=self.index.conditional_logs(self.signals[j]);native=self.decode(ctx)
            B.Generation.equal(entry(native),self.refs[j]['native'])
            zero=self.decode(ctx,q,0.);assert zero['paths']==native['paths']
            np.testing.assert_array_equal(zero['scores'],native['scores'])
            B.Generation.equal(entry(self.decode(ctx,q,.3)),self.refs[j]['fixed03'])
            B.Generation.equal(entry(self.decode(ctx,q,float(self.user_alpha[j]))),self.refs[j]['user_kl1'])
            self.decode(ctx,q,self.alpha)
            self.samples[j]['target']=2;changed=self.context(j);self.samples[j]['target']=1
            torch.testing.assert_close(ctx['ids'],changed['ids'],atol=0,rtol=0)
            torch.testing.assert_close(ctx['mask'],changed['mask'],atol=0,rtol=0)
            checks.append(dict(row=j,native_reproduced=True,zero_exact=True,fixed03_reproduced=True,
                user_kl1_reproduced=True,new_arm_path_rescore_passed=True,target_placeholder_invariant=True))
        write(self.folder/'smoke.json',dict(state='PASSED',checks=checks))
        self.stage='global_kl1_generation';self.status()
        with (self.folder/'predictions.jsonl').open('x') as stream:
            for j,sample in enumerate(self.truth):
                began=time.monotonic();ctx=self.context(j);q=self.index.conditional_logs(self.signals[j])
                value=self.decode(ctx,q,self.alpha);row=dict(self.refs[j]);row[ARM]=entry(value)
                stream.write(json.dumps(row)+'\n');stream.flush()
                self.users_completed=j+1;self.row_seconds.append(time.monotonic()-began)
                if j==0 or (j+1)%16==0 or j+1==512:self.status()
        assert self.digest()==self.initial_digest
        write(self.folder/'runtime_checks.json',dict(state='PASSED',gram_model_digest_unchanged=True,
            global_alpha_constant=True,new_arm_complete=True,reference_predictions_reused=True,
            legal_mass_preserved=True,history_filter=False,history_penalty=False,test_read=False))
        self.stage='complete';self.status('COMPLETED')


def summarize():
    result={};lines=['# 全局α＋同KL损失匹配对照','','## Material Passport','',
        '- Origin Skill: academic-research-suite / experiment-agent，内联执行','- Origin Mode: validate',
        '- Origin Date: 2026-10-17','- Verification Status: VERIFIED_SAVED_OUTPUTS；同批开发集、单种子',
        '- Version Label: global_kl_control_20261017_v1','',
        '新增global_kl1一参数头，与user_kl1三十三参数头使用同一NLL+.1KL损失。SASRec/BGE等权；不读test。',
        'CPU拟合先重放历史GPU global_nll，通过固定阈值数值检查；所有参照预测原样保留，新增组真实生成。','']
    for d in DOMAINS:
        p=OUT/d;samples=read(p/'samples.json');refs=read(p/'references.json')
        ranks={k:[] for k in (*REFERENCES,ARM)}
        for j,line in enumerate((p/'predictions.jsonl').read_text().splitlines()):
            row=json.loads(line);assert all(row[k]==samples[j][k] for k in ('user_id','target','history'))
            assert all(row[k]==refs[j][k] for k in REFERENCES)
            for k in ranks:ranks[k].append(independent_rank(row[k],samples[j]['target']))
        assert j+1==len(samples)==512
        ranks={k:np.asarray(v) for k,v in ranks.items()};measured={k:M.metrics(v) for k,v in ranks.items()}
        for k,rr in ranks.items():
            for cut in (5,10,20,50):assert abs(measured[k]['N'+str(cut)]-float(utility(rr,cut).mean()))<1e-14
        comparisons={ARM+'_vs_'+k:M.contrast(ranks[ARM],ranks[k],seed=20261017) for k in REFERENCES}
        decomp={ARM+'_vs_'+k:decomposition(ranks[ARM],ranks[k]) for k in REFERENCES}
        result[d]=dict(users=512,metrics=measured,comparisons=comparisons,decompositions=decomp,
            training=read(p/'training.json'),replay_checks=read(p/'replay_checks.json'))
        np.savez_compressed(p/'ranks.npz',**ranks);write(p/'summary.json',result[d])
        lines += [f'## {d}','','| 方案 | NDCG@10 | 对GRAM相对变化 | Hit@10人数 |','|---|---:|---:|---:|']
        for k,r in measured.items():
            rel=100*(r['N10']/measured['native']['N10']-1)
            lines.append(f"| {k} | {r['N10']:.6f} | {rel:+.2f}% | {int((ranks[k]<=10).sum())} |")
        ci=comparisons[ARM+'_vs_user_kl1']['descriptive_95pct_N10']
        lines += ['',f"全局α={result[d]['training']['alpha']:.6f}；相对同KL用户α，N10绝对差{comparisons[ARM+'_vs_user_kl1']['delta_N10']:+.6f}，描述区间[{ci[0]:+.6f},{ci[1]:+.6f}]。",'']
        files=[p/name for name in ('predictions.jsonl','ranks.npz','summary.json','smoke.json','runtime_checks.json')]
        write(p/'output_manifest.json',dict(sha256={str(f.relative_to(ROOT)):sha(f) for f in files}))
    review=dict(coverage='11/11',simpson='Domains reported separately.',ecological='Per-user gains/losses retained.',
        berkson='Processed development cohort.',collider='No outcome-based user selection.',
        base_rate='All512 users included in every arm.',regression_to_mean='No independent confirmation, one seed.',
        survivorship='Both domains and all references retained.',look_elsewhere='All ten contrasts saved; unadjusted descriptive intervals.',
        forking_paths='Post-hoc control following known512 results; no new lambda search.',
        causation='Same cached loss and fit folds, one vs33 parameters; CPU replay guard quantifies backend difference.',
        reverse_causality='No targets in inference features; disjoint fit and development users.')
    write(OUT/'summary.json',dict(domains=result,statistical_review=review,test_read=False,automatic_final_selection=False))
    lines += ['2000次配对bootstrap区间未校正多重比较；这批开发用户已重复使用，区间不包含训练种子和方案选择不确定性。',
        '统计检查11/11。没有自动扩大4096或确定最终方法。']
    REPORT.write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('test','prepare','train','supervise','worker','cache','fit','generate'))
    parser.add_argument('--domain',choices=DOMAINS);parser.add_argument('--lease-pid',type=int);args=parser.parse_args()
    if args.mode!='generate':
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
    try:
        if args.mode=='test':print(json.dumps(tests()),flush=True)
        elif args.mode=='prepare':prepare()
        elif args.mode=='train':
            for d in ([args.domain] if args.domain else DOMAINS):train(d)
        elif args.mode=='supervise':
            for d in DOMAINS:verify_training(d)
            B.OUT,B.CFG,B.MODULE,B.REPORT,B.summarize=OUT,CFG,MODULE,REPORT,summarize
            B.supervise()
        elif args.mode=='worker':
            OLD.OUT,OLD.CFG,OLD.MODULE,OLD.progress=OUT,CFG,MODULE,progress
            return OLD.worker(args.domain)
        elif args.mode in ('cache','fit'):
            verify_training(args.domain);progress(args.domain,'reused_completed_cpu_training',gram_forward_passes=0)
        else:
            # CLEAN.Worker owns its own thread initialization in this process.
            Generation(args.domain,args.lease_pid).run()
        return 0
    except BaseException as exc:
        if OUT.exists():
            p=OUT/args.domain if args.domain else OUT
            write(p/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),updated_at=stamp()))
            write(p/'status.json',dict(state='FAILED',error=repr(exc),updated_at=stamp(),test_read=False))
        raise


if __name__=='__main__':sys.exit(main())
