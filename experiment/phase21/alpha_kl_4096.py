"""Frozen SASRec/BGE alpha confirmation on the existing wide4096 cohort.

Reuse completed native/fixed predictions and the first512 learned predictions.
Only the two already-trained KL=.1 heads generate on the remaining3584 users.
LIME-derived experts and the AGRec-inspired legal-mass-preserving guide remain
unchanged. The cohort is development data, not an independent test set.
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
import torch
from experiment.phase21 import alpha_mechanism as M
from experiment.phase21 import global_kl_control as G
from experiment.phase21 import fixed_expert_screen_4096 as F
from experiment.phase21 import global_kl_control_recovery as R

ROOT=M.ROOT
OUT=ROOT/'artifacts/phase21/alpha_kl_4096_20261017_v1'
SMALL=ROOT/'artifacts/phase21/global_kl_control_20261017_v1_recovery1'
SCREEN=F.OUT
PLAN=ROOT/'plan/第二十一阶段/21.59_SASRec_BGE冻结KL干预强度4096用户验证_20261017.md'
REPORT=ROOT/'report/第二十一阶段/SASRec_BGE冻结KL干预强度4096用户验证_20261017.md'
MODULE='experiment.phase21.alpha_kl_4096'
DOMAINS=('Toys','Beauty')
ARMS=('native','fixed03','global_kl1','user_kl1')
JOBS={f'{d}_{s}':dict(domain=d,shard=s) for s in range(2) for d in DOMAINS}
CFG=dict(users_per_domain=4096,reused_learned_users_per_domain=512,
    new_learned_users_per_domain=3584,arms=list(ARMS),beta=[.5,.5],expert_pair=['SASRec','BGE'],
    lambda_value=.1,score_scale=20.,beam_size=50,length_penalty=1.,
    frozen_models=True,training_steps=0,gram_optimizer_steps=0,expert_training_steps=0,
    history_filter=False,history_penalty=False,text_self_exclusion=False,test_read=False,
    evaluation_blocks=5,block_seed=F.CFG['block_seed'],shards_per_domain=2,
    total_lease_mib=16384,expected_workload_peak_mib=12288,minimum_free_mib=18432,
    allocation_limit_gib=11,workload_preallocation_mib=10240,stable_snapshots=3,
    poll_seconds=5,worker_timeout_seconds=21600,resource_wait_seconds=86400,
    scientific_auto_retry=False,per_gpu_active_jobs=1,
    cohort='existing wide4096 development users; includes previously inspected512',
    groups=['all4096','previous512','additional3584','block0','block1','block2','block3','block4'],
    primary_contrast='user_kl1 vs global_kl1 on additional3584, domains separately',
    stop_after='all four arms for both domains; no automatic next experiment')
read,write,sha,stamp=G.read,G.write,G.sha,G.stamp


def assigned(job):
    return list(range(512+JOBS[job]['shard'],4096,2))


def check_rows(rows,samples,arms):
    assert len(rows)==len(samples)
    for row,s in zip(rows,samples):
        assert all(row[k]==s[k] for k in ('user_id','history','target'))
        for a in arms:G.independent_rank(row[a],s['target'])


def prepare():
    assert read(SMALL/'status.json')['state']==read(SCREEN/'status.json')['state']=='COMPLETED'
    assert read(SMALL/'review.json')['state']=='VERIFIED_SAVED_OUTPUTS'
    OUT.mkdir(exist_ok=False);write(OUT/'config.json',CFG);shutil.copy2(PLAN,OUT/'protocol.md')
    manifest={};domain_inputs={}
    def use(path,expected=None):
        rel=str(path.relative_to(ROOT));digest=manifest.get(rel) or sha(path)
        assert expected is None or digest==expected,rel
        manifest[rel]=digest;return rel
    common=[]
    # Source dependencies were frozen by the completed small matched control.
    for rel,digest in read(SMALL/'manifest.json')['sha256'].items():
        if rel.startswith(('experiment/','GRAM/src/','.cache/huggingface/')):
            common.append(use(ROOT/rel,digest))
    for f in (Path(__file__),ROOT/'experiment/phase21/run_alpha_kl_4096.sh',PLAN,OUT/'config.json'):
        common.append(use(f))
    for d in DOMAINS:
        p=OUT/d;p.mkdir();files=[]
        for source in (SMALL/d,SCREEN/d):
            for rel,digest in read(source/'output_manifest.json')['sha256'].items():
                files.append(use(ROOT/rel,digest))
        for name in ('catalog.json','data_config.json','teacher_checks.json'):
            src=SMALL/d/name;assert sha(src)==sha(SCREEN/d/name)
            files.append(use(src));shutil.copy2(src,p/name);files.append(use(p/name))
        for name,src in [('samples.json',SCREEN/d/'samples.json'),
                         ('fit_samples.json',SMALL/d/'fit_samples.json'),
                         ('parent_full.pt',M.OUT/d/'parent_full.pt'),
                         ('user_kl1.pt',SMALL/d/'user_kl1.pt'),
                         ('global_kl1.pt',SMALL/d/'global_kl1.pt')]:
            files.append(use(src));shutil.copy2(src,p/name);files.append(use(p/name))
        assert sha(p/'user_kl1.pt')==sha(M.OUT/d/'user_kl1.pt')
        samples=read(p/'samples.json');fit=read(p/'fit_samples.json')
        assert len(samples)==len({s['user_id'] for s in samples})==4096
        assert {s['user_id'] for s in samples}.isdisjoint(s['user_id'] for s in fit)
        small_samples=read(SMALL/d/'samples.json')
        assert all(all(a[k]==b[k] for k in ('user_id','history','full_history','target'))
                   for a,b in zip(samples[:512],small_samples))
        parent=torch.load(p/'parent_full.pt',map_location='cpu')
        head=torch.load(p/'user_kl1.pt',map_location='cpu')
        global_head=torch.load(p/'global_kl1.pt',map_location='cpu')
        assert parent['fit_users']==head['fit_users']==global_head['fit_users']==[s['user_id'] for s in fit]
        assert head['lambda_value']==.1 and not head['global_only'] and not head['use_evidence']
        assert global_head['lambda_value']==.1 and global_head['global_only'] and not global_head['use_evidence']
        for f in (M.OUT/d/'alphas.npz',M.OUT/d/'dev_user.npy',SMALL/d/'training.json'):
            files.append(use(f))
        features=M.B.OUT/d/'features/validation'
        feature_manifest=read(M.B.OUT/d/'feature_manifest.json')['sha256']
        for name in ('SASRec__BGE.npy','experts.npy','metadata.json'):
            f=features/name;files.append(use(f,feature_manifest[str(f.relative_to(ROOT))]))
        meta=read(features/'metadata.json')
        assert meta['expert_channels']==list(F.EXPERTS)
        assert meta['users_order'][512:]==[s['user_id'] for s in samples]
        user=np.load(features/'SASRec__BGE.npy',mmap_mode='r')[512:]
        np.testing.assert_array_equal(user[:512],np.load(M.OUT/d/'dev_user.npy'))
        model=M.C.Head();model.load_state_dict(head['state_dict']);model.eval().requires_grad_(False)
        with torch.no_grad():inferred=model(M.hidden(parent,user,torch.device('cpu'))).double().numpy()
        assert np.isfinite(inferred).all() and ((inferred>0)&(inferred<1)).all()
        known=np.load(M.OUT/d/'alphas.npz')['user_kl1']
        error=float(np.max(np.abs(inferred[:512]-known)))
        np.testing.assert_allclose(inferred[:512],known,atol=5e-7,rtol=0)
        # Preserve the completed GPU-generated512 exactly after CPU inference replay.
        inferred[:512]=known
        global_alpha=float(read(SMALL/d/'training.json')['alpha'])
        global_model=M.C.Head(True);global_model.load_state_dict(global_head['state_dict'])
        assert abs(float(global_model(torch.zeros(1,32))[0])-global_alpha)<1e-7
        np.savez(p/'alphas.npz',global_kl1=np.full(4096,global_alpha),user_kl1=inferred)
        raw=np.load(features/'experts.npy',mmap_mode='r')
        signals=np.lib.format.open_memmap(p/'signals.npy',mode='w+',dtype='float32',shape=(4096,raw.shape[1]))
        for j in range(0,4096,64):
            signals[j:j+64]=np.float32(.5)*raw[j+512:j+576,:,0]+np.float32(.5)*raw[j+512:j+576,:,3]
        signals.flush();np.testing.assert_array_equal(signals[:512],np.load(SMALL/d/'signals.npy'));del signals
        old=[json.loads(s) for s in (SMALL/d/'predictions.jsonl').read_text().splitlines()]
        check_rows(old,small_samples,ARMS)
        refs=[]
        for j,line in enumerate((SCREEN/d/'predictions.jsonl').read_text().splitlines()):
            row=json.loads(line);s=samples[j]
            assert all(row[k]==s[k] for k in ('user_id','target','history'))
            ref={k:row[k] for k in ('user_id','target','history','native')}
            ref.update(evaluation_index=j,fixed03=row['SASRec__BGE']);refs.append(ref)
            if j<512:
                assert all(ref[a]==old[j][a] for a in ('native','fixed03'))
        check_rows(refs,samples,('native','fixed03'));write(p/'references.json',refs)
        with (p/'reused_predictions.jsonl').open('x') as stream:
            for j,row in enumerate(old):
                stream.write(json.dumps(dict(refs[j],global_kl1=row['global_kl1'],user_kl1=row['user_kl1']))+'\n')
        audit=dict(state='PASSED',users=4096,unique_users=True,fit_disjoint=True,
            first512_same_users_and_predictions=True,baseline4096_reused=True,
            expert_signal_replay_exact=True,user_alpha_cpu_replay_max_abs=error,
            cpu_replay_tolerance=5e-7,first512_alphas_retained_exactly=True,
            new_users=3584,global_alpha=global_alpha,training_steps=0,test_read=False)
        write(p/'input_checks.json',audit)
        for name in ('alphas.npz','signals.npy','references.json','reused_predictions.jsonl','input_checks.json'):
            files.append(use(p/name))
        cfg=read(p/'data_config.json');files.append(use(ROOT/cfg['parent_checkpoint'],cfg['parent_sha256']))
        domain_inputs[d]=common+files
        write(p/'status.json',dict(state='PREPARED',users_completed=512,users_total=4096,updated_at=stamp()))
    for job in JOBS:
        (OUT/'jobs'/job).mkdir(parents=True)
        write(OUT/'jobs'/job/'status.json',dict(state='PREPARED',users_completed=0,users_total=len(assigned(job)),updated_at=stamp()))
    for rel in manifest:
        src=ROOT/rel
        if src.suffix in ('.py','.sh','.md'):
            dest=OUT/'sources'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    write(OUT/'manifest.json',dict(sha256=manifest,domain_inputs=domain_inputs))
    write(OUT/'status.json',dict(state='PREPARED',users_completed=1024,users_total=8192,
        new_guided_generations_total=14336,updated_at=stamp(),test_read=False))
    REPORT.write_text('# SASRec+BGE冻结KL干预强度4096用户验证\n\n状态：PREPARED，尚无本轮结果。\n')
    print(json.dumps(dict(state='PREPARED',hashes=len(manifest),training_steps=0)),flush=True)


class Generation(M.CLEAN.Worker):
    def __init__(self,job,lease_pid):
        self.job=job;self.spec=JOBS[job];self.jobfolder=OUT/'jobs'/job
        self.lease=M.B.S.LeaseHandle(lease_pid);self.row_seconds=[]
        M.CLEAN.OUT=OUT;super().__init__(self.spec['domain'])
        self.truth=self.samples
        from experiment.phase18.core.diff_data import DiffDataset
        self.samples=[dict(s,target=1) for s in self.truth];self.data=DiffDataset(self.samples,self.catalog)
        self.refs=read(self.folder/'references.json')
        self.signal=np.load(self.folder/'signals.npy',mmap_mode='r')
        self.alphas=np.load(self.folder/'alphas.npz')
        self.known=[json.loads(s) for s in (self.folder/'reused_predictions.jsonl').read_text().splitlines()]
        assert self.initial_digest==read(self.folder/'teacher_checks.json')['model_digest']

    def context(self,j,override_index=None):return super().context(j,override_index=j)

    def status(self,state='RUNNING',**extra):
        assert self.lease.poll() is None,'GPU lease exited'
        assert torch.cuda.memory_reserved()/2**20>=CFG['workload_preallocation_mib'],'Workload cache released'
        done=getattr(self,'users_completed',0);total=len(assigned(self.job))
        rate=float(np.median(self.row_seconds[-32:])) if self.row_seconds else None
        write(self.jobfolder/'status.json',dict(state=state,stage=self.stage,job=self.job,pid=os.getpid(),
            updated_at=stamp(),gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),
            users_completed=done,users_total=total,completed_decodes=self.completed,
            elapsed_seconds=time.monotonic()-self.started,median_seconds_per_user=rate,
            remaining_generation_seconds_estimate=rate*(total-done) if rate is not None else None,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            reserved_mib=torch.cuda.memory_reserved()/2**20,test_read=False,**extra))

    def run(self):
        def entry(v):return dict(ids=v['ids'],scores=v['scores'].tolist())
        self.stage='matched_reference_smoke';self.status();checks=[]
        ix=assigned(self.job)
        for j in (0,ix[0],ix[-1]):
            ctx=self.context(j);q=self.index.conditional_logs(self.signal[j])
            native=self.decode(ctx);M.B.Generation.equal(entry(native),self.refs[j]['native'])
            zero=self.decode(ctx,q,0.);assert zero['paths']==native['paths']
            np.testing.assert_array_equal(zero['scores'],native['scores'])
            M.B.Generation.equal(entry(self.decode(ctx,q,.3)),self.refs[j]['fixed03'])
            for a in ('global_kl1','user_kl1'):
                v=entry(self.decode(ctx,q,float(self.alphas[a][j])))
                if j<512:M.B.Generation.equal(v,self.known[j][a])
            self.samples[j]['target']=2;changed=self.context(j);self.samples[j]['target']=1
            torch.testing.assert_close(ctx['ids'],changed['ids'],atol=0,rtol=0)
            torch.testing.assert_close(ctx['mask'],changed['mask'],atol=0,rtol=0)
            checks.append(dict(row=j,native_reproduced=True,zero_exact=True,fixed03_reproduced=True,
                previous_learned_replayed=j<512,new_arm_path_rescore_passed=True,target_placeholder_invariant=True))
        write(self.jobfolder/'smoke.json',dict(state='PASSED',checks=checks))
        self.stage='two_frozen_alpha_generation';self.status()
        with (self.jobfolder/'predictions.jsonl').open('x') as stream:
            for n,j in enumerate(ix):
                began=time.monotonic();ctx=self.context(j);q=self.index.conditional_logs(self.signal[j])
                row=dict(self.refs[j])
                for a in ('global_kl1','user_kl1'):row[a]=entry(self.decode(ctx,q,float(self.alphas[a][j])))
                stream.write(json.dumps(row)+'\n');stream.flush()
                self.users_completed=n+1;self.row_seconds.append(time.monotonic()-began)
                if n==0 or (n+1)%16==0 or n+1==len(ix):self.status()
        assert self.digest()==self.initial_digest
        write(self.jobfolder/'runtime_checks.json',dict(state='PASSED',gram_digest_unchanged=True,
            frozen_alpha_inputs=True,legal_mass_preserved=True,history_filter=False,history_penalty=False,test_read=False))
        write(self.jobfolder/'output_manifest.json',dict(sha256={str(f.relative_to(ROOT)):sha(f) for f in
            [self.jobfolder/n for n in ('predictions.jsonl','smoke.json','runtime_checks.json')]}))
        self.stage='complete';self.status('COMPLETED')


def generate(job,lease_pid,attempt):
    p=OUT/'jobs'/job
    write(p/'status.json',dict(state='RUNNING',stage='reserve_working_memory',users_completed=0,
        users_total=len(assigned(job)),pid=os.getpid(),updated_at=stamp()))
    before=None
    try:
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(11*2**30/torch.cuda.get_device_properties(0).total_memory)
        before=R.memory_snapshot()
        held=torch.empty(CFG['workload_preallocation_mib']*2**20,device='cuda:0',dtype=torch.uint8)
        torch.cuda.synchronize();del held
        cached=R.memory_snapshot();assert cached['reserved_mib']>=CFG['workload_preallocation_mib']
    except RuntimeError as exc:
        if 'out of memory' not in str(exc).lower():raise
        write(p/f'reservation_attempt{attempt}.json',dict(state='ADMISSION_DEFERRED',reason=repr(exc),
            before=before,model_loaded=False,predictions_written=False,updated_at=stamp()))
        write(p/'status.json',dict(state='ADMISSION_DEFERRED',stage='reservation',users_completed=0,
            users_total=len(assigned(job)),updated_at=stamp()))
        return 75
    write(p/f'reservation_attempt{attempt}.json',dict(state='RESERVED',before=before,cached_for_model=cached,updated_at=stamp()))
    torch.cuda.reset_peak_memory_stats();Generation(job,lease_pid).run()
    return 0


def worker(job,attempt):
    lease=child=None;p=OUT/'jobs'/job
    def interrupted(signum,frame):raise InterruptedError('Worker interrupted or hard timeout')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGALRM,interrupted)
    signal.alarm(CFG['worker_timeout_seconds'])
    try:
        manifest=read(OUT/'manifest.json');assert read(OUT/'config.json')==CFG
        for rel in manifest['domain_inputs'][JOBS[job]['domain']]:assert sha(ROOT/rel)==manifest['sha256'][rel],rel
        gpu=os.environ['CUDA_VISIBLE_DEVICES']
        free=next(g['free_mib'] for g in M.B.S.OLD.gpu_inventory() if g['uuid']==gpu)
        write(p/f'admission_attempt{attempt}.json',dict(gpu_uuid=gpu,free_before_lease_mib=free,updated_at=stamp()))
        def defer(stage):
            write(p/'status.json',dict(state='ADMISSION_DEFERRED',stage=stage,users_completed=0,
                users_total=len(assigned(job)),updated_at=stamp()));return 75
        if free<CFG['minimum_free_mib']:return defer('free_memory_changed_before_lease')
        with (p/f'lease_attempt{attempt}.log').open('x') as log:
            lease_path=p/f'lease_status_attempt{attempt}.json'
            lease=subprocess.Popen([sys.executable,str(ROOT/'experiment/gpu_memory_lease.py'),'--gpu','0',
                '--total-lease-mib','16384','--expected-workload-peak-mib','12288','--status-path',str(lease_path)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(p/f'lease_process_attempt{attempt}.json',dict(pid=lease.pid,gpu_uuid=gpu,updated_at=stamp()))
            began=time.monotonic()
            while not lease_path.exists() or read(lease_path).get('state')!='holding':
                assert lease.poll() is None,'Lease exited';assert time.monotonic()-began<180,'Lease timeout';time.sleep(1)
            free_after=next(g['free_mib'] for g in M.B.S.OLD.gpu_inventory() if g['uuid']==gpu)
            write(p/f'post_lease_attempt{attempt}.json',dict(gpu_uuid=gpu,free_mib=free_after,updated_at=stamp()))
            if free_after<CFG['workload_preallocation_mib']+2048:return defer('free_memory_changed_after_lease')
            child=subprocess.Popen([sys.executable,'-u','-m',MODULE,'generate','--job',job,
                '--lease-pid',str(lease.pid),'--attempt',str(attempt)],cwd=ROOT,start_new_session=True)
            code=child.wait()
            if code==75 and read(p/'status.json')['state']=='ADMISSION_DEFERRED':return 75
            assert code==0,('Generation failed; see failure_generate.json',code)
        return 0
    finally:
        signal.alarm(0);M.B.stop(child);M.B.stop(lease)


def summarize():
    result={};hashes={};lines=['# SASRec+BGE冻结KL干预强度4096用户验证','','## Material Passport','',
        '- Origin Skill: academic-research-suite / experiment-agent，内联执行','- Origin Mode: run + validate',
        '- Origin Date: 2026-10-17','- Verification Status: VERIFIED_SAVED_OUTPUTS；冻结模型开发集扩样',
        '- Version Label: alpha_kl_4096_20261017_v1','',
        '复用已完成的原生/固定α预测及此前512人学习α预测；其余3584人真实生成两组冻结α。没有训练或新超参搜索。',
        '新增用户未用于此前α头的训练或512结果选择；其原生/固定预测及其他实验可能已被查看，因此不是独立测试集。','']
    pairs=[('user_kl1','global_kl1'),('global_kl1','fixed03'),('user_kl1','fixed03'),
           ('fixed03','native'),('global_kl1','native'),('user_kl1','native')]
    for d in DOMAINS:
        p=OUT/d;samples=read(p/'samples.json');indexed={}
        files=[p/'reused_predictions.jsonl']
        for job,spec in JOBS.items():
            if spec['domain']!=d:continue
            jp=OUT/'jobs'/job
            assert read(jp/'smoke.json')['state']==read(jp/'runtime_checks.json')['state']=='PASSED'
            for rel,digest in read(jp/'output_manifest.json')['sha256'].items():assert sha(ROOT/rel)==digest,rel
            files.append(jp/'predictions.jsonl')
        for f in files:
            for line in f.read_text().splitlines():
                row=json.loads(line);j=row['evaluation_index'];assert j not in indexed
                indexed[j]=row
        assert set(indexed)==set(range(4096));rows=[indexed[j] for j in range(4096)]
        check_rows(rows,samples,ARMS)
        refs=read(p/'references.json')
        assert all(all(r[a]==ref[a] for a in ('native','fixed03')) for r,ref in zip(rows,refs))
        with (p/'predictions.jsonl').open('x') as stream:
            for row in rows:stream.write(json.dumps(row)+'\n')
        ranks={a:np.asarray([G.independent_rank(r[a],s['target']) for r,s in zip(rows,samples)]) for a in ARMS}
        np.savez_compressed(p/'ranks.npz',**ranks)
        groups=dict(all4096=np.arange(4096),previous512=np.arange(512),additional3584=np.arange(512,4096))
        for b in range(5):groups['block'+str(b)]=np.asarray([j for j,s in enumerate(samples) if s['evaluation_block']==b])
        values={}
        for group,ix in groups.items():
            assert len(ix)
            values[group]=dict(users=len(ix),metrics={a:M.metrics(r[ix]) for a,r in ranks.items()},
                comparisons={a+'_vs_'+b:M.contrast(ranks[a][ix],ranks[b][ix],seed=20261017) for a,b in pairs},
                decompositions={a+'_vs_'+b:G.decomposition(ranks[a][ix],ranks[b][ix]) for a,b in pairs})
        # Recalculate all cuts using an independently expressed utility formula.
        for group,ix in groups.items():
            for a in ARMS:
                for cut in (5,10,20,50):
                    measured=float(sum(1/np.log2(int(r)+1) if r<=cut else 0. for r in ranks[a][ix])/len(ix))
                    assert abs(measured-values[group]['metrics'][a]['N'+str(cut)])<1e-14
        result[d]=dict(groups=values,inputs=read(p/'input_checks.json'))
        write(p/'summary.json',result[d]);lines += [f'## {d}','',
            '| 方案 | 全4096 NDCG@10（对GRAM） | 新增3584 NDCG@10（对GRAM） | 全4096 Hit@10人数 |',
            '|---|---:|---:|---:|']
        for a in ARMS:
            cells=[]
            for group in ('all4096','additional3584'):
                m=values[group]['metrics'];cells.append(f"{m[a]['N10']:.6f}（{100*(m[a]['N10']/m['native']['N10']-1):+.2f}%）")
            lines.append(f"| {a} | {cells[0]} | {cells[1]} | {int((ranks[a]<=10).sum())} |")
        for group in ('all4096','additional3584'):
            c=values[group]['comparisons']['user_kl1_vs_global_kl1'];ci=c['descriptive_95pct_N10']
            lines += ['',f"{group} 用户对全局α：N10绝对差{c['delta_N10']:+.6f}，95%描述区间[{ci[0]:+.6f},{ci[1]:+.6f}]。"]
        outputs=[p/n for n in ('predictions.jsonl','ranks.npz','summary.json')]
        hm={str(f.relative_to(ROOT)):sha(f) for f in outputs};hashes.update(hm)
        write(p/'output_manifest.json',dict(sha256=hm))
    review=dict(coverage='11/11',simpson='Domains and fixed hash blocks separate.',
        ecological='Per-user gained/lost hits and common-hit rank contributions retained.',
        berkson='Processed development cohort, not population sampling.',collider='No outcome-based user selection.',
        base_rate='All4096 and baseline hit rates retained.',regression_to_mean='Frozen heads; one training seed; additional3584 separately reported.',
        survivorship='All four arms in both domains, no pruning.',look_elsewhere='Six paired comparisons in eight groups per domain; descriptive unadjusted intervals.',
        forking_paths='Four arms and primary comparison frozen before new generation; no new hyperparameters.',
        causation='Same expert mixture/GRAM and matched KL objective; cohort and prior selection limit interpretation.',
        reverse_causality='No target inference features; fit users disjoint; labels only for verification and metrics.')
    write(OUT/'summary.json',dict(domains=result,statistical_review=review,test_read=False,
        automatic_final_selection=False,output_hashes=hashes))
    lines += ['', '2000次配对bootstrap未校正多重比较，仅作描述；未包含训练种子不确定性。扩样仍属开发集验证，跨0不代表等效。',
        '五个固定hash块用于稳定性检查，不是五折重新训练。统计检查11/11；未启动下一实验。']
    REPORT.write_text('\n'.join(lines)+'\n')


def supervise():
    assert read(OUT/'status.json')['state']=='PREPARED'
    pending=list(JOBS);active={};finished={};attempts={j:0 for j in JOBS};stable={};began=time.monotonic()
    def statuses():return {j:read(OUT/'jobs'/j/'status.json') for j in JOBS}
    try:
        while pending or active:
            for job,t in list(active.items()):
                code=t['process'].poll()
                if code is None and time.monotonic()-t['started']>CFG['worker_timeout_seconds']+120:
                    M.B.stop(t['process']);code=t['process'].returncode
                if code is None:continue
                t['log'].close();p=OUT/'jobs'/job;s=read(p/'status.json')
                lp=p/f"lease_status_attempt{attempts[job]}.json"
                released=not lp.exists() or read(lp).get('state')=='released'
                write(p/f"attempt{attempts[job]}_result.json",dict(exit_code=code,status=s,lease_released=released))
                if code==75 and s['state']=='ADMISSION_DEFERRED':pending.append(job)
                else:finished[job]=dict(exit_code=code,state=s['state'],lease_released=released)
                stable[t['gpu']['uuid']]=0;del active[job]
            inventory=M.B.S.OLD.gpu_inventory();occupied={t['gpu']['uuid'] for t in active.values()};eligible=[]
            for g in inventory:
                ok=g['uuid'] not in occupied and g['free_mib']>=CFG['minimum_free_mib']
                stable[g['uuid']]=stable.get(g['uuid'],0)+1 if ok else 0
                if stable[g['uuid']]>=CFG['stable_snapshots']:eligible.append(g)
            eligible.sort(key=lambda g:(-g['free_mib'],g['utilization'],g['index']))
            for job in list(pending):
                if not eligible:break
                gpu=eligible.pop(0);pending.remove(job);attempts[job]+=1
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                    OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
                log=(OUT/'jobs'/job/f'worker_attempt{attempts[job]}.log').open('x')
                write(OUT/'jobs'/job/'status.json',dict(state='STARTING',users_completed=0,
                    users_total=len(assigned(job)),gpu_index=gpu['index'],updated_at=stamp()))
                process=subprocess.Popen([sys.executable,'-u','-m',MODULE,'worker','--job',job,
                    '--attempt',str(attempts[job])],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[job]=dict(process=process,log=log,gpu=gpu,started=time.monotonic())
            if pending and time.monotonic()-began>CFG['resource_wait_seconds']:
                for job in pending:
                    finished[job]=dict(exit_code=124,state='RESOURCE_WAIT_LIMIT',lease_released=True)
                    write(OUT/'jobs'/job/'status.json',dict(state='RESOURCE_WAIT_LIMIT',users_completed=0,
                        users_total=len(assigned(job)),updated_at=stamp()))
                pending=[]
            ss=statuses();domains={}
            for d in DOMAINS:
                members={j:s for j,s in ss.items() if JOBS[j]['domain']==d}
                state='COMPLETED' if all(s['state']=='COMPLETED' for s in members.values()) else (
                    'INCOMPLETE' if any(j in finished and finished[j]['exit_code'] for j in members) else 'RUNNING')
                domains[d]=dict(state=state,users_completed=512+sum(s.get('users_completed',0) for s in members.values()),
                    users_total=4096,updated_at=stamp(),jobs=members)
                write(OUT/d/'status.json',domains[d])
            write(OUT/'status.json',dict(state='RUNNING' if active else 'WAITING_FOR_RESOURCES' if pending else 'FINALIZING',
                pid=os.getpid(),updated_at=stamp(),elapsed_seconds=time.monotonic()-began,
                users_completed=sum(s['users_completed'] for s in domains.values()),users_total=8192,
                reused_users=1024,new_guided_generations_total=14336,pending=pending,finished=finished,
                domains=domains,gpu_inventory=inventory,test_read=False,
                active={j:dict(pid=t['process'].pid,gpu_index=t['gpu']['index'],gpu_uuid=t['gpu']['uuid'],worker=ss[j]) for j,t in active.items()}))
            if pending or active:time.sleep(CFG['poll_seconds'])
        complete=len(finished)==len(JOBS) and all(t['exit_code']==0 and t['state']=='COMPLETED' and t['lease_released'] for t in finished.values())
        if complete:summarize()
        write(OUT/'status.json',dict(state='COMPLETED' if complete else 'INCOMPLETE',finished=finished,
            users_completed=sum(s['users_completed'] for s in domains.values()),users_total=8192,
            updated_at=stamp(),elapsed_seconds=time.monotonic()-began,report=str(REPORT),test_read=False))
    finally:
        for t in active.values():M.B.stop(t['process']);t['log'].close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','supervise','worker','generate'))
    parser.add_argument('--job',choices=JOBS);parser.add_argument('--lease-pid',type=int)
    parser.add_argument('--attempt',type=int,default=1);args=parser.parse_args()
    def interrupted(signum,frame):raise InterruptedError('Experiment interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    if args.mode=='prepare':torch.set_num_threads(2);torch.set_num_interop_threads(1)
    try:
        if args.mode=='prepare':prepare()
        elif args.mode=='supervise':supervise()
        elif args.mode=='worker':return worker(args.job,args.attempt)
        else:return generate(args.job,args.lease_pid,args.attempt)
        return 0
    except BaseException as exc:
        p=OUT/'jobs'/args.job if args.job else OUT
        if p.exists():
            write(p/('failure_'+args.mode+'.json'),dict(error=repr(exc),traceback=traceback.format_exc(),updated_at=stamp()))
            error=read(p/'failure_generate.json')['error'] if args.mode=='worker' and (p/'failure_generate.json').exists() else repr(exc)
            previous=read(p/'status.json') if (p/'status.json').exists() else {}
            previous.update(state='FAILED',stage=args.mode,error=error,updated_at=stamp(),test_read=False)
            write(p/'status.json',previous)
        raise


if __name__=='__main__':sys.exit(main())
