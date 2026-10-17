"""Extend the frozen Qwen4 alpha screen from new512 to new1024 users.

Evaluation only: preserve the first512 prediction bytes and generate the next
512 users, keeping every trained checkpoint, feature and decoder rule fixed.
"""
import argparse
import hashlib
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
from experiment.phase21 import qwen4_alpha_rank as R
from experiment.phase20.catalog_scope_core import metrics, contrast

ROOT, PREVIOUS, SOURCE = R.ROOT, R.OUT, R.INPUT
SCREEN = ROOT/'artifacts/phase21/old10_expert_screen_staged_20261017_v1'
OUT = ROOT/'artifacts/phase21/qwen4_alpha_rank_1024_20261017_v1'
PLAN = ROOT/'plan/第二十一阶段/21.47_Qwen4强度对照扩至新增1024用户_20261017.md'
REPORT = ROOT/'report/第二十一阶段/Qwen4干预强度对照_新增1024用户_20261017.md'
MODULE = 'experiment.phase21.qwen4_alpha_rank_1024'
ARMS, DOMAINS = R.ARMS, R.DOMAINS
KEYS = ('native',*ARMS,'joint_user_reference','sas_only_reference')
CFG = dict(R.CFG,evaluation_users=1024,input_validation_slice=[512,1536],
    evaluation_only=True,reused_prediction_users=512,new_generation_users=512,
    model_selection=False,training_steps=0,groups=['batch1_512','batch2_512','all1024'],
    previous_run=str(PREVIOUS.relative_to(ROOT)),
    stop_after='append second512 for each domain, report both batches and all1024; no retraining')
read, write, sha, stamp = R.read, R.write, R.sha, R.stamp


def prepare():
    assert read(PREVIOUS/'status.json')['state']=='COMPLETED'
    assert read(SCREEN/'status.json')['state']=='COMPLETED'
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    OUT.mkdir(exist_ok=False)
    write(OUT/'config.json',CFG);shutil.copy2(PLAN,OUT/'protocol.md')
    write(OUT/'status.json',dict(state='PREPARING',updated_at=stamp(),test_read=False))
    manifest={};domain_inputs={};audit={}
    def use(path,expected=None):
        rel=str(path.relative_to(ROOT));digest=manifest.get(rel) or sha(path)
        assert expected is None or digest==expected,rel
        manifest[rel]=digest;return rel
    for domain in DOMAINS:
        p=OUT/domain;p.mkdir();files=[]
        old=PREVIOUS/domain
        for name in ('output_manifest.json','training_manifest.json'):
            for rel,digest in read(old/name)['sha256'].items():assert sha(ROOT/rel)==digest,rel
            files.append(use(old/name))
        for name in ('catalog.json','data_config.json','teacher_checks.json','fit_samples.json',
                     'parent_full.pt','user_nll.pt','user_rank.pt','training.json'):
            files.append(use(old/name));shutil.copy2(old/name,p/name);files.append(use(p/name))
        samples=read(SOURCE/domain/'samples.json')[512:1536]
        assert len(samples)==len({s['user_id'] for s in samples})==1024
        assert samples[:512]==read(old/'samples.json')
        assert {s['user_id'] for s in read(p/'fit_samples.json')}.isdisjoint(s['user_id'] for s in samples)
        assert all(s['cohort']=='wide4096' for s in samples)
        write(p/'samples.json',samples);files += [use(SOURCE/domain/'samples.json'),use(p/'samples.json')]
        snapshot=SCREEN/domain/'stage1024_predictions.jsonl'
        assert sha(snapshot)==read(SCREEN/domain/'stage1024_snapshot.json')['sha256']
        reference_rows=[json.loads(x) for x in snapshot.read_text().splitlines()][512:1536]
        refs=[]
        for s,r in zip(samples,reference_rows):
            assert all(s[k]==r[k] for k in ('user_id','target','history'))
            refs.append(dict(user_id=r['user_id'],target=r['target'],history=r['history'],native=r['native'],
                joint_user_reference=r['SASRec__Qwen4'],sas_only_reference=r['SASRec__only']))
        previous_refs=read(old/'references.json')
        for current,previous in zip(refs[:512],previous_refs):
            assert all(current[k]==previous[k] for k in current)
        write(p/'references.json',refs);files += [use(snapshot),use(p/'references.json')]
        prefix=(old/'predictions.jsonl').read_bytes()
        rows=[json.loads(x) for x in prefix.splitlines()]
        assert prefix.endswith(b'\n') and len(rows)==512
        for s,row in zip(samples,rows):
            assert all(s[k]==row[k] for k in ('user_id','target','history'))
        (p/'reused_prefix.jsonl').write_bytes(prefix)
        write(p/'prefix_checks.json',dict(rows=512,bytes=len(prefix),sha256=hashlib.sha256(prefix).hexdigest(),
            original=str((old/'predictions.jsonl').relative_to(ROOT)),original_bytes_unchanged=True))
        files += [use(old/'predictions.jsonl'),use(p/'reused_prefix.jsonl'),use(p/'prefix_checks.json')]
        user_path=SOURCE/domain/'features/validation/SASRec__Qwen4.npy'
        expert_path=SOURCE/domain/'features/validation/experts.npy'
        users=np.array(np.load(user_path,mmap_mode='r')[512:1536])
        experts=np.take(np.load(expert_path,mmap_mode='r')[512:1536],(0,4),axis=-1)
        np.testing.assert_array_equal(users[:512],np.load(old/'dev_user.npy'))
        np.testing.assert_array_equal(experts[:512],np.load(old/'dev_experts.npy'))
        # Freeze exactly the values used here, avoiding hashing/re-reading the
        # unrelated remaining3584-user large source arrays on each worker.
        np.save(p/'dev_user.npy',users);np.save(p/'dev_experts.npy',experts)
        parent=torch.load(p/'parent_full.pt',map_location='cpu')
        assert parent['fit_users']==[s['user_id'] for s in read(p/'fit_samples.json')]
        x=R.J.normalize(torch.from_numpy(users),*[torch.from_numpy(np.asarray(parent[k][:10])) for k in ('mean','std')])
        base=R.A.AlphaOnly(parent['state_dict'])
        with torch.no_grad():beta=base(x)[0].numpy()
        expected={k:v for k,v in base.state_dict().items() if not k.startswith('strength.')}
        alphas={'fixed':np.full(1024,.3)}
        for arm in ARMS[1:]:
            ck=torch.load(p/(arm+'.pt'),map_location='cpu')
            assert ck['fit_users']==parent['fit_users']
            for k,v in expected.items():torch.testing.assert_close(ck['state_dict'][k],v,rtol=0,atol=0)
            np.testing.assert_array_equal(ck['mean'][:10],parent['mean'][:10])
            np.testing.assert_array_equal(ck['std'][:10],parent['std'][:10])
            model=R.A.AlphaOnly(parent['state_dict']);model.load_state_dict(ck['state_dict'])
            with torch.no_grad():b,a=model(x)
            np.testing.assert_array_equal(b.numpy(),beta)
            alphas[arm]=a.double().numpy()
            # Independent NumPy MLP verifies the CPU exports.
            nb,na=R.B.numpy_forward(ck['state_dict'],users,parent['mean'][:10],parent['std'][:10])
            np.testing.assert_allclose(beta,nb,atol=2e-6,rtol=2e-5)
            np.testing.assert_allclose(alphas[arm],na,atol=2e-6,rtol=2e-5)
        scores=(beta[:,None,:]*experts).sum(-1)
        previous_alphas=np.load(old/'alphas.npz')
        for arm in ARMS:
            np.testing.assert_allclose(alphas[arm][:512],previous_alphas[arm],atol=2e-6,rtol=2e-5)
            alphas[arm][:512]=previous_alphas[arm]
        np.testing.assert_allclose(beta[:512],np.load(old/'betas.npy'),atol=2e-6,rtol=2e-5)
        np.testing.assert_allclose(scores[:512],np.load(old/'signals.npy'),atol=3e-6,rtol=2e-5)
        # Exact historical replay on the preserved first512, avoiding backend
        # roundoff in a CPU re-export from changing completed predictions.
        scores[:512]=np.load(old/'signals.npy');beta[:512]=np.load(old/'betas.npy')
        assert np.isfinite(scores).all()
        assert all(np.isfinite(a).all() and np.all((a>0)&(a<1)) for a in alphas.values())
        np.save(p/'signals.npy',scores);np.save(p/'betas.npy',beta);np.savez(p/'alphas.npz',**alphas)
        write(p/'export_checks.json',dict(frozen_trunk_beta_exact=True,trained_checkpoints_reused=True,
            first512_inputs_equal=True,first512_exports_match_and_replayed_exactly=True,
            independent_numpy_verified_users=1024,training_steps=0,target_used_in_features=False,
            alpha_means={k:float(a.mean()) for k,a in alphas.items()},
            source_features=dict(user=str(user_path.relative_to(ROOT)),experts=str(expert_path.relative_to(ROOT)),
                source_slice=[512,1536],expert_columns=[0,4])))
        files += [use(p/name) for name in ('signals.npy','betas.npy','alphas.npz','dev_user.npy','dev_experts.npy','export_checks.json')]
        cfg=read(p/'data_config.json');files.append(use(ROOT/cfg['parent_checkpoint'],cfg['parent_sha256']))
        domain_inputs[domain]=files
        audit[domain]=dict(reused_users=512,new_generation_users=512,training_steps=0,fit_dev_disjoint=True)
        write(p/'status.json',dict(state='PREPARED',updated_at=stamp(),users_completed=512,users_total=1024,
            new_users_completed=0,new_users_total=512,test_read=False))
    common=[]
    for path in (Path(__file__),PLAN,OUT/'config.json',ROOT/'experiment/phase21/qwen4_alpha_rank.py',
                 ROOT/'experiment/phase21/qwen4_alpha_rank_core.py',ROOT/'experiment/phase21/joint_fusion_core.py',
                 ROOT/'experiment/phase21/old10_expert_selection.py',ROOT/'experiment/phase20/lime_decode_screen_core.py'):
        common.append(use(path));dest=OUT/'sources'/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    write(OUT/'manifest.json',dict(sha256=manifest,domain_inputs={d:common+domain_inputs[d] for d in DOMAINS}))
    write(OUT/'preflight_checks.json',audit)
    write(OUT/'status.json',dict(state='PREPARED',updated_at=stamp(),test_read=False,evaluation_only=True,
        users_per_domain=1024,reused_per_domain=512,new_per_domain=512))
    print(json.dumps(dict(state='PREPARED',checks=audit,input_files=len(manifest))),flush=True)


def bind():
    R.OUT=OUT;R.CFG=CFG


class Generation(R.Generation):
    def __init__(self,domain,lease_pid):
        bind();super().__init__(domain,lease_pid)
        self.users_completed=512

    def status(self,state='RUNNING',**extra):
        assert self.lease.poll() is None,'GPU lease exited'
        done=max(512,getattr(self,'users_completed',0))
        rate=float(np.median(self.row_seconds[-32:])) if self.row_seconds else None
        write(OUT/self.domain/'status.json',dict(state=state,stage=self.stage,pid=os.getpid(),updated_at=stamp(),
            gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),users_completed=done,users_total=1024,
            reused_users=512,new_users_completed=done-512,new_users_total=512,completed_decodes=self.completed,
            elapsed_seconds=time.monotonic()-self.started,median_seconds_per_new_user=rate,
            remaining_generation_seconds_estimate=rate*(1024-done) if rate else None,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,test_read=False,training_steps=0,**extra))

    def run(self):
        p=self.folder;prefix=(p/'reused_prefix.jsonl').read_bytes()
        assert hashlib.sha256(prefix).hexdigest()==read(p/'prefix_checks.json')['sha256']
        previous=[json.loads(x) for x in prefix.splitlines()]
        assert self.initial_digest==read(p/'teacher_checks.json')['model_digest']
        self.stage='prefix_replay_and_new_users_smoke';self.status();checks=[]
        for j in (0,511,512,767,1023):
            ctx=self.context(j);native=self.decode(ctx)
            R.B.Generation.equal(dict(ids=native['ids'],scores=native['scores'].tolist()),self.refs[j]['native'])
            q=self.index.conditional_logs(self.scores[j]);zero=self.decode(ctx,q,0.)
            assert native['paths']==zero['paths'];np.testing.assert_array_equal(native['scores'],zero['scores'])
            if j<512:
                for arm in ARMS:
                    result=self.decode(ctx,q,float(self.alpha[arm][j]))
                    R.B.Generation.equal(dict(ids=result['ids'],scores=result['scores'].tolist()),previous[j][arm])
            self.samples[j]['target']=2;changed=self.context(j);self.samples[j]['target']=1
            torch.testing.assert_close(ctx['ids'],changed['ids'],atol=0,rtol=0)
            torch.testing.assert_close(ctx['mask'],changed['mask'],atol=0,rtol=0)
            checks.append(dict(row=j,native_matches=True,alpha_zero_exact=True,
                old_arms_replayed=j<512,target_placeholder_invariant=True))
        write(p/'smoke.json',dict(state='PASSED',checks=checks))
        self.stage='second512_generation';self.status()
        with (p/'predictions.jsonl').open('xb') as stream:
            stream.write(prefix);stream.flush()
            for j in range(512,1024):
                began=time.monotonic();s=self.truth[j];ctx=self.context(j)
                q=self.index.conditional_logs(self.scores[j])
                row=dict(user_id=s['user_id'],target=s['target'],history=s['history'],native=self.refs[j]['native'],
                    joint_user_reference=self.refs[j]['joint_user_reference'],sas_only_reference=self.refs[j]['sas_only_reference'])
                same_alpha={}
                for arm in ARMS:
                    a=float(self.alpha[arm][j])
                    if a in same_alpha:row[arm]=row[same_alpha[a]];continue
                    result=self.decode(ctx,q,a);same_alpha[a]=arm
                    row[arm]=dict(ids=result['ids'],scores=result['scores'].tolist())
                stream.write((json.dumps(row)+'\n').encode());stream.flush()
                self.users_completed=j+1;self.row_seconds.append(time.monotonic()-began)
                if (j+1)%16==0 or j==512:self.status()
        with (p/'predictions.jsonl').open('rb') as stream:assert stream.read(len(prefix))==prefix
        assert self.digest()==self.initial_digest
        write(p/'runtime_checks.json',dict(state='PASSED',gram_model_digest_unchanged=True,
            training_steps=0,first512_prediction_bytes_unchanged=True,new_users_generated=512,
            all_arms_retained=True,history_filter=False))
        self.stage='complete';self.status('COMPLETED')


def worker(domain):
    p=OUT/domain;lease=child=None
    def interrupted(signum,frame):raise InterruptedError('1024 evaluation interrupted or timed out')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGALRM,interrupted)
    signal.alarm(CFG['worker_timeout_seconds'])
    try:
        write(p/'status.json',dict(state='RUNNING',stage='verify_inputs',updated_at=stamp(),pid=os.getpid(),
            users_completed=512,users_total=1024,test_read=False))
        m=read(OUT/'manifest.json');assert read(OUT/'config.json')==CFG
        for rel in m['domain_inputs'][domain]:assert sha(ROOT/rel)==m['sha256'][rel],rel
        gpu=os.environ['CUDA_VISIBLE_DEVICES'];free=next(g['free_mib'] for g in R.B.S.OLD.gpu_inventory() if g['uuid']==gpu)
        if free<CFG['minimum_free_mib']:
            write(p/'status.json',dict(state='ADMISSION_DEFERRED',updated_at=stamp(),free_mib=free));return 75
        with (p/'lease.log').open('x') as log:
            leasepath=p/'lease_status.json'
            lease=subprocess.Popen([sys.executable,str(ROOT/'experiment/gpu_memory_lease.py'),'--gpu','0',
                '--total-lease-mib','16384','--expected-workload-peak-mib','12288','--status-path',str(leasepath)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(p/'lease_process.json',dict(pid=lease.pid,gpu_uuid=gpu,started_at=stamp()))
            began=time.monotonic()
            while not leasepath.exists() or read(leasepath).get('state')!='holding':
                assert lease.poll() is None,'Lease exited';assert time.monotonic()-began<180,'Lease startup timeout';time.sleep(1)
            child=subprocess.Popen([sys.executable,'-u','-m',MODULE,'generate','--domain',domain,'--lease-pid',str(lease.pid)],
                cwd=ROOT,start_new_session=True)
            assert child.wait()==0,'Generation failed; no automatic retry'
        return 0
    finally:
        signal.alarm(0);R.B.stop(child);R.B.stop(lease)


def summarize():
    result={};lines=['# Qwen4干预强度对照：新增1024用户','','## Material Passport','',
        '- Origin Skill: academic-research-suite / experiment-agent','- Origin Mode: validate',
        '- Origin Date: 2026-10-17','- Verification Status: ANALYZED；固定模型真实生成开发集，非独立test',
        '- Version Label: qwen4_alpha_rank_1024_v1','',
        '每域同一固定新增1024名用户，复用首批512预测，只新增第二批512；融合器、训练epoch、输入特征与生成规则均不改变。',
        '不排除历史商品，beam50。旧joint和SAS-only为不同融合器的参考，不能代替当前模型的严格专家消融。','']
    groups=[('batch1_512',np.arange(512)),('batch2_512',np.arange(512,1024)),('all1024',np.arange(1024))]
    pairs=[(arm,'native') for arm in ARMS]+[(arm,'fixed') for arm in ARMS[1:]]+[('user_rank','user_nll')]
    for domain in DOMAINS:
        p=OUT/domain;samples=read(p/'samples.json')
        raw=(p/'predictions.jsonl').read_bytes();prefix=(p/'reused_prefix.jsonl').read_bytes()
        assert raw.startswith(prefix)
        rows=[json.loads(x) for x in raw.splitlines()];assert len(rows)==1024
        ranks={k:[] for k in KEYS}
        for sample,row in zip(samples,rows):
            assert all(sample[k]==row[k] for k in ('user_id','target','history'))
            for k in KEYS:
                rank=R.B.rank(row[k],sample['target'])
                ids,scores=np.asarray(row[k]['ids']),np.asarray(row[k]['scores']);at=np.flatnonzero(ids==sample['target'])
                independent=2**30 if not len(at) else 1+int(((scores>scores[at[0]])|((scores==scores[at[0]])&(ids<sample['target']))).sum())
                assert rank==independent;ranks[k].append(rank)
        ranks={k:np.asarray(v,np.int64) for k,v in ranks.items()}
        previous=np.load(PREVIOUS/domain/'ranks.npz')
        for k in KEYS:np.testing.assert_array_equal(ranks[k][:512],previous[k])
        np.savez_compressed(p/'ranks.npz',users=np.asarray([s['user_id'] for s in samples]),**ranks)
        result[domain]={}
        for group,ix in groups:
            measured={k:metrics(r[ix]) for k,r in ranks.items()}
            result[domain][group]=dict(users=len(ix),metrics=measured,
                comparisons={a+'_vs_'+b:contrast(ranks[a][ix],ranks[b][ix],repeats=2000,seed=20261017) for a,b in pairs},
                relative_to_gram_pct={k:100*(v['N10']/measured['native']['N10']-1) for k,v in measured.items()})
        write(p/'summary.json',result[domain])
        files=[p/n for n in ('ranks.npz','predictions.jsonl','summary.json','runtime_checks.json','smoke.json','export_checks.json')]
        write(p/'output_manifest.json',dict(sha256={str(f.relative_to(ROOT)):sha(f) for f in files}))
    for group,_ in groups:
        lines += [f'## {group}','','| 方法 | Toys N10 | 对GRAM | Beauty N10 | 对GRAM |','|---|---:|---:|---:|---:|']
        for k in KEYS:
            a,b=[result[d][group] for d in DOMAINS]
            lines.append(f'| {k} | {a["metrics"][k]["N10"]:.6f} | {a["relative_to_gram_pct"][k]:+.2f}% | '
                f'{b["metrics"][k]["N10"]:.6f} | {b["relative_to_gram_pct"][k]:+.2f}% |')
    lines += ['','## 合计1024上的配对比较','','| 数据集 | 比较 | N10绝对差 | 描述性95%区间 |','|---|---|---:|---|']
    for d in DOMAINS:
        for name,x in result[d]['all1024']['comparisons'].items():
            ci=x['descriptive_95pct_N10'];lines.append(f'| {d} | {name} | {x["delta_N10"]:+.6f} | [{ci[0]:+.6f}, {ci[1]:+.6f}] |')
    review=dict(coverage='11/11',simpson='Domains, each512 and combined1024 reported separately.',
        ecological='Paired user gains/losses retained; no every-user benefit claim.',
        berkson='Fixed eligible validation hash sample; generalization scope limited.',
        collider='No hit or outcome based resampling.',base_rate='All1024 prescribed users retained.',
        regression_to_mean='First512 examined before extension; both batches are development, not independent confirmation.',
        survivorship='All three arms retained including prior negative results; failures recorded.',
        look_elsewhere='36 descriptive comparisons across three groups and two domains, unadjusted intervals.',
        forking_paths='User-requested sample expansion after first512; weights and rules frozen.',
        causation='Inference limited to these fixed heads and decoder, not expert necessity.',
        reverse_causality='No fitting or model selection on evaluation; target used only for metrics.')
    write(OUT/'summary.json',dict(domains=result,statistical_review=review,test_read=False,automatic_final_selection=False))
    lines += ['','Beauty排序组原来选择epoch0，仍等同固定α初始化；扩样不重新选epoch。',
        '1024减少抽样波动，但不能消除训练种子、研究选择或方法本身的不稳定性。两批方向不一致时必须保留该事实。',
        '统计解释检查11/11；36项区间未做多重比较校正，不含训练与方法选择不确定性。相对变化按同批GRAM计算，区间跨0不表示等效；没有读取test或自动启动下一轮。']
    REPORT.write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','supervise','worker','generate'))
    parser.add_argument('--domain',choices=DOMAINS);parser.add_argument('--lease-pid',type=int);args=parser.parse_args()
    def interrupted(signum,frame):raise InterruptedError('1024 evaluation interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        if args.mode=='prepare':prepare()
        elif args.mode=='supervise':
            R.B.OUT=OUT;R.B.CFG=CFG;R.B.MODULE=MODULE;R.B.REPORT=REPORT;R.B.summarize=summarize
            R.B.supervise()
        elif args.mode=='worker':return worker(args.domain)
        else:Generation(args.domain,args.lease_pid).run()
        return 0
    except BaseException as exc:
        if OUT.exists():
            p=OUT/args.domain if args.domain else OUT
            write(p/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),updated_at=stamp()))
            write(p/'status.json',dict(state='FAILED',error=repr(exc),updated_at=stamp(),test_read=False))
        raise


if __name__=='__main__':sys.exit(main())
