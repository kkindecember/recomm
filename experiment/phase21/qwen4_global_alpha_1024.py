"""Evaluate an existing frozen-beta learned-global-alpha control on same1024.

Reuses the original 'calibrate' checkpoints: only one scalar alpha was trained
with target-path NLL. No new fitting, decoder changes or population expansion.
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
from experiment.phase21 import qwen4_alpha_rank_1024 as X
from experiment.phase21 import review_screen1024 as REVIEW

ROOT,SOURCE=X.ROOT,X.OUT
OUT=ROOT/'artifacts/phase21/qwen4_global_alpha_1024_20261017_v1'
PLAN=ROOT/'plan/第二十一阶段/21.48_1024筛选收尾与统一学习强度对照_20261017.md'
MODULE='experiment.phase21.qwen4_global_alpha_1024'
DOMAINS=X.DOMAINS
CFG=dict(X.CFG,arms=['frozen_global'],new_generation_users=1024,reused_prediction_users=0,
    checkpoint_source='existing qwen4_calibrate: frozen CE trunk/beta, one learned global alpha',
    training_steps=0,evaluation_only=True,previous_run=str(SOURCE.relative_to(ROOT)),
    stop_after='one existing global alpha control on same1024, then finish combined review; no next run')
read,write,sha,stamp=X.read,X.write,X.sha,X.stamp


def prepare():
    assert read(SOURCE/'status.json')['state']=='COMPLETED'
    OUT.mkdir(exist_ok=False);write(OUT/'config.json',CFG);shutil.copy2(PLAN,OUT/'protocol.md')
    write(OUT/'status.json',dict(state='PREPARING',updated_at=stamp(),test_read=False))
    torch.set_num_threads(2);manifest={};inputs={}
    def use(path,expected=None):
        rel=str(path.relative_to(ROOT));digest=manifest.get(rel) or sha(path)
        assert expected is None or digest==expected,rel
        manifest[rel]=digest;return rel
    for d in DOMAINS:
        p=OUT/d;p.mkdir();files=[]
        for name in ('samples.json','references.json','catalog.json','data_config.json','teacher_checks.json',
                     'signals.npy','betas.npy','dev_user.npy','parent_full.pt'):
            files.append(use(SOURCE/d/name));shutil.copy2(SOURCE/d/name,p/name);files.append(use(p/name))
        src=X.R.OLD.OUT/d/'qwen4_calibrate.pt'
        expected=read(X.R.OLD.OUT/d/'training_manifest.json')['sha256'][str(src.relative_to(ROOT))]
        files.append(use(src,expected));shutil.copy2(src,p/'global_checkpoint.pt');files.append(use(p/'global_checkpoint.pt'))
        ck=torch.load(src,map_location='cpu');parent=torch.load(p/'parent_full.pt',map_location='cpu')
        assert ck['mode']=='calibrate' and ck['fit_users']==parent['fit_users']
        for k in ('mean','std'):np.testing.assert_array_equal(ck[k][:10],parent[k][:10])
        model=X.R.J.JointFusion('calibrate',parent['state_dict']);model.load_state_dict(ck['state_dict'])
        reference=X.R.A.AlphaOnly(parent['state_dict'])
        for k,v in reference.state_dict().items():
            if not k.startswith('strength.'):torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0)
        users=np.load(p/'dev_user.npy');normal=X.R.J.normalize(torch.from_numpy(users),
            *[torch.from_numpy(np.asarray(parent[k][:10])) for k in ('mean','std')])
        with torch.no_grad():beta,alpha=model(normal)
        np.testing.assert_allclose(beta.numpy(),np.load(p/'betas.npy'),atol=2e-6,rtol=2e-5)
        a=alpha.double().numpy();assert np.unique(a).size==1 and 0<a[0]<1
        alpha_source=X.R.OLD.OUT/d/'alphas.npz'
        expected_a=np.load(alpha_source)['qwen4_calibrate']
        assert np.unique(expected_a).size==1
        # CPU and CUDA sigmoid may differ by one float32 ULP. Verify that
        # bound and replay the original exported scalar exactly, rather than
        # silently replacing the established control with a rounded variant.
        cpu_difference=float(np.max(np.abs(a[:512]-expected_a)))
        np.testing.assert_allclose(a[:512],expected_a,atol=np.finfo(np.float32).eps,rtol=0)
        a=np.full(1024,float(expected_a[0]))
        files.append(use(alpha_source,read(X.R.OLD.OUT/d/'training_manifest.json')['sha256'][str(alpha_source.relative_to(ROOT))]))
        np.save(p/'alpha.npy',a)
        write(p/'checkpoint_checks.json',dict(state='PASSED',frozen_beta_and_trunk_exact=True,
            matching_feature_normalization=True,fit_users_same=True,export_users=1024,
            learned_global_alpha=float(a[0]),selected_epochs=ck['selected_epochs'],
            cpu_sigmoid_vs_cached_max_abs=cpu_difference,cached_alpha_replayed_exactly=True,
            trained_parameters=1,training_steps_this_run=0,source=str(src.relative_to(ROOT)),
            comparison='same CE beta, same target-path NLL and internal-fold budget; only alpha parameterization differs',
            reference_user_alpha_checkpoint=str((SOURCE/d/'user_nll.pt').relative_to(ROOT))))
        files += [use(p/'alpha.npy'),use(p/'checkpoint_checks.json')]
        cfg=read(p/'data_config.json');files.append(use(ROOT/cfg['parent_checkpoint'],cfg['parent_sha256']))
        inputs[d]=files
        write(p/'status.json',dict(state='PREPARED',updated_at=stamp(),users_completed=0,users_total=1024,test_read=False))
    common=[]
    for path in (Path(__file__),Path(REVIEW.__file__),PLAN,OUT/'config.json',
                 ROOT/'experiment/phase21/old10_expert_selection.py',ROOT/'experiment/phase20/lime_decode_screen_core.py'):
        common.append(use(path));dest=OUT/'sources'/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    write(OUT/'manifest.json',dict(sha256=manifest,domain_inputs={d:common+inputs[d] for d in DOMAINS}))
    write(OUT/'status.json',dict(state='PREPARED',updated_at=stamp(),test_read=False,training_steps=0))
    print('PREPARED: existing global alpha, identical frozen beta and1024 users, no training',flush=True)


class Generation(X.R.CLEAN.Worker):
    def __init__(self,d,lease_pid):
        self.lease=X.R.B.S.LeaseHandle(lease_pid);self.row_seconds=[];X.R.CLEAN.OUT=OUT
        super().__init__(d)
        self.truth=self.samples
        from experiment.phase18.core.diff_data import DiffDataset
        self.samples=[dict(s,target=1) for s in self.truth];self.data=DiffDataset(self.samples,self.catalog)
        self.refs=read(self.folder/'references.json');self.scores=np.load(self.folder/'signals.npy')
        self.alpha=np.load(self.folder/'alpha.npy')

    def context(self,j,override_index=None):return super().context(j,override_index=j)

    def status(self,state='RUNNING',**extra):
        assert self.lease.poll() is None,'GPU lease exited'
        done=getattr(self,'users_completed',0);rate=float(np.median(self.row_seconds[-32:])) if self.row_seconds else None
        write(OUT/self.domain/'status.json',dict(state=state,stage=self.stage,pid=os.getpid(),updated_at=stamp(),
            gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),users_completed=done,users_total=1024,
            median_seconds_per_user=rate,remaining_generation_seconds_estimate=rate*(1024-done) if rate else None,
            elapsed_seconds=time.monotonic()-self.started,completed_decodes=self.completed,
            test_read=False,training_steps=0,**extra))

    def run(self):
        assert self.initial_digest==read(self.folder/'teacher_checks.json')['model_digest']
        self.stage='same_baseline_zero_and_target_smoke';self.status();checks=[]
        for j in (0,511,512,1023):
            ctx=self.context(j);native=self.decode(ctx)
            X.R.B.Generation.equal(dict(ids=native['ids'],scores=native['scores'].tolist()),self.refs[j]['native'])
            q=self.index.conditional_logs(self.scores[j]);zero=self.decode(ctx,q,0.)
            assert zero['paths']==native['paths'];np.testing.assert_array_equal(zero['scores'],native['scores'])
            self.samples[j]['target']=2;changed=self.context(j);self.samples[j]['target']=1
            torch.testing.assert_close(ctx['ids'],changed['ids'],rtol=0,atol=0)
            torch.testing.assert_close(ctx['mask'],changed['mask'],rtol=0,atol=0)
            checks.append(dict(row=j,native_matches=True,zero_exact=True,target_invariant=True))
        write(self.folder/'smoke.json',dict(state='PASSED',checks=checks))
        self.stage='learned_global_alpha_generation';self.status()
        with (self.folder/'predictions.jsonl').open('x') as stream:
            for j,s in enumerate(self.truth):
                started=time.monotonic();ctx=self.context(j);q=self.index.conditional_logs(self.scores[j])
                decoded=self.decode(ctx,q,float(self.alpha[j]))
                row=dict(user_id=s['user_id'],target=s['target'],history=s['history'],native=self.refs[j]['native'],
                    frozen_global=dict(ids=decoded['ids'],scores=decoded['scores'].tolist()))
                stream.write(json.dumps(row)+'\n');stream.flush()
                self.users_completed=j+1;self.row_seconds.append(time.monotonic()-started)
                if j==0 or (j+1)%16==0 or j==1023:self.status()
        assert self.digest()==self.initial_digest
        write(self.folder/'runtime_checks.json',dict(state='PASSED',users=1024,
            gram_model_digest_unchanged=True,trained_model_unchanged=True,history_filter=False,training_steps=0))
        self.stage='complete';self.status('COMPLETED')


def worker(d):
    p=OUT/d;lease=child=None
    def interrupted(signum,frame):raise InterruptedError('Global alpha evaluation interrupted or timed out')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGALRM,interrupted);signal.alarm(CFG['worker_timeout_seconds'])
    try:
        manifest=read(OUT/'manifest.json');assert read(OUT/'config.json')==CFG
        for rel in manifest['domain_inputs'][d]:assert sha(ROOT/rel)==manifest['sha256'][rel],rel
        gpu=os.environ['CUDA_VISIBLE_DEVICES'];free=next(g['free_mib'] for g in X.R.B.S.OLD.gpu_inventory() if g['uuid']==gpu)
        if free<CFG['minimum_free_mib']:
            write(p/'status.json',dict(state='ADMISSION_DEFERRED',updated_at=stamp(),free_mib=free));return 75
        with (p/'lease.log').open('x') as log:
            path=p/'lease_status.json'
            lease=subprocess.Popen([sys.executable,str(ROOT/'experiment/gpu_memory_lease.py'),'--gpu','0',
                '--total-lease-mib','16384','--expected-workload-peak-mib','12288','--status-path',str(path)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(p/'lease_process.json',dict(pid=lease.pid,gpu_uuid=gpu,started_at=stamp()))
            began=time.monotonic()
            while not path.exists() or read(path).get('state')!='holding':
                assert lease.poll() is None;assert time.monotonic()-began<180;time.sleep(1)
            child=subprocess.Popen([sys.executable,'-u','-m',MODULE,'generate','--domain',d,'--lease-pid',str(lease.pid)],
                cwd=ROOT,start_new_session=True)
            assert child.wait()==0,'Global alpha evaluation failed; no automatic retry'
        return 0
    finally:
        signal.alarm(0);X.R.B.stop(child);X.R.B.stop(lease)


def summarize():
    result={}
    for d in DOMAINS:
        p=OUT/d;rows=[json.loads(x) for x in (p/'predictions.jsonl').read_text().splitlines()]
        samples=read(p/'samples.json');assert len(rows)==len(samples)==1024
        rr={k:[] for k in ('native','frozen_global')}
        for r,s in zip(rows,samples):
            assert all(r[k]==s[k] for k in ('user_id','target','history'))
            for k in rr:rr[k].append(X.R.B.rank(r[k],s['target']))
        rr={k:np.asarray(v,np.int64) for k,v in rr.items()}
        np.testing.assert_array_equal(rr['native'],np.load(SOURCE/d/'ranks.npz')['native'])
        np.savez_compressed(p/'ranks.npz',users=np.asarray([s['user_id'] for s in samples]),**rr)
        result[d]=dict(metrics={k:X.metrics(v) for k,v in rr.items()},checkpoint=read(p/'checkpoint_checks.json'))
        write(p/'summary.json',result[d])
        files=[p/name for name in ('predictions.jsonl','ranks.npz','summary.json','smoke.json','runtime_checks.json','checkpoint_checks.json')]
        write(p/'output_manifest.json',dict(sha256={str(f.relative_to(ROOT)):sha(f) for f in files}))
    write(OUT/'summary.json',dict(domains=result,test_read=False,training_steps=0))
    REVIEW.run(global_ready=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','supervise','worker','generate'))
    parser.add_argument('--domain',choices=DOMAINS);parser.add_argument('--lease-pid',type=int);args=parser.parse_args()
    def interrupted(signum,frame):raise InterruptedError('Global alpha control interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        if args.mode=='prepare':prepare()
        elif args.mode=='supervise':
            X.R.B.OUT=OUT;X.R.B.CFG=CFG;X.R.B.MODULE=MODULE;X.R.B.REPORT=REVIEW.REPORT;X.R.B.summarize=summarize
            X.R.B.supervise()
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
