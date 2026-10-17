"""Three-seed one-epoch text heads and genuinely matched guided calibration."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
import torch

from experiment.phase21 import text_residual_screen as R
from experiment.phase21 import text_residual_prefix_screen as V
from experiment.phase21 import short_text_calibration_core as C
from experiment.phase20.catalog_scope_core import metrics, contrast

F = R.P.A.F
ROOT, SOURCE, PREFIX = R.ROOT, R.OUT, V.OUT
OUT = ROOT/'artifacts/phase21/short_text_calibration_1024_20261017_v1'
GUIDED = OUT/'guided_cache'
PLAN = ROOT/'plan/第二十一阶段/21.55_原容量短训练多种子与候选匹配校准_20261017.md'
REPORT = ROOT/'report/第二十一阶段/原容量短训练多种子与候选匹配校准_20261017.md'
MODULE = 'experiment.phase21.short_text_calibration'
DOMAINS, POOLS = R.DOMAINS, R.PROPOSALS
SEEDS = (0,1,2)
TRAIN = {kind+'_s'+str(seed):dict(seed=seed,objective=objective) for seed in SEEDS for kind,objective in
         (('residual','residual_hard'),('text','text_hard'),('random','residual_random'))}
JOBS = {d+'_cal':dict(domain=d) for d in DOMAINS}
EXTERNAL = (F.OUT,R.P.A.OUT,ROOT/'artifacts/phase21/single_switch_controls_1024_20261017_v1')
CFG = dict(users_per_domain=1024,seeds=list(SEEDS),epochs=1,rank=32,bound=2.,batch_size=128,
    lr=.001,weight_decay=.01,regularization=.01,strengths=[0.,.1,.25,.5,1.,2.],
    train_arms=TRAIN,source_indices=[1024,2048],alpha=.3,score_scale=20.,beta_single=1.,
    history_filter=False,history_penalty=False,text_self_exclusion=False,test_read=False,
    cpu_threads=2,max_parallel=1,global_project_gpu_cap=4,total_lease_mib=16384,
    expected_workload_peak_mib=12288,minimum_free_mib=18432,allocation_limit_gib=11,
    worker_timeout_seconds=3600,resource_wait_seconds=86400,scientific_auto_retry=False,
    calibration_policies=['catalog_cal','native_cal','guided_cal','fixed1'],
    primary='guided_cal residual rank32 epoch1; report all seeds without selecting one',
    automatic_next_experiment=False,bootstrap_repeats=2000)
read,write,sha,stamp = R.read,R.write,R.sha,R.stamp


def progress(domain,stage,**extra):
    write(OUT/domain/'status.json',dict(state='RUNNING',domain=domain,stage=stage,pid=os.getpid(),
        updated_at=stamp(),test_read=False,gpu_used=False,**extra))


def prepare():
    checks = tests()
    OUT.mkdir(exist_ok=False)
    write(OUT/'status.json',dict(state='PREPARING',updated_at=stamp(),test_read=False))
    write(OUT/'config.json',CFG)
    write(OUT/'tests.json',checks)
    shutil.copy2(PLAN,OUT/'protocol.md')
    GUIDED.mkdir()
    write(GUIDED/'config.json',CFG)
    expected = read(SOURCE/'manifest.json')['sha256']
    old_gpu = read(F.OUT/'manifest.json')['sha256']
    manifest,inputs = {},{}
    def use(path):
        rel = str(path.relative_to(ROOT))
        digest = manifest.get(rel) or sha(path)
        reference = expected.get(rel,old_gpu.get(rel))
        assert reference is None or reference==digest,rel
        manifest[rel] = digest
        return rel
    common = []
    for rel,digest in old_gpu.items():
        if rel.startswith(('experiment/','GRAM/src/','.cache/huggingface/')):
            assert sha(ROOT/rel)==digest,rel
            manifest[rel] = digest
            common.append(rel)
    for module in list(sys.modules.values()):
        name = getattr(module,'__file__',None)
        if name and name.endswith('.py'):
            path = Path(name).resolve()
            if ROOT in path.parents:
                common.append(use(path))
    common += [use(PLAN),use(OUT/'config.json'),use(GUIDED/'config.json')]
    for domain in DOMAINS:
        folder = OUT/domain
        folder.mkdir()
        files = []
        for name in ('samples.json','references.json','fit_samples.json','folds.npy','fit_inputs.npy',
            'evaluation_inputs.npy','item_vectors.npz','proposals.npz','calibration_proposals.npz',
            'training_pools.npz','data_config.json'):
            source = SOURCE/domain/name
            files += [use(source)]
            (folder/name).symlink_to(source)
            files += [use(folder/name)]
        source = R.P.OUT/domain/'fit_candidates.npz'
        files += [use(source)]
        (folder/'native_calibration_pool.npz').symlink_to(source)
        files += [use(folder/'native_calibration_pool.npz')]
        parity = PREFIX/domain/'models/terminal_rank32.pt'
        digest = read(PREFIX/domain/'output_manifest.json')['sha256'][str(parity.relative_to(ROOT))]
        assert sha(parity)==digest
        files += [use(parity)]
        fits,dev = read(folder/'fit_samples.json'),read(folder/'samples.json')
        folds = np.load(folder/'folds.npy')
        assert len(dev)==1024 and {s['user_id'] for s in fits}.isdisjoint(s['user_id'] for s in dev)
        cal = np.flatnonzero(folds==0)
        cache = GUIDED/domain
        cache.mkdir()
        samples = [dict(fits[j],fit_index=int(j),validation_index=k) for k,j in enumerate(cal)]
        write(cache/'samples.json',samples)
        native = np.load(folder/'native_calibration_pool.npz')
        write(cache/'references.json',[dict(native=dict(ids=native['ids'][j].tolist(),scores=native['scores'][j].tolist())) for j in cal])
        for name,source in [('catalog.json',R.P.OUT/domain/'catalog.json'),
                            ('data_config.json',folder/'data_config.json'),
                            ('teacher_checks.json',R.P.OUT/domain/'teacher_checks.json')]:
            files += [use(source)]
            shutil.copy2(source,cache/name)
            files += [use(cache/name)]
        for name in ('samples.json','references.json'):
            files += [use(cache/name)]
        dc = read(folder/'data_config.json')
        files += [use(ROOT/dc['parent_checkpoint']),use(ROOT/dc['original_config'])]
        files += [use(R.P.FIT/domain/'features/train/experts.npy')]
        inputs[domain] = sorted(set(common+files))
        write(folder/'status.json',dict(state='PREPARED',updated_at=stamp(),test_read=False))
        job = OUT/'jobs'/(domain+'_cal')
        job.mkdir(parents=True)
        write(job/'status.json',dict(state='PREPARED',users_completed=0,users_total=len(cal),updated_at=stamp()))
    for rel in manifest:
        path = ROOT/rel
        if path.suffix in ('.py','.md','.sh'):
            dest = OUT/'sources'/rel
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,dest)
    write(OUT/'manifest.json',dict(sha256=manifest,domain_inputs=inputs))
    write(OUT/'status.json',dict(state='PREPARED',updated_at=stamp(),users_total=2048,test_read=False))
    REPORT.write_text('# 原容量短训练多种子与候选匹配校准\n\n状态：PREPARED；guided_cal尚未完成。\n')
    print(json.dumps(dict(state='PREPARED',hashes=len(manifest),tests=checks)),flush=True)


def fit_models(folder,parity_path=None):
    x,folds = np.load(folder/'fit_inputs.npy'),np.load(folder/'folds.npy')
    fit = np.flatnonzero(folds!=0)
    mean,std = x[folds>=2].mean(0),np.maximum(x[folds>=2].std(0),.01)
    xt = torch.from_numpy(((x-mean)/std).astype(np.float32))
    bank = np.load(folder/'item_vectors.npz')['bank']
    item = torch.from_numpy(bank)
    pools = dict(np.load(folder/'training_pools.npz'))
    (folder/'models').mkdir()
    result = {}
    for name,spec in TRAIN.items():
        torch.manual_seed(spec['seed'])
        model = R.C.TextAdapter(x.shape[1],bank.shape[1],32,2.)
        optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
        kind = 'random' if spec['objective']=='residual_random' else 'hard'
        ids,valid,sas = [torch.from_numpy(pools[kind+'_'+k]-(1 if k=='ids' else 0))
                         if k!='valid' else torch.from_numpy(pools[kind+'_'+k]) for k in ('ids','valid','sas')]
        order = np.random.default_rng(spec['seed']).permutation(fit)
        total = 0.
        for start in range(0,len(order),128):
            jj = order[start:start+128]
            optimizer.zero_grad()
            value = R.C.objective(model(xt[jj],item[ids[jj]]),sas[jj],valid[jj],spec['objective'],.01)
            assert torch.isfinite(value)
            value.backward()
            optimizer.step()
            total += float(value)*len(jj)
        model.eval().requires_grad_(False)
        assert model.item.abs().sum()>0
        info = dict(seed=spec['seed'],objective=spec['objective'],epochs=1,rank=32,
                    parameters=sum(v.numel() for v in model.parameters()),users=len(fit),
                    train_loss=total/len(fit),calibration_users=int((folds==0).sum()))
        torch.save(dict(state_dict=model.state_dict(),mean=mean,std=std,info=info),folder/'models'/(name+'.pt'))
        np.save(folder/('training_order_'+name+'.npy'),order)
        result[name] = info
    if parity_path:
        actual = torch.load(folder/'models/residual_s0.pt',map_location='cpu')['state_dict']
        expected = torch.load(parity_path,map_location='cpu')['state_dict']
        for key in actual:
            torch.testing.assert_close(actual[key],expected[key],atol=0,rtol=0)
    write(folder/'training.json',dict(models=result,epoch_fixed_before_new_run=True,
        calibration_labels_used_for_fit=False,seed0_old_model_exact=bool(parity_path),test_read=False))


def load_models(folder):
    bank = np.load(folder/'item_vectors.npz')['bank']
    models = {}
    for name in TRAIN:
        saved = torch.load(folder/'models'/(name+'.pt'),map_location='cpu')
        model = R.C.TextAdapter(len(saved['mean']),bank.shape[1],32,2.)
        model.load_state_dict(saved['state_dict'])
        model.eval().requires_grad_(False)
        models[name] = (model,saved['mean'],saved['std'])
    return models,bank


def signals(models,bank,x,ids,bge,samples=None,domain=None):
    result = {'bge':2*bge-1}
    for name,(model,mean,std) in models.items():
        value = R.predict(model,x,bank,ids,mean,std)/2
        state = {k:v.numpy() for k,v in model.state_dict().items()}
        jj = np.array(sorted(set((0,len(x)//2,len(x)-1))))
        check = R.C.numpy_delta(state,((x[jj]-mean)/std).astype(np.float32),bank[ids[jj]-1])/2
        np.testing.assert_allclose(value[jj],check,atol=2e-6,rtol=2e-5)
        result[name] = value
        if samples is not None and name.startswith('residual_'):
            donor = R.P.C.history_donors(samples,domain)
            result[name+'_shuffled'] = R.predict(model,x[donor],bank,ids,mean,std)/2
    return result


def calibrate(folder,models,bank,policies):
    folds = np.load(folder/'folds.npy')
    cal = np.flatnonzero(folds==0)
    x = np.load(folder/'fit_inputs.npy')[cal]
    selections,grids = {},{}
    for policy in policies:
        if policy=='catalog_cal':
            packed = np.load(folder/'calibration_proposals.npz')
            ids,base,bge,targets = packed['ids'][cal],20*packed['sas'][cal],packed['bge'][cal],packed['targets'][cal]
        elif policy=='native_cal':
            packed = np.load(folder/'native_calibration_pool.npz')
            ids,base,bge,targets = [packed[k][cal] for k in ('ids','scores','bge','targets')]
        else:
            packed = np.load(folder/'guided_calibration_pool.npz')
            ids,base,bge,targets = [packed[k] for k in ('ids','scores','bge','targets')]
            np.testing.assert_array_equal(packed['fit_indices'],cal)
        values = signals(models,bank,x,ids,bge)
        selections[policy],grids[policy] = {},{}
        for name,value in values.items():
            gamma,grid = C.choose(ids,base,value,targets,CFG['strengths'])
            selections[policy][name],grids[policy][name] = gamma,grid
    return dict(selected=selections,grids=grids,calibration_users=len(cal),
                same_users_across_policies=True,evaluation_targets_used_for_calibration=False)


def evaluate(domain,folder,models,bank,selection,stage,verify_old=True):
    samples = read(folder/'samples.json')
    targets = np.array([s['target'] for s in samples])
    x = np.load(folder/'evaluation_inputs.npy')
    proposals = dict(np.load(folder/'proposals.npz'))
    scores,orders,rr = {},{},{}
    diagnostics = {}
    policies = ('fixed1',*selection['selected'])
    for pool in POOLS:
        ids,base,bge = [proposals[pool+'_'+key] for key in ('ids','scores','bge')]
        values = signals(models,bank,x,ids,bge,samples,domain)
        key = pool+'__off'
        scores[key],orders[key],rr[key] = C.prediction(ids,base,np.zeros(ids.shape),0.,targets)
        for name,value in values.items():
            for policy in policies:
                key = pool+'__'+name+'__'+policy
                source = name[:-len('_shuffled')] if name.endswith('_shuffled') else name
                gamma = 1. if policy=='fixed1' else selection['selected'][policy][source]
                scores[key],orders[key],rr[key] = C.prediction(ids,base,value,gamma,targets)
                np.testing.assert_array_equal(rr[key]<=50,rr[pool+'__off']<=50)
                diagnostics[key] = dict(strength=gamma,mean_absolute_signal=float(np.abs(value).mean()),
                    modified_orders=int((orders[key]!=orders[pool+'__off']).any(1).sum()))
        progress(domain,'evaluate_'+stage,pool=pool,methods=len(rr))
    refs = read(folder/'references.json')
    for name in ('native','sas_only','bge_only','pair_raw'):
        rr['reference__'+name] = np.array([R.P.A.B.rank(row[name],s['target']) for row,s in zip(refs,samples)])
    np.testing.assert_array_equal(rr['gram_sas__off'],rr['reference__sas_only'])
    np.testing.assert_array_equal(rr['gram_native__off'],rr['reference__native'])
    # Confirm the former weak signal is replayed rather than silently changed.
    if stage=='cached' and verify_old:
        old = np.load(PREFIX/domain/'ranks.npz')
        np.testing.assert_array_equal(rr['gram_sas__residual_s0__fixed1'],old['gram_sas__terminal_rank32_fixed1'])
    np.savez_compressed(folder/(stage+'_scores.npz'),**scores)
    np.savez_compressed(folder/(stage+'_orders.npz'),**orders)
    np.savez_compressed(folder/(stage+'_ranks.npz'),users=np.array([s['user_id'] for s in samples]),**rr)
    write(folder/(stage+'_diagnostics.json'),diagnostics)
    write(folder/(stage+'_selection.json'),selection)
    write(folder/(stage+'_checks.json'),dict(state='PASSED',users=len(samples),all_orders_and_ranks_independent=True,
        model_numpy_checks=True,baseline_replay_exact=True,history_filter=False,candidate_sets_unchanged=True,
        H50_unchanged=True,target_free_proposals=True,test_read=False))


def cpu_worker(domain):
    folder = OUT/domain
    signal.alarm(3600)
    verify_inputs(domain)
    progress(domain,'train_rank32_epoch1_three_seeds')
    fit_models(folder,PREFIX/domain/'models/terminal_rank32.pt')
    models,bank = load_models(folder)
    selection = calibrate(folder,models,bank,('catalog_cal','native_cal'))
    evaluate(domain,folder,models,bank,selection,'cached')
    paths = list((folder/'models').glob('*.pt'))+list(folder.glob('training*'))+list(folder.glob('cached_*'))
    write(folder/'cpu_output_manifest.json',dict(sha256={str(p.relative_to(ROOT)):sha(p) for p in paths if p.is_file()}))
    write(folder/'status.json',dict(state='CPU_COMPLETED',updated_at=stamp(),test_read=False))
    signal.alarm(0)


def verify_inputs(domain):
    manifest = read(OUT/'manifest.json')
    assert read(OUT/'config.json')==CFG
    for rel in manifest['domain_inputs'][domain]:
        assert sha(ROOT/rel)==manifest['sha256'][rel],rel


class Generation(F.B.CLEAN.Worker):
    def __init__(self,job,lease_pid):
        self.job,self.spec = job,JOBS[job]
        self.jobfolder = OUT/'jobs'/job
        self.lease = F.B.S.LeaseHandle(lease_pid)
        self.row_seconds,self.users_completed = [],0
        F.B.CLEAN.OUT = GUIDED
        super().__init__(self.spec['domain'])
        self.truth = self.samples
        from experiment.phase18.core.diff_data import DiffDataset
        self.samples = [dict(s,target=1) for s in self.truth]
        self.data = DiffDataset(self.samples,self.catalog)
        self.refs = read(self.folder/'references.json')
        self.raw = np.load(R.P.FIT/self.domain/'features/train/experts.npy',mmap_mode='r')
        assert self.initial_digest==read(self.folder/'teacher_checks.json')['model_digest']

    def context(self,j,override_index=None):
        return super().context(j,override_index=j)

    def status(self,state='RUNNING',**extra):
        assert self.lease.poll() is None
        done = self.users_completed
        count = len(self.truth) if hasattr(self,'truth') else len(read(GUIDED/self.spec['domain']/'samples.json'))
        rate = float(np.median(self.row_seconds[-32:])) if self.row_seconds else None
        write(self.jobfolder/'status.json',dict(state=state,stage=self.stage,job=self.job,
            domain=self.spec['domain'],pid=os.getpid(),updated_at=stamp(),users_completed=done,users_total=count,
            gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),median_seconds_per_user=rate,
            remaining_generation_seconds_estimate=(count-done)*rate if rate else None,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            alpha=.3,history_filter=False,test_read=False,**extra))

    def run(self):
        self.stage = 'matched_calibration_smoke'
        self.status()
        checks = []
        for j in (0,len(self.truth)//2,len(self.truth)-1):
            ctx = self.context(j)
            native = self.decode(ctx)
            F.B.Generation.equal(dict(ids=native['ids'],scores=native['scores'].tolist()),self.refs[j]['native'])
            q = self.index.conditional_logs(self.raw[self.truth[j]['fit_index'],:,0])
            zero = self.decode(ctx,q,0.)
            assert native['paths']==zero['paths']
            np.testing.assert_array_equal(native['scores'],zero['scores'])
            self.samples[j]['target']=2
            other = self.context(j)
            self.samples[j]['target']=1
            torch.testing.assert_close(ctx['ids'],other['ids'],atol=0,rtol=0)
            torch.testing.assert_close(ctx['mask'],other['mask'],atol=0,rtol=0)
            checks.append(dict(user=self.truth[j]['user_id'],native_replay=True,alpha_zero_exact=True,target_placeholder_invariant=True))
        write(self.jobfolder/'smoke.json',dict(state='PASSED',checks=checks))
        self.stage = 'generate_actual_SASRec_guided_calibration_pool'
        self.status()
        with (self.jobfolder/'predictions.jsonl').open('x') as stream:
            for j,s in enumerate(self.truth):
                began = time.monotonic()
                ctx = self.context(j)
                q = self.index.conditional_logs(self.raw[s['fit_index'],:,0],temperature_scale=20.)
                value = self.decode(ctx,q,.3)
                stream.write(json.dumps(dict(user_id=s['user_id'],fit_index=s['fit_index'],target=s['target'],
                    ids=value['ids'],scores=value['scores'].tolist()))+'\n')
                stream.flush()
                self.users_completed += 1
                self.row_seconds.append(time.monotonic()-began)
                if self.users_completed==1 or self.users_completed%8==0 or j==len(self.truth)-1:
                    os.fsync(stream.fileno())
                    self.status()
        assert self.digest()==self.initial_digest
        write(self.jobfolder/'runtime_checks.json',dict(state='PASSED',users=len(self.truth),
            gram_digest_unchanged=True,history_filter=False,target_labels_not_in_generation=True,test_read=False))
        self.stage = 'complete'
        self.status('COMPLETED')


def gpu_worker(job):
    F.OUT,F.CFG,F.MODULE,F.JOBS = OUT,CFG,MODULE,JOBS
    return F.worker(job)


def external_state():
    active,pending = [],[]
    for root in EXTERNAL:
        value = read(root/'status.json')
        if value.get('pending'):
            pending.append(str(root.relative_to(ROOT)))
        for job,task in value.get('active',{}).items():
            active.append(dict(source=str(root.relative_to(ROOT)),job=job,gpu_uuid=task['gpu_uuid']))
    return active,pending


def finalize_guided():
    for domain in DOMAINS:
        folder = OUT/domain
        job = OUT/'jobs'/(domain+'_cal')
        assert read(job/'status.json')['state']=='COMPLETED'
        assert read(job/'runtime_checks.json')['state']=='PASSED'
        assert read(job/'lease_status.json')['state']=='released'
        for rel,digest in read(folder/'cpu_output_manifest.json')['sha256'].items():
            assert sha(ROOT/rel)==digest,rel
        rows = [json.loads(line) for line in (job/'predictions.jsonl').read_text().splitlines()]
        samples = read(GUIDED/domain/'samples.json')
        assert len(rows)==len(samples) and len({r['user_id'] for r in rows})==len(rows)
        for row,s in zip(rows,samples):
            assert all(row[k]==s[k] for k in ('user_id','fit_index','target'))
            assert len(row['ids'])==len(set(row['ids']))==50
        raw = np.load(R.P.FIT/domain/'features/train/experts.npy',mmap_mode='r')
        np.savez_compressed(folder/'guided_calibration_pool.npz',
            ids=np.array([r['ids'] for r in rows]),scores=np.array([r['scores'] for r in rows]),
            targets=np.array([r['target'] for r in rows]),fit_indices=np.array([r['fit_index'] for r in rows]),
            bge=np.stack([raw[r['fit_index'],np.array(r['ids'])-1,3] for r in rows]))
        models,bank = load_models(folder)
        selection = calibrate(folder,models,bank,('catalog_cal','native_cal','guided_cal'))
        evaluate(domain,folder,models,bank,selection,'final')
        write(folder/'final_output_manifest.json',dict(sha256={str(p.relative_to(ROOT)):sha(p)
            for p in list(folder.glob('final_*'))+[folder/'guided_calibration_pool.npz'] if p.is_file() and p.name!='final_output_manifest.json'}))
        write(folder/'status.json',dict(state='COMPLETED',users_completed=1024,updated_at=stamp(),test_read=False))


def summarize(stage):
    domains = {}
    policies = ('fixed1','catalog_cal','native_cal')+(('guided_cal',) if stage=='final' else ())
    lines = ['# 原容量短训练多种子与候选匹配校准','','## Material Passport','',
        '- Origin Skill: academic-research-suite / experiment-agent','- Origin Mode: run / validate',
        '- Origin Date: 2026-10-17','- Verification Status: ANALYZED, reused development users','',
        '阶段：'+('COMPLETED，包括真实SASRec引导候选校准。' if stage=='final' else 'PRELIMINARY，仅缓存校准；真实guided_cal仍排队。'),
        'rank32、1轮、三个种子全部报告；均值是指标汇总，不是集成模型。候选固定，不读test。','']
    for domain in DOMAINS:
        folder = OUT/domain
        rr = dict(np.load(folder/(stage+'_ranks.npz')))
        users = rr.pop('users')
        samples = read(folder/'samples.json')
        assert users.tolist()==[s['user_id'] for s in samples]
        groups = dict(all1024=np.arange(1024),first512=np.arange(512),second512=np.arange(512,1024))
        blocks = np.array([s['evaluation_block'] for s in samples])
        groups.update({f'block{b+1}':np.flatnonzero(blocks==b) for b in range(5)})
        domains[domain] = {}
        for group,ix in groups.items():
            entry = dict(users=len(ix),metrics={k:metrics(v[ix]) for k,v in rr.items()})
            entry['residual_seed_summary'] = {policy:dict(
                N10_mean=float(np.mean([entry['metrics'][f'gram_sas__residual_s{s}__{policy}']['N10'] for s in SEEDS])),
                N10_min=float(min(entry['metrics'][f'gram_sas__residual_s{s}__{policy}']['N10'] for s in SEEDS)),
                N10_max=float(max(entry['metrics'][f'gram_sas__residual_s{s}__{policy}']['N10'] for s in SEEDS))) for policy in policies}
            if group in ('all1024','first512','second512'):
                pairs = []
                for seed in SEEDS:
                    for policy in policies:
                        a = f'gram_sas__residual_s{seed}__{policy}'
                        pairs += [(a,'gram_sas__off')]+[(a,f'gram_sas__{name}__{policy}') for name in
                            (f'text_s{seed}',f'random_s{seed}',f'residual_s{seed}_shuffled','bge')]
                        if policy!='catalog_cal':
                            pairs.append((a,f'gram_sas__residual_s{seed}__catalog_cal'))
                entry['comparisons'] = {a+'_vs_'+b:contrast(rr[a][ix],rr[b][ix],2000,20261017) for a,b in pairs}
            domains[domain][group] = entry
        write(folder/(stage+'_summary.json'),domains[domain])
        lines += [f'## {domain}','','| 校准方式 | seed0 N10 | seed1 N10 | seed2 N10 | 均值 |','|---|---:|---:|---:|---:|']
        mm = domains[domain]['all1024']['metrics']
        for policy in policies:
            vv = [mm[f'gram_sas__residual_s{s}__{policy}']['N10'] for s in SEEDS]
            lines.append(f'| {policy} | {vv[0]:.6f} | {vv[1]:.6f} | {vv[2]:.6f} | {np.mean(vv):.6f} |')
        lines += ['',f'同用户GRAM+SAS原排序N10={mm["gram_sas__off"]["N10"]:.6f}。','']
    write(OUT/(stage+'_summary.json'),dict(stage=stage,domains=domains,config=CFG,test_read=False,
        statistical_review=read(SOURCE/'summary.json')['statistical_review'],
        intervals='Unadjusted descriptive paired-user intervals; three seeds are not independent users.',
        next_experiment_started=False))
    lines += ['原生GRAM校准是对照；只有guided_cal与SASRec引导候选协议匹配。重复开发与多重比较、单个上游模型等局限保留。',
              '完整分数和顺序以NPZ保存，按proposals中的ID矩阵与orders重建全部50项预测；不得选最好种子宣布成功。']
    target = REPORT if stage=='final' else OUT/'cached_report.md'
    target.write_text('\n'.join(lines)+'\n')


def supervise():
    assert read(OUT/'status.json')['state']=='PREPARED'
    started = time.monotonic()
    for domain in DOMAINS:
        with (OUT/domain/'cpu_worker.log').open('x') as stream:
            p = subprocess.Popen([sys.executable,'-u','-m',MODULE,'cpu-worker','--domain',domain],cwd=ROOT,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            while p.poll() is None:
                write(OUT/'status.json',dict(state='CPU_RUNNING',domain=domain,pid=os.getpid(),worker_pid=p.pid,
                    worker=read(OUT/domain/'status.json'),updated_at=stamp(),elapsed_seconds=time.monotonic()-started,test_read=False))
                time.sleep(5)
        assert p.returncode==0 and read(OUT/domain/'status.json')['state']=='CPU_COMPLETED','CPU scientific failure; no retry'
    summarize('cached')
    pending,finished,active = list(JOBS),{},None
    attempts = {j:0 for j in JOBS}
    waiting = time.monotonic()
    while pending or active:
        if active and active['process'].poll() is not None:
            code = active['process'].returncode
            active['log'].close()
            job = active['job']
            state = read(OUT/'jobs'/job/'status.json')['state']
            if code==75 and state=='ADMISSION_DEFERRED':
                pending.append(job)
            else:
                finished[job] = dict(exit_code=code,state=state)
                assert code==0 and state=='COMPLETED','Guided calibration scientific failure; no retry'
            active = None
        external,external_pending = external_state()
        if pending and active is None and not external_pending and len(external)<4:
            excluded = {t['gpu_uuid'] for t in external}
            available = sorted([g for g in F.B.S.OLD.gpu_inventory()
                if g['uuid'] not in excluded and g['free_mib']>=18432],
                key=lambda g:(g['utilization'],-g['free_mib'],g['index']))
            if available:
                gpu,job = available[0],pending.pop(0)
                attempts[job]+=1
                log = (OUT/'jobs'/job/f'worker_attempt{attempts[job]}.log').open('x')
                env = dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                    OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
                process = subprocess.Popen([sys.executable,'-u','-m',MODULE,'gpu-worker','--job',job],cwd=ROOT,
                    env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active = dict(job=job,gpu=gpu,process=process,log=log)
        assert time.monotonic()-waiting<86400,'Resource wait limit'
        write(OUT/'status.json',dict(state='GUIDED_CALIBRATION_RUNNING' if active else 'WAITING_FOR_GPU_SLOT' if pending else 'FINALIZING',
            updated_at=stamp(),pid=os.getpid(),cached_results='cached_report.md',pending=pending,finished=finished,
            external_active=external,external_pending=external_pending,
            active={active['job']:dict(pid=active['process'].pid,gpu_uuid=active['gpu']['uuid'],gpu_index=active['gpu']['index'])} if active else {},
            elapsed_seconds=time.monotonic()-started,test_read=False))
        if pending or active:
            time.sleep(10)
    finalize_guided()
    summarize('final')
    write(OUT/'status.json',dict(state='COMPLETED',updated_at=stamp(),users_completed=2048,users_total=2048,
        elapsed_seconds=time.monotonic()-started,finished=finished,report=str(REPORT.relative_to(ROOT)),
        test_read=False,next_experiment_started=False))


def tests():
    result = dict(calibration=C.tests(),adapter=R.C.tests())
    with tempfile.TemporaryDirectory(prefix='short-text-test-') as tmp:
        folder = Path(tmp)
        rng = np.random.default_rng(55)
        x = rng.normal(size=(15,7)).astype(np.float32)
        np.save(folder/'fit_inputs.npy',x)
        np.save(folder/'folds.npy',np.arange(15)%5)
        np.savez(folder/'item_vectors.npz',bank=rng.normal(size=(60,5)).astype(np.float32))
        ids,valid = zip(*(R.C.training_pool(np.arange(1,51),1+j) for j in range(15)))
        ids,valid = np.array(ids),np.array(valid)
        sas = rng.random(ids.shape).astype(np.float32)
        np.savez(folder/'training_pools.npz',hard_ids=ids,hard_valid=valid,hard_sas=sas,
                 random_ids=ids,random_valid=valid,random_sas=sas)
        fit_models(folder)
        info = read(folder/'training.json')
        assert len(info['models'])==9 and all(v['epochs']==1 for v in info['models'].values())
        for seed in SEEDS:
            orders = [np.load(folder/f'training_order_{kind}_s{seed}.npy') for kind in ('residual','text','random')]
            for order in orders[1:]:
                np.testing.assert_array_equal(order,orders[0])
        result['three_seeds_three_objectives_fixture'] = 'PASSED'
        result['same_seed_optimizer_order_matched'] = True
        ids = np.tile(np.arange(1,51),(15,1))
        targets = np.arange(1,16)
        np.savez(folder/'calibration_proposals.npz',ids=ids,sas=rng.random((15,50)).astype(np.float32),
                 bge=rng.random((15,50)).astype(np.float32),targets=targets)
        np.savez(folder/'native_calibration_pool.npz',ids=ids,scores=rng.random((15,50)),
                 bge=rng.random((15,50)).astype(np.float32),targets=targets)
        samples = [dict(user_id=f'u{j}',target=1+j%60,full_history=[1,2,3],history=[1,2,3],
                        evaluation_block=j%5) for j in range(64)]
        write(folder/'samples.json',samples)
        np.save(folder/'evaluation_inputs.npy',rng.normal(size=(64,7)).astype(np.float32))
        ids = np.tile(np.arange(1,51),(64,1))
        scores = np.tile(np.linspace(1,0,50),(64,1))
        proposals = {}
        for pool in POOLS:
            proposals.update({pool+'_ids':ids,pool+'_scores':scores,
                              pool+'_bge':rng.random((64,50)).astype(np.float32)})
        np.savez(folder/'proposals.npz',**proposals)
        reference = dict(ids=ids[0].tolist(),scores=scores[0].tolist())
        write(folder/'references.json',[dict(native=reference,sas_only=reference,bge_only=reference,pair_raw=reference)
                                        for _ in samples])
        models,bank = load_models(folder)
        selection = calibrate(folder,models,bank,('catalog_cal','native_cal'))
        old_progress = globals()['progress']
        globals()['progress'] = lambda *a,**kw:None
        try:
            evaluate('synthetic',folder,models,bank,selection,'cached',verify_old=False)
        finally:
            globals()['progress'] = old_progress
        assert read(folder/'cached_checks.json')['state']=='PASSED'
        packed = np.load(folder/'cached_orders.npz')
        assert len(packed)==120
        result['all_cached_policies_and_controls_fixture'] = 'PASSED'
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=('test','prepare','supervise','cpu-worker','gpu-worker','generate'))
    parser.add_argument('--domain',choices=DOMAINS)
    parser.add_argument('--job',choices=JOBS)
    parser.add_argument('--lease-pid',type=int)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    def interrupted(signum,frame):
        raise InterruptedError('Short text calibration interrupted or hard timeout')
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGALRM,interrupted)
    try:
        if args.mode=='test':
            print(json.dumps(tests()),flush=True)
        elif args.mode=='prepare':
            prepare()
        elif args.mode=='supervise':
            supervise()
        elif args.mode=='cpu-worker':
            cpu_worker(args.domain)
        elif args.mode=='gpu-worker':
            return gpu_worker(args.job)
        else:
            Generation(args.job,args.lease_pid).run()
    except BaseException as exc:
        if OUT.exists():
            folder = OUT/'jobs'/args.job if args.job else OUT/args.domain if args.domain else OUT
            value = dict(state='FAILED',error=repr(exc),traceback=traceback.format_exc(),updated_at=stamp(),test_read=False)
            write(folder/'failure.json',value)
            write(folder/'status.json',value)
        raise
    return 0


if __name__=='__main__':
    sys.exit(main())
