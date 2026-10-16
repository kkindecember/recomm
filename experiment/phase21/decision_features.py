"""Matched old10 vs observable decision7 features, frozen TiSASRec/BGE/GRAM.

Only the user-level fuser is fitted. Generation guidance remains active at
every prefix; start-of-generation distributions are used only as gate inputs.
"""
import argparse
import gc
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

from experiment.phase21 import single_fusion as S
from experiment.phase21 import joint_fusion as J
from experiment.phase21 import joint_fusion_core as JC
from experiment.phase21 import decision_features_core as C
from experiment.phase21.candidate_gate_core import fold
from experiment.phase20.catalog_scope_core import metrics, contrast

ROOT = S.ROOT
PRIOR = ROOT/'artifacts/phase21/joint_fusion_20261015_v2'
TIME = ROOT/'artifacts/phase21/time_behavior_expert_20261015_v1'
OUT = ROOT/'artifacts/phase21/decision_features_20261016_v1'
PLAN = ROOT/'plan/第二十一阶段/21.36_生成决策统计特征匹配对照_20261016.md'
REPORT = ROOT/'report/第二十一阶段/生成决策统计特征匹配对照_20261016.md'
MODULE = 'experiment.phase21.decision_features'
DOMAINS, ARMS = ('Toys','Beauty'), ('old10','decision7')
read, write, sha, stamp = S.read, S.write, S.sha, S.stamp
LEASE = None
CFG = dict(seed=0, epochs=20, checkpoints=[0,4,8,12,20], batch_size=32, lr=.002,
    weight_decay=.001, score_scale=20., alpha_initial=.3, beam_size=50, length_penalty=1.,
    evaluation_users=512, worker_timeout_seconds=21600, resource_wait_seconds=86400,
    minimum_free_mib=18432, total_lease_mib=16384, expected_workload_peak_mib=12288,
    allocation_limit_gib=11, max_parallel=2, gpu_pool='all GPUs, lowest utilization first',
    resource_authorization='User preference: reduce previous 30 GiB reserve; retain 16 GiB per domain',
    experts=['TiSASRec','BGE'], arms=list(ARMS), hidden_width=32,
    features=dict(old10=S.C.USER_NAMES, decision7=C.NAMES), parameters=dict(old10=451,decision7=355),
    feature_space='same legal first-token branches; no target prefix',
    entropy='natural-log entropy / log(branch count)', js='JS / log(2), not sqrt(JS)',
    gate_scope='one beta/alpha per current user context shared by all items and generation prefixes',
    guide_scope='existing conditional expert mass at EVERY generation prefix',
    loss='full catalog expert CE parent, then mean length-normalized target-path NLL',
    epoch_selection='same user-hash80/20; CE and NLL checkpoints0/4/8/12/20; refit all ranker_fit',
    parent_initialization='existing C.Fusion prototype first d columns; identical scale and common random prefix',
    old_parent='reuse verified full and fit-only CE parents with original CE selection',
    test_read=False, experts_frozen=True, gram_optimizer_steps=0,
    scientific_auto_retry=False, stop_after='two arms x two reused development512 pools, seed0')


def progress(domain, stage, **values):
    if LEASE is not None:
        assert LEASE.poll() is None, 'GPU lease exited'
    value = dict(state='RUNNING', stage=stage, pid=os.getpid(), updated_at=stamp(),
        gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'), test_read=False, **values)
    write(OUT/domain/'status.json', value)
    print(json.dumps(dict(domain=domain, **value)), flush=True)


def bind():
    S.OUT, S.REPORT, S.MODULE, S.CFG = OUT, REPORT, MODULE, CFG
    S.progress = progress


def feature_source(domain, part):
    return TIME/domain/'features/tisas'/part


def prepare():
    checks = C.tests()
    assert read(PRIOR/'status.json')['state'] == 'COMPLETED'
    OUT.mkdir(exist_ok=False)
    write(OUT/'status.json', dict(state='PREPARING', updated_at=stamp()))
    write(OUT/'config.json', CFG); write(OUT/'tests.json', checks)
    shutil.copy2(PLAN, OUT/'protocol.md')
    manifest = {}
    previous = read(PRIOR/'manifest.json')['sha256']
    def use(path, expected=None):
        key = str(path.relative_to(ROOT))
        digest = manifest.get(key) or sha(path)
        assert expected is None or digest == expected, key
        manifest[key] = digest
    # Pin the actual reused code/checkpoint/tokenizer dependencies, not unrelated
    # Qwen caches or historical resource overrides.
    for rel, digest in previous.items():
        if rel.startswith(('experiment/', 'GRAM/src/', '.cache/huggingface/')):
            use(ROOT/rel, digest)
    for domain in DOMAINS:
        p = OUT/domain; p.mkdir()
        for name in ('samples.json','fit_samples.json','catalog.json','data_config.json','teacher.npz','teacher_checks.json'):
            src = PRIOR/domain/name
            use(src); shutil.copy2(src, p/name); use(p/name)
        fit, dev = read(p/'fit_samples.json'), read(p/'samples.json')
        assert len(dev) == len({s['user_id'] for s in dev}) == 512
        assert {s['user_id'] for s in fit}.isdisjoint(s['user_id'] for s in dev)
        old_train = read(PRIOR/domain/'training_manifest.json')['sha256']
        old_outputs = read(PRIOR/domain/'output_manifest.json')['sha256']
        for src in (PRIOR/domain/'parents/tisas/fold.pt', PRIOR/domain/'tisas_user.pt',
                    PRIOR/domain/'signals.npz', PRIOR/domain/'alphas.npz', PRIOR/domain/'betas.npz',
                    PRIOR/domain/'predictions.jsonl', PRIOR/domain/'ranks.npz', PRIOR/domain/'training.json'):
            rel = str(src.relative_to(ROOT))
            use(src, old_train.get(rel, old_outputs.get(rel)))
        assert sha(p/'teacher.npz') == read(p/'teacher_checks.json')['sha256']
        use(TIME/'models/tisas'/domain/'user.pt', previous[str((TIME/'models/tisas'/domain/'user.pt').relative_to(ROOT))])
        for part, samples in (('train',fit),('validation',dev)):
            src = feature_source(domain,part)
            meta = read(src/'metadata.json')
            assert meta['users'] == len(samples) and meta['variant'] == 'tisas'
            history_path = TIME/domain/('fit_histories.npz' if part == 'train' else 'validation_histories.npz')
            use(history_path, previous.get(str(history_path.relative_to(ROOT))))
            with np.load(history_path) as histories:
                assert histories['users'].tolist() == [s['user_id'] for s in samples]
                for s,items in zip(samples,histories['items']):
                    assert items[items != 0].tolist() == s['full_history'][-50:]
            for name in ('user.npy','item.npy','metadata.json'):
                path = src/name; use(path,previous[str(path.relative_to(ROOT))])
            assert np.load(src/'user.npy',mmap_mode='r').shape == (len(samples),10)
        config = read(p/'data_config.json')
        use(ROOT/config['parent_checkpoint'],config['parent_sha256'])
        rows = [json.loads(x) for x in (PRIOR/domain/'predictions.jsonl').read_text().splitlines()]
        refs = []
        for sample,row in zip(dev,rows):
            assert all(sample[k] == row[k] for k in ('user_id','target','history'))
            refs.append(dict(user_id=sample['user_id'],native=row['native'],
                historical_guided=row['tisas_user_guided'], historical_late=row['tisas_user_late']))
        assert len(rows) == len(refs) == 512
        write(p/'references.json',refs); use(p/'references.json')
        write(p/'status.json',dict(state='PREPARED',users_total=512,updated_at=stamp()))
    for p in (Path(__file__), ROOT/'experiment/phase21/decision_features_core.py',
              ROOT/'experiment/phase21/run_decision_features.sh', PLAN, OUT/'config.json'):
        use(p)
    for rel in manifest:
        src = ROOT/rel
        if src.suffix in ('.py','.sh','.md'):
            dst = OUT/'sources'/rel; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
    write(OUT/'manifest.json',dict(sha256=manifest,created_at=stamp()))
    write(OUT/'status.json',dict(state='PREPARED',updated_at=stamp(),test_read=False))
    REPORT.write_text('# 生成决策统计特征匹配对照\n\n状态：PREPARED，尚无新效果结论。\n')
    print(json.dumps(dict(state='PREPARED',hashes=len(manifest),tests=checks)),flush=True)


class RootWorker(S.G.clean.Worker):
    def status(self,state='RUNNING',**kw):
        progress(self.domain,'root_'+self.stage,**kw)

    def context(self,j,override_index=None):
        return super().context(j,override_index=j)

    def distribution(self,ctx,tokens):
        from model.gram_t5_outputs import BaseModelOutputWithPastAndCrossAttentions
        with torch.no_grad():
            result = self.model(attention_mask=ctx['mask'].reshape(1,-1),
                encoder_outputs=BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=ctx['hidden']),
                decoder_input_ids=torch.zeros((1,1),dtype=torch.long,device=self.device),
                use_cache=False,return_dict=True)
            logs = result.logits[0,0].log_softmax(-1)
            selected = logs[torch.as_tensor(tokens,device=self.device)]
            mass = torch.logsumexp(selected,0)
            return (selected-mass).exp().cpu().double().numpy(), float(mass)


def cache(domain):
    from experiment.phase18.core.diff_data import DiffDataset
    S.G.clean.OUT = OUT
    job = RootWorker(domain)
    assert job.initial_digest == read(OUT/domain/'teacher_checks.json')['model_digest']
    tokens, groups = C.root_mapping(job.catalog['decoder_paths'])
    assert len(tokens) == {'Toys':30,'Beauty':108}[domain]
    checks = []
    begin = time.monotonic()
    for part, name in (('train','fit_samples.json'),('validation','samples.json')):
        samples = read(OUT/domain/name)
        # Remove the real target from all feature-extraction records before
        # constructing the legacy adapter/collator. Nothing selects a gold path.
        job.samples = [dict(s,target=1) for s in samples]
        job.data = DiffDataset(job.samples,job.catalog)
        e = np.load(feature_source(domain,part)/'item.npy',mmap_mode='r')
        n = len(samples)
        roots = np.empty((n,3,len(tokens)),np.float64)
        user = np.empty((n,7),np.float32)
        masses = np.empty(n,np.float64)
        ticks = []
        for j,sample in enumerate(samples):
            tick = time.monotonic()
            ctx = job.context(j)
            p,mass = job.distribution(ctx,tokens)
            b = C.expert_root(e[j,:,0],groups,len(tokens))
            t = C.expert_root(e[j,:,1],groups,len(tokens))
            roots[j], masses[j] = np.stack((p,b,t)), mass
            user[j] = C.features(len(sample['full_history']),p,b,t)
            if j in (0,n//2,n-1):
                for channel,actual in ((0,b),(1,t)):
                    reference = job.index.conditional_logs(e[j,:,channel])[(0,)]
                    np.testing.assert_allclose(actual,[np.exp(reference[int(x)]) for x in tokens],atol=2e-12,rtol=1e-10)
                native = job.decode(ctx)
                tt, lp, lm = native['guide'].trace[(0,)]
                order = [tt.index(int(x)) for x in tokens]
                np.testing.assert_allclose(p,np.exp(lp[order]-lm),atol=2e-6,rtol=2e-5)
                np.testing.assert_allclose(mass,lm,atol=2e-5,rtol=1e-5)
                if part == 'validation':
                    reference = read(OUT/domain/'references.json')[j]['native']
                    assert native['ids'] == reference['ids']
                # The target-independent feature path is tested by changing the
                # placeholder, not by using gold tokens as model inputs.
                job.samples[j]['target'] = 2
                other = job.context(j)
                torch.testing.assert_close(ctx['ids'],other['ids'],atol=0,rtol=0)
                torch.testing.assert_close(ctx['mask'],other['mask'],atol=0,rtol=0)
                pp,mm = job.distribution(other,tokens)
                np.testing.assert_array_equal(p,pp); assert mass == mm
                job.samples[j]['target'] = 1
                checks.append(dict(part=part,row=j,root_matches_beam=True,expert_matches_trie=True,
                    placeholder_target_invariant=True))
                del other,native
            del ctx
            ticks.append(time.monotonic()-tick)
            if j == 0 or (j+1)%64 == 0 or j+1 == n:
                progress(domain,'cache_root_features',part=part,completed=j+1,total=n,
                    elapsed_seconds=time.monotonic()-begin,
                    remaining_part_seconds_estimate=float(np.median(ticks[-64:])*(n-j-1)))
        dest = OUT/domain/'features'/part; dest.mkdir(parents=True)
        np.save(dest/'decision7.npy',user)
        np.savez_compressed(dest/'root_probabilities.npz',probabilities=roots,legal_log_mass=masses,
            tokens=tokens,users=np.asarray([s['user_id'] for s in samples]))
        np.testing.assert_allclose(user[:,0],np.load(feature_source(domain,part)/'user.npy')[:,0],atol=0,rtol=0)
        write(dest/'metadata.json',dict(users_order=[s['user_id'] for s in samples],features=C.NAMES,
            branches=len(tokens),target_used_in_features=False,decoder_input=[0],
            feature_min=user.min(0).tolist(),feature_max=user.max(0).tolist(),feature_std=user.std(0).tolist()))
    assert job.digest() == job.initial_digest
    write(OUT/domain/'feature_checks.json',dict(state='PASSED',checks=checks,
        frozen_gram_digest=job.initial_digest,target_used_in_features=False,elapsed_seconds=time.monotonic()-begin))
    paths = list((OUT/domain/'features').glob('*/*'))+[OUT/domain/'feature_checks.json']
    write(OUT/domain/'feature_manifest.json',dict(sha256={str(p.relative_to(ROOT)):sha(p) for p in paths}))


def fit(domain):
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    device = torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(11*2**30/torch.cuda.get_device_properties(0).total_memory)
    p = OUT/domain
    for rel,digest in read(p/'feature_manifest.json')['sha256'].items():
        assert sha(ROOT/rel) == digest,rel
    samples,dev = read(p/'fit_samples.json'),read(p/'samples.json')
    fi = np.asarray([j for j,s in enumerate(samples) if fold(s['user_id'],domain) != 0])
    hi = np.asarray([j for j,s in enumerate(samples) if fold(s['user_id'],domain) == 0])
    all_ix = np.arange(len(samples))
    yt = torch.tensor([s['target']-1 for s in samples],device=device)
    teacher = np.load(p/'teacher.npz')
    assert sha(p/'teacher.npz') == read(p/'teacher_checks.json')['sha256']
    np.testing.assert_array_equal(teacher['users'],[s['user_id'] for s in samples])
    np.testing.assert_array_equal(teacher['targets'],yt.cpu().numpy())
    nt,mt = [torch.from_numpy(teacher[k]).to(device) for k in ('native','legal')]
    et = torch.from_numpy(np.ascontiguousarray(np.load(feature_source(domain,'train')/'item.npy',mmap_mode='r')[...,:2])).to(device)
    ve = np.ascontiguousarray(np.load(feature_source(domain,'validation')/'item.npy',mmap_mode='r')[...,:2])
    tree = JC.PathMass(read(p/'catalog.json')['decoder_paths'],device)
    exported,alphas,betas,info = {},{},{},{}

    for arm in ARMS:
        dimension = 10 if arm == 'old10' else 7
        def users(part):
            return np.load(feature_source(domain,part)/'user.npy' if arm == 'old10' else p/'features'/part/'decision7.npy')
        u,v = users('train'),users('validation')
        ut = torch.from_numpy(u).to(device)
        fn,an = J.user_norm(u,fi),J.user_norm(u,all_ix)

        def run(indices,norm,epochs,phase,objective,parent=None):
            torch.manual_seed(0); torch.cuda.manual_seed_all(0)
            model = C.Fusion(dimension,parent).to(device)
            if objective == 'ce': model.strength.requires_grad_(False)
            opt = torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=CFG['lr'],weight_decay=CFG['weight_decay'])
            rng = np.random.default_rng(0)
            normal = C.normalize(ut,*[torch.from_numpy(x).to(device) for x in norm])
            def loss(ix):
                beta,alpha = model(normal[ix])
                scores = (beta[:,None,:]*et[ix]).sum(-1)
                if objective == 'ce': return torch.nn.functional.cross_entropy(20*scores,yt[ix])
                return JC.path_nll(nt[ix],mt[ix],tree.logs(scores,yt[ix]),alpha,tree.valid[yt[ix]]).mean()
            curve,best_state,best = [],None,float('inf')
            for epoch in range(epochs+1):
                total = 0.
                if epoch:
                    order = rng.permutation(indices)
                    for start in range(0,len(order),32):
                        ix = order[start:start+32]
                        opt.zero_grad(); value = loss(ix)
                        assert torch.isfinite(value)
                        value.backward()
                        assert all(x.grad is None or torch.isfinite(x.grad).all() for x in model.parameters())
                        opt.step(); total += float(value)*len(ix)
                val = None
                if phase == 'selection' and epoch in CFG['checkpoints']:
                    with torch.no_grad():
                        val = sum(float(loss(hi[k:k+32]))*len(hi[k:k+32]) for k in range(0,len(hi),32))/len(hi)
                    curve.append(dict(epoch=epoch,loss=val))
                    if val < best:
                        best = val; best_state = {k:x.detach().cpu().clone() for k,x in model.state_dict().items()}
                progress(domain,objective+'_'+phase,arm=arm,epoch=epoch,epochs=epochs,
                    training_loss=total/len(indices) if epoch else None,internal_heldout_loss=val)
                write(p/(arm+'_'+objective+'_'+phase+'_curve.json'),dict(curve=curve,completed_epochs=epoch))
            if phase == 'selection':
                model.load_state_dict(best_state)
            return {k:x.detach().cpu().clone() for k,x in model.state_dict().items()},curve

        if arm == 'old10':
            parent = torch.load(TIME/'models/tisas'/domain/'user.pt',map_location='cpu')
            internal = torch.load(PRIOR/domain/'parents/tisas/fold.pt',map_location='cpu')
            assert parent['fit_users'] == [s['user_id'] for s in samples]
            assert internal['fit_users'] == [samples[j]['user_id'] for j in fi]
            for actual,expected in zip(an,(parent['mean'][:10],parent['std'][:10])):
                np.testing.assert_array_equal(actual,expected)
            for actual,expected in zip(fn,(internal['mean'],internal['std'])):
                np.testing.assert_array_equal(actual,expected)
            fold_state,full_state = internal['state_dict'],parent['state_dict']
            ce_epochs,ce_curve = parent['selected_epochs'],parent['selection_curve']
        else:
            fold_state,ce_curve = run(fi,fn,20,'selection','ce')
            ce_epochs = min(ce_curve,key=lambda x:(x['loss'],x['epoch']))['epoch']
            full_state,_ = run(all_ix,an,ce_epochs,'refit','ce')
        parent_dir = p/'parents'/arm; parent_dir.mkdir(parents=True)
        for name,state,norm,ix in (('fold',fold_state,fn,fi),('full',full_state,an,all_ix)):
            torch.save(dict(state_dict=state,mean=norm[0],std=norm[1],selected_epochs=ce_epochs,
                fit_users=[samples[j]['user_id'] for j in ix],selection_curve=ce_curve),parent_dir/(name+'.pt'))
        _,curve = run(fi,fn,20,'selection','nll',fold_state)
        epochs = min(curve,key=lambda x:(x['loss'],x['epoch']))['epoch']
        state,_ = run(all_ix,an,epochs,'refit','nll',full_state)
        ck = dict(state_dict=state,mean=an[0],std=an[1],dimension=dimension,mode='user',
            selected_epochs=epochs,parent_epochs=ce_epochs,selection_curve=curve,
            fit_users=[s['user_id'] for s in samples],config=CFG)
        torch.save(ck,p/(arm+'.pt'))
        model = C.Fusion(dimension).to(device); model.load_state_dict(state)
        with torch.no_grad():
            beta,alpha = model(C.normalize(torch.from_numpy(v).to(device),*[torch.from_numpy(x).to(device) for x in an]))
            bn,aa = beta.cpu().numpy(),alpha.cpu().double().numpy()
        nb,na = C.numpy_forward(torch.load(p/(arm+'.pt'),map_location='cpu')['state_dict'],v,*an)
        np.testing.assert_allclose(bn,nb,atol=2e-6,rtol=2e-5)
        np.testing.assert_allclose(aa,na,atol=2e-6,rtol=2e-5)
        scores = (bn[:,None,:]*ve).sum(-1)
        np.testing.assert_allclose(scores,(nb[:,None,:]*ve).sum(-1),atol=3e-6,rtol=2e-5)
        exported[arm],alphas[arm],betas[arm] = scores,aa,bn
        parity = None
        if arm == 'old10':
            old_s = np.load(PRIOR/domain/'signals.npz')['tisas_user']
            old_a = np.load(PRIOR/domain/'alphas.npz')['tisas_user']
            old_b = np.load(PRIOR/domain/'betas.npz')['tisas_user']
            assert epochs == read(PRIOR/domain/'training.json')['fits']['tisas_user']['selected_epochs']
            np.testing.assert_allclose(scores,old_s,atol=3e-6,rtol=2e-5)
            np.testing.assert_allclose(aa,old_a,atol=2e-6,rtol=2e-5)
            np.testing.assert_allclose(bn,old_b,atol=2e-6,rtol=2e-5)
            parity = dict(max_score_error=float(np.max(np.abs(scores-old_s))),max_alpha_error=float(np.max(np.abs(aa-old_a))))
        info[arm] = dict(dimension=dimension,parameters=sum(x.numel() for x in model.parameters()),
            parent_epochs=ce_epochs,selected_epochs=epochs,selection_curve=curve,parent_selection_curve=ce_curve,
            selection_fit_users=len(fi),heldout_users=len(hi),refit_users=len(samples),
            alpha_mean=float(aa.mean()),alpha_quantiles=np.quantile(aa,[0,.1,.5,.9,1]).tolist(),
            behavior_weight_mean=float(bn[:,0].mean()),numpy_verified_users=len(v),historical_parity=parity)
        del ut,model; gc.collect(); torch.cuda.empty_cache()
    np.savez(p/'signals.npz',**exported); np.savez(p/'alphas.npz',**alphas); np.savez(p/'betas.npz',**betas)
    write(p/'training.json',dict(fits=info,test_read=False,dev_used_for_selection=False,
        fit_dev_disjoint=True,internal_parent_weights_exclude_heldout_labels=True))
    paths = [p/n for n in ('signals.npz','alphas.npz','betas.npz','training.json','old10.pt','decision7.pt','feature_manifest.json')]
    paths += list((p/'parents').glob('*/*.pt'))+list(p.glob('*curve.json'))
    write(p/'training_manifest.json',dict(sha256={str(x.relative_to(ROOT)):sha(x) for x in paths}))


class Generation(S.Generation):
    def run(self):
        p = self.folder
        for rel,digest in read(p/'training_manifest.json')['sha256'].items():
            assert sha(ROOT/rel) == digest,rel
        self.alpha = dict(np.load(p/'alphas.npz'))
        self.stage = 'native_zero_historical_smoke'; self.status()
        checks = []
        for j in (0,256,511):
            ctx = self.context(j); native = self.decode(ctx)
            plain = self.decode(ctx,native_entry=True,record=False)
            q = self.index.conditional_logs(self.signals['decision7'][j])
            zero = self.decode(ctx,q,0.)
            for other in (plain,zero):
                assert native['paths'] == other['paths']
                np.testing.assert_array_equal(native['scores'],other['scores'])
            checks.append(dict(row=j,native_exact=True,alpha_zero_exact=True))
        write(p/'smoke.json',dict(state='PASSED',checks=checks))
        self.stage = 'matched_generation'; self.status()
        old_exact = 0
        with (p/'predictions.jsonl').open('x') as stream:
            for j,sample in enumerate(self.samples):
                begin = time.monotonic(); ctx = self.context(j); native = self.decode(ctx)
                ref = self.reference[sample['user_id']]
                assert native['ids'] == ref['native']['ids']
                np.testing.assert_allclose(native['scores'],ref['native']['scores'],atol=2e-6,rtol=1e-5)
                row = dict(user_id=sample['user_id'],target=sample['target'],history=sample['history'],
                    native=dict(ids=native['ids'],scores=native['scores'].tolist()),
                    historical_guided=ref['historical_guided'],historical_late=ref['historical_late'])
                for arm in ARMS:
                    a = float(self.alpha[arm][j])
                    q = self.index.conditional_logs(self.signals[arm][j],temperature_scale=20.)
                    late = native['guide'].rescore(native['paths'],q,a)
                    guided = self.decode(ctx,q,a)
                    row[arm+'_late'] = dict(ids=native['ids'],scores=late.tolist())
                    row[arm+'_guided'] = dict(ids=guided['ids'],scores=guided['scores'].tolist())
                    if arm == 'old10':
                        for suffix in ('guided','late'):
                            actual,expected = row[arm+'_'+suffix],ref['historical_'+suffix]
                            # Numerical parity is asserted; ties can permute
                            # beam presentation, so compare ID/score mappings.
                            assert set(actual['ids']) == set(expected['ids']),(j,suffix)
                            mapping = dict(zip(expected['ids'],expected['scores']))
                            np.testing.assert_allclose(actual['scores'],[mapping[x] for x in actual['ids']],atol=3e-6,rtol=2e-5)
                        old_exact += int(guided['ids'] == ref['historical_guided']['ids'])
                    del guided
                stream.write(json.dumps(row)+'\n'); stream.flush(); os.fsync(stream.fileno())
                self.users_completed = j+1; self.row_seconds.append(time.monotonic()-begin)
                if j == 0 or (j+1)%8 == 0 or j == 511: self.status(arms_per_user=2)
                del ctx,native
        assert self.digest() == self.initial_digest
        write(p/'runtime_checks.json',dict(state='PASSED',gram_model_digest_unchanged=True,
            model_digest=self.initial_digest,native_reproduced_users=512,old_guided_order_exact_users=old_exact,
            old_item_score_mappings_verified_users=512,gram_optimizer_steps=0,all_arms_generated=True))
        self.stage = 'complete'; self.status('COMPLETED')


def summarize():
    results = {}
    keys = ('native','historical_guided','historical_late')+tuple(a+s for a in ARMS for s in ('_guided','_late'))
    lines = ['# 生成决策统计特征匹配对照','', '状态：COMPLETED。两域各512历史开发用户，seed0；新结果为探索性证据。','',
        '| 数据集 | GRAM N10 | 原10维联合融合 N10 | 新7维联合融合 N10 | 新7维同规则后重排 N10 |',
        '|---|---:|---:|---:|---:|']
    for domain in DOMAINS:
        p = OUT/domain
        assert read(p/'status.json')['state'] == 'COMPLETED'
        assert read(p/'lease_status.json')['state'] == 'released'
        for name in ('feature_manifest.json','training_manifest.json'):
            for rel,digest in read(p/name)['sha256'].items(): assert sha(ROOT/rel) == digest,rel
        samples = read(p/'samples.json')
        rows = [json.loads(x) for x in (p/'predictions.jsonl').read_text().splitlines()]
        assert len(rows) == len(samples) == 512
        rr = {k:[] for k in keys}
        for row,sample in zip(rows,samples):
            assert all(row[k] == sample[k] for k in ('user_id','target','history'))
            for key in keys:
                value = row[key]; rank = S.OLD.record_rank(value,sample)
                ids,scores = np.asarray(value['ids']),np.asarray(value['scores'])
                keep = ~np.isin(ids,sample['history']); ids,scores = ids[keep],scores[keep]
                order = ids[np.lexsort((ids,-scores))].tolist()
                assert rank == (order.index(sample['target'])+1 if sample['target'] in order else 2**30)
                rr[key].append(rank)
        rr = {k:np.asarray(v) for k,v in rr.items()}
        old = np.load(PRIOR/domain/'ranks.npz')
        for a,b in (('native','native'),('historical_guided','tisas_user_guided'),('historical_late','tisas_user_late')):
            np.testing.assert_array_equal(rr[a],old[b])
        pairs = [('decision7_guided','old10_guided'),('decision7_guided','native'),
                 ('old10_guided','native'),('decision7_guided','decision7_late'),
                 ('old10_guided','old10_late'),('old10_guided','historical_guided')]
        result = dict(users=512,seed=0,metrics={k:metrics(v) for k,v in rr.items()},
            comparisons={a+'_vs_'+b:contrast(rr[a],rr[b],repeats=2000,seed=20261016) for a,b in pairs},
            training=read(p/'training.json'),test_read=False)
        np.savez_compressed(p/'ranks.npz',users=np.asarray([s['user_id'] for s in samples]),**rr)
        write(p/'summary.json',result); results[domain] = result
        write(p/'verification.json',dict(state='VERIFIED_SAVED_PREDICTIONS',users=512,
            independently_sorted_rank_arrays=len(keys),historical_rank_arrays_verified=3))
        paths = [p/n for n in ('predictions.jsonl','ranks.npz','summary.json','verification.json','runtime_checks.json','smoke.json')]
        write(p/'output_manifest.json',dict(sha256={str(x.relative_to(ROOT)):sha(x) for x in paths}))
        lines.append('| '+domain+' | '+' | '.join(f'{result["metrics"][k]["N10"]:.6f}' for k in
            ('native','old10_guided','decision7_guided','decision7_late'))+' |')
    write(OUT/'summary.json',dict(domains=results,test_read=False))
    lines += ['', '新特征仅观察生成开始时的首层分布；专家仍在所有生成步骤参与概率引导。',
        '两组隐藏层均为32，参数451/355；特征整体替换及输入维数变化，不能归因为某单项特征。',
        '旧CE父模型按原规则复用，新CE父模型按相同规则拟合；内部验证跨阶段复用。',
        '熵和JS是可观察的预测状态，不能直接称作真实可靠性或纠错收益。首层状态不覆盖后续路径。',
        '重复开发池、单seed、多个未经校正的描述性区间不构成正式确认；没有test访问。',
        '评分与引导分别沿用LIME-Rec衍生实现及AGRec启发的概率混合，应保留必要归属。',
        '本轮完成即停止；未开展每组特征消融、专家消融或整套论文实验。']
    REPORT.write_text('\n'.join(lines)+'\n')


def stop(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM)
        try: process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL); process.wait()


def worker(domain):
    p = OUT/domain; lease = child = None
    def interrupted(signum,frame): raise InterruptedError('Decision-feature worker interrupted/time limit')
    signal.signal(signal.SIGTERM,interrupted); signal.signal(signal.SIGALRM,interrupted)
    signal.alarm(CFG['worker_timeout_seconds'])
    try:
        assert read(OUT/'config.json') == CFG
        progress(domain,'verify_inputs')
        for rel,digest in read(OUT/'manifest.json')['sha256'].items(): assert sha(ROOT/rel) == digest,rel
        gpu = os.environ['CUDA_VISIBLE_DEVICES']
        free = next(g['free_mib'] for g in S.OLD.gpu_inventory() if g['uuid'] == gpu)
        if free < CFG['minimum_free_mib']:
            write(p/'status.json',dict(state='ADMISSION_DEFERRED',free_mib=free,
                required_free_mib=CFG['minimum_free_mib'],updated_at=stamp()))
            return 75
        with (p/'lease.log').open('a') as log:
            path = p/'lease_status.json'
            write(path,dict(state='starting'))
            lease = subprocess.Popen([sys.executable,str(ROOT/'experiment/gpu_memory_lease.py'),'--gpu','0',
                '--total-lease-mib',str(CFG['total_lease_mib']),
                '--expected-workload-peak-mib',str(CFG['expected_workload_peak_mib']),
                '--status-path',str(path)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(p/'lease_process.json',dict(pid=lease.pid,gpu_uuid=gpu,started_at=stamp()))
            start = time.monotonic()
            while True:
                try: holding = read(path).get('state') == 'holding'
                except json.JSONDecodeError: holding = False
                if holding: break
                assert lease.poll() is None and time.monotonic()-start < 180,'Lease startup failed'
                time.sleep(1)
            for phase in ('cache','fit','generate'):
                assert lease.poll() is None,'Lease exited'
                child = subprocess.Popen([sys.executable,'-u','-m',MODULE,phase,'--domain',domain,
                    '--lease-pid',str(lease.pid)],cwd=ROOT,start_new_session=True)
                code = child.wait(); assert code == 0,(phase,code)
        return 0
    except BaseException as exc:
        write(p/'status.json',dict(state='FAILED',error=repr(exc),traceback=traceback.format_exc(),updated_at=stamp()))
        raise
    finally:
        signal.alarm(0); stop(child); stop(lease)


def supervise():
    assert read(OUT/'status.json')['state'] == 'PREPARED'
    start = time.monotonic(); pending = list(DOMAINS); active = {}; finished = {}; attempts = dict.fromkeys(DOMAINS,0)
    try:
        while pending or active:
            for d,job in list(active.items()):
                code = job['process'].poll()
                if code is None and time.monotonic()-job['started'] > CFG['worker_timeout_seconds']+120:
                    stop(job['process']); code = job['process'].returncode
                if code is not None:
                    job['log'].close(); state = read(OUT/d/'status.json')
                    if code == 75 and state['state'] == 'ADMISSION_DEFERRED': pending.append(d)
                    else:
                        lease = OUT/d/'lease_status.json'
                        finished[d] = dict(exit_code=code,state=state['state'],
                            lease_released=lease.exists() and read(lease).get('state') == 'released')
                    del active[d]
            inventory = S.OLD.gpu_inventory()
            occupied = {j['gpu']['uuid'] for j in active.values()}
            available = sorted([g for g in inventory if g['uuid'] not in occupied and g['free_mib'] >= CFG['minimum_free_mib']],
                key=lambda g:(g['utilization'],-g['free_mib'],g['index']))
            for d in list(pending):
                if not available or len(active) >= CFG['max_parallel']: break
                gpu = available.pop(0); pending.remove(d); attempts[d] += 1
                log = (OUT/d/f'worker_attempt{attempts[d]}.log').open('x')
                env = dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                    MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
                write(OUT/d/'status.json',dict(state='STARTING',gpu_index=gpu['index'],gpu_uuid=gpu['uuid'],updated_at=stamp()))
                process = subprocess.Popen([sys.executable,'-u','-m',MODULE,'worker','--domain',d],cwd=ROOT,
                    env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[d] = dict(process=process,log=log,gpu=gpu,started=time.monotonic())
            if pending and time.monotonic()-start > CFG['resource_wait_seconds']:
                for d in pending:
                    write(OUT/d/'status.json',dict(state='RESOURCE_WAIT_LIMIT',updated_at=stamp()))
                    finished[d] = dict(exit_code=124,state='RESOURCE_WAIT_LIMIT',lease_released=True)
                pending = []
            write(OUT/'status.json',dict(state='RUNNING' if active else 'WAITING_FOR_RESOURCES' if pending else 'FINALIZING',
                pid=os.getpid(),updated_at=stamp(),elapsed_seconds=time.monotonic()-start,pending=pending,finished=finished,
                required_free_mib=CFG['minimum_free_mib'],total_lease_mib=CFG['total_lease_mib'],
                active={d:dict(pid=j['process'].pid,gpu_index=j['gpu']['index'],gpu_uuid=j['gpu']['uuid'],
                    worker=read(OUT/d/'status.json')) for d,j in active.items()},gpu_inventory=inventory,test_read=False))
            if pending or active: time.sleep(10)
        complete = len(finished) == 2 and all(j['exit_code'] == 0 and j['state'] == 'COMPLETED' and j['lease_released'] for j in finished.values())
        if complete: summarize()
        write(OUT/'status.json',dict(state='COMPLETED' if complete else 'INCOMPLETE',finished=finished,
            updated_at=stamp(),elapsed_seconds=time.monotonic()-start,test_read=False,report=str(REPORT.relative_to(ROOT))))
    finally:
        for job in active.values(): stop(job['process']); job['log'].close()


def main():
    global LEASE
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=('test','prepare','supervise','worker','cache','fit','generate','summarize'))
    parser.add_argument('--domain',choices=DOMAINS); parser.add_argument('--lease-pid',type=int)
    args = parser.parse_args(); bind()
    def interrupted(signum,frame): raise InterruptedError('Decision features interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        if args.mode == 'test': print(json.dumps(C.tests()))
        elif args.mode == 'prepare': prepare()
        elif args.mode == 'supervise': supervise()
        elif args.mode == 'worker': return worker(args.domain)
        elif args.mode == 'summarize': summarize()
        else:
            assert args.domain and args.lease_pid
            LEASE = S.LeaseHandle(args.lease_pid); assert LEASE.poll() is None
            torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
            if args.mode == 'cache': cache(args.domain)
            elif args.mode == 'fit': fit(args.domain)
            else: Generation(args.domain,LEASE).run()
        return 0
    except BaseException as exc:
        if OUT.exists() and args.mode != 'test':
            dest = OUT/args.domain if args.domain else OUT
            write(dest/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),updated_at=stamp()))
            write(dest/'status.json',dict(state='FAILED',error=repr(exc),updated_at=stamp(),test_read=False))
        raise


if __name__ == '__main__': sys.exit(main())
