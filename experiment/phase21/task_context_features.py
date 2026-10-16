"""Matched original10, retained7, and task-aware10 joint user fusers.

Reuse verified immutable score/root caches. Train each new CE parent and joint
path-NLL head under the same protocol, then freshly generate all three arms.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback
import numpy as np
import torch
from experiment.phase21 import decision_features as D
from experiment.phase21 import task_context_features_core as C
from experiment.phase21.candidate_gate_core import fold
from experiment.phase20.catalog_scope_core import metrics, contrast

ROOT, S, J, JC = D.ROOT, D.S, D.J, D.JC
PRIOR, TIME = D.PRIOR, D.TIME
BASE = ROOT/'artifacts/phase21/decision_features_20261016_v1'
OUT = ROOT/'artifacts/phase21/task_context_features_20261016_v1'
PLAN = ROOT/'plan/第二十一阶段/21.40_面向生成引导的十维上下文匹配对照_20261016.md'
REPORT = ROOT/'report/第二十一阶段/面向生成引导的十维上下文匹配对照_20261016.md'
MODULE = 'experiment.phase21.task_context_features'
DOMAINS, ARMS = ('Toys', 'Beauty'), C.ARMS
read, write, sha, stamp, progress = D.read, D.write, D.sha, D.stamp, D.progress
feature_source = D.feature_source
CFG = dict(D.CFG, arms=list(ARMS), features=C.NAMES,
    parameters=dict(old10=451, pruned7=355, task10=451),
    feature_space='retained expert/history statistics plus legal first-token GRAM context',
    retained_original_columns=C.KEEP, removed_original_columns=C.REPLACED,
    extra_features=C.EXTRA_NAMES, extra_source_columns=[1,5,6],
    architecture='all active inputs through shared32 ReLU; beta32to2 and alpha32to1',
    beta_sees_extra=True, same_hidden_width=True,
    equal_parameter_count_old10_task10=True, equal_parameter_count_all_arms=False,
    parent_initialization='common retained input columns identical; new columns use replaced-column scale',
    stop_after='three arms x two development512 pools, seed0; no automatic follow-up')


def bind():
    D.OUT, D.REPORT, D.MODULE, D.CFG, D.ARMS = OUT, REPORT, MODULE, CFG, ARMS
    D.summarize = summarize
    D.bind()


def prepare():
    checks = C.tests()
    assert read(BASE/'status.json')['state'] == 'COMPLETED'
    OUT.mkdir(exist_ok=False)
    write(OUT/'status.json', dict(state='PREPARING', updated_at=stamp(), test_read=False))
    write(OUT/'config.json', CFG); write(OUT/'tests.json', checks)
    shutil.copy2(PLAN, OUT/'protocol.md')
    manifest = {}
    def use(path, expected=None):
        rel = str(path.relative_to(ROOT))
        digest = manifest.get(rel) or sha(path)
        assert expected is None or digest == expected, rel
        manifest[rel] = digest
    manifests = [BASE/'manifest.json']
    for domain in DOMAINS:
        manifests += [BASE/domain/(name+'.json') for name in
                      ('feature_manifest', 'training_manifest', 'output_manifest')]
    for path in manifests:
        use(path)
        for rel, digest in read(path)['sha256'].items(): use(ROOT/rel, digest)
    for domain in DOMAINS:
        p = OUT/domain; p.mkdir()
        for name in ('samples.json', 'fit_samples.json', 'catalog.json', 'data_config.json',
                     'teacher.npz', 'teacher_checks.json'):
            shutil.copy2(BASE/domain/name, p/name); use(p/name)
        fit, dev = read(p/'fit_samples.json'), read(p/'samples.json')
        assert len(dev) == len({s['user_id'] for s in dev}) == 512
        assert {s['user_id'] for s in fit}.isdisjoint(s['user_id'] for s in dev)
        assert sha(p/'teacher.npz') == read(p/'teacher_checks.json')['sha256']
        rows = [json.loads(line) for line in (BASE/domain/'predictions.jsonl').read_text().splitlines()]
        assert len(rows) == len(dev)
        refs = []
        for row, sample in zip(rows, dev):
            assert all(row[k] == sample[k] for k in ('user_id', 'target', 'history'))
            refs.append(dict(user_id=sample['user_id'], native=row['native'],
                             historical_guided=row['old10_guided'], historical_late=row['old10_late']))
        write(p/'references.json', refs); use(p/'references.json')
        write(p/'status.json', dict(state='PREPARED', users_total=512, updated_at=stamp()))
    for path in (Path(__file__), ROOT/'experiment/phase21/task_context_features_core.py',
                 ROOT/'experiment/phase21/run_task_context_features.sh',
                 ROOT/'experiment/phase21/alpha_gram_features_core.py', PLAN, OUT/'config.json'):
        use(path)
    for rel in manifest:
        p = ROOT/rel
        if p.suffix in ('.py', '.sh', '.md'):
            dest = OUT/'sources'/rel; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dest)
    write(OUT/'manifest.json', dict(sha256=manifest, created_at=stamp()))
    write(OUT/'status.json', dict(state='PREPARED', updated_at=stamp(), test_read=False))
    REPORT.write_text('# 面向生成引导的十维上下文匹配对照\n\n状态：PREPARED，尚无新效果结论。\n')
    print(json.dumps(dict(state='PREPARED', hashes=len(manifest), tests=checks)), flush=True)


def cache(domain):
    p = OUT/domain; checks = {}
    tokens, _ = D.C.root_mapping(read(p/'catalog.json')['decoder_paths'])
    assert read(BASE/domain/'feature_checks.json')['frozen_gram_digest'] == read(p/'teacher_checks.json')['model_digest']
    for part, name in (('train', 'fit_samples.json'), ('validation', 'samples.json')):
        samples = read(p/name)
        assert samples == read(BASE/domain/name)
        old = np.load(feature_source(domain, part)/'user.npy')
        root = BASE/domain/'features'/part
        meta = read(root/'metadata.json')
        assert meta['users_order'] == [s['user_id'] for s in samples]
        assert meta['decoder_input'] == [0] and not meta['target_used_in_features']
        roots = np.load(root/'root_probabilities.npz')
        np.testing.assert_array_equal(roots['users'], [s['user_id'] for s in samples])
        np.testing.assert_array_equal(roots['tokens'], tokens)
        assert roots['probabilities'].shape == (len(samples), 3, len(tokens))
        dest = p/'features'/part; dest.mkdir(parents=True)
        arrays = {arm: C.features(old, roots['probabilities'], arm) for arm in ARMS}
        np.testing.assert_array_equal(arrays['old10'], old)
        np.testing.assert_array_equal(arrays['pruned7'], old[:, C.KEEP])
        np.testing.assert_array_equal(arrays['task10'][:, :7], arrays['pruned7'])
        expected = np.load(root/'decision7.npy')[:, [1,5,6]]
        np.testing.assert_array_equal(arrays['task10'][:, 7:], expected)
        # Independently evaluate KL, rather than only comparing two callers of
        # the inherited entropy-identity implementation.
        prob = roots['probabilities']; prob = prob/prob.sum(-1,keepdims=True)
        ent = -(prob[:,0]*np.log(np.maximum(prob[:,0],1e-300))).sum(-1)/np.log(len(tokens))
        direct = [ent]
        for j in (1,2):
            a,b = prob[:,0],prob[:,j]; m = (a+b)/2
            direct.append(.5*(a*(np.log(np.maximum(a,1e-300))-np.log(np.maximum(m,1e-300)))).sum(-1)/np.log(2)
                          +.5*(b*(np.log(np.maximum(b,1e-300))-np.log(np.maximum(m,1e-300)))).sum(-1)/np.log(2))
        np.testing.assert_allclose(arrays['task10'][:,7:],np.stack(direct,axis=1),atol=1e-7,rtol=1e-6)
        for arm,value in arrays.items():
            assert value.shape == (len(samples),len(C.NAMES[arm])) and np.isfinite(value).all()
            np.save(dest/(arm+'.npy'),value)
        checks[part] = dict(users=len(samples),retained_columns=C.KEEP,replaced_columns=C.REPLACED,
                           old10_exact=True,retained_seven_exact=True,extra_source_columns=[1,5,6],
                           extra_matches_previous_exact=True,independent_kl_verified=True,branches=len(tokens))
        write(dest/'metadata.json', dict(users_order=[s['user_id'] for s in samples],
                                         features=C.NAMES,decoder_input=[0],target_used_in_features=False,**checks[part]))
        progress(domain,'cache_task_context_features',part=part,completed=len(samples),total=len(samples))
    prior_checks=read(BASE/domain/'feature_checks.json')
    assert all(x['placeholder_target_invariant'] and x['root_matches_beam'] and x['expert_matches_trie']
               for x in prior_checks['checks'])
    write(p/'feature_checks.json',dict(state='PASSED',parts=checks,no_new_gram_forward=True,
                                       target_used_in_features=False,verified_source=prior_checks))
    paths=list((p/'features').glob('*/*'))+[p/'feature_checks.json']
    write(p/'feature_manifest.json',dict(sha256={str(x.relative_to(ROOT)):sha(x) for x in paths}))


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
        dimension = len(C.NAMES[arm])
        def users(part):
            return np.load(p/'features'/part/(arm+'.npy'))
        u,v = users('train'),users('validation')
        ut = torch.from_numpy(u).to(device)
        fn,an = J.user_norm(u,fi),J.user_norm(u,all_ix)

        def run(indices,norm,epochs,phase,objective,parent=None):
            torch.manual_seed(0); torch.cuda.manual_seed_all(0)
            model = C.Fusion(arm,parent).to(device)
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
        model = C.Fusion(arm).to(device); model.load_state_dict(state)
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
    paths = [p/n for n in ('signals.npz','alphas.npz','betas.npz','training.json','old10.pt','pruned7.pt','task10.pt','feature_manifest.json')]
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
            q = self.index.conditional_logs(self.signals['task10'][j])
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
                if j == 0 or (j+1)%8 == 0 or j == 511: self.status(arms_per_user=len(ARMS))
                del ctx,native
        assert self.digest() == self.initial_digest
        write(p/'runtime_checks.json',dict(state='PASSED',gram_model_digest_unchanged=True,
            model_digest=self.initial_digest,native_reproduced_users=512,old_guided_order_exact_users=old_exact,
            old_item_score_mappings_verified_users=512,gram_optimizer_steps=0,all_arms_generated=True))
        self.stage = 'complete'; self.status('COMPLETED')



def summarize():
    results = {}
    keys = ('native','historical_guided','historical_late')+tuple(a+s for a in ARMS for s in ('_guided','_late'))
    lines = ['# 面向生成引导的十维上下文匹配对照','',
        '状态：COMPLETED。两域各512历史开发用户，seed0；新结果为探索性证据。','',
        '| 数据集 | GRAM N10 | 原10维 N10 | 精简7维 N10 | 生成上下文10维 N10 |',
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
                assert len(value['ids']) == len(set(value['ids'])) == len(value['scores']) == 50
                assert np.isfinite(value['scores']).all()
                ids,scores = np.asarray(value['ids']),np.asarray(value['scores'])
                keep = ~np.isin(ids,sample['history']); ids,scores = ids[keep],scores[keep]
                order = ids[np.lexsort((ids,-scores))].tolist()
                assert rank == (order.index(sample['target'])+1 if sample['target'] in order else 2**30)
                rr[key].append(rank)
        rr = {k:np.asarray(v) for k,v in rr.items()}
        old = np.load(PRIOR/domain/'ranks.npz')
        for a,b in (('native','native'),('historical_guided','tisas_user_guided'),('historical_late','tisas_user_late')):
            np.testing.assert_array_equal(rr[a],old[b])
        for suffix in ('guided','late'):
            np.testing.assert_array_equal(rr['old10_'+suffix],rr['historical_'+suffix])
        pairs = [('task10_guided','old10_guided'),('task10_guided','pruned7_guided'),
                 ('pruned7_guided','old10_guided')]
        pairs += [(a+'_guided','native') for a in ARMS]
        pairs += [(a+'_guided',a+'_late') for a in ARMS]
        result = dict(users=512,seed=0,metrics={k:metrics(v) for k,v in rr.items()},
            comparisons={a+'_vs_'+b:contrast(rr[a],rr[b],repeats=2000,seed=20261016) for a,b in pairs},
            training=read(p/'training.json'),test_read=False)
        np.savez_compressed(p/'ranks.npz',users=np.asarray([s['user_id'] for s in samples]),**rr)
        write(p/'summary.json',result); results[domain]=result
        write(p/'verification.json',dict(state='VERIFIED_SAVED_PREDICTIONS',users=512,
            independently_sorted_rank_arrays=len(keys),historical_rank_arrays_verified=3))
        paths=[p/n for n in ('predictions.jsonl','ranks.npz','summary.json','verification.json','runtime_checks.json','smoke.json')]
        write(p/'output_manifest.json',dict(sha256={str(x.relative_to(ROOT)):sha(x) for x in paths}))
        lines.append('| '+domain+' | '+' | '.join(f'{result["metrics"][k]["N10"]:.6f}' for k in
            ('native','old10_guided','pruned7_guided','task10_guided'))+' |')
    write(OUT/'summary.json',dict(domains=results,test_read=False))
    lines += ['', '精简7维为原10维子集[0,1,2,3,4,7,8]，与先前熵/JS七维方案不同。',
        '生成上下文10维在这7项上添加GRAM首次合法分支熵及两个专家/GRAM的JS，共同进入32隐藏单元并供beta/alpha使用。',
        '参数量451/355/451；task10对old10同参数量，对pruned7同时增加了输入信息和参数。',
        '首层状态不是整个生成过程的GRAM状态；所有前缀仍由相同专家条件概率规则引导。',
        '两项标准差只是待检验的精简取舍，不预先声称冗余或无用；熵与分歧也不是正确率。',
        '原10维复用CE父模型、重新拟合联合头并fresh生成；新增两组按相同CE/NLL规则选轮并全量重拟合。',
        '单seed、重复开发512、未经校正描述性区间，不作正式显著性、等效或泛化结论；没有test访问。',
        '保留LIME-Rec专家/继承特征及AGRec启发生成混合的来源，不据维数变化声称首次提出。',
        '本轮完成即停止，不自动扩大网络、扫特征或开展完整论文实验。']
    REPORT.write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('test', 'prepare', 'supervise', 'worker', 'cache', 'fit', 'generate', 'summarize'))
    parser.add_argument('--domain', choices=DOMAINS); parser.add_argument('--lease-pid', type=int)
    args = parser.parse_args(); bind()
    def interrupted(signum, frame): raise InterruptedError('Task-context-features worker interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.mode == 'test': print(json.dumps(C.tests()))
        elif args.mode == 'prepare': prepare()
        elif args.mode == 'supervise': D.supervise()
        elif args.mode == 'worker': return D.worker(args.domain)
        elif args.mode == 'summarize': summarize()
        else:
            assert args.domain and args.lease_pid
            D.LEASE = S.LeaseHandle(args.lease_pid); assert D.LEASE.poll() is None
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
            if args.mode == 'cache': cache(args.domain)
            elif args.mode == 'fit': fit(args.domain)
            else: Generation(args.domain, D.LEASE).run()
        return 0
    except BaseException as exc:
        if OUT.exists() and args.mode != 'test':
            dest = OUT/args.domain if args.domain else OUT
            write(dest/'failure.json', dict(error=repr(exc), traceback=traceback.format_exc(), updated_at=stamp()))
            write(dest/'status.json', dict(state='FAILED', error=repr(exc), updated_at=stamp(), test_read=False))
        raise


if __name__ == '__main__': sys.exit(main())
