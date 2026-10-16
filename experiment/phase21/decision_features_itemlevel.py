"""Matched seven-feature granularity experiment, with frozen recommenders.

Uses the established decision-feature worker/supervisor for resource admission,
leases, status, and timeouts. Scientific methods live here; prior run files and
source files are never edited. Each subprocess explicitly binds its own paths.
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
from experiment.phase21 import decision_features_core as C
from experiment.phase21 import decision_features_itemlevel_core as F
from experiment.phase21.candidate_gate_core import fold
from experiment.phase20.catalog_scope_core import metrics, contrast

ROOT, S, J, JC = D.ROOT, D.S, D.J, D.JC
BASE = ROOT/'artifacts/phase21/decision_features_20261016_v1'
TIME = D.TIME
OUT = ROOT/'artifacts/phase21/decision_features_itemlevel_20261016_v1'
PLAN = ROOT/'plan/第二十一阶段/21.37_七维特征商品粒度匹配对照_20261016.md'
REPORT = ROOT/'report/第二十一阶段/七维特征商品粒度匹配对照_20261016.md'
MODULE = 'experiment.phase21.decision_features_itemlevel'
DOMAINS, ARMS = ('Toys', 'Beauty'), ('root7', 'item7')
read, write, sha, stamp = D.read, D.write, D.sha, D.stamp
CFG = dict(D.CFG, arms=list(ARMS), parameters=dict(root7=355, item7=355),
    features=dict(root7=C.NAMES, item7=F.NAMES),
    feature_space='expert confidence and expert-expert JS at item level; GRAM comparisons at legal root level',
    replaced_coordinates=[2, 3, 4], same_hidden_width_and_parameter_count=True,
    parent_training='both arms fresh CE selection/refit; root7 must reproduce previous decision7',
    root_cache='reuse verified frozen initial distributions; no new GRAM feature inference',
    stop_after='one item-granularity variant and matched root7, two development512 pools, seed0')
CFG.pop('old_parent')


def bind():
    D.OUT, D.REPORT, D.MODULE, D.CFG, D.ARMS = OUT, REPORT, MODULE, CFG, ARMS
    D.summarize = summarize
    D.bind()


def source(domain, part):
    return TIME/domain/'features/tisas'/part


def prepare():
    checks = F.tests()
    assert read(BASE/'status.json')['state'] == 'COMPLETED'
    OUT.mkdir(exist_ok=False)
    write(OUT/'status.json', dict(state='PREPARING', updated_at=stamp(), test_read=False))
    write(OUT/'config.json', CFG); write(OUT/'tests.json', checks)
    shutil.copy2(PLAN, OUT/'protocol.md')
    manifest = {}

    def use(path, expected=None):
        rel = str(path.relative_to(ROOT))
        value = manifest.get(rel) or sha(path)
        assert expected is None or value == expected, rel
        manifest[rel] = value

    # Keep all inherited dependencies pinned, including frozen expert scores,
    # tokenizer, GRAM, label-clean root caches, and prior reference outputs.
    manifests = [BASE/'manifest.json']
    for domain in DOMAINS:
        manifests += [BASE/domain/(name+'.json') for name in
                      ('feature_manifest', 'training_manifest', 'output_manifest')]
    for path in manifests:
        use(path)
        for rel, digest in read(path)['sha256'].items():
            use(ROOT/rel, digest)
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
                             prior_root_guided=row['decision7_guided'], prior_root_late=row['decision7_late'],
                             old10_guided=row['old10_guided'], old10_late=row['old10_late']))
        write(p/'references.json', refs); use(p/'references.json')
        write(p/'status.json', dict(state='PREPARED', users_total=512, updated_at=stamp()))
    for path in (Path(__file__), ROOT/'experiment/phase21/decision_features_itemlevel_core.py',
                 ROOT/'experiment/phase21/run_decision_features_itemlevel.sh', PLAN, OUT/'config.json'):
        use(path)
    for rel in manifest:
        p = ROOT/rel
        if p.suffix in ('.py', '.sh', '.md'):
            dest = OUT/'sources'/rel; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dest)
    write(OUT/'manifest.json', dict(sha256=manifest, created_at=stamp()))
    write(OUT/'status.json', dict(state='PREPARED', updated_at=stamp(), test_read=False))
    REPORT.write_text('# 七维特征商品粒度匹配对照\n\n状态：PREPARED，尚无新效果结论。\n')
    print(json.dumps(dict(state='PREPARED', hashes=len(manifest), tests=checks)), flush=True)


def cache(domain):
    p = OUT/domain
    tokens, groups = C.root_mapping(read(p/'catalog.json')['decoder_paths'])
    info = {}
    for part, name in (('train', 'fit_samples.json'), ('validation', 'samples.json')):
        samples = read(p/name)
        origin = BASE/domain/'features'/part
        roots = np.load(origin/'root_probabilities.npz')
        np.testing.assert_array_equal(roots['users'], [s['user_id'] for s in samples])
        np.testing.assert_array_equal(roots['tokens'], tokens)
        old = np.load(origin/'decision7.npy')
        item = np.load(source(domain, part)/'item.npy', mmap_mode='r')
        new = np.empty_like(old)
        deltas = []
        for j, sample in enumerate(samples):
            probs = roots['probabilities'][j]
            new[j], old_check = F.features(len(sample['full_history']), probs[0],
                                          item[j, :, 0], item[j, :, 1], groups)
            np.testing.assert_allclose(old_check, old[j], atol=1e-7, rtol=1e-6)
            np.testing.assert_array_equal(new[j, [0, 1, 5, 6]], old[j, [0, 1, 5, 6]])
            for channel in (0, 1):
                root = C.expert_root(item[j, :, channel], groups, len(tokens))
                np.testing.assert_allclose(root, probs[channel+1], atol=2e-12, rtol=1e-10)
            # Separate softmax, entropy and entropy-form JS verify every item
            # statistic without calling the new feature constructor's helpers.
            logits = np.asarray(item[j, :, :2], dtype=np.float64)*20
            mass = np.exp(logits-logits.max(0)); mass /= mass.sum(0)
            h = lambda x: -(x*np.log(np.maximum(x, 1e-300))).sum(0)
            expected = [*(h(mass)/np.log(len(mass))),
                        (h(mass.mean(1))-h(mass).mean())/np.log(2)]
            np.testing.assert_allclose(new[j, 2:5], expected, atol=1e-7, rtol=1e-6)
            assert new[j, 4] >= old[j, 4]-1e-6
            deltas.append(float(new[j, 4]-old[j, 4]))
            if j == 0 or (j+1) % 256 == 0 or j+1 == len(samples):
                D.progress(domain, 'cache_item_features', part=part, completed=j+1, total=len(samples))
        dest = p/'features'/part; dest.mkdir(parents=True)
        np.save(dest/'root7.npy', old); np.save(dest/'item7.npy', new)
        info[part] = dict(users=len(samples), branches=len(tokens), all_features_verified=True,
                          four_unchanged_coordinates_exact=True,
                          mean_additional_item_disagreement=float(np.mean(deltas)),
                          root_feature_std=old.std(0).tolist(), item_feature_std=new.std(0).tolist())
        write(dest/'metadata.json', dict(users_order=[s['user_id'] for s in samples],
                                         features=CFG['features'], target_used_in_features=False, **info[part]))
    write(p/'feature_checks.json', dict(state='PASSED', parts=info, root_cache_hash_verified=True,
                                        target_used_in_features=False, new_gram_forward_passes=0))
    paths = list((p/'features').glob('*/*'))+[p/'feature_checks.json']
    write(p/'feature_manifest.json', dict(sha256={str(x.relative_to(ROOT)): sha(x) for x in paths}))


def fit(domain):
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    device = torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(11*2**30/torch.cuda.get_device_properties(0).total_memory)
    p = OUT/domain
    for rel, digest in read(p/'feature_manifest.json')['sha256'].items():
        assert sha(ROOT/rel) == digest, rel
    samples, dev = read(p/'fit_samples.json'), read(p/'samples.json')
    fi = np.asarray([j for j, s in enumerate(samples) if fold(s['user_id'], domain) != 0])
    hi = np.asarray([j for j, s in enumerate(samples) if fold(s['user_id'], domain) == 0])
    all_ix = np.arange(len(samples))
    yt = torch.tensor([s['target']-1 for s in samples], device=device)
    teacher = np.load(p/'teacher.npz')
    assert sha(p/'teacher.npz') == read(p/'teacher_checks.json')['sha256']
    np.testing.assert_array_equal(teacher['users'], [s['user_id'] for s in samples])
    np.testing.assert_array_equal(teacher['targets'], yt.cpu().numpy())
    nt, mt = [torch.from_numpy(teacher[k]).to(device) for k in ('native', 'legal')]
    et = torch.from_numpy(np.ascontiguousarray(np.load(source(domain, 'train')/'item.npy', mmap_mode='r')[..., :2])).to(device)
    ve = np.ascontiguousarray(np.load(source(domain, 'validation')/'item.npy', mmap_mode='r')[..., :2])
    tree = JC.PathMass(read(p/'catalog.json')['decoder_paths'], device)
    exported, alphas, betas, info = {}, {}, {}, {}
    for arm in ARMS:
        u = np.load(p/'features/train'/(arm+'.npy'))
        v = np.load(p/'features/validation'/(arm+'.npy'))
        ut = torch.from_numpy(u).to(device)
        fn, an = J.user_norm(u, fi), J.user_norm(u, all_ix)

        def run(indices, norm, epochs, phase, objective, parent=None):
            torch.manual_seed(0); torch.cuda.manual_seed_all(0)
            model = C.Fusion(7, parent).to(device)
            if objective == 'ce': model.strength.requires_grad_(False)
            opt = torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],
                                   lr=CFG['lr'], weight_decay=CFG['weight_decay'])
            rng = np.random.default_rng(0)
            normal = C.normalize(ut, *[torch.from_numpy(x).to(device) for x in norm])

            def loss(ix):
                beta, alpha = model(normal[ix])
                scores = (beta[:, None, :]*et[ix]).sum(-1)
                if objective == 'ce':
                    return torch.nn.functional.cross_entropy(20*scores, yt[ix])
                return JC.path_nll(nt[ix], mt[ix], tree.logs(scores, yt[ix]), alpha, tree.valid[yt[ix]]).mean()

            curve, best_state, best = [], None, float('inf')
            for epoch in range(epochs+1):
                total = 0.
                if epoch:
                    order = rng.permutation(indices)
                    for start in range(0, len(order), 32):
                        ix = order[start:start+32]
                        opt.zero_grad(); value = loss(ix)
                        assert torch.isfinite(value)
                        value.backward()
                        assert all(x.grad is None or torch.isfinite(x.grad).all() for x in model.parameters())
                        opt.step(); total += float(value)*len(ix)
                val = None
                if phase == 'selection' and epoch in CFG['checkpoints']:
                    with torch.no_grad():
                        val = sum(float(loss(hi[k:k+32]))*len(hi[k:k+32]) for k in range(0, len(hi), 32))/len(hi)
                    curve.append(dict(epoch=epoch, loss=val))
                    if val < best:
                        best = val; best_state = {k: x.detach().cpu().clone() for k, x in model.state_dict().items()}
                D.progress(domain, objective+'_'+phase, arm=arm, epoch=epoch, epochs=epochs,
                           training_loss=total/len(indices) if epoch else None, internal_heldout_loss=val)
                write(p/(arm+'_'+objective+'_'+phase+'_curve.json'), dict(curve=curve, completed_epochs=epoch))
            if phase == 'selection': model.load_state_dict(best_state)
            return {k: x.detach().cpu().clone() for k, x in model.state_dict().items()}, curve

        fold_state, ce_curve = run(fi, fn, 20, 'selection', 'ce')
        ce_epochs = min(ce_curve, key=lambda x: (x['loss'], x['epoch']))['epoch']
        full_state, _ = run(all_ix, an, ce_epochs, 'refit', 'ce')
        parent_dir = p/'parents'/arm; parent_dir.mkdir(parents=True)
        for name, state, norm, ix in (('fold', fold_state, fn, fi), ('full', full_state, an, all_ix)):
            torch.save(dict(state_dict=state, mean=norm[0], std=norm[1], selected_epochs=ce_epochs,
                            fit_users=[samples[j]['user_id'] for j in ix], selection_curve=ce_curve), parent_dir/(name+'.pt'))
        _, curve = run(fi, fn, 20, 'selection', 'nll', fold_state)
        epochs = min(curve, key=lambda x: (x['loss'], x['epoch']))['epoch']
        state, _ = run(all_ix, an, epochs, 'refit', 'nll', full_state)
        ck = dict(state_dict=state, mean=an[0], std=an[1], dimension=7, mode='user',
                  selected_epochs=epochs, parent_epochs=ce_epochs, selection_curve=curve,
                  fit_users=[s['user_id'] for s in samples], config=CFG)
        torch.save(ck, p/(arm+'.pt'))
        model = C.Fusion(7).to(device); model.load_state_dict(state)
        with torch.no_grad():
            beta, alpha = model(C.normalize(torch.from_numpy(v).to(device), *[torch.from_numpy(x).to(device) for x in an]))
            bn, aa = beta.cpu().numpy(), alpha.cpu().double().numpy()
        nb, na = C.numpy_forward(torch.load(p/(arm+'.pt'), map_location='cpu')['state_dict'], v, *an)
        np.testing.assert_allclose(bn, nb, atol=2e-6, rtol=2e-5)
        np.testing.assert_allclose(aa, na, atol=2e-6, rtol=2e-5)
        scores = (bn[:, None, :]*ve).sum(-1)
        np.testing.assert_allclose(scores, (nb[:, None, :]*ve).sum(-1), atol=3e-6, rtol=2e-5)
        parity = None
        if arm == 'root7':
            old = read(BASE/domain/'training.json')['fits']['decision7']
            assert (ce_epochs, epochs) == (old['parent_epochs'], old['selected_epochs'])
            parity = {}
            for name, value in (('signals', scores), ('alphas', aa), ('betas', bn)):
                reference = np.load(BASE/domain/(name+'.npz'))['decision7']
                np.testing.assert_allclose(value, reference, atol=3e-6, rtol=2e-5)
                parity['max_'+name+'_error'] = float(np.max(np.abs(value-reference)))
        exported[arm], alphas[arm], betas[arm] = scores, aa, bn
        info[arm] = dict(dimension=7, parameters=sum(x.numel() for x in model.parameters()),
                         parent_epochs=ce_epochs, selected_epochs=epochs, selection_curve=curve,
                         parent_selection_curve=ce_curve, selection_fit_users=len(fi), heldout_users=len(hi),
                         refit_users=len(samples), alpha_mean=float(aa.mean()),
                         alpha_quantiles=np.quantile(aa, [0, .1, .5, .9, 1]).tolist(),
                         behavior_weight_mean=float(bn[:, 0].mean()), numpy_verified_users=len(v),
                         historical_parity=parity)
        del ut, model; gc.collect(); torch.cuda.empty_cache()
    np.savez(p/'signals.npz', **exported); np.savez(p/'alphas.npz', **alphas); np.savez(p/'betas.npz', **betas)
    write(p/'training.json', dict(fits=info, test_read=False, dev_used_for_selection=False,
                                  fit_dev_disjoint=True, internal_parent_weights_exclude_heldout_labels=True))
    paths = [p/n for n in ('signals.npz', 'alphas.npz', 'betas.npz', 'training.json',
                          'root7.pt', 'item7.pt', 'feature_manifest.json')]
    paths += list((p/'parents').glob('*/*.pt'))+list(p.glob('*curve.json'))
    write(p/'training_manifest.json', dict(sha256={str(x.relative_to(ROOT)): sha(x) for x in paths}))


class Generation(S.Generation):
    def run(self):
        p = self.folder
        for rel, digest in read(p/'training_manifest.json')['sha256'].items():
            assert sha(ROOT/rel) == digest, rel
        self.alpha = dict(np.load(p/'alphas.npz'))
        self.stage = 'native_zero_prior_root_smoke'; self.status()
        checks = []
        for j in (0, 256, 511):
            ctx = self.context(j); native = self.decode(ctx)
            plain = self.decode(ctx, native_entry=True, record=False)
            q = self.index.conditional_logs(self.signals['item7'][j])
            zero = self.decode(ctx, q, 0.)
            for other in (plain, zero):
                assert native['paths'] == other['paths']
                np.testing.assert_array_equal(native['scores'], other['scores'])
            checks.append(dict(row=j, native_exact=True, alpha_zero_exact=True))
            del ctx, native, plain, zero
        write(p/'smoke.json', dict(state='PASSED', checks=checks))
        self.stage = 'matched_generation'; self.status()
        root_exact = 0
        with (p/'predictions.jsonl').open('x') as stream:
            for j, sample in enumerate(self.samples):
                begin = time.monotonic(); ctx = self.context(j); native = self.decode(ctx)
                ref = self.reference[sample['user_id']]
                assert native['ids'] == ref['native']['ids']
                np.testing.assert_allclose(native['scores'], ref['native']['scores'], atol=2e-6, rtol=1e-5)
                row = dict(user_id=sample['user_id'], target=sample['target'], history=sample['history'],
                           native=dict(ids=native['ids'], scores=native['scores'].tolist()),
                           old10_guided=ref['old10_guided'], old10_late=ref['old10_late'],
                           prior_root_guided=ref['prior_root_guided'], prior_root_late=ref['prior_root_late'])
                for arm in ARMS:
                    a = float(self.alpha[arm][j])
                    q = self.index.conditional_logs(self.signals[arm][j], temperature_scale=20.)
                    late = native['guide'].rescore(native['paths'], q, a)
                    guided = self.decode(ctx, q, a)
                    row[arm+'_late'] = dict(ids=native['ids'], scores=late.tolist())
                    row[arm+'_guided'] = dict(ids=guided['ids'], scores=guided['scores'].tolist())
                    if arm == 'root7':
                        for suffix in ('guided', 'late'):
                            actual, expected = row[arm+'_'+suffix], ref['prior_root_'+suffix]
                            assert set(actual['ids']) == set(expected['ids']), (j, suffix)
                            mapping = dict(zip(expected['ids'], expected['scores']))
                            np.testing.assert_allclose(actual['scores'], [mapping[x] for x in actual['ids']], atol=3e-6, rtol=2e-5)
                        root_exact += int(guided['ids'] == ref['prior_root_guided']['ids'])
                    del guided
                stream.write(json.dumps(row)+'\n'); stream.flush(); os.fsync(stream.fileno())
                self.users_completed = j+1; self.row_seconds.append(time.monotonic()-begin)
                if j == 0 or (j+1) % 8 == 0 or j == 511: self.status(arms_per_user=2)
                del ctx, native
        assert self.digest() == self.initial_digest
        write(p/'runtime_checks.json', dict(state='PASSED', gram_model_digest_unchanged=True,
                                            model_digest=self.initial_digest, native_reproduced_users=512,
                                            prior_root_order_exact_users=root_exact,
                                            prior_root_score_mappings_verified_users=512,
                                            gram_optimizer_steps=0, all_arms_generated=True))
        self.stage = 'complete'; self.status('COMPLETED')


def summarize():
    results = {}
    keys = ('native', 'old10_guided', 'old10_late', 'prior_root_guided', 'prior_root_late')+tuple(a+s for a in ARMS for s in ('_guided', '_late'))
    lines = ['# 七维特征商品粒度匹配对照', '', '状态：COMPLETED。两域各512历史开发用户、seed0；探索性结果。', '',
             '| 数据集 | GRAM N10 | 历史10维 N10 | 原7维 N10 | 商品粒度7维 N10 | 商品粒度7维后重排 N10 |',
             '|---|---:|---:|---:|---:|---:|']
    for domain in DOMAINS:
        p = OUT/domain
        assert read(p/'status.json')['state'] == 'COMPLETED'
        assert read(p/'lease_status.json')['state'] == 'released'
        for name in ('feature_manifest.json', 'training_manifest.json'):
            for rel, digest in read(p/name)['sha256'].items(): assert sha(ROOT/rel) == digest, rel
        samples = read(p/'samples.json')
        rows = [json.loads(x) for x in (p/'predictions.jsonl').read_text().splitlines()]
        assert len(rows) == len(samples) == 512
        rr = {k: [] for k in keys}
        for row, sample in zip(rows, samples):
            assert all(row[k] == sample[k] for k in ('user_id', 'target', 'history'))
            for key in keys:
                value = row[key]
                assert len(value['ids']) == len(set(value['ids'])) == len(value['scores']) == 50
                assert np.isfinite(value['scores']).all()
                rank = S.OLD.record_rank(value, sample)
                eligible = sorted(((i, s) for i, s in zip(value['ids'], value['scores']) if i not in sample['history']),
                                  key=lambda x: (-x[1], x[0]))
                ids = [x[0] for x in eligible]
                assert rank == (ids.index(sample['target'])+1 if sample['target'] in ids else 2**30)
                rr[key].append(rank)
        rr = {k: np.asarray(v) for k, v in rr.items()}
        previous = np.load(BASE/domain/'ranks.npz')
        for actual, expected in (('native', 'native'), ('old10_guided', 'old10_guided'), ('old10_late', 'old10_late'),
                                 ('root7_guided', 'decision7_guided'), ('root7_late', 'decision7_late')):
            np.testing.assert_array_equal(rr[actual], previous[expected])
        pairs = [('item7_guided', 'root7_guided'), ('item7_guided', 'old10_guided'),
                 ('item7_guided', 'native'), ('item7_guided', 'item7_late'),
                 ('root7_guided', 'root7_late'), ('root7_guided', 'prior_root_guided')]
        a, b = rr['item7_guided'], rr['root7_guided']
        delta = np.where(a <= 10, 1/np.log2(a+1), 0)-np.where(b <= 10, 1/np.log2(b+1), 0)
        masks = dict(gained=(a <= 10) & (b > 10), lost=(b <= 10) & (a > 10),
                     retained=(a <= 10) & (b <= 10), neither=(a > 10) & (b > 10))
        weights = {}
        for arm in ARMS:
            alpha = np.load(p/'alphas.npz')[arm]; beta = np.load(p/'betas.npz')[arm][:, 0]
            weights[arm] = {name: dict(mean=float(x.mean()), std=float(x.std()), quantiles=np.quantile(x, [0, .1, .5, .9, 1]).tolist())
                            for name, x in (('alpha', alpha), ('behavior_weight', beta))}
        result = dict(users=512, seed=0, metrics={k: metrics(v) for k, v in rr.items()},
                      comparisons={a+'_vs_'+b: contrast(rr[a], rr[b], repeats=2000, seed=20261016) for a, b in pairs},
                      top10_change_decomposition={k: dict(users=int(m.sum()), delta_N10_contribution=float(delta[m].sum()/512)) for k, m in masks.items()},
                      weights=weights, training=read(p/'training.json'), test_read=False)
        np.savez_compressed(p/'ranks.npz', users=np.asarray([s['user_id'] for s in samples]), **rr)
        write(p/'summary.json', result); results[domain] = result
        write(p/'verification.json', dict(state='VERIFIED_SAVED_PREDICTIONS', users=512,
                                          independently_sorted_rank_arrays=len(keys), historical_rank_arrays_verified=5))
        paths = [p/n for n in ('predictions.jsonl', 'ranks.npz', 'summary.json', 'verification.json', 'runtime_checks.json', 'smoke.json')]
        write(p/'output_manifest.json', dict(sha256={str(x.relative_to(ROOT)): sha(x) for x in paths}))
        lines.append('| '+domain+' | '+' | '.join(f'{result["metrics"][k]["N10"]:.6f}' for k in
                     ('native', 'old10_guided', 'root7_guided', 'item7_guided', 'item7_late'))+' |')
    write(OUT/'summary.json', dict(domains=results, test_read=False))
    lines += ['', '两组均7维、355参数、隐藏宽度32、同一初始化/优化/内部验证选轮规则；原7维完整重训及重生成，并核对历史结果。',
              '只将两专家熵和专家间JS从首层改到全商品；GRAM相关比较仍在合法首层空间。专家仍影响每一步生成。',
              '特征组整体变化，不宣称单项特征必要或真实可靠性；首层GRAM统计仍不覆盖所有后续路径。',
              '单seed、重复开发池、12组未经校正描述性区间；没有test访问，不作正式显著性或稳定泛化结论。',
              '本轮完成即停止，没有自动追加网络宽度、专家组合或完整论文实验。']
    REPORT.write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('test', 'prepare', 'supervise', 'worker', 'cache', 'fit', 'generate', 'summarize'))
    parser.add_argument('--domain', choices=DOMAINS); parser.add_argument('--lease-pid', type=int)
    args = parser.parse_args(); bind()
    def interrupted(signum, frame): raise InterruptedError('Item-level decision features interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.mode == 'test': print(json.dumps(F.tests()))
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
