"""Bounded CPU mechanism screening using already exposed development diagnostics."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import signal
import time
import traceback

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from experiment.phase20.guided_decoding_job import ROOT, read, write, sha, now
from experiment.phase20.lime_learnability import arrays, cohorts, training_bundle, fingerprint, predict
from experiment.phase20.source_competition_core import ResidualRanker, row_losses, independent_ranks, target_positions
from experiment.phase20.catalog_scope_core import ranks, metrics, utilities

OLD = ROOT/'artifacts/phase20/lime_learnability_20260920_v1'
CACHE = ROOT/'artifacts/phase20/source_competition_20260920_v2'
OUT = ROOT/'artifacts/phase20/lime_source_probe_20260921_v1'
CONTRACT = ROOT/'plan/第二十阶段/LIME候选来源与普通校准_并行机制筛查_20260921.md'
ARMS = ('full_mlp', 'no_source_mlp', 'shuffled_source_mlp', 'linear_residual')
SEEDS = (20260920, 20260921, 20260922)
SOURCE_COLUMNS = (4, 5, 6, 7)


def transformed(x, ids, users, arm, domain):
    result = x.copy()
    if arm == 'no_source_mlp':
        result[:, :, SOURCE_COLUMNS] = 0.
    elif arm == 'shuffled_source_mlp':
        for i, user in enumerate(users):
            seed = int.from_bytes(hashlib.sha256(f'20260921|source-probe|{domain}|{user}'.encode()).digest()[:4], 'big')
            valid = np.flatnonzero(ids[i] > 0)
            shuffled = np.random.RandomState(seed).permutation(valid)
            result[i][np.ix_(valid, SOURCE_COLUMNS)] = x[i][np.ix_(shuffled, SOURCE_COLUMNS)]
    else:
        assert arm in ('full_mlp', 'linear_residual')
    return result


class LinearResidual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Linear(10, 1)
        self.bound = 2.
        torch.nn.init.zeros_(self.network.weight)
        torch.nn.init.zeros_(self.network.bias)

    def forward(self, x, prior):
        return prior + self.bound*torch.tanh(self.network(x).squeeze(-1))


def new_model(arm):
    return LinearResidual() if arm == 'linear_residual' else ResidualRanker(hidden=32, bound=2.)


def fit_model(bundle, arm, config, seed, steps=30):
    torch.manual_seed(seed)
    model = new_model(arm)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    curve = []
    for _ in range(steps):
        optimizer.zero_grad()
        total = 0.
        for start in range(0, len(bundle['x']), config['chunk_size']):
            sl = slice(start, start+config['chunk_size'])
            scores = model(bundle['x'][sl], bundle['prior'][sl])
            loss = row_losses(scores, bundle['valid'][sl], bundle['pos'][sl]).sum()/len(bundle['x'])
            assert torch.isfinite(loss)
            loss.backward()
            total += float(loss.detach())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        curve.append(total)
    return model, curve


def simple_contrast(a, b):
    return dict(delta_N10=float((utilities(a)-utilities(b)).mean()),
        top10_gained=int(((a <= 10) & (b > 10)).sum()),
        top10_lost=int(((b <= 10) & (a > 10)).sum()),
        top10_net=int((a <= 10).sum()-(b <= 10).sum()))


def verify_metrics(rr, table):
    for k in (5, 10, 20, 50):
        hit = rr <= k
        assert abs(float(hit.sum()/len(rr))-table[f'H{k}']) < 1e-12
        assert abs(float((1/np.log2(rr[hit]+1)).sum()/len(rr))-table[f'N{k}']) < 1e-12


def checks():
    rng = np.random.RandomState(43)
    x = rng.randn(6, 12, 10).astype(np.float32)
    ids = np.tile(np.arange(12), (6, 1))
    users = np.asarray([f'u{i}' for i in range(6)])
    for arm in ARMS:
        first = transformed(x, ids, users, arm, 'Toys')
        order = np.asarray([4, 2, 5, 1, 0, 3])
        np.testing.assert_array_equal(first[order], transformed(x[order], ids[order], users[order], arm, 'Toys'))
        other = [i for i in range(10) if i not in SOURCE_COLUMNS]
        np.testing.assert_array_equal(first[:, :, other], x[:, :, other])
        if arm == 'shuffled_source_mlp':
            np.testing.assert_array_equal(first[ids == 0], x[ids == 0])
            for i in range(len(ids)):
                a = first[i, ids[i] > 0][:, SOURCE_COLUMNS]
                b = x[i, ids[i] > 0][:, SOURCE_COLUMNS]
                assert sorted(map(tuple, a)) == sorted(map(tuple, b))
        elif arm == 'no_source_mlp':
            assert (first[:, :, SOURCE_COLUMNS] == 0).all()
    # Check existing fitted-only normalization and cohort isolation as well.
    import unittest
    from experiment.phase20.test_lime_learnability import LearnabilityTests
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(LearnabilityTests)
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    assert result.wasSuccessful(), log.getvalue()
    return dict(state='PASSED', prior_isolation_tests=result.testsRun,
        deterministic_user_permutation=True, block_value_preservation=True,
        unrelated_features_unchanged=True, test_log=log.getvalue())


class Run:
    def __init__(self):
        self.started = time.monotonic()
        self.inputs = {}
        self.config = read(ROOT/'experiment/phase20/lime_learnability_20260920_v1.json')
        self.parent = read(OLD/'summary.json')
        self.cache_receipt = read(CACHE/'summary.json')
        self.count = 0

    def status(self, stage, state='RUNNING', **extra):
        value = dict(state=state, stage=stage, pid=os.getpid(), updated_at=now(),
            elapsed_seconds=time.monotonic()-self.started, models_fitted=self.count,
            expected_models=100, cpu_threads=4, gpu_used=False, test_read=False,
            development_labels_used_for_fit=True, formal_method_evaluation=False, **extra)
        write(OUT/'status.json', value)
        print(value, flush=True)

    def use(self, path, expected=None):
        relative = str(path.relative_to(ROOT))
        digest = sha(path)
        if expected:
            assert digest == expected, relative
        self.inputs[relative] = digest
        return path

    def load_domain(self, domain):
        values = []
        for name in ('train.npz', 'development.npz'):
            path = CACHE/domain/name
            self.use(path, self.cache_receipt['output_sha256'][str(path.relative_to(ROOT))])
            value = arrays(path)
            np.testing.assert_array_equal(target_positions(value['ids'], value['targets']), value['positions'])
            assert ((value['ing'] | value['inl']) == (value['ids'] > 0)).all()
            values.append(value)
        train, dev = values
        folds, match, parts, = cohorts(train, dev, domain, self.config)
        return dev, folds, match, parts

    def freeze(self):
        self.status('冻结协议和来源')
        assert self.parent['state'] == 'COMPLETED'
        assert self.cache_receipt['state'] == 'COMPLETED'
        paths = [Path(__file__), CONTRACT, ROOT/'experiment/phase20/lime_learnability.py',
            ROOT/'experiment/phase20/lime_learnability_20260920_v1.json',
            ROOT/'experiment/phase20/source_competition_core.py',
            ROOT/'experiment/phase20/catalog_scope_core.py', ROOT/'experiment/phase20/guided_decoding_job.py',
            ROOT/'experiment/phase20/test_lime_learnability.py']
        for p in paths:
            self.use(p)
            dest = OUT/'sources'/p.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
        self.use(OLD/'summary.json')
        self.use(CACHE/'summary.json')
        write(OUT/'config.json', dict(base_config=self.config, arms=ARMS, seeds=SEEDS,
            source_columns=SOURCE_COLUMNS, linear_seeds=SEEDS[:1], max_seconds=900,
            hypotheses='linear calibration; aligned source information; fixed-head candidate restrictions',
            hyperparameter_search=False, validation_labels_used=True))
        write(OUT/'checks.json', checks())

    def train_domain(self, domain):
        folder = OUT/domain
        folder.mkdir()
        dev, folds, match, parts = self.load_domain(domain)
        for arm in ARMS:
            modified = dict(dev, x=transformed(dev['x'], dev['ids'], dev['users'], arm, domain))
            for part in parts:
                ix = part['C_next_interaction']
                bundle, mean, std, used = training_bundle(modified, ix)
                assert not set(dev['users'][used]) & set(dev['users'][part['evaluation']])
                for seed in (SEEDS[:1] if arm == 'linear_residual' else SEEDS):
                    model, curve = fit_model(bundle, arm, self.config, seed)
                    digest = fingerprint(model)
                    if arm == 'full_mlp' and seed == SEEDS[0]:
                        p = OLD/domain/f'fold{part["fold"]}.C_next_interaction.pt'
                        old = torch.load(self.use(p, self.parent['output_sha256'][str(p.relative_to(ROOT))]), map_location='cpu')
                        np.testing.assert_array_equal(used, old['fit_indices'])
                        np.testing.assert_array_equal(mean, old['feature_mean'])
                        np.testing.assert_array_equal(std, old['feature_std'])
                        assert digest == old['final_sha256'], 'Original full model must exactly reproduce'
                    path = folder/f'{arm}.seed{seed}.fold{part["fold"]}.pt'
                    torch.save(dict(state_dict=model.state_dict(), mean=mean, std=std, fit_indices=used,
                        evaluation_indices=part['evaluation'], arm=arm, seed=seed, fold=part['fold'],
                        fingerprint=digest, curve=curve, domain=domain), path)
                    self.count += 1
                    self.status('拟合固定对照，尚未揭示新评价指标', domain=domain, arm=arm, seed=seed, fold=part['fold'])

    def evaluate(self, domain):
        self.status('统一折外评价及保存模型复核', domain=domain)
        dev, folds, match, parts = self.load_domain(domain)
        folder = OUT/domain
        results = {}
        for arm in ARMS:
            x = transformed(dev['x'], dev['ids'], dev['users'], arm, domain)
            for seed in (SEEDS[:1] if arm == 'linear_residual' else SEEDS):
                score = np.full(dev['prior'].shape, -np.inf, dtype=np.float32)
                visits = np.zeros(len(folds), dtype=np.int8)
                for part in parts:
                    cp = torch.load(folder/f'{arm}.seed{seed}.fold{part["fold"]}.pt', map_location='cpu')
                    model = new_model(arm)
                    model.load_state_dict(cp['state_dict'])
                    assert fingerprint(model) == cp['fingerprint']
                    ix = cp['evaluation_indices']
                    np.testing.assert_array_equal(ix, part['evaluation'])
                    assert not set(dev['users'][cp['fit_indices']]) & set(dev['users'][ix])
                    predicted = predict(model, x[ix], dev['prior'][ix], dev['ids'][ix], cp['mean'], cp['std'])
                    score[ix] = predicted
                    visits[ix] += 1
                assert (visits == 1).all()
                rr = ranks(score, dev['ids'], dev['targets'])
                np.testing.assert_array_equal(rr, independent_ranks(score, dev['ids'], dev['targets']))
                key = f'{arm}.{seed}'
                results[key] = rr
                if arm == 'full_mlp':
                    for source in ('gram', 'lime'):
                        restricted = np.where(dev['ing' if source == 'gram' else 'inl'], score, -np.inf)
                        r = ranks(restricted, dev['ids'], dev['targets'])
                        np.testing.assert_array_equal(r, independent_ranks(restricted, dev['ids'], dev['targets']))
                        results[f'{key}.{source}_only'] = r
                np.savez_compressed(folder/f'{key}.predictions.npz', users=dev['users'], scores=score, ranks=rr, folds=folds)
        old_path = OLD/domain/'predictions.npz'
        old = arrays(self.use(old_path, self.parent['output_sha256'][str(old_path.relative_to(ROOT))]))
        old_index = list(old['arm_names']).index('C_next_interaction')
        np.testing.assert_array_equal(results[f'full_mlp.{SEEDS[0]}'], old['ranks'][old_index])
        results['fixed_gram_lime'] = dev['gram_lime_ranks']
        results['fixed_union_lime'] = ranks(np.where(dev['ids'] > 0, dev['prior'], -np.inf), dev['ids'], dev['targets'])
        target_match = dev['ids'] == dev['targets'][:, None]
        in_g = (target_match & dev['ing']).any(1)
        in_l = (target_match & dev['inl']).any(1)
        groups = dict(all=np.ones(len(folds), bool), low_frequency=dev['tail'],
            gram_only_target=in_g & ~in_l, lime_only_target=in_l & ~in_g,
            both_target=in_g & in_l, uncovered=~in_g & ~in_l)
        table = {key:{g:metrics(rr[mask]) for g, mask in groups.items()} for key, rr in results.items()}
        for key, rr in results.items():
            for g, mask in groups.items():
                verify_metrics(rr[mask], table[key][g])
        byfold = {key:[metrics(rr[folds == f]) for f in range(5)] for key, rr in results.items()}
        means = {}
        for arm in ARMS:
            keys = [f'{arm}.{s}' for s in (SEEDS[:1] if arm == 'linear_residual' else SEEDS)]
            means[arm] = {g:{metric:float(np.mean([table[k][g][metric] for k in keys])) for metric in table[keys[0]][g]} for g in groups}
        checks = {}
        for arm in ('no_source_mlp', 'shuffled_source_mlp'):
            full, control = means['full_mlp']['all'], means[arm]['all']
            seed_deltas = [table[f'full_mlp.{s}']['all']['N10']-table[f'{arm}.{s}']['all']['N10'] for s in SEEDS]
            fold_deltas = [float(np.mean([byfold[f'full_mlp.{s}'][f]['N10']-byfold[f'{arm}.{s}'][f]['N10'] for s in SEEDS])) for f in range(5)]
            checks[arm] = dict(relative_N10=full['N10']/control['N10']-1,
                delta_H10=full['H10']-control['H10'], seed_N10_deltas=seed_deltas,
                fold_N10_deltas=fold_deltas, pass_screen=bool(full['N10'] >= control['N10']*1.005 and full['H10'] >= control['H10'] and min(seed_deltas) > 0 and sum(d > 0 for d in fold_deltas) >= 4))
        candidate = {}
        for source in ('gram', 'lime'):
            candidate[source] = []
            for seed in SEEDS:
                a, b = results[f'full_mlp.{seed}'], results[f'full_mlp.{seed}.{source}_only']
                candidate[source].append(dict(seed=seed, **simple_contrast(a,b),
                    full_population_N10_contributions={g:float((utilities(a)-utilities(b))[mask].sum()/len(a)) for g,mask in groups.items() if g != 'all'}))
        np.savez_compressed(folder/'all_ranks.npz', users=dev['users'], targets=dev['targets'], folds=folds,
            arm_names=np.asarray(list(results)), ranks=np.stack(list(results.values())), **{f'mask_{k}':v for k,v in groups.items()})
        result = dict(metrics=table, means=means, fold_metrics=byfold, source_checks=checks,
            candidate_restriction=candidate, group_sizes={k:int(v.sum()) for k,v in groups.items()},
            original_full_seed_exact_reproduction=True, every_user_once_per_arm=True,
            independent_rankings_and_metrics_verified=True,
            average_candidate_counts=dict(union=float((dev['ids']>0).sum(1).mean()), gram=float(dev['ing'].sum(1).mean()), lime=float(dev['inl'].sum(1).mean())),
            test_read=False, development_labels_used_for_fit=True)
        write(folder/'summary.json', result)
        return result

    def run(self):
        self.freeze()
        for domain in ('Toys','Beauty'):
            self.train_domain(domain)
        assert self.count == 100
        results = {d:self.evaluate(d) for d in ('Toys','Beauty')}
        for p,h in self.inputs.items():
            assert sha(ROOT/p) == h, p
        passed = all(c['pass_screen'] for r in results.values() for c in r['source_checks'].values())
        outputs = {str(p.relative_to(ROOT)):sha(p) for p in OUT.rglob('*') if p.is_file() and 'sources' not in p.parts and p.name not in ('status.json','run.log')}
        summary = dict(state='COMPLETED', completed_at=now(), seconds=time.monotonic()-self.started,
            models_fitted=self.count, optimizer_updates=self.count*30, domains=results,
            source_information_screen_passed=passed, input_sha256=self.inputs, output_sha256=outputs,
            gpu_used=False, test_read=False, formal_method_evaluation=False, development_labels_used_for_fit=True)
        write(OUT/'summary.json', summary)
        self.status('筛查完成，保留全部对照', state='COMPLETED', source_information_screen_passed=passed)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    runner = Run()
    def timeout(*_):
        raise TimeoutError('Fixed 900-second CPU diagnostic budget exceeded')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(900)
    write(OUT/'launch.json', dict(pid=os.getpid(), started_at=now(), max_seconds=900, gpu_used=False))
    with threadpool_limits(4), (OUT/'run.log').open('x',buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            runner.run()
        except BaseException as e:
            runner.status('运行失败，保留记录', state='FAILED', error=repr(e))
            (OUT/'failure.txt').write_text(traceback.format_exc())
            raise
        finally:
            signal.alarm(0)


if __name__ == '__main__':
    main()
