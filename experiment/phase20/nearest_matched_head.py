"""Registered ordinary-head controls; fixed upstream caches, 12 heads / 360 updates.

All new heads finish fitting before their development metrics are computed.
--verify replays checkpoints and independently recomputes stable target ranks.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import signal
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiment.phase20.future_factor_screen import Factor, z, ranks, metrics, sha
from experiment.phase20.future_fusion_alignment import standardize
from experiment.phase20.future_label_multiseed import write, fingerprint, describe, compare
from experiment.phase20.talos_candidate_screen import row_loss

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/phase20/nearest_matched_head_20260918'
DESIGN = ROOT / 'experiment/phase20/nearest_matched_head_plan_v1.json'
CONTRACT = ROOT / 'plan/第二十阶段/当前完整方案与近邻方法_公平对照准备_20260918.md'
REFERENCE = ROOT / 'artifacts/phase20/talos_candidate_screen_20260917'
CONTEXT = ROOT / 'artifacts/phase20/future_fusion_alignment_20260917'
VALIDATION = ROOT / 'artifacts/phase20/future_factor_screen_20260917'
PAIR = ROOT / 'artifacts/phase20/pair_uncertainty_screen_20260917'
SEEDS = (20260917, 20260918, 20260919)
DOMAINS = ('Toys', 'Beauty')
NEW_ARMS = ('linear_residual', 'ordinary_mlp32')
ARMS = ('frozen_B1', *NEW_ARMS, 'factor_softmax')
COUNTS = dict(linear_residual=7, ordinary_mlp32=257, factor_softmax=258)
CONTRASTS = [(a, 'factor_softmax') for a in NEW_ARMS] + [(a, 'frozen_B1') for a in ARMS[1:]]
METRICS = tuple(f'{m}{k}' for k in (5, 10, 20, 50) for m in ('N', 'H'))


class OrdinaryHead(nn.Module):
    def __init__(self, arm):
        super().__init__()
        if arm == 'linear_residual':
            self.network = nn.Sequential(nn.Linear(6, 1))
        elif arm == 'ordinary_mlp32':
            self.network = nn.Sequential(nn.Linear(6, 32), nn.Tanh(), nn.Linear(32, 1))
        else:
            raise ValueError(arm)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features, prior):
        return prior + (2 * torch.tanh(self.network(features).squeeze(-1))).double(), None


def model_for(arm):
    return Factor('next_only') if arm == 'factor_softmax' else OrdinaryHead(arm)


def checks():
    """Check the scientific implementation, including chunk gradient accumulation."""
    rng = np.random.RandomState(721)
    x = torch.tensor(rng.normal(size=(8, 50, 6)), dtype=torch.float32)
    p = torch.tensor(z(rng.normal(size=(8, 50))) - np.log(49))
    cf = torch.tensor(rng.normal(size=(8, 50)))
    pos = torch.tensor([0, 2, 4, 9, 15, 23, 37, 49])
    weights = torch.tensor([.1] * 5 + [1 / 6] * 3)
    updates = 0
    for arm in COUNTS:
        gradients = []
        for chunk in (8, 3):
            torch.manual_seed(SEEDS[0])
            model = model_for(arm)
            assert sum(p.numel() for p in model.parameters()) == COUNTS[arm]
            for start in range(0, len(x), chunk):
                sl = slice(start, start + chunk)
                raw, _ = model(x[sl], p[sl])
                np.testing.assert_allclose(raw.detach(), p[sl], atol=1e-12, rtol=0)
                score = standardize(raw) + cf[sl]
                loss = row_loss(score, pos[sl], 'softmax')
                np.testing.assert_allclose(loss.detach(), F.cross_entropy(score, pos[sl], reduction='none').detach(), atol=1e-12, rtol=0)
                (loss * weights[sl]).sum().div(50).backward()
            gradients.append(np.concatenate([p.grad.numpy().ravel() for p in model.parameters()]))
            assert np.isfinite(gradients[-1]).all()
        np.testing.assert_allclose(*gradients, atol=1e-7, rtol=1e-5)
        before = fingerprint(model)
        torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.01).step()
        updates += 1
        assert fingerprint(model) != before
        with torch.no_grad():
            score, _ = model(x, p)
            if arm != 'factor_softmax':
                assert torch.max(torch.abs(score - p)) <= 2
    synthetic = np.array([[3., 3., 1.], [0., 0., 0.], [2., 4., 1.]])
    np.testing.assert_array_equal(independent_ranks(synthetic, np.array([1, 2, -1])), [2, 3, 51])
    return dict(parameter_counts=COUNTS, common_initial_output=True,
                full_vs_chunk_gradients=True, ordinary_cross_entropy_parity=True,
                finite_gradients=True, residual_bound=True, tie_and_missing_rank=True,
                discarded_synthetic_optimizer_updates=updates)


def independent_ranks(scores, positions):
    """Count candidates ahead of the target, including the stable tie rule."""
    col = np.arange(scores.shape[1])[None]
    target = scores[np.arange(len(scores)), np.maximum(positions, 0)][:, None]
    ahead = (scores > target) | ((scores == target) & (col < positions[:, None]))
    return np.where(positions >= 0, 1 + ahead.sum(1), 51)


def load_npz(path, keys=None):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in (data.files if keys is None else keys)}


def read_data(domain):
    cache = load_npz(ROOT / f'artifacts/phase20/gacr_fusion_train_v1/{domain}/training_cache.npz')
    ctx = load_npz(CONTEXT / f'{domain}.training_context.npz')
    ix = ctx['fit_indices']
    pos = cache['positions'][ix]
    assert len(ix) == dict(Toys=6884, Beauty=8309)[domain]
    assert len(np.unique(ix)) == len(ix) and cache['fit'][ix].all() and (pos >= 0).all()
    np.testing.assert_array_equal(cache['candidate_ids'][ix], ctx['candidate_ids'])
    np.testing.assert_array_equal(ctx['targets'], ctx['candidate_ids'][np.arange(len(ix)), pos])
    head = cache['head'][ix]
    weights = np.where(head, .5 / head.sum(), .5 / (~head).sum())
    np.testing.assert_allclose([weights[head].sum(), weights[~head].sum()], [.5, .5], atol=1e-14, rtol=0)
    train = dict(x=torch.as_tensor(cache['features'][ix]), prior=torch.as_tensor(ctx['prior']),
                 cf=torch.as_tensor(ctx['cf_mean']), pos=torch.as_tensor(pos), weights=torch.as_tensor(weights))
    v = load_npz(VALIDATION / f'{domain}.audit_validation_cache.npz')
    v['in_fit'] = np.isin(v['users'], np.unique(ctx['users']))
    assert len(set(v['users'])) == len(v['users'])
    for tensor in train.values():
        assert torch.isfinite(tensor).all()
    masks = dict(all=np.ones(len(v['users']), bool), tail=v['tail_mask'], non_tail=~v['tail_mask'],
                 fit=v['in_fit'], nonfit=~v['in_fit'])
    metadata = dict(fit_prefixes=len(ix), fit_users=len(np.unique(ctx['users'])),
                    head_prefixes=int(head.sum()), tail_prefixes=int((~head).sum()),
                    development_users=len(v['users']), covered_users=int((v['positions'] >= 0).sum()),
                    group_users={k: int(m.sum()) for k, m in masks.items()})
    return train, v, masks, metadata


def predict(model, v):
    model.eval()
    pred = []
    with torch.no_grad():
        for start in range(0, len(v['users']), 256):
            sl = slice(start, start + 256)
            values, _ = model(torch.as_tensor(v['features'][sl]), torch.as_tensor(v['prior'][sl]))
            pred.append(values.numpy())
    return z(np.concatenate(pred))


def final_loss(model, train):
    model.eval()
    total = 0.
    with torch.no_grad():
        for start in range(0, len(train['x']), 256):
            sl = slice(start, start + 256)
            values, _ = model(train['x'][sl], train['prior'][sl])
            score = standardize(values) + train['cf'][sl]
            total += float((F.cross_entropy(score, train['pos'][sl], reduction='none') * train['weights'][sl]).sum())
    return total


def summarize_run(rr, masks):
    return dict(metrics={a: {g: metrics(r[m]) for g, m in masks.items()} for a, r in rr.items()},
                comparisons={a + '__vs__' + b: {g: compare(rr[a][m], rr[b][m]) for g, m in masks.items()}
                             for a, b in CONTRASTS})


def summarize_domain(runs, metadata):
    sample = next(iter(runs.values()))
    groups = metadata['group_users']
    return dict(metadata=metadata, seeds=runs,
                aggregate_metrics={a: {g: {k: describe([r['metrics'][a][g][k] for r in runs.values()])
                                          for k in METRICS} for g in groups} for a in ARMS},
                aggregate_comparisons={c: {g: {k: describe([r['comparisons'][c][g][k] for r in runs.values()])
                                              for k in sample['comparisons'][c][g]} for g in groups}
                                       for c in sample['comparisons']})


def verify():
    started = time.monotonic()
    summary = json.loads((OUT / 'summary.json').read_text())
    assert summary['state'] == 'COMPLETED' and summary['training_steps'] == 360
    assert not summary['test_read']
    for mapping in (summary['input_sha256'], summary['output_sha256']):
        for p, h in mapping.items():
            assert sha(ROOT / p) == h, p
    total = 0
    errors = {}
    for domain in DOMAINS:
        train, v, masks, meta = read_data(domain)
        runs = {}
        for seed in SEEDS:
            saved = load_npz(OUT / f'{domain}.{seed}.predictions.npz')
            for k in ('users', 'candidate_ids', 'positions', 'tail_mask', 'in_fit'):
                np.testing.assert_array_equal(saved[k], v[k])
            assert saved['arm_names'].tolist() == list(ARMS)
            training = json.loads((OUT / f'{domain}.{seed}.training.json').read_text())
            rr = {}
            for i, arm in enumerate(ARMS):
                if arm == 'frozen_B1':
                    raw = z(v['prior'])
                else:
                    model = model_for(arm)
                    path = (REFERENCE / f'{domain}.{seed}.softmax.pt') if arm == 'factor_softmax' else (OUT / f'{domain}.{seed}.{arm}.pt')
                    model.load_state_dict(torch.load(path, map_location='cpu'), strict=True)
                    raw = predict(model, v)
                    if arm in NEW_ARMS:
                        loss = final_loss(model, train)
                        np.testing.assert_allclose(loss, training[arm]['final_loss'], atol=1e-12, rtol=0)
                        assert len(training[arm]['curve']) == 30
                error = float(np.max(np.abs(raw - saved['raw_scores'][i])))
                assert error <= 1e-12
                errors[f'{domain}.{seed}.{arm}'] = error
                score = raw + v['cf_mean']
                rr[arm] = independent_ranks(score, v['positions'])
                np.testing.assert_array_equal(rr[arm], saved['ranks'][i])
                # Independently compute all eight metrics on every group.
                for group, mask in masks.items():
                    rank = rr[arm][mask]
                    for k in (5, 10, 20, 50):
                        hit = rank <= k
                        n = float(np.sum(1 / np.log2(rank[hit] + 1)) / len(rank))
                        h = float(np.count_nonzero(hit) / len(rank))
                        expected = summary['domains'][domain]['seeds'][str(seed)]['metrics'][arm][group]
                        np.testing.assert_allclose([n, h], [expected[f'N{k}'], expected[f'H{k}']], atol=1e-15, rtol=0)
                total += len(rr[arm])
            runs[str(seed)] = summarize_run(rr, masks)
        assert summarize_domain(runs, meta) == summary['domains'][domain]
    result = dict(state='VERIFIED_CHECKPOINT_REPLAY', elapsed_seconds=time.monotonic() - started,
                  full_training_repeated=False, checkpoint_max_errors=errors, new_checkpoints_replayed=12,
                  existing_checkpoints_replayed=6, independently_recomputed_target_ranks=total,
                  input_hashes=len(summary['input_sha256']), output_hashes=len(summary['output_sha256']),
                  all_metrics_groups_and_paired_summaries_checked=True, test_read=False)
    write(OUT / 'verification.json', result)
    print(json.dumps(result), flush=True)


def run():
    health = checks()
    config = json.loads(DESIGN.read_text())
    assert config['formal_updates'] == 360 and config['new_arms'] == {a: COUNTS[a] for a in NEW_ARMS}
    assert config['head_seeds'] == list(SEEDS) and config['domains'] == list(DOMAINS)
    assert (config['loss_divisor'], config['learning_rate'], config['weight_decay'], config['gradient_clip'],
            config['full_fit_updates'], config['chunk_size']) == (50, .01, .01, 10, 30, 256)
    OUT.mkdir(exist_ok=False)
    started = time.monotonic()
    steps = 0
    hashes = {}

    def use(path):
        key = str(path.relative_to(ROOT))
        if key not in hashes:
            hashes[key] = sha(path)
        return path

    def status(state, **kwargs):
        write(OUT / 'status.json', dict(state=state, elapsed_seconds=time.monotonic() - started,
            updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), pid=os.getpid(), training_steps=steps,
            planned_training_steps=360, gpu_used=False, test_read=False, automatic_next_experiment=False, **kwargs))

    def timeout(sig, frame):
        raise TimeoutError('Registered 1800 second CPU execution cap')

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(1800)
    try:
        status('RUNNING', stage='input_and_reference_verification')
        for path in (Path(__file__), DESIGN, CONTRACT,
                     ROOT / 'experiment/phase20/future_factor_screen.py',
                     ROOT / 'experiment/phase20/future_fusion_alignment.py',
                     ROOT / 'experiment/phase20/future_label_multiseed.py',
                     ROOT / 'experiment/phase20/talos_candidate_screen.py'):
            use(path)
        old = json.loads(use(REFERENCE / 'summary.json').read_text())
        audit = json.loads(use(REFERENCE / 'verification.json').read_text())
        assert old['state'] == 'COMPLETED' and audit['state'] == 'VERIFIED'
        for mapping in (old['input_sha256'], audit['output_sha256']):
            for p, h in mapping.items():
                assert sha(use(ROOT / p)) == h, p
        bank = {}
        reference_ranks = {}
        preflight = {}
        for domain in DOMAINS:
            for path in (ROOT / f'artifacts/phase20/gacr_fusion_train_v1/{domain}/training_cache.npz',
                         CONTEXT / f'{domain}.training_context.npz', VALIDATION / f'{domain}.audit_validation_cache.npz',
                         PAIR / f'{domain}.predictions.npz'):
                use(path)
            train, v, masks, meta = read_data(domain)
            bank[domain] = (train, v, masks, meta)
            pair = load_npz(PAIR / f'{domain}.predictions.npz', ('users', 'candidate_ids', 'positions', 'arm_names', 'ranks'))
            for k in ('users', 'candidate_ids', 'positions'):
                np.testing.assert_array_equal(v[k], pair[k])
            rr0 = ranks(z(v['prior']) + v['cf_mean'], v['positions'])
            np.testing.assert_array_equal(rr0, pair['ranks'][pair['arm_names'].tolist().index('mean')])
            for seed in SEEDS:
                old_pred = load_npz(use(REFERENCE / f'{domain}.{seed}.predictions.npz'))
                for k in ('users', 'candidate_ids', 'positions', 'tail_mask', 'in_fit'):
                    np.testing.assert_array_equal(v[k], old_pred[k])
                model = model_for('factor_softmax')
                model.load_state_dict(torch.load(use(REFERENCE / f'{domain}.{seed}.softmax.pt'), map_location='cpu'))
                raw = predict(model, v)
                saved_raw = old_pred['raw_scores'][old_pred['raw_names'].tolist().index('softmax')]
                np.testing.assert_allclose(raw, saved_raw, atol=1e-12, rtol=0)
                rr = ranks(raw + v['cf_mean'], v['positions'])
                np.testing.assert_array_equal(rr, old_pred['ranks'][old_pred['rank_names'].tolist().index('softmax.mean')])
                assert metrics(rr) == old['domains'][domain]['seeds'][str(seed)]['metrics']['softmax.mean']['all']
                reference_ranks[(domain, seed)] = rr
                preflight[f'{domain}.{seed}'] = dict(reference_exact_ranks=True, reference_max_score_error=float(np.max(np.abs(raw - saved_raw))))
                for arm in NEW_ARMS:
                    torch.manual_seed(seed)
                    initial = model_for(arm)
                    # Check all actual FIT priors; no development metric for a newly fitted head.
                    with torch.no_grad():
                        for start in range(0, len(train['x']), 256):
                            sl = slice(start, start + 256)
                            score, _ = initial(train['x'][sl], train['prior'][sl])
                            np.testing.assert_array_equal(score.numpy(), train['prior'][sl].numpy())
            del pair, old_pred
        snapshot = OUT / 'source_snapshot'
        snapshot.mkdir()
        for path in (Path(__file__), DESIGN, CONTRACT):
            (snapshot / path.name).write_bytes(path.read_bytes())
        write(OUT / 'protocol.json', dict(design=config, implementation=str(Path(__file__).relative_to(ROOT)),
            arms=ARMS, new_arms=NEW_ARMS, contrasts=CONTRASTS, primary='N10', deployment='mean_CF',
            checks=health, reference_replay=preflight, runtime=dict(python=platform.python_version(),
            torch=torch.__version__, numpy=np.__version__, threads=4), input_sha256=hashes,
            train_all_before_new_development_scoring=True, development_reused=True, test_read=False))
        # Fit all 12 models without reading new development outcomes.
        trained = {}
        for domain in DOMAINS:
            train = bank[domain][0]
            for seed in SEEDS:
                details = {}
                for arm in NEW_ARMS:
                    torch.manual_seed(seed)
                    model = model_for(arm)
                    initial = fingerprint(model)
                    initial_state = {k: v.clone() for k, v in model.state_dict().items()}
                    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.01)
                    curve = []
                    for step in range(1, 31):
                        model.train()
                        optimizer.zero_grad(set_to_none=True)
                        total = 0.
                        for start in range(0, len(train['x']), 256):
                            sl = slice(start, start + 256)
                            values, _ = model(train['x'][sl], train['prior'][sl])
                            score = standardize(values) + train['cf'][sl]
                            loss = (row_loss(score, train['pos'][sl], 'softmax') * train['weights'][sl]).sum() / 50
                            assert torch.isfinite(loss)
                            loss.backward()
                            total += float(loss.detach())
                        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
                        optimizer.step()
                        steps += 1
                        curve.append(dict(step=step, scaled_loss=total, loss=total * 50, grad_norm=float(grad)))
                        status('RUNNING', stage='training', domain=domain, seed=seed, arm=arm, step=step)
                    delta = float(torch.sqrt(sum((v - initial_state[k]).square().sum() for k, v in model.state_dict().items())))
                    assert np.isfinite(delta) and delta > 0
                    details[arm] = dict(initial_sha256=initial, final_sha256=fingerprint(model), curve=curve,
                                        final_loss=final_loss(model, train), parameter_delta_l2=delta,
                                        parameters=COUNTS[arm], updates=30)
                    torch.save(model.state_dict(), OUT / f'{domain}.{seed}.{arm}.pt')
                    trained[(domain, seed, arm)] = model.eval()
                write(OUT / f'{domain}.{seed}.training.json', details)
        assert steps == 360
        write(OUT / 'training_complete.json', dict(training_steps=steps, new_checkpoints=12,
            elapsed_seconds=time.monotonic() - started, new_development_predictions_computed=False))
        results = {}
        for domain in DOMAINS:
            _, v, masks, meta = bank[domain]
            runs = {}
            for seed in SEEDS:
                status('RUNNING', stage='fixed_final_validation', domain=domain, seed=seed)
                raw = {'frozen_B1': z(v['prior'])}
                for arm in NEW_ARMS:
                    raw[arm] = predict(trained[(domain, seed, arm)], v)
                model = model_for('factor_softmax')
                model.load_state_dict(torch.load(REFERENCE / f'{domain}.{seed}.softmax.pt', map_location='cpu'))
                raw['factor_softmax'] = predict(model, v)
                rr = {a: ranks(r + v['cf_mean'], v['positions']) for a, r in raw.items()}
                np.testing.assert_array_equal(rr['factor_softmax'], reference_ranks[(domain, seed)])
                runs[str(seed)] = summarize_run(rr, masks)
                np.savez_compressed(OUT / f'{domain}.{seed}.predictions.npz',
                    **{k: v[k] for k in ('users', 'candidate_ids', 'positions', 'tail_mask', 'in_fit')},
                    arm_names=np.asarray(ARMS), raw_scores=np.stack([raw[a] for a in ARMS]),
                    ranks=np.stack([rr[a] for a in ARMS]))
            results[domain] = summarize_domain(runs, meta)
            write(OUT / f'{domain}.summary.json', results[domain])
        for p, h in hashes.items():
            assert sha(ROOT / p) == h, p
        output_hashes = {str(p.relative_to(ROOT)): sha(p) for p in sorted(OUT.rglob('*'))
                         if p.is_file() and p.name != 'status.json'}
        write(OUT / 'summary.json', dict(state='COMPLETED', training_steps=steps, domains=results,
            input_sha256=hashes, output_sha256=output_hashes, checks=health,
            development_reused=True, test_read=False, automatic_followup=False,
            interpretation='Descriptive matched-recipe head comparison, not an equivalence or novelty test.'))
        status('COMPLETED', new_checkpoints=12)
        print(json.dumps({d: {a: {m: results[d]['aggregate_metrics'][a]['all'][m]['mean']
                                 for m in ('N10', 'H10')} for a in ARMS} for d in DOMAINS}), flush=True)
    except BaseException as exc:
        status('FAILED', error=repr(exc))
        raise
    finally:
        signal.alarm(0)


def main():
    parser = argparse.ArgumentParser()
    options = parser.add_mutually_exclusive_group()
    options.add_argument('--check-only', action='store_true')
    options.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if args.check_only:
        print(json.dumps(checks()))
    elif args.verify:
        verify()
    else:
        run()


if __name__ == '__main__':
    main()
