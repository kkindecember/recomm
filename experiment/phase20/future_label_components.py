"""Frozen 2x2 soft-label mass/placement ablation, with leave-user-out priors."""
import argparse
import json
import os
import signal
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiment.phase20.future_factor_screen import Factor, old_score, z, ranks, metrics, sha

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/phase20/future_label_components_20260917'
PARENT = ROOT / 'artifacts/phase20/future_factor_screen_20260917'
SPECIFICITY = ROOT / 'artifacts/phase20/future_label_specificity_20260917'
CONTRACT = ROOT / 'plan/第二十阶段/去除逐前缀未来信息_质量与位置拆解合同_20260917.md'
ARMS = ('global_uniform', 'global_local', 'rowmass_globalshape',
        'globalmass_rowshape', 'rowmass_rowshape')


def negative_order(prior, positions):
    output = []
    for row, p in zip(prior, positions):
        negatives = np.flatnonzero(np.arange(50) != p)
        output.append(negatives[np.argsort(-row[negatives], kind='stable')])
    return np.asarray(output)


def leave_user_out(bin_mass, users, head):
    """User's own labels do not enter numerator OR denominator of their prior."""
    unique, inverse = np.unique(users, return_inverse=True)
    count = np.zeros((len(unique), 2), np.int64)
    total = np.zeros((len(unique), 2, 7), np.float64)
    groups = np.asarray(head, np.int64)
    np.add.at(count, (inverse, groups), 1)
    np.add.at(total, (inverse, groups), bin_mass)
    remaining_count = count.sum(0)[None] - count
    remaining_total = total.sum(0)[None] - total
    if np.any(remaining_count <= 0):
        raise ValueError('Cannot estimate both frequency groups without query user')
    per_user = (remaining_total / remaining_count[:, :, None]).mean(1)
    assert np.all(per_user >= -1e-12)
    per_user = np.maximum(per_user, 0)
    return per_user[inverse]


def global_targets(positions, order, prior_bins, uniform=False):
    """No future labels argument. Only the current target and other-user prior."""
    values = prior_bins.copy()
    if uniform:
        values[:] = values.sum(1, keepdims=True) / 7
    y = np.zeros((len(positions), 50), np.float64)
    y[np.arange(len(y))[:, None], order] = np.repeat(values / 7, 7, axis=1)
    y[np.arange(len(y)), positions] = 1
    return y


def build_targets(positions, distances, prior, users, head):
    order = negative_order(prior, positions)
    immediate = np.arange(50)[None] == positions[:, None]
    future = ((distances > 0) & (distances <= 3) & ~immediate) / (np.maximum(distances, 0) + 1)
    bin_mass = np.take_along_axis(future, order, 1).reshape(-1, 7, 7).sum(2)
    priors = leave_user_out(bin_mass, users, head)
    global_mass = priors.sum(1, keepdims=True)
    row_mass = bin_mass.sum(1, keepdims=True)
    assert np.all(global_mass > 0)
    global_shape = priors / global_mass
    row_shape = np.divide(bin_mass, row_mass, out=global_shape.copy(), where=row_mass > 0)
    targets = {
        'global_uniform': global_targets(positions, order, priors, uniform=True),
        'global_local': global_targets(positions, order, priors),
        'rowmass_globalshape': global_targets(positions, order, row_mass * global_shape),
        'globalmass_rowshape': global_targets(positions, order, global_mass * row_shape),
        'rowmass_rowshape': global_targets(positions, order, bin_mass),
    }
    for arm, y in targets.items():
        assert np.isfinite(y).all() and (y >= 0).all() and (y <= 1).all()
        np.testing.assert_array_equal(y[immediate], 1)
        expected = row_mass if arm in ('rowmass_globalshape', 'rowmass_rowshape') else global_mass
        np.testing.assert_allclose(y.sum(1) - 1, expected[:, 0], atol=1e-12, rtol=0)
    return targets, dict(order=order, bin_mass=bin_mass, prior_bins=priors,
        row_mass=row_mass[:, 0], global_mass=global_mass[:, 0], actual=future + immediate)


def checks():
    rng = np.random.RandomState(42)
    n = 12; positions = rng.randint(0, 50, n); prior = rng.normal(size=(n, 50))
    distances = np.full((n, 50), -1, np.int64)
    distances[1:, 7] = 1; distances[2:, 23] = 2; distances[4:, 41] = 3
    users = np.repeat(np.array(['a', 'b', 'c', 'd']), 3)
    head = np.arange(n) % 2 == 0
    targets, stats = build_targets(positions, distances, prior, users, head)
    for user in np.unique(users):
        other = users != user
        expected = np.stack([stats['bin_mass'][other & (head == h)].mean(0) for h in (False, True)]).mean(0)
        np.testing.assert_allclose(stats['prior_bins'][users == user], np.broadcast_to(expected, (3, 7)), atol=1e-12, rtol=0)
    changed = distances.copy(); changed[users == 'a'] = 2
    alternative, _ = build_targets(positions, changed, prior, users, head)
    for arm in ('global_uniform', 'global_local'):
        np.testing.assert_allclose(targets[arm][users == 'a'], alternative[arm][users == 'a'], atol=1e-12, rtol=0)
    permutation = rng.permutation(50); inverse = np.argsort(permutation)
    reordered, _ = build_targets(inverse[positions], distances[:, permutation], prior[:, permutation], users, head)
    for arm in ARMS:
        np.testing.assert_allclose(targets[arm][:, permutation], reordered[arm], atol=1e-12, rtol=0)
    zero = stats['row_mass'] == 0
    assert zero.any()
    np.testing.assert_allclose(targets['globalmass_rowshape'][zero], targets['global_local'][zero], atol=1e-12, rtol=0)
    assert (targets['rowmass_rowshape'][zero].sum(1) == 1).all()
    torch.manual_seed(20260917); model = Factor('next_only')
    assert sum(p.numel() for p in model.parameters()) == 258
    x = torch.tensor(rng.normal(size=(n, 50, 6)), dtype=torch.float32)
    score, _ = model(x, torch.tensor(prior))
    loss = F.binary_cross_entropy_with_logits(score, torch.tensor(targets['global_local']))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    return dict(leave_user_out_matches_explicit=True, own_future_intervention_invariance=True,
        candidate_permutation_equivariance=True, mass_and_target_invariants=True,
        zero_mass_fallback=True, parameters=258, finite_gradients=True)


def comparison(a, b):
    na = ((a <= 10) / np.log2(a + 1)).mean()
    nb = ((b <= 10) / np.log2(b + 1)).mean()
    return dict(delta_N10=float(na - nb), relative_N10=float(na / nb - 1),
        hit_gained=int(((a <= 10) & (b > 10)).sum()), hit_lost=int(((a > 10) & (b <= 10)).sum()))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--check-only', action='store_true'); args = parser.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    health = checks()
    if args.check_only:
        print(json.dumps(health)); return
    OUT.mkdir(exist_ok=False)
    start = time.monotonic(); steps = 0; hashes = {}; result = {}

    def use(p):
        hashes[str(p.relative_to(ROOT))] = sha(p); return p

    def write(path, payload):
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n'); temporary.replace(path)

    def status(state, **kw):
        write(OUT / 'status.json', dict(state=state, updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            elapsed_seconds=time.monotonic() - start, pid=os.getpid(), training_steps=steps,
            gpu_used=False, test_read=False, automatic_next_experiment=False, **kw))

    def timeout(sig, frame):
        raise TimeoutError('15 minute CPU wall budget')

    signal.signal(signal.SIGALRM, timeout); signal.alarm(900)
    try:
        status('RUNNING', stage='verify')
        use(Path(__file__)); use(CONTRACT); use(ROOT / 'experiment/phase20/future_factor_screen.py')
        parent = json.loads(use(PARENT / 'summary.json').read_text())
        audit = json.loads(use(PARENT / 'verification.json').read_text())
        previous = json.loads(use(SPECIFICITY / 'summary.json').read_text())
        previous_audit = json.loads(use(SPECIFICITY / 'verification.json').read_text())
        assert audit['state'] == previous_audit['state'] == 'VERIFIED'
        for group in (parent['input_sha256'], audit['output_sha256'], previous['input_sha256']):
            for p, h in group.items():
                if p not in hashes: use(ROOT / p)
                assert hashes[p] == h, p
        write(OUT / 'protocol.json', dict(contract=str(CONTRACT.relative_to(ROOT)), checks=health,
            arms=ARMS, seed=20260917, steps_per_arm=30, threads=4, timeout_seconds=900,
            estimator='leave_entire_user_out_group_balanced_mean', source_sha256=hashes))
        for domain in ('Toys', 'Beauty'):
            status('RUNNING', dataset=domain, stage='load_and_build_labels')
            old = ROOT / f'artifacts/phase20/gacr_fusion_train_v1/{domain}'
            with np.load(use(old / 'training_cache.npz')) as f: a = {k: f[k] for k in f.files}
            with np.load(use(ROOT / f'artifacts/phase20/future_horizon_audit_20260917/{domain}.labels.npz')) as f: b = {k: f[k] for k in f.files}
            for k in ('candidate_ids', 'positions', 'fit'): np.testing.assert_array_equal(a[k], b[k])
            valid = a['fit'] & (a['positions'] >= 0) & (b['remaining'] >= 3)
            head = a['head'][valid]; users = b['users'][valid]
            xall = torch.as_tensor(a['features'])
            g = old_score(xall, torch.as_tensor(a['base']), torch.load(use(old / 'B1.pt'), map_location='cpu'))
            prior = z(g[valid]) - np.log(49)
            targets, stats = build_targets(a['positions'][valid], b['distances'][valid], prior, users, head)
            # Match earlier true labels exactly before using any new outcome.
            with np.load(use(SPECIFICITY / f'{domain}.training_labels.npz')) as f:
                np.testing.assert_array_equal(stats['actual'], f['actual'])
            np.savez_compressed(OUT / f'{domain}.training_labels.npz', **targets)
            np.savez_compressed(OUT / f'{domain}.label_statistics.npz', users=users, head=head,
                positions=a['positions'][valid], fit_indices=np.flatnonzero(valid), **stats)
            x = xall[valid]; prior = torch.as_tensor(prior)
            weights = torch.as_tensor(np.where(head, 1 / (2 * head.sum()), 1 / (2 * (~head).sum())))
            models = {}; curves = {}
            for arm in ARMS:
                torch.manual_seed(20260917); model = Factor('next_only')
                optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.01)
                y = torch.as_tensor(targets[arm]); curve = []
                for step in range(1, 31):
                    optimizer.zero_grad(set_to_none=True); total_loss = 0.
                    for offset in range(0, len(x), 256):
                        sl = slice(offset, offset + 256)
                        scores, _ = model(x[sl], prior[sl])
                        loss = (F.binary_cross_entropy_with_logits(scores, y[sl], reduction='none').mean(1) * weights[sl]).sum()
                        assert torch.isfinite(loss); loss.backward(); total_loss += float(loss.detach())
                    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
                    optimizer.step(); steps += 1
                    curve.append(dict(step=step, loss=total_loss, grad_norm=float(grad)))
                    status('RUNNING', dataset=domain, stage='training', arm=arm, step=step, total_steps=30)
                models[arm] = model.eval(); curves[arm] = curve
                torch.save(model.state_dict(), OUT / f'{domain}.{arm}.pt')
            write(OUT / f'{domain}.training.json', dict(curves=curves, prefixes=len(x), users=len(set(users)),
                zero_future_prefixes=int((stats['row_mass'] == 0).sum()),
                weighted_mean_bins=np.average(stats['bin_mass'], axis=0, weights=weights.numpy()).tolist()))
            with np.load(use(PARENT / f'{domain}.audit_validation_cache.npz')) as f: v = {k: f[k] for k in f.files}
            with np.load(use(SPECIFICITY / f'{domain}.predictions.npz')) as f: prev = {k: f[k] for k in f.files}
            np.testing.assert_array_equal(v['users'], prev['users']); np.testing.assert_array_equal(v['candidate_ids'], prev['candidate_ids'])
            fit_users = set(users)
            in_fit = np.asarray([u in fit_users for u in v['users']])
            np.testing.assert_array_equal(in_fit, prev['in_fit'])
            out_scores = {}; out_ranks = {}
            for arm, model in models.items():
                status('RUNNING', dataset=domain, stage='validation', arm=arm)
                chunks = []
                with torch.no_grad():
                    for offset in range(0, len(v['users']), 256):
                        sl = slice(offset, offset + 256)
                        scores, _ = model(torch.as_tensor(v['features'][sl]), torch.as_tensor(v['prior'][sl]))
                        chunks.append(scores.numpy())
                raw = z(np.concatenate(chunks))
                for fusion in ('raw', 'current', 'mean'):
                    name = arm + '.' + fusion
                    out_scores[name] = raw if fusion == 'raw' else raw + v['cf_' + fusion]
                    out_ranks[name] = ranks(out_scores[name], v['positions'])
            for arm in ('soft_future', 'next_only', 'score_matched_shuffle', 'frozen'):
                for fusion in ('current', 'mean'):
                    name = arm + '.' + fusion; j = list(prev['arm_names']).index(name)
                    out_scores[name] = prev['scores'][j]; out_ranks[name] = prev['ranks'][j]
            masks = dict(all=np.ones(len(in_fit), bool), tail=v['tail_mask'], non_tail=~v['tail_mask'], fit=in_fit, nonfit=~in_fit)
            met = {name: {group: metrics(rr[mask]) for group, mask in masks.items()} for name, rr in out_ranks.items()}
            contrasts = [('soft_future', 'global_local'), ('rowmass_rowshape', 'global_local'),
                ('global_local', 'global_uniform'), ('rowmass_globalshape', 'global_local'),
                ('globalmass_rowshape', 'global_local'), ('rowmass_rowshape', 'globalmass_rowshape'),
                ('rowmass_rowshape', 'rowmass_globalshape'), ('rowmass_rowshape', 'soft_future'),
                ('global_local', 'next_only')]
            comps = {a + '_vs_' + b: {group: comparison(out_ranks[a + '.mean'][mask], out_ranks[b + '.mean'][mask])
                for group, mask in masks.items()} for a, b in contrasts}
            primary = comps['soft_future_vs_global_local']
            future_passed = primary['all']['relative_N10'] >= .005 and primary['nonfit']['relative_N10'] > 0
            pooled = met['rowmass_rowshape.mean']['all']
            controls = [a for a in ARMS if a != 'rowmass_rowshape'] + ['soft_future', 'next_only']
            group_passed = (comps['rowmass_rowshape_vs_global_local']['all']['relative_N10'] >= .005 and
                all(pooled['N10'] > met[a + '.mean']['all']['N10'] for a in controls) and
                all(pooled['H10'] >= met[a + '.mean']['all']['H10'] for a in ('global_local', 'next_only')))
            result[domain] = dict(metrics=met, comparisons=comps, future_passed=future_passed, group_passed=group_passed)
            names = list(out_scores)
            np.savez_compressed(OUT / f'{domain}.predictions.npz', users=v['users'], candidate_ids=v['candidate_ids'],
                positions=v['positions'], tail_mask=v['tail_mask'], in_fit=in_fit, arm_names=np.asarray(names),
                scores=np.stack([out_scores[n] for n in names]), ranks=np.stack([out_ranks[n] for n in names]))
            write(OUT / f'{domain}.summary.json', result[domain])
        for p, h in hashes.items(): assert sha(ROOT / p) == h, p
        decisions = dict(per_prefix_future_supported=all(v['future_passed'] for v in result.values()),
            group_supervision_followup_supported=all(v['group_passed'] for v in result.values()))
        write(OUT / 'summary.json', dict(state='COMPLETED', domains=result, decisions=decisions, checks=health,
            input_sha256=hashes, development_reused=True, test_read=False, training_steps=steps))
        status('COMPLETED', stage='finished', decisions=decisions)
        print(json.dumps(decisions, indent=2))
    except Exception as error:
        status('FAILED', error=repr(error)); raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
