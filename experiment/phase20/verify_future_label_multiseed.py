"""Independent checkpoint/label replay and complete multiseed metric verification."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiment.phase20.future_factor_screen import Factor
from experiment.phase20.verify_future_label_components import replay
from experiment.phase20.verify_future_factor_screen import sha, z

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/phase20/future_label_multiseed_20260917'
COMP = ROOT / 'artifacts/phase20/future_label_components_20260917'
PARENT = ROOT / 'artifacts/phase20/future_factor_screen_20260917'
SEEDS = (20260917, 20260918, 20260919)
ARMS = ('next_only', 'global_local', 'soft_future')


def main():
    torch.set_num_threads(4)
    summary = json.loads((OUT / 'summary.json').read_text())
    protocol = json.loads((OUT / 'protocol.json').read_text())
    for p, h in summary['input_sha256'].items(): assert sha(ROOT / p) == h, p
    hashes = {}; scores_checked = 0; metric_count = 0; comparison_count = 0; aggregate_count = 0
    errors = {}; gates = {}; domains = {}

    def read(folder, name):
        p = folder / name
        if folder == OUT: hashes[str(p.relative_to(ROOT))] = sha(p)
        with np.load(p) as f: return {k: f[k] for k in f.files}

    init_hashes = {}
    for seed in SEEDS:
        torch.manual_seed(seed); model = Factor('next_only'); h = hashlib.sha256()
        for name, value in model.state_dict().items():
            if '.2.' in name: assert torch.count_nonzero(value) == 0
            h.update(name.encode()); h.update(value.numpy().tobytes())
        init_hashes[str(seed)] = h.hexdigest()
    assert init_hashes == protocol['initialization_sha256']; assert len(set(init_hashes.values())) == 3

    for domain in ('Toys', 'Beauty'):
        report = summary['domains'][domain]
        v = read(PARENT, f'{domain}.audit_validation_cache.npz')
        stats = read(COMP, f'{domain}.label_statistics.npz')
        ys = read(COMP, f'{domain}.training_labels.npz')
        future = read(ROOT / 'artifacts/phase20/future_horizon_audit_20260917', f'{domain}.labels.npz')
        cache = read(ROOT / f'artifacts/phase20/gacr_fusion_train_v1/{domain}', 'training_cache.npz')
        valid = future['fit'] & (future['positions'] >= 0) & (future['remaining'] >= 3)
        ix = np.flatnonzero(valid); np.testing.assert_array_equal(ix, stats['fit_indices'])
        np.testing.assert_array_equal(cache['positions'][ix], stats['positions'])
        np.testing.assert_array_equal(future['users'][ix], stats['users'])
        immediate = (np.arange(50)[None] == stats['positions'][:, None]).astype(float)
        dist = future['distances'][ix]
        actual = immediate + ((dist > 0) & (dist <= 3) & (immediate == 0)) / (np.maximum(dist, 0) + 1)
        np.testing.assert_array_equal(actual, stats['actual'])
        expected_local = immediate.copy()
        expected_local[np.arange(len(ix))[:, None], stats['order']] = np.repeat(stats['prior_bins'] / 7, 7, axis=1)
        np.testing.assert_array_equal(expected_local, ys['global_local'])
        state = torch.load(ROOT / f'artifacts/phase20/gacr_fusion_train_v1/{domain}/B1.pt', map_location='cpu')
        with torch.no_grad():
            x = torch.as_tensor(cache['features'][ix])
            hidden = torch.tanh(F.linear(x, state['network.0.weight'], state['network.0.bias']))
            g = (torch.as_tensor(cache['base'][ix]) + .2 * torch.tanh(F.linear(hidden, state['network.2.weight'], state['network.2.bias'])).squeeze(-1)).numpy().astype(float)
        prior = z(g) - np.log(49); head = cache['head'][ix]
        weights = np.where(head, .5 / head.sum(), .5 / (~head).sum())
        targets = dict(next_only=immediate, global_local=expected_local, soft_future=actual)
        initial_losses = {arm: float(((np.logaddexp(0, prior) - y * prior).mean(1) * weights).sum()) for arm, y in targets.items()}
        old_curves = json.loads((PARENT / f'{domain}.training.json').read_text())['curves']
        component_curves = json.loads((COMP / f'{domain}.training.json').read_text())['curves']
        for arm in ARMS:
            source = component_curves if arm == 'global_local' else old_curves
            assert abs(source[arm][0]['loss'] - initial_losses[arm]) < 1e-12
        frozen = read(COMP, f'{domain}.predictions.npz'); original = read(PARENT, f'{domain}.predictions.npz')
        all_ranks = {}; all_metrics = {}; all_comparisons = {}
        for seed in SEEDS:
            data = read(OUT, f'{domain}.{seed}.predictions.npz')
            for k in ('users', 'candidate_ids', 'positions', 'tail_mask'): np.testing.assert_array_equal(data[k], v[k])
            np.testing.assert_array_equal(data['in_fit'], np.isin(v['users'], np.unique(stats['users'])))
            names = list(data['arm_names'])
            assert set(names) == {a + '.' + f for a in ARMS for f in ('raw', 'current', 'mean')}
            training_path = OUT / f'{domain}.{seed}.training.json'
            train = json.loads(training_path.read_text()); hashes[str(training_path.relative_to(ROOT))] = sha(training_path)
            assert train['reused'] == (seed == SEEDS[0])
            if seed == SEEDS[0]: assert train['curves'] == {} and train['initial_sha256'] == {}
            for arm in ARMS:
                if seed == SEEDS[0]:
                    for fusion in ('raw', 'current', 'mean'):
                        name = arm + '.' + fusion
                        old = original if fusion == 'raw' and arm != 'global_local' else frozen
                        np.testing.assert_array_equal(data['scores'][names.index(name)], old['scores'][list(old['arm_names']).index(name)])
                    continue
                assert train['initial_sha256'][arm] == init_hashes[str(seed)]
                curve = train['curves'][arm]; assert len(curve) == 30
                assert [c['step'] for c in curve] == list(range(1, 31))
                assert all(np.isfinite([c['loss'], c['grad_norm']]).all() for c in curve)
                assert abs(curve[0]['loss'] - initial_losses[arm]) < 1e-12
                ckpt = OUT / f'{domain}.{seed}.{arm}.pt'; hashes[str(ckpt.relative_to(ROOT))] = sha(ckpt)
                raw = replay(torch.load(ckpt, map_location='cpu'), v['features'], v['prior'])
                errors[f'{domain}.{seed}.{arm}'] = 0.
                for fusion in ('raw', 'current', 'mean'):
                    predicted = raw if fusion == 'raw' else raw + v['cf_' + fusion]
                    err = float(np.max(np.abs(predicted - data['scores'][names.index(arm + '.' + fusion)])))
                    assert err < 1e-11; scores_checked += 1
                    errors[f'{domain}.{seed}.{arm}'] = max(errors[f'{domain}.{seed}.{arm}'], err)
            masks = dict(all=np.ones(len(v['users']), bool), tail=data['tail_mask'], non_tail=~data['tail_mask'], fit=data['in_fit'], nonfit=~data['in_fit'])
            rr = {}; mets = {}; comparisons = {}
            for j, name in enumerate(names):
                s = data['scores'][j]; pos = np.maximum(data['positions'], 0)
                gold = s[np.arange(len(pos)), pos, None]
                rank = np.where(data['positions'] >= 0, 1 + (s > gold).sum(1) + ((s == gold) & (np.arange(50)[None] < pos[:, None])).sum(1), 51)
                np.testing.assert_array_equal(rank, data['ranks'][j]); rr[name] = rank; mets[name] = {}
                for group, mask in masks.items():
                    m = {'users': int(mask.sum())}
                    for k in (5, 10, 20, 50):
                        m[f'H{k}'] = float((rank[mask] <= k).mean())
                        m[f'N{k}'] = float(((rank[mask] <= k) / np.log2(rank[mask] + 1)).mean())
                    saved = report['seeds'][str(seed)]['metrics'][name][group]
                    for key, value in m.items(): assert abs(value - saved[key]) < 1e-14
                    metric_count += 8; mets[name][group] = m
            for control in ('next_only', 'global_local'):
                comparisons[control] = {}
                for fusion in ('raw', 'current', 'mean'):
                    comparisons[control][fusion] = {}
                    for group, mask in masks.items():
                        a = rr['soft_future.' + fusion][mask]; b = rr[control + '.' + fusion][mask]
                        na = ((a <= 10) / np.log2(a + 1)).mean(); nb = ((b <= 10) / np.log2(b + 1)).mean()
                        c = dict(delta_N10=float(na - nb), relative_N10=float(na / nb - 1),
                            delta_H10=float((a <= 10).mean() - (b <= 10).mean()),
                            hit_net=int((a <= 10).sum() - (b <= 10).sum()), hit_gained=int(((a <= 10) & (b > 10)).sum()),
                            hit_lost=int(((a > 10) & (b <= 10)).sum()))
                        saved = report['seeds'][str(seed)]['comparisons'][control][fusion][group]
                        for key, value in c.items(): assert abs(value - saved[key]) < 1e-14
                        comparison_count += len(c); comparisons[control][fusion][group] = c
            all_ranks[seed] = rr; all_metrics[seed] = mets; all_comparisons[seed] = comparisons

        def check_stats(saved, values):
            nonlocal aggregate_count
            np.testing.assert_array_equal(saved['values'], values)
            assert abs(saved['mean'] - np.mean(values)) < 1e-14
            assert abs(saved['sd'] - np.std(values, ddof=1)) < 1e-14
            aggregate_count += 1

        for name, groups in report['aggregate_metrics'].items():
            for group, keys in groups.items():
                for key, saved in keys.items(): check_stats(saved, [all_metrics[s][name][group][key] for s in SEEDS])
        for control, fusions in report['aggregate_comparisons'].items():
            for fusion, groups in fusions.items():
                for group, keys in groups.items():
                    for key, saved in keys.items(): check_stats(saved, [all_comparisons[s][control][fusion][group][key] for s in SEEDS])
        for control, groups in report['fusion_interactions'].items():
            for group, saved in groups.items(): check_stats(saved, [all_comparisons[s][control]['mean'][group]['delta_N10'] - all_comparisons[s][control]['current'][group]['delta_N10'] for s in SEEDS])
        gates[domain] = {}
        for fusion in ('current', 'mean'):
            positive = all(all_comparisons[s][c][fusion][g]['relative_N10'] > 0 for s in SEEDS for c in ('next_only', 'global_local') for g in ('all', 'nonfit'))
            basic = bool(positive and np.mean([all_comparisons[s]['next_only'][fusion]['all']['relative_N10'] for s in SEEDS]) >= .005 and
                sum(all_comparisons[s]['next_only'][fusion]['all']['hit_net'] for s in SEEDS) >= 0)
            residual = bool(basic and np.mean([all_comparisons[s]['global_local'][fusion]['all']['relative_N10'] for s in SEEDS]) >= .005 and
                sum(all_comparisons[s]['global_local'][fusion]['all']['hit_net'] for s in SEEDS) >= 0)
            gates[domain][fusion] = dict(basic_stability=basic, residual_margin=residual, all_seed_directions_positive=positive)
        assert gates[domain] == report['gates']
        domains[domain] = dict(validation_users=len(v['users']), train_prefixes=len(ix), train_users=len(np.unique(stats['users'])), initial_losses=initial_losses)
    decisions = {f: {g: all(d[f][g] for d in gates.values()) for g in ('basic_stability', 'residual_margin')} for f in ('current', 'mean')}
    assert decisions == summary['decisions']; assert summary['training_steps'] == 360
    for name in ('summary.json', 'protocol.json'):
        p = OUT / name; hashes[str(p.relative_to(ROOT))] = sha(p)
    verification = dict(state='VERIFIED', input_hashes=len(summary['input_sha256']), new_checkpoint_count=len(errors),
        prediction_arrays_replayed=scores_checked, metrics_checked=metric_count, comparison_values_checked=comparison_count,
        aggregate_distributions_checked=aggregate_count, checkpoint_max_errors=errors, domains=domains, decisions=decisions,
        output_sha256=hashes, source_sha256=sha(Path(__file__)), replay_source_sha256=sha(ROOT / 'experiment/phase20/verify_future_label_components.py'), test_read=False)
    (OUT / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    print(json.dumps({k: v for k, v in verification.items() if k != 'output_sha256'}, indent=2))


if __name__ == '__main__': main()
