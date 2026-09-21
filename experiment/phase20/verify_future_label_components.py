"""Independent label reconstruction, score replay and metric checks for component ablation."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from experiment.phase20.verify_future_factor_screen import sha, z

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/phase20/future_label_components_20260917'
PARENT = ROOT / 'artifacts/phase20/future_factor_screen_20260917'
SPECIFICITY = ROOT / 'artifacts/phase20/future_label_specificity_20260917'
ARMS = ('global_uniform', 'global_local', 'rowmass_globalshape', 'globalmass_rowshape', 'rowmass_rowshape')


def replay(state, features, prior):
    values = []
    with torch.no_grad():
        for offset in range(0, len(prior), 256):
            sl = slice(offset, offset + 256); x = torch.as_tensor(features[sl]); delta = []
            for j in range(2):
                hidden = torch.tanh(F.linear(x, state[f'heads.{j}.0.weight'], state[f'heads.{j}.0.bias']))
                delta.append((2 * torch.tanh(F.linear(hidden, state[f'heads.{j}.2.weight'], state[f'heads.{j}.2.bias'])).squeeze(-1)).numpy().astype(np.float64))
            root = prior[sl] + np.log1p(np.sqrt(1 + np.exp(-prior[sl])))
            probability = 1 / (1 + np.exp(-root - delta[0])) / (1 + np.exp(-root - delta[1]))
            values.append(np.log(probability) - np.log1p(-probability))
    return z(np.concatenate(values))


def main():
    torch.set_num_threads(4)
    summary = json.loads((OUT / 'summary.json').read_text())
    for p, h in summary['input_sha256'].items(): assert sha(ROOT / p) == h, p
    domains = {}; metric_count = 0; user_count = 0; output_hashes = {}
    for domain in ('Toys', 'Beauty'):
        def read_npz(folder, name):
            p = folder / f'{domain}.{name}.npz'
            if folder == OUT: output_hashes[str(p.relative_to(ROOT))] = sha(p)
            with np.load(p) as f: return {k: f[k] for k in f.files}
        a = read_npz(OUT, 'predictions'); stats = read_npz(OUT, 'label_statistics'); ys = read_npz(OUT, 'training_labels')
        v = read_npz(PARENT, 'audit_validation_cache'); old = read_npz(SPECIFICITY, 'predictions')
        labels = read_npz(ROOT / 'artifacts/phase20/future_horizon_audit_20260917', 'labels')
        with np.load(ROOT / f'artifacts/phase20/gacr_fusion_train_v1/{domain}/training_cache.npz') as f:
            cache = {k: f[k] for k in f.files}
        valid = labels['fit'] & (labels['positions'] >= 0) & (labels['remaining'] >= 3)
        np.testing.assert_array_equal(stats['fit_indices'], np.flatnonzero(valid))
        np.testing.assert_array_equal(stats['users'], labels['users'][valid])
        np.testing.assert_array_equal(stats['head'], cache['head'][valid])
        pos = labels['positions'][valid]; dist = labels['distances'][valid]; n = len(pos)
        immediate = np.arange(50)[None] == pos[:, None]
        true = ((dist > 0) & (dist <= 3) & ~immediate) / (np.maximum(dist, 0) + 1)
        np.testing.assert_array_equal(stats['actual'], true + immediate)
        # Verify grouping from frozen B1 weights without the experiment's helper.
        base_state = torch.load(ROOT / f'artifacts/phase20/gacr_fusion_train_v1/{domain}/B1.pt', map_location='cpu')
        with torch.no_grad():
            x = torch.as_tensor(cache['features'][valid])
            h = torch.tanh(F.linear(x, base_state['network.0.weight'], base_state['network.0.bias']))
            g = (torch.as_tensor(cache['base'][valid]) + .2 * torch.tanh(F.linear(h, base_state['network.2.weight'], base_state['network.2.bias'])).squeeze(-1)).numpy()
        masked = g.copy(); masked[np.arange(n), pos] = -np.inf
        order = np.argsort(-masked, axis=1, kind='stable')[:, :49]
        np.testing.assert_array_equal(order, stats['order'])
        bins = np.take_along_axis(true, order, 1).reshape(n, 7, 7).sum(2)
        np.testing.assert_array_equal(bins, stats['bin_mass'])
        # Deliberately recompute each user's prior from ONLY other users.
        prior = np.empty_like(bins)
        for u in np.unique(stats['users']):
            same = stats['users'] == u
            mu = (bins[(~same) & stats['head']].mean(0) + bins[(~same) & (~stats['head'])].mean(0)) / 2
            prior[same] = mu; user_count += 1
        np.testing.assert_allclose(prior, stats['prior_bins'], atol=1e-12, rtol=0)
        grow = prior.sum(1); mrow = bins.sum(1)
        np.testing.assert_allclose(stats['row_mass'], mrow, atol=1e-12, rtol=0)
        np.testing.assert_allclose(stats['global_mass'], grow, atol=1e-12, rtol=0)
        pglobal = prior / grow[:, None]; plocal = pglobal.copy()
        nz = mrow > 0; plocal[nz] = bins[nz] / mrow[nz, None]
        expected_bins = dict(global_uniform=np.repeat(grow[:, None] / 7, 7, axis=1), global_local=prior,
            rowmass_globalshape=mrow[:, None] * pglobal, globalmass_rowshape=grow[:, None] * plocal,
            rowmass_rowshape=bins)
        for arm in ARMS:
            expected = immediate.astype(float)
            expected[np.arange(n)[:, None], order] = np.repeat(expected_bins[arm] / 7, 7, axis=1)
            np.testing.assert_allclose(expected, ys[arm], atol=1e-12, rtol=0)
        np.testing.assert_array_equal(a['users'], v['users']); np.testing.assert_array_equal(a['candidate_ids'], v['candidate_ids'])
        fit_users = set(stats['users']); np.testing.assert_array_equal(a['in_fit'], [u in fit_users for u in a['users']])
        names = list(a['arm_names']); errors = {}
        for arm in ARMS:
            ckpt = OUT / f'{domain}.{arm}.pt'; output_hashes[str(ckpt.relative_to(ROOT))] = sha(ckpt)
            raw = replay(torch.load(ckpt, map_location='cpu'), v['features'], v['prior']); errors[arm] = 0.
            for fusion in ('raw', 'current', 'mean'):
                score = raw if fusion == 'raw' else raw + v['cf_' + fusion]
                saved = a['scores'][names.index(arm + '.' + fusion)]
                err = float(np.max(np.abs(score - saved))); errors[arm] = max(errors[arm], err); assert err < 1e-11
        for name in names:
            if name.split('.')[0] in ARMS: continue
            j = list(old['arm_names']).index(name)
            np.testing.assert_array_equal(a['scores'][names.index(name)], old['scores'][j])
        masks = dict(all=np.ones(len(a['users']), bool), tail=a['tail_mask'], non_tail=~a['tail_mask'], fit=a['in_fit'], nonfit=~a['in_fit'])
        for j, name in enumerate(names):
            score = a['scores'][j]; positions = a['positions']; ix = np.maximum(positions, 0)
            gold = score[np.arange(len(ix)), ix, None]
            rr = np.where(positions >= 0, 1 + (score > gold).sum(1) + ((score == gold) & (np.arange(50)[None] < ix[:, None])).sum(1), 51)
            np.testing.assert_array_equal(rr, a['ranks'][j])
            for group, mask in masks.items():
                m = summary['domains'][domain]['metrics'][name][group]
                assert m['users'] == int(mask.sum())
                for k in (5, 10, 20, 50):
                    for metric, value in [(f'H{k}', (rr[mask] <= k).mean()), (f'N{k}', ((rr[mask] <= k) / np.log2(rr[mask] + 1)).mean())]:
                        assert abs(value - m[metric]) < 1e-14; metric_count += 1
        for name, comparisons in summary['domains'][domain]['comparisons'].items():
            first, second = name.split('_vs_'); ra = a['ranks'][names.index(first + '.mean')]; rb = a['ranks'][names.index(second + '.mean')]
            for group, mask in masks.items():
                aa = ra[mask]; bb = rb[mask]
                na = ((aa <= 10) / np.log2(aa + 1)).mean(); nb = ((bb <= 10) / np.log2(bb + 1)).mean()
                c = comparisons[group]
                assert abs(c['relative_N10'] - (na / nb - 1)) < 1e-14
                assert abs(c['delta_N10'] - (na - nb)) < 1e-14
                assert c['hit_gained'] == int(((aa <= 10) & (bb > 10)).sum())
                assert c['hit_lost'] == int(((aa > 10) & (bb <= 10)).sum())
        m = summary['domains'][domain]['metrics']; p = summary['domains'][domain]['comparisons']['soft_future_vs_global_local']
        assert summary['domains'][domain]['future_passed'] == (p['all']['relative_N10'] >= .005 and p['nonfit']['relative_N10'] > 0)
        pooled = m['rowmass_rowshape.mean']['all']; controls = [x for x in ARMS if x != 'rowmass_rowshape'] + ['soft_future', 'next_only']
        group_passed = (pooled['N10'] / m['global_local.mean']['all']['N10'] - 1 >= .005 and
            all(pooled['N10'] > m[x + '.mean']['all']['N10'] for x in controls) and
            all(pooled['H10'] >= m[x + '.mean']['all']['H10'] for x in ('global_local', 'next_only')))
        assert summary['domains'][domain]['group_passed'] == group_passed
        domains[domain] = dict(checkpoint_max_errors=errors, users=len(a['users']),
            train_prefixes=n, zero_mass_prefixes=int((mrow == 0).sum()),
            mean_other_user_mass=float(grow.mean()), mean_other_user_bin_profile=prior.mean(0).tolist())
    actual_decisions = dict(per_prefix_future_supported=all(v['future_passed'] for v in summary['domains'].values()),
        group_supervision_followup_supported=all(v['group_passed'] for v in summary['domains'].values()))
    assert actual_decisions == summary['decisions']
    result = dict(state='VERIFIED', input_hashes=len(summary['input_sha256']), metrics_checked=metric_count,
        users_with_explicit_exclusion_checked=user_count, checkpoint_count=10, domains=domains,
        output_sha256=output_hashes, source_sha256=sha(Path(__file__)), test_read=False)
    (OUT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'output_sha256'}, indent=2))


if __name__ == '__main__': main()
