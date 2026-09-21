"""Independent numerical replay, frozen CF scoring and complete metric verification."""
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

from experiment.phase20.verify_future_label_components import replay
from experiment.phase20.verify_future_factor_screen import sha, z

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiment/phase9'))
from train_cf0_b2_item_head import CF0B2ItemHead

OUT = ROOT / 'artifacts/phase20/future_fusion_alignment_20260917'
PREVIOUS = ROOT / 'artifacts/phase20/future_label_multiseed_20260917'
COMP = ROOT / 'artifacts/phase20/future_label_components_20260917'
FACTOR = ROOT / 'artifacts/phase20/future_factor_screen_20260917'
PAIR = ROOT / 'artifacts/phase20/pair_uncertainty_screen_20260917'
SEEDS = (20260917, 20260918, 20260919)
LABELS = ('next_only', 'global_local', 'soft_future')
MODES = ('legacy', 'normalized', 'fusion_mean')
FUSIONS = ('raw', 'current', 'seed2024', 'seed2025', 'mean')


def direct_ranks(s, pos):
    chosen = np.maximum(pos, 0)
    gold = s[np.arange(len(pos)), chosen, None]
    return np.where(pos >= 0, 1 + (s > gold).sum(1) + ((s == gold) & (np.arange(50)[None] < chosen[:, None])).sum(1), 51)


def main():
    torch.set_num_threads(4)
    summary = json.loads((OUT / 'summary.json').read_text()); assert summary['state'] == 'COMPLETED'
    for p, h in summary['input_sha256'].items(): assert sha(ROOT / p) == h, p
    protocol = json.loads((OUT / 'protocol.json').read_text())
    init = json.loads((PREVIOUS / 'protocol.json').read_text())['initialization_sha256']
    config = json.loads((ROOT / 'experiment/phase20/gacr_fusion_train_v1.json').read_text())
    metric_count = 0; comparison_count = 0; aggregate_count = 0; checkpoint_errors = {}; cf_errors = {}; output_hashes = {}; gates = {}; domains = {}

    def read(folder, name):
        p = folder / name
        if folder == OUT: output_hashes[str(p.relative_to(ROOT))] = sha(p)
        with np.load(p) as f: return {k: f[k] for k in f.files}

    def check_distribution(saved, values):
        nonlocal aggregate_count
        np.testing.assert_allclose(saved['values'], values, atol=1e-14, rtol=0)
        assert abs(saved['mean'] - np.mean(values)) < 1e-14
        assert abs(saved['sd'] - np.std(values, ddof=1)) < 1e-14
        aggregate_count += 1

    for domain in ('Toys', 'Beauty'):
        cfg = config['domains'][domain]; report = summary['domains'][domain]
        a = read(ROOT / f'artifacts/phase20/gacr_fusion_train_v1/{domain}', 'training_cache.npz')
        ctx = read(OUT, f'{domain}.training_context.npz'); stats = read(COMP, f'{domain}.label_statistics.npz')
        future = read(ROOT / 'artifacts/phase20/future_horizon_audit_20260917', f'{domain}.labels.npz')
        labels = read(COMP, f'{domain}.training_labels.npz')
        ix = stats['fit_indices']; n = len(ix)
        np.testing.assert_array_equal(ix, ctx['fit_indices']); np.testing.assert_array_equal(ix, np.flatnonzero(a['fit'] & (a['positions'] >= 0) & (future['remaining'] >= 3)))
        np.testing.assert_array_equal(ctx['users'], stats['users']); np.testing.assert_array_equal(ctx['candidate_ids'], a['candidate_ids'][ix])
        np.testing.assert_array_equal(ctx['frequencies'], a['frequencies'][ix]); np.testing.assert_array_equal(ctx['cf_scores'][0], a['cf'][ix])
        np.testing.assert_array_equal(ctx['targets'], ctx['candidate_ids'][np.arange(n), a['positions'][ix]])
        # Reconstruct histories directly from prepared train rows, independently of candidate JSONL.
        expected_by_index = {int(t): j for j, t in enumerate(ctx['train_indices'])}; assert len(expected_by_index) == n
        found = set()
        with (ROOT / cfg['validation']).with_name('train.jsonl').open() as f:
            for i, line in enumerate(f):
                if i not in expected_by_index: continue
                j = expected_by_index[i]; r = json.loads(line); found.add(i)
                assert r['user_id'] == ctx['users'][j] and r['target'] == ctx['targets'][j]
                h = np.zeros(20, np.int64); h[:len(r['history'])] = r['history']
                np.testing.assert_array_equal(h, ctx['history'][j])
        assert found == set(expected_by_index)
        # Re-score disjoint positions across the training cache from every existing CF checkpoint.
        sampled = np.linspace(0, n - 1, 96, dtype=int)
        for cf_index, path in enumerate([ROOT / cfg['item_head']] + [ROOT / f'artifacts/phase9/p9s_multiseed/{domain}/seed{s}/item_head/best_item_head.pt' for s in (2024, 2025)]):
            checkpoint = torch.load(path, map_location='cpu'); c = checkpoint['model_config']
            model = CF0B2ItemHead(c['num_items'], c['max_history'], c['d_model'], c['num_layers'], c['num_heads'], c['dropout'], c['temperature_initial'])
            model.load_state_dict(checkpoint['model_state_dict'], strict=True); model.eval()
            with torch.no_grad():
                h = torch.as_tensor(ctx['history'][sampled]); u = model.encode(h, h.ne(0))
                item = F.normalize(model.item_embedding.weight, dim=-1)[torch.as_tensor(ctx['candidate_ids'][sampled])]
                predicted = (model.logit_scale.exp().clamp(max=100) * torch.bmm(item, u[:, :, None]).squeeze(-1)).numpy()
            err = float(np.abs(predicted - ctx['cf_scores'][cf_index, sampled]).max())
            assert err <= 1e-4; cf_errors[f'{domain}.{cf_index}'] = err
            del model
        b1 = torch.load(ROOT / f'artifacts/phase20/gacr_fusion_train_v1/{domain}/B1.pt', map_location='cpu')
        with torch.no_grad():
            hidden = torch.tanh(F.linear(torch.as_tensor(a['features'][ix]), b1['network.0.weight'], b1['network.0.bias']))
            g = (torch.as_tensor(a['base'][ix]) + .2 * torch.tanh(F.linear(hidden, b1['network.2.weight'], b1['network.2.bias'])).squeeze(-1)).numpy().astype(float)
        prior = z(g) - np.log(49); np.testing.assert_array_equal(prior, ctx['prior'])
        order = np.argsort(-g, axis=1, kind='stable')
        r = 1 - (np.take_along_axis(ctx['frequencies'], order[:, :10], 1) <= cfg['q1']).mean(1)
        np.testing.assert_array_equal(r, ctx['reliability'])
        adjusted = []
        for single in ctx['cf_scores']: adjusted.append(z(z(single) - .5 * z(np.log1p(ctx['frequencies']))))
        component = r[:, None] * np.mean(adjusted, axis=0)
        np.testing.assert_allclose(component, ctx['cf_mean'], atol=1e-12, rtol=0)
        immediate = (np.arange(50)[None] == a['positions'][ix, None]).astype(float)
        dist = future['distances'][ix]
        actual = immediate + ((dist > 0) & (dist <= 3) & (immediate == 0)) / (np.maximum(dist, 0) + 1)
        np.testing.assert_array_equal(actual, stats['actual'])
        local = immediate.copy()
        local[np.arange(n)[:, None], stats['order']] = np.repeat(stats['prior_bins'] / 7, 7, axis=1)
        np.testing.assert_array_equal(local, labels['global_local'])
        targets = dict(next_only=immediate, global_local=local, soft_future=actual)
        head = a['head'][ix]; weights = np.where(head, .5 / head.sum(), .5 / (~head).sum())
        first_losses = {}
        for mode in MODES[1:]:
            initial_logit = z(prior) - np.log(49) + (component if mode == 'fusion_mean' else 0)
            for label, y in targets.items(): first_losses[mode + '.' + label] = float(((np.logaddexp(0, initial_logit) - y * initial_logit).mean(1) * weights).sum())
        v = read(FACTOR, f'{domain}.audit_validation_cache.npz'); pair = read(PAIR, f'{domain}.predictions.npz')
        for key in ('users', 'candidate_ids', 'positions', 'tail_mask'): np.testing.assert_array_equal(v[key], pair[key])
        parts = {f: pair['scores'][list(pair['arm_names']).index(f)] - v['prior'] - np.log(49) for f in FUSIONS[1:]}
        all_metrics = {}; all_comparisons = {}
        for seed in SEEDS:
            data = read(OUT, f'{domain}.{seed}.predictions.npz'); previous = read(PREVIOUS, f'{domain}.{seed}.predictions.npz')
            for key in ('users', 'candidate_ids', 'positions', 'tail_mask'): np.testing.assert_array_equal(data[key], v[key])
            np.testing.assert_array_equal(data['in_fit'], np.isin(v['users'], np.unique(ctx['users'])))
            raw_names = list(data['raw_names']); rank_names = list(data['rank_names'])
            assert set(raw_names) == {m + '.' + y for m in MODES for y in LABELS}
            assert set(rank_names) == {a + '.' + f for a in raw_names for f in FUSIONS}
            train_path = OUT / f'{domain}.{seed}.training.json'; train = json.loads(train_path.read_text()); output_hashes[str(train_path.relative_to(ROOT))] = sha(train_path)
            for mode in MODES:
                for label in LABELS:
                    name = mode + '.' + label; raw = data['raw_scores'][raw_names.index(name)]
                    if mode == 'legacy':
                        np.testing.assert_array_equal(raw, previous['scores'][list(previous['arm_names']).index(label + '.raw')]); continue
                    assert train['initial_sha256'][name] == init[str(seed)]
                    curve = train['curves'][name]; assert [c['step'] for c in curve] == list(range(1, 31))
                    assert all(np.isfinite([c['loss'], c['grad_norm']]).all() for c in curve)
                    assert abs(first_losses[name] - curve[0]['loss']) < 1e-12
                    ckpt = OUT / f'{domain}.{seed}.{name}.pt'; output_hashes[str(ckpt.relative_to(ROOT))] = sha(ckpt)
                    prediction = replay(torch.load(ckpt, map_location='cpu'), v['features'], v['prior'])
                    error = float(np.abs(prediction - raw).max()); assert error < 1e-11
                    checkpoint_errors[f'{domain}.{seed}.{name}'] = error
            masks = dict(all=np.ones(len(v['users']), bool), tail=data['tail_mask'], non_tail=~data['tail_mask'], fit=data['in_fit'], nonfit=~data['in_fit'])
            rr = {}; met = {}
            for j, name in enumerate(rank_names):
                mode, label, fusion = name.split('.')
                raw = data['raw_scores'][raw_names.index(mode + '.' + label)]
                rank = direct_ranks(raw if fusion == 'raw' else raw + parts[fusion], v['positions'])
                np.testing.assert_array_equal(rank, data['ranks'][j]); rr[name] = rank; met[name] = {}
                for group, mask in masks.items():
                    saved = report['seeds'][str(seed)]['metrics'][name][group]; m = dict(users=int(mask.sum()))
                    for k in (5, 10, 20, 50):
                        m[f'H{k}'] = float((rank[mask] <= k).mean())
                        m[f'N{k}'] = float(np.where(rank[mask] <= k, 1 / np.log2(rank[mask] + 1), 0).mean())
                    for key, value in m.items(): assert abs(saved[key] - value) < 1e-14
                    metric_count += 8; met[name][group] = m
            comparisons = {}
            for name, by_fusion in report['seeds'][str(seed)]['comparisons'].items():
                left, right = name.split('__vs__'); comparisons[name] = {}
                for fusion, by_group in by_fusion.items():
                    comparisons[name][fusion] = {}
                    for group, saved in by_group.items():
                        aa = rr[left + '.' + fusion][masks[group]]; bb = rr[right + '.' + fusion][masks[group]]
                        na = np.where(aa <= 10, 1 / np.log2(aa + 1), 0).mean(); nb = np.where(bb <= 10, 1 / np.log2(bb + 1), 0).mean()
                        c = dict(delta_N10=float(na - nb), relative_N10=float(na / nb - 1),
                            delta_H10=float((aa <= 10).mean() - (bb <= 10).mean()), hit_net=int((aa <= 10).sum() - (bb <= 10).sum()),
                            hit_gained=int(((aa <= 10) & (bb > 10)).sum()), hit_lost=int(((aa > 10) & (bb <= 10)).sum()))
                        for key, value in c.items(): assert abs(saved[key] - value) < 1e-14
                        comparison_count += len(c); comparisons[name][fusion][group] = c
            all_metrics[seed] = met; all_comparisons[seed] = comparisons
        for name, by_group in report['aggregate_metrics'].items():
            for group, by_key in by_group.items():
                for key, saved in by_key.items(): check_distribution(saved, [all_metrics[s][name][group][key] for s in SEEDS])
        for name, by_fusion in report['aggregate_comparisons'].items():
            for fusion, by_group in by_fusion.items():
                for group, by_key in by_group.items():
                    for key, saved in by_key.items(): check_distribution(saved, [all_comparisons[s][name][fusion][group][key] for s in SEEDS])
        required = ['legacy.soft_future', 'normalized.soft_future', 'fusion_mean.next_only', 'fusion_mean.global_local']
        margin = all(np.mean([all_comparisons[s]['fusion_mean.soft_future__vs__' + b]['mean']['all']['relative_N10'] for s in SEEDS]) >= .005 for b in required)
        directions = all(all_comparisons[s]['fusion_mean.soft_future__vs__legacy.soft_future']['mean'][g]['relative_N10'] > 0 for s in SEEDS for g in ('all', 'nonfit'))
        hits = all(sum(all_comparisons[s]['fusion_mean.soft_future__vs__' + b]['mean'][g]['hit_net'] for s in SEEDS) >= 0 for b in ('legacy.soft_future', 'fusion_mean.next_only') for g in ('all', 'tail'))
        gate = dict(margins=bool(margin), directions=bool(directions), hit_protection=bool(hits), passed=bool(margin and directions and hits))
        assert gate == report['gate']; gates[domain] = gate
        domains[domain] = dict(train_prefixes=n, train_users=len(np.unique(ctx['users'])), validation_users=len(v['users']), initial_losses=first_losses)
    expected = 'SUPPORTED_FOR_FOLLOWUP' if all(g['passed'] for g in gates.values()) else 'FUSION_ALIGNMENT_ADVANTAGE_NOT_ESTABLISHED'
    assert summary['decision'] == expected and summary['training_steps'] == 1080
    for filename in ('summary.json', 'protocol.json'):
        p = OUT / filename; output_hashes[str(p.relative_to(ROOT))] = sha(p)
    result = dict(state='VERIFIED', input_hashes=len(summary['input_sha256']), checkpoints_replayed=len(checkpoint_errors),
        checkpoint_max_errors=checkpoint_errors, cf_checkpoint_sample_errors=cf_errors, sampled_training_CF_scores=96 * 50 * 6,
        metrics_checked=metric_count, comparison_values_checked=comparison_count, aggregate_distributions_checked=aggregate_count,
        domains=domains, gates=gates, decision=expected, output_sha256=output_hashes, source_sha256=sha(Path(__file__)),
        replay_source_sha256=sha(ROOT / 'experiment/phase20/verify_future_label_components.py'), test_read=False)
    (OUT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('output_sha256', 'checkpoint_max_errors')}, indent=2))


if __name__ == '__main__': main()
