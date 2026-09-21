"""Independent checkpoint, training-objective and full rank/metric replay for Talos adapter."""
import json
from pathlib import Path

import numpy as np
import torch

from experiment.phase20.verify_future_label_components import replay
from experiment.phase20.verify_future_factor_screen import sha, z

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/phase20/talos_candidate_screen_20260917'
PREV = ROOT / 'artifacts/phase20/future_fusion_alignment_20260917'
ARMS = ('softmax', 'negative_softmax', 'talos')
REFS = ('normalized.next_only', 'fusion_mean.next_only', 'legacy.soft_future')
FUSIONS = ('raw', 'current', 'seed2024', 'seed2025', 'mean')
SEEDS = (20260917, 20260918, 20260919)


def objective(score, pos, arm):
    beta = np.sort(score, axis=1)[:, 40]
    transformed = -np.logaddexp(0, beta[:, None] - score) if arm == 'talos' else score.copy()
    positive = transformed[np.arange(len(pos)), pos].copy()
    if arm != 'softmax': transformed[np.arange(len(pos)), pos] = -np.inf
    maximum = transformed.max(1)
    loss = maximum + np.log(np.exp(transformed - maximum[:, None]).sum(1)) - positive
    return loss, [float(beta.min()), float(beta.max())]


def rank_by_count(score, pos):
    chosen = np.maximum(pos, 0)
    gold = score[np.arange(len(pos)), chosen, None]
    return np.where(pos >= 0, 1 + (score > gold).sum(1) + ((score == gold) & (np.arange(50)[None] < chosen[:, None])).sum(1), 51)


def check_gate(c, specific=False):
    all_ = c['all']
    ndcg = np.mean(all_['relative_N10']['values']) >= .005 and sum(all_['hit_net']['values']) >= 0
    hit = np.mean(all_['delta_H10']['values']) >= .001 and np.mean(all_['delta_N10']['values']) >= 0
    protection = sum(c['tail']['hit_net']['values']) >= 0 and np.mean(c['nonfit']['delta_N10']['values']) >= 0
    if specific:
        return dict(gain=bool(ndcg or hit), protection=bool(protection), passed=bool((ndcg or hit) and protection))
    protection = protection and np.mean(c['tail']['delta_N10']['values']) >= 0
    directions = np.count_nonzero(np.asarray(all_['delta_N10']['values']) >= 0) >= 2
    return dict(ndcg_route=bool(ndcg), hit_route=bool(hit), protection=bool(protection), directions=bool(directions),
                passed=bool((ndcg or hit) and protection and directions))


def main():
    torch.set_num_threads(4)
    summary = json.loads((OUT / 'summary.json').read_text()); assert summary['state'] == 'COMPLETED'
    for path, h in summary['input_sha256'].items(): assert sha(ROOT / path) == h, path
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert protocol['temperature'] == 1 and protocol['top_k'] == 10 and protocol['loss_divisor'] == 50
    assert protocol['steps_per_arm'] == 30 and protocol['new_training_steps'] == 540
    initial_map = json.loads((ROOT / 'artifacts/phase20/future_label_multiseed_20260917/protocol.json').read_text())['initialization_sha256']
    registered = json.loads((ROOT / 'artifacts/phase20/strong_baseline_20260917/baseline.json').read_text())
    assert registered['canonical_arm'] == 'normalized.next_only.mean'
    metric_count = comparison_count = aggregate_count = 0
    checkpoint_errors = {}; train_errors = {}; output_hashes = {}; gates = {}

    def read(folder, filename):
        path = folder / filename
        if folder == OUT: output_hashes[str(path.relative_to(ROOT))] = sha(path)
        with np.load(path, allow_pickle=False) as data: return {k: data[k] for k in data.files}

    def distribution(saved, values):
        nonlocal aggregate_count
        np.testing.assert_allclose(saved['values'], values, atol=1e-14, rtol=0)
        assert abs(saved['mean'] - np.mean(values)) < 1e-14
        assert abs(saved['sd'] - np.std(values, ddof=1)) < 1e-14
        aggregate_count += 1

    for domain, report in summary['domains'].items():
        ctx = read(PREV, f'{domain}.training_context.npz')
        a = read(ROOT / f'artifacts/phase20/gacr_fusion_train_v1/{domain}', 'training_cache.npz')
        future = read(ROOT / 'artifacts/phase20/future_horizon_audit_20260917', f'{domain}.labels.npz')
        ix = ctx['fit_indices']; pos = a['positions'][ix]
        np.testing.assert_array_equal(ix, np.flatnonzero(a['fit'] & (a['positions'] >= 0) & (future['remaining'] >= 3)))
        np.testing.assert_array_equal(ctx['candidate_ids'], a['candidate_ids'][ix])
        np.testing.assert_array_equal(ctx['targets'], ctx['candidate_ids'][np.arange(len(ix)), pos])
        head = a['head'][ix]; weights = np.where(head, .5 / head.sum(), .5 / (~head).sum())
        initial_scores = z(ctx['prior']) + ctx['cf_mean']
        initial_losses = {arm: float(weights @ objective(initial_scores, pos, arm)[0]) for arm in ARMS}
        v = read(ROOT / 'artifacts/phase20/future_factor_screen_20260917', f'{domain}.audit_validation_cache.npz')
        pair = read(ROOT / 'artifacts/phase20/pair_uncertainty_screen_20260917', f'{domain}.predictions.npz')
        for key in ('users', 'candidate_ids', 'positions', 'tail_mask'): np.testing.assert_array_equal(v[key], pair[key])
        parts = {f: pair['scores'][list(pair['arm_names']).index(f)] - v['prior'] - np.log(49) for f in FUSIONS[1:]}
        all_metrics = {}; all_comps = {}
        for seed in SEEDS:
            p = read(OUT, f'{domain}.{seed}.predictions.npz'); old = read(PREV, f'{domain}.{seed}.predictions.npz')
            for key in ('users', 'candidate_ids', 'positions', 'tail_mask'): np.testing.assert_array_equal(p[key], v[key])
            np.testing.assert_array_equal(p['in_fit'], np.isin(v['users'], np.unique(ctx['users'])))
            raw_names = list(p['raw_names']); rank_names = list(p['rank_names'])
            assert set(raw_names) == set(REFS + ARMS)
            assert set(rank_names) == {name + '.' + f for name in raw_names for f in FUSIONS}
            training_path = OUT / f'{domain}.{seed}.training.json'
            output_hashes[str(training_path.relative_to(ROOT))] = sha(training_path)
            training = json.loads(training_path.read_text()); assert set(training) == set(ARMS)
            for arm in ARMS:
                trace = training[arm]
                assert trace['initial_sha256'] == initial_map[str(seed)]
                assert [c['step'] for c in trace['curve']] == list(range(1, 31))
                assert abs(trace['curve'][0]['loss'] - initial_losses[arm]) < 1e-11
                for point in trace['curve']:
                    assert np.isfinite([point['loss'], point['scaled_loss'], point['grad_norm']]).all()
                    assert abs(point['scaled_loss'] * 50 - point['loss']) < 1e-13
                assert trace['parameter_delta_l2'] > 0 and np.isfinite(trace['parameter_delta_l2'])
                ckpt = OUT / f'{domain}.{seed}.{arm}.pt'
                output_hashes[str(ckpt.relative_to(ROOT))] = sha(ckpt)
                state = torch.load(ckpt, map_location='cpu'); assert sum(v.numel() for v in state.values()) == 258
                prediction = replay(state, v['features'], v['prior'])
                err = float(np.abs(prediction - p['raw_scores'][raw_names.index(arm)]).max()); assert err < 1e-11
                checkpoint_errors[f'{domain}.{seed}.{arm}'] = err
                train_score = replay(state, a['features'][ix], ctx['prior']) + ctx['cf_mean']
                final, beta = objective(train_score, pos, arm)
                error = abs(float(weights @ final) - trace['final_loss']); assert error < 1e-11
                np.testing.assert_allclose(beta, trace['final_threshold_range'], atol=1e-11, rtol=0)
                train_errors[f'{domain}.{seed}.{arm}'] = error
            for reference in REFS:
                np.testing.assert_array_equal(p['raw_scores'][raw_names.index(reference)], old['raw_scores'][list(old['raw_names']).index(reference)])
            masks = dict(all=np.ones(len(v['users']), bool), tail=p['tail_mask'], non_tail=~p['tail_mask'], fit=p['in_fit'], nonfit=~p['in_fit'])
            rr = {}; met = {}
            for j, name in enumerate(rank_names):
                arm, fusion = name.rsplit('.', 1)
                raw = p['raw_scores'][raw_names.index(arm)]
                rank = rank_by_count(raw if fusion == 'raw' else raw + parts[fusion], v['positions'])
                np.testing.assert_array_equal(rank, p['ranks'][j]); rr[name] = rank; met[name] = {}
                if arm in REFS: np.testing.assert_array_equal(rank, old['ranks'][list(old['rank_names']).index(name)])
                for group, mask in masks.items():
                    saved = report['seeds'][str(seed)]['metrics'][name][group]; measured = dict(users=int(mask.sum()))
                    for k in (5, 10, 20, 50):
                        measured[f'H{k}'] = float(np.mean(rank[mask] <= k))
                        measured[f'N{k}'] = float(np.mean(np.where(rank[mask] <= k, 1 / np.log2(rank[mask] + 1), 0)))
                    for key, value in measured.items(): assert abs(value - saved[key]) < 1e-14
                    met[name][group] = measured; metric_count += 8
            # Explicitly link the registered strong baseline to this run's compared ranks.
            assert met['normalized.next_only.mean'] == registered['domains'][domain]['per_seed_metrics'][str(seed)]
            comparisons = {}
            expected_contrasts = {left + '__vs__' + right for left, right in protocol['contrasts']}
            assert set(report['seeds'][str(seed)]['comparisons']) == expected_contrasts
            for contrast, by_fusion in report['seeds'][str(seed)]['comparisons'].items():
                left, right = contrast.split('__vs__'); comparisons[contrast] = {}
                for fusion, by_group in by_fusion.items():
                    comparisons[contrast][fusion] = {}
                    for group, saved in by_group.items():
                        r1, r0 = rr[left + '.' + fusion][masks[group]], rr[right + '.' + fusion][masks[group]]
                        n1 = np.where(r1 <= 10, 1 / np.log2(r1 + 1), 0).mean()
                        n0 = np.where(r0 <= 10, 1 / np.log2(r0 + 1), 0).mean()
                        measured = dict(delta_N10=float(n1 - n0), relative_N10=float(n1 / n0 - 1),
                            delta_H10=float(np.mean(r1 <= 10) - np.mean(r0 <= 10)), hit_net=int((r1 <= 10).sum() - (r0 <= 10).sum()),
                            hit_gained=int(((r1 <= 10) & (r0 > 10)).sum()), hit_lost=int(((r1 > 10) & (r0 <= 10)).sum()))
                        for key, value in measured.items(): assert abs(value - saved[key]) < 1e-14
                        comparisons[contrast][fusion][group] = measured; comparison_count += 6
            all_metrics[seed] = met; all_comps[seed] = comparisons
        for name, by_group in report['aggregate_metrics'].items():
            for group, by_metric in by_group.items():
                for key, saved in by_metric.items(): distribution(saved, [all_metrics[seed][name][group][key] for seed in SEEDS])
        for name, by_fusion in report['aggregate_comparisons'].items():
            for fusion, by_group in by_fusion.items():
                for group, by_metric in by_group.items():
                    for key, saved in by_metric.items(): distribution(saved, [all_comps[seed][name][fusion][group][key] for seed in SEEDS])
        agg = report['aggregate_comparisons']
        utility = {arm: check_gate(agg[arm + '__vs__normalized.next_only']['mean']) for arm in ARMS}
        specificity = {arm: check_gate(agg['talos__vs__' + arm]['mean'], specific=True) for arm in ARMS[:2]}
        assert utility == report['utility_gates'] and specificity == report['specificity_gates']
        topk = bool(utility['talos']['passed'] and all(v['passed'] for v in specificity.values()))
        assert topk == report['topk_gate']
        gates[domain] = dict(utility=utility, specificity=specificity, topk=topk)
    retained = [a for a in ARMS if all(g['utility'][a]['passed'] for g in gates.values())]
    decision = 'TOPK_SPECIFIC_INCREMENT_SUPPORTED' if all(g['topk'] for g in gates.values()) else ('PRACTICAL_BASELINE_ONLY' if retained else 'NO_ADVANCEMENT_GATE_PASSED')
    assert summary['decision'] == decision and summary['retained_arms'] == retained and summary['training_steps'] == 540
    for filename in ('summary.json', 'protocol.json'):
        path = OUT / filename; output_hashes[str(path.relative_to(ROOT))] = sha(path)
    result = dict(state='VERIFIED', input_hashes=len(summary['input_sha256']), checkpoints_replayed=len(checkpoint_errors),
        checkpoint_max_errors=checkpoint_errors, final_training_loss_errors=train_errors, metrics_checked=metric_count,
        comparison_values_checked=comparison_count, aggregate_distributions_checked=aggregate_count,
        registered_strong_baseline_exact_parity=True, gates=gates, decision=decision, retained_arms=retained,
        output_sha256=output_hashes, source_sha256=sha(Path(__file__)), test_read=False)
    (OUT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('checkpoint_max_errors', 'final_training_loss_errors', 'output_sha256')}, indent=2))


if __name__ == '__main__': main()
