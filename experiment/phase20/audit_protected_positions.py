"""Frozen cached mechanism screen with no model fitting or new beam generation."""
import csv
import hashlib
import itertools
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from experiment.phase20.audit_b1_fusion_tradeoff import digest, rank, read, zscore, WEIGHTS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/phase20/protected_position_audit_20260917'
OLD = ROOT / 'artifacts/phase20/gacr_fusion_train_v1'
ARMS = ('current', 'plain', 'pop_constant', 'group_slots', 'protect_tail', 'protect_non_tail')


def project(base_order, proposal_order, item_tail, mode):
    """Inputs are predicted permutations and item frequencies; never targets."""
    result = base_order.copy()
    movable = (False, True) if mode == 'group_slots' else ((False,) if mode == 'protect_tail' else (True,))
    assert mode in ARMS[3:]
    for n, (base, proposal, tail) in enumerate(zip(base_order, proposal_order, item_tail)):
        for group in movable:
            slots = np.flatnonzero(tail[base] == group)
            items = proposal[tail[proposal] == group]
            assert len(slots) == len(items)
            result[n, slots] = items
    return result


def target_ranks(order, positions):
    inv = np.argsort(order, axis=1)
    return np.where(positions >= 0, inv[np.arange(len(order)), np.maximum(positions, 0)] + 1, 51)


def metric(ranks, users=None):
    ranks = np.asarray(ranks, dtype=np.int64)
    result = {'rows': len(ranks)}
    if users is not None:
        unique, inv, count = np.unique(users, return_inverse=True, return_counts=True)
        weights = 1 / (len(unique) * count[inv])
        result['users'] = len(unique)
    else:
        weights = np.full(len(ranks), 1 / len(ranks))
    for k in (5, 10, 20, 50):
        result[f'H{k}'] = float(((ranks <= k) * weights).sum())
        result[f'N{k}'] = float((np.where(ranks <= k, 1 / np.log2(ranks + 1), 0) * weights).sum())
    return result


def evaluate(g, cf, frequency, positions, target_frequency, users, q1):
    order = np.argsort(-g, axis=1, kind='stable')
    positions = np.where(positions >= 0, np.argsort(order, axis=1)[np.arange(len(g)), np.maximum(positions, 0)], -1)
    g, cf, frequency = [np.take_along_axis(x, order, 1) for x in (g, cf, frequency)]
    gz, cz, pz = zscore(g), zscore(cf), zscore(np.log1p(frequency))
    corrected = zscore(cz - .5 * pz)
    reliability = 1 - (frequency[:, :10] <= q1).mean(1)
    scores = {'current': gz + reliability[:, None] * corrected,
              'plain': gz + zscore(cz), 'pop_constant': gz + corrected}
    orders = {k: np.argsort(-v, axis=1, kind='stable') for k, v in scores.items()}
    low = frequency <= q1
    for arm in ARMS[3:]:
        orders[arm] = project(orders['current'], orders['plain'], low, arm)
    parent_inverse = np.argsort(orders['current'], axis=1)
    for arm, output in orders.items():
        np.testing.assert_array_equal(np.sort(output, axis=1), np.broadcast_to(np.arange(50), output.shape))
        inverse = np.argsort(output, axis=1)
        if arm in ('protect_tail', 'protect_non_tail'):
            mask = low if arm == 'protect_tail' else ~low
            np.testing.assert_array_equal(inverse[mask], parent_inverse[mask])
        if arm in ARMS[3:]:
            np.testing.assert_array_equal(np.take_along_axis(low, output, 1), np.take_along_axis(low, orders['current'], 1))
    ranks = {k: target_ranks(v, positions) for k, v in orders.items()}
    ranks['raw'] = rank(gz, positions)
    ranks['cf_only'] = rank(cz, positions)
    tail = target_frequency <= q1
    masks = {'all': np.ones(len(g), bool), 'tail': tail, 'non_tail': ~tail}
    groups = {name: {arm: {'row_equal': metric(r[mask]), 'user_equal': metric(r[mask], users[mask])}
                     for arm, r in ranks.items()} for name, mask in masks.items() if mask.any()}
    pairs = {}
    for arm, r in ranks.items():
        if arm in ('raw', 'cf_only', 'current'):
            continue
        pairs[arm] = {}
        for name, mask in masks.items():
            a, b = r[mask], ranks['current'][mask]
            if not len(a):
                continue
            pairs[arm][name] = dict(delta_N10=metric(a)['N10'] - metric(b)['N10'],
                relative_N10=metric(a)['N10'] / metric(b)['N10'] - 1,
                hit10_gained=int(((a <= 10) & (b > 10)).sum()), hit10_lost=int(((a > 10) & (b <= 10)).sum()),
                rank_improved=int((a < b).sum()), rank_worsened=int((a > b).sum()))
    # Target-free identity explains the effective coefficients after restandardizing.
    var_c = cz.var(1); var_p = pz.var(1); cov = (cz * pz).mean(1)
    direct = (cz - .5 * pz).std(1)
    implied = np.sqrt(np.maximum(var_c + .25 * var_p - cov, 0))
    assert np.max(np.abs(direct - implied)) < 1e-12
    alpha = reliability / np.maximum(direct, 1e-6)
    reexpressed = gz + alpha[:, None] * cz - .5 * alpha[:, None] * pz
    assert np.max(np.abs(scores['current'] - reexpressed)) < 1e-12
    def dist(a):
        return dict(mean=float(a.mean()), quantiles=np.quantile(a, [0, .1, .5, .9, 1]).tolist())
    geometry = dict(correlation_cf_frequency=dist(cov), reliability=dist(reliability),
        corrected_score_std=dist(direct), effective_cf_coefficient=dist(alpha),
        equivalent_score_max_error=float(np.max(np.abs(scores['current'] - reexpressed))))
    return dict(groups=groups, comparisons=pairs, geometry=geometry), ranks, (gz, cz, corrected, positions, tail)


def structural_checks():
    base = np.asarray([[0, 1, 2, 3, 4, 5]])
    tail = np.asarray([[False, True, False, True, False, False]])
    for perm in itertools.permutations(range(6)):
        proposal = np.asarray([perm])
        for mode in ARMS[3:]:
            projected = project(base, proposal, tail, mode)
            assert sorted(projected[0]) == list(range(6))
            if mode == 'protect_tail':
                assert projected[0, 1] == 1 and projected[0, 3] == 3
            if mode == 'protect_non_tail':
                assert projected[0, [0, 2, 4, 5]].tolist() == [0, 2, 4, 5]
            assert np.array_equal(tail[0, projected[0]], tail[0])
    return 720 * 3


def main():
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=False)
    started = time.monotonic()
    hashes = {}

    def use(path):
        path = Path(path)
        hashes[str(path.relative_to(ROOT))] = digest(path)
        return path

    def status(state, **extra):
        data = dict(state=state, pid=os.getpid(), updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            elapsed_seconds=round(time.monotonic() - started, 2), new_training_steps=0, gpu_used=False,
            new_beam=False, test_read=False, automatic_next_experiment=False, **extra)
        p = OUT / 'status.tmp'
        p.write_text(json.dumps(data, indent=2) + '\n')
        p.replace(OUT / 'status.json')
        assert time.monotonic() - started < 300, 'CPU_budget_exceeded'

    protocol = dict(arms=ARMS, main='protect_tail', fixed_plain_weight=1, q1={'Toys': 5, 'Beauty': 6},
        fit_parameters=False, model_forward='Frozen 129-parameter B1 only, historical training cache, CPU',
        decision='Both domains N10 > current and H10 >= current with all protected positions exactly identical',
        selection='No new settings selected for deployment; historical calibration weight choice only diagnoses supervision',
        calibration_grid_weights=WEIGHTS + ['cf_only_limit'], budget_seconds=300,
        source_sha256=digest(use(Path(__file__))),
        contract_sha256=digest(use(ROOT / 'plan/第二十阶段/固定低频位置重排_机制核查合同_20260917.md')))
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    try:
        status('RUNNING', stage='structural_invariants')
        structure_cases = structural_checks()
        config = read(use(ROOT / 'experiment/phase20/gacr_fusion_train_v1.json'))
        use(ROOT / 'experiment/phase20/audit_b1_fusion_tradeoff.py')
        old_curve = read(use(ROOT / 'artifacts/phase20/b1_contribution_audit_20260917/summary.json'))
        results = {}
        for dataset, domain in config['domains'].items():
            status('RUNNING', stage='load_frozen_validation', dataset=dataset)
            run = OLD / dataset
            prepared = (ROOT / domain['validation']).parent
            counts, first, train_users = Counter(), {}, set()
            for line in use(prepared / 'train.jsonl').open():
                row = json.loads(line); counts[row['target']] += 1; train_users.add(row['user_id'])
                if len(row['history']) == 1:
                    first.setdefault(row['user_id'], row['history'][0])
            assert set(first) == train_users
            counts.update(first.values())
            frequency = np.asarray([counts[i] for i in range(len(read(use(prepared / 'catalog.json'))['items']))])
            validation = {r['user_id']: r for r in map(json.loads, use(ROOT / domain['validation']).open())}
            with np.load(use(run / 'B1.validation_scores.npz')) as z:
                users, ids, g = [z[k].copy() for k in ('users', 'candidate_ids', 'residual_scores')]
            with np.load(use(ROOT / domain['c1_cf'])) as z:
                assert np.array_equal(users, z['users']) and np.array_equal(ids, z['candidate_ids'])
                cf = z['scores'].astype(np.float64)
            assert len(set(users)) == len(users) == len(validation) == domain['users']
            targets = np.asarray([validation[u]['target'] for u in users])
            hit = ids == targets[:, None]
            positions = np.where(hit.any(1), hit.argmax(1), -1)
            result, ranks, values = evaluate(g, cf, frequency[ids], positions, frequency[targets], users, domain['q1'])
            with use(run / 'per_user.tsv').open() as handle:
                reference = {r['user_id']: r for r in csv.DictReader(handle, delimiter='\t')}
            for arm, col in [('current', 'B1'), ('raw', 'B1_raw')]:
                np.testing.assert_array_equal(ranks[arm], [int(reference[u][col]) for u in users])
            with np.load(use(ROOT / f'artifacts/phase20/b1_contribution_audit_20260917/{dataset}.ranks.npz')) as z:
                assert np.array_equal(users, z['users'])
                names = z['arm_names'].tolist()
                for arm, name in [('plain', 'plain:1.0'), ('pop_constant', 'pop_adjusted:1.0')]:
                    np.testing.assert_array_equal(ranks[arm], z['curve_ranks'][names.index(name)])
            np.savez_compressed(OUT / f'{dataset}.validation_ranks.npz', users=users,
                tail_mask=frequency[targets] <= domain['q1'], **ranks)
            result['exact_validation_rank_parity'] = True
            result['all_candidate_protection_checked'] = True
            status('RUNNING', stage='historical_training_supervision', dataset=dataset)
            manifest = read(use(run / 'cache_manifest.json'))
            cache_path = use(run / 'training_cache.npz')
            assert digest(cache_path) == manifest['cache_sha256']
            records_path = use(run / 'train_candidates.jsonl')
            assert digest(records_path) == manifest['candidate_jsonl_sha256']
            records = list(map(json.loads, records_path.open()))
            train_users = np.asarray([r['user_id'] for r in records])
            train_targets = np.asarray([r['target'] for r in records])
            with np.load(cache_path) as z:
                arrays = {k: z[k].copy() for k in z.files}
            training = read(use(run / 'B1.training.json'))
            state_path = use(run / 'B1.pt')
            assert digest(state_path) == training['checkpoint_sha256']
            state = torch.load(state_path, map_location='cpu')
            state = {k: v.detach() for k, v in state.items()}
            def score(features, base):
                # Exact frozen architecture, no optimizer or autograd.
                x = torch.as_tensor(features, dtype=torch.float32)
                b = torch.as_tensor(base, dtype=torch.float32)
                with torch.no_grad():
                    h = torch.tanh(torch.nn.functional.linear(x, state['network.0.weight'], state['network.0.bias']))
                    y = torch.nn.functional.linear(h, state['network.2.weight'], state['network.2.bias']).squeeze(-1)
                    return (b + .2 * torch.tanh(y)).numpy().astype(np.float64)
            train_g = np.concatenate([score(arrays['features'][i:i + 128], arrays['base'][i:i + 128])
                                      for i in range(0, len(train_users), 128)])
            result['historical_supervision'] = {}
            calibration_choices = {}
            for partition, mask in [('FIT', arrays['fit']), ('calibration', ~arrays['fit'])]:
                part, rr, zz = evaluate(train_g[mask], arrays['cf'][mask], arrays['frequencies'][mask],
                    arrays['positions'][mask], frequency[train_targets[mask]], train_users[mask], domain['q1'])
                result['historical_supervision'][partition] = part
                if partition == 'calibration':
                    for arm, old_arm in [('raw', 'raw'), ('current', 'pcrf')]:
                        for key, old_key in [('N10', 'ndcg@10'), ('H10', 'hit@10'), ('H50', 'hit@50')]:
                            actual = part['groups']['all'][arm]['row_equal'][key]
                            expected = training['train_only_calibration'][old_arm][old_key]
                            assert abs(actual - expected) < 1e-12, ('CPU_calibration_metric_parity_failed', dataset, arm, key, actual, expected)
                    gz, cz, corrected, pos, low = zz
                    for family, cf_score in [('plain', zscore(cz)), ('pop_adjusted', corrected)]:
                        curve = []
                        for weight in WEIGHTS + [None]:
                            r = rank(cf_score if weight is None else gz + weight * cf_score, pos)
                            curve.append(dict(weight='cf_only_limit' if weight is None else weight,
                                **metric(r, train_users[mask])))
                        best = max(curve, key=lambda r: r['N10'])
                        deployed = next(r for r in old_curve['domains'][dataset]['fixed_scalar_curves']
                                        if r['family'] == family and r['weight'] == best['weight'])
                        calibration_choices[family] = dict(user_equal_calibration_choice=best,
                            old_development_result_at_that_weight=deployed, all_calibration_choices=curve,
                            used_for_new_deployment=False)
            result['exact_CPU_calibration_metric_parity'] = True
            result['calibration_selected_weight_diagnostic'] = calibration_choices
            results[dataset] = result
        for path, h in hashes.items():
            assert digest(ROOT / path) == h, ('source_or_input_changed', path)
        supported = all(d['comparisons']['protect_tail']['all']['delta_N10'] > 0 and
                        d['comparisons']['protect_tail']['all']['hit10_gained'] >= d['comparisons']['protect_tail']['all']['hit10_lost']
                        for d in results.values())
        summary = dict(state='COMPLETED', decision='DUAL_DOMAIN_FEASIBILITY_SUPPORTED' if supported else 'DUAL_DOMAIN_FEASIBILITY_NOT_ESTABLISHED',
            domains=results, structural_permutations_checked=structure_cases, input_sha256=hashes,
            new_training_steps=0, gpu_used=False, test_read=False, automatic_next_experiment=False,
            model_forward='Frozen B1 training-cache CPU only', limitations='Repeated development validation; '
                'one upstream/residual seed; historical calibration seen by upstream models; '
                'deterministic position protection is not a novelty or general accuracy guarantee.')
        (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        status('COMPLETED', stage='finished', decision=summary['decision'])
        print(json.dumps({'decision': summary['decision'], 'comparisons': {k: v['comparisons'] for k, v in results.items()}}, indent=2))
    except Exception as error:
        status('FAILED', stage='stopped', error=repr(error))
        raise


if __name__ == '__main__':
    main()
