"""Descriptive cached-validation audit; no fitting or deployment selection.

Compare the current B1+PCRF point with fixed scalar-fusion curves on the same
scores/candidates. Validation-selected envelopes are intentionally optimistic
references, not held-out performance estimates of newly selected methods.
"""
import csv
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/phase20/b1_contribution_audit_20260917'
WEIGHTS = [i / 40 for i in range(81)] + [4., 8.]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def zscore(x):
    return (x - x.mean(1, keepdims=True)) / np.maximum(x.std(1, keepdims=True), 1e-6)


def rank(scores, positions):
    order = np.argsort(-scores, axis=1, kind='stable')
    inv = np.argsort(order, axis=1)
    return np.where(positions >= 0, inv[np.arange(len(scores)), np.maximum(positions, 0)] + 1, 51)


def metrics(ranks, tail):
    result = {}
    for name, mask in [('all', np.ones(len(ranks), bool)), ('tail', tail), ('non_tail', ~tail)]:
        r = ranks[mask]
        result[name] = {'users': len(r), 'N10': float(np.where(r <= 10, 1 / np.log2(r + 1), 0).mean()),
                        'H10': float((r <= 10).mean()), 'H50': float((r <= 50).mean())}
    return result


def main():
    OUT.mkdir(exist_ok=False)
    started = time.monotonic()

    def status(state, **kwargs):
        data = dict(state=state, updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), pid=os.getpid(),
                    elapsed_seconds=round(time.monotonic() - started, 2), new_training_steps=0,
                    model_forward=False, gpu_used=False, test_read=False, automatic_next_experiment=False, **kwargs)
        temporary = OUT / 'status.tmp'
        temporary.write_text(json.dumps(data, indent=2) + '\n')
        temporary.replace(OUT / 'status.json')

    config_path = ROOT / 'experiment/phase20/gacr_fusion_train_v1.json'
    protocol = dict(scope=__doc__, datasets=['Toys', 'Beauty'], base='frozen B1 residual scores and C1 beam50',
                    families=['z(B1)+lambda*z(CF)', 'z(B1)+lambda*z(z(CF)-0.5*z(log1p(train_frequency)))'],
                    scalar_weights=WEIGHTS, include_cf_only_limit=True,
                    baseline_comparison_axes=['all.N10', 'tail.N10'], secondary_axes=['all.H10', 'tail.H10'],
                    baseline_envelope_selection='same repeated validation, exploratory only; never deploy selected weight',
                    budget_seconds=300, old_config_sha256=digest(config_path), source_sha256=digest(Path(__file__)))
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    hashes = {str(config_path.relative_to(ROOT)): digest(config_path), str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))}
    status('RUNNING', stage='load_cached_scores')
    try:
        config = read(config_path)
        results = {}
        for dataset, domain in config['domains'].items():
            status('RUNNING', stage='reference_parity', dataset=dataset)
            run = ROOT / 'artifacts/phase20/gacr_fusion_train_v1' / dataset
            prepared = (ROOT / domain['validation']).parent
            inputs = [run / 'B1.validation_scores.npz', ROOT / domain['c1_cf'], run / 'per_user.tsv',
                      run / 'summary.json', prepared / 'catalog.json', prepared / 'train.jsonl', ROOT / domain['validation']]
            for path in inputs:
                h = digest(path)
                rel = str(path.relative_to(ROOT))
                if rel in domain['input_sha256']:
                    assert h == domain['input_sha256'][rel], ('frozen_input_changed', rel)
                hashes[rel] = h
            with np.load(inputs[0]) as data:
                users, ids, g, saved_rel = (data[k].copy() for k in ['users', 'candidate_ids', 'residual_scores', 'reliability'])
            with np.load(inputs[1]) as data:
                assert np.array_equal(users, data['users']) and np.array_equal(ids, data['candidate_ids'])
                cf = data['scores'].astype(np.float64)
            counts, first, train_users = Counter(), {}, set()
            for line in inputs[5].open():
                row = json.loads(line)
                counts[row['target']] += 1
                train_users.add(row['user_id'])
                if len(row['history']) == 1:
                    first.setdefault(row['user_id'], row['history'][0])
            assert set(first) == train_users
            counts.update(first.values())
            freq = np.asarray([counts[i] for i in range(len(read(inputs[4])['items']))])
            validation = {row['user_id']: row for row in map(json.loads, inputs[6].open())}
            assert len(users) == len(set(users)) == len(validation) == domain['users']
            targets = np.asarray([validation[u]['target'] for u in users])
            matches = ids == targets[:, None]
            assert (matches.sum(1) <= 1).all() and np.isfinite(g).all() and np.isfinite(cf).all()
            positions = np.where(matches.any(1), matches.argmax(1), -1)
            tail = freq[targets] <= domain['q1']
            order = np.argsort(-g, axis=1, kind='stable')
            positions = np.where(positions >= 0, np.argsort(order, axis=1)[np.arange(len(g)), np.maximum(positions, 0)], -1)
            g = np.take_along_axis(g, order, 1)
            cf = np.take_along_axis(cf, order, 1)
            pop = np.take_along_axis(freq[ids], order, 1)
            reliability = 1 - (pop[:, :10] <= domain['q1']).mean(1)
            np.testing.assert_array_equal(reliability, saved_rel)
            gz, cz, pz = zscore(g), zscore(cf), zscore(np.log1p(pop))
            adjusted = zscore(cz - .5 * pz)
            pcrf_ranks = rank(gz + reliability[:, None] * adjusted, positions)
            with inputs[2].open() as handle:
                reference = {r['user_id']: r for r in csv.DictReader(handle, delimiter='\t')}
            np.testing.assert_array_equal(pcrf_ranks, [int(reference[u]['B1']) for u in users])
            np.testing.assert_array_equal(rank(gz, positions), [int(reference[u]['B1_raw']) for u in users])
            current = metrics(pcrf_ranks, tail)
            old_summary = read(inputs[3])
            for group, old_group in [('all', 'all_validation'), ('tail', f"target_frequency_le{domain['q1']}")]:
                for key, old_key in [('N10', 'ndcg@10'), ('H10', 'hit@10'), ('H50', 'hit@50')]:
                    assert abs(current[group][key] - old_summary['groups'][old_group]['B1'][old_key]) < 1e-12
            curves, all_ranks, names = [], [], []
            for family, scores in [('plain', zscore(cz)), ('pop_adjusted', adjusted)]:
                status('RUNNING', stage='scalar_curve', dataset=dataset, family=family)
                for weight in WEIGHTS + [None]:
                    assert time.monotonic() - started < protocol['budget_seconds'], 'audit_budget_exceeded'
                    ranks = rank(scores if weight is None else gz + weight * scores, positions)
                    assert np.array_equal(ranks <= 50, pcrf_ranks <= 50)
                    row = dict(family=family, weight='cf_only_limit' if weight is None else weight, **metrics(ranks, tail))
                    curves.append(row)
                    all_ranks.append(ranks.astype(np.int16))
                    names.append(f"{family}:{row['weight']}")
            envelopes = {}
            for family in ('plain', 'pop_adjusted'):
                rows = [r for r in curves if r['family'] == family]
                feasible = [r for r in rows if r['tail']['N10'] >= current['tail']['N10']]
                dominant = [r for r in feasible if r['all']['N10'] >= current['all']['N10']]
                envelopes[family] = dict(best_all_N10=max(rows, key=lambda r: r['all']['N10']),
                    best_all_at_least_current_tail_N10=max(feasible, key=lambda r: r['all']['N10']) if feasible else None,
                    points_dominating_current_on_N10_axes=dominant)
            results[dataset] = dict(q1=domain['q1'], exact_B1_and_raw_rank_parity=True,
                current_B1_PCRF=current, B1_raw=metrics(rank(gz, positions), tail),
                fixed_scalar_curves=curves, exploratory_envelopes=envelopes)
            np.savez_compressed(OUT / f'{dataset}.ranks.npz', users=users, tail_mask=tail,
                current=pcrf_ranks, arm_names=np.asarray(names), curve_ranks=np.stack(all_ranks))
        for p, h in hashes.items():
            assert digest(ROOT / p) == h, ('input_changed_during_analysis', p)
        summary = dict(state='COMPLETED', domains=results, input_sha256=hashes,
            new_training_steps=0, model_forward=False, gpu_used=False, test_read=False,
            automatic_next_experiment=False, selected_deployment_method=None,
            limitations='Repeated validation; one frozen upstream/residual seed; 168 scalar settings per domain; '
                        'descriptive optimistic envelope, no significance/novelty or held-out tuning claim; '
                        'only two specific families and the stated finite grid, not all learned/simple fusion methods.')
        (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        with (OUT / 'tradeoff.csv').open('w') as handle:
            fields = ['dataset', 'family', 'weight', 'all_N10', 'tail_N10', 'all_H10', 'tail_H10']
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for dataset, data in results.items():
                for row in [dict(family='current_B1_PCRF', weight='', **data['current_B1_PCRF'])] + data['fixed_scalar_curves']:
                    writer.writerow(dict(dataset=dataset, family=row['family'], weight=row['weight'],
                        **{f'{group}_{key}': row[group][key] for group in ('all', 'tail') for key in ('N10', 'H10')}))
        status('COMPLETED', stage='finished', summary=str(OUT / 'summary.json'))
        print(json.dumps({d: {'current': r['current_B1_PCRF'], 'envelopes': r['exploratory_envelopes']}
                          for d, r in results.items()}, indent=2))
    except Exception as error:
        status('FAILED', stage='stopped', error=repr(error))
        raise


if __name__ == '__main__':
    main()
