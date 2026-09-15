"""Independent cached-result audit; does not import experiment/model code."""
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'artifacts/phase20/gacr_controls_multiseed_v1'
KEYS = [f'{kind}@{k}' for kind in ('ndcg', 'hit') for k in (5, 10, 20, 50)]


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def ranks(path):
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        names = reader.fieldnames[1:]
        result = {}
        for row in reader:
            user = row['user_id']
            assert user not in result
            result[user] = {name: int(row[name]) for name in names}
    assert all(1 <= rank <= 51 for row in result.values() for rank in row.values())
    return names, result


def value(rank, key):
    kind, k = key.split('@')
    return (1. if kind == 'hit' else 1. / math.log2(rank + 1)) if rank <= int(k) else 0.


def close(actual, expected):
    assert abs(actual - expected) < 1e-12, (actual, expected)


def main():
    summary, status = read(OUT / 'summary.json'), read(OUT / 'status.json')
    config_path = ROOT / 'experiment/phase20/gacr_controls_multiseed_v1.json'
    config = read(config_path)
    assert status['state'] == summary['state'] == 'COMPLETED'
    assert status['completed_scientific_phases'] == status['total_scientific_phases'] == 6
    assert all(t['state'] == 'COMPLETED' and t['exit_code'] == 0 for t in status['tasks'])
    assert not summary['test_read'] and summary['new_optimizer_steps'] == 0
    checked = Counter()
    details = {}
    for path, expected in config['source_sha256'].items():
        assert sha(ROOT / path) == expected, path
        checked['source_hashes'] += 1
    for domain, cfg in config['domains'].items():
        base = OUT / domain
        ds, protocol = read(base / 'summary.json'), read(base / 'protocol.json')
        assert ds == summary['domains'][domain]
        assert protocol['config'] == config and protocol['config_sha256'] == sha(config_path)
        paths = {name: ROOT / path for name, path in cfg['inputs'].items()}
        for name, expected in cfg['expected_sha256'].items():
            assert sha(paths[name]) == expected == ds['input_sha256'][name] == protocol['input_sha256'][name], name
            checked['domain_input_hashes'] += 1
        validation = {r['user_id']: r for r in rows(paths['validation'])}
        assert len(validation) == cfg['users']
        counts, first = Counter(), {}
        for row in rows(paths['train']):
            counts[row['target']] += 1
            if len(row['history']) == 1:
                first.setdefault(row['user_id'], row['history'][0])
        counts.update(first.values())
        fit = set(paths['c1_training_users'].read_text().splitlines())
        groups = {
            'all_validation': list(validation),
            f'target_frequency_le{cfg["q1"]}': [u for u, r in validation.items() if counts[r['target']] <= cfg['q1']],
            f'target_frequency_gt{cfg["q1"]}': [u for u, r in validation.items() if counts[r['target']] > cfg['q1']],
            'outside_c1_training_users': [u for u in validation if u not in fit],
            'inside_c1_training_users': [u for u in validation if u in fit],
        }
        c1 = {r['user_id']: set(r['ranked_item_ids']) for r in rows(paths['c1'])}
        union = {r['user_id']: set(r['union_ids']) for r in rows(paths['features'])}
        _, reference = ranks(paths['reference_ranks'])
        seed2023 = None
        phase_hashes = {}
        for seed in (2023, 2024, 2025):
            phase = 'controls_seed2023' if seed == 2023 else f'seed{seed}'
            folder = base / phase
            result = read(folder / 'summary.json')
            names, records = ranks(folder / 'per_user.tsv')
            assert set(records) == set(validation)
            if seed == 2023:
                seed2023 = records
                for u in records:
                    for name in reference[u]:
                        assert records[u][name] == reference[u][name]
                if 'controls_reference' in paths:
                    assert ranks(paths['controls_reference'])[1] == records
            else:
                for u, r in records.items():
                    assert r['original_pcrf'] == seed2023[u]['original_pcrf']
                    assert r['seed2023_full'] == seed2023[u]['gacr_v6_pcrf']
                    assert r['seed2023_fixed'] == seed2023[u]['c1_fixed_candidates_residual_pcrf']
            for group, users in groups.items():
                for name in names:
                    expected = result['groups'][group][name]
                    assert expected['users'] == len(users)
                    for key in KEYS:
                        close(math.fsum(value(records[u][name], key) for u in users) / len(users), expected[key])
                        checked['metric_values'] += 1
                for comparison, group_values in result['comparisons'].items():
                    a, b = comparison.split('_minus_')
                    expected = group_values[group]
                    assert expected['users'] == len(users)
                    for key in ('ndcg@10', 'hit@10'):
                        close(math.fsum(value(records[u][a], key) - value(records[u][b], key) for u in users) / len(users), expected[key]['delta'])
                    for k in (5, 10, 20, 50):
                        assert sum(records[u][a] <= k < records[u][b] for u in users) == expected[f'hit{k}_gained']
                        assert sum(records[u][b] <= k < records[u][a] for u in users) == expected[f'hit{k}_lost']
                    checked['paired_comparisons'] += 1
            for method in (('fixed',) if seed == 2023 else ('full', 'fixed')):
                seen = set()
                for row in rows(folder / f'{method}.predictions.jsonl'):
                    u, ids, scores = row['user_id'], row['ranked_item_ids'], row['scores']
                    assert u in validation and u not in seen
                    seen.add(u)
                    target = validation[u]['target']
                    assert row['split'] == 'validation' and row['gold_item_id'] == target
                    assert len(ids) == len(set(ids)) == len(scores) == 50
                    assert all(math.isfinite(x) for x in scores)
                    assert all(x >= y for x, y in zip(scores, scores[1:]))
                    assert set(ids) <= union[u]
                    arm = 'c1_fixed_candidates_residual_pcrf' if seed == 2023 else method
                    assert (records[u][arm] <= 50) == (target in ids)
                    if method == 'fixed':
                        assert set(ids) == c1[u]
                        checked['fixed_candidate_sets'] += 1
                    else:
                        assert records[u]['gacr_raw'] == (ids.index(target) + 1 if target in ids else 51)
                        checked['full_raw_ranks'] += 1
                    checked['prediction_rows'] += 1
                assert seen == set(validation)
            for method in ('full', 'fixed'):
                arm = {'full': 'gacr_v6_pcrf', 'fixed': 'c1_fixed_candidates_residual_pcrf'}[method] if seed == 2023 else method
                assert ds['seed_metrics'][str(seed)][method] == result['groups']['all_validation'][arm]
            phase_hashes[phase] = {name: sha(folder / name) for name in ('summary.json', 'per_user.tsv')}
        for method in ('full', 'fixed'):
            for key in KEYS:
                values = [ds['seed_metrics'][str(seed)][method][key] for seed in (2023, 2024, 2025)]
                expected = ds['aggregate'][method][key]
                for name, actual in [('mean', statistics.mean(values)), ('min', min(values)), ('max', max(values)),
                                     ('sample_sd_across_residual_seeds', statistics.stdev(values)),
                                     ('mean_delta_vs_original_pcrf', statistics.mean(values) - ds['baseline'][key])]:
                    close(actual, expected[name])
                assert sum(v > ds['baseline'][key] for v in values) == expected['positive_seed_count_vs_original_pcrf']
                checked['aggregate_metrics'] += 1
        details[domain] = {'users': len(validation), 'phase_hashes': phase_hashes}
        print(domain, 'PASSED', flush=True)
    output = {
        'state': 'PASSED', 'verified_at': datetime.now().astimezone().isoformat(), 'checks': dict(checked),
        'domains': details, 'summary_sha256': sha(OUT / 'summary.json'), 'metric_tolerance': 1e-12,
        'scope': 'Independent standard-library recomputation from integer ranks, candidate membership/order checks, and frozen source/input hashes.',
        'limits': 'Does not rerun model scores, final PCRF sorting, or bootstrap intervals. Prediction JSONL stores pre-PCRF residual order; TSV stores final PCRF ranks.',
        'new_training': False, 'new_beam_inference': False, 'test_read': False,
    }
    Path(__file__).with_name('multiseed_completion_verification.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'state': output['state'], 'checks': output['checks']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
