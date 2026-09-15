"""Audit completed validation artifacts without model inference or new arms."""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
METHODS = ('original_gram', 'original_pcrf', 'c1', 'c1_pcrf', 'gacr_v6', 'gacr_v6_pcrf')


def read_json(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def metric(rank, kind, k):
    return (1.0 if kind == 'hit' else 1.0 / math.log2(rank + 1)) if rank <= k else 0.0


def compare(records, users, a, b):
    result = {'users': len(users)}
    for k in (5, 10, 20, 50):
        gained = sum(records[u][a] <= k < records[u][b] for u in users)
        lost = sum(records[u][b] <= k < records[u][a] for u in users)
        result[f'hit@{k}'] = {'gained': gained, 'lost': lost, 'net': gained - lost,
                              'delta': (gained - lost) / len(users)}
        result[f'ndcg@{k}'] = math.fsum(metric(records[u][a], 'ndcg', k) -
                                        metric(records[u][b], 'ndcg', k) for u in users) / len(users)
    partition = {'both_covered': [], 'gained_coverage': [], 'lost_coverage': [], 'neither_covered': []}
    for u in users:
        ac, bc = records[u][a] <= 50, records[u][b] <= 50
        key = 'both_covered' if ac and bc else 'gained_coverage' if ac else 'lost_coverage' if bc else 'neither_covered'
        partition[key].append(u)
    result['coverage_accounting'] = {
        key: {'users': len(us), 'ndcg10_contribution_to_full_group': math.fsum(
            metric(records[u][a], 'ndcg', 10) - metric(records[u][b], 'ndcg', 10) for u in us) / len(users),
            'hit10_net': sum((records[u][a] <= 10) - (records[u][b] <= 10) for u in us)}
        for key, us in partition.items()
    }
    return result


def main():
    cfg = read_json(ROOT / 'experiment/phase20/gacr_v6_pcrf_beauty_v1.json')
    run, parent = ROOT / cfg['output'], ROOT / cfg['shared_parent_output']
    summary = read_json(run / 'summary.json')
    for path in (run, parent, ROOT / cfg['queue_output']):
        assert read_json(path / 'status.json')['state'] == 'COMPLETED', path
    assert summary['users'] == 22363 and summary['new_optimizer_steps'] == 0 and not summary['test_read']
    hash_results = {name: sha(ROOT / name) for name in cfg['frozen_inputs']}
    assert hash_results == cfg['frozen_inputs'], 'Frozen source drift'
    prepared = ROOT / read_json(ROOT / cfg['data_config'])['prepared']
    validation_rows = list(rows(prepared / 'validation.jsonl'))
    validation = {r['user_id']: r for r in validation_rows}
    assert len(validation) == len(validation_rows) == 22363
    counts, first = Counter(), {}
    for row in rows(prepared / 'train.jsonl'):
        counts[row['target']] += 1
        if len(row['history']) == 1:
            first.setdefault(row['user_id'], row['history'][0])
    counts.update(first.values())
    fit = set((ROOT / cfg['c1_training_users']).read_text().splitlines())
    with (run / 'per_user.tsv').open() as handle:
        table = list(csv.DictReader(handle, delimiter='\t'))
    records = {r['user_id']: {name: int(r[name]) for name in METHODS} for r in table}
    assert len(records) == len(table) == len(validation) and set(records) == set(validation)
    assert all(1 <= r <= 51 for row in records.values() for r in row.values())
    groups = {
        'all_validation': list(records),
        'outside_c1_training_users': [u for u in records if u not in fit],
        'inside_c1_training_users': [u for u in records if u in fit],
        'target_frequency_le6': [u for u in records if counts[validation[u]['target']] <= 6],
        'target_frequency_gt6': [u for u in records if counts[validation[u]['target']] > 6],
    }
    checked, max_error, csv_rows = 0, 0.0, []
    for group, users in groups.items():
        for name in METHODS:
            expected = summary['groups'][group]['metrics'][name]
            assert expected['users'] == len(users)
            result = {'dataset': 'Beauty', 'group': group, 'method': name, 'users': len(users)}
            for k in (5, 10, 20, 50):
                for kind in ('hit', 'ndcg'):
                    key = f'{kind}@{k}'
                    actual = math.fsum(metric(records[u][name], kind, k) for u in users) / len(users)
                    error = abs(actual - expected[key])
                    assert error < 1e-12, (group, name, key)
                    if group == 'all_validation':
                        assert abs(actual - summary['metrics'][name][key]) < 1e-12
                    max_error = max(max_error, error)
                    result[key] = actual
                    checked += 1
            csv_rows.append(result)
    comparisons = {}
    for b in ('original_gram', 'original_pcrf', 'c1_pcrf', 'gacr_v6', 'c1'):
        comparisons[b] = {g: compare(records, users, 'gacr_v6_pcrf', b) for g, users in groups.items()}
        if b in summary['paired_comparisons']:
            for kind in ('hit', 'ndcg'):
                v = comparisons[b]['all_validation'][f'{kind}@10']
                delta = v['delta'] if kind == 'hit' else v
                assert abs(delta - summary['paired_comparisons'][b][f'{kind}@10']['delta']) < 1e-12
    for group in groups:
        pair = comparisons['original_pcrf'][group]
        for kind in ('hit', 'ndcg'):
            v = pair[f'{kind}@10']
            delta = v['delta'] if kind == 'hit' else v
            assert abs(delta - summary['groups'][group]['combined_vs_original_pcrf'][f'{kind}@10']['delta']) < 1e-12
    predictions_checked = 0
    for path, name in ((parent / 'parent_full.predictions.jsonl', 'original_gram'),
                       (run / 'c1.predictions.jsonl', 'c1'), (run / 'gacr.predictions.jsonl', 'gacr_v6')):
        seen = set()
        for row in rows(path):
            u, candidates, scores = row['user_id'], row['ranked_item_ids'], row['scores']
            assert u not in seen and u in validation
            seen.add(u)
            assert row['split'] == 'validation' and row['gold_item_id'] == validation[u]['target']
            assert len(candidates) == len(set(candidates)) == len(scores) == 50
            assert all(math.isfinite(s) for s in scores)
            assert all(a >= b for a, b in zip(scores, scores[1:]))
            target = validation[u]['target']
            rank = candidates.index(target) + 1 if target in candidates else 51
            assert rank == records[u][name]
            pcrf_name = {'original_gram': 'original_pcrf', 'c1': 'c1_pcrf', 'gacr_v6': 'gacr_v6_pcrf'}[name]
            assert (rank <= 50) == (records[u][pcrf_name] <= 50)
            predictions_checked += 1
        assert seen == set(validation)
    matched = read_json(parent / 'matched_parent_reference.json')
    assert set(matched) == set(records)
    for u, row in matched.items():
        assert int(row['baseline_rank']) == records[u]['original_gram']
        assert int(row['pcrf_rank']) == records[u]['original_pcrf']
        assert int(row['target_frequency']) == counts[validation[u]['target']]
    source_paths = [run / 'summary.json', run / 'per_user.tsv', parent / 'summary.json',
                    parent / 'matched_parent_reference.json', ROOT / cfg['config_path']]
    result = {
        'state': 'PASSED', 'verification_status': 'ANALYZED', 'users': len(records),
        'metric_values_checked': checked, 'max_metric_error': max_error,
        'prediction_rows_checked': predictions_checked, 'frozen_input_hashes_checked': len(hash_results),
        'parent_reference_users_checked': len(matched),
        'group_sizes': {g: len(us) for g, us in groups.items()},
        'comparisons_combined_minus': comparisons,
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in source_paths},
        'bootstrap_repeated': False, 'new_training': False, 'new_model_inference': False, 'test_read': False,
        'scope': 'Independent standard-library rank metrics, hit counts, coverage accounting and source hashes. '
                 'PCRF scoring and bootstrap are not independently rerun; this is not training reproducibility.',
    }
    (OUT / 'beauty_completion_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    with (OUT / 'beauty_full_validation_metrics.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps({k: v for k, v in result.items() if k not in ('comparisons_combined_minus', 'source_sha256')}, ensure_ascii=False))
    print(json.dumps({'combined_minus_original_pcrf': comparisons['original_pcrf']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
