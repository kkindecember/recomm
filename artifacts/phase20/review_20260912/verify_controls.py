"""Independently verify all reported cached-control metrics from integer ranks."""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'artifacts/phase20/gacr_score_controls_toys_v2'


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


def value(rank, kind, k):
    return (1.0 if kind == 'hit' else 1.0 / math.log2(rank + 1)) if rank <= k else 0.0


def main():
    summary = json.loads((OUT / 'summary.json').read_text())
    cfg = summary['config']
    paths = {key: ROOT / name for key, name in cfg['inputs'].items()}
    paths.update(config=ROOT / cfg['config_path'], code=ROOT / 'experiment/phase20/gacr_score_controls.py')
    assert summary['input_sha256'] == {key: sha(path) for key, path in paths.items()}
    validation = {row['user_id']: row for row in rows(paths['validation'])}
    counts, first = Counter(), {}
    for row in rows(paths['train']):
        counts[row['target']] += 1
        if len(row['history']) == 1:
            first.setdefault(row['user_id'], row['history'][0])
    counts.update(first.values())
    fit = set(paths['c1_training_users'].read_text().splitlines())
    with (OUT / 'per_user.tsv').open() as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        methods = reader.fieldnames[1:]
        records = {r['user_id']: {name: int(r[name]) for name in methods} for r in reader}
    assert len(records) == 19412 and set(records) == set(validation)
    assert all(1 <= rank <= 51 for row in records.values() for rank in row.values())
    groups = {
        'all_validation': list(records),
        'target_frequency_le5': [u for u in records if counts[validation[u]['target']] <= 5],
        'target_frequency_gt5': [u for u in records if counts[validation[u]['target']] > 5],
        'outside_c1_training_users': [u for u in records if u not in fit],
        'inside_c1_training_users': [u for u in records if u in fit],
    }
    checked_metrics = 0
    for group, users in groups.items():
        for name in methods:
            expected = summary['groups'][group][name]
            assert len(users) == expected['users']
            for kind in ('hit', 'ndcg'):
                for k in (5, 10, 20, 50):
                    actual = math.fsum(value(records[u][name], kind, k) for u in users) / len(users)
                    assert abs(actual - expected[f'{kind}@{k}']) < 1e-12
                    checked_metrics += 1
    checked_contrasts = 0
    for a, b in cfg['comparisons']:
        for group, users in groups.items():
            expected = summary['comparisons'][a + '_minus_' + b][group]
            gained = sum(records[u][a] <= 10 < records[u][b] for u in users)
            lost = sum(records[u][b] <= 10 < records[u][a] for u in users)
            assert gained == expected['hit10_gained'] and lost == expected['hit10_lost']
            for kind in ('hit', 'ndcg'):
                delta = math.fsum(value(records[u][a], kind, 10) - value(records[u][b], kind, 10) for u in users) / len(users)
                assert abs(delta - expected[f'{kind}@10']['delta']) < 1e-12
            checked_contrasts += 1
    c1 = {r['user_id']: r for r in rows(paths['c1'])}
    fixed_users = set()
    for row in rows(OUT / 'fixed_candidates_residual.predictions.jsonl'):
        u = row['user_id']
        assert u not in fixed_users
        fixed_users.add(u)
        assert row['split'] == 'validation' and row['gold_item_id'] == validation[u]['target']
        assert len(row['ranked_item_ids']) == len(set(row['ranked_item_ids'])) == 50
        assert set(row['ranked_item_ids']) == set(c1[u]['ranked_item_ids'])
        assert all(math.isfinite(x) for x in row['scores'])
        assert all(x >= y for x, y in zip(row['scores'], row['scores'][1:]))
        assert (records[u]['c1_fixed_candidates_residual_pcrf'] <= 50) == (records[u]['c1'] <= 50)
    assert fixed_users == set(records)
    result = {
        'state': 'PASSED', 'users': len(records), 'metric_values_checked': checked_metrics,
        'paired_contrasts_checked': checked_contrasts, 'fixed_candidate_sets_checked': len(fixed_users),
        'source_hashes_unchanged': True, 'metric_tolerance': 1e-12,
        'bootstrap_repeated': False, 'new_training': False, 'test_read': False,
        'summary_sha256': sha(OUT / 'summary.json'), 'per_user_sha256': sha(OUT / 'per_user.tsv'),
        'scope': 'Independent standard-library metric/count calculation; no additional model arms or inference.',
    }
    target = Path(__file__).with_name('controls_verification.json')
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
