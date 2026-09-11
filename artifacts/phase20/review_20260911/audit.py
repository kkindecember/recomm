"""CPU-only audit of completed stage-20 validation outputs and train banks.

No training modules are imported; no workloads are signalled or modified.
Intervals describe reused selection validation, not independent confirmation.
"""
import collections
import csv
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
P20 = ROOT / 'artifacts/phase20'
PREPARED = ROOT / 'artifacts/phase18/diff_gram/prepared_v2_categories'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ndcg(rank):
    return (rank <= 10) / np.log2(rank + 1)


def paired(a, b):
    delta = ndcg(a) - ndcg(b)
    rng = np.random.default_rng(20260911)
    draws = np.concatenate([
        delta[rng.integers(len(a), size=(50, len(a)))].mean(axis=1)
        for _ in range(80)
    ])
    return {
        'ndcg10_delta': float(delta.mean()),
        'ndcg10_relative_percent': float(delta.mean() / ndcg(b).mean() * 100),
        'descriptive_paired_bootstrap_95ci': np.quantile(draws, [.025, .975]).tolist(),
        'hit10_gained_users': int(((a <= 10) & (b > 10)).sum()),
        'hit10_lost_users': int(((a > 10) & (b <= 10)).sum()),
        'hit50_gained_users': int(((a <= 50) & (b > 50)).sum()),
        'hit50_lost_users': int(((a > 50) & (b <= 50)).sum()),
    }


train = rows(PREPARED / 'train.jsonl')
validation = rows(PREPARED / 'validation.jsonl')
catalog = read(PREPARED / 'catalog.json')
paths = catalog['decoder_paths']
frequencies = collections.Counter(row['target'] for row in train)
initial = {r['user_id']: r['history'][0] for r in train if len(r['history']) == 1}
frequencies.update(initial.values())
indices = read(P20 / 'sprec_gram_toys_v1/selection_indices.json')
assert indices == read(P20 / 'sdpo_gram_toys_v1/selection_indices.json')
users = [validation[i]['user_id'] for i in indices]
assert len(users) == len(set(users)) == 3000
parent = {r['user_id']: r for r in rows(P20 / 'sprec_gram_toys_v1/parent_full.predictions.jsonl')}
reference = read(P20 / 'sprec_gram_toys_v1/matched_parent_reference.json')
output = {
    'captured_at': datetime.datetime.now().astimezone().isoformat(),
    'verification_status': 'ANALYZED',
    'scope': 'single-seed, shared 3000-user selection validation; no new inference or test read',
    'bootstrap': {'seed': 20260911, 'repeats': 4000, 'multiple_comparison_adjusted': False},
    'selection_indices_identical': True,
    'artifact_sha256': {}, 'results': {}, 'banks': {}, 'jobs': {},
}


def prediction_stats(predictions):
    top1 = collections.Counter(r['ranked_item_ids'][0] for r in predictions)
    return {
        'top1_distinct_items': len(top1),
        'most_frequent_top1_count': top1.most_common(1)[0][1],
        'mean_top1_train_frequency': float(np.mean([frequencies[r['ranked_item_ids'][0]] for r in predictions])),
        'mean_top1_response_tokens': float(np.mean([len(paths[r['ranked_item_ids'][0]-1])-1 for r in predictions])),
    }


output['parent_selection_predictions'] = prediction_stats([parent[u] for u in users])
rank_cache = {}
for name in ['sprec_gram_toys_v1', 'sdpo_gram_toys_v1']:
    directory = P20 / name
    manifest = read(directory / 'manifest.json')
    assert manifest['data_manifest']['prepared_files']['train.jsonl'] == digest(PREPARED / 'train.jsonl')
    for key in ['catalog.json', 'validation.jsonl']:
        assert manifest['data_manifest']['prepared_files'][key] == digest(PREPARED / key)
    output['jobs'][name] = read(directory / 'status.json')
    selection = read(directory / 'selection.json')
    for result in selection['outcomes']:
        tag = result['tag']
        with (directory / f'{tag}.ranks.tsv').open() as handle:
            ranks = list(csv.DictReader(handle, delimiter='\t'))
        predictions = rows(directory / f'{tag}.predictions.jsonl')
        assert [r['user_id'] for r in ranks] == users
        assert [r['user_id'] for r in predictions] == users
        for pred, rank in zip(predictions, ranks):
            items = pred['ranked_item_ids']
            assert len(items) == len(set(items)) == 50
            assert all(1 <= item <= len(paths) for item in items)
            assert len(pred['scores']) == 50 and np.isfinite(pred['scores']).all()
            assert pred['gold_item_id'] == parent[pred['user_id']]['gold_item_id']
            expected = items.index(pred['gold_item_id']) + 1 if pred['gold_item_id'] in items else 51
            assert int(rank['raw_rank']) == expected
            assert int(rank['baseline_rank']) == int(reference[pred['user_id']]['baseline_rank'])
            assert int(rank['baseline_pcrf_rank']) == int(reference[pred['user_id']]['pcrf_rank'])
        columns = {key: np.array([int(r[key]) for r in ranks]) for key in ['raw_rank', 'pcrf_rank', 'baseline_rank', 'baseline_pcrf_rank']}
        for metric_key, col in [('raw', 'raw_rank'), ('pcrf', 'pcrf_rank'), ('original_gram', 'baseline_rank'), ('original_gram_pcrf', 'baseline_pcrf_rank')]:
            for k in [5, 10, 20, 50]:
                r = columns[col]
                assert abs(float(((r <= k) / np.log2(r + 1)).mean()) - result[metric_key][f'ndcg@{k}']) < 1e-12
                assert abs(float((r <= k).mean()) - result[metric_key][f'hit@{k}']) < 1e-12
        entry = {'raw': result['raw'], 'pcrf': result['pcrf'],
                 'raw_vs_parent': paired(columns['raw_rank'], columns['baseline_rank']),
                 'pcrf_vs_parent_pcrf': paired(columns['pcrf_rank'], columns['baseline_pcrf_rank']),
                 'prediction_stats': prediction_stats(predictions), 'groups': {}}
        target_freq = np.array([frequencies[r['gold_item_id']] for r in predictions])
        for group, mask in [('frequency_le5', target_freq <= 5), ('frequency_gt5', target_freq > 5)]:
            entry['groups'][group] = {'users': int(mask.sum()), 'raw_ndcg10_delta': float((ndcg(columns['raw_rank'][mask]) - ndcg(columns['baseline_rank'][mask])).mean())}
        output['results'][tag] = entry
        rank_cache[tag] = columns['raw_rank']
        for suffix in ['ranks.tsv', 'metrics.json', 'predictions.jsonl']:
            path = directory / f'{tag}.{suffix}'
            output['artifact_sha256'][str(path.relative_to(ROOT))] = digest(path)
    events = rows(directory / 'events.jsonl')
    output['jobs'][name]['completed_events'] = [
        {k: v for k, v in e.items() if k not in ['raw','pcrf','original_gram','original_gram_pcrf','historical_gram','historical_gram_pcrf','descriptive_paired_ci']}
        for e in events if e['event'] in ['training_started','training_complete','mining_complete','validation_complete']
    ]
    banks = sorted(directory.glob('round*_train_negatives.jsonl')) + list(directory.glob('train_preference_bank.jsonl'))
    for path in banks:
        records = rows(path)
        assert len(records) == len(train)
        seen, history_negatives, total_negatives, collisions = set(), 0, 0, 0
        for record in records:
            i = record['train_index']
            assert i not in seen
            seen.add(i)
            assert record['split'] == 'train' and record['chosen'] == train[i]['target']
            assert record['user_id'] == train[i]['user_id']
            negatives = record['rejected'] if isinstance(record['rejected'], list) else [record['rejected']]
            assert len(set(negatives)) == len(negatives)
            assert all(1 <= item <= len(paths) for item in negatives)
            collision = record['chosen'] in negatives
            collisions += int(collision)
            if 'valid' in record:
                assert record['valid'] == (not collision)
            else:
                assert not collision and len(negatives) == 4
                assert np.isfinite([record['reference_chosen']] + record['reference_rejected']).all()
            history_negatives += sum(item in train[i]['history'] for item in negatives)
            total_negatives += len(negatives)
        output['banks'][path.name] = {'rows': len(records), 'unique_train_indices': len(seen), 'positive_collisions': collisions,
                                      'collision_rate': collisions / len(records), 'negative_slots': total_negatives,
                                      'negative_slots_in_observed_history': history_negatives,
                                      'observed_history_overlap_rate': history_negatives / total_negatives,
                                      'sha256': digest(path)}
for round_index in [1, 2]:
    dpo, sft = f'round{round_index}_dpo', f'round{round_index}_sft'
    output['results'][dpo]['within_round_vs_sft_descriptive'] = paired(rank_cache[dpo], rank_cache[sft])
output['integrity_checks'] = 'All asserted checks passed: users, targets, 50 unique legal candidates, finite scores, recomputed metrics, matched references, prepared hashes, and complete train-bank row alignment.'
destination = OUT / 'audit.json'
destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'output': str(destination.relative_to(ROOT)), 'integrity': output['integrity_checks'],
                  'results': {k: {'raw_vs_parent': v['raw_vs_parent'], 'pcrf_vs_parent_pcrf': v['pcrf_vs_parent_pcrf'], 'prediction_stats': v['prediction_stats'], 'groups': v['groups']} for k, v in output['results'].items()},
                  'parent_selection_predictions': output['parent_selection_predictions'], 'banks': output['banks']}, ensure_ascii=False, indent=2))
