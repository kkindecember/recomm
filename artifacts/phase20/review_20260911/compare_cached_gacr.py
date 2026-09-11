"""Compare existing GACR-v6 and original/PCRF ranks on identical users.

This is a historical cross-pipeline comparison, not new matched inference.
The old GACR reference is the continued C1 backbone, not untouched GRAM.
"""
import csv
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def rank(value):
    return int(value) if value else 51


def metrics(ranks):
    ranks = np.asarray(ranks)
    return {'ndcg10': float(((ranks <= 10) / np.log2(ranks + 1)).mean()),
            'hit10': float((ranks <= 10).mean()), 'hit50': float((ranks <= 50).mean())}


result = {'captured_at': datetime.datetime.now().astimezone().isoformat(),
          'scope': 'Historical cached results, same users and verified validation targets; different model/numerical pipelines. No training or inference.',
          'old_baseline': 'C1 backbone trained with lexical CE and catalog balanced softmax; not untouched original GRAM',
          'seed_scope': 'GACR residual seeds 2023/2024/2025; PCRF item-head seed2023 fixed',
          'test_read': False, 'cells': {}, 'sha256': {}}
for domain in ['Toys', 'Beauty']:
    config = json.loads((ROOT / f'experiment/phase18/config/s18_diff_gram_{domain.lower()}.json').read_text())
    prepared = ROOT / config['prepared']
    catalog = json.loads((prepared / 'catalog.json').read_text())
    with (prepared / 'validation.jsonl').open() as handle:
        targets = {r['user_id']: catalog['items'][r['target']] for r in map(json.loads, handle)}
    reference_path = ROOT / f'artifacts/phase9/p9s_multiseed/{domain}/seed2023/validation/per_user.tsv'
    with reference_path.open() as handle:
        reference = {r['user_id']: r for r in csv.DictReader(handle, delimiter='\t')}
    result['sha256'][str(reference_path.relative_to(ROOT))] = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    for seed in [2023, 2024, 2025]:
        path = ROOT / f'artifacts/phase6/gacr_v6_validation_recovery/{domain}/gacr_v6_seed{seed}_per_user.csv'
        with path.open() as handle:
            records = list(csv.DictReader(handle))
        users = []
        for row in records:
            user, split, target = row['sample_key'].split(':')
            assert split == 'validation' and targets[user] == target
            users.append(user)
        assert len(users) == len(set(users)) == 1024
        ranks = {
            'c1_baseline': np.array([rank(r['baseline_rank']) for r in records]),
            'gacr_v6': np.array([rank(r['candidate_rank']) for r in records]),
            'original_gram': np.array([rank(reference[u]['baseline_rank']) for u in users]),
            'original_pcrf': np.array([rank(reference[u]['pcrf_rank']) for u in users]),
        }
        cell = {'users': len(users), 'metrics': {key: metrics(value) for key, value in ranks.items()},
                'old_baseline_vs_original_rank_changed_users': int((ranks['c1_baseline'] != ranks['original_gram']).sum())}
        cand, base = ranks['gacr_v6'], ranks['original_pcrf']
        cell['gacr_vs_pcrf_hit10_gained_users'] = int(((cand <= 10) & (base > 10)).sum())
        cell['gacr_vs_pcrf_hit10_lost_users'] = int(((cand > 10) & (base <= 10)).sum())
        for comparator in ['original_gram', 'original_pcrf']:
            cell[f'ndcg10_relative_percent_vs_{comparator}'] = 100 * (cell['metrics']['gacr_v6']['ndcg10'] / cell['metrics'][comparator]['ndcg10'] - 1)
        result['cells'][f'{domain}_{seed}'] = cell
        result['sha256'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
destination = OUT / 'gacr_cached_comparison.json'
destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'output': str(destination.relative_to(ROOT)), 'cells': result['cells']}, ensure_ascii=False, indent=2))
