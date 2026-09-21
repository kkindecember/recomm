"""Read-only diagnostic of current candidate support and train transition support."""
import csv
import hashlib
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/phase20/candidate_transition_audit_20260917'
CONTRACT = ROOT / 'plan/第二十阶段/候选覆盖与转移关系_零训练核查合同_20260917.md'
BASELINE = ROOT / 'artifacts/phase20/strong_baseline_20260917/baseline.json'
SEEDS = [20260917, 20260918, 20260919]
GROUPS = ['last_direct', 'earlier_direct_only', 'two_edge_only', 'no_path_within_two_edges']
INPUTS = {}


def record(path):
    path = Path(path)
    INPUTS[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def save(name, data):
    path = OUT / name
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def status(state, **kwargs):
    save('status.json', dict(state=state, updated_at=datetime.now().astimezone().isoformat(),
         training_steps=0, gpu_used=False, test_read=False, **kwargs))


def metric(rank, mask):
    r = rank[mask]
    return {'users': int(len(r)), 'N10': float(np.where(r <= 10, 1 / np.log2(r + 1), 0).mean()),
            'H10': float((r <= 10).mean())} if len(r) else {'users': 0, 'N10': None, 'H10': None}


def main():
    start = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=False)
    status('RUNNING', stage='input_and_graph_audit')
    record(__file__)
    record(CONTRACT)
    base = json.loads(record(BASELINE).read_text())
    summary = {'interpretation': 'Development diagnostic; target-aware strata and oracle availability, not a new method.',
               'groups': GROUPS, 'domains': {}, 'seeds': SEEDS}
    for domain, prep, original in [
        ('Toys', 'artifacts/phase18/diff_gram/prepared_v2_categories', 'cf0_b2_toys_item_p2a'),
        ('Beauty', 'artifacts/phase18/diff_gram/beauty/prepared_v1', 'p9x_beauty_item_head')]:
        status('RUNNING', domain=domain, stage='cached_rank_analysis')
        prep = ROOT / prep
        record(prep / 'catalog.json')
        graph = defaultdict(set)
        reverse = defaultdict(set)
        with record(prep / 'train.jsonl').open() as f:
            for line in f:
                row = json.loads(line)
                a, b = row['history'][-1], row['target']
                graph[a].add(b)
                reverse[b].add(a)
        with record(prep / 'validation.jsonl').open() as f:
            rows = [json.loads(line) for line in f]
        lookup = {r['user_id']: r for r in rows}
        assert len(lookup) == len(rows)
        ranks, raw = [], []
        for seed in SEEDS:
            p = ROOT / f'artifacts/phase20/future_fusion_alignment_20260917/{domain}.{seed}.predictions.npz'
            with np.load(record(p), allow_pickle=False) as a:
                if not ranks:
                    users = a['users'].copy()
                    ids = a['candidate_ids'].copy()
                    pos = a['positions'].copy()
                    tail = a['tail_mask'].copy()
                else:
                    for key, expected in [('users', users), ('candidate_ids', ids), ('positions', pos), ('tail_mask', tail)]:
                        assert np.array_equal(a[key], expected), key
                names = a['rank_names'].tolist()
                ranks.append(a['ranks'][names.index('normalized.next_only.mean')].copy())
                raw.append(a['ranks'][names.index('normalized.next_only.raw')].copy())
        ranks, raw = np.array(ranks), np.array(raw)
        assert set(users) == set(lookup)
        targets = np.array([lookup[u]['target'] for u in users])
        covered = pos >= 0
        assert np.array_equal((ids == targets[:, None]).any(1), covered)
        assert np.array_equal(ids[np.where(covered)[0], pos[covered]], targets[covered])
        assert np.all(ranks[:, ~covered] == 51)
        cf = []
        for seed in [2023, 2024, 2025]:
            p = ROOT / 'artifacts/phase9' / (f'{original}/best_validation_ranks.tsv' if seed == 2023 else
                  f'p9s_multiseed/{domain}/seed{seed}/item_head/best_validation_ranks.tsv')
            with record(p).open() as f:
                recs = list(csv.DictReader(f, delimiter='\t'))
            c = {r['user_id']: r for r in recs}
            assert len(c) == len(recs) and set(c) == set(users)
            assert all(int(c[u]['target_item_id']) == targets[i] for i, u in enumerate(users))
            cf.append([int(c[u]['item_rank']) for u in users])
        cf = np.array(cf)
        assert np.all(cf >= 1)
        category = []
        for u, t in zip(users, targets):
            h = lookup[u]['history'][-20:]
            if t in graph[h[-1]]:
                g = 0
            elif any(t in graph[x] for x in h):
                g = 1
            elif any(graph[x] & reverse[t] for x in h):
                g = 2
            else:
                g = 3
            category.append(g)
        category = np.array(category)
        result = {'users': len(users), 'covered': int(covered.sum()), 'coverage': float(covered.mean()),
                  'unique_train_edges': sum(map(len, graph.values())), 'groups': {}}
        for g, mask in [('all', np.ones(len(users), dtype=bool))] + [(name, category == i) for i, name in enumerate(GROUPS)]:
            value = {'users': int(mask.sum()), 'candidate_covered': int((mask & covered).sum()),
                     'baseline': [metric(r, mask) for r in ranks], 'raw': [metric(r, mask) for r in raw],
                     'cf': [metric(r, mask) for r in cf], 'fusion_minus_raw': [], 'outside_candidate_cf': {}}
            for r, b in zip(ranks, raw):
                gains = int((mask & (r <= 10) & (b > 10)).sum())
                losses = int((mask & (r > 10) & (b <= 10)).sum())
                value['fusion_minus_raw'].append(dict(hit_gained=gains, hit_lost=losses, hit_net=gains-losses,
                    delta_N10=metric(r, mask)['N10'] - metric(b, mask)['N10'] if mask.any() else None))
            for k in [10, 50]:
                counts = [int((mask & ~covered & (r <= k)).sum()) for r in cf]
                value['outside_candidate_cf'][str(k)] = dict(per_cf_seed=counts,
                    any_cf_oracle=int((mask & ~covered & (cf <= k).any(0)).sum()),
                    all_cf=int((mask & ~covered & (cf <= k).all(0)).sum()))
            result['groups'][g] = value
        for i in range(3):
            for metric_name in ['N10', 'H10']:
                expected = base['domains'][domain]['aggregate_metrics']['all'][metric_name]['values'][i]
                assert abs(result['groups']['all']['baseline'][i][metric_name] - expected) < 1e-12
        np.savez_compressed(OUT / f'{domain}.per_user.npz', users=users, targets=targets, positions=pos,
                            category=category, baseline_ranks=ranks, raw_ranks=raw, cf_ranks=cf, tail=tail)
        summary['domains'][domain] = result
        assert time.monotonic() - start < 600
    summary['elapsed_seconds'] = time.monotonic() - start
    summary['input_sha256'] = INPUTS
    summary['output_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.npz')}
    save('summary.json', summary)
    status('COMPLETED', elapsed_seconds=summary['elapsed_seconds'], verification='BASELINE_IDENTITY_AND_INPUT_ALIGNMENT_PASSED',
           automatic_training=False, paper_method_confirmed=False)
    print(json.dumps({d: {k: v for k, v in r.items() if k != 'groups'} for d, r in summary['domains'].items()}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if OUT.exists():
            status('FAILED', error=repr(error))
        raise
