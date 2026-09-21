"""Frozen, training-free candidate-wise transition-support screen."""
import hashlib
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/phase20/direct_transition_screen_20260917'
SEEDS = [20260917, 20260918, 20260919]
ARMS = ['registered', 'raw', 'direct_support', 'uniform_mass', 'frequency_shuffle', 'softmax', 'negative_softmax']
INPUTS = {}


def source(path):
    p = ROOT / path
    INPUTS[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return p


def save(name, data):
    p = OUT / name
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(p)


def status(state, **kw):
    save('status.json', dict(state=state, updated_at=datetime.now().astimezone().isoformat(),
         gpu_used=False, test_read=False, training_steps=0, **kw))


def rank(scores, pos):
    order = np.argsort(-scores, axis=1, kind='stable')
    hit = order == pos[:, None]
    return np.where(pos >= 0, np.argmax(hit, axis=1) + 1, 51)


def metrics(r, mask):
    r = r[mask]
    return {'users': len(r), 'N10': float(np.where(r <= 10, 1 / np.log2(r + 1), 0).mean()),
            'H10': float((r <= 10).mean()), 'H50': float((r <= 50).mean())}


def compare(r, b, mask):
    rm, bm = metrics(r, mask), metrics(b, mask)
    gained = int((mask & (r <= 10) & (b > 10)).sum())
    lost = int((mask & (r > 10) & (b <= 10)).sum())
    return dict(delta_N10=rm['N10']-bm['N10'], relative_N10=rm['N10']/bm['N10']-1,
                delta_H10=rm['H10']-bm['H10'], gained=gained, lost=lost, hit_net=gained-lost)


def main():
    start = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=False)
    status('RUNNING')
    source('experiment/phase20/screen_direct_transition_support.py')
    source('plan/第二十阶段/直接转移支持修正_零训练筛查合同_20260917.md')
    source('artifacts/phase20/strong_baseline_20260917/baseline.json')
    source('artifacts/phase20/strong_baseline_20260917/list_loss_controls.json')
    summary = {'arms': ARMS, 'seeds': SEEDS, 'domains': {}, 'training_steps': 0}
    for d, prep in [('Toys','artifacts/phase18/diff_gram/prepared_v2_categories'),
                    ('Beauty','artifacts/phase18/diff_gram/beauty/prepared_v1')]:
        status('RUNNING', domain=d, stage='candidate_masks')
        graph = defaultdict(set)
        with source(prep + '/train.jsonl').open() as f:
            for line in f:
                row = json.loads(line)
                graph[row['history'][-1]].add(row['target'])
        with source(prep + '/validation.jsonl').open() as f:
            rows = {r['user_id']: r for r in map(json.loads, f)}
        with np.load(source(f'artifacts/phase20/future_factor_screen_20260917/{d}.audit_validation_cache.npz')) as a:
            users, ids, pos, tail, cf = [a[k].copy() for k in ['users','candidate_ids','positions','tail_mask','cf_mean']]
        with np.load(source(f'artifacts/phase20/candidate_transition_audit_20260917/{d}.per_user.npz')) as a:
            assert np.array_equal(users, a['users'])
            categories = a['category'].copy()
        cf_file = 'cf0_b2_toys_item_p2a' if d == 'Toys' else 'p9x_beauty_item_head'
        import csv
        with source(f'artifacts/phase9/{cf_file}/best_validation_ranks.tsv').open() as f:
            freq_rows = {r['user_id']: r for r in csv.DictReader(f, delimiter='\t')}
        frequencies = defaultdict(int)
        # Full training sequences are reconstructed as first item plus every target.
        seen_users = set()
        with (ROOT / prep / 'train.jsonl').open() as f:
            for line in f:
                r = json.loads(line)
                if r['user_id'] not in seen_users:
                    assert len(r['history']) == 1
                    frequencies[r['history'][0]] += 1
                    seen_users.add(r['user_id'])
                frequencies[r['target']] += 1
        for u in users:
            assert frequencies[rows[u]['target']] == int(freq_rows[u]['train_frequency'])
        mask = np.empty(ids.shape, dtype=bool)
        freq = np.empty(ids.shape, dtype=np.int64)
        for i, u in enumerate(users):
            supported = set().union(*(graph[x] for x in rows[u]['history'][-20:]))
            mask[i] = [x in supported for x in ids[i]]
            freq[i] = [frequencies[x] for x in ids[i]]
        shuffled = mask.copy()
        rng = np.random.default_rng(20260917)
        freq_order = np.argsort(freq, axis=1, kind='stable')
        for i in range(len(users)):
            for j in range(0, 50, 10):
                idx = freq_order[i, j:j+10]
                shuffled[i, idx] = rng.permutation(mask[i, idx])
                assert shuffled[i, idx].sum() == mask[i, idx].sum()
        assert np.array_equal(mask.sum(1), shuffled.sum(1))
        domain = {'users': len(users), 'support_fraction': float(mask.mean()), 'seeds': {}}
        for seed in SEEDS:
            p = source(f'artifacts/phase20/talos_candidate_screen_20260917/{d}.{seed}.predictions.npz')
            with np.load(p) as a:
                assert np.array_equal(a['users'], users) and np.array_equal(a['candidate_ids'], ids)
                names = a['raw_names'].tolist()
                g = a['raw_scores'][names.index('normalized.next_only')]
                score = [g+cf, g, g+mask*cf, g+mask.mean(1)[:,None]*cf, g+shuffled*cf,
                         a['raw_scores'][names.index('softmax')]+cf,
                         a['raw_scores'][names.index('negative_softmax')]+cf]
                ranks = np.array([rank(s, pos) for s in score])
                old_names = a['rank_names'].tolist()
                for idx, old in [(0,'normalized.next_only.mean'),(1,'normalized.next_only.raw'),
                                 (5,'softmax.mean'),(6,'negative_softmax.mean')]:
                    assert np.array_equal(ranks[idx], a['ranks'][old_names.index(old)])
                in_fit = a['in_fit'].copy()
            groups = {'all': np.ones(len(users),dtype=bool), 'tail': tail, 'non_fit': ~in_fit}
            groups.update({f'relation_{i}': categories == i for i in range(4)})
            cell = {'metrics': {}, 'comparisons': {}}
            for i, arm in enumerate(ARMS):
                cell['metrics'][arm] = {gn: metrics(ranks[i], gm) for gn, gm in groups.items()}
                cell['comparisons'][arm] = {gn: compare(ranks[i], ranks[0], gm) for gn, gm in groups.items()}
            domain['seeds'][str(seed)] = cell
            np.savez_compressed(OUT / f'{d}.{seed}.npz', users=users, candidate_ids=ids, positions=pos,
                                masks=mask, shuffled_masks=shuffled, frequencies=freq, tail=tail,
                                in_fit=in_fit, categories=categories, arm_names=np.array(ARMS), ranks=ranks)
        mean = lambda arm, group, key: float(np.mean([domain['seeds'][str(s)]['comparisons'][arm][group][key] for s in SEEDS]))
        decisions = {'relative_N10': mean('direct_support','all','relative_N10') >= .005,
                     'overall_hits': mean('direct_support','all','hit_net') >= 0,
                     'tail_N10': mean('direct_support','tail','delta_N10') >= 0,
                     'tail_hits': mean('direct_support','tail','hit_net') >= 0,
                     'two_positive_seeds': sum(domain['seeds'][str(s)]['comparisons']['direct_support']['all']['delta_N10'] > 0 for s in SEEDS) >= 2,
                     'beats_uniform': mean('direct_support','all','delta_N10') > mean('uniform_mass','all','delta_N10'),
                     'beats_shuffle': mean('direct_support','all','delta_N10') > mean('frequency_shuffle','all','delta_N10')}
        domain['advancement_checks'] = decisions
        domain['advance'] = all(decisions.values())
        summary['domains'][d] = domain
        assert time.monotonic()-start < 600
    summary.update(elapsed_seconds=time.monotonic()-start, input_sha256=INPUTS,
                   advance=all(d['advance'] for d in summary['domains'].values()))
    summary['output_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.npz')}
    save('summary.json', summary)
    status('COMPLETED', elapsed_seconds=summary['elapsed_seconds'], advance=summary['advance'],
           automatic_training=False, paper_method_confirmed=False)
    print(json.dumps({d: {'advance': r['advance'], 'checks': r['advancement_checks']} for d,r in summary['domains'].items()}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if OUT.exists():
            status('FAILED', error=repr(error))
        raise
