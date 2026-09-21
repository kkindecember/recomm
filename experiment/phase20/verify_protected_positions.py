"""Independent cached-rank checks; no fitting or additional search."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/phase20/protected_position_audit_20260917'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    summary = json.loads((OUT / 'summary.json').read_text())
    config = json.loads((ROOT / 'experiment/phase20/gacr_fusion_train_v1.json').read_text())
    for name, expected in summary['input_sha256'].items():
        assert sha(ROOT / name) == expected, name
    verified_metrics = verified_ranks = 0
    details = {}
    for dataset, spec in config['domains'].items():
        result = summary['domains'][dataset]
        with np.load(OUT / f'{dataset}.validation_ranks.npz') as f:
            saved = {k: f[k] for k in f.files}
        users = saved['users']
        assert len(users) == len(set(users))
        tail = saved['tail_mask']
        lookup = np.zeros(52)
        lookup[1:11] = [1 / np.log2(r + 1) for r in range(1, 11)]
        # Recount all metrics from rank histograms rather than row-wise means.
        for group, mask in [('all', np.ones(len(users), bool)), ('tail', tail), ('non_tail', ~tail)]:
            for arm, reported in result['groups'][group].items():
                hist = np.bincount(saved[arm][mask], minlength=52)
                assert reported['row_equal']['rows'] == hist.sum()
                for k in (5, 10, 20, 50):
                    hits = hist[1:k + 1].sum() / hist.sum()
                    ndcg = sum(hist[r] / np.log2(r + 1) for r in range(1, k + 1)) / hist.sum()
                    for key, value in [(f'H{k}', hits), (f'N{k}', ndcg)]:
                        for weighting in ('row_equal', 'user_equal'):
                            assert abs(value - reported[weighting][key]) < 1e-12
                            verified_metrics += 1
        for arm, groups in result['comparisons'].items():
            for group, mask in [('all', np.ones(len(users), bool)), ('tail', tail), ('non_tail', ~tail)]:
                old, new = saved['current'][mask], saved[arm][mask]
                expected = groups[group]
                delta = float((lookup[new] - lookup[old]).mean())
                assert abs(delta - expected['delta_N10']) < 1e-12
                for key, value in {
                    'hit10_gained': ((old > 10) & (new <= 10)).sum(),
                    'hit10_lost': ((old <= 10) & (new > 10)).sum(),
                    'rank_improved': (new < old).sum(),
                    'rank_worsened': (new > old).sum(),
                }.items():
                    assert value == expected[key]
        np.testing.assert_array_equal(saved['protect_tail'][tail], saved['current'][tail])
        np.testing.assert_array_equal(saved['protect_non_tail'][~tail], saved['current'][~tail])
        np.testing.assert_array_equal(saved['group_slots'], np.where(tail, saved['protect_non_tail'], saved['protect_tail']))
        # Rebuild the target's rank by counting preceding same-group candidates;
        # do not call the original permutation projection or evaluation functions.
        prepared = (ROOT / spec['validation']).parent
        count, first = Counter(), {}
        for line in (prepared / 'train.jsonl').open():
            row = json.loads(line)
            count[row['target']] += 1
            if len(row['history']) == 1:
                first.setdefault(row['user_id'], row['history'][0])
        count.update(first.values())
        target = {r['user_id']: r['target'] for r in map(json.loads, (ROOT / spec['validation']).open())}
        with np.load(ROOT / f'artifacts/phase20/gacr_fusion_train_v1/{dataset}/B1.validation_scores.npz') as f:
            ids, g = f['candidate_ids'], f['residual_scores']
            np.testing.assert_array_equal(users, f['users'])
        with np.load(ROOT / spec['c1_cf']) as f:
            cf = f['scores'].astype(np.float64)
            np.testing.assert_array_equal(ids, f['candidate_ids'])
        freq = np.asarray([[count[int(i)] for i in row] for row in ids])
        q = spec['q1']

        def z(a):
            centered = a - a.mean()
            return centered / max(float(np.sqrt(np.mean(centered ** 2))), 1e-6)

        for row, uid in enumerate(users):
            initial = np.argsort(-g[row], kind='stable')
            gg, cc, ff, ii = g[row, initial], cf[row, initial], freq[row, initial], ids[row, initial]
            low = ff <= q
            assert tail[row] == (count[target[uid]] <= q)
            current_order = np.argsort(-(z(gg) + (1 - low[:10].mean()) * z(z(cc) - .5 * z(np.log1p(ff)))), kind='stable')
            plain_order = np.argsort(-(z(gg) + z(z(cc))), kind='stable')
            location = np.flatnonzero(ii == target[uid])
            predicted = {a: 51 for a in ('current', 'plain', 'group_slots', 'protect_tail', 'protect_non_tail')}
            if len(location):
                t = location.item()
                original_rank = int(np.flatnonzero(current_order == t).item()) + 1
                plain_rank = int(np.flatnonzero(plain_order == t).item()) + 1
                group = low[t]
                slots = np.flatnonzero(low[current_order] == group)
                preceding = int((low[plain_order[:plain_rank - 1]] == group).sum())
                projected_rank = int(slots[preceding]) + 1
                predicted.update(current=original_rank, plain=plain_rank, group_slots=projected_rank,
                                 protect_tail=original_rank if group else projected_rank,
                                 protect_non_tail=projected_rank if group else original_rank)
            for arm, value in predicted.items():
                assert saved[arm][row] == value, (dataset, row, arm)
                verified_ranks += 1
        # Post-hoc, pointwise descriptive uncertainty; not adjusted for prior
        # development-set reuse and not a confirmatory significance claim.
        rng = np.random.default_rng(20260917)
        intervals = {}
        for arm in result['comparisons']:
            delta = lookup[saved[arm]] - lookup[saved['current']]
            values, counts = np.unique(delta, return_counts=True)
            samples = rng.multinomial(len(delta), counts / len(delta), size=2000)
            means = (samples @ values) / len(delta)
            intervals[arm] = dict(delta_N10=float(delta.mean()),
                descriptive_paired_user_bootstrap_95_percent=np.quantile(means, [.025, .975]).tolist())
        details[dataset] = dict(rows=len(users), tail_rows=int(tail.sum()),
                               bootstrap=intervals, seed=20260917, resamples=2000)
    verdict = dict(state='VERIFIED', input_hashes_checked=len(summary['input_sha256']),
        metrics_checked=verified_metrics, independently_rebuilt_target_ranks=verified_ranks,
        domains=details, original_decision=summary['decision'], new_training_steps=0, gpu_used=False,
        statistical_scope='Post-hoc pointwise paired-user intervals on reused development data; '
                          'no adjustment for adaptive research, no seed uncertainty, no confirmatory claims.',
        source_sha256=sha(Path(__file__)), summary_sha256=sha(OUT / 'summary.json'))
    (OUT / 'verification.json').write_text(json.dumps(verdict, indent=2) + '\n')
    print(json.dumps(verdict, indent=2))


if __name__ == '__main__':
    main()
