"""Post-hoc cached diagnostic; no fitting, inference, or test-set access."""
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'artifacts/phase21/single_fusion_20261014_v1'
OUT = ROOT / 'artifacts/phase21/expert_motivation_20261015_v1'


def read(path):
    return json.loads(path.read_text())


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def rank(ids, values, target, history):
    valid = ~np.isin(ids, history)
    order = ids[valid][np.lexsort((ids[valid], -values[valid]))]
    pos = np.flatnonzero(order == target)
    return (int(pos[0]) + 1 if len(pos) else 2**30), order


def lcp(a, b):
    for j, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return j
    return min(len(a), len(b))


def main():
    begin = time.monotonic()
    OUT.mkdir(exist_ok=False)
    write(OUT / 'status.json', dict(state='RUNNING', training=False, gpu=False, test_read=False))
    results, hashes = {}, {}
    for domain in ('Toys', 'Beauty'):
        folder = SOURCE / domain
        for name in ('samples.json', 'catalog.json', 'predictions.jsonl', 'ranks.npz',
                     'pure_ranks.npz', 'validation/item.npy'):
            p = folder / name
            h = hashlib.sha256()
            with p.open('rb') as stream:
                for chunk in iter(lambda: stream.read(2**20), b''):
                    h.update(chunk)
            hashes[str(p.relative_to(ROOT))] = h.hexdigest()
        samples = read(folder / 'samples.json')
        catalog = read(folder / 'catalog.json')
        paths = [[]] + [p[1:-1] for p in catalog['decoder_paths']]
        assert all(p[0] == 0 and p[-1] == 1 for p in catalog['decoder_paths'])
        assert len(paths) == len(catalog['items'])
        predictions = [json.loads(s) for s in (folder / 'predictions.jsonl').read_text().splitlines()]
        features = np.load(folder / 'validation/item.npy', mmap_mode='r')
        saved = np.load(folder / 'ranks.npz')
        pure = np.load(folder / 'pure_ranks.npz')
        assert len(samples) == len(predictions) == len(features) == 512
        ids = np.arange(1, len(paths))
        rows = []
        for j, (sample, pred) in enumerate(zip(samples, predictions)):
            assert sample['role'] == 'development'
            assert sample['user_id'] == pred['user_id'] == saved['users'][j]
            assert sample['history'] == sample['full_history'][-20:]
            target, history = sample['target'], sample['history']
            row = dict(user_id=sample['user_id'], history_length=len(sample['full_history']))
            for name in ('native', 'user_late', 'user_guided'):
                r, order = rank(np.asarray(pred[name]['ids']), np.asarray(pred[name]['scores']), target, history)
                assert r == saved[name][j]
                row[name + '_rank'] = r
                if name == 'native':
                    row['max_target_prefix_shared_by_native_top10'] = max(lcp(paths[target], paths[int(i)]) for i in order[:10])
            for col, name in enumerate(('sasrec', 'text')):
                row[name + '_rank'], _ = rank(ids, features[j, :, col], target, history)
            mean_rank, _ = rank(ids, features[j, :, :2].mean(-1), target, history)
            assert mean_rank == pure['mean'][j]
            rows.append(row)
        with (OUT / (domain + '_per_user.csv')).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        groups = {
            'all': lambda r: True,
            'native_miss10': lambda r: r['native_rank'] > 10,
            'text_hit10_native_miss10': lambda r: r['text_rank'] <= 10 < r['native_rank'],
            'sasrec_hit10_native_miss10': lambda r: r['sasrec_rank'] <= 10 < r['native_rank'],
            'user_guided_rescue10': lambda r: r['user_guided_rank'] <= 10 < r['native_rank'],
            'user_guided_harm10': lambda r: r['native_rank'] <= 10 < r['user_guided_rank'],
        }
        desc = {}
        for name, select in groups.items():
            subset = [r for r in rows if select(r)]
            desc[name] = dict(n=len(subset), history_longer_than20=sum(r['history_length'] > 20 for r in subset),
                max_shared_prefix_histogram=dict(sorted(Counter(r['max_target_prefix_shared_by_native_top10'] for r in subset).items())),
                target_in_native50=sum(r['native_rank'] <= 50 for r in subset))
        short = [r for r in rows if r['history_length'] <= 20]
        desc['history_at_most20'] = dict(n=len(short),
            metrics={name: dict(N10=float(np.mean([1 / np.log2(r[name + '_rank'] + 1) if r[name + '_rank'] <= 10 else 0 for r in short])),
                               H10=float(np.mean([r[name + '_rank'] <= 10 for r in short])))
                     for name in ('native', 'user_late', 'user_guided')})
        results[domain] = desc
    write(OUT / 'summary.json', dict(domains=results, scope='post-hoc reused development512; descriptive only',
        limitations=['Lexical prefix sharing is not an audited attribute or semantic-error label.',
                     'Expert hit opportunity is not a realizable selection policy.',
                     'Short-history subgroup controls visible event count only, not model training or text budget.',
                     'No new recommendation method, causal attribution, or significance claim.']))
    hashes[str(Path(__file__).relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write(OUT / 'manifest.json', dict(sha256=hashes))
    write(OUT / 'status.json', dict(state='COMPLETED', elapsed_seconds=time.monotonic()-begin,
        training=False, gpu=False, test_read=False, checked_users=1024,
        checks='native/user_late/user_guided ranks and pure mean ranks recomputed for all users'))
    print(json.dumps(results, ensure_ascii=False))


if __name__ == '__main__':
    main()
