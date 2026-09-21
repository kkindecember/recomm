"""Audit completed domains without inventing results for cost-blocked domains."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def read(p):
    return json.loads(p.read_text())


def measure(ranks):
    return {f'{name}{k}': sum((int(r <= k) if name == 'H' else
            int(r <= k) / math.log2(r + 1)) for r in ranks) / len(ranks)
            for k in (5, 10, 20, 50) for name in ('N', 'H')}


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='artifacts/phase20/lime_decode_screen_20260921_v1/bounded_v3')
    parser.add_argument('--beauty-output', default=None)
    args = parser.parse_args()
    folder = ROOT / args.output
    verified, unavailable = {}, {}
    for domain in ('Toys', 'Beauty'):
        root = (ROOT / args.beauty_output if domain == 'Beauty' and args.beauty_output else folder) / domain
        status = read(root / 'status.json')
        if status['state'] != 'COMPLETED':
            unavailable[domain] = {'state': status['state'], 'stage': status['stage'],
                                   'error': status.get('error')}
            continue
        summary = read(root / 'summary.json')
        budget = read(root / 'budget.json')
        samples = read(root / 'samples.json')
        signals = np.load(root / 'signals.npz', allow_pickle=False)
        rows = {s: [json.loads(line) for line in (root / f'{s}.jsonl').read_text().splitlines()]
                for s in ('tuning', 'evaluation')}
        assert len(rows['tuning']) == budget['tuning_users']
        assert len(rows['evaluation']) == budget['evaluation_users']
        assert [r['sample_index'] for r in rows['tuning']] == list(range(3, 3 + budget['tuning_users']))
        assert [r['sample_index'] for r in rows['evaluation']] == list(range(131, 131 + budget['evaluation_users']))
        assert set(r['user_id'] for r in rows['tuning']).isdisjoint(r['user_id'] for r in rows['evaluation'])
        checked = 0
        for split, group in rows.items():
            for row in group:
                j = row['sample_index']
                for key in ('split', 'user_id', 'target', 'history'):
                    assert row[key] == samples[j][key]
                assert row['split'] == split
                hist = set(row['history'])
                for name, arm in row['arms'].items():
                    if name == 'LIME':
                        expected = signals['lime_top50'][j].tolist()
                    else:
                        raw = row['candidate_scores'][name]
                        assert len(raw['ids']) == len(set(raw['ids'])) == len(raw['scores']) == 50
                        assert all(math.isfinite(v) for v in raw['scores'])
                        expected = [i for i, s in sorted(zip(raw['ids'], raw['scores']),
                                    key=lambda v: (-v[1], v[0])) if i not in hist]
                    assert expected == arm['items'] and not set(expected).intersection(hist)
                    expected_rank = next((i + 1 for i, item in enumerate(expected) if item == row['target']), 51)
                    assert arm['rank'] == expected_rank
                    checked += 1
        selection = read(root / 'selection.json')
        for signal in ('sasrec', 'lime'):
            options = []
            for alpha in (0., .1, .3):
                key = 'GRAM' if alpha == 0 else f'{signal}_{alpha}'
                value = measure([r['arms'][key]['rank'] for r in rows['tuning']])['N10']
                options.append((value, -alpha, alpha))
            assert max(options)[2] == selection['selected_alpha'][signal] == summary['selected_alpha'][signal]
        for name, metric in summary['metrics'].items():
            calc = measure([r['arms'][name]['rank'] for r in rows['evaluation']])
            for key, value in calc.items():
                assert abs(value - metric[key]) < 1e-12
        for name, comparison in summary['comparisons'].items():
            left, right = name.split('_vs_', 1)
            if right == 'late':
                right = left + '_late'
            a = [r['arms'][left]['rank'] for r in rows['evaluation']]
            b = [r['arms'][right]['rank'] for r in rows['evaluation']]
            delta = measure(a)['N10'] - measure(b)['N10']
            assert abs(delta - comparison['delta_N10']) < 1e-12
            for k in (10, 50):
                assert sum(x <= k and y > k for x, y in zip(a, b)) == comparison[f'gained_H{k}']
                assert sum(x > k and y <= k for x, y in zip(a, b)) == comparison[f'lost_H{k}']
        assert summary['model_unchanged'] and not summary['test_read']
        verified[domain] = {'tuning_users': len(rows['tuning']), 'evaluation_users': len(rows['evaluation']),
                            'predictions_checked': checked, 'summary_sha256': sha(root / 'summary.json'),
                            'selected_alpha': selection['selected_alpha'],
                            'metrics': summary['metrics'], 'comparisons': summary['comparisons']}
        signals.close()
    assert verified, 'No completed domain available to audit'
    manifest = read(folder / 'manifest.json')['sha256']
    for relative, digest in manifest.items():
        assert sha(ROOT / relative) == digest, relative
    if args.beauty_output:
        for relative, digest in read(ROOT / args.beauty_output / 'manifest.json')['sha256'].items():
            assert sha(ROOT / relative) == digest, relative
    result = {'verified_domains': verified, 'unavailable_domains': unavailable,
              'input_hashes_verified': len(manifest), 'independent_of_runner_code': True,
              'two_domain_conclusion_available': len(verified) == 2, 'test_read': False,
              'beauty_output': args.beauty_output}
    path = folder / 'available_domains_audit.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'written': str(path), 'verified_domains': list(verified),
                      'unavailable_domains': list(unavailable)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
