"""Post-outcome descriptive decomposition of verified rank changes; no fitting."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/phase20/future_fusion_alignment_20260917'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decompose(before, after):
    n = len(before)
    assert n > 0
    old_hit, new_hit = before <= 10, after <= 10
    old_gain = np.where(old_hit, 1 / np.log2(before + 1), 0.)
    new_gain = np.where(new_hit, 1 / np.log2(after + 1), 0.)
    groups = {
        'both_hit': old_hit & new_hit,
        'entered': ~old_hit & new_hit,
        'left': old_hit & ~new_hit,
        'both_miss': ~old_hit & ~new_hit,
    }
    assert np.all(sum(m.astype(int) for m in groups.values()) == 1)
    result = {'users': n}
    for name, m in groups.items():
        result[name + '_users'] = int(m.sum())
        result[name + '_N10_contribution'] = float((new_gain[m] - old_gain[m]).sum() / n)
    result.update(
        delta_N10=float((new_gain - old_gain).mean()),
        hit_net=int(new_hit.sum() - old_hit.sum()),
        retained_rank_better=int((groups['both_hit'] & (after < before)).sum()),
        retained_rank_worse=int((groups['both_hit'] & (after > before)).sum()),
        retained_rank_same=int((groups['both_hit'] & (after == before)).sum()),
    )
    np.testing.assert_allclose(sum(result[k + '_N10_contribution'] for k in groups), result['delta_N10'], atol=1e-15, rtol=0)
    assert result['entered_users'] - result['left_users'] == result['hit_net']
    assert result['both_miss_N10_contribution'] == 0
    return result


def main():
    started = time.monotonic()
    summary = json.loads((OUT / 'summary.json').read_text())
    verification = json.loads((OUT / 'verification.json').read_text())
    assert verification['state'] == 'VERIFIED'
    inputs = [OUT / 'summary.json', OUT / 'verification.json']
    result = {'state': 'CHECKED', 'analysis_timing': 'post_outcome_exploratory',
              'training_steps': 0, 'test_read': False, 'domains': {}}
    cells = 0
    for domain, dv in summary['domains'].items():
        outputs = {}
        for seed, sv in dv['seeds'].items():
            path = OUT / f'{domain}.{seed}.predictions.npz'
            inputs.append(path)
            with np.load(path, allow_pickle=False) as p:
                ranks = dict(zip(p['rank_names'].tolist(), p['ranks']))
                masks = dict(all=np.ones(len(p['users']), dtype=bool), tail=p['tail_mask'],
                             non_tail=~p['tail_mask'], fit=p['in_fit'], nonfit=~p['in_fit'])
                record = {}
                for contrast, cv in sv['comparisons'].items():
                    left, right = contrast.split('__vs__')
                    record[contrast] = {}
                    for deployment, groups in cv.items():
                        record[contrast][deployment] = {}
                        for group, reference in groups.items():
                            m = masks[group]
                            values = decompose(ranks[right + '.' + deployment][m], ranks[left + '.' + deployment][m])
                            np.testing.assert_allclose(values['delta_N10'], reference['delta_N10'], rtol=0, atol=1e-14)
                            assert values['hit_net'] == reference['hit_net']
                            assert values['entered_users'] == reference['hit_gained']
                            assert values['left_users'] == reference['hit_lost']
                            record[contrast][deployment][group] = values
                            cells += 1
                outputs[seed] = record
        aggregate = {}
        exemplar = next(iter(outputs.values()))
        for contrast, cv in exemplar.items():
            aggregate[contrast] = {}
            for deployment, groups in cv.items():
                aggregate[contrast][deployment] = {}
                for group, values in groups.items():
                    aggregate[contrast][deployment][group] = {
                        key: {'values': [seed[contrast][deployment][group][key] for seed in outputs.values()],
                              'mean': float(np.mean([seed[contrast][deployment][group][key] for seed in outputs.values()]))}
                        for key in values
                    }
        result['domains'][domain] = {'seeds': outputs, 'aggregate': aggregate}
    result.update(cells_checked=cells, input_sha256={str(p.relative_to(ROOT)): sha(p) for p in inputs},
                  source_sha256=sha(Path(__file__)), elapsed_seconds=time.monotonic() - started)
    (OUT / 'rank_attribution.json').write_text(json.dumps(result, indent=2) + '\n')
    selected = (
        'normalized.next_only__vs__legacy.next_only',
        'fusion_mean.next_only__vs__normalized.next_only',
        'fusion_mean.soft_future__vs__legacy.soft_future',
        'fusion_mean.soft_future__vs__normalized.soft_future',
        'fusion_mean.soft_future__vs__fusion_mean.next_only',
    )
    lines = ['# Rank attribution (post-outcome, descriptive)', '',
             'All contrasts, seeds, deployments, and groups are in rank_attribution.json. No training or test.', '',
             'N10 contributions below sum to the total delta; each uses the full group denominator.', '',
             '|Domain|Left vs right|Group|New hits|Lost hits|Retained-hit contribution|New-hit contribution|Lost-hit contribution|Total N10 delta|',
             '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for domain, dv in result['domains'].items():
        for contrast in selected:
            for group in ('all', 'tail'):
                a = dv['aggregate'][contrast]['mean'][group]
                keys = ('entered_users', 'left_users', 'both_hit_N10_contribution', 'entered_N10_contribution', 'left_N10_contribution', 'delta_N10')
                values = [a[k]['mean'] for k in keys]
                lines.append(f"|{domain}|{contrast.replace('__vs__', ' vs ')}|{group}|{values[0]:.2f}|{values[1]:.2f}|{values[2]:+.9f}|{values[3]:+.9f}|{values[4]:+.9f}|{values[5]:+.9f}|")
    (OUT / 'rank_attribution.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('domains', 'input_sha256')}, indent=2))


if __name__ == '__main__':
    main()
