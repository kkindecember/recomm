"""Complete descriptive tables from the independently checked Talos adapter run."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/phase20/talos_exact_threshold_20260917'
ARMS = ('normalized.next_only', 'fusion_mean.next_only', 'legacy.soft_future', 'softmax', 'negative_softmax', 'talos', 'talos_exact')


def main():
    s = json.loads((OUT / 'summary.json').read_text())
    v = json.loads((OUT / 'verification.json').read_text()); assert v['state'] == 'VERIFIED'
    lines = ['# Talos exact-threshold revision: complete descriptive tables', '',
             'Reused development set; three small-head initializations with shared upstream models. No confirmatory test.', '',
             'Primary deployment is the frozen mean of three existing CF components. Other deployments are sensitivity checks.', '']
    for domain, d in s['domains'].items():
        lines += [f'## {domain}', '', '### Primary metrics (all three seeds)', '',
                  '|Arm|N10 mean|N10 SD|H10 mean|H10 SD|Tail N10 mean|Tail H10 mean|Nonfit N10 mean|',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for arm in ARMS:
            m = d['aggregate_metrics'][arm + '.mean']
            entries = [m['all']['N10']['mean'], m['all']['N10']['sd'], m['all']['H10']['mean'], m['all']['H10']['sd'],
                       m['tail']['N10']['mean'], m['tail']['H10']['mean'], m['nonfit']['N10']['mean']]
            lines.append('| ' + arm + ' | ' + ' | '.join(f'{x:.9f}' for x in entries) + ' |')
        lines += ['', '### Every prespecified paired contrast, mean-CF deployment', '',
                  '|Left vs right|Relative N10 mean (%)|SD (percentage points)|All net hits by seed|Tail net hits by seed|Nonfit relative N10 by seed (%)|',
                  '|---|---:|---:|---|---|---|']
        for name, by_fusion in d['aggregate_comparisons'].items():
            c = by_fusion['mean']; gain = c['all']['relative_N10']
            nonfit = ', '.join(f'{x*100:+.5f}' for x in c['nonfit']['relative_N10']['values'])
            lines.append(f"|{name.replace('__vs__', ' vs ')}|{gain['mean']*100:+.5f}|{gain['sd']*100:.5f}|{c['all']['hit_net']['values']}|{c['tail']['hit_net']['values']}|{nonfit}|")
        lines += ['', '### Mean gained and lost Top10 hits (three seeds)', '',
                  '|Left vs right|All gained|All lost|Tail gained|Tail lost|', '|---|---:|---:|---:|---:|']
        for name in d['aggregate_comparisons']:
            values = []
            for group in ('all', 'tail'):
                for field in ('hit_gained', 'hit_lost'):
                    values.append(sum(seed['comparisons'][name]['mean'][group][field] for seed in d['seeds'].values()) / 3)
            lines.append('| ' + name.replace('__vs__', ' vs ') + ' | ' + ' | '.join(f'{x:.3f}' for x in values) + ' |')
        lines += ['', '### All deployments', '', '|Arm|Deployment|N10 mean|H10 mean|Tail N10 mean|Tail H10 mean|', '|---|---|---:|---:|---:|---:|']
        for arm in ARMS:
            for fusion in ('raw', 'current', 'seed2024', 'seed2025', 'mean'):
                m = d['aggregate_metrics'][arm + '.' + fusion]
                values = [m[group][metric]['mean'] for group in ('all', 'tail') for metric in ('N10', 'H10')]
                lines.append('| ' + arm + ' | ' + fusion + ' | ' + ' | '.join(f'{x:.9f}' for x in values) + ' |')
        lines += ['', '### Training objective and update checks', '',
                  '|Seed|Arm|Initial unscaled loss|Final unscaled loss|Max parameter gradient norm|Parameter change L2|',
                  '|---|---|---:|---:|---:|---:|']
        for seed in (20260917, 20260918, 20260919):
            train = json.loads((OUT / f'{domain}.{seed}.training.json').read_text())
            for arm, t in train.items():
                lines.append(f"|{seed}|{arm}|{t['curve'][0]['loss']:.9f}|{t['final_loss']:.9f}|{max(x['grad_norm'] for x in t['curve']):.9f}|{t['parameter_delta_l2']:.9f}|")
        lines += ['', '### Prespecified gates', '', '```json', json.dumps({key: d[key] for key in ('utility_gates', 'specificity_gates', 'topk_gate')}, indent=2), '```', '']
    lines += ['Decision: ' + s['decision'], 'Retained arms: ' + repr(s['retained_arms']), 'Independent verification: ' + v['state'], '']
    (OUT / 'review_tables.md').write_text('\n'.join(lines))
    print(OUT / 'review_tables.md')


if __name__ == '__main__': main()
