"""Create complete, unselected comparison tables from the verified alignment screen."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/phase20/future_fusion_alignment_20260917'


def main():
    s = json.loads((OUT / 'summary.json').read_text())
    a = json.loads((OUT / 'verification.json').read_text())
    assert a['state'] == 'VERIFIED'
    lines = ['# Final-fusion training: complete descriptive tables', '',
        'All numbers use the reused development set. Seeds vary only the small correction. SD is descriptive.', '']
    for domain, v in s['domains'].items():
        lines += [f'## {domain}', '', '### Mean-CF deployment', '',
            '|Training mode|Labels|N10 mean|N10 SD|H10 mean|Tail N10 mean|Tail H10 mean|',
            '|---|---|---:|---:|---:|---:|---:|']
        for mode in ('legacy', 'normalized', 'fusion_mean'):
            for label in ('next_only', 'global_local', 'soft_future'):
                m = v['aggregate_metrics'][mode + '.' + label + '.mean']
                lines.append(f"|{mode}|{label}|{m['all']['N10']['mean']:.9f}|{m['all']['N10']['sd']:.9f}|{m['all']['H10']['mean']:.9f}|{m['tail']['N10']['mean']:.9f}|{m['tail']['H10']['mean']:.9f}|")
        lines += ['', '### Primary mean-CF paired contrasts', '',
            '|Left vs right|Relative N10 mean (%)|SD (percentage points)|All hit net by seed|Tail hit net by seed|Nonfit N10 by seed (%)|',
            '|---|---:|---:|---|---|---|']
        for name, fusions in v['aggregate_comparisons'].items():
            c = fusions['mean']; m = c['all']['relative_N10']; hit = c['all']['hit_net']['values']; tail = c['tail']['hit_net']['values']
            nonfit = ', '.join(f'{n*100:+.4f}' for n in c['nonfit']['relative_N10']['values'])
            lines.append(f"|{name.replace('__vs__',' vs ')}|{m['mean']*100:+.5f}|{m['sd']*100:.5f}|{hit}|{tail}|{nonfit}|")
        lines += ['', '### All deployments: soft-future per training mode', '',
            '|Training mode|Deployment|N10 mean|H10 mean|Tail N10 mean|Tail H10 mean|', '|---|---|---:|---:|---:|---:|']
        for mode in ('legacy', 'normalized', 'fusion_mean'):
            for fusion in ('raw', 'current', 'seed2024', 'seed2025', 'mean'):
                m = v['aggregate_metrics'][mode + '.soft_future.' + fusion]
                lines.append(f"|{mode}|{fusion}|{m['all']['N10']['mean']:.9f}|{m['all']['H10']['mean']:.9f}|{m['tail']['N10']['mean']:.9f}|{m['tail']['H10']['mean']:.9f}|")
        lines += ['', '### Prespecified gate', '', '```json', json.dumps(v['gate'], indent=2), '```', '']
    lines += ['Decision: ' + s['decision'], '', 'Independent verification: ' + a['state'], '']
    (OUT / 'review_tables.md').write_text('\n'.join(lines))
    print(OUT / 'review_tables.md')


if __name__ == '__main__': main()
