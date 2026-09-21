"""Plot the completed descriptive score-fusion audit and recount saved ranks."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiment.phase20.audit_b1_fusion_tradeoff import OUT, metrics


def main():
    summary = json.loads((OUT / 'summary.json').read_text())
    checks = 0
    for dataset, result in summary['domains'].items():
        with np.load(OUT / f'{dataset}.ranks.npz') as saved:
            rows = [result['current_B1_PCRF']] + result['fixed_scalar_curves']
            ranks = [np.asarray(r, dtype=np.int64) for r in [saved['current']] + list(saved['curve_ranks'])]
            assert len(rows) == len(ranks) == 169
            for row, r in zip(rows, ranks):
                found = metrics(r, saved['tail_mask'])
                for group in ('all', 'tail', 'non_tail'):
                    for key in ('N10', 'H10', 'H50'):
                        assert abs(found[group][key] - row[group][key]) < 1e-12
                        checks += 1
            # Independent formula: arithmetic mean over nonzero discounts.
            for row, r in zip(rows, ranks):
                for group, mask in [('all', np.ones(len(r), bool)), ('tail', saved['tail_mask'])]:
                    rr = r[mask]
                    n10 = sum(1 / np.log2(float(v) + 1) for v in rr if v <= 10) / len(rr)
                    assert abs(n10 - row[group]['N10']) < 1e-12
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    for ax, (dataset, result) in zip(axes, summary['domains'].items()):
        for family, color, label in [('plain', '#2874A6', 'Plain CF fusion'),
                                      ('pop_adjusted', '#D9822B', 'Frequency-corrected CF fusion')]:
            rows = [r for r in result['fixed_scalar_curves']
                    if r['family'] == family and isinstance(r['weight'], (float, int)) and r['weight'] <= 2]
            ax.plot([r['tail']['N10'] for r in rows], [r['all']['N10'] for r in rows],
                    color=color, lw=1.7, label=label)
        current = result['current_B1_PCRF']
        ax.scatter(current['tail']['N10'], current['all']['N10'], s=170, marker='*',
                   color='#17202A', zorder=5, label='Current B1 + PCRF')
        ax.set_title(f"{dataset}  |  {current['all']['users']:,} users", weight='bold')
        ax.set_xlabel(f"Low-frequency NDCG@10 (train count <= {result['q1']})")
        ax.set_ylabel('Overall NDCG@10')
        ax.grid(alpha=.2)
        ax.ticklabel_format(axis='both', style='plain', useOffset=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(.5, .035), frameon=False)
    fig.suptitle('Same B1 scores and 50 candidates: overall vs. low-frequency accuracy', fontsize=13)
    fig.text(.5, .006, 'Reused development validation; descriptive weight curves (0 to 2). Higher and farther right is better. No new training.',
             ha='center', fontsize=9, color='#555555')
    fig.tight_layout(rect=[0, .12, 1, .95])
    for suffix in ('png', 'svg'):
        fig.savefig(OUT / f'tradeoff.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    (OUT / 'verification.json').write_text(json.dumps(dict(state='VERIFIED', metric_values_recounted=checks,
        independent_ndcg_formula=True, new_training_steps=0, test_read=False), indent=2) + '\n')
    print(json.dumps({'metric_values_recounted': checks, 'plot': str(OUT / 'tradeoff.png')}))


if __name__ == '__main__':
    main()
