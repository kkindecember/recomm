"""Export complete descriptive tables after checkpoint replay; no fitting or selection."""
import csv
import json
from pathlib import Path

from experiment.phase20.future_factor_screen import sha
from experiment.phase20.future_label_multiseed import write

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'artifacts/phase20/nearest_matched_head_20260918'
OUT = SOURCE / 'tables'
LABELS = {'frozen_B1': '原B1＋三CF（不新增小头）', 'linear_residual': '线性残差（7参数）',
          'ordinary_mlp32': '普通MLP（257参数）', 'factor_softmax': '原双头（258参数）'}


def csv_file(name, rows):
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    summary = json.loads((SOURCE / 'summary.json').read_text())
    audit = json.loads((SOURCE / 'verification.json').read_text())
    assert summary['state'] == 'COMPLETED' and audit['state'] == 'VERIFIED_CHECKPOINT_REPLAY'
    OUT.mkdir(exist_ok=False)
    per_seed, means, comparisons = [], [], []
    for domain, result in summary['domains'].items():
        for seed, run in result['seeds'].items():
            for arm, groups in run['metrics'].items():
                for group, values in groups.items():
                    per_seed.append(dict(domain=domain, seed=seed, arm=arm, group=group, **values))
            for contrast, groups in run['comparisons'].items():
                for group, values in groups.items():
                    comparisons.append(dict(domain=domain, seed=seed, contrast=contrast, group=group, **values))
        for arm, groups in result['aggregate_metrics'].items():
            for group, values in groups.items():
                means.append(dict(domain=domain, arm=arm, group=group,
                                  users=result['metadata']['group_users'][group],
                                  **{f'{k}_{stat}': v[stat] for k, v in values.items() for stat in ('mean', 'sd')}))
    csv_file('per_seed_metrics.csv', per_seed)
    csv_file('mean_and_sd_metrics.csv', means)
    csv_file('paired_comparisons.csv', comparisons)
    lines = ['# 当前目标下的小头对照：完整描述性结果', '',
             '三小头初始化分别计算指标再取均值；未集成三个小头，未重复训练上游。', '',
             '| 数据集 | 版本 | NDCG@10均值 | 小头初始化SD | Hit@10均值 |',
             '|---|---|---:|---:|---:|']
    for row in means:
        if row['group'] == 'all':
            lines.append(f"| {row['domain']} | {LABELS[row['arm']]} | {row['N10_mean']:.9f} | {row['N10_sd']:.9f} | {row['H10_mean']:.9f} |")
    lines += ['', '## 普通小头相对原双头的配对结果', '',
              '| 数据集 | 版本 | seed | ΔNDCG@10 | 相对变化 | Top10新增命中 | 丢失命中 | 净命中 |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in comparisons:
        if row['group'] == 'all' and row['contrast'].endswith('__vs__factor_softmax'):
            lines.append(f"| {row['domain']} | {LABELS[row['contrast'].split('__vs__')[0]]} | {row['seed']} | {row['delta_N10']:+.9f} | {100*row['relative_N10']:+.4f}% | {row['hit_gained']} | {row['hit_lost']} | {row['hit_net']:+d} |")
    lines += ['', '## 分组结果', '', '| 数据集 | 版本 | 分组 | 人数 | NDCG@10均值 | Hit@10均值 |',
              '|---|---|---|---:|---:|---:|']
    for row in means:
        if row['group'] != 'all':
            lines.append(f"| {row['domain']} | {LABELS[row['arm']]} | {row['group']} | {row['users']} | {row['N10_mean']:.9f} | {row['H10_mean']:.9f} |")
    lines += ['', '## 全体用户八指标', '',
              '| 数据集 | 版本 | N5 | H5 | N10 | H10 | N20 | H20 | N50 | H50 |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in means:
        if row['group'] == 'all':
            values = ' | '.join(f"{row[m+'_mean']:.9f}" for m in ('N5', 'H5', 'N10', 'H10', 'N20', 'H20', 'N50', 'H50'))
            lines.append(f"| {row['domain']} | {LABELS[row['arm']]} | {values} |")
    (OUT / 'results.md').write_text('\n'.join(lines) + '\n')
    write(OUT / 'manifest.json', dict(state='EXPORTED', training_updates=0, test_read=False,
          input_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                        (SOURCE / 'summary.json', SOURCE / 'verification.json', Path(__file__))},
          rows=dict(per_seed_metrics=len(per_seed), means=len(means), paired_comparisons=len(comparisons)),
          output_sha256={str(p.relative_to(ROOT)): sha(p) for p in sorted(OUT.iterdir())}))
    print(json.dumps(dict(per_seed_rows=len(per_seed), mean_rows=len(means), paired_rows=len(comparisons))))


if __name__ == '__main__':
    main()
