"""Isolated adapter reusing the frozen generation runner for two removal arms.

Bindings below are process-local. Files/configuration of the running text
experiment are never modified; its source hashes remain valid.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from experiment.phase21 import text_contribution as runner
from experiment.phase21.other_expert_contribution_core import ARMS, transform_scores, fuse
from experiment.phase20.catalog_scope_core import metrics, contrast

ROOT = runner.ROOT
OUT = ROOT / 'artifacts/phase21/other_expert_contribution_20260923_v1'
PLAN = ROOT / 'plan/第二十一阶段/21.3C_SASRec与ItemCF保留价值对照_20260923.md'
REPORT = ROOT / 'report/第二十一阶段/SASRec与ItemCF保留价值对照_20260923.md'
TEXT = ROOT / 'artifacts/phase21/text_contribution_20260923_v1'
GPUS = dict(without_sasrec='GPU-7952ad6e-2db7-fb2e-581b-34be56287640',
            without_itemcf='GPU-4cc17bbe-a85f-bdab-44ef-bbe29d1afc09')


def summarize():
    domains = {}; checks = {arm: {} for arm in ARMS}; decisions = {arm: {} for arm in ARMS}
    lines = ['# SASRec与ItemCF保留价值对照', '', '状态：COMPLETED；两组、两域真实生成及排名核验完成。', '',
        'Origin Skill：academic-research-suite。共用已核验的完整组合基线；上游冻结、每组融合器独立拟合4轮。', '',
        '| 域 | 组 | 真实生成 N10 | H10 | H50 | 后重排 N10 |', '|---|---|---:|---:|---:|---:|']
    for domain in runner.DOMAINS:
        previous = runner.base.arrays(runner.SOURCE / domain / 'ranks.npz')
        reference = dict(users=previous['development_users'], GRAM=previous['dev_GRAM'],
                         late=previous['dev_fixed_late'], guided=previous['dev_fixed_decode'])
        replay = TEXT / 'real' / domain / 'ranks.npz'
        replay_checked = False
        if replay.exists():
            actual = runner.base.arrays(replay)
            for name in reference: np.testing.assert_array_equal(reference[name], actual[name])
            replay_checked = True
        native = {k: metrics(reference[k]) for k in ('GRAM', 'late', 'guided')}
        g = native['guided']
        lines.append(f'| {domain} | 完整组合（共享） | {g["N10"]:.6f} | {g["H10"]:.6f} | {g["H50"]:.6f} | {native["late"]["N10"]:.6f} |')
        results = {}
        for arm in ARMS:
            folder = OUT / arm / domain
            result = runner.read(folder / 'summary.json'); rr = runner.base.arrays(folder / 'ranks.npz')
            np.testing.assert_array_equal(rr['users'], reference['users'])
            np.testing.assert_array_equal(rr['GRAM'], reference['GRAM'])
            comparisons = {scope: contrast(reference[scope], rr[scope], repeats=2000, seed=20260923)
                           for scope in ('guided', 'late')}
            m = result['metrics']['guided']; checks[arm][domain] = bool(g['N10'] > m['N10'] and g['H10'] >= m['H10'])
            improve_remove = m['N10'] > g['N10'] and m['H10'] >= g['H10']
            decisions[arm][domain] = ('RETENTION_DIRECTION' if checks[arm][domain] else
                                      'SIMPLIFICATION_DIRECTION' if improve_remove else 'INCONCLUSIVE_OR_TRADEOFF')
            results[arm] = dict(result=result, real_minus_removed=comparisons, interpretation=decisions[arm][domain])
            lines.append(f'| {domain} | {arm} | {m["N10"]:.6f} | {m["H10"]:.6f} | {m["H50"]:.6f} | {result["metrics"]["late"]["N10"]:.6f} |')
        domains[domain] = dict(reference_metrics=native, real_replay_crosscheck_completed=replay_checked, arms=results)
    lines += ['', '| 域 | 完整组减去删减组 | ΔN10 | 描述性95%区间 | 前十新增/丢失 |', '|---|---|---:|---|---|']
    for domain in runner.DOMAINS:
        for arm in ARMS:
            c = domains[domain]['arms'][arm]['real_minus_removed']['guided']; ci = c['descriptive_95pct_N10']
            lines.append(f'| {domain} | {arm} | {c["delta_N10"]:+.6f} | [{ci[0]:+.6f}, {ci[1]:+.6f}] | {c["top10_gained"]}/{c["top10_lost"]} |')
    lines += ['', '预设逐域方向判断：', '', '```json', json.dumps(decisions, ensure_ascii=False, indent=2), '```', '',
        'RETENTION_DIRECTION：该域支持保留方向；SIMPLIFICATION_DIRECTION：该域支持进一步验证删减；INCONCLUSIVE_OR_TRADEOFF：证据不足或指标取舍。它们不是显著性或最终方法判定。', '',
        '去SASRec只删除外部评分；GRAM内部既有协同近邻仍保留。两组均屏蔽移除专家的评分和四个统计特征，并重新学习其余融合权重。', '',
        '移除结果是相对其余组件的条件贡献；不能从两个单独删减实验推断可同时删掉两个专家。当前流程复用旧评分缓存，计时不是最终部署节省。', '',
        '一个上游seed，开发池多次使用；配对区间不包括训练随机性和研究选择。无新机制、无调参搜索、无test读取。', '',
        '[冻结协议](../../plan/第二十一阶段/21.3C_SASRec与ItemCF保留价值对照_20260923.md)；[完整结果](../../artifacts/phase21/other_expert_contribution_20260923_v1/summary.json)。']
    REPORT.write_text('\n'.join(lines) + '\n')
    summary = dict(state='COMPLETED', domains=domains, directional_support=checks, decisions=decisions,
                   dual_domain_support=all(all(v.values()) for v in checks.values()), test_read=False,
                   shared_reference='artifacts/phase20/lime_clean_benefit_20260922_v1')
    runner.write(OUT / 'summary.json', summary)
    return summary


def bind_runtime():
    runner.OUT = OUT; runner.PLAN = PLAN; runner.REPORT = REPORT
    runner.MODULE = 'experiment.phase21.other_expert_contribution'
    runner.ARMS = ARMS; runner.GPUS = GPUS
    runner.transform_scores = transform_scores; runner.fuse = fuse
    runner.summarize = summarize


def prepare():
    # Reuse exactly the same dataset/supervision/decoding preparation.
    original_sources = runner.read(TEXT / 'manifest.json')['sha256']
    runtime_files = [ROOT / 'experiment/phase21/text_contribution.py',
                     ROOT / 'experiment/phase21/text_contribution_core.py']
    for path in runtime_files:
        assert runner.sha(path) == original_sources[str(path.relative_to(ROOT))]
    runner.prepare()
    receipt = runner.read(OUT / 'manifest.json')
    for path in [Path(__file__), ROOT / 'experiment/phase21/other_expert_contribution_core.py',
                 ROOT / 'experiment/phase21/test_other_expert_contribution.py',
                 ROOT / 'experiment/phase21/run_other_expert_contribution.sh']:
        receipt['sha256'][str(path.relative_to(ROOT))] = runner.sha(path)
        dest = OUT / 'sources' / path.relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    receipt['runtime_adapter'] = dict(module='experiment.phase21.other_expert_contribution',
                                    process_local_only=True, active_text_files_unchanged=True)
    runner.write(OUT / 'manifest.json', receipt)
    for path in runtime_files:
        assert runner.sha(path) == original_sources[str(path.relative_to(ROOT))]
    REPORT.write_text('# SASRec与ItemCF保留价值对照\n\n状态：PREPARED。两组配置固定，尚无新的效果结论。\n')
    print(json.dumps(dict(state='PREPARED', runtime_adapter_frozen=True)), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['prepare', 'supervise', 'worker'])
    parser.add_argument('--arm', choices=ARMS); parser.add_argument('--domain', choices=runner.DOMAINS)
    args = parser.parse_args(); bind_runtime()
    if args.action == 'prepare': prepare()
    elif args.action == 'supervise':
        runner.supervise()
        # Correct the shared runner's generic failure-report title only for this output directory.
        if runner.read(OUT / 'status.json')['state'] == 'FAILED' and REPORT.exists():
            REPORT.write_text(REPORT.read_text().replace('文本专家贡献三组并行对照', 'SASRec与ItemCF保留价值对照'))
    else:
        assert args.arm and args.domain
        runner.worker(args.arm, args.domain)


if __name__ == '__main__': main()
