"""Audit the evaluation mask and gate sample counts from saved outputs only.

Both ranking policies reuse the identical frozen head outputs and beam50 lists.
This is an evaluation sensitivity analysis, not a full official reproduction.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from experiment.phase20.catalog_scope_core import contrast, metrics

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'artifacts/phase21/expert_signal_refinement_20261016_v1'
DESIGN = ROOT/'artifacts/phase20/lime_clean_training_design_20260920_v1'
OUT = ROOT/'artifacts/phase21/history_filter_audit_20261016_v1'
REPORT = ROOT/'report/第二十一阶段/历史过滤来源与融合器训练样本审计_20261016.md'
KEYS = ('native', 'old10_reference_guided', 'base7_guided', 'mass7_guided',
        'no_gap5_guided', 'mass_no_gap5_guided')


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def rank(items, target):
    return next((j+1 for j, (item, _) in enumerate(items) if item == target), 2**30)


def main():
    OUT.mkdir(exist_ok=False)
    write(OUT/'status.json', dict(state='RUNNING', test_read=False, new_training=False, new_generation=False))
    assert read(RUN/'status.json')['state'] == 'COMPLETED'
    design = read(DESIGN/'summary.json')['domains']
    hashes = {}; results = {}
    for domain in ('Toys', 'Beauty'):
        p = RUN/domain
        expected = read(p/'output_manifest.json')['sha256']
        for name in ('samples.json', 'fit_samples.json', 'training.json', 'predictions.jsonl', 'ranks.npz'):
            rel = str((p/name).relative_to(ROOT)); hashes[rel] = sha(p/name)
            if rel in expected:
                assert hashes[rel] == expected[rel], rel
        fit, samples = read(p/'fit_samples.json'), read(p/'samples.json')
        assert len(fit) == len({s['user_id'] for s in fit})
        assert {s['user_id'] for s in fit}.isdisjoint(s['user_id'] for s in samples)
        rows = [json.loads(line) for line in (p/'predictions.jsonl').read_text().splitlines()]
        assert len(rows) == len(samples) == 512
        filtered, unfiltered, inspection = {}, {}, {}
        saved = np.load(p/'ranks.npz')
        for key in KEYS:
            f, u = [], []; history_in_top10 = history_in_beam = changed = 0
            for row, sample in zip(rows, samples):
                assert all(row[k] == sample[k] for k in ('user_id', 'target', 'history'))
                value = row[key]
                assert len(value['ids']) == len(value['scores']) == len(set(value['ids'])) == 50
                pairs = sorted(zip(value['ids'], value['scores']), key=lambda x:(-x[1], x[0]))
                history = set(sample['history'])
                u.append(rank(pairs, sample['target']))
                f.append(rank([(item, score) for item, score in pairs if item not in history], sample['target']))
                history_in_top10 += sum(item in history for item, _ in pairs[:10])
                history_in_beam += sum(item in history for item, _ in pairs)
                changed += int(f[-1] != u[-1])
            filtered[key], unfiltered[key] = np.asarray(f), np.asarray(u)
            np.testing.assert_array_equal(filtered[key], saved[key])
            inspection[key] = dict(target_ranks_changed=changed,
                historical_items_in_top10_total=history_in_top10, historical_items_in_beam_total=history_in_beam,
                filtered_metrics=metrics(filtered[key]), unfiltered_metrics=metrics(unfiltered[key]))
        training = read(p/'training.json')['fits']
        count = design[domain]
        calibration = count['gate_calibration_users']
        reserved = [json.loads(line) for line in (DESIGN/domain/'terminal_holdout.jsonl').read_text().splitlines()]
        extra = [r for r in reserved if r['role'] == 'gate_calibration']
        assert len(extra) == calibration
        assert {r['user_id'] for r in extra}.isdisjoint(s['user_id'] for s in samples)
        assert {r['user_id'] for r in extra}.isdisjoint(s['user_id'] for s in fit)
        assert {r['user_id'] for r in reserved} == {r['user_id'] for r in extra+fit}
        comparisons = {key+'_vs_native':contrast(unfiltered[key], unfiltered['native'], repeats=2000, seed=20261016)
            for key in KEYS if key != 'native'}
        np.savez_compressed(OUT/(domain+'_ranks.npz'), users=[s['user_id'] for s in samples],
            **{key+'_filtered':value for key, value in filtered.items()},
            **{key+'_unfiltered':value for key, value in unfiltered.items()})
        results[domain] = dict(users=512, arms=inspection, unfiltered_comparisons=comparisons,
            gate_training=dict(reserved_terminal_events=count['heldout_events'],
                current_fit_rows=len(fit), current_fit_users=len({s['user_id'] for s in fit}),
                internal_selection_fit=training['base7']['selection_fit_users'],
                internal_selection_heldout=training['base7']['heldout_users'],
                final_refit=len(fit), external_development=len(samples),
                extra_previous_calibration=calibration, potential_fit_if_reassigned=len(fit)+calibration,
                potential_increase_pct=100*calibration/len(fit),
                upstream_train_prefixes_before_reserve=count['original_train_prefixes'],
                upstream_train_prefixes_after_reserve=count['clean_train_prefixes'],
                target_in_history_fit=sum(s['target'] in s['full_history'] for s in fit),
                target_in_history_dev=sum(s['target'] in s['full_history'] for s in samples),
                extra_fit_and_dev_user_disjoint=True))
        print(domain, json.dumps(results[domain]['gate_training']), flush=True)
    for path in (Path(__file__), DESIGN/'summary.json',
                 ROOT/'experiment/phase20/prepare_lime_clean_training.py',
                 ROOT/'experiment/phase20/lime_decode_screen.py',
                 ROOT/'experiment/phase20/lime_decode_screen_core.py',
                 ROOT/'experiment/phase21/dual_candidate_generation.py',
                 ROOT/'experiment/phase21/expert_signal_refinement.py'):
        hashes[str(path.relative_to(ROOT))] = sha(path)
    for domain in ('Toys', 'Beauty'):
        path = DESIGN/domain/'terminal_holdout.jsonl'; hashes[str(path.relative_to(ROOT))] = sha(path)
    write(OUT/'summary.json', dict(domains=results, test_read=False, new_training=False, new_generation=False,
        scope='Saved beam50 rescoring changes history filtering only; ID tie-break unchanged. Not full official reproduction.',
        origin=dict(official_runner='https://github.com/skleee/GRAM/blob/main/src/runner/single_runner_gram.py',
            official_metric='https://github.com/skleee/GRAM/blob/main/src/utils/evaluate.py',
            official_path='Global catalog trie -> decode -> rel_results(predictions, targets, scores, k); no recent20 mask in inspected official path.',
            local_path='phase20 lime_decode_screen.py/core.py introduced/retained the historical mask, now inherited by phase21.',
            local_archive_commit='f5dceb5 (2026-09-21); archive proves presence, not human decision author.',
            limit='Cannot identify who originally chose the rule from code alone; do not attribute it to official GRAM.')))
    write(OUT/'input_hashes.json', hashes)
    lines = ['# 历史过滤来源与融合器训练样本审计', '', '## Material Passport', '',
        '- Origin Skill: academic-research-suite / experiment-agent', '- Origin Mode: validate',
        '- Origin Date: 2026-10-16',
        '- Verification Status: ANALYZED；已核对保存候选与排名，没有重跑模型',
        '- Version Label: history_filter_audit_v1', '',
        '官方GRAM所核对评测路径没有按最近20历史排除候选：全目录树生成后直接调用rel_results比较预测与目标。',
        '来源：[官方生成/评测路径](https://github.com/skleee/GRAM/blob/main/src/runner/single_runner_gram.py)、[官方指标函数](https://github.com/skleee/GRAM/blob/main/src/utils/evaluate.py)。',
        '本地排除最近20历史发生在项目后续评测适配层，至少在2026-09-21的phase20归档f5dceb5中已经存在；提交归档不等于谁人工决定了这条规则，不能据此指认个人。',
        '之前所有方法与GRAM对照同用过滤，内部比较没有只给新方法过滤；但过滤后的GRAM不能直接作为官方评测口径。', '',
        '## 相同保存候选，不过滤历史的NDCG@10', '',
        '| 方法 | Toys 过滤后 | Toys 不过滤 | Beauty 过滤后 | Beauty 不过滤 |',
        '|---|---:|---:|---:|---:|']
    for label, key in zip(('GRAM', '原10维', '原7维', '只替换Top5的7维', '只删分差5维', '组合5维'), KEYS):
        cells = [label]
        for domain in ('Toys', 'Beauty'):
            values = results[domain]['arms'][key]
            cells += [f'{values[mode+"_metrics"]["N10"]:.6f}' for mode in ('filtered', 'unfiltered')]
        lines.append('| '+' | '.join(cells)+' |')
    lines += ['', '这只是既有beam50的过滤敏感性复算：训练样本、冻结参数、候选、分数、ID打破平局规则都不变；没有复现官方完整数据划分、训练或论文数字。',
        '不做历史过滤时，mass7两域N10点估计高于GRAM；组合5维的Beauty略低于GRAM。因此之前优先选择5维的建议仅适用于原过滤协议，不能直接迁移到官方不屏蔽设定。',
        '新质量特征此时仍是历史集中度的可观察统计，但不能再解释为落到“不能输出商品”的概率。也没有证据保证其单调代表专家错误率或可靠性。', '',
        '## 实际权重训练样本', '',
        '| 数据集 | 原预留末端事件 | 当前融合器拟合样本 | 内部训练/选轮 | 最终重拟合 | 外部开发 | 早期校准预留 |',
        '|---|---:|---:|---|---:|---:|---:|']
    for domain, result in results.items():
        c = result['gate_training']
        lines.append(f'| {domain} | {c["reserved_terminal_events"]} | {c["current_fit_rows"]} | '
            f'{c["internal_selection_fit"]}/{c["internal_selection_heldout"]} | {c["final_refit"]} | '
            f'{c["external_development"]} | {c["extra_previous_calibration"]} |')
    lines += ['', '每条当前拟合样本是一名用户的一段可见历史和一个留出的下一商品标签；每域一个网络，并非把两域合成同一个训练集。CE父阶段与GRAM路径NLL阶段反复使用这些标签，不等于有两倍独立样本。',
        '预留标签来自原训练序列末端，已在上游GRAM/行为专家训练前移除；不是把本次外部512开发用户用于梯度训练。当前外部开发池被反复用于方法探索，仍不能充当最终独立确认。',
        '当前291/355参数的小融合器已能学习；单凭样本/参数比不能确定充分性。是否值得增加应通过固定留出下的学习曲线判断；更多样本不保证NDCG必涨，也不同于多跑epoch。',
        '已有721/775早期校准事件与当前fit和dev用户不交叠，也在原上游预留集合中。若不再承担旧校准角色，可优先审计并改分配给新融合器，总量可到3494/4085，分别增加约26.0%/23.4%；需新提取对应专家及路径缓存并按匹配规则训练。',
        '继续从已用于上游训练的标签中划出事件会改变“上游未见融合训练答案”的当前条件；应重新构造上游留出或使用折外预测，不能把扩大训练样本简单当成无影响操作。',
        '建议次序：先固定评测是否排除历史，再做低成本样本量曲线，然后决定是否重新分配早期校准池。未修改已完成实验、主线配置或开启训练。', '',
        '## 统计解释检查：11/11', '',
        'Simpson：两域分别报告；Ecological：平均收益不代表所有用户；Berkson：固定512开发池限制外推；Collider：不按命中选择新训练用户；Base rate：全512保留；Regression to mean：相同固定候选配对；Survivorship：全部六组均保留；Look elsewhere：两规则敏感性与多方法比较为描述性，未校正；Forking paths：发现协议差异后的分析为事后探索，不为保留最好结果选规则；Causation：仅改变评测过滤，不代表重新训练后效果或错误机制；Reverse causality：过滤只用可见历史，不接触未来目标。']
    REPORT.write_text('\n'.join(lines)+'\n')
    write(OUT/'status.json', dict(state='COMPLETED', test_read=False, new_training=False, new_generation=False,
        report=str(REPORT.relative_to(ROOT))))
    print('COMPLETED', str(REPORT), flush=True)


if __name__ == '__main__':
    main()
