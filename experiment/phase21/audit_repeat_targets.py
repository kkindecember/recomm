"""Count train/validation repeat targets and explain saved ranking-mask gains.

Reads no test split. Does not modify model parameters or candidate generation.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from experiment.phase20.personal_popularity_core import reconstruct_training

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'artifacts/phase21/expert_signal_refinement_20261016_v1'
OUT = ROOT/'artifacts/phase21/repeat_target_audit_20261016_v1'
REPORT = ROOT/'report/第二十一阶段/重复目标与历史过滤指标变化解释_20261016.md'
DATA = dict(Toys='artifacts/phase18/diff_gram/prepared_v2_categories',
            Beauty='artifacts/phase18/diff_gram/beauty/prepared_v1')


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def utility(rank):
    return 1/np.log2(rank+1) if rank <= 10 else 0.


def main():
    OUT.mkdir(exist_ok=False)
    write(OUT/'status.json', dict(state='RUNNING', test_read=False))
    domains, paths = {}, [Path(__file__)]
    for domain, relative in DATA.items():
        source = ROOT/relative
        paths += [source/'train.jsonl', source/'validation.jsonl', RUN/domain/'predictions.jsonl']
        train, valid = records(source/'train.jsonl'), records(source/'validation.jsonl')
        users, sequences, _, _ = reconstruct_training(train)
        duplicate_events = sum(len(seq)-len(set(seq)) for seq in sequences.values())
        counts = dict(train_users=len(users), train_transition_rows=len(train),
            train_interactions=sum(map(len, sequences.values())),
            train_users_with_repeated_items=sum(len(seq) != len(set(seq)) for seq in sequences.values()),
            train_repeated_events_full_history=duplicate_events,
            train_targets_in_recent20=sum(row['target'] in row['history'] for row in train),
            validation_users=len(valid), validation_targets_in_full_history=0,
            validation_targets_in_recent20=0)
        for row in valid:
            assert row['history'] == sequences[row['user_id']][-20:]
            counts['validation_targets_in_full_history'] += int(row['target'] in sequences[row['user_id']])
            counts['validation_targets_in_recent20'] += int(row['target'] in row['history'])
        predictions = records(RUN/domain/'predictions.jsonl')
        assert len(predictions) == 512
        details = {}
        for key in ('native', 'base7_guided', 'mass7_guided', 'mass_no_gap5_guided'):
            before, after = [], []
            improved = entered = target_in_beam = historical10 = historical50 = 0
            gain_entered = gain_retained = 0.
            example = None
            for row in predictions:
                history, target = set(row['history']), row['target']
                assert target not in history
                value = row[key]
                ordered = sorted(zip(value['ids'], value['scores']), key=lambda x:(-x[1], x[0]))
                ids = [item for item, _ in ordered]
                historical10 += sum(item in history for item in ids[:10])
                historical50 += sum(item in history for item in ids)
                if target not in ids:
                    before.append(2**30); after.append(2**30)
                    continue
                target_in_beam += 1
                rank = ids.index(target)+1
                excluded_above = sum(item in history for item in ids[:rank-1])
                filtered_rank = [item for item in ids if item not in history].index(target)+1
                assert filtered_rank == rank-excluded_above
                before.append(rank); after.append(filtered_rank)
                improved += int(excluded_above > 0)
                delta = (utility(filtered_rank)-utility(rank))/512
                if rank > 10 and filtered_rank <= 10:
                    entered += 1; gain_entered += delta
                elif rank <= 10:
                    gain_retained += delta
                if excluded_above and rank <= 10 and example is None:
                    example = dict(unfiltered_rank=rank, filtered_rank=filtered_rank,
                                   historical_items_above_target=excluded_above)
            original = float(np.mean([utility(r) for r in before]))
            filtered = float(np.mean([utility(r) for r in after]))
            assert abs(filtered-original-gain_entered-gain_retained) < 1e-12
            details[key] = dict(unfiltered_N10=original, filtered_N10=filtered,
                relative_drop_without_filter_pct=100*(original/filtered-1),
                target_present_in_beam=target_in_beam, target_rank_improved_by_filter=improved,
                new_top10_hits_from_filter=entered, existing_top10_reranking_N10_gain=gain_retained,
                new_top10_N10_gain=gain_entered, historical_items_in_top10_total=historical10,
                mean_historical_items_in_top10=historical10/512,
                historical_items_in_beam_total=historical50, rank_example=example)
        domains[domain] = dict(corpus_repeat_counts=counts, saved512_ranking_explanation=details)
        print(domain, json.dumps(counts), flush=True)
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    write(OUT/'input_hashes.json', hashes)
    write(OUT/'summary.json', dict(domains=domains, test_read=False, new_training=False, new_generation=False,
        scope='All processed train/validation records; not raw purchase transactions or held-out test.',
        inference_limit='Zero observed repeated targets does not establish why repeats are absent or rule out real repurchases.'))
    lines = ['# 重复目标与历史过滤指标变化解释', '', '## Material Passport', '',
        '- Origin Skill: academic-research-suite / experiment-agent', '- Origin Mode: validate',
        '- Origin Date: 2026-10-16', '- Verification Status: ANALYZED；确定性计数与保存排名已核对',
        '- Version Label: repeat_target_audit_v1', '',
        '范围：当前处理版全量训练和验证。测试集未读取；不能将结果推广为原始日志或现实购买行为无复购。', '',
        '| 数据集 | 训练用户数 | 训练下一商品样本 | 训练完整序列重复事件 | 验证下一商品样本 | 验证目标出现在完整历史 |',
        '|---|---:|---:|---:|---:|---:|']
    for domain, result in domains.items():
        c = result['corpus_repeat_counts']
        lines.append(f'| {domain} | {c["train_users"]} | {c["train_transition_rows"]} | '
            f'{c["train_repeated_events_full_history"]} | {c["validation_users"]} | {c["validation_targets_in_full_history"]} |')
    lines += ['', '这些计数证明所查处理版没有重复目标；仅凭计数不能确认是原数据天然无重复、评论记录方式或上游去重造成。',
        '当前512用户的所有目标也都不在最近20历史中。固定候选和分数后，过滤不会删掉正确目标，只能保持或改善其排名。',
        '恒等式：过滤后目标排名 = 原排名 − 排在目标之前且被过滤的历史商品数。目标不在beam50时，过滤不能让它凭空进入候选。', '',
        '| 数据集/方案 | 不过滤N10 | 过滤后N10 | 因过滤改善排名人数 | 前10历史商品数/用户 |',
        '|---|---:|---:|---:|---:|']
    for domain, result in domains.items():
        for key, value in result['saved512_ranking_explanation'].items():
            lines.append(f'| {domain}/{key} | {value["unfiltered_N10"]:.6f} | {value["filtered_N10"]:.6f} | '
                f'{value["target_rank_improved_by_filter"]}/512 | {value["mean_historical_items_in_top10"]:.3f} |')
    lines += ['', '以[A历史商品,B历史商品,C正确目标]为例，过滤将C从第3提升到第1，NDCG@10从0.5提升到1；模型没有重新训练，预测分数没有改变。',
        '在该512用户中，专家引导方案的原始前10列表包含更多历史商品；过滤对它们的提升更大。因此过滤协议下的相对GRAM收益不能直接解释为无过滤时相同的生成排序改善。',
        '不重复推荐是可以定义的任务/系统后处理，但非技术必需，也不是官方GRAM当前所查评测流程的默认操作。主对照应对齐官方不做该屏蔽的口径；统一历史过滤可另作明确的附加设置，所有基线规则一致。',
        '若任务允许重复交互、存在历史目标，直接屏蔽会删掉真实答案。不能因为当前处理版无重复就声称现实复购不存在。',
        '此报告只解释排名恒等关系与处理版计数；没有新增训练、生成、方法选择或显著性主张。']
    REPORT.write_text('\n'.join(lines)+'\n')
    write(OUT/'status.json', dict(state='COMPLETED', test_read=False,
        new_training=False, new_generation=False, report=str(REPORT.relative_to(ROOT))))
    print('COMPLETED', str(REPORT), flush=True)


if __name__ == '__main__':
    main()
