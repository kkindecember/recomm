"""Independent CPU audit of the saved Top5/mass x retain/drop gap screen.

Recompute features, gate outputs, ranks and descriptive intervals. No fitting,
generation, test access or changes to the completed experiment's artifacts.
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from experiment.phase21 import review_decision_features as V
from experiment.phase21.candidate_gate_core import fold

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'artifacts/phase21/expert_signal_refinement_20261016_v1'
BASE = ROOT/'artifacts/phase21/task_context_features_20261016_v1'
TIME = ROOT/'artifacts/phase21/time_behavior_expert_20261015_v1'
OUT = ROOT/'artifacts/phase21/expert_signal_refinement_review_20261016_v1'
REPORT = ROOT/'report/第二十一阶段/精简七维历史质量与头部间隔结果复核_20261016.md'
ARMS = ('base7', 'mass7', 'no_gap5', 'mass_no_gap5')
KEEP = [0, 1, 2, 3, 4, 7, 8]
WITHOUT_GAP = [0, 1, 2, 5, 6]
read, write = V.read, V.write


def norm(values):
    x = values.astype(np.float64)
    mean = x.mean(0)
    std = np.sqrt(np.maximum((x*x).mean(0)-mean*mean, 0))
    std[std < 1e-5] = 1.
    return mean.astype(np.float32), std.astype(np.float32)


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    write(OUT/'status.json', dict(state='RUNNING', stage='saved_outputs_audit', test_read=False))
    checked = {}

    def verify(path):
        for rel, expected in read(path)['sha256'].items():
            if rel not in checked:
                h = hashlib.sha256()
                with (ROOT/rel).open('rb') as stream:
                    for block in iter(lambda: stream.read(8*1024*1024), b''):
                        h.update(block)
                checked[rel] = h.hexdigest()
            assert checked[rel] == expected, rel

    status = read(RUN/'status.json')
    assert status['state'] == 'COMPLETED'
    assert set(status['finished']) == {'Toys', 'Beauty'}
    assert all(v['exit_code'] == 0 and v['lease_released'] for v in status['finished'].values())
    verify(RUN/'manifest.json')
    print('Input/source hashes verified', flush=True)
    domains = {}
    for domain in ('Toys', 'Beauty'):
        p = RUN/domain
        for name in ('feature_manifest', 'training_manifest', 'output_manifest'):
            verify(p/(name+'.json'))
        assert read(p/'lease_status.json')['state'] == 'released'
        runtime = read(p/'runtime_checks.json')
        assert runtime['gram_model_digest_unchanged'] and runtime['base7_guided_order_exact_users'] == 512
        samples, fit = read(p/'samples.json'), read(p/'fit_samples.json')
        assert {s['user_id'] for s in fit}.isdisjoint(s['user_id'] for s in samples)
        rows = [json.loads(line) for line in (p/'predictions.jsonl').read_text().splitlines()]
        assert len(rows) == len(samples) == len({s['user_id'] for s in samples}) == 512
        summary, archive = read(p/'summary.json'), np.load(p/'ranks.npz')
        np.testing.assert_array_equal(archive['users'], [s['user_id'] for s in samples])
        ranks = {k: [] for k in summary['metrics']}
        for row, sample in zip(rows, samples):
            assert all(row[k] == sample[k] for k in ('user_id', 'target', 'history'))
            for key in ranks:
                value = row[key]
                assert len(value['ids']) == len(value['scores']) == len(set(value['ids'])) == 50
                assert np.isfinite(value['scores']).all()
                ordered = sorted(((i, s) for i, s in zip(value['ids'], value['scores'])
                                  if i not in sample['history']), key=lambda x: (-x[1], x[0]))
                ids = [i for i, _ in ordered]
                ranks[key].append(ids.index(sample['target'])+1 if sample['target'] in ids else 2**30)
        ranks = {k: np.asarray(v) for k, v in ranks.items()}
        for key, values in ranks.items():
            np.testing.assert_array_equal(values, archive[key])
            for metric, value in V.metrics(values).items():
                assert abs(value-summary['metrics'][key][metric]) < 1e-12
        prior = np.load(BASE/domain/'ranks.npz')
        for key, old in (('native', 'native'), ('old10_reference_guided', 'old10_guided'),
                         ('base7_guided', 'pruned7_guided'), ('base7_late', 'pruned7_late')):
            np.testing.assert_array_equal(ranks[key], prior[old])
        for suffix in ('guided', 'late'):
            np.testing.assert_array_equal(ranks['historical_'+suffix], ranks['base7_'+suffix])
        indices = np.random.RandomState(20261016).randint(0, 512, size=(2000, 512))
        for key, comparison in summary['comparisons'].items():
            first, second = key.split('_vs_'); a, b = ranks[first], ranks[second]
            for label, delta in (('N10', V.utility(a)-V.utility(b)),
                                 ('H10', (a <= 10).astype(float)-(b <= 10).astype(float))):
                assert abs(delta.mean()-comparison['delta_'+label]) < 1e-12
                np.testing.assert_allclose(np.percentile(delta[indices].mean(1), [2.5, 97.5]),
                    comparison['descriptive_95pct_'+label], atol=1e-12, rtol=0)
            assert comparison['top10_gained'] == int(((a <= 10) & (b > 10)).sum())
            assert comparison['top10_lost'] == int(((a > 10) & (b <= 10)).sum())

        feature_checks = {}
        for part, records in (('train', fit), ('validation', samples)):
            source = TIME/domain/'features/tisas'/part
            old = np.load(source/'user.npy')
            item = np.load(source/'item.npy', mmap_mode='r')
            expected7, masses = [], []
            for j, sample in enumerate(records):
                e = np.ascontiguousarray(item[j, :, :2].T)
                assert sample['history'] == sample['full_history'][-20:]
                history_mask = np.isin(np.arange(1, e.shape[1]+1), sample['full_history'])
                order = [np.argsort(-x, kind='stable') for x in e]
                top = np.asarray([e[k, ix[:2]] for k, ix in enumerate(order)])
                expected7.append(np.r_[np.log1p(len(sample['full_history']))/5,
                    top[:, 0]-e.mean(1), top[:, 0]-top[:, 1],
                    [history_mask[ix[:5]].mean() for ix in order]])
                # Direct dense softmax, independently of the production
                # extractor's shifted exponentials. Boolean mask deduplicates.
                prob = np.exp(20.*e.astype(np.float64)); prob /= prob.sum(1, keepdims=True)
                mask = np.isin(np.arange(1, e.shape[1]+1), sample['history'])
                masses.append(prob[:, mask].sum(1))
            expected7, masses = np.asarray(expected7, np.float32), np.asarray(masses, np.float32)
            folder = p/'features'/part
            assert read(folder/'metadata.json')['users_order'] == [s['user_id'] for s in records]
            assert not read(folder/'metadata.json')['target_used_in_features']
            assert not read(folder/'metadata.json')['gram_state_used_in_features']
            seven, mass7 = [np.load(folder/(a+'.npy')) for a in ('base7', 'mass7')]
            np.testing.assert_array_equal(seven, old[:, KEEP])
            np.testing.assert_allclose(seven, expected7, atol=2e-6, rtol=1e-5)
            np.testing.assert_array_equal(mass7[:, :5], seven[:, :5])
            np.testing.assert_allclose(mass7[:, 5:], masses, atol=3e-8, rtol=1e-6)
            for arm, full in (('no_gap5', seven), ('mass_no_gap5', mass7)):
                np.testing.assert_array_equal(np.load(folder/(arm+'.npy')), full[:, WITHOUT_GAP])
            feature_checks[part] = dict(users=len(records), all_seven_formulas_recomputed=True,
                both_mass_formulas_recomputed=True, actual_recent20_verified=True,
                all_deleted_and_unchanged_columns_exact=True,
                max_old_error=float(np.abs(expected7-seven).max()),
                max_mass_error=float(np.abs(masses-mass7[:, 5:]).max()),
                behavior_filtered_mass=V.describe(masses[:, 0]), text_filtered_mass=V.describe(masses[:, 1]))

        heads = {}
        signals, alphas, betas = [np.load(p/name) for name in ('signals.npz', 'alphas.npz', 'betas.npz')]
        item = np.load(TIME/domain/'features/tisas/validation/item.npy', mmap_mode='r')[:, :, :2]
        fi = np.asarray([j for j, s in enumerate(fit) if fold(s['user_id'], domain) != 0])
        for arm in ARMS:
            ck = torch.load(p/(arm+'.pt'), map_location='cpu')
            assert ck['fit_users'] == [s['user_id'] for s in fit]
            train = np.load(p/'features/train'/(arm+'.npy'))
            for actual, expected in zip((ck['mean'], ck['std']), norm(train)):
                np.testing.assert_array_equal(actual, expected)
            for name, ix in (('full', np.arange(len(fit))), ('fold', fi)):
                parent = torch.load(p/'parents'/arm/(name+'.pt'), map_location='cpu')
                assert parent['fit_users'] == [fit[j]['user_id'] for j in ix]
                for actual, expected in zip((parent['mean'], parent['std']), norm(train[ix])):
                    np.testing.assert_array_equal(actual, expected)
                assert parent['selected_epochs'] == min(parent['selection_curve'], key=lambda x:(x['loss'], x['epoch']))['epoch']
            assert ck['selected_epochs'] == min(ck['selection_curve'], key=lambda x:(x['loss'], x['epoch']))['epoch']
            sd = {k: v.numpy() for k, v in ck['state_dict'].items()}
            user = np.load(p/'features/validation'/(arm+'.npy'))
            x = np.clip((user-ck['mean'])/ck['std'], -10, 10)
            hidden = np.maximum(x@sd['input.weight'].T+sd['input.bias'], 0)
            logits = hidden@sd['beta.weight'].T+sd['beta.bias']
            beta = np.exp(logits-logits.max(1, keepdims=True)); beta /= beta.sum(1, keepdims=True)
            alpha = np.clip(1/(1+np.exp(-(hidden@sd['strength.weight'].T+sd['strength.bias']).ravel())), 1e-6, 1-1e-6)
            np.testing.assert_allclose(beta, betas[arm], atol=2e-6, rtol=2e-5)
            np.testing.assert_allclose(alpha, alphas[arm], atol=2e-6, rtol=2e-5)
            np.testing.assert_allclose((beta[:, None, :]*item).sum(-1), signals[arm], atol=3e-6, rtol=2e-5)
            count = sum(v.size for v in sd.values()); assert count == (355 if user.shape[1] == 7 else 291)
            if arm == 'base7':
                for name, values in (('signals.npz', signals[arm]), ('alphas.npz', alphas[arm]), ('betas.npz', betas[arm])):
                    np.testing.assert_array_equal(values, np.load(BASE/domain/name)['pruned7'])
            heads[arm] = dict(parameters=count, all512_outputs_verified=True,
                fit_only_normalization_and_parent_scope_verified=True,
                alpha=V.describe(alphas[arm]), behavior_weight=V.describe(betas[arm][:, 0]))

        decompositions = {}
        for key in summary['comparisons']:
            first, second = key.split('_vs_'); a, b = ranks[first], ranks[second]
            delta = V.utility(a)-V.utility(b)
            masks = dict(gained=(a <= 10) & (b > 10), lost=(a > 10) & (b <= 10),
                         retained=(a <= 10) & (b <= 10), neither=(a > 10) & (b > 10))
            decompositions[key] = dict(groups={k:dict(users=int(v.sum()),
                delta_N10_contribution=float(delta[v].sum()/512)) for k, v in masks.items()},
                retained_rank_changes={k:int(v.sum()) for k, v in dict(
                    improved=masks['retained'] & (a < b), worsened=masks['retained'] & (a > b),
                    unchanged=masks['retained'] & (a == b)).items()})
        interaction = (V.utility(ranks['mass_no_gap5_guided'])-V.utility(ranks['no_gap5_guided'])
                       -V.utility(ranks['mass7_guided'])+V.utility(ranks['base7_guided']))
        summary.update(independent_feature_checks=feature_checks, independent_heads=heads,
            top10_change_decompositions=decompositions,
            factorial_interaction_N10=dict(delta=float(interaction.mean()),
                descriptive_95pct=np.percentile(interaction[indices].mean(1), [2.5, 97.5]).tolist(),
                interpretation='Replacement effect after deleting gaps minus replacement effect retaining gaps; exploratory'),
            relative_N10_pct={a:{b:100*(summary['metrics'][a+'_guided']['N10']/summary['metrics'][b]['N10']-1)
                for b in ('native', 'base7_guided', 'old10_reference_guided')} for a in ARMS})
        domains[domain] = summary
        print(domain, 'VERIFIED: 12 rank arrays, all feature rows, four heads and 17 paired comparisons', flush=True)

    review = dict(state='VERIFIED_SAVED_OUTPUTS', hashes_checked=len(checked), domains=domains,
        elapsed_experiment_seconds=status['elapsed_seconds'], test_read=False, new_training=False, new_generation=False,
        statistical_review=dict(coverage='11/11', confidence='CAUTION: exploratory',
            simpson='Report both domains separately. Mass-only and gap-deletion effects vary by domain/configuration.',
            ecological='Aggregate gains and mean weights do not imply benefit for every user.',
            berkson='Repeated fixed development512 pools; no population extrapolation.',
            collider='Hit-conditioned decomposition is descriptive only, not causal subgroup evidence.',
            base_rate='All512 denominators and gained/lost/retained users retained.',
            regression_to_mean='Base7 exactly reproduced; no extreme-user subgroup selected.',
            survivorship='All four prescribed arms and both complete512 cohorts included.',
            look_elsewhere='17 paired comparisons/domain with N10/H10 intervals plus one interaction N10 interval/domain; all uncorrected descriptive.',
            forking_paths='Repeated development searches and post-result method preference disclosed; not confirmatory.',
            causation='Mass replacement changes weighting and history scope together. Gap deletion changes dimension; optimization and selected epochs also change.',
            reverse_causality='Only visible historical items and frozen expert scores enter gate; target labels used for fitting/evaluation, not feature extraction.'))
    write(OUT/'summary.json', review); write(OUT/'verified_hashes.json', checked)
    lines = ['# 精简七维历史质量与头部间隔结果复核', '', '## Material Passport', '',
        '- Origin Skill: academic-research-suite / experiment-agent', '- Origin Mode: validate',
        '- Origin Date: 2026-10-16',
        '- Verification Status: VERIFIED_SAVED_OUTPUTS；统计解释 ANALYZED，未重跑训练或生成',
        '- Version Label: expert_signal_refinement_review_v1', '',
        f'两域四组全部完成，耗时{status["elapsed_seconds"]/60:.1f}分钟，GPU租约释放。独立核对{len(checked)}个文件哈希、24份排名、全部特征公式、8个融合头前向、34组配对比较及标准化/CE父模型拟合范围。原7维与历史精简7维逐元素复现；没有test访问。', '',
        '| 方法 | Toys N10 | 对原7维 | Beauty N10 | 对原7维 |',
        '|---|---:|---:|---:|---:|']
    methods = [('GRAM', 'native'), ('原10维只读参照', 'old10_reference_guided')]+[
        (label, arm+'_guided') for label, arm in zip(
            ('原7维', '只换历史概率质量：7维', '只删两项分差：5维', '两处同时改：5维'), ARMS)]
    for label, key in methods:
        cells = [label]
        for domain in ('Toys', 'Beauty'):
            m = domains[domain]['metrics']; value = m[key]['N10']
            cells += [f'{value:.6f}', f'{100*(value/m["base7_guided"]["N10"]-1):+.3f}%']
        lines.append('| '+' | '.join(cells)+' |')
    lines += ['', '两处同时改的5维是值得优先讨论的简化候选：本轮N10双域略高于原7维，也都高于GRAM。原7维仍为用户此前选择的主线，未擅自修改模型或启动后续实验。',
        '这不是严格支配：该5维相对原7维的Beauty N20略降；相对只替换的7维，Toys N10下降而Beauty上升。两域均高于原7维的N10点估计，不等于每个指标或每个用户都更好。',
        '只删除分差两域N10都下降，不能据此说分差没有价值。替换历史指标后再删分差，Toys下降而Beauty上升；当前证据不支持一个跨配置一致的分差结论。', '',
        '| 数据集 | 组合5维对GRAM | 对原10维 | 对原7维N10差值95%描述区间 | Top10命中：原7/改7/删5/组合5 | Top50命中：原7/改7/删5/组合5 |',
        '|---|---:|---:|---|---|---|']
    for domain, result in domains.items():
        m = result['metrics']; rel = result['relative_N10_pct']['mass_no_gap5']
        ci = result['comparisons']['mass_no_gap5_guided_vs_base7_guided']['descriptive_95pct_N10']
        counts = lambda k:'/'.join(str(round(m[a+'_guided']['H'+str(k)]*512)) for a in ARMS)
        lines.append(f'| {domain} | {rel["native"]:+.3f}% | {rel["old10_reference_guided"]:+.3f}% | [{ci[0]:+.7f}, {ci[1]:+.7f}] | {counts(10)} | {counts(50)} |')
    lines += ['', '组合5维对原7维的两域N10差值区间均跨0。只能作为低成本简化候选，不能宣称显著胜出或等效。每域17组比较同时报告N10/H10描述区间；另有1项N10交互区间，均未经多重比较校正。',
        '组合5维内容：完整历史长度1项、两个专家最高分减全目录均值2项、两个专家独立分布落在实际过滤最近20历史上的质量2项。',
        '任务解释依次为历史证据规模、专家候选突出程度、专家意见与当前输出过滤规则的适配情况。这些是可观察状态，不直接等同于真实可靠性或预测正确率。',
        '不再使用TopK计数或Top1-Top2分差；仍沿用固定TiSASRec+BGE、32隐藏单元共享干路、用户级beta/alpha及全路径生成引导。5维291参数，7维355参数。',
        '概率质量：p_e=softmax(20*s_e)，R_e为去重sample.history内p_e之和。每专家分别计算，不是最终beta融合分布的实际质量；也不预设R_e与beta/alpha的单调关系。',
        '新质量输入在开发集的行为/文本均值如下：']
    for domain, result in domains.items():
        f = result['independent_feature_checks']['validation']
        lines.append(f'{domain}：{f["behavior_filtered_mass"]["mean"]:.4f}/{f["text_filtered_mass"]["mean"]:.4f}。这描述分配给已交互商品的概率，不是有同样比例的预测错误。')
    lines += ['', '| 数据集 | 各组CE选轮：原7/改7/删5/组合5 | 各组NLL选轮 | 组合5维alpha/行为beta均值 | 组合5维生成/同规则后重排N10 |',
        '|---|---|---|---|---|']
    for domain, result in domains.items():
        fits, m = result['training']['fits'], result['metrics']
        epochs = lambda key:'/'.join(str(fits[a][key]) for a in ARMS)
        g = fits['mass_no_gap5']
        lines.append(f'| {domain} | {epochs("parent_epochs")} | {epochs("selected_epochs")} | {g["alpha_mean"]:.6f}/{g["behavior_weight_mean"]:.6f} | {m["mass_no_gap5_guided"]["N10"]:.6f}/{m["mass_no_gap5_late"]["N10"]:.6f} |')
    lines += ['', '各组预算、内部hash留出和选轮规则匹配，但选中轮数不同。两域所有5维CE都选20轮预算上限，不能据单轮实验排除更充分优化的影响。本轮不追加训练；不把组合差值单独归因为特征冗余。',
        '组合5维生成优于同规则后重排的两域点估计，但对应N10描述区间均跨0。',
        '统计解释11/11检查已记录。重复使用开发512用户、单seed、方法选择后的偏乐观风险保留；没有训练随机性区间或独立泛化确认。',
        '原特征来自LIME衍生实现，生成概率混合受AGRec启发，继续保留来源。任务对齐的替换可使设计理由更具体，不能仅以精简或更名主张方法新颖性。']
    REPORT.write_text('\n'.join(lines)+'\n')
    write(OUT/'status.json', dict(state='COMPLETED', verification='VERIFIED_SAVED_OUTPUTS',
        hashes_checked=len(checked), updated_at=datetime.now().astimezone().isoformat(), test_read=False,
        new_training=False, new_generation=False, report=str(REPORT.relative_to(ROOT))))
    print(json.dumps(dict(state='COMPLETED', hashes_checked=len(checked), report=str(REPORT)), ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if OUT.exists():
            write(OUT/'status.json', dict(state='FAILED', error=repr(exc), test_read=False))
        raise
