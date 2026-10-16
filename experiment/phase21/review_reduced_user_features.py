"""Independent CPU audit of the saved original10/reduced9 comparison."""
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from experiment.phase21 import review_decision_features as V

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'artifacts/phase21/reduced_user_features_20261016_v1'
BASE = ROOT/'artifacts/phase21/decision_features_20261016_v1'
TIME = ROOT/'artifacts/phase21/time_behavior_expert_20261015_v1'
OUT = ROOT/'artifacts/phase21/reduced_user_features_review_20261016_v1'
REPORT = ROOT/'report/第二十一阶段/原十维与精简九维结果复核_20261016.md'
read, write = V.read, V.write


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
    assert all(v['exit_code'] == 0 and v['lease_released'] for v in status['finished'].values())
    verify(RUN/'manifest.json')
    print('Input/source hashes verified', flush=True)
    domains = {}
    for domain in ('Toys', 'Beauty'):
        p = RUN/domain
        for name in ('feature_manifest', 'training_manifest', 'output_manifest'):
            verify(p/(name+'.json'))
        assert read(p/'lease_status.json')['state'] == 'released'
        assert read(p/'runtime_checks.json')['gram_model_digest_unchanged']
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
        for key, r in ranks.items():
            np.testing.assert_array_equal(r, archive[key])
            for metric, value in V.metrics(r).items():
                assert abs(value-summary['metrics'][key][metric]) < 1e-12
        prior = np.load(BASE/domain/'ranks.npz')
        for key in ('native', 'old10_guided', 'old10_late'):
            np.testing.assert_array_equal(ranks[key], prior[key])
        for suffix in ('guided', 'late'):
            np.testing.assert_array_equal(ranks['historical_'+suffix], ranks['old10_'+suffix])
        indices = np.random.RandomState(20261016).randint(0, 512, size=(2000, 512))
        for key, c in summary['comparisons'].items():
            first, second = key.split('_vs_'); a, b = ranks[first], ranks[second]
            for label, delta in (('N10', V.utility(a)-V.utility(b)),
                                 ('H10', (a<=10).astype(float)-(b<=10).astype(float))):
                assert abs(delta.mean()-c['delta_'+label]) < 1e-12
                np.testing.assert_allclose(np.percentile(delta[indices].mean(1), [2.5, 97.5]),
                                           c['descriptive_95pct_'+label], atol=1e-12, rtol=0)
            assert c['top10_gained'] == int(((a<=10)&(b>10)).sum())
            assert c['top10_lost'] == int(((a>10)&(b<=10)).sum())

        features = {}
        for part, ss in (('train', fit), ('validation', samples)):
            src = TIME/domain/'features/tisas'/part
            old = np.load(src/'user.npy')
            new = np.load(p/'features'/part/'reduced9.npy')
            item = np.load(src/'item.npy', mmap_mode='r')
            assert old.shape == (len(ss), 10) and new.shape == (len(ss), 9)
            np.testing.assert_array_equal(new, old[:, :9])
            assert read(p/'features'/part/'metadata.json')['users_order'] == [s['user_id'] for s in ss]
            expected = []
            for row, sample in enumerate(ss):
                history = sample['full_history']
                assert len(set(history)) == len(history)
                e = np.ascontiguousarray(item[row, :, :2].T)
                mask = np.isin(np.arange(1, e.shape[1]+1), history)
                order = [np.argsort(-x, kind='stable') for x in e]
                top = np.asarray([e[j, ix[:2]] for j, ix in enumerate(order)])
                expected.append(np.r_[np.log1p(len(history))/5,
                                      top[:, 0]-e.mean(1), top[:, 0]-top[:, 1],
                                      e.std(1, ddof=1), [mask[ix[:5]].mean() for ix in order], mask.mean()])
            expected = np.asarray(expected, dtype=np.float32)
            np.testing.assert_allclose(old, expected, atol=2e-6, rtol=1e-5)
            features[part] = dict(users=len(ss), first_nine_exact=True,
                all_ten_formulas_recomputed=True, max_error=float(np.abs(old-expected).max()),
                histories_with_duplicates=0, removed_feature_determined_by_length=True)

        signals, alphas, betas = [np.load(p/name) for name in ('signals.npz', 'alphas.npz', 'betas.npz')]
        item = np.load(TIME/domain/'features/tisas/validation/item.npy', mmap_mode='r')[:, :, :2]
        heads = {}
        for arm, dimension in (('old10', 10), ('reduced9', 9)):
            ck = torch.load(p/(arm+'.pt'), map_location='cpu')
            assert ck['fit_users'] == [s['user_id'] for s in fit]
            sd = {k: v.numpy() for k, v in ck['state_dict'].items()}
            u = np.load(TIME/domain/'features/tisas/validation/user.npy')[:, :dimension]
            train = np.load(TIME/domain/'features/tisas/train/user.npy')[:, :dimension].astype(np.float64)
            mean = train.mean(0)
            std = np.sqrt(np.maximum((train**2).mean(0)-mean**2, 0)); std[std<1e-5] = 1
            np.testing.assert_array_equal(ck['mean'], mean.astype(np.float32))
            np.testing.assert_array_equal(ck['std'], std.astype(np.float32))
            x = np.clip((u-ck['mean'])/ck['std'], -10, 10)
            hidden = np.maximum(x@sd['input.weight'].T+sd['input.bias'], 0)
            logits = hidden@sd['beta.weight'].T+sd['beta.bias']
            w = np.exp(logits-logits.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
            alpha = np.clip(1/(1+np.exp(-(hidden@sd['strength.weight'].T+sd['strength.bias']).ravel())), 1e-6, 1-1e-6)
            np.testing.assert_allclose(w, betas[arm], atol=2e-6, rtol=2e-5)
            np.testing.assert_allclose(alpha, alphas[arm], atol=2e-6, rtol=2e-5)
            np.testing.assert_allclose((w[:, None, :]*item).sum(-1), signals[arm], atol=3e-6, rtol=2e-5)
            count = sum(v.size for v in sd.values()); assert count == {10: 451, 9: 419}[dimension]
            if arm == 'old10':
                for filename, values in (('signals.npz', signals[arm]), ('alphas.npz', alphas[arm]), ('betas.npz', betas[arm])):
                    np.testing.assert_array_equal(values, np.load(BASE/domain/filename)['old10'])
            heads[arm] = dict(parameters=count, all512_heads_verified=True,
                              alpha=V.describe(alphas[arm]), behavior_weight=V.describe(betas[arm][:, 0]))
        a, b = ranks['reduced9_guided'], ranks['old10_guided']
        delta = V.utility(a)-V.utility(b)
        masks = dict(gained=(a<=10)&(b>10), lost=(a>10)&(b<=10),
                     retained=(a<=10)&(b<=10), neither=(a>10)&(b>10))
        decomposition = {k: dict(users=int(mask.sum()), delta_N10_contribution=float(delta[mask].sum()/512))
                         for k, mask in masks.items()}
        hit = masks['retained']
        movements = dict(improved=int((hit&(a<b)).sum()), worsened=int((hit&(a>b)).sum()),
                         unchanged=int((hit&(a==b)).sum()))
        m = summary['metrics']
        summary['derived_relative_N10_pct'] = {ref: 100*(m['reduced9_guided']['N10']/m[ref]['N10']-1)
                                               for ref in ('native', 'old10_guided', 'reduced9_late')}
        summary.update(head_checks=heads, feature_checks=features,
                       top10_change_decomposition=decomposition, retained_target_rank_changes=movements)
        domains[domain] = summary
        print(domain, 'VERIFIED: all features, 7 ranks, 2 heads, old10 exact, 6 paired comparisons', flush=True)
    review = dict(state='VERIFIED_SAVED_OUTPUTS', hashes_checked=len(checked), domains=domains,
                  elapsed_experiment_seconds=status['elapsed_seconds'], test_read=False,
                  new_training=False, new_generation=False,
                  statistical_review=dict(coverage='11/11', confidence='CAUTION: exploratory',
                    simpson='Both datasets reported separately; no pooled superiority.',
                    ecological='No inference of individual expert reliability from mean weights.',
                    berkson='Repeated development512 pool; not a new random population sample.',
                    collider='Hit/rank decomposition is descriptive, not causal subgroup evidence.',
                    base_rate='All512 users retained in denominators, including misses.',
                    regression_to_mean='Original10 reproduced exactly; no stability claim from one seed.',
                    survivorship='Both arms completed for all planned evaluation users.',
                    look_elsewhere='12 uncorrected paired descriptive comparisons; Beauty negative interval disclosed.',
                    forking_paths='Feature deletion chosen after earlier exploration, not confirmatory testing.',
                    causation='Deletion also reduces parameters and changes representation; no proof removed feature useless.',
                    reverse_causality='Inputs computed from history/expert scores, not current target labels.'))
    write(OUT/'summary.json', review); write(OUT/'verified_hashes.json', checked)
    lines = ['# 原十维与精简九维结果复核', '', '## Material Passport', '',
             '- Origin Skill: academic-research-suite / experiment-agent', '- Origin Mode: validate',
             '- Origin Date: 2026-10-16', '- Verification Status: VERIFIED_SAVED_OUTPUTS；统计解释 ANALYZED，未重跑训练或生成',
             '- Version Label: reduced_user_features_review_v1', '',
             f'双域完整结束，耗时{status["elapsed_seconds"]/60:.1f}分钟，租约已释放。独立核对{len(checked)}个文件哈希、14份排名、全部输入特征公式及4个融合头输出。原10维输出与历史精确一致。无test访问。', '',
             '| 数据集 | GRAM N10 | 原10维 N10 | 精简9维 N10 | 9维对10维相对变化 | 9维对GRAM相对提升 |',
             '|---|---:|---:|---:|---:|---:|']
    for d, v in domains.items():
        m, rel = v['metrics'], v['derived_relative_N10_pct']
        lines.append('| '+d+' | '+' | '.join(f'{m[k]["N10"]:.6f}' for k in ('native', 'old10_guided', 'reduced9_guided'))+
                     f' | {rel["old10_guided"]:+.2f}% | {rel["native"]:+.2f}% |')
    lines += ['', '9维两域NDCG@10均略低于10维，但均保留对GRAM的提升。按用户接受适度性能取舍、优先方法简洁的探索标准，可将9维作为下一步主方案候选；不是声称9维性能更优或统计等效。', '',
              '| 数据集 | 10/9维Top10命中人数 | 10/9维Top50命中人数 | 9维生成/后重排N10 | 9-10维N10差值描述性95%区间 |',
              '|---|---:|---:|---:|---|']
    for d, v in domains.items():
        m = v['metrics']; o, n = m['old10_guided'], m['reduced9_guided']
        ci = v['comparisons']['reduced9_guided_vs_old10_guided']['descriptive_95pct_N10']
        lines.append(f'| {d} | {round(o["H10"]*512)}/{round(n["H10"]*512)} | {round(o["H50"]*512)}/{round(n["H50"]*512)} | {n["N10"]:.6f}/{m["reduced9_late"]["N10"]:.6f} | [{ci[0]:+.7f}, {ci[1]:+.7f}] |')
    lines += ['', '两域新旧Top10命中用户集合完全相同，因此NDCG差值全部来自共同命中商品的名次变化。']
    for d, v in domains.items():
        z = v['retained_target_rank_changes']
        lines.append(f'{d}：共同命中用户中，目标名次改善{z["improved"]}人、变差{z["worsened"]}人、不变{z["unchanged"]}人。')
    lines += ['', 'Beauty的9-10维配对bootstrap区间完全为负，不能笼统称两域下降都是随机波动；Toys区间跨0也不证明等效。这里只反映固定模型、当前512用户的重采样变化，未覆盖训练seed和反复开发选择的不确定性。',
              '9维生成均高于同规则后重排，但Toys相对差距仅约0.24%，Beauty约3.77%；两域差值区间均跨0，不据此声称稳定或显著优势。', '',
              '9维的设计解释：1维历史支持量；6维专家分数区分结构（每个专家最高分减均值、头两名差距、分数标准差）；2维Top5与完整历史重合比例。融合器据这些上下文统计学习专家比例与引导强度。它们不是经校准的正确概率，也不直接测量GRAM的不确定性或专家与GRAM的冲突。',
              '当前训练和开发历史均无重复、目录固定，因此删除项history_ratio由历史长度确定；但其非线性表达仍可能帮助有限网络，不能称完全无用。此次实验也没有证明剩余9项逐项必需。',
              '其余条件保持一致：TiSASRec+BGE冻结；隐藏32、参数451/419；CE父模型同规则（旧组复用已核验父模型，新组重训），两组重训路径NLL；两域两组均选CE12/NLL4；fresh beam50。',
              '建议先保留9维作为简洁候选，保存10维性能参照，不因本轮结果自动追加更宽网络或更多特征搜索。所有结论限于各512用户、单seed的开发探索。统计解释检查11/11；12组未经校正的描述性比较，不作正式显著性或等效结论。']
    REPORT.write_text('\n'.join(lines)+'\n')
    write(OUT/'status.json', dict(state='COMPLETED', verification='VERIFIED_SAVED_OUTPUTS',
                                 updated_at=datetime.now().astimezone().isoformat(), test_read=False,
                                 new_training=False, report=str(REPORT.relative_to(ROOT))))


if __name__ == '__main__':
    main()
