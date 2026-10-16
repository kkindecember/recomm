"""Independent CPU audit of saved alpha-only GRAM-feature augmentation."""
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from experiment.phase21 import review_decision_features as V

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'artifacts/phase21/alpha_gram_features_20261016_v1'
BASE = ROOT/'artifacts/phase21/reduced_user_features_20261016_v1'
ROOTBASE = ROOT/'artifacts/phase21/decision_features_20261016_v1'
TIME = ROOT/'artifacts/phase21/time_behavior_expert_20261015_v1'
OUT = ROOT/'artifacts/phase21/alpha_gram_features_review_20261016_v1'
REPORT = ROOT/'report/第二十一阶段/九维融合器补GRAM状态结果复核_20261016.md'
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
        for current, previous in (('native','native'), ('base9_guided','reduced9_guided'),
                                  ('base9_late','reduced9_late'), ('old10_reference_guided','old10_guided')):
            np.testing.assert_array_equal(ranks[current], prior[previous])
        for suffix in ('guided', 'late'):
            np.testing.assert_array_equal(ranks['historical_'+suffix], ranks['base9_'+suffix])
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
            base = np.load(p/'features'/part/'base9.npy')
            new = np.load(p/'features'/part/'alpha12.npy')
            np.testing.assert_array_equal(base, np.load(BASE/domain/'features'/part/'reduced9.npy'))
            np.testing.assert_array_equal(new[:, :9], base)
            assert base.shape == (len(ss),9) and new.shape == (len(ss),12)
            assert read(p/'features'/part/'metadata.json')['users_order'] == [s['user_id'] for s in ss]
            roots = np.load(ROOTBASE/domain/'features'/part/'root_probabilities.npz')
            np.testing.assert_array_equal(roots['users'], [s['user_id'] for s in ss])
            prob = roots['probabilities'].astype(np.float64)
            prob /= prob.sum(-1, keepdims=True)
            gram = prob[:,0]
            entropy = -(gram*np.log(np.maximum(gram,1e-300))).sum(-1)/np.log(gram.shape[-1])
            divergences=[]
            for channel in (1,2):
                expert=prob[:,channel]; middle=(gram+expert)/2
                # Independently use the KL definition rather than entropy differences.
                kl_g=(gram*(np.log(np.maximum(gram,1e-300))-np.log(np.maximum(middle,1e-300)))).sum(-1)
                kl_e=(expert*(np.log(np.maximum(expert,1e-300))-np.log(np.maximum(middle,1e-300)))).sum(-1)
                divergences.append((kl_g+kl_e)/(2*np.log(2)))
            expected=np.column_stack((entropy,*divergences))
            np.testing.assert_allclose(new[:,9:],expected,atol=1e-7,rtol=1e-6)
            features[part]=dict(users=len(ss),first_nine_exact=True,extra_formulas_recomputed=True,
                                max_extra_error=float(np.abs(new[:,9:]-expected).max()))

        signals, alphas, betas = [np.load(p/name) for name in ('signals.npz', 'alphas.npz', 'betas.npz')]
        item = np.load(TIME/domain/'features/tisas/validation/item.npy', mmap_mode='r')[:, :, :2]
        heads = {}
        for arm, dimension in (('base9', 9), ('alpha12', 12)):
            ck = torch.load(p/(arm+'.pt'), map_location='cpu')
            assert ck['fit_users'] == [s['user_id'] for s in fit]
            sd = {k: v.numpy() for k, v in ck['state_dict'].items()}
            u = np.load(p/'features/validation'/(arm+'.npy'))
            train = np.load(p/'features/train'/(arm+'.npy')).astype(np.float64)
            mean = train.mean(0)
            std = np.sqrt(np.maximum((train**2).mean(0)-mean**2, 0)); std[std<1e-5] = 1
            np.testing.assert_array_equal(ck['mean'], mean.astype(np.float32))
            np.testing.assert_array_equal(ck['std'], std.astype(np.float32))
            x = np.clip((u-ck['mean'])/ck['std'], -10, 10)
            hidden = np.maximum(x[:,:9]@sd['input.weight'].T+sd['input.bias'], 0)
            logits = hidden@sd['beta.weight'].T+sd['beta.bias']
            w = np.exp(logits-logits.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
            base_logit=(hidden@sd['strength.weight'].T+sd['strength.bias']).ravel()
            alpha_without_extra=np.clip(1/(1+np.exp(-base_logit)),1e-6,1-1e-6)
            added_logit=(x[:,9:]@sd['gram_strength.weight'].T).ravel() if arm=='alpha12' else np.zeros(len(x),np.float32)
            alpha=np.clip(1/(1+np.exp(-(base_logit+added_logit))),1e-6,1-1e-6)
            effects=None
            if arm=='alpha12':
                magnitude=float(np.abs(alpha-alpha_without_extra).mean())
                assert abs(magnitude-summary['training']['fits'][arm]['mean_abs_alpha_effect_of_extra'])<2e-6
                effects=dict(standardized_coefficients=sd['gram_strength.weight'].ravel().tolist(),
                             alpha_mean_absolute_effect_at_fixed_head=magnitude,
                             added_logit=V.describe(added_logit),
                             alpha_without_extra=V.describe(alpha_without_extra))
            np.testing.assert_allclose(w, betas[arm], atol=2e-6, rtol=2e-5)
            np.testing.assert_allclose(alpha, alphas[arm], atol=2e-6, rtol=2e-5)
            np.testing.assert_allclose((w[:, None, :]*item).sum(-1), signals[arm], atol=3e-6, rtol=2e-5)
            count = sum(v.size for v in sd.values()); assert count == {9:419, 12:422}[dimension]
            if arm == 'base9':
                for filename, values in (('signals.npz', signals[arm]), ('alphas.npz', alphas[arm]), ('betas.npz', betas[arm])):
                    np.testing.assert_array_equal(values, np.load(BASE/domain/filename)['reduced9'])
                old_state=torch.load(BASE/domain/'reduced9.pt',map_location='cpu')['state_dict']
                for key,value in ck['state_dict'].items():
                    torch.testing.assert_close(value,old_state[key],atol=0,rtol=0)
            heads[arm] = dict(parameters=count, all512_heads_verified=True, extra_effect=effects,
                              alpha=V.describe(alphas[arm]), behavior_weight=V.describe(betas[arm][:, 0]))
        a, b = ranks['alpha12_guided'], ranks['base9_guided']
        delta = V.utility(a)-V.utility(b)
        masks = dict(gained=(a<=10)&(b>10), lost=(a>10)&(b<=10),
                     retained=(a<=10)&(b<=10), neither=(a>10)&(b>10))
        decomposition = {k: dict(users=int(mask.sum()), delta_N10_contribution=float(delta[mask].sum()/512))
                         for k, mask in masks.items()}
        hit = masks['retained']
        movements = dict(improved=int((hit&(a<b)).sum()), worsened=int((hit&(a>b)).sum()),
                         unchanged=int((hit&(a==b)).sum()))
        m = summary['metrics']
        summary['derived_relative_N10_pct'] = {ref: 100*(m['alpha12_guided']['N10']/m[ref]['N10']-1)
                                               for ref in ('native', 'base9_guided', 'old10_reference_guided', 'alpha12_late')}
        summary.update(head_checks=heads, feature_checks=features,
                       top10_change_decomposition=decomposition, retained_target_rank_changes=movements)
        domains[domain] = summary
        print(domain, 'VERIFIED: feature formulas, 8 ranks, 2 heads, base9 exact, 7 paired comparisons', flush=True)
    review = dict(state='VERIFIED_SAVED_OUTPUTS', hashes_checked=len(checked), domains=domains,
                  elapsed_experiment_seconds=status['elapsed_seconds'], test_read=False,
                  new_training=False, new_generation=False,
                  statistical_review=dict(coverage='11/11', confidence='CAUTION: exploratory',
                    simpson='Both datasets separately reported; no pooled benefit claim.',
                    ecological='Similar mean alpha does not mean no per-user changes.',
                    berkson='Repeated development512 pools; no population extrapolation.',
                    collider='Hit/rank decomposition is descriptive, not causal subgroup evidence.',
                    base_rate='All512 users retained in all metric denominators.',
                    regression_to_mean='Base9 weights and predictions exactly reproduced; single training seed.',
                    survivorship='Both arms completed for every planned evaluation user.',
                    look_elsewhere='14 uncorrected descriptive paired comparisons; Beauty negative interval disclosed.',
                    forking_paths='Follow-up chosen after earlier exploration, not confirmatory.',
                    causation='Three features and weights introduced together; joint shared-trunk training also changes beta.',
                    reverse_causality='Root statistics use start token only; target labels used in training/evaluation.'))
    write(OUT/'summary.json', review); write(OUT/'verified_hashes.json', checked)
    lines = ['# 九维融合器补GRAM状态结果复核', '', '## Material Passport', '',
             '- Origin Skill: academic-research-suite / experiment-agent', '- Origin Mode: validate',
             '- Origin Date: 2026-10-16', '- Verification Status: VERIFIED_SAVED_OUTPUTS；统计解释 ANALYZED，未重跑训练或生成',
             '- Version Label: alpha_gram_features_review_v1', '',
             f'双域完整结束，耗时{status["elapsed_seconds"]/60:.1f}分钟，租约已释放。独立核对{len(checked)}个文件哈希、16份排名、全部新增特征公式及4个融合头输出。原9维权重及输出与历史精确一致。无test访问。', '',
             '| 数据集 | GRAM N10 | 原9维 N10 | alpha补GRAM状态 N10 | 相对9维 | 相对GRAM |',
             '|---|---:|---:|---:|---:|---:|']
    for d,v in domains.items():
        m,rel=v['metrics'],v['derived_relative_N10_pct']
        lines.append('| '+d+' | '+' | '.join(f'{m[k]["N10"]:.6f}' for k in ('native','base9_guided','alpha12_guided'))+
                     f' | {rel["base9_guided"]:+.2f}% | {rel["native"]:+.2f}% |')
    lines += ['', '新增信息确实影响alpha，但当前简单加性接入没有带来双域NDCG@10收益；Beauty下降更明显。它仍达到用户允许“双域点估计高于GRAM”的探索条件，不能仅按曾讨论过的2%下降线否决；同样不能据此声称新增三项有益或必需。', '',
              '| 数据集 | 原/新Top10命中 | 原/新Top50命中 | 新-原N10描述性95%区间 | 新生成/后重排N10 |',
              '|---|---:|---:|---|---:|']
    for d,v in domains.items():
        m=v['metrics']; o,n=m['base9_guided'],m['alpha12_guided']
        ci=v['comparisons']['alpha12_guided_vs_base9_guided']['descriptive_95pct_N10']
        lines.append(f'| {d} | {round(o["H10"]*512)}/{round(n["H10"]*512)} | {round(o["H50"]*512)}/{round(n["H50"]*512)} | [{ci[0]:+.7f}, {ci[1]:+.7f}] | {n["N10"]:.6f}/{m["alpha12_late"]["N10"]:.6f} |')
    lines += ['', 'Toys新增1个Top10命中、丢失0个，共同命中排序净损失抵消新增贡献。Beauty新增0个、丢失2个，同时共同命中也有排序净损失。完整分解见summary.json。',
              'Beauty差值描述性区间完全为负，不能称两域下降都是随机波动；Toys区间跨0不证明等效。固定模型的用户重采样不覆盖训练seed或反复开发选择的不确定性。', '',
              '| 数据集 | 原/新alpha均值 | 固定新模型、三项归零后alpha平均绝对变化 | 三项标准化输入的logit系数：GRAM熵/行为JS/内容JS |',
              '|---|---:|---:|---|']
    for d,v in domains.items():
        o,n=v['head_checks']['base9'],v['head_checks']['alpha12']; e=n['extra_effect']
        lines.append(f'| {d} | {o["alpha"]["mean"]:.6f}/{n["alpha"]["mean"]:.6f} | {e["alpha_mean_absolute_effect_at_fixed_head"]:.6f} | '+', '.join(f'{x:+.6f}' for x in e['standardized_coefficients'])+' |')
    lines += ['', '三项归零指把标准化后的额外特征置为0，即原始特征置为训练均值；固定新模型参数。此诊断仅验证输出敏感性，未重新生成反事实排名，不等于新增信息的推荐收益，也不是新旧alpha之差。beta在固定参数下不读取新增三项，但联合训练改变共享干路，最终beta仍可不同。',
              '两域新组的内部验证路径NLL均更低：', '']
    for d,v in domains.items():
        fits=v['training']['fits']
        values=[min(fits[k]['selection_curve'],key=lambda x:x['loss'])['loss'] for k in ('base9','alpha12')]
        assert values[1]<values[0]
        lines.append(f'- {d}：{values[0]:.9f} → {values[1]:.9f}；两组均选择4轮。')
    lines += ['', '因此“没有学到”不准确：新增权重非零、alpha确实改变、内部NLL略改善。内部NLL和外部开发NDCG并非同一指标、同一用户集合；不能单凭方向不一致断言损失函数是唯一原因。首层信息粒度、简单加性模型的交互限制及开发样本差异均未被独立定位。',
              '当前建议保留9维作为性能参照，9+3保留为机制候选，不宣称其输入具有增益，也不自动扩大网络或替换训练目标。若用户优先机制解释，可接受性能取舍，但方法定稿仍需面对新增输入的消融结果。',
              '专家仍为TiSASRec+BGE；隐藏32，参数419/422；同CE父模型、相同联合训练预算与选轮、fresh beam50。两域新生成优于同规则后重排，相关描述性区间均跨0，不能称稳定显著优势。',
              '单seed、重复开发池各512；14组未经校正描述性比较；统计解释检查11/11。未自动启动后续实验。']
    REPORT.write_text('\n'.join(lines)+'\n')
    write(OUT/'status.json',dict(state='COMPLETED',verification='VERIFIED_SAVED_OUTPUTS',
                               updated_at=datetime.now().astimezone().isoformat(),test_read=False,
                               new_training=False,report=str(REPORT.relative_to(ROOT))))


if __name__ == '__main__':
    main()
