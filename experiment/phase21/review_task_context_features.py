"""Independent CPU audit of saved original10/pruned7/task10 outputs."""
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from experiment.phase21 import review_decision_features as V

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'artifacts/phase21/task_context_features_20261016_v1'
BASE = ROOT/'artifacts/phase21/decision_features_20261016_v1'
TIME = ROOT/'artifacts/phase21/time_behavior_expert_20261015_v1'
OUT = ROOT/'artifacts/phase21/task_context_features_review_20261016_v1'
REPORT = ROOT/'report/第二十一阶段/生成引导十维与精简七维结果复核_20261016.md'
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
            item = np.load(src/'item.npy', mmap_mode='r')
            expected = []
            for row, sample in enumerate(ss):
                history = sample['full_history']
                e = np.ascontiguousarray(item[row, :, :2].T)
                mask = np.isin(np.arange(1, e.shape[1]+1), history)
                order = [np.argsort(-x, kind='stable') for x in e]
                top = np.asarray([e[j, ix[:2]] for j, ix in enumerate(order)])
                expected.append(np.r_[np.log1p(len(history))/5,top[:,0]-e.mean(1),
                    top[:,0]-top[:,1],e.std(1,ddof=1),[mask[ix[:5]].mean() for ix in order],mask.mean()])
            expected=np.asarray(expected,dtype=np.float32)
            np.testing.assert_allclose(old,expected,atol=2e-6,rtol=1e-5)
            folder=p/'features'/part
            assert read(folder/'metadata.json')['users_order']==[s['user_id'] for s in ss]
            np.testing.assert_array_equal(np.load(folder/'old10.npy'),old)
            seven=np.load(folder/'pruned7.npy'); ten=np.load(folder/'task10.npy')
            np.testing.assert_array_equal(seven,old[:,[0,1,2,3,4,7,8]])
            np.testing.assert_array_equal(ten[:,:7],seven)
            roots=np.load(BASE/domain/'features'/part/'root_probabilities.npz')
            np.testing.assert_array_equal(roots['users'],[s['user_id'] for s in ss])
            prob=roots['probabilities'].astype(np.float64); prob/=prob.sum(-1,keepdims=True)
            gram=prob[:,0]
            entropy=-(gram*np.log(np.maximum(gram,1e-300))).sum(-1)/np.log(gram.shape[-1])
            extra=[entropy]
            for j in (1,2):
                expert=prob[:,j]; middle=(gram+expert)/2
                kg=(gram*(np.log(np.maximum(gram,1e-300))-np.log(np.maximum(middle,1e-300)))).sum(-1)
                ke=(expert*(np.log(np.maximum(expert,1e-300))-np.log(np.maximum(middle,1e-300)))).sum(-1)
                extra.append((kg+ke)/(2*np.log(2)))
            extra=np.column_stack(extra)
            np.testing.assert_allclose(ten[:,7:],extra,atol=1e-7,rtol=1e-6)
            features[part]=dict(users=len(ss),all_original_formulas_recomputed=True,
                retained_seven_exact=True,extra_three_independent_kl_verified=True,
                max_old_error=float(np.abs(old-expected).max()),max_extra_error=float(np.abs(ten[:,7:]-extra).max()))

        signals, alphas, betas = [np.load(p/name) for name in ('signals.npz', 'alphas.npz', 'betas.npz')]
        item = np.load(TIME/domain/'features/tisas/validation/item.npy', mmap_mode='r')[:, :, :2]
        heads = {}
        for arm, dimension in (('old10', 10), ('pruned7', 7), ('task10', 10)):
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
            hidden = np.maximum(x@sd['input.weight'].T+sd['input.bias'], 0)
            logits = hidden@sd['beta.weight'].T+sd['beta.bias']
            w = np.exp(logits-logits.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
            alpha = np.clip(1/(1+np.exp(-(hidden@sd['strength.weight'].T+sd['strength.bias']).ravel())), 1e-6, 1-1e-6)
            np.testing.assert_allclose(w, betas[arm], atol=2e-6, rtol=2e-5)
            np.testing.assert_allclose(alpha, alphas[arm], atol=2e-6, rtol=2e-5)
            np.testing.assert_allclose((w[:, None, :]*item).sum(-1), signals[arm], atol=3e-6, rtol=2e-5)
            count = sum(v.size for v in sd.values()); assert count == {10: 451, 7: 355}[dimension]
            if arm == 'old10':
                for filename, values in (('signals.npz', signals[arm]), ('alphas.npz', alphas[arm]), ('betas.npz', betas[arm])):
                    np.testing.assert_array_equal(values, np.load(BASE/domain/filename)['old10'])
            heads[arm] = dict(parameters=count, all512_heads_verified=True,
                              alpha=V.describe(alphas[arm]), behavior_weight=V.describe(betas[arm][:, 0]))
        decompositions={}
        for first,second in [('task10_guided','old10_guided'),('task10_guided','pruned7_guided'),
                             ('pruned7_guided','old10_guided')]:
            a,b=ranks[first],ranks[second]; delta=V.utility(a)-V.utility(b)
            masks=dict(gained=(a<=10)&(b>10),lost=(a>10)&(b<=10),
                       retained=(a<=10)&(b<=10),neither=(a>10)&(b>10))
            groups={k:dict(users=int(mask.sum()),delta_N10_contribution=float(delta[mask].sum()/512))
                    for k,mask in masks.items()}
            retained=masks['retained']
            movements=dict(improved=int((retained&(a<b)).sum()),worsened=int((retained&(a>b)).sum()),
                           unchanged=int((retained&(a==b)).sum()))
            decompositions[first+'_vs_'+second]=dict(groups=groups,retained_rank_changes=movements)
        # Fixed fitted head, set the standardized GRAM columns to zero (their
        # training means). This diagnoses output use, not recommendation benefit.
        ck=torch.load(p/'task10.pt',map_location='cpu')
        sd={k:v.numpy() for k,v in ck['state_dict'].items()}
        u=np.load(p/'features/validation/task10.npy')
        x=np.clip((u-ck['mean'])/ck['std'],-10,10); x[:,7:]=0
        hidden=np.maximum(x@sd['input.weight'].T+sd['input.bias'],0)
        z=hidden@sd['beta.weight'].T+sd['beta.bias']; w0=np.exp(z-z.max(1,keepdims=True)); w0/=w0.sum(1,keepdims=True)
        a0=np.clip(1/(1+np.exp(-(hidden@sd['strength.weight'].T+sd['strength.bias']).ravel())),1e-6,1-1e-6)
        sensitivity=dict(mean_abs_behavior_weight_change=float(np.abs(w0[:,0]-betas['task10'][:,0]).mean()),
                         mean_abs_alpha_change=float(np.abs(a0-alphas['task10']).mean()),
                         intervention='standardized GRAM inputs set to zero at fixed trained head',
                         fresh_counterfactual_generation=False)
        m=summary['metrics']
        relative={a:{ref:100*(m[a+'_guided']['N10']/m[ref]['N10']-1)
                         for ref in ('native','old10_guided','pruned7_guided',a+'_late')}
                  for a in ('pruned7','task10')}
        summary.update(head_checks=heads,feature_checks=features,top10_change_decompositions=decompositions,
                       derived_relative_N10_pct=relative,task10_gram_input_sensitivity=sensitivity)
        domains[domain]=summary
        print(domain,'VERIFIED: 9 rank arrays, all feature formulas, 3 heads, 9 paired comparisons',flush=True)
    review=dict(state='VERIFIED_SAVED_OUTPUTS',hashes_checked=len(checked),domains=domains,
        elapsed_experiment_seconds=status['elapsed_seconds'],test_read=False,new_training=False,new_generation=False,
        statistical_review=dict(coverage='11/11',confidence='CAUTION: exploratory',
            simpson='Domains separately reported; added GRAM features have opposite point-estimate changes.',
            ecological='Mean beta/alpha and total hit counts do not imply user-level equivalence or uniform benefit.',
            berkson='Reused development512 pools; no population extrapolation.',
            collider='Hit/rank decomposition descriptive only; no causal subgroup inference.',
            base_rate='All512 users retained; gained/lost/retained denominators explicit.',
            regression_to_mean='Original10 control predictions reproduced; single seed and no post-selected cohorts.',
            survivorship='Three prespecified arms and both domains completed.',
            look_elsewhere='18 uncorrected descriptive paired comparisons; intervals not formal confirmation.',
            forking_paths='Follow-up chosen after several feature experiments; repeated development use disclosed.',
            causation='Bundle/architecture dimensions differ; individual input necessity and latent causes unproven.',
            reverse_causality='Features use visible history and initial distributions; no true target prefix.'))
    write(OUT/'summary.json',review); write(OUT/'verified_hashes.json',checked)
    lines=['# 生成引导十维与精简七维结果复核','','## Material Passport','',
        '- Origin Skill: academic-research-suite / experiment-agent','- Origin Mode: validate',
        '- Origin Date: 2026-10-16','- Verification Status: VERIFIED_SAVED_OUTPUTS；统计解释 ANALYZED，未重跑训练或生成',
        '- Version Label: task_context_features_review_v1','',
        f'两域完成，耗时{status["elapsed_seconds"]/60:.1f}分钟，GPU租约释放。独立核对{len(checked)}个文件哈希、18份排名、全部原特征/新增特征公式及6个融合头输出；原10维复现历史结果。无test访问。','',
        '| 数据集 | GRAM N10 | 原10维 | 精简7维 | 新10维 | 新对原10维 | 新对精简7维 | 新对GRAM |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for d,v in domains.items():
        m=v['metrics'];r=v['derived_relative_N10_pct']['task10']
        lines.append('| '+d+' | '+' | '.join(f'{m[k]["N10"]:.6f}' for k in
            ('native','old10_guided','pruned7_guided','task10_guided'))+
            f' | {r["old10_guided"]:+.2f}% | {r["pruned7_guided"]:+.2f}% | {r["native"]:+.2f}% |')
    lines += ['', '新10维两域仍高于GRAM，达到用户允许适度性能取舍的探索条件；不能按旧2%阈值否决。相比精简7维，新增GRAM信息在Toys点估计提高、Beauty点估计下降，未得到双域收益支持。',
        '新10维对旧10维及对精简7维的两域N10差值描述性95%区间均跨0：不证明等效、无效或显著退化。原10维优势及各替换项的必要性仍不能由单seed512用户确定。','',
        '| 数据集 | 原/精简/新Top10命中人数 | 原/精简/新Top50命中人数 | 新10维生成/后重排N10 | 新对精简7维N10差值区间 |',
        '|---|---:|---:|---:|---|']
    for d,v in domains.items():
        m=v['metrics'];ci=v['comparisons']['task10_guided_vs_pruned7_guided']['descriptive_95pct_N10']
        h=lambda k:'/'.join(str(round(m[a+'_guided']['H'+str(k)]*512)) for a in ('old10','pruned7','task10'))
        lines.append(f'| {d} | {h(10)} | {h(50)} | {m["task10_guided"]["N10"]:.6f}/{m["task10_late"]["N10"]:.6f} | [{ci[0]:+.7f}, {ci[1]:+.7f}] |')
    lines += ['', 'Top10命中人数增加不等于NDCG提高；以下是新10维相对原10维的逐用户分解：']
    for d,v in domains.items():
        z=v['top10_change_decompositions']['task10_guided_vs_old10_guided'];g=z['groups'];r=z['retained_rank_changes']
        lines.append(f'{d}：新增{g["gained"]["users"]}、丢失{g["lost"]["users"]}；共同命中中改善{r["improved"]}、变差{r["worsened"]}、不变{r["unchanged"]}。新增/丢失/共同命中的N10贡献分别为'+
            '/'.join(f'{g[k]["delta_N10_contribution"]:+.8f}' for k in ('gained','lost','retained'))+'。')
    lines += ['', '新增输入已进入拟合输出，不宜称为没有学到。固定新模型，将新增三项置为训练均值后的输出敏感性如下；未重新生成反事实推荐，不是性能消融。','',
        '| 数据集 | 原/精简/新alpha均值 | 原/精简/新行为权重均值 | 新输入置均值后alpha平均绝对变化 | 行为权重平均绝对变化 |',
        '|---|---|---|---:|---:|']
    for d,v in domains.items():
        fits=v['training']['fits'];s=v['task10_gram_input_sensitivity']
        means=lambda key:'/'.join(f'{fits[a][key]:.6f}' for a in ('old10','pruned7','task10'))
        lines.append(f'| {d} | {means("alpha_mean")} | {means("behavior_weight_mean")} | {s["mean_abs_alpha_change"]:.6f} | {s["mean_abs_behavior_weight_change"]:.6f} |')
    lines += ['', '精简7维为历史长度1＋头部突出2＋头部分差2＋完整历史Top5重合2，与先前首层熵/JS七维不同；新10维在其上添加GRAM首层熵及两项专家/GRAM JS，全部进入共享非线性干路。',
        '同协议选轮并非相同选定轮数：Toys原/精简/新CE均12、NLL为4/4/8；Beauty CE为12/12/20、NLL均4。按内部留出规则选择，未按本次外部开发NDCG选轮。不能因外部结果下降就称未学习或单独归因于损失、某特征、首层粒度。',
        '新10维与原10维同为451参数；精简7维355参数。task10对pruned7比较同时增加信息和参数，不是严格等参数的信息消融。',
        '两个新方案均保留作为方法候选；新增GRAM三项尚无双域正收益证据，不直接定稿。未自动启动Top5替换、头部分差消融、更多特征搜索或完整论文实验。',
        '两域新生成优于同规则后重排，但N10差值区间均跨0，不称稳定显著生成优势。',
        '统计解释检查11/11；18组未经校正描述性比较、重复开发池、单seed。原始作者评分/特征及生成混合来源继续保留。']
    REPORT.write_text('\n'.join(lines)+'\n')
    write(OUT/'status.json',dict(state='COMPLETED',verification='VERIFIED_SAVED_OUTPUTS',
        updated_at=datetime.now().astimezone().isoformat(),test_read=False,new_training=False,
        new_generation=False,report=str(REPORT.relative_to(ROOT))))


if __name__ == '__main__':
    main()
