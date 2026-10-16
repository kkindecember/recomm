"""Independent CPU verification of saved item-granularity feature experiment."""
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from experiment.phase21 import review_decision_features as V

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'artifacts/phase21/decision_features_itemlevel_20261016_v1'
BASE = ROOT/'artifacts/phase21/decision_features_20261016_v1'
TIME = ROOT/'artifacts/phase21/time_behavior_expert_20261015_v1'
OUT = ROOT/'artifacts/phase21/decision_features_itemlevel_review_20261016_v1'
REPORT = ROOT/'report/第二十一阶段/七维特征商品粒度结果复核_20261016.md'
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
                    for block in iter(lambda: stream.read(8*1024*1024), b''): h.update(block)
                checked[rel] = h.hexdigest()
            assert checked[rel] == expected, rel

    status = read(RUN/'status.json')
    assert status['state'] == 'COMPLETED'
    assert all(v['exit_code'] == 0 and v['lease_released'] for v in status['finished'].values())
    verify(RUN/'manifest.json')
    domains = {}
    for domain in ('Toys', 'Beauty'):
        p = RUN/domain
        for name in ('feature_manifest', 'training_manifest', 'output_manifest'): verify(p/(name+'.json'))
        assert read(p/'lease_status.json')['state'] == 'released'
        assert read(p/'runtime_checks.json')['gram_model_digest_unchanged']
        samples, fit = read(p/'samples.json'), read(p/'fit_samples.json')
        assert {s['user_id'] for s in fit}.isdisjoint(s['user_id'] for s in samples)
        rows = [json.loads(line) for line in (p/'predictions.jsonl').read_text().splitlines()]
        assert len(rows) == len(samples) == len({s['user_id'] for s in samples}) == 512
        summary, archive = read(p/'summary.json'), np.load(p/'ranks.npz')
        ranks = {k: [] for k in summary['metrics']}
        for row, sample in zip(rows, samples):
            assert all(row[k] == sample[k] for k in ('user_id', 'target', 'history'))
            for key in ranks:
                value = row[key]
                assert len(value['ids']) == len(value['scores']) == len(set(value['ids'])) == 50
                assert np.isfinite(value['scores']).all()
                ordered = sorted(((i, s) for i, s in zip(value['ids'], value['scores']) if i not in sample['history']),
                                 key=lambda x: (-x[1], x[0]))
                ids = [i for i, _ in ordered]
                ranks[key].append(ids.index(sample['target'])+1 if sample['target'] in ids else 2**30)
        ranks = {k: np.asarray(v) for k, v in ranks.items()}
        for key, r in ranks.items():
            np.testing.assert_array_equal(r, archive[key])
            for m, value in V.metrics(r).items(): assert abs(value-summary['metrics'][key][m]) < 1e-12
        prior = np.load(BASE/domain/'ranks.npz')
        for a, b in (('native', 'native'), ('old10_guided', 'old10_guided'), ('old10_late', 'old10_late'),
                     ('root7_guided', 'decision7_guided'), ('root7_late', 'decision7_late')):
            np.testing.assert_array_equal(ranks[a], prior[b])
        for key, c in summary['comparisons'].items():
            first, second = key.split('_vs_'); a, b = ranks[first], ranks[second]
            indices = np.random.RandomState(20261016).randint(0, 512, size=(2000, 512))
            for label, delta in (('N10', V.utility(a)-V.utility(b)), ('H10', (a<=10).astype(float)-(b<=10).astype(float))):
                assert abs(delta.mean()-c['delta_'+label]) < 1e-12
                np.testing.assert_allclose(np.percentile(delta[indices].mean(1), [2.5, 97.5]),
                                           c['descriptive_95pct_'+label], atol=1e-12, rtol=0)
            assert c['top10_gained'] == int(((a<=10)&(b>10)).sum())
            assert c['top10_lost'] == int(((a>10)&(b<=10)).sum())

        feature_checks = {}
        for part, ss in (('train', fit), ('validation', samples)):
            old = np.load(p/'features'/part/'root7.npy')
            new = np.load(p/'features'/part/'item7.npy')
            np.testing.assert_array_equal(old, np.load(BASE/domain/'features'/part/'decision7.npy'))
            np.testing.assert_array_equal(new[:, [0, 1, 5, 6]], old[:, [0, 1, 5, 6]])
            assert read(p/'features'/part/'metadata.json')['users_order'] == [s['user_id'] for s in ss]
            item = np.load(TIME/domain/'features/tisas'/part/'item.npy', mmap_mode='r')
            max_error = 0.
            for start in range(0, len(ss), 64):
                stop = min(start+64, len(ss))
                scores = np.asarray(item[start:stop, :, :2], dtype=np.float64)*20
                probability = np.exp(scores-scores.max(1, keepdims=True))
                probability /= probability.sum(1, keepdims=True)
                entropy = -(probability*np.log(np.maximum(probability, 1e-300))).sum(1)
                middle = probability.mean(2)
                middle_entropy = -(middle*np.log(np.maximum(middle, 1e-300))).sum(1)
                expected = np.column_stack((entropy/np.log(item.shape[1]),
                                             (middle_entropy-entropy.mean(1))/np.log(2)))
                np.testing.assert_allclose(new[start:stop, 2:5], expected, atol=1e-7, rtol=1e-6)
                max_error = max(max_error, float(np.abs(new[start:stop, 2:5]-expected).max()))
            feature_checks[part] = dict(users=len(ss), unchanged_coordinates_exact=True,
                                         all_item_statistics_recomputed=True, max_error=max_error)

        signals, alphas, betas = [np.load(p/name) for name in ('signals.npz', 'alphas.npz', 'betas.npz')]
        item = np.load(TIME/domain/'features/tisas/validation/item.npy', mmap_mode='r')[:, :, :2]
        heads = {}
        for arm in ('root7', 'item7'):
            ck = torch.load(p/(arm+'.pt'), map_location='cpu')
            assert ck['fit_users'] == [s['user_id'] for s in fit]
            sd = {k: v.numpy() for k, v in ck['state_dict'].items()}
            u = np.load(p/'features/validation'/(arm+'.npy'))
            x = np.clip((u-ck['mean'])/ck['std'], -10, 10)
            hidden = np.maximum(x@sd['input.weight'].T+sd['input.bias'], 0)
            logits = hidden@sd['beta.weight'].T+sd['beta.bias']
            w = np.exp(logits-logits.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
            alpha = np.clip(1/(1+np.exp(-(hidden@sd['strength.weight'].T+sd['strength.bias']).ravel())), 1e-6, 1-1e-6)
            np.testing.assert_allclose(w, betas[arm], atol=2e-6, rtol=2e-5)
            np.testing.assert_allclose(alpha, alphas[arm], atol=2e-6, rtol=2e-5)
            np.testing.assert_allclose((w[:, None, :]*item).sum(-1), signals[arm], atol=3e-6, rtol=2e-5)
            assert sum(v.size for v in sd.values()) == 355
            if arm == 'root7':
                for filename, values in (('signals.npz', signals[arm]), ('alphas.npz', alphas[arm]), ('betas.npz', betas[arm])):
                    np.testing.assert_array_equal(values, np.load(BASE/domain/filename)['decision7'])
            heads[arm] = dict(parameters=355, all512_heads_verified=True,
                              alpha=V.describe(alphas[arm]), behavior_weight=V.describe(betas[arm][:, 0]))
        a, b = ranks['item7_guided'], ranks['root7_guided']
        delta = V.utility(a)-V.utility(b)
        masks = dict(gained=(a<=10)&(b>10), lost=(a>10)&(b<=10), retained=(a<=10)&(b<=10), neither=(a>10)&(b>10))
        for k, mask in masks.items():
            c = summary['top10_change_decomposition'][k]
            assert c['users'] == int(mask.sum())
            assert abs(c['delta_N10_contribution']-delta[mask].sum()/512) < 1e-12
        m = summary['metrics']
        summary['derived'] = {reference: 100*(m['item7_guided']['N10']/m[reference]['N10']-1)
                              for reference in ('native', 'old10_guided', 'root7_guided', 'item7_late')}
        summary.update(head_checks=heads, feature_checks=feature_checks)
        domains[domain] = summary
        print(domain, 'VERIFIED: all features, 9 ranks, 2 heads, original7 exact, 6 paired comparisons', flush=True)
    review = dict(state='VERIFIED_SAVED_OUTPUTS', hashes_checked=len(checked), domains=domains,
                  elapsed_experiment_seconds=status['elapsed_seconds'], test_read=False,
                  new_training=False, new_generation=False,
                  statistical_review=dict(coverage='11/11',
                    simpson='Opposite domain directions reported separately; no pooled superiority.',
                    ecological='No individual reliability claim from mean weights or mean performance.',
                    berkson='Reused development512 samples; no population extrapolation.',
                    collider='Gained/lost/retained decomposition is descriptive, not a causal subgroup analysis.',
                    base_rate='All512 denominators preserved, including misses.',
                    regression_to_mean='Original7 fully reproduced; small recovery not claimed stable.',
                    survivorship='Both prespecified arms and every evaluation user completed.',
                    look_elsewhere='12 uncorrected descriptive comparisons; no formal significance claim.',
                    forking_paths='Follow-up designed after earlier results; still exploratory.',
                    causation='Three features changed together; no single-feature or loss-function cause established.',
                    reverse_causality='Features use history and predictions; targets only in fitting/evaluation.'))
    write(OUT/'summary.json', review); write(OUT/'verified_hashes.json', checked)
    lines = ['# 七维特征商品粒度结果复核', '', '## Material Passport', '',
             '- Origin Skill: academic-research-suite / experiment-agent', '- Origin Mode: validate',
             '- Origin Date: 2026-10-16', '- Verification Status: VERIFIED_SAVED_OUTPUTS；统计解释 ANALYZED，未重跑训练或生成',
             '- Version Label: decision_features_itemlevel_review_v1', '',
             f'实验耗时{status["elapsed_seconds"]/60:.1f}分钟，双域完整结束且租约释放。独立核对{len(checked)}个文件哈希、18份排名、全部特征和4个融合头输出。原7维权重、商品分数和最终排名精确复现。无test访问。', '',
             '| 数据集 | GRAM N10 | 旧10维 N10 | 原7维 N10 | 商品粒度7维 N10 | 对原7维 | 对GRAM |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for d, v in domains.items():
        m = v['metrics']; derived = v['derived']
        lines.append('| '+d+' | '+' | '.join(f'{m[k]["N10"]:.6f}' for k in ('native', 'old10_guided', 'root7_guided', 'item7_guided'))+
                     f' | {derived["root7_guided"]:+.2f}% | {derived["native"]:+.2f}% |')
    lines += ['', '商品粒度7维没有双域一致NDCG收益；新旧7维差值区间均跨0，不宣称显著差异或已证明等效。两种7维仍低于旧10维NDCG点估计。', '',
              '| 数据集 | 原/新7维Top10命中 | 原/新7维Top50命中 | 新生成N10 | 新后重排N10 |',
              '|---|---:|---:|---:|---:|']
    for d, v in domains.items():
        m = v['metrics']; o, n = m['root7_guided'], m['item7_guided']
        lines.append(f'| {d} | {round(o["H10"]*512)}/{round(n["H10"]*512)} | {round(o["H50"]*512)}/{round(n["H50"]*512)} | {n["N10"]:.6f} | {m["item7_late"]["N10"]:.6f} |')
    lines += ['', 'Toys新增2个Top10命中、丢失3个，共同命中排序净变化接近零；Beauty新增3个、丢失0个，但共同命中排序损失抵消大部分新增贡献。两域Top50命中人数提高，并不等于Top10排序改善。',
              '商品粒度版在两域生成NDCG均高于同规则后重排，差值区间跨0。不能把描述性优势表述为统计确认。',
              'Toys商品粒度版内部验证路径NLL更低但开发NDCG更低；Beauty内部验证NLL略高而开发NDCG略高。训练目标与Top10排序并非一致，但本轮没有隔离训练目标的作用，也不能据此断言损失函数是原因。',
              '新方案有清楚的商品/分支分工，可保留为7维候选；当前证据不支持宣称它优于原7维，也不支持继续靠反复更换特征保证收益。本次未追加实验或更改最终方法选择。',
              '两域各512、单seed、重复开发池；12组未经校正描述性区间。统计解释检查11/11；详细分解及权重见复核summary.json。']
    REPORT.write_text('\n'.join(lines)+'\n')
    write(OUT/'status.json', dict(state='COMPLETED', verification='VERIFIED_SAVED_OUTPUTS',
                                 updated_at=datetime.now().astimezone().isoformat(), test_read=False,
                                 new_training=False, report=str(REPORT.relative_to(ROOT))))


if __name__ == '__main__': main()
