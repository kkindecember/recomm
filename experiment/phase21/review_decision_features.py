"""CPU audit of saved decision-feature outputs; no training or generation."""
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'artifacts/phase21/decision_features_20261016_v1'
PRIOR = ROOT / 'artifacts/phase21/joint_fusion_20261015_v2'
TIME = ROOT / 'artifacts/phase21/time_behavior_expert_20261015_v1'
OUT = ROOT / 'artifacts/phase21/decision_features_review_20261016_v1'
REPORT = ROOT / 'report/第二十一阶段/生成决策统计特征结果复核_20261016.md'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def utility(ranks):
    return np.where(ranks <= 10, 1 / np.log2(ranks + 1), 0.)


def metrics(ranks):
    return {f'{m}{k}': float(np.mean(ranks <= k) if m == 'H' else
            np.mean(np.where(ranks <= k, 1 / np.log2(ranks + 1), 0.)))
            for k in (5, 10, 20, 50) for m in ('H', 'N')}


def describe(values):
    return dict(mean=float(values.mean()), std=float(values.std()),
                quantiles=np.quantile(values, [0, .1, .5, .9, 1]).tolist())


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    write(OUT / 'status.json', dict(state='RUNNING', stage='saved_outputs_audit', test_read=False))
    checked = {}

    def manifest(path):
        for rel, expected in read(path)['sha256'].items():
            if rel not in checked:
                h = hashlib.sha256()
                with (ROOT / rel).open('rb') as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                        h.update(chunk)
                checked[rel] = h.hexdigest()
            assert checked[rel] == expected, rel

    status = read(RUN / 'status.json')
    assert status['state'] == 'COMPLETED'
    assert all(v['exit_code'] == 0 and v['lease_released'] for v in status['finished'].values())
    manifest(RUN / 'manifest.json')
    domains = {}
    for domain in ('Toys', 'Beauty'):
        p = RUN / domain
        for name in ('feature_manifest', 'training_manifest', 'output_manifest'):
            manifest(p / (name + '.json'))
        assert read(p / 'lease_status.json')['state'] == 'released'
        assert read(p / 'runtime_checks.json')['gram_model_digest_unchanged']
        samples, fit = read(p / 'samples.json'), read(p / 'fit_samples.json')
        assert {s['user_id'] for s in fit}.isdisjoint(s['user_id'] for s in samples)
        rows = [json.loads(line) for line in (p / 'predictions.jsonl').read_text().splitlines()]
        assert len(rows) == len(samples) == len({s['user_id'] for s in samples}) == 512
        summary, saved = read(p / 'summary.json'), np.load(p / 'ranks.npz')
        rr = {k: [] for k in summary['metrics']}
        for row, sample in zip(rows, samples):
            assert all(row[k] == sample[k] for k in ('user_id', 'target', 'history'))
            for key in rr:
                value = row[key]
                assert len(value['ids']) == len(value['scores']) == len(set(value['ids'])) == 50
                assert np.isfinite(value['scores']).all()
                ordered = sorted(((i, s) for i, s in zip(value['ids'], value['scores'])
                                  if i not in sample['history']), key=lambda x: (-x[1], x[0]))
                ids = [i for i, _ in ordered]
                rr[key].append(ids.index(sample['target']) + 1 if sample['target'] in ids else 2**30)
        rr = {k: np.asarray(v) for k, v in rr.items()}
        for key, ranks in rr.items():
            np.testing.assert_array_equal(ranks, saved[key])
            for m, value in metrics(ranks).items():
                assert abs(value - summary['metrics'][key][m]) < 1e-12
        original = np.load(PRIOR / domain / 'ranks.npz')
        np.testing.assert_array_equal(rr['native'], original['native'])
        for suffix in ('guided', 'late'):
            np.testing.assert_array_equal(rr['old10_' + suffix], original['tisas_user_' + suffix])
        for key, comparison in summary['comparisons'].items():
            first, second = key.split('_vs_')
            a, b = rr[first], rr[second]
            dn = utility(a) - utility(b)
            dh = (a <= 10).astype(float) - (b <= 10).astype(float)
            ix = np.random.RandomState(20261016).randint(0, len(a), size=(2000, len(a)))
            for label, delta in (('N10', dn), ('H10', dh)):
                assert abs(comparison['delta_' + label] - delta.mean()) < 1e-12
                np.testing.assert_allclose(comparison['descriptive_95pct_' + label],
                                           np.percentile(delta[ix].mean(1), [2.5, 97.5]), atol=1e-12, rtol=0)
            assert comparison['top10_gained'] == int(((a <= 10) & (b > 10)).sum())
            assert comparison['top10_lost'] == int(((b <= 10) & (a > 10)).sum())

        feature_checks = {}
        for part, ss in (('train', fit), ('validation', samples)):
            folder = p / 'features' / part
            roots = np.load(folder / 'root_probabilities.npz')
            np.testing.assert_array_equal(roots['users'], [s['user_id'] for s in ss])
            probs = roots['probabilities'].copy()
            assert np.isfinite(probs).all() and (probs >= 0).all()
            np.testing.assert_allclose(probs.sum(-1), 1., atol=2e-6, rtol=0)
            probs /= probs.sum(-1, keepdims=True)
            entropy = -(probs * np.log(np.maximum(probs, 1e-300))).sum(-1) / np.log(probs.shape[-1])
            js = []
            for i, j in ((1, 2), (1, 0), (2, 0)):
                a, b = probs[:, i], probs[:, j]
                middle = (a + b) / 2
                kl = lambda x: (x * (np.log(np.maximum(x, 1e-300)) -
                                    np.log(np.maximum(middle, 1e-300)))).sum(-1)
                js.append(np.clip((kl(a) + kl(b)) / (2 * np.log(2)), 0, 1))
            lengths = np.log1p([len(s.get('full_history', s['history'])) for s in ss]) / 5
            features = np.column_stack([lengths, entropy, *js])
            np.testing.assert_allclose(features, np.load(folder / 'decision7.npy'), atol=1e-7, rtol=1e-6)
            feature_checks[part] = dict(users=len(ss), all_seven_features_recomputed_from_cached_roots=True)

        heads = {}
        alphas, betas, signals = [np.load(p / f) for f in ('alphas.npz', 'betas.npz', 'signals.npz')]
        source = TIME / domain / 'features/tisas/validation'
        item = np.load(source / 'item.npy', mmap_mode='r')[:, :, :2]
        for arm in ('old10', 'decision7'):
            ck = torch.load(p / (arm + '.pt'), map_location='cpu')
            assert ck['fit_users'] == [s['user_id'] for s in fit]
            sd = {k: v.numpy() for k, v in ck['state_dict'].items()}
            inputs = np.load(source / 'user.npy' if arm == 'old10' else p / 'features/validation/decision7.npy')
            x = np.clip((inputs - ck['mean']) / ck['std'], -10, 10)
            h = np.maximum(x @ sd['input.weight'].T + sd['input.bias'], 0)
            logits = h @ sd['beta.weight'].T + sd['beta.bias']
            beta = np.exp(logits - logits.max(1, keepdims=True)); beta /= beta.sum(1, keepdims=True)
            alpha = np.clip(1 / (1 + np.exp(-(h @ sd['strength.weight'].T + sd['strength.bias']).ravel())), 1e-6, 1-1e-6)
            np.testing.assert_allclose(beta, betas[arm], atol=2e-6, rtol=2e-5)
            np.testing.assert_allclose(alpha, alphas[arm], atol=2e-6, rtol=2e-5)
            np.testing.assert_allclose((beta[:, None, :] * item).sum(-1), signals[arm], atol=3e-6, rtol=2e-5)
            heads[arm] = dict(parameters=sum(v.size for v in sd.values()),
                              alpha=describe(alphas[arm]), behavior_weight=describe(betas[arm][:, 0]),
                              all512_head_outputs_verified=True)
        a, b = rr['decision7_guided'], rr['old10_guided']
        change = utility(a) - utility(b)
        groups = dict(gained=(a <= 10) & (b > 10), lost=(b <= 10) & (a > 10),
                      retained=(a <= 10) & (b <= 10), neither=(a > 10) & (b > 10))
        decomposition = {k: dict(users=int(mask.sum()), contribution_to_delta_N10=float(change[mask].sum()/512))
                         for k, mask in groups.items()}
        values = summary['metrics']
        domains[domain] = dict(metrics=values, comparisons=summary['comparisons'], heads=heads,
                              training=summary['training'], feature_checks=feature_checks,
                              top10_change_decomposition=decomposition,
                              relative_new_vs_old_pct=100*(values['decision7_guided']['N10']/values['old10_guided']['N10']-1),
                              relative_new_vs_gram_pct=100*(values['decision7_guided']['N10']/values['native']['N10']-1))
        print(domain, 'VERIFIED: 7 rank arrays, all feature rows, 2 heads, paired intervals', flush=True)
    review = dict(state='VERIFIED_SAVED_OUTPUTS', hashes_checked=len(checked), domains=domains,
                  fresh_training=False, fresh_generation=False, test_read=False,
                  statistical_review=dict(coverage='11/11',
                    simpson='Both domains reported separately; no pooled winner claim.',
                    ecological='Mean weights do not imply constant weights or benefit for every user.',
                    berkson='Reused development512 cohorts; no population extrapolation.',
                    collider='No outcome-conditioned subgroup performance claims.',
                    base_rate='All512 denominators and gained/lost/retained counts retained.',
                    regression_to_mean='Matched old control reproduced; no selected-low subgroup claims.',
                    survivorship='Both prespecified arms and all users completed.',
                    look_elsewhere='Twelve comparisons; uncorrected descriptive intervals only.',
                    forking_paths='Repeated development use; exploratory, not confirmatory.',
                    causation='Whole feature bundle and input dimension changed; no individual-feature cause established.',
                    reverse_causality='Inputs use visible history and cached initial distributions; labels only in fitting/evaluation.'))
    write(OUT / 'summary.json', review)
    write(OUT / 'verified_hashes.json', checked)
    lines = ['# 生成决策统计特征结果复核', '', '## Material Passport', '',
             '- Origin Skill: academic-research-suite / experiment-agent', '- Origin Mode: validate',
             '- Origin Date: 2026-10-16', '- Verification Status: VERIFIED_SAVED_OUTPUTS；统计解释 ANALYZED，未重跑训练或生成',
             '- Version Label: decision_features_review_v1', '',
             f'两域各512开发用户、seed0，全部完成且显存租约释放。独立核对{len(checked)}个文件哈希、14份排名、全部新特征与4个融合头输出；旧10维结果精确复现历史结果。无test访问。', '',
             '| 数据集 | GRAM N10 | 旧10维 N10 | 新7维 N10 | 新对旧相对变化 | 新对GRAM相对变化 |',
             '|---|---:|---:|---:|---:|---:|']
    for d, v in domains.items():
        m = v['metrics']
        lines.append(f'| {d} | {m["native"]["N10"]:.6f} | {m["old10_guided"]["N10"]:.6f} | {m["decision7_guided"]["N10"]:.6f} | {v["relative_new_vs_old_pct"]:+.2f}% | {v["relative_new_vs_gram_pct"]:+.2f}% |')
    lines += ['', '新7维两域仍高于GRAM，符合用户允许一定性能取舍的探索标准；未显示优于旧10维的N10点估计。新旧差值描述性区间均跨0，不能声称正式显著退化或等效。', '',
              '| 数据集 | 旧/新Top10命中人数 | 旧/新Top50命中人数 | 新生成N10 | 新后重排N10 |',
              '|---|---:|---:|---:|---:|']
    for d, v in domains.items():
        m = v['metrics']; old, new = m['old10_guided'], m['decision7_guided']
        lines.append(f'| {d} | {round(old["H10"]*512)}/{round(new["H10"]*512)} | {round(old["H50"]*512)}/{round(new["H50"]*512)} | {new["N10"]:.6f} | {m["decision7_late"]["N10"]:.6f} |')
    lines += ['', 'Toys新增4个Top10命中、丢失1个，但共同命中用户的排序损失超过净新增贡献；Beauty新增0个、丢失2个，且共同命中排序也下降。新7维生成仍在两域高于同规则后重排，区间均跨0。', '',
              '| 数据集 | 旧/新平均行为权重 | 旧/新平均专家强度 |', '|---|---:|---:|']
    for d, v in domains.items():
        o, n = v['heads']['old10'], v['heads']['decision7']
        lines.append(f'| {d} | {o["behavior_weight"]["mean"]:.4f}/{n["behavior_weight"]["mean"]:.4f} | {o["alpha"]["mean"]:.4f}/{n["alpha"]["mean"]:.4f} |')
    lines += ['', '新头没有退化成常数；内部验证路径NLL两域均略低于旧头，但这不是NDCG目标，不能据此宣称排序更好。隐藏宽度同为32，参数451降到355；不据此自动扩宽网络。',
              '首层分布聚合会丢失同一首层分支内商品间的差异，因此只用首层熵/JS可能不足以替代原商品分数统计。这是结构上合理的解释候选，尚未通过特征组消融确定为本轮下降原因。',
              '方法判断：新7维可保留为简洁且仍超GRAM的候选，旧10维保留为性能参照。若继续，优先限定一次保留商品级分辨能力的紧凑特征对照；本次复核未启动该实验。',
              '重复开发池、单seed、12组未经校正的描述性区间；统计解释检查11/11。完整权重分布、差值区间及命中/排序分解见复核summary.json。']
    REPORT.write_text('\n'.join(lines) + '\n')
    write(OUT / 'status.json', dict(state='COMPLETED', verification='VERIFIED_SAVED_OUTPUTS',
          updated_at=datetime.now().astimezone().isoformat(), test_read=False, fresh_training=False,
          report=str(REPORT.relative_to(ROOT))))


if __name__ == '__main__':
    main()
