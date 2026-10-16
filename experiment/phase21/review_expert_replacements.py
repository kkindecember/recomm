"""CPU post-run audit of saved Qwen/TiSASRec experiments; no model selection fit."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'artifacts/phase21/single_fusion_20261014_v1'
OUT = ROOT/'artifacts/phase21/expert_replacement_review_20261015_v1'
REPORT = ROOT/'report/第二十一阶段/Qwen与TiSASRec替换结果复核_20261015.md'
JOBS = dict(qwen=('qwen_text_expert_20261015_v1', ('qwen06', 'qwen4')),
            time=('time_behavior_expert_20261015_v1', ('tisas', 'tisas_order')))


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def metrics(ranks):
    ranks = np.asarray(ranks)
    return {kind+str(k): float(np.mean((ranks <= k) if kind == 'H' else
        np.where(ranks <= k, np.log(2)/np.log(ranks+1), 0.)))
        for k in (5, 10, 20, 50) for kind in ('H', 'N')}


def sorted_rank(ids, values, sample, absent):
    seen = set(sample['history'])
    ordered = sorted(((float(v), int(i)) for i, v in zip(ids, values) if int(i) not in seen), key=lambda z: (-z[0], z[1]))
    return next((j+1 for j, (_, item) in enumerate(ordered) if item == sample['target']), absent)


def catalog_ranks(matrix, samples):
    ids = np.arange(1, matrix.shape[1]+1)
    ranks = []
    for row, sample in zip(matrix, samples):
        keep = ~np.isin(ids, sample['history'])
        ordered = ids[keep][np.lexsort((ids[keep], -row[keep]))]
        match = np.flatnonzero(ordered == sample['target'])
        ranks.append(int(match[0])+1 if len(match) else len(ids)+1)
    return np.asarray(ranks)


def head_weights(checkpoint, users):
    # Independent batched user-only MLP: candidate input columns do not enter.
    state = {k: v.numpy() for k, v in checkpoint['state_dict'].items()}
    u = np.clip((users-checkpoint['mean'][:10])/checkpoint['std'][:10], -10, 10)
    h = np.maximum(u@state['input.weight'][:, :10].T+state['input.bias'], 0)
    z = h@state['output.weight'].T+state['output.bias']
    w = np.exp(z-z.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
    return w


def main():
    OUT.mkdir(exist_ok=True)
    write(OUT/'status.json', dict(state='RUNNING', stage='saved_outputs_audit', no_new_training=True))
    checked, domains, details = {}, {d: {} for d in ('Toys', 'Beauty')}, {}
    for job, (dirname, variants) in JOBS.items():
        root = ROOT/'artifacts/phase21'/dirname
        status = read(root/'status.json')
        assert status['state'] == 'COMPLETED'
        assert all(x['exit_code'] == 0 and x['lease_released'] for x in status['finished'].values())
        for d in domains:
            p = root/d
            # Validate persisted outputs and trained weights/vectors against the
            # writer's frozen output manifests, without regenerating any result.
            for name in ('output_manifest.json', 'training_manifest.json'):
                for rel, expected in read(p/name)['sha256'].items():
                    if rel not in checked:
                        checked[rel] = digest(ROOT/rel)
                    assert checked[rel] == expected, rel
            summary = read(p/'summary.json')
            samples = read(p/'samples.json')
            assert samples == read(BASE/d/'samples.json') and len(samples) == 512
            rows = [json.loads(line) for line in (p/'predictions.jsonl').read_text().splitlines()]
            assert len(rows) == 512
            archive = np.load(p/'ranks.npz')
            independent = {}
            for key in summary['metrics']:
                ranks = []
                for sample, row in zip(samples, rows):
                    assert all(sample[k] == row[k] for k in ('user_id', 'target', 'history'))
                    ranks.append(sorted_rank(row[key]['ids'], row[key]['scores'], sample, 2**30))
                np.testing.assert_array_equal(ranks, archive[key])
                independent[key] = np.asarray(ranks)
                for metric, value in metrics(ranks).items():
                    assert abs(value-summary['metrics'][key][metric]) < 1e-12
            old_ranks = np.load(BASE/d/'ranks.npz')
            for key, old in (('native', 'native'), ('full_guided', 'user_guided'), ('full_late', 'user_late')):
                np.testing.assert_array_equal(independent[key], old_ranks[old])
            # Validate comparison point estimates and user gains/losses without
            # rerunning the bootstrap or treating the descriptive CI as formal.
            for pair, comparison in summary['comparisons'].items():
                a, b = [independent[k] for k in pair.split('_vs_')]
                expected = metrics(a)['N10']-metrics(b)['N10']
                assert abs(expected-comparison['delta_N10']) < 1e-12
                assert comparison['top10_gained'] == int(((a <= 10)&(b > 10)).sum())
                assert comparison['top10_lost'] == int(((b <= 10)&(a > 10)).sum())
            signals, pure = np.load(p/'signals.npz'), np.load(p/'pure_ranks.npz')
            old_items = np.load(BASE/d/'validation/item.npy', mmap_mode='r')
            baseline_channels = {k: catalog_ranks(old_items[:, :, j], samples) for j, k in enumerate(('behavior', 'text'))}
            components = {}
            for arm in ('full',)+variants:
                ranks = catalog_ranks(signals[arm], samples)
                np.testing.assert_array_equal(ranks, pure[arm])
                source = BASE/d if arm == 'full' else p/'features'/arm
                users = np.load(source/'validation/user.npy')
                items = np.load(source/'validation/item.npy', mmap_mode='r')
                checkpoint_path = BASE/d/'user.pt' if arm == 'full' else root/'models'/arm/d/'user.pt'
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                weights = head_weights(checkpoint, users)
                predicted = (weights[:, None, :]*items[:, :, :2]).sum(2)
                np.testing.assert_allclose(predicted, signals[arm], atol=3e-6, rtol=2e-5)
                if arm != 'full':
                    retained_channel = 0 if job == 'qwen' else 1
                    np.testing.assert_array_equal(items[:, :, retained_channel], old_items[:, :, retained_channel])
                channel_ranks = baseline_channels if arm == 'full' else {
                    k: catalog_ranks(items[:, :, j], samples) for j, k in enumerate(('behavior', 'text'))}
                if arm != 'full' and job == 'qwen':
                    np.testing.assert_array_equal(channel_ranks['text'], summary['training']['checks'][arm]['text_alone_ranks'])
                if job == 'time':
                    np.testing.assert_array_equal(channel_ranks['behavior'], np.load(p/'behavior_ranks.npz')[arm])
                components[arm] = dict(fused=metrics(ranks), standalone={k: metrics(v) for k,v in channel_ranks.items()},
                    behavior_weight_mean=float(weights[:, 0].mean()), behavior_weight_quantiles=np.quantile(weights[:, 0], [.1,.5,.9]).tolist(),
                    all512_fuser_scores_recomputed=True, head_epochs=checkpoint['selected_epochs'])
            baseline = summary['metrics']['full_guided']['N10']
            native = summary['metrics']['native']['N10']
            domains[d]['native'] = native
            domains[d]['sas_bge'] = dict(N10=baseline, relative_to_current_pct=0., relative_to_gram_pct=100*(baseline/native-1))
            for arm in variants:
                value = summary['metrics'][arm+'_guided']['N10']
                domains[d][arm] = dict(N10=value, relative_to_current_pct=100*(value/baseline-1),
                    relative_to_gram_pct=100*(value/native-1), comparison=summary['comparisons'][arm+'_guided_vs_full_guided'])
            details[job+'_'+d] = dict(components=components, comparisons=summary['comparisons'],
                time_strata=summary.get('visible_history_date_strata'), generated_ranks_verified=7*512,
                pure_ranks_verified=3*512, saved_training=summary['training']['fits'])
            print(job, d, 'VERIFIED', flush=True)
    audit = dict(state='VERIFIED', domains=domains, details=details, hashes_checked=len(checked),
        verification='Independent Python sorting, NDCG/H calculation, full-catalog ranks and all-user fusion recomputation',
        scope='Reused development users512/domain, seed0, fixed inputs/objectives, no new GPU work/test evaluation',
        decision='Retain SASRec+BGE as common provisional configuration; no two-domain replacement established',
        statistical_review=dict(coverage='11/11',
          simpson='Dataset signs differ; keep datasets separate, do not hide tradeoff with pooled mean.',
          ecological='No individual benefit or population-wide superiority inferred from domain averages.',
          berkson='Historical development sample reused; external generalization not established.',
          collider='Date strata descriptive only; no causal adjustment claims.',
          base_rate='512-user denominator and gains/losses preserved; no diagnostic-prevalence claim.',
          regression_to_mean='No new subset or next experiment selected from extreme subgroup gains.',
          survivorship='All planned arms and all512 users completed in both domains.',
          look_elsewhere='Many uncorrected paired intervals; no confirmatory significance claim.',
          forking_paths='Frozen protocol followed; same reused dev set remains exploratory.',
          causation='Architecture substitution differs beyond time; real-vs-order is one-seed controlled pipeline contrast.',
          reverse_causality='History-only chronology checked in preflight; no target dates used.'))
    write(OUT/'summary.json', audit)
    write(OUT/'verified_hashes.json', checked)
    lines = ['# Qwen 与 TiSASRec 替换结果复核', '',
        '结论：本轮继续保留 SASRec+BGE 作为两域统一的暂定组合。Qwen与TiSASRec都没有给出两个数据集一致的替换收益。', '',
        '本轮每域512名已重复使用的开发用户、seed0；GRAM、alpha=.3、beam50保持一致，新专家各自重训同结构用户融合器。所有值为完整引导生成的NDCG@10。', '',
        '| 方案 | Toys N10 | 相对当前 | Beauty N10 | 相对当前 |', '|---|---:|---:|---:|---:|']
    names = dict(sas_bge='SASRec+BGE', qwen06='SASRec+Qwen0.6B', qwen4='SASRec+Qwen4B',
                 tisas='TiSASRec+BGE（真实时间）', tisas_order='TiSASRec+BGE（等间隔）')
    lines.append(f'| GRAM原生 | {domains["Toys"]["native"]:.6f} | — | {domains["Beauty"]["native"]:.6f} | — |')
    for arm, label in names.items():
        t, b = domains['Toys'][arm], domains['Beauty'][arm]
        lines.append(f'| {label} | {t["N10"]:.6f} | {t["relative_to_current_pct"]:+.2f}% | {b["N10"]:.6f} | {b["relative_to_current_pct"]:+.2f}% |')
    lines += ['', '## 可以支持的判断', '',
        '- Qwen：Toys的0.6B几乎持平、4B小幅上升，Beauty分别下降5.65%/7.54%。4B并未在两个域均优于0.6B；当前BGE历史向量聚合方式下，没有统一替换依据。并不据此断言Qwen在所有检索/推荐设置都较差。',
        '- TiSASRec：相对现有SAS，Toys上升8.36%，Beauty下降8.83%。等间隔对照为Toys+3.50%、Beauty−3.25%；真实时间相对等间隔的生成N10差值分别+0.003254/−0.004606，两域该描述性95%区间均跨0。因此时间信息尚未被证明在两个域稳定有效。',
        '- 现有SAS+BGE相对同批用户原生GRAM为Toys+7.16%、Beauty+15.16%。替换组也都高于各自原生GRAM，所以它们不是完全无效；当前选择依据是统一配置的收益与损失。',
        '- 本轮仅检验了因果前缀时间归一化TiSASRec、固定200轮全目录CE；不是对原论文训练配方或全部时间模型的否定。', '',
        '## 独立专家与融合阶段', '',
        '| 域 | 行为专家 | 单独排序N10 | 与BGE融合后N10 | 引导GRAM N10 |', '|---|---|---:|---:|---:|']
    for d in domains:
        for a, label in (('full','SASRec'),('tisas','TiSASRec真实时间'),('tisas_order','TiSASRec等间隔')):
            c = details['time_'+d]['components'][a]
            guided = domains[d]['sas_bge' if a == 'full' else a]['N10']
            lines.append(f'| {d} | {label} | {c["standalone"]["behavior"]["N10"]:.6f} | {c["fused"]["N10"]:.6f} | {guided:.6f} |')
    lines += ['', 'TiSASRec在Beauty的单独行为排序已低于SASRec，不能把全部下降归因于GRAM生成；融合后差距较小，进入固定引导后差距扩大。最后一项是观察，尚不能证明学习alpha即可解决。',
        'Qwen两种规格在两个域的融合专家单独排序均低于原组合；Toys的最终引导却略有上升，说明专家独立指标与对GRAM的补充收益并非简单单调对应。', '',
        '## 统计与实现边界', '',
        '独立复核四个域任务：保存预测逐用户重新排序、NDCG/H重新计算、原生和旧基线全512回放一致、全目录融合分数从checkpoint独立重算、未替换的专家通道逐项一致、输出清单哈希一致；上游训练、编码检查和租约释放均通过。',
        'Qwen两种规格Beauty的描述性N10差值区间均在0以下；TiSASRec Toys对SAS的区间在0以上。其余多处区间跨0。所有区间为未校正的探索性区间，复用开发集和单seed不支持正式显著性结论。',
        '时间分组只作描述：Toys相对SAS的加权改善主要来自同日历史用户；不能由此编造“较大时间跨度纠错”的机制解释。',
        '本轮不启动8B、Qwen+TiSAS组合、多seed、全量或test。也不能把本次替换对照当作两个专家在两个域各自必要的消融证据。',
        '统计解释检查11/11完成；逐项说明、专家单独排序、融合权重分布、配对收益/误伤及哈希核对见 artifacts/phase21/expert_replacement_review_20261015_v1/summary.json。']
    REPORT.write_text('\n'.join(lines)+'\n')
    write(OUT/'status.json', dict(state='COMPLETED', verified=True, no_new_training=True,
        updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), report=str(REPORT.relative_to(ROOT))))
    print(json.dumps(domains), flush=True)


if __name__ == '__main__':
    main()
