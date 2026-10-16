"""Check frozen expert-screen outputs; never fit a model or read test data."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'artifacts/phase21/expert_reselection_20261016_v2'
OUT = ROOT/'artifacts/phase21/expert_reselection_review_20261016_v1'
REPORT = ROOT/'report/第二十一阶段/专家重选结果独立复核与4096样本解释_20261016.md'
SOURCE = ROOT/'artifacts/phase20/lime_clean_benefit_20260922_v1'
FULL = ROOT/'artifacts/phase21/full_validation_20260924_v1'
QWEN = ROOT/'artifacts/phase21/qwen_text_expert_20261015_v1'
TIME = ROOT/'artifacts/phase21/time_behavior_expert_20261015_v1'
PAIRS = [('SASRec', 'TiSASRec'), ('SASRec', 'ItemCF'),
         ('BGE_loo', 'BGE'), ('Qwen4_loo', 'Qwen4'),
         ('BGE_loo', 'Qwen4_loo'), ('BGE_loo', 'BGE_uniform'),
         ('Qwen4_loo', 'Qwen4_uniform')]


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def utility(ranks):
    ranks = np.asarray(ranks)
    return np.where(ranks <= 10, 1/np.log2(1+ranks), 0.)


def contrast(a, b):
    delta = utility(a)-utility(b)
    rng = np.random.RandomState(20261016)
    means = np.empty(2000)
    for start in range(0, 2000, 50):
        means[start:start+50] = delta[rng.randint(0, len(delta), (50, len(delta)))].mean(1)
    return dict(delta_N10=float(delta.mean()),
        relative_N10_percent=float(100*(utility(a).mean()/utility(b).mean()-1)),
        descriptive_95pct_N10=np.percentile(means, [2.5, 97.5]).tolist(),
        top10_gained=int(((a <= 10) & (b > 10)).sum()),
        top10_lost=int(((b <= 10) & (a > 10)).sum()))


def main():
    start = time.monotonic()
    OUT.mkdir(exist_ok=False)
    write(OUT/'status.json', dict(state='RUNNING', test_read=False))
    assert read(RUN/'status.json')['state'] == 'COMPLETED'
    manifest = read(RUN/'manifest.json')['sha256']
    for domain in ('Toys', 'Beauty'):
        manifest.update(read(RUN/domain/'output_manifest.json')['sha256'])
    for index, (rel, expected) in enumerate(manifest.items()):
        assert sha(Path(rel) if rel.startswith('/') else ROOT/rel) == expected, rel
        if index % 30 == 0:
            print('hashes', index+1, '/', len(manifest), flush=True)
    checked = dict(manifest_files=len(manifest), rank_arrays=0,
                   feature_arrays=0, independent_sorted_rows=0)
    results = {}
    for domain in ('Toys', 'Beauty'):
        folder = RUN/domain
        seen = []
        results[domain] = {}
        for part in ('fit', 'old512', 'wide4096'):
            samples = read(folder/(part+'_samples.json'))
            meta = read(folder/(part+'_metadata.json'))
            summary = read(folder/(part+'_summary.json'))
            ranks = dict(np.load(folder/(part+'_ranks.npz')))
            measured = dict(np.load(folder/(part+'_measurements.npz')))
            users = [s['user_id'] for s in samples]
            assert users == meta['users_order'] == ranks.pop('users').tolist()
            assert len(users) == len(set(users)) == summary['users']
            assert all(s['target'] not in s['full_history'] for s in samples)
            assert all(s['history'] == s['full_history'][-20:] for s in samples)
            seen.append(set(users))
            base = np.load((SOURCE if part == 'fit' else FULL)/domain/'experts.npy', mmap_mode='r')
            ix = np.asarray(meta['source_indices'])
            files = folder/'features'/part
            banks = {}
            for name in ranks:
                if name in ('SASRec', 'ItemCF', 'BGE'):
                    banks[name] = (base, {'SASRec': 0, 'ItemCF': 1, 'BGE': 2}[name])
                elif part == 'wide4096' or '_' in name:
                    banks[name] = (np.load(files/(name+'.npy'), mmap_mode='r'), None)
                else:
                    root, variant, channel = (TIME, 'tisas', 0) if name == 'TiSASRec' else (QWEN, 'qwen4', 1)
                    path = root/domain/'features'/variant/('train' if part == 'fit' else 'validation')/'item.npy'
                    banks[name] = (np.load(path, mmap_mode='r'), channel)
            n = len(samples)
            # Target-free deterministic row subsample for full-sort tie checking.
            selected = set(np.linspace(0, n-1, 12, dtype=int).tolist())
            for name, rank in ranks.items():
                assert rank.shape == (n,) and (rank > 0).all()
                for k in (5, 10, 20, 50):
                    for metric, value in ((f'H{k}', (rank <= k).mean()),
                                          (f'N{k}', np.where(rank <= k, 1/np.log2(rank+1), 0.).mean())):
                        assert abs(value-summary['experts'][name]['metrics'][metric]) < 1e-12
                array, channel = banks[name]
                for j, sample in enumerate(samples):
                    scores = array[ix[j], channel] if name in ('SASRec', 'ItemCF', 'BGE') else (
                        array[j] if channel is None else array[j, :, channel])
                    assert np.isfinite(scores).all()
                    target = sample['target']-1
                    expected = 1+np.count_nonzero(scores > scores[target])+np.count_nonzero(scores[:target] == scores[target])
                    assert rank[j] == expected, (domain, part, name, j)
                    if j in selected:
                        order = np.lexsort((np.arange(len(scores)), -scores))
                        assert rank[j] == np.flatnonzero(order == target)[0]+1
                        historical = np.asarray(sample['full_history'])-1
                        assert measured[name][j, 0] == int(order[0] in historical)
                        assert measured[name][j, 1] == np.isin(order[:10], historical).sum()
                        checked['independent_sorted_rows'] += 1
                checked['rank_arrays'] += 1
            for path in files.glob('*_features6.npy'):
                value = np.load(path)
                assert value.shape == (n, 6) and np.isfinite(value).all()
                np.testing.assert_allclose(value[:, 0], np.log1p([len(s['full_history']) for s in samples]), atol=3e-7)
                behavior, text = path.stem[:-len('_features6')].split('__')
                np.testing.assert_allclose(value[:, 1], measured[behavior][:, 2], atol=3e-8)
                np.testing.assert_allclose(value[:, 2], measured[text][:, 2], atol=3e-8)
                assert (value[:, 3:] >= -1e-6).all() and (value[:, 3:] <= 1+1e-6).all()
                checked['feature_arrays'] += 1
            comparisons = {a+'_vs_'+b: contrast(ranks[a], ranks[b]) for a, b in PAIRS}
            # Oracle hit coverage describes possible complementarity, not a learned method.
            coverage = {}
            for k in (10, 50):
                sas = ranks['SASRec'] <= k
                for name in ('BGE_loo', 'Qwen4_loo', 'ItemCF', 'TiSASRec'):
                    hit = ranks[name] <= k
                    coverage[f'SASRec__{name}__{k}'] = dict(
                        base_hits=int(sas.sum()), other_hits=int(hit.sum()),
                        unique_other_hits=int((hit & ~sas).sum()),
                        oracle_union_hits=int((hit | sas).sum()), users=n)
            results[domain][part] = dict(users=n, comparisons=comparisons,
                oracle_hit_coverage_not_model_performance=coverage)
            print(domain, part, json.dumps(comparisons), flush=True)
        assert all(seen[a].isdisjoint(seen[b]) for a in range(3) for b in range(a))
    stats = dict(coverage='11/11', confidence='Exploratory validation; not independent test',
        simpson='Both domains and each cohort are reported separately; Toys SAS/TiSA direction reverses between old512 and wide4096.',
        ecological='No claim of universal per-user improvement; paired gain/loss counts retained.',
        berkson='4096 are hash-selected without targets from an explicitly restricted validation population.',
        collider='No hit-, score-, or target-dependent sample selection.',
        base_rate='All prescribed users included; processed targets have zero repeats in visible history.',
        regression_to_mean='Frozen models; no promise that old512 rankings generalize.',
        survivorship='Both successful domains, all three cohorts and all nine channels retained; V1 failure preserved.',
        look_elsewhere='Seven paired comparisons per cohort/domain, 42 total; bootstrap intervals descriptive and unadjusted.',
        forking_paths='Exploration with prior validation exposure; not preregistered confirmation.',
        causation='LOO and uniform both suppress history support; direct rank gains do not prove superior expert evidence or GRAM generation.',
        reverse_causality='Features use visible history and expert scores; target used only to measure rank.')
    write(OUT/'summary.json', dict(domains=results, checks=checked, statistical_review=stats,
        test_read=False, fusion_training=False, new_generation=False))
    write(OUT/'input_hashes.json', dict(pinned_inputs_and_outputs=manifest,
        review_source_sha256=sha(Path(__file__))))
    lines = ['# 专家重选结果独立复核与4096样本解释', '', '## Material Passport', '',
        '- Origin Skill: academic-research-suite / experiment-agent', '- Origin Mode: validate',
        '- Origin Date: 2026-10-16', '- Verification Status: ANALYZED；已有输出复算，无重新训练或生成',
        '- Version Label: expert_reselection_review_v1', '',
        '原512用户保留为旧开发回放；每域额外4096用户用于扩大开发诊断。两组不重叠，4096排除早期融合拟合和校准用户。不是增加训练样本；Toys拟合仍2773事件，Beauty仍3310事件。',
        '4096仍来自已经看过总体结果的validation，不是独立test。该子集的结果不能直接与旧512的绝对指标比较。', '',
        '## 4096人专家直接全目录排序，不过滤历史', '',
        '| 专家/评分 | Toys NDCG@10 | Beauty NDCG@10 |', '|---|---:|---:|']
    for name in ('SASRec', 'TiSASRec', 'ItemCF', 'BGE', 'BGE_loo', 'BGE_uniform', 'Qwen4', 'Qwen4_loo', 'Qwen4_uniform'):
        values = [read(RUN/d/'wide4096_summary.json')['experts'][name]['metrics']['N10'] for d in ('Toys', 'Beauty')]
        lines.append(f'| {name} | {values[0]:.6f} | {values[1]:.6f} |')
    lines += ['', '## 4096人配对差值', '',
        '| 数据集 | 比较 | N10绝对差 | 描述性95%区间 | 相对变化 |', '|---|---|---:|---|---:|']
    for domain in ('Toys', 'Beauty'):
        for pair, value in results[domain]['wide4096']['comparisons'].items():
            ci = value['descriptive_95pct_N10']
            lines.append(f'| {domain} | {pair} | {value["delta_N10"]:+.6f} | [{ci[0]:+.6f}, {ci[1]:+.6f}] | {value["relative_N10_percent"]:+.2f}% |')
    lines += ['', 'SASRec在4096人两域的直接排序均值高于TiSASRec，但不能据此证明它在最终融合引导中必胜。Toys旧512呈反方向，因此不能仅依赖旧512选择行为专家。',
        'BGE和Qwen去自匹配都降低了历史偏重，但等历史概率质量的统一降分得到接近结果，尚无去自匹配的额外质量优势证据。非历史候选原分数和相对顺序均未被改善。',
        'BGE_loo的单专家排序均值在两域高于Qwen4_loo。仍需保留组合验证：单专家强弱不能替代内容专家相对行为/GRAM的增量价值；并集命中是知道答案后计算的上界，不是可实现模型成绩。',
        '当前融合器没有训练，六维与原十维/七维的优劣没有实验结论。历史倾向只是一个特征组，目标仍是下一商品预测。', '',
        f'核验：{checked["manifest_files"]}份输入/输出哈希、{checked["rank_arrays"]}组完整逐用户排名及指标、{checked["independent_sorted_rows"]}行独立全排序、{checked["feature_arrays"]}个特征缓存接口与历史质量对应关系全部通过。',
        '统计解释覆盖11/11；42项配对比较均保存，2000次用户bootstrap，seed20261016。区间未经多重比较校正，只作当前固定模型的描述；不包含训练种子或模型选择不确定性。']
    REPORT.write_text('\n'.join(lines)+'\n')
    write(OUT/'status.json', dict(state='COMPLETED', elapsed_seconds=time.monotonic()-start,
        test_read=False, checks=checked, report=str(REPORT.relative_to(ROOT))))
    print('COMPLETED', checked, flush=True)


if __name__ == '__main__':
    main()
