"""Read-only saved-output audit and descriptive interpretation for stage 21.22.

No fitting, validation-based head selection, new generation, or test access.
Review artifacts and a report appendix are the only outputs.
"""
import json
from pathlib import Path
import time

import numpy as np

from experiment.phase21 import dual_candidate_gate as D

OUT = D.OUT/'review'


def metrics_per_user(ranks):
    # Independent metric definition, without the experiment's utility helper.
    return np.column_stack([np.where(ranks <= 10, 1./np.log2(ranks+1), 0.),
                            ranks <= 10, ranks <= 50])


def signal_check(domain, samples, rr):
    gen = D.read(D.ROOT/'artifacts/phase21/candidate_gate_generation_20261010_v1'/domain/'samples.json')
    by_user = {s['user_id']: i for i, s in enumerate(samples)}
    assert len(by_user) == len(samples)
    selected = [by_user[s['user_id']] for s in gen]
    for s, k in zip(gen, selected):
        assert s['target'] == samples[k]['target'] and s['history'] == samples[k]['history']
    result = {}
    for arm in D.ARMS:
        scores = np.load(D.OUT/domain/f'{arm}_generation_signals.npy', mmap_mode='r')
        assert len(scores) == len(gen)
        found = []; sort_checks = 0
        for start in range(0, len(gen), 64):
            ss = gen[start:start+64]
            f = np.array(scores[start:start+64], copy=True)
            assert np.isfinite(f).all()
            y = np.array([s['target']-1 for s in ss])
            for j, s in enumerate(ss):
                f[j, np.asarray(s['history'])-1] = -np.inf
            t = f[np.arange(len(ss)), y]
            ranks = 1+(f > t[:, None]).sum(1)+((f == t[:, None]) & (np.arange(f.shape[1]) < y[:, None])).sum(1)
            ranks[~np.isfinite(t)] = f.shape[1]+1
            if start % 512 == 0:
                for j in (0, len(ss)-1):
                    ids = np.flatnonzero(np.isfinite(f[j]))
                    order = ids[np.lexsort((ids, -f[j, ids]))]
                    where = np.flatnonzero(order == y[j])
                    expected = int(where[0])+1 if len(where) else f.shape[1]+1
                    assert expected == ranks[j]
                    sort_checks += 1
            found.extend(ranks.tolist())
        np.testing.assert_array_equal(found, rr[f'{arm}_0'][selected])
        result[arm] = dict(exported_users=len(gen), target_ranks_matched=len(found), independent_sorts=sort_checks)
    return result


def review_domain(domain):
    summary = D.read(D.OUT/domain/'summary.json')
    rows = D.read(D.OUT/domain/'validation/users.json')
    samples = D.read(D.FULL/domain/'samples.json')
    assert len(rows) == len(samples)
    for i, (row, sample) in enumerate(zip(rows, samples)):
        assert row['source_index'] == i and row['user_id'] == sample['user_id'] and row['target'] == sample['target']-1
    rr = np.load(D.OUT/domain/'ranks.npz')
    expected_keys = {'baseline'} | {f'{arm}_{seed}' for arm in D.ARMS for seed in D.CFG['seeds']}
    assert set(rr.files) == expected_keys
    metric_checks = 0; results = {}
    for name in ('primary', 'previous512', 'all'):
        mask = np.ones(len(rows), bool) if name == 'all' else np.array([r['cohort'] == name for r in rows])
        assert int(mask.sum()) == summary['results'][name]['users']
        for key in expected_keys:
            assert len(rr[key]) == len(rows)
            calculated = metrics_per_user(rr[key][mask]).mean(0)
            saved = summary['results'][name]['metrics'][key]
            np.testing.assert_allclose(calculated, [saved[m] for m in ('N10', 'H10', 'H50')], atol=2e-14, rtol=0)
            metric_checks += 3
        arms = {'baseline': metrics_per_user(rr['baseline'][mask])}
        for arm in D.ARMS:
            arms[arm] = np.mean([metrics_per_user(rr[f'{arm}_{seed}'][mask]) for seed in D.CFG['seeds']], axis=0)
        means = {a: dict(zip(('N10', 'H10', 'H50'), x.mean(0).tolist())) for a, x in arms.items()}
        result = dict(users=int(mask.sum()), means=means)
        if name == 'primary':
            contrasts = [('candidate', b) for b in ('user', 'scalar', 'baseline')]
            diff = np.stack([arms[a]-arms[b] for a, b in contrasts], axis=1)
            rng = np.random.default_rng(20261014); boots = []
            for _ in range(40):
                ix = rng.integers(0, len(diff), size=(50, len(diff)))
                boots.append(diff[ix].mean(1))
            ci = np.quantile(np.concatenate(boots), [.025, .975], axis=0)
            comparisons = {}
            for j, (a, b) in enumerate(contrasts):
                by_metric = {}
                for m, metric in enumerate(('N10', 'H10', 'H50')):
                    seed_changes = []
                    for seed in D.CFG['seeds']:
                        ra = metrics_per_user(rr[f'{a}_{seed}'][mask])[:, m]
                        rb = metrics_per_user(rr['baseline' if b == 'baseline' else f'{b}_{seed}'][mask])[:, m]
                        seed_changes.append(float((ra-rb).mean()))
                    by_metric[metric] = dict(delta=float(diff[:, j, m].mean()),
                        relative_pct=100*(means[a][metric]/means[b][metric]-1),
                        ci95_descriptive=ci[:, j, m].tolist(), seed_deltas=seed_changes)
                old = summary['results'][name]['comparisons'][f'{a}_vs_{b}']
                np.testing.assert_allclose(by_metric['N10']['ci95_descriptive'], old['ci95_descriptive'], atol=2e-14, rtol=0)
                comparisons[f'{a}_vs_{b}'] = by_metric
            result['comparisons'] = comparisons
        results[name] = result
    return dict(results=results, aggregate_metrics_recomputed=metric_checks,
                generation_signal_checks=signal_check(domain, samples, rr))


def main():
    begin = time.monotonic()
    OUT.mkdir(exist_ok=False)
    D.write(OUT/'status.json', dict(state='RUNNING', updated_at=D.stamp()))
    assert D.read(D.OUT/'status.json')['state'] == 'COMPLETED'
    manifest_checks = 0
    for path in [D.OUT/'manifest.json', *[D.OUT/d/'output_manifest.json' for d in D.DOMAINS]]:
        for rel, digest in D.read(path)['sha256'].items():
            assert D.sha(D.ROOT/rel) == digest, rel
            manifest_checks += 1
    domains = {d: review_domain(d) for d in D.DOMAINS}
    D.write(OUT/'diagnostic.json', dict(domains=domains, bootstrap='paired users; average head-seed utilities before resampling; descriptive only'))
    checks = [
        ('simpson', 'CAUTION', 'Report domains and cohorts separately. Beauty candidate-minus-scalar N10 changes sign between primary and all; retain predeclared primary pool, do not choose the favorable denominator.'),
        ('ecological', 'CHECKED', 'Mean benefit does not mean every user benefits; retain per-seed top10 gains/losses.'),
        ('berkson', 'CAUTION', 'Primary excludes previous512 and fit-role users by fixed membership; generalization beyond this selected, repeatedly used validation cohort is unproven.'),
        ('collider', 'CHECKED', 'No restriction to target-in-candidates or outcome-selected responders; all planned rows retained.'),
        ('base_rate', 'CHECKED', 'Report cohort sizes, absolute N10/H10/H50, absolute differences and relative differences.'),
        ('regression_to_mean', 'CAUTION', 'Same-user matched controls; configuration informed by past exploration, so gains are not fresh confirmation.'),
        ('survivorship', 'CHECKED', 'Both domains and all three seeds of all three arms completed; no dropped evaluation rows.'),
        ('look_elsewhere', 'CAUTION', 'Multiple exploratory comparisons without multiplicity correction; no confirmatory significance claim. H50 is a predeclared secondary metric, not a replacement for the weak primary N10 contrast.'),
        ('forking_paths', 'CAUTION', 'Protocol and code hashes match the launch; past three-expert experiments and failed compact heads informed this design and remain part of the record.'),
        ('causation', 'CHECKED', 'Matched computational contrast supports this configuration only; H50 ranking coverage is not proof of improved generation or causal user behavior.'),
        ('reverse_causality', 'CHECKED', 'Features are score/history based; targets are used for fit losses/evaluation, not candidate construction. No behavioral causal direction inferred.')]
    D.write(OUT/'interpretation_checks.json', dict(coverage='11/11', checks=[dict(name=n, status=s, note=t) for n, s, t in checks]))
    verification = dict(state='VERIFIED_SAVED_OUTPUTS', hashes_checked=manifest_checks,
                        aggregate_metrics_recomputed=sum(v['aggregate_metrics_recomputed'] for v in domains.values()),
                        signal_rows_checked=sum(c['target_ranks_matched'] for v in domains.values() for c in v['generation_signal_checks'].values()),
                        training_rerun=False, generation_rerun=False, test_read=False,
                        elapsed_seconds=time.monotonic()-begin, updated_at=D.stamp())
    D.write(OUT/'verification.json', verification)
    lines = ['','## 完成后复核与判断','',
             'Verification Status: VERIFIED_SAVED_OUTPUTS。只重算已保存输出；未重新训练或生成，未读取test。总后台耗时225.30秒，约3分45秒。两域小头与输出均完整。','',
             '主问题：候选权重相对用户级权重的NDCG@10额外收益较小（Toys +0.96%，Beauty +0.31%），两个描述区间均跨0，Beauty有一个小头seed为负。因此目前不足以认定候选粒度带来稳定的前十排序优势。候选权重相对冻结双专家基线为 +2.23%/+7.18%，但用户级调整同样有收益，不能把全部提升归于候选粒度。','',
             '直接分数修正的主池NDCG@10均值在两域略高，候选减去分数修正的区间也跨0；没有证明候选权重更好，也没有证明二者等效或分数修正稳定更好。','',
             '| 域 | 用户级 H50 | 候选级 H50 | 分数修正 H50 | 候选−用户（描述95%区间） | 候选−分数（描述95%区间） |',
             '|---|---:|---:|---:|---|---|']
    for domain, r in domains.items():
        p = r['results']['primary']; m = p['means']; comps = p['comparisons']
        def show(key):
            c = comps[key]['H50']; lo, hi = c['ci95_descriptive']
            return f'{c["delta"]:+.6f} [{lo:+.6f}, {hi:+.6f}]'
        lines.append(f'| {domain} | {m["user"]["H50"]:.6f} | {m["candidate"]["H50"]:.6f} | {m["scalar"]["H50"]:.6f} | {show("candidate_vs_user")} | {show("candidate_vs_scalar")} |')
    lines += ['',
              'H50是预先列出的次要指标；这是专家融合分布的前50覆盖，不能当作GRAM束搜索的覆盖。它提供了继续小规模真实生成验证的理由，不抵消主N10比较证据偏弱的结论。建议下一步只用一个预先固定小头种子、同批512用户，比较用户级/候选级/分数修正三组真实生成，并复用同批GRAM与原双专家参照；不再先扩大调参或完整论文实验。若真实生成仍不能超过用户级和分数对照，应降低候选权重作为主机制的优先级。','',
              '本次未启动后续生成。没有按验证结果选择最好seed；保持seed0作为后续若开展的预设种子，并明确其Beauty候选−用户N10在本轮为负。统计解释扫描11/11，详见 review/interpretation_checks.json；全部指标与分母详见 review/diagnostic.json。']
    marker='\n## 完成后复核与判断\n'
    old = D.REPORT.read_text()
    assert marker not in old
    D.REPORT.write_text(old+'\n'.join(lines)+'\n')
    files=[Path(__file__),D.REPORT,*[OUT/n for n in ('diagnostic.json','verification.json','interpretation_checks.json')]]
    D.write(OUT/'manifest.json', dict(sha256={str(p.relative_to(D.ROOT)):D.sha(p) for p in files}))
    D.write(OUT/'status.json',dict(state='COMPLETED',updated_at=D.stamp(),elapsed_seconds=time.monotonic()-begin))
    print(json.dumps(dict(verification=verification, primary={d:v['results']['primary'] for d,v in domains.items()}),ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
