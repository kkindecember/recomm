"""Read completed LIME seed outputs for descriptive coverage/ranking diagnostics.

No new model fitting, parameter selection, or test-set access. Post-hoc findings
are development evidence; oracle unions below are not deployable predictions.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'artifacts/phase20/lime_comparison_20260920_v1/recovery_v2'


def ndcg(ranks):
    return np.where(ranks <= 10, 1 / np.log2(ranks + 1), 0.)


def quadrants(first, second):
    return dict(both=int((first & second).sum()), first_only=int((first & ~second).sum()),
                second_only=int((~first & second).sum()), neither=int((~first & ~second).sum()))


def union_control(domain, saved, labels, rr):
    from experiment.phase20.catalog_scope_core import ranks
    old = ROOT/f'artifacts/phase20/catalog_scope_20260919_v1/{domain}'
    w = json.loads((old/'selection.json').read_text())['primary_weight']
    directory = np.load(old/'directory_z.npy', mmap_mode='r')
    collaborative = np.load(old/'collaborative.npy', mmap_mode='r')
    rows = saved['indices'][:, None]
    targets = saved['targets']
    g = saved['top50'][labels.index('common_G50')]
    lime = saved['top50'][labels.index('common_LIME50')]
    def score(ids):
        values = (1-w)*np.asarray(directory[rows, ids], dtype=np.float64)+w*np.asarray(collaborative[rows, ids], dtype=np.float64)
        return np.where(ids > 0, values, -np.inf)
    # Verify the very same scoring rule before examining a larger candidate set.
    np.testing.assert_array_equal(ranks(score(g), g, targets), rr['common_G50'])
    np.testing.assert_array_equal(ranks(score(lime), lime, targets), rr['common_LIME50'])
    ids = np.sort(np.concatenate([g, lime], axis=1), axis=1)
    duplicate = np.concatenate([np.zeros((len(ids), 1), dtype=bool), ids[:, 1:] == ids[:, :-1]], axis=1)
    ids[duplicate] = 0
    scores = score(ids)
    union_ranks = ranks(scores, ids, targets)
    # Independent complete sorting of the at-most-100 unique items.
    order = np.lexsort((ids, -scores), axis=1)
    hit = np.take_along_axis(ids, order, axis=1) == targets[:, None]
    checked = np.where(hit.any(1), hit.argmax(1)+1, 2**30)
    np.testing.assert_array_equal(union_ranks, checked)
    covered = rr['common_G50'] <= 50
    assert np.all(union_ranks[covered] >= rr['common_G50'][covered])
    delta = ndcg(union_ranks)-ndcg(rr['common_G50'])
    return dict(N10=float(ndcg(union_ranks).mean()), H10=float((union_ranks <= 10).mean()),
        H50=float((union_ranks <= 50).mean()), mean_candidate_count=float((ids > 0).sum(1).mean()),
        N10_delta_vs_G50=float(delta.mean()),
        existing_covered_contribution=float(delta[covered].sum()/len(delta)),
        newly_covered_contribution=float(delta[~covered].sum()/len(delta)),
        gained_top10=int(((union_ranks <= 10) & (rr['common_G50'] > 10)).sum()),
        lost_top10=int(((union_ranks > 10) & (rr['common_G50'] <= 10)).sum()),
        changed_scoring=False, parameter_search=False, new_training_steps=0)


def run(out, include_union=False):
    records, inputs = [], {}
    for domain in ['Toys', 'Beauty']:
        for summary_path in sorted((RUN/domain).glob('summary_seed*_crossfit.json')):
            summary = json.loads(summary_path.read_text())
            seed = summary['seed']
            path = RUN/domain/f'evaluation_seed{seed}_crossfit.npz'
            with np.load(path, allow_pickle=False) as saved:
                labels = saved['arm_names'].tolist()
                rr = dict(zip(labels, saved['ranks']))
                targets = saved['targets']
                top = saved['top50']
                a = saved['A_ranks'][0]
                n = len(a)
                assert n == summary['evaluation_users']
                for j, label in enumerate(labels):
                    hit = top[j] == targets[:, None]
                    rebuilt = np.where(hit.any(1), hit.argmax(1)+1, 2**30)
                    np.testing.assert_array_equal(np.minimum(rebuilt, 51), np.minimum(rr[label], 51))
                    m = summary['metrics'][label]['all']
                    assert abs(ndcg(rr[label]).mean()-m['N10']) < 1e-12
                    assert abs((rr[label] <= 10).mean()-m['H10']) < 1e-12
                    assert abs((rr[label] <= 50).mean()-m['H50']) < 1e-12
                g = rr['common_G50'] <= 50
                lime = rr['LIME']
                restricted = rr['LIME_scores_G50']
                # Restriction removes competitors, without changing shared scores.
                assert np.all(restricted[g] <= lime[g])
                delta = ndcg(restricted)-ndcg(lime)
                assert abs(delta.mean()-(delta[g].sum()+delta[~g].sum())/n) < 1e-12
                metrics = {label: {k: summary['metrics'][label]['all'][k] for k in ['N10', 'H10', 'H50']}
                           for label in labels}
                metrics['A_primary'] = dict(N10=float(ndcg(a).mean()), H10=float((a <= 10).mean()),
                                            H50=float((a <= 50).mean()))
                records.append(dict(domain=domain, seed=seed, users=n, metrics=metrics,
                    A_vs_LIME_top10=quadrants(a <= 10, lime <= 10),
                    G50_vs_LIME50_target_coverage=quadrants(g, lime <= 50),
                    target_coverage_union_oracle=float((g | (lime <= 50)).mean()),
                    restricted_minus_full=dict(N10=float(delta.mean()),
                        covered_contribution=float(delta[g].sum()/n),
                        excluded_contribution=float(delta[~g].sum()/n),
                        gained_top10=int(((restricted <= 10) & (lime > 10)).sum()),
                        lost_top10=int(((restricted > 10) & (lime <= 10)).sum())),
                    A_vs_LIME_scored_G50_top10=quadrants(a <= 10, restricted <= 10)))
                if include_union:
                    records[-1]['fixed_common_score_union'] = union_control(domain, saved, labels, rr)
            for p in [summary_path, path]:
                inputs[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    complete = len(records) == 6 and json.loads((RUN/'status.json').read_text())['state'] == 'COMPLETED'
    data = dict(created_at=datetime.now().astimezone().isoformat(timespec='seconds'),
                state='COMPLETE_DESCRIPTIVE_REVIEW' if complete else 'INTERIM_DESCRIPTIVE_REVIEW',
                completed_seed_outputs=len(records), expected_seed_outputs=6,
                new_training_steps=0, test_read=False, input_sha256=inputs, records=records,
                limitations=['Post-hoc analysis on repeatedly used development validation.',
                    'Oracle target unions are diagnostic upper bounds, not achievable gains.',
                    'A and LIME differ in fitting supervision and history length.',
                    'Seeds vary SASRec for LIME; A varies only its small head.',
                    'No new significance, novelty, or causal claims.'])
    out.mkdir(parents=True, exist_ok=False)
    (out/'summary.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    lines = ['# LIME候选互补与排序拆解', '',
        'Origin Skill: academic-research-suite。AI辅助的本地缓存分析，无新训练/test。',
        f'状态：{data["state"]}；已完成种子输出{len(records)}/6。', '',
        '| 域 | seed | A N10 | LIME N10 | LIME评分＋生成50 N10 | 生成覆盖H50 | LIME覆盖H50 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        m = r['metrics']
        lines.append(f'| {r["domain"]} | {r["seed"]} | {m["A_primary"]["N10"]:.6f} | {m["LIME"]["N10"]:.6f} | {m["LIME_scores_G50"]["N10"]:.6f} | {m["common_G50"]["H50"]:.6f} | {m["LIME"]["H50"]:.6f} |')
    lines += ['', '完整逐用户命中得失及同分数候选限制的贡献拆解见summary.json。',
        '该分析是事后开发证据。覆盖并集使用真实目标识别互补，只作诊断上界，不能直接当成新方法可取得的收益。',
        '两套模型监督来源和历史长度不同；本表不证明严格同监督优势或新颖性。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(state=data['state'], outputs=len(records), output=str(out.relative_to(ROOT))), ensure_ascii=False))
    for r in records:
        print(json.dumps({k:r[k] for k in ['domain', 'seed', 'A_vs_LIME_top10',
            'G50_vs_LIME50_target_coverage', 'restricted_minus_full']}, ensure_ascii=False))
        if include_union:
            print(json.dumps(dict(domain=r['domain'], seed=r['seed'], union=r['fixed_common_score_union']), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--include-union', action='store_true')
    args = parser.parse_args()
    run(ROOT/args.output, args.include_union)
