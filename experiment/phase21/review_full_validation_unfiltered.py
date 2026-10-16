"""Re-score the original full three-expert validation without history masking.

Reuse saved beam50 candidates/scores. Keep the old fuser's learned history
penalty intact; remove only the final evaluation mask. No fitting or generation.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from experiment.phase20.catalog_scope_core import metrics, contrast

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'artifacts/phase21/full_validation_20260924_v1'
OUT = ROOT/'artifacts/phase21/full_validation_unfiltered_review_20261016_v1'
REPORT = ROOT/'report/第二十一阶段/原三专家全量验证取消历史过滤复核_20261016.md'
COUNTS = dict(Toys=19412, Beauty=22363)
FIELDS = dict(GRAM='native', late='late', guided='guided')


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def describe(ranks, mask, intervals):
    selected = {key:values[mask] for key, values in ranks.items()}
    measured = {key:metrics(values) for key, values in selected.items()}
    pairs = [('guided', 'GRAM'), ('late', 'GRAM'), ('guided', 'late')]
    return dict(users=int(mask.sum()), metrics=measured,
        relative_N10_pct={a+'_vs_'+b:100*(measured[a]['N10']/measured[b]['N10']-1) for a,b in pairs},
        comparisons={a+'_vs_'+b:contrast(selected[a], selected[b], repeats=2000, seed=20260924)
                     for a,b in pairs} if intervals else {})


def main():
    begin = time.monotonic()
    OUT.mkdir(exist_ok=False)
    write(OUT/'status.json', dict(state='RUNNING', stage='saved_full_validation_rescore', test_read=False))
    assert read(RUN/'status.json')['state'] == 'COMPLETED'
    inputs, results = {}, {}
    for path in (RUN/'config.json', RUN/'manifest.json',
                 ROOT/'experiment/phase21/full_validation.py',
                 ROOT/'plan/第二十一阶段/21.5_两域全量验证_20260924.md', Path(__file__)):
        inputs[str(path.relative_to(ROOT))] = sha(path)
    for domain, count in COUNTS.items():
        folder = RUN/domain
        for name in ('samples.json', 'predictions.jsonl', 'ranks.npz', 'summary.json', 'status.json'):
            path = folder/name; inputs[str(path.relative_to(ROOT))] = sha(path)
        assert read(folder/'status.json')['state'] == 'COMPLETED'
        samples, old = read(folder/'samples.json'), read(folder/'summary.json')
        archive = np.load(folder/'ranks.npz')
        assert len(samples) == len({s['user_id'] for s in samples}) == count
        np.testing.assert_array_equal(archive['users'], [s['user_id'] for s in samples])
        filtered = {key:np.empty(count, np.int64) for key in FIELDS}
        unfiltered = {key:np.empty(count, np.int64) for key in FIELDS}
        historical10 = dict.fromkeys(FIELDS, 0)
        target_history = 0
        actual_count = 0
        with (folder/'predictions.jsonl').open() as stream:
            for j, line in enumerate(stream):
                assert j < count
                row, sample = json.loads(line), samples[j]
                assert row['sample_index'] == j
                assert all(row[k] == sample[k] for k in ('user_id', 'target', 'history'))
                history, target = set(sample['history']), sample['target']
                target_history += int(target in history)
                assert row['late']['ids'] == row['native']['ids']
                for key, field in FIELDS.items():
                    value = row[field]
                    ids, scores = np.asarray(value['ids']), np.asarray(value['scores'])
                    assert len(ids) == len(set(ids.tolist())) == len(scores) == 50
                    assert np.isfinite(scores).all()
                    order = np.lexsort((ids, -scores))
                    ordered = ids[order].tolist()
                    historical10[key] += sum(item in history for item in ordered[:10])
                    position = ordered.index(target)+1 if target in ordered else 2**30
                    unfiltered[key][j] = position
                    # Independent rank-by-count with score/ID tie rule.
                    if position != 2**30:
                        target_score = scores[np.flatnonzero(ids == target)[0]]
                        counted = 1+int(((scores > target_score) | ((scores == target_score) & (ids < target))).sum())
                        assert position == counted
                    kept = [item for item in ordered if item not in history]
                    filtered[key][j] = kept.index(target)+1 if target in kept else 2**30
                actual_count += 1
        assert actual_count == count
        for key in FIELDS:
            np.testing.assert_array_equal(filtered[key], archive[key])
            for label, value in metrics(filtered[key]).items():
                assert abs(value-old['metrics'][key][label]) < 1e-12
        existing = np.asarray([s['previously_evaluated'] for s in samples], bool)
        assert int(existing.sum()) == 512
        np.testing.assert_array_equal(existing, archive['previously_evaluated'])
        all_users = np.ones(count, bool)
        result = dict(users=count, filtered=describe(filtered, all_users, False),
            unfiltered=describe(unfiltered, all_users, True),
            subgroups=dict(previous512=describe(unfiltered, existing, False),
                           remaining=describe(unfiltered, ~existing, True)),
            target_in_recent_history=target_history,
            historical_items_in_unfiltered_top10_total=historical10,
            checks=dict(all_users_preserved=True, historical_filtered_rank_arrays_exact=True,
                filtered_metrics_reproduced=True, independent_target_rank_count_verified=True,
                same_native_candidates_for_late=True),
            original_generation=dict(generated_users=old['generated_users'], reused_users=old['reused_users']))
        np.savez_compressed(OUT/(domain+'_ranks.npz'), users=archive['users'], previously_evaluated=existing,
            **{k+'_filtered':v for k,v in filtered.items()}, **{k+'_unfiltered':v for k,v in unfiltered.items()})
        results[domain] = result
        write(OUT/(domain+'_summary.json'), result)
        write(OUT/'status.json', dict(state='RUNNING', stage='domain_completed', completed_domains=list(results),
            updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'), test_read=False))
        print(domain, json.dumps(result['unfiltered']), flush=True)
    for rel, expected in inputs.items():
        assert sha(ROOT/rel) == expected, ('Source changed during audit', rel)
    write(OUT/'input_hashes.json', inputs)
    write(OUT/'summary.json', dict(state='COMPLETED', domains=results, test_read=False,
        new_training=False, new_generation=False, elapsed_seconds=time.monotonic()-begin,
        experts=['SASRec', 'ItemCF', 'BGE'], alpha=.3, beam_size=50,
        removed='Final evaluation exclusion of recent20 history only',
        retained='Original learned history penalty in three-expert fusion, expert weights, all model parameters/candidate scores',
        statistical_review=dict(coverage='11/11', confidence='CAUTION: exploratory validation',
            simpson='Both domains and previous512/remaining separately reported.',
            ecological='Aggregate N10 does not imply every-user benefit.',
            berkson='Full processed validation; not raw population or test.',
            collider='No outcome-based user selection.',
            base_rate='All19412/22363 retained; target-in-history explicitly counted.',
            regression_to_mean='Exact same saved predictions under both rules, no retraining.',
            survivorship='All originally prescribed users and three methods included.',
            look_elsewhere='Three paired N10/H10 comparisons per domain/pool; uncorrected descriptive intervals.',
            forking_paths='Post-discovery evaluation sensitivity analysis; not preregistered confirmation.',
            causation='Only ranking mask changed. Cannot infer effects of removing the learned score penalty or restoring ItemCF to the new model.',
            reverse_causality='Mask uses visible history only; no future target injected into model.')))
    lines = ['# 原三专家全量验证取消历史过滤复核', '', '## Material Passport', '',
        '- Origin Skill: academic-research-suite / experiment-agent', '- Origin Mode: validate',
        '- Origin Date: 2026-10-16', '- Verification Status: ANALYZED；原过滤排名与指标逐项复现，未重跑训练或生成',
        '- Version Label: full_validation_unfiltered_review_v1', '',
        '复核对象：2026-09-24的21.5全量验证。SASRec＋ItemCF＋BGE、原融合器、alpha=.3、beam50。Toys19412、Beauty22363，共41775名验证用户。',
        '仅取消最终排序时排除最近20历史的操作。原融合器对完整历史商品的已学习惩罚仍保留；这不等于去掉所有历史校准。', '',
        '| 数据集 | 方法 | 原过滤N10 | 不过滤N10 | 不过滤时相对GRAM |',
        '|---|---|---:|---:|---:|']
    for domain, result in results.items():
        original, current = result['filtered']['metrics'], result['unfiltered']['metrics']
        for key, label in (('GRAM', 'GRAM'), ('late', '三专家后重排'), ('guided', '三专家生成引导')):
            relative = 100*(current[key]['N10']/current['GRAM']['N10']-1)
            lines.append(f'| {domain} | {label} | {original[key]["N10"]:.6f} | {current[key]["N10"]:.6f} | {relative:+.2f}% |')
    lines += ['', '| 数据集 | 比较 | 不过滤N10绝对差 | 配对描述性95%区间 | Top10新增/丢失人数 |',
        '|---|---|---:|---|---|']
    for domain, result in results.items():
        for pair, value in result['unfiltered']['comparisons'].items():
            ci = value['descriptive_95pct_N10']
            lines.append(f'| {domain} | {pair} | {value["delta_N10"]:+.7f} | [{ci[0]:+.7f}, {ci[1]:+.7f}] | {value["top10_gained"]}/{value["top10_lost"]} |')
    lines += ['', '| 数据集/用户组 | 人数 | GRAM N10 | 三专家后重排N10 | 三专家引导N10 | 引导对GRAM |',
        '|---|---:|---:|---:|---:|---:|']
    for domain, result in results.items():
        for group, values in result['subgroups'].items():
            m = values['metrics']; relative = values['relative_N10_pct']['guided_vs_GRAM']
            lines.append(f'| {domain}/{group} | {values["users"]} | {m["GRAM"]["N10"]:.6f} | {m["late"]["N10"]:.6f} | {m["guided"]["N10"]:.6f} | {relative:+.2f}% |')
    lines += ['', '全量与512人的基准数值不同，不能直接拿不同用户集的绝对N10比较模型。较大样本降低了固定模型下的用户抽样不确定性，不消除单次训练或研究选择的不确定性。',
        '这轮使用完整validation，含先前融合拟合用户的后续事件，不能称作独立test或全部用户外泛化。原512以外的用户也不自动成为从未参与研究的确认集。',
        '统计解释覆盖11/11，区间为固定训练模型下的描述性配对bootstrap，2000次、seed20260924，未经多重比较校正。没有新训练或GPU推理。',
        '完整H/N@5/10/20/50、分组与全部排名见同名artifacts目录。不能据旧三专家的整体结果直接证明ItemCF在当前TiSASRec双专家结构中的增量必要性。']
    REPORT.write_text('\n'.join(lines)+'\n')
    write(OUT/'status.json', dict(state='COMPLETED', elapsed_seconds=time.monotonic()-begin,
        test_read=False, new_training=False, new_generation=False, report=str(REPORT.relative_to(ROOT))))
    print('COMPLETED', str(REPORT), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if OUT.exists():
            write(OUT/'status.json', dict(state='FAILED', error=repr(exc), test_read=False))
        raise
