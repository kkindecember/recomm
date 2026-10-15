"""CPU-only frozen expert deletion diagnostics on completed decode caches."""
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / 'artifacts/phase20/lime_decode_screen_20260921_v1/full_parallel_v4'
OUT = ROOT / 'artifacts/phase20/lime_expert_ablation_20260921_v1'
PROTOCOL = ROOT / 'plan/第二十阶段/LIME专家删减_CPU并行诊断_20260921.md'
REPORT = ROOT / 'report/第二十阶段/LIME专家删减_CPU并行诊断_20260921.md'
VARIANTS = OrderedDict([
    ('full', ([0, 1, 2], True, False)),
    ('without_itemcf', ([0, 2], True, False)),
    ('without_text', ([0, 1], True, False)),
    ('without_sasrec', ([1, 2], True, False)),
    ('without_history_penalty', ([0, 1, 2], False, False)),
    ('equal_weights', ([0, 1, 2], True, True)),
    ('sasrec_only', ([0], False, False)),
])
SCOPES = ('catalog', 'GRAM50', 'guided50')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def status(state, **kwargs):
    write(OUT / 'status.json', dict(state=state, updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        pid=os.getpid(), resource='CPU only, 2 threads', test_read=False,
        new_model_training_steps=0, new_decodes=0, automatic_new_experiments=False, **kwargs))


def normalize(raw):
    valid = raw > -1e8
    low = np.where(valid, raw, np.inf).min(axis=1, keepdims=True)
    high = np.where(valid, raw, -np.inf).max(axis=1, keepdims=True)
    empty = ~valid.any(axis=1, keepdims=True)
    low, high = np.where(empty, 0., low), np.where(empty, 0., high)
    return (raw - low) / (high - low + 1e-8)


def blend(experts, weights, penalty, old_history, name):
    selected, calibrate, equal = VARIANTS[name]
    effective = np.zeros(3, dtype=np.float32)
    if equal:
        effective[selected] = np.float32(1 / len(selected))
    elif len(selected) == 3:
        # Keep full-method floating-point arithmetic identical to the parent.
        effective[:] = weights
    else:
        effective[selected] = weights[selected] / np.sum(weights[selected])
    scores = (experts * effective[:, None]).sum(axis=0, dtype=np.float32)
    if calibrate:
        scores[np.asarray(old_history, dtype=np.int64) - 1] -= penalty
    return scores, effective


def metrics(ranks):
    r = np.asarray(ranks)
    return {f'{n}{k}': float(np.mean(r <= k) if n == 'H' else np.mean((r <= k) / np.log2(r + 1)))
            for k in (5, 10, 20, 50) for n in ('N', 'H')}


def run():
    OUT.mkdir(exist_ok=False)
    started = time.monotonic()
    status('PREPARING', stage='freeze_sources')
    inputs = {}
    def use(path):
        path = Path(path)
        rel = str(path.relative_to(ROOT))
        assert '/test/' not in rel and path.name != 'test.jsonl'
        if rel not in inputs:
            status('PREPARING', stage='hash_input', current_file=rel,
                   elapsed_seconds=time.monotonic() - started)
            inputs[rel] = sha(path)
        return path
    for path in (Path(__file__).resolve(), PROTOCOL,
                 ROOT / 'experiment/phase20/verify_lime_expert_ablation.py',
                 ROOT / 'experiment/phase20/verify_lime_mechanism.py',
                 ROOT / 'experiment/phase20/lime_decode_screen_core.py',
                 ROOT / 'experiment/phase20/run_lime_expert_ablation.sh'):
        use(path)
    (OUT / 'protocol.md').write_text(PROTOCOL.read_text())
    config = dict(variants={n: dict(experts=v[0], history_penalty=v[1], equal_weights=v[2])
                            for n, v in VARIANTS.items()}, scopes=SCOPES,
                  evaluation_users=512, score_scale=20., fit_steps=0, test_read=False,
                  scope='Post-hoc analysis on completed 512-user decode evaluations; no current running outcomes read.')
    write(OUT / 'config.json', config)
    use(OUT / 'config.json'); use(OUT / 'protocol.md')
    prepared = {}
    for domain in ('Toys', 'Beauty'):
        folder = OUT / domain; folder.mkdir()
        parent = PARENT / domain
        records = [json.loads(s) for s in use(parent / 'evaluation.jsonl').read_text().splitlines()]
        samples = read(use(parent / 'samples.json'))
        indices = np.asarray([r['sample_index'] for r in records])
        assert indices.tolist() == list(range(131, 643))
        assert all(r['user_id'] == samples[j]['user_id'] for r, j in zip(records, indices))
        query_indices = np.asarray([samples[j]['query_index'] for j in indices])
        recovery = ROOT / f'artifacts/phase20/lime_comparison_20260920_v1/recovery_v2/{domain}'
        source_paths = [recovery / 'sasrec_scores_seed0.npy',
            ROOT / f'artifacts/phase20/lime_comparison_20260920_v1/{domain}/itemcf_scores.npy',
            recovery / 'semantic_scores.npy']
        expert = np.stack([normalize(np.asarray(np.load(use(p), mmap_mode='r')[query_indices]))
                           for p in source_paths], axis=1)
        with np.load(use(parent / 'signals.npz'), allow_pickle=False) as bank:
            weights = bank['weights'][indices].copy()
            penalty = bank['penalty'][indices].copy()
            reference = bank['lime'][indices].copy()
            np.testing.assert_array_equal(expert[:, 0], bank['sasrec'][indices])
        with np.load(use(ROOT / f'artifacts/phase20/nearest_comparison_20260918/{domain}.train_sequences.npz'),
                     allow_pickle=False) as history_bank:
            ui = {str(u): i for i, u in enumerate(history_bank['users'])}
            histories = []
            for r in records:
                i = ui[r['user_id']]
                h = history_bank['items'][history_bank['offsets'][i]:history_bank['offsets'][i + 1]]
                assert h[-20:].tolist() == r['history']
                histories.append(np.unique(h).tolist())
        # No generated candidates from the ongoing mechanism screen are read.
        for i, r in enumerate(records):
            reconstructed, _ = blend(expert[i], weights[i], penalty[i], histories[i], 'full')
            np.testing.assert_allclose(reconstructed, reference[i], atol=2e-7, rtol=0)
        np.savez_compressed(folder / 'inputs.npz', experts=expert, weights=weights, penalty=penalty,
            reference_full=reference, targets=np.asarray([r['target'] for r in records]),
            gram_ids=np.asarray([r['candidate_scores']['GRAM']['ids'] for r in records]),
            guided_ids=np.asarray([r['candidate_scores']['lime']['ids'] for r in records]))
        metadata = [dict(user_id=r['user_id'], target=r['target'], history=r['history'],
                         full_history_unique=histories[i], sample_index=r['sample_index'])
                    for i, r in enumerate(records)]
        write(folder / 'samples.json', metadata)
        use(folder / 'inputs.npz'); use(folder / 'samples.json')
        prepared[domain] = (expert, weights, penalty, reference, records, histories)
    # Freeze both domains before calculating any new ablation outcome.
    write(OUT / 'manifest.json', dict(sha256=inputs, frozen_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    summaries = {}
    for domain, (expert, weights, penalty, reference, records, histories) in prepared.items():
        folder = OUT / domain
        ranks = {name: {scope: [] for scope in SCOPES} for name in VARIANTS}
        entropy = {name: [] for name in VARIANTS}
        all_ids = np.arange(1, expert.shape[2] + 1)
        with (folder / 'predictions.jsonl').open('x') as f:
            for i, r in enumerate(records):
                out = dict(user_id=r['user_id'], target=r['target'], sample_index=r['sample_index'], arms={})
                for name in VARIANTS:
                    scores, effective = blend(expert[i], weights[i], penalty[i], histories[i], name)
                    # Measure exactly the catalog signal supplied to the trie, before its history filter.
                    z = 20. * (scores.astype('float64') - float(scores.max()))
                    p = np.exp(z); p /= p.sum()
                    entropy[name].append(float(-(p * np.log(np.maximum(p, 1e-300))).sum()))
                    out['arms'][name] = dict(effective_weights=effective.tolist(), scopes={})
                    for scope in SCOPES:
                        ids = (all_ids if scope == 'catalog' else np.asarray(r['candidate_scores'][
                            'GRAM' if scope == 'GRAM50' else 'lime']['ids']))
                        ids = ids[~np.isin(ids, r['history'])]
                        order = ids[np.lexsort((ids, -scores[ids - 1]))][:50].tolist()
                        rank = order.index(r['target']) + 1 if r['target'] in order else 51
                        ranks[name][scope].append(rank)
                        out['arms'][name]['scopes'][scope] = dict(items=order, rank=rank)
                        if name == 'full' and scope in ('catalog', 'GRAM50'):
                            old_arm = 'LIME' if scope == 'catalog' else 'LIME_after_GRAM'
                            assert order == r['arms'][old_arm]['items'] and rank == r['arms'][old_arm]['rank']
                f.write(json.dumps(out) + '\n'); f.flush()
                if (i + 1) % 16 == 0:
                    status('RUNNING', stage='CPU_rank_ablation', domain=domain, completed_users=i + 1,
                           total_users=512, completed_domains=list(summaries), elapsed_seconds=time.monotonic() - started)
        from experiment.phase20.lime_decode_screen_core import contrast
        summary = dict(domain=domain, users=512,
            metrics={n: {s: metrics(rr) for s, rr in scopes.items()} for n, scopes in ranks.items()},
            comparisons={n: {s: contrast(rr, ranks['full'][s]) for s, rr in scopes.items()}
                         for n, scopes in ranks.items() if n != 'full'},
            catalog_signal_entropy={n: float(np.mean(v)) for n, v in entropy.items()},
            mean_original_weights=weights.mean(axis=0).tolist(), test_read=False, new_decodes=0,
            original_LIME_catalog_and_GRAM50_exact=True)
        write(folder / 'summary.json', summary)
        summaries[domain] = summary
    status('VERIFYING', stage='independent_audit', elapsed_seconds=time.monotonic() - started)
    subprocess.run([sys.executable, '-m', 'experiment.phase20.verify_lime_expert_ablation'], cwd=ROOT, check=True)
    flags = {}
    for name in VARIANTS:
        if name == 'full':
            continue
        good = all(s['comparisons'][name][scope]['relative_N10'] >= 0
                   and s['comparisons'][name][scope]['net_H10'] >= 0
                   for s in summaries.values() for scope in ('catalog', 'GRAM50'))
        increment = any(s['comparisons'][name][scope]['relative_N10'] >= .005
                        for s in summaries.values() for scope in ('catalog', 'GRAM50'))
        flags[name] = 'WORTH_CONSIDERING_IN_DECODE' if good and increment else 'NO_CLEAR_DUAL_SCOPE_DUAL_DOMAIN_SIGNAL'
    write(OUT / 'summary.json', dict(domains=summaries, screening_flags=flags,
        elapsed_seconds=time.monotonic() - started, no_automatic_GPU_followup=True,
        limitation='Frozen-gate deletion changes remaining relative strength; no refit and no new generation. Historical development reused.'))
    labels = {'full': '完整LIME', 'without_itemcf': '去ItemCF', 'without_text': '去文本',
              'without_sasrec': '去SASRec', 'without_history_penalty': '去历史校准',
              'equal_weights': '三专家等权', 'sasrec_only': 'SASRec单独'}
    lines = ['# LIME专家删减：CPU并行诊断', '', 'Origin Skill：academic-research-suite / experiment-agent，内联执行。',
        '', '状态：COMPLETED / VERIFIED。旧模型、旧缓存、原512评价用户/域；无新生成、无训练、无test读取。',
        '本轮属于事后开发诊断。删专家后保留其余权重比例再归一，不重新拟合；不能据此认定该专家在最优系统中无用。', '',
        '| 信号 | Toys全目录N10 | Beauty全目录N10 | Toys固定GRAM50 N10 | Beauty固定GRAM50 N10 |',
        '|---|---:|---:|---:|---:|']
    for n in VARIANTS:
        a, b = summaries['Toys']['metrics'][n], summaries['Beauty']['metrics'][n]
        lines.append(f'| {labels[n]} | {a["catalog"]["N10"]:.6f} | {b["catalog"]["N10"]:.6f} | {a["GRAM50"]["N10"]:.6f} | {b["GRAM50"]["N10"]:.6f} |')
    lines += ['', '## 解释边界', '',
        '固定候选排序变好不等于解码变好；full/guided50的候选均由旧方法生成，尤其guided50有原LIME选择偏差。',
        '两个域、全目录和原GRAM50口径均不退且至少一处N10相对+0.5%的规则，才标为值得考虑下一次解码。',
        '筛查门槛是资源分配规则，不是非劣检验或正式显著性。旧512用户已经看过，多变体结果不能作为独立确认。',
        '不读取正在运行的新机制评价，不根据本轮修改其规则。没有自动启动GPU跟进。', '',
        json.dumps(flags, ensure_ascii=False, indent=2), '',
        '完整Hit@10/50、引导候选固定排序、描述性配对区间及信号熵见summary.json。',
        f'耗时{time.monotonic() - started:.1f}秒（包含输入校验和独立复算）；CPU两线程。', '',
        '实时状态：`artifacts/phase20/lime_expert_ablation_20260921_v1/status.json`。']
    REPORT.write_text('\n'.join(lines) + '\n')
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n')
    status('COMPLETED', stage='done', elapsed_seconds=time.monotonic() - started,
           independently_verified=True, screening_flags=flags, summary='summary.json', report=str(REPORT.relative_to(ROOT)))


if __name__ == '__main__':
    try:
        run()
    except Exception:
        detail = traceback.format_exc()
        print(detail, flush=True)
        if OUT.exists():
            status('FAILED', stage='failed', error=detail, automatic_retry=False)
        raise
