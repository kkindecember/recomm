"""Frozen two-policy follow-up on disjoint development evaluation users."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

from experiment.phase20 import lime_decode_screen as base
from experiment.phase20.lime_decode_screen_core import metrics, contrast, ordered, target_rank
from experiment.phase20.lime_mechanism_core import ScheduledGuide, select_alpha

ROOT = base.ROOT
PARENT = ROOT / 'artifacts/phase20/lime_decode_screen_20260921_v1/full_parallel_v4'
OUT = ROOT / 'artifacts/phase20/lime_mechanism_screen_20260921_v1'
PROTOCOL = ROOT / 'plan/第二十阶段/LIME生成机制_分支延后与分歧抑制筛查_20260921.md'
REPORT = ROOT / 'report/第二十阶段/LIME生成机制_分支延后与分歧抑制筛查_20260921.md'
MODULE = 'experiment.phase20.lime_mechanism_screen'
GPUS = {'Toys': 'GPU-cfd866c2-ccc1-8a3c-00b8-594d7a8cd059',
        'Beauty': 'GPU-4cc17bbe-a85f-bdab-44ef-bbe29d1afc09'}
METHODS = {'sasrec': ('sasrec', 'constant'), 'lime': ('lime', 'constant'),
           'delayed_branch': ('lime', 'delayed_branch'), 'agreement': ('lime', 'agreement')}
NEW = ('delayed_branch', 'agreement')
base.OUT = OUT


def rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def historical_diagnostics():
    result = {}
    for domain in base.DOMAINS:
        folder = PARENT / domain
        signals = base.arrays(folder / 'signals.npz')['lime']
        data = rows(folder / 'evaluation.jsonl')
        groups, movements, contributions = Counter(), Counter(), Counter()
        order_changes, reranked = 0, []
        for r in data:
            a, b = r['arms']['lime'], r['arms']['lime_late']
            sa, sb = set(a['items']), set(b['items'])
            target = r['target']
            group = ('both' if target in sa & sb else 'gained' if target in sa
                     else 'lost' if target in sb else 'neither')
            groups[group] += 1
            nd = lambda rank: 1 / np.log2(rank + 1) if rank <= 10 else 0.
            contributions[group] += nd(a['rank']) - nd(b['rank'])
            if group == 'both':
                movements['better' if a['rank'] < b['rank'] else
                          'worse' if a['rank'] > b['rank'] else 'equal'] += 1
            order_changes += ([i for i in a['items'] if i in sb] !=
                              [i for i in b['items'] if i in sa])
            ids = np.asarray(r['candidate_scores']['lime']['ids'])
            reranked.append(target_rank(ordered(ids, signals[r['sample_index']][ids - 1],
                                                r['history']), target))
        expected = base.read(folder / 'summary.json')['comparisons']['lime_vs_late']['delta_N10']
        actual = sum(contributions.values()) / len(data)
        assert abs(expected - actual) < 1e-12
        result[domain] = dict(users=len(data), coverage_groups=dict(groups),
            shared_target_rank_changes=dict(movements), shared_order_changed_users=order_changes,
            N10_contributions={k: float(v / len(data)) for k, v in contributions.items()},
            LIME_rerank_guided_candidates=metrics(reranked),
            caveat='Post-hoc development diagnosis; does not select follow-up evaluation users.')
    base.write(OUT / 'historical_diagnostics.json', result)
    return result


def prepare():
    import torch
    sys.path.insert(0, str(ROOT / 'third_party/LIME-Rec'))
    from lime_rec.evaluation import normalize_score_rows
    from experiment.phase20.catalog_scope_core import topk
    torch.set_num_threads(2)
    assert base.read(PARENT / 'status.json')['state'] == 'COMPLETED'
    OUT.mkdir(exist_ok=False)
    base.write(OUT / 'status.json', dict(state='PREPARING', updated_at=base.stamp(), test_read=False))
    cfg = dict(domains=base.DOMAINS, gpus=GPUS, tuning_users=128, evaluation_users=512,
        alpha_grid=[0., .1, .3], new_policies=list(NEW), score_scale=20., beam_size=50,
        length_penalty=1., allocation_limit_gib=11, minimum_free_mib=12288,
        worker_timeout_seconds=None, no_model_training=True, test_read=False,
        selection='Reuse original 128 tuning users; evaluate on 512 disjoint decode-followup users.',
        outcome_peeking='No new evaluation metrics are read until selection.json is frozen.',
        authorization='User: ok研究吧; background execution, shared GPUs and status files authorized.')
    base.write(OUT / 'config.json', cfg)
    (OUT / 'protocol.md').write_text(PROTOCOL.read_text())
    historical_diagnostics()
    manifest = dict(base.read(PARENT / 'manifest.json')['sha256'])
    def use(path):
        path = Path(path)
        rel = str(path.relative_to(ROOT))
        assert '/test/' not in rel and not path.name.startswith('test.json')
        digest = base.sha(path)
        if rel in manifest:
            assert manifest[rel] == digest, ('Changed inherited input', rel)
        manifest[rel] = digest
        return path
    for domain in base.DOMAINS:
        folder = OUT / domain
        folder.mkdir()
        old_samples = base.read(use(PARENT / domain / 'samples.json'))
        old_users = {s['user_id'] for s in old_samples}
        data_cfg = base.read(use(PARENT / domain / 'data_config.json'))
        prepared = ROOT / data_cfg['prepared']
        catalog = base.read(use(prepared / 'catalog.json'))
        validation = rows(use(prepared / 'validation.jsonl'))
        vi = {r['user_id']: i for i, r in enumerate(validation)}
        query = base.arrays(use(ROOT / f'artifacts/phase20/catalog_scope_20260919_v1/{domain}/queries.npz'))
        recovery = ROOT / f'artifacts/phase20/lime_comparison_20260920_v1/recovery_v2/{domain}'
        old = base.arrays(use(recovery / 'evaluation_seed0_cal2048.npz'))
        gate = torch.load(use(recovery / 'gate_seed0_cal2048.pt'), map_location='cpu')
        np.testing.assert_array_equal(np.sort(gate['fit_indices']), query['calibration_indices'])
        np.testing.assert_array_equal(old['users'], query['users'][old['indices']])
        assert set(old['indices']).isdisjoint(gate['fit_indices'])
        pos_by_user = {str(u): i for i, u in enumerate(old['users'])}
        eligible = [i for i, u in enumerate(old['users']) if str(u) not in old_users]
        chosen = sorted(eligible, key=lambda i: hashlib.sha256(
            f'20260921|lime-mechanism-v1|{domain}|{old["users"][i]}'.encode()).digest())[:512]
        assert len(chosen) == 512
        positions = [pos_by_user[s['user_id']] for s in old_samples[:131]] + chosen
        samples = []
        for j, pos in enumerate(positions):
            qi = int(old['indices'][pos]); user = str(old['users'][pos]); row = validation[vi[user]]
            history = query['history'][qi]; history = history[history > 0].tolist()
            assert row['history'] == history and row['target'] == int(old['targets'][pos])
            samples.append(dict(user_id=user, target=row['target'], history=history,
                validation_index=vi[user], query_index=qi, evaluation_cache_index=pos,
                split='smoke' if j < 3 else 'tuning' if j < 131 else 'evaluation'))
        assert samples[:131] == old_samples[:131]
        assert len({s['user_id'] for s in samples}) == 643
        assert {s['user_id'] for s in samples[131:]}.isdisjoint(old_users)
        ix = np.asarray([r['query_index'] for r in samples])
        weights = old['weights'][positions].astype('float32')
        penalty = old['penalty'][positions].astype('float32')
        paths = [recovery / 'sasrec_scores_seed0.npy',
                 ROOT / f'artifacts/phase20/lime_comparison_20260920_v1/{domain}/itemcf_scores.npy',
                 recovery / 'semantic_scores.npy']
        raw = [np.asarray(np.load(use(p), mmap_mode='r')[ix]) for p in paths]
        normalized = np.stack([normalize_score_rows(s) for s in raw], axis=1)
        fused = (torch.from_numpy(weights)[:, :, None] * torch.from_numpy(normalized)).sum(1)
        bank = base.arrays(use(ROOT / f'artifacts/phase20/nearest_comparison_20260918/{domain}.train_sequences.npz'))
        ui = {str(u): i for i, u in enumerate(bank['users'])}
        for j, sample in enumerate(samples):
            k = ui[sample['user_id']]
            history = bank['items'][bank['offsets'][k]:bank['offsets'][k + 1]]
            assert history[-20:].tolist() == sample['history']
            fused[j, np.unique(history) - 1] -= penalty[j]
        fused = fused.numpy()
        masked = fused.copy()
        for j, sample in enumerate(samples):
            masked[j, np.asarray(sample['history']) - 1] = -np.inf
        tops = topk(masked, np.arange(1, len(catalog['items'])))
        arm_index = old['arm_names'].tolist().index('LIME')
        np.testing.assert_array_equal(tops, old['top50'][arm_index, positions])
        previous = base.arrays(use(PARENT / domain / 'signals.npz'))
        np.testing.assert_array_equal(fused[:131], previous['lime'][:131])
        np.testing.assert_array_equal(normalized[:131, 0], previous['sasrec'][:131])
        np.savez_compressed(folder / 'signals.npz', lime=fused, sasrec=normalized[:, 0],
                            lime_top50=tops, weights=weights, penalty=penalty)
        base.write(folder / 'samples.json', samples)
        base.write(folder / 'data_config.json', data_cfg)
        base.write(folder / 'preflight.json', dict(samples=643, new_evaluation_users=512,
            old_all_643_users_excluded_from_new_evaluation=True, gate_fit_disjoint=True,
            original_tuning_signals_exact=True, LIME_top50_exact=True,
            development_reuse_caveat='New only relative to the prior decode cohort, not untouched historical development data.'))
        base.write(folder / 'status.json', dict(state='PENDING', stage='pending', domain=domain, updated_at=base.stamp()))
        for name in ('samples.json', 'signals.npz', 'data_config.json', 'preflight.json'):
            use(folder / name)
        for name in ('tuning.jsonl', 'evaluation.jsonl', 'selection.json', 'summary.json'):
            use(PARENT / domain / name)
    for path in (Path(__file__).resolve(), PROTOCOL, OUT / 'config.json', OUT / 'protocol.md',
                 OUT / 'historical_diagnostics.json',
                 ROOT / 'experiment/phase20/lime_mechanism_core.py',
                 ROOT / 'experiment/phase20/test_lime_mechanism.py',
                 ROOT / 'experiment/phase20/verify_lime_mechanism.py',
                 ROOT / 'experiment/phase20/run_lime_mechanism_screen.sh',
                 ROOT / 'experiment/phase18/core/diff_generation_cache.py'):
        use(path)
    base.write(OUT / 'manifest.json', dict(sha256=manifest, created_at=base.stamp()))
    base.write(OUT / 'status.json', dict(state='PREPARED', updated_at=base.stamp(), test_read=False,
                                        frozen_inputs=len(manifest), domains=list(base.DOMAINS)))
    print(json.dumps(dict(prepared=str(OUT), frozen_inputs=len(manifest))), flush=True)


class Worker(base.Worker):
    def __init__(self, domain):
        super().__init__(domain)
        from experiment.phase18.core.diff_generation_cache import install_cross_cache_reuse
        install_cross_cache_reuse(self.model, 50)
        self.cached = {r['sample_index']: r for r in rows(PARENT / domain / 'tuning.jsonl')}

    def status(self, state='RUNNING', **kw):
        import torch
        data = dict(state=state, stage=self.stage, domain=self.domain, pid=os.getpid(),
            updated_at=base.stamp(), elapsed_seconds=time.monotonic() - self.started,
            completed_decodes=self.completed, current_sample=getattr(self, 'current_sample', None),
            current_arm=getattr(self, 'current_arm', None),
            mean_decode_seconds=float(np.mean(self.seconds[-16:])) if self.seconds else None,
            peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
            test_read=False, optimizer_steps=0)
        previous = base.read(self.folder / 'status.json') if (self.folder / 'status.json').exists() else {}
        for key in ('tuning_users_written', 'evaluation_users_written', 'selection', 'projected_remaining_hours'):
            if key in previous:
                data[key] = previous[key]
        data.update(kw)
        base.write(self.folder / 'status.json', data)

    def guided_decode(self, ctx, q, alpha, policy):
        import torch
        from transformers import LogitsProcessorList
        from model.gram import GRAM
        from model.gram_t5_outputs import BaseModelOutputWithPastAndCrossAttentions
        guide = ScheduledGuide(self.index, q, alpha, policy)
        begin = time.monotonic()
        with torch.no_grad():
            encoded = BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=ctx['hidden'])
            output = super(GRAM, self.model).generate(input_ids=ctx['ids'].reshape(1, -1),
                attention_mask=ctx['mask'].reshape(1, -1), encoder_outputs=encoded,
                max_length=self.max_length, prefix_allowed_tokens_fn=self.prefix_fn,
                num_beams=50, num_return_sequences=50, return_dict_in_generate=True,
                output_scores=True, length_penalty=1., logits_processor=LogitsProcessorList([guide]))
        paths = [p[:p.index(1) + 1] for p in output.sequences.cpu().tolist()]
        ids = [self.index.path_to_item[tuple(p)] for p in paths]
        scores = output.sequences_scores.detach().cpu().double().numpy()
        assert len(set(ids)) == 50 and np.isfinite(scores).all()
        np.testing.assert_allclose(guide.rescore(paths), scores, atol=2e-6, rtol=1e-5)
        del output
        torch.cuda.synchronize()
        self.seconds.append(time.monotonic() - begin)
        self.completed += 1
        return dict(paths=paths, ids=ids, scores=scores, guide=guide)

    def smoke(self):
        self.stage = 'engineering_smoke'
        checks = []
        for j in range(3):
            self.current_sample = j
            self.current_arm = 'original_GRAM'
            self.status()
            ctx = self.context(j)
            # Base uses a numeric timeout only; disable it in memory.
            self.config['worker_timeout_seconds'] = float('inf')
            native = self.decode(ctx)
            q = self.index.conditional_logs(self.signals['lime'][j])
            zero = self.guided_decode(ctx, q, 0., 'agreement')
            assert zero['paths'] == native['paths']
            np.testing.assert_array_equal(zero['scores'], native['scores'])
            old_mix = self.decode(ctx, q, .3)
            constant = self.guided_decode(ctx, q, .3, 'constant')
            assert constant['paths'] == old_mix['paths']
            np.testing.assert_array_equal(constant['scores'], old_mix['scores'])
            for policy in NEW:
                self.current_arm = policy
                self.status()
                generated = self.guided_decode(ctx, q, .3, policy)
                late = ScheduledGuide(self.index, q, .3, policy)
                late.trace = native['guide'].trace
                values = late.rescore(native['paths'])
                assert np.isfinite(values).all()
                # Compare shared complete-item scores across search placements.
                lookup = dict(zip(native['ids'], values))
                for item, score in zip(generated['ids'], generated['scores']):
                    if item in lookup:
                        np.testing.assert_allclose(score, lookup[item], atol=5e-6, rtol=1e-5)
            checks.append(dict(user=self.samples[j]['user_id'], zero_exact=True,
                constant_matches_original_exact=True, legal_paths=True,
                path_scores_reconstructed=True, common_items_match_late=True))
            del ctx, native, zero, old_mix, constant, generated, late
        assert self.digest() == self.initial_digest
        base.write(self.folder / 'smoke.json', dict(checks=checks, decode_seconds=self.seconds,
            initial_digest=self.initial_digest, max_error_tolerance=5e-6))
        self.status(projected_remaining_hours=float(np.mean(self.seconds[-12:])) * 3200 / 3600)

    def one(self, j, choices, split):
        self.current_sample = j
        self.current_arm = 'GRAM'
        self.status()
        sample = self.samples[j]
        ctx = self.context(j)
        native = self.guided_decode(ctx, None, 0., 'constant')
        history, target = sample['history'], sample['target']
        arms, saved, diagnostics = {}, {}, {}
        def add(name, ids, score):
            order = ordered(ids, score, history)[:50]
            arms[name] = dict(items=order, rank=target_rank(order, target))
            saved[name] = dict(ids=list(map(int, ids)), scores=np.asarray(score).tolist())
        add('GRAM', native['ids'], native['scores'])
        if split == 'tuning':
            old = self.cached[j]
            assert all(old[k] == sample[k] for k in ('user_id', 'target', 'history', 'split'))
            assert native['paths'] == old['baseline_paths']
            np.testing.assert_allclose(native['scores'], old['candidate_scores']['GRAM']['scores'], atol=5e-6, rtol=5e-6)
            assert arms['GRAM'] == old['arms']['GRAM']
            for name, arm in old['arms'].items():
                if name != 'GRAM':
                    arms[name] = arm
                    if name in old['candidate_scores']:
                        saved[name] = old['candidate_scores'][name]
        else:
            ids = np.arange(1, len(self.catalog['items']))
            order = ordered(ids, self.signals['lime'][j], history)[:50]
            arms['LIME'] = dict(items=order, rank=target_rank(order, target))
            add('LIME_after_GRAM', native['ids'], self.signals['lime'][j][np.asarray(native['ids']) - 1])
        qs = {signal: self.index.conditional_logs(self.signals[signal][j])
              for signal in {METHODS[name][0] for name in choices}}
        for name, strengths in choices.items():
            signal, policy = METHODS[name]
            q = qs[signal]
            for alpha in strengths:
                label = name if split == 'evaluation' else f'{name}_{alpha}'
                self.current_arm = label
                self.status()
                if alpha:
                    out = self.guided_decode(ctx, q, alpha, policy)
                    add(label, out['ids'], out['scores'])
                    late = ScheduledGuide(self.index, q, alpha, policy)
                    late.trace = native['guide'].trace
                    add(label + '_late', native['ids'], late.rescore(native['paths']))
                    diagnostics[label] = out['guide'].diagnostics()
                    if split == 'evaluation' and name == 'lime':
                        add('LIME_on_guided_candidates', out['ids'],
                            self.signals['lime'][j][np.asarray(out['ids']) - 1])
                else:
                    for alias in (label, label + '_late'):
                        arms[alias], saved[alias] = arms['GRAM'], saved['GRAM']
                    if split == 'evaluation' and name == 'lime':
                        arms['LIME_on_guided_candidates'] = arms['LIME_after_GRAM']
                        saved['LIME_on_guided_candidates'] = saved['LIME_after_GRAM']
        return dict(user_id=sample['user_id'], target=target, history=history, sample_index=j,
            split=split, arms=arms, candidate_scores=saved, baseline_paths=native['paths'],
            guidance_diagnostics=diagnostics)

    def run(self):
        self.smoke()
        self.stage = 'tuning'
        tuning = []
        with (self.folder / 'tuning.jsonl').open('x') as f:
            for k, j in enumerate(range(3, 131), 1):
                row = self.one(j, {n: [.1, .3] for n in NEW}, 'tuning')
                tuning.append(row)
                f.write(json.dumps(row) + '\n'); f.flush()
                self.status(tuning_users_written=k)
        choices, table = {}, {}
        for name in METHODS:
            choices[name], table[name] = select_alpha(tuning, name)
        parent_selection = base.read(PARENT / self.domain / 'selection.json')['selected_alpha']
        assert all(choices[name] == parent_selection[name] for name in ('sasrec', 'lime'))
        base.write(self.folder / 'selection.json', dict(selected_alpha=choices,
            tuning_N10=table, selected_at=base.stamp(), evaluation_users_read=0,
            criterion='Per-method NDCG@10 on original 128 tuning users; ties favor smaller alpha.'))
        selection_hash = base.sha(self.folder / 'selection.json')
        self.stage = 'evaluation'
        self.status(selection=choices)
        evaluation = []
        with (self.folder / 'evaluation.jsonl').open('x') as f:
            for k, j in enumerate(range(131, 643), 1):
                row = self.one(j, {n: [choices[n]] for n in METHODS}, 'evaluation')
                evaluation.append(row)
                f.write(json.dumps(row) + '\n'); f.flush()
                remain = (512 - k) * (1 + sum(a > 0 for a in choices.values()))
                self.status(evaluation_users_written=k,
                    projected_remaining_hours=remain * float(np.mean(self.seconds[-16:])) / 3600)
        assert base.sha(self.folder / 'selection.json') == selection_hash
        ranks = {name: np.asarray([r['arms'][name]['rank'] for r in evaluation])
                 for name in evaluation[0]['arms']}
        comparisons = {name + '_vs_GRAM': contrast(rr, ranks['GRAM'])
                       for name, rr in ranks.items() if name != 'GRAM'}
        for name in METHODS:
            comparisons[name + '_vs_late'] = contrast(ranks[name], ranks[name + '_late'])
        for name in NEW:
            comparisons[name + '_vs_lime'] = contrast(ranks[name], ranks['lime'])
            comparisons[name + '_vs_sasrec'] = contrast(ranks[name], ranks['sasrec'])
            comparisons[name + '_vs_direct_LIME_rerank'] = contrast(ranks[name], ranks['LIME_after_GRAM'])
        assert self.digest() == self.initial_digest
        for path, digest in self.loaded_source.items():
            assert base.sha(ROOT / path) == digest, ('Changed loaded source', path)
        base.write(self.folder / 'summary.json', dict(domain=self.domain, selected_alpha=choices,
            tuning_users=128, evaluation_users=512, metrics={n: metrics(rr) for n, rr in ranks.items()},
            comparisons=comparisons, model_unchanged=True, test_read=False, optimizer_steps=0,
            development_diagnostic=True, selection_sha256=selection_hash,
            elapsed_seconds=time.monotonic() - self.started, completed_decodes=self.completed))
        self.stage = 'done'
        self.status('COMPLETED')


def finish():
    subprocess.run([sys.executable, '-m', 'experiment.phase20.verify_lime_mechanism'],
                   cwd=ROOT, check=True)
    summaries = {d: base.read(OUT / d / 'summary.json') for d in base.DOMAINS}
    decisions = {}
    for name in NEW:
        perf = all(s['comparisons'][name + '_vs_GRAM']['relative_N10'] >= .01
                   and s['comparisons'][name + '_vs_GRAM']['net_H10'] >= 0 for s in summaries.values())
        extra = all(s['comparisons'][name + '_vs_lime']['relative_N10'] >= .005
                    and s['comparisons'][name + '_vs_lime']['net_H10'] >= 0 for s in summaries.values())
        decisions[name] = 'RETEST_IN_CLEAN_PIPELINE' if perf and extra else 'NO_CLEAR_DUAL_DOMAIN_INCREMENT'
    base.write(OUT / 'summary.json', dict(domains=summaries, decisions=decisions,
        test_read=False, automatic_new_experiments=False, independent_verification='verification.json',
        caveat='Development-only adaptive follow-up; new evaluation cohort is disjoint only from the prior decode study.'))
    lines = ['# LIME 生成机制筛查', '', 'Origin Skill：academic-research-suite / experiment-agent，内联执行。', '',
             '状态：COMPLETED / VERIFIED。两个新规则、固定旧模型；每域128人选参、512名与前轮互斥的解码评价用户。',
             '仍属于反复使用的历史开发集，不是最终test，也不证明机制新颖性。', '',
             '分支延后：首个有多个合法选项的分支不引导；之后沿用原概率混合。',
             '分歧抑制：alpha_eff=alpha×sum(min(p_GRAM,p_expert))，不是正确率估计。', '']
    for d, s in summaries.items():
        lines += [f'## {d}', '', f'选中权重：{s["selected_alpha"]}', '',
                  '| 方法 | NDCG@10 | Hit@10 | Hit@50 |', '|---|---:|---:|---:|']
        for name, m in s['metrics'].items():
            lines.append(f'| {name} | {m["N10"]:.6f} | {m["H10"]:.6f} | {m["H50"]:.6f} |')
        lines += ['']
    lines += ['## 筛查判定', '', json.dumps(decisions, ensure_ascii=False), '',
              '继续门槛仅用于资源分配；不作正式显著性、因果识别或接受概率结论。所有方法和负结果均保留。', '',
              '结构化结果：`artifacts/phase20/lime_mechanism_screen_20260921_v1/summary.json`。',
              '独立核验：同目录 `verification.json`。未自动追加参数或启动新实验。']
    REPORT.write_text('\n'.join(lines) + '\n')
    return decisions


def supervise():
    config = base.read(OUT / 'config.json')
    pending, active, done = set(base.DOMAINS), {}, {}
    started = time.monotonic()
    while pending or active:
        for d in sorted(pending):
            if base.gpu_free(GPUS[d]) < config['minimum_free_mib']:
                continue
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPUS[d], PYTHONUNBUFFERED='1',
                       OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                       TOKENIZERS_PARALLELISM='false')
            with (OUT / d / 'worker.log').open('x') as log:
                active[d] = subprocess.Popen([sys.executable, '-m', MODULE, 'worker', '--domain', d],
                    cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
            pending.remove(d)
        for d, child in list(active.items()):
            if child.poll() is not None:
                done[d] = child.returncode
                del active[d]
                if child.returncode:
                    previous = base.read(OUT / d / 'status.json')
                    base.write(OUT / d / 'status.json', dict(previous, state='FAILED',
                        updated_at=base.stamp(), exit_code=child.returncode))
        status = dict(state='RUNNING', updated_at=base.stamp(), supervisor_pid=os.getpid(),
            background=True, active={d: p.pid for d, p in active.items()}, pending=sorted(pending),
            completed=done, elapsed_seconds=time.monotonic() - started,
            domains={d: base.read(OUT / d / 'status.json') for d in base.DOMAINS},
            test_read=False, automatic_new_experiments=False, wall_clock_timeout_seconds=None,
            result_path=str((OUT / 'summary.json').relative_to(ROOT)), report_path=str(REPORT.relative_to(ROOT)))
        for d in pending:
            status['domains'][d]['state'] = 'WAITING_FOR_MEMORY'
        if not pending and not active:
            if len(done) == 2 and all(rc == 0 for rc in done.values()):
                try:
                    status.update(state='COMPLETED', decisions=finish(), independently_verified=True)
                except Exception:
                    status.update(state='AUDIT_FAILED', error=traceback.format_exc())
            else:
                status['state'] = 'INCOMPLETE'
        base.write(OUT / 'status.json', status)
        if pending or active:
            time.sleep(15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'launch', 'supervise', 'worker'])
    parser.add_argument('--domain', choices=base.DOMAINS)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'launch':
        assert base.read(OUT / 'status.json')['state'] == 'PREPARED'
        assert not (OUT / 'launch.json').exists()
        with (OUT / 'supervisor.log').open('x') as log:
            child = subprocess.Popen([sys.executable, '-m', MODULE, 'supervise'], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        base.write(OUT / 'launch.json', dict(pid=child.pid, started_at=base.stamp()))
        print(json.dumps(dict(pid=child.pid, status=str(OUT / 'status.json'))), flush=True)
    elif args.action == 'supervise':
        try:
            supervise()
        except Exception:
            base.write(OUT / 'status.json', dict(state='SUPERVISOR_FAILED', updated_at=base.stamp(),
                error=traceback.format_exc(), test_read=False, automatic_retry=False))
            raise
    else:
        try:
            Worker(args.domain).run()
        except Exception:
            detail = traceback.format_exc()
            print(detail, flush=True)
            previous = base.read(OUT / args.domain / 'status.json')
            base.write(OUT / args.domain / 'status.json', dict(previous, state='FAILED',
                       updated_at=base.stamp(), error=detail, automatic_retry=False))
            raise


if __name__ == '__main__':
    main()
