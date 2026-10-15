"""Frozen four-cell, shared-alpha prefix expert weighting screen."""
import argparse
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
from experiment.phase20.lime_prefix_expert_core import Descendants, PrefixExperts

ROOT = base.ROOT
PARENT = ROOT / 'artifacts/phase20/lime_decode_screen_20260921_v1/full_parallel_v4'
PRIOR = ROOT / 'artifacts/phase20/lime_mechanism_screen_20260921_v1'
OUT = ROOT / 'artifacts/phase20/lime_prefix_expert_20260921_v1'
PROTOCOL = ROOT / 'plan/第二十阶段/LIME前缀专家权重_四组筛查_20260921.md'
REPORT = ROOT / 'report/第二十阶段/LIME前缀专家权重_四组筛查_20260921.md'
MODULE = 'experiment.phase20.lime_prefix_expert'
GPUS = {'Toys': 'GPU-2afcc1df-7df8-1f3f-8970-a252a06c6160',
        'Beauty': 'GPU-e2e37406-aaaa-da68-cf74-1a487b1c93e2'}
METHODS = ('fixed_cal', 'fixed_no_cal', 'dynamic_cal', 'dynamic_no_cal')
base.OUT = OUT


def rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def prepare():
    import torch
    sys.path.insert(0, str(ROOT / 'third_party/LIME-Rec'))
    from lime_rec.evaluation import normalize_score_rows
    from experiment.phase20.catalog_scope_core import topk
    torch.set_num_threads(2)
    assert base.read(PARENT / 'status.json')['state'] == 'COMPLETED'
    OUT.mkdir(exist_ok=False)
    base.write(OUT / 'status.json', dict(state='PREPARING', updated_at=base.stamp(), test_read=False))
    cfg = dict(domains=base.DOMAINS, gpus=GPUS, reserved_old_tuning_users=128, new_tuning_users=0, evaluation_users=512,
        alpha=.3, coefficient=1., arms=list(METHODS), score_scale=20., beam_size=50,
        length_penalty=1., allocation_limit_gib=11, minimum_free_mib=12288,
        worker_timeout_seconds=None, no_model_training=True, test_read=False,
        selection='Freeze alpha=.3 from original LIME tuning; no new parameter selection; 512 users disjoint from both previous decode cohorts.',
        outcome_peeking='No new evaluation metrics are read until selection.json is frozen.',
        authorization='User: 那做吧, authorizing four-cell prefix weighting screen; background and shared GPUs authorized.')
    base.write(OUT / 'config.json', cfg)
    (OUT / 'protocol.md').write_text(PROTOCOL.read_text())
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
        previous_followup = base.read(use(PRIOR / domain / 'samples.json'))
        old_users = {s['user_id'] for s in old_samples + previous_followup}
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
            f'20260921|lime-prefix-expert-v1|{domain}|{old["users"][i]}'.encode()).digest())[:512]
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
        history_mask = np.zeros((len(samples), normalized.shape[2]), dtype=bool)
        for j, sample in enumerate(samples):
            k = ui[sample['user_id']]
            history = bank['items'][bank['offsets'][k]:bank['offsets'][k + 1]]
            assert history[-20:].tolist() == sample['history']
            history_mask[j, np.unique(history) - 1] = True
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
                            lime_top50=tops, weights=weights, penalty=penalty,
                            experts=normalized, history_mask=history_mask)
        base.write(folder / 'samples.json', samples)
        base.write(folder / 'data_config.json', data_cfg)
        base.write(folder / 'preflight.json', dict(samples=643, new_evaluation_users=512,
            both_previous_decode_cohorts_excluded=True, gate_fit_disjoint=True,
            original_tuning_signals_exact=True, LIME_top50_exact=True,
            development_reuse_caveat='Disjoint from both previous decode cohorts, but still historically reused development data.'))
        base.write(folder / 'status.json', dict(state='PENDING', stage='pending', domain=domain, updated_at=base.stamp()))
        parent_selection = base.read(use(PARENT / domain / 'selection.json'))
        assert parent_selection['selected_alpha']['lime'] == .3
        base.write(folder / 'selection.json', dict(alpha=.3, coefficient=1., arms=list(METHODS),
            evaluation_users_read=0, selected_at=base.stamp(),
            criterion='All four arms use original LIME alpha=.3; no new tuning.'))
        for name in ('samples.json', 'signals.npz', 'data_config.json', 'preflight.json', 'selection.json'):
            use(folder / name)
    for path in (Path(__file__).resolve(), PROTOCOL, OUT / 'config.json', OUT / 'protocol.md',
                 ROOT / 'experiment/phase20/lime_prefix_expert_core.py',
                 ROOT / 'experiment/phase20/test_lime_prefix_expert.py',
                 ROOT / 'experiment/phase20/verify_lime_prefix_expert.py',
                 ROOT / 'experiment/phase20/run_lime_prefix_expert.sh',
                 ROOT / 'experiment/phase18/core/diff_generation_cache.py',
                 ROOT / 'experiment/phase20/verify_lime_mechanism.py'):
        use(path)
    base.write(OUT / 'manifest.json', dict(sha256=manifest, created_at=base.stamp()))
    base.write(OUT / 'status.json', dict(state='PREPARED', updated_at=base.stamp(), test_read=False,
                                        frozen_inputs=len(manifest), domains=list(base.DOMAINS)))
    print(json.dumps(dict(prepared=str(OUT), frozen_inputs=len(manifest))), flush=True)


def comparison_pairs(names):
    pairs = {n + '_vs_GRAM': (n, 'GRAM') for n in names if n != 'GRAM'}
    for name in METHODS:
        pairs[name + '_vs_late'] = (name, name + '_late')
        pairs[name + '_vs_direct_LIME_rerank'] = (name, 'LIME_after_GRAM')
    for a, b in [('dynamic_cal', 'fixed_cal'), ('dynamic_no_cal', 'fixed_no_cal'),
                 ('fixed_no_cal', 'fixed_cal'), ('dynamic_no_cal', 'dynamic_cal'),
                 ('dynamic_no_cal', 'fixed_cal')]:
        pairs[a + '_vs_' + b] = (a, b)
    return pairs


class Worker(base.Worker):
    def __init__(self, domain):
        super().__init__(domain)
        from experiment.phase18.core.diff_generation_cache import install_cross_cache_reuse
        install_cross_cache_reuse(self.model, 50)
        self.config['worker_timeout_seconds'] = float('inf')
        self.structure = Descendants(self.index)
        self.user_seconds = []
        self.selection_hash = base.sha(self.folder / 'selection.json')
        self.loaded_source[str(Path(__file__).resolve().relative_to(ROOT))] = base.sha(Path(__file__))
        base.write(self.folder / 'loaded_sources.json', self.loaded_source)

    def status(self, state='RUNNING', **kw):
        import torch
        previous = base.read(self.folder / 'status.json') if (self.folder / 'status.json').exists() else {}
        data = dict(previous, state=state, stage=self.stage, domain=self.domain, pid=os.getpid(),
            updated_at=base.stamp(), elapsed_seconds=time.monotonic() - self.started,
            completed_decodes=self.completed, current_sample=getattr(self, 'current_sample', None),
            current_arm=getattr(self, 'current_arm', None),
            mean_decode_seconds=float(np.mean(self.seconds[-16:])) if self.seconds else None,
            peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
            test_read=False, optimizer_steps=0, alpha=.3)
        data.update(kw)
        base.write(self.folder / 'status.json', data)

    def providers(self, j):
        import torch
        fixed_no_cal = (torch.from_numpy(self.signals['weights'][j])[:, None]
                        * torch.from_numpy(self.signals['experts'][j])).sum(0).numpy()
        reconstructed = fixed_no_cal.copy()
        reconstructed[self.signals['history_mask'][j]] -= self.signals['penalty'][j]
        np.testing.assert_array_equal(reconstructed, self.signals['lime'][j])
        p = PrefixExperts(self.structure, self.signals['experts'][j], self.signals['weights'][j],
            np.flatnonzero(self.signals['history_mask'][j]) + 1, self.signals['penalty'][j])
        qs = dict(fixed_cal=self.index.conditional_logs(self.signals['lime'][j]),
                  fixed_no_cal=self.index.conditional_logs(fixed_no_cal),
                  dynamic_cal=p.views[True], dynamic_no_cal=p.views[False])
        return p, qs

    def smoke(self):
        self.stage = 'engineering_smoke'
        checks = []
        for j in range(3):
            begin = time.monotonic()
            self.current_sample, self.current_arm = j, 'GRAM'
            self.status()
            ctx = self.context(j)
            native = self.decode(ctx)
            provider, qs = self.providers(j)
            zero = self.decode(ctx, qs['dynamic_cal'], 0.)
            assert zero['paths'] == native['paths']
            np.testing.assert_array_equal(zero['scores'], native['scores'])
            for name in METHODS:
                self.current_arm = name
                self.status()
                out = self.decode(ctx, qs[name], .3)
                late = native['guide'].rescore(native['paths'], qs[name], .3)
                lookup = dict(zip(native['ids'], late))
                for item, value in zip(out['ids'], out['scores']):
                    if item in lookup:
                        np.testing.assert_allclose(value, lookup[item], atol=5e-6, rtol=1e-5)
            diag = provider.diagnostics()
            assert diag['max_weight_change'] > 0
            checks.append(dict(user_id=self.samples[j]['user_id'], zero_exact=True,
                original_fusion_exact=True, legal_paths=True, path_scores_reconstructed=True,
                shared_items_match_late=True, weight_changes=diag,
                elapsed_seconds=time.monotonic() - begin))
        assert self.digest() == self.initial_digest
        base.write(self.folder / 'smoke.json', dict(checks=checks,
            initial_digest=self.initial_digest, decode_seconds=self.seconds))
        self.status(engineering_checks_passed=True)

    def one(self, j):
        self.current_sample, self.current_arm = j, 'GRAM'
        self.status()
        sample = self.samples[j]
        ctx = self.context(j)
        native = self.decode(ctx)
        provider, qs = self.providers(j)
        arms, saved, timing = {}, {}, {}
        def add(name, ids, scores):
            order = ordered(ids, scores, sample['history'])[:50]
            arms[name] = dict(items=order, rank=target_rank(order, sample['target']))
            saved[name] = dict(ids=list(map(int, ids)), scores=np.asarray(scores).tolist())
        add('GRAM', native['ids'], native['scores'])
        timing['GRAM'] = self.seconds[-1]
        ids = self.signals['lime_top50'][j].tolist()
        arms['LIME'] = dict(items=ids, rank=target_rank(ids, sample['target']))
        add('LIME_after_GRAM', native['ids'], self.signals['lime'][j][np.asarray(native['ids']) - 1])
        # Rotate order by a predeclared, outcome-independent rule to spread GPU
        # contention / warm-cache effects across cells. Native always first.
        shift = (j - 131) % 4
        for name in METHODS[shift:] + METHODS[:shift]:
            self.current_arm = name
            self.status()
            out = self.decode(ctx, qs[name], .3)
            timing[name] = self.seconds[-1]
            add(name, out['ids'], out['scores'])
            late = native['guide'].rescore(native['paths'], qs[name], .3)
            add(name + '_late', native['ids'], late)
            lookup = dict(zip(native['ids'], late))
            for item, value in zip(out['ids'], out['scores']):
                if item in lookup:
                    np.testing.assert_allclose(value, lookup[item], atol=5e-6, rtol=1e-5)
        return dict(user_id=sample['user_id'], target=sample['target'], history=sample['history'],
            sample_index=j, split='evaluation', arms=arms, candidate_scores=saved,
            baseline_paths=native['paths'], prefix_diagnostics=provider.diagnostics(),
            decode_seconds=timing)

    def run(self):
        self.smoke()
        self.stage = 'evaluation'
        evaluation = []
        with (self.folder / 'evaluation.jsonl').open('x') as f:
            for k, j in enumerate(range(131, 643), 1):
                begin = time.monotonic()
                row = self.one(j)
                evaluation.append(row)
                f.write(json.dumps(row) + '\n'); f.flush()
                self.user_seconds.append(time.monotonic() - begin)
                per_user = float(np.mean(self.user_seconds[-16:]))
                self.status(evaluation_users_written=k, evaluation_users_total=512,
                    mean_recent_user_seconds=per_user,
                    projected_remaining_hours=(512 - k) * per_user / 3600,
                    eta_caveat='Measured full-user wall time; shared GPU contention can change it.')
        assert base.sha(self.folder / 'selection.json') == self.selection_hash
        ranks = {n: np.asarray([r['arms'][n]['rank'] for r in evaluation]) for n in evaluation[0]['arms']}
        pairs = comparison_pairs(ranks)
        comparisons = {key: contrast(ranks[a], ranks[b]) for key, (a, b) in pairs.items()}
        assert self.digest() == self.initial_digest
        for path, digest in self.loaded_source.items():
            assert base.sha(ROOT / path) == digest, ('Changed loaded source', path)
        base.write(self.folder / 'summary.json', dict(domain=self.domain, alpha=.3, coefficient=1.,
            new_tuning_users=0, evaluation_users=512, metrics={n: metrics(rr) for n, rr in ranks.items()},
            comparisons=comparisons, comparison_pairs=pairs, model_unchanged=True, test_read=False,
            optimizer_steps=0, development_diagnostic=True, selection_sha256=self.selection_hash,
            elapsed_seconds=time.monotonic() - self.started, completed_decodes=self.completed,
            mean_decode_seconds_by_arm={n: float(np.mean([r['decode_seconds'][n] for r in evaluation]))
                                        for n in ['GRAM'] + list(METHODS)},
            weight_changed_users=sum(r['prefix_diagnostics']['max_weight_change'] > 0 for r in evaluation)))
        self.stage = 'done'
        self.status('COMPLETED', projected_remaining_hours=0.)


def finish():
    summaries = {d: base.read(OUT / d / 'summary.json') for d in base.DOMAINS}
    decisions = {}
    for method, control in [('dynamic_cal', 'fixed_cal'), ('dynamic_no_cal', 'fixed_no_cal')]:
        gain = all(s['comparisons'][method + '_vs_' + control]['relative_N10'] >= .005
                   and s['comparisons'][method + '_vs_' + control]['net_H10'] >= 0 for s in summaries.values())
        beats_gram = all(s['comparisons'][method + '_vs_GRAM']['relative_N10'] >= .01
                        and s['comparisons'][method + '_vs_GRAM']['net_H10'] >= 0 for s in summaries.values())
        decisions[method] = 'CANDIDATE_FOR_CLEAN_RETEST' if gain and beats_gram else 'NO_CLEAR_DUAL_DOMAIN_INCREMENT'
    for method, control in [('fixed_no_cal', 'fixed_cal'), ('dynamic_no_cal', 'dynamic_cal')]:
        little_loss = all(s['comparisons'][method + '_vs_' + control]['relative_N10'] >= -.005
                         and s['comparisons'][method + '_vs_' + control]['net_H10'] >= -1
                         and s['comparisons'][method + '_vs_' + control]['gained_H50']
                             - s['comparisons'][method + '_vs_' + control]['lost_H50'] >= -1
                         for s in summaries.values())
        decisions[method + '_simplification'] = ('CANDIDATE_SIMPLIFICATION_RETEST' if little_loss
                                                else 'OBSERVED_SIMPLIFICATION_TRADEOFF')
    result = dict(domains=summaries, decisions=decisions, test_read=False,
        automatic_new_experiments=False, independent_verification='verification.json',
        caveat='Historically reused development screen; heuristic local discrimination, not reliability or novelty evidence. Simplification tolerance is not a statistical equivalence test.')
    base.write(OUT / 'summary.json', result)
    lines = ['# LIME 前缀专家权重四组筛查', '', '状态：COMPLETED / VERIFIED。', '',
             '四组统一 alpha=0.3；动态权重系数=1，无新增选参。每域512名用户避开前两轮解码评价及门控拟合用户。',
             '仍为历史开发数据；不能代替最终测试、多种子复验或新颖性核查。', '']
    for domain, summary in summaries.items():
        lines += [f'## {domain}', '', '| 方法 | NDCG@10 | Hit@10 | Hit@50 |', '|---|---:|---:|---:|']
        for name, m in summary['metrics'].items():
            lines.append(f'| {name} | {m["N10"]:.6f} | {m["H10"]:.6f} | {m["H50"]:.6f} |')
        lines += ['']
    lines += ['## 资源筛查判定', '', json.dumps(decisions, ensure_ascii=False), '',
              '正增益与删减价值分开判定。阈值用于资源分配，不代表等效性、显著性或论文接受概率。',
              '所有四组、同规则后重排及直接LIME后重排结果完整保留。未自动追加实验。']
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
                    cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
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
        base.write(OUT / 'status.json', status)
        if pending or active:
            time.sleep(15)
    if len(done) == 2 and all(rc == 0 for rc in done.values()):
        # Separate audit process; parent continues updating status every 5 s.
        with (OUT / 'audit.log').open('x') as log:
            child = subprocess.Popen([sys.executable, '-m', 'experiment.phase20.verify_lime_prefix_expert'],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            while child.poll() is None:
                status.update(state='VERIFYING', updated_at=base.stamp(), audit_pid=child.pid,
                              elapsed_seconds=time.monotonic() - started)
                base.write(OUT / 'status.json', status)
                time.sleep(5)
        if child.returncode == 0:
            status.update(state='COMPLETED', decisions=finish(), independently_verified=True)
        else:
            status.update(state='AUDIT_FAILED', audit_exit_code=child.returncode)
    else:
        status['state'] = 'INCOMPLETE'
    status.update(updated_at=base.stamp(), elapsed_seconds=time.monotonic() - started)
    base.write(OUT / 'status.json', status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'supervise', 'worker'])
    parser.add_argument('--domain', choices=base.DOMAINS)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'supervise':
        assert base.read(OUT / 'status.json')['state'] == 'PREPARED'
        assert not (OUT / 'launch.json').exists()
        base.write(OUT / 'launch.json', dict(pid=os.getpid(), started_at=base.stamp(), host_tmux=True))
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
