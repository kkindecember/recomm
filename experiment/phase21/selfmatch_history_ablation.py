"""Independent 2x2 text-self-matching/history-penalty development ablation.

Existing expert scoring is LIME-derived; probability guidance is AGRec-inspired.
Never change the concurrently running fixed-expert screening experiment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from experiment.phase21 import fixed_expert_screen_4096 as F
from experiment.phase21 import selfmatch_history_core as C
from experiment.phase20.catalog_scope_core import metrics, contrast, utilities

B = F.B
ROOT = F.ROOT
SOURCE = F.OUT
OUT = ROOT/'artifacts/phase21/selfmatch_history_1024_20261017_v1'
PLAN = ROOT/'plan/第二十一阶段/21.50_SASRec与BGE自匹配及历史惩罚对照_20261017.md'
REPORT = ROOT/'report/第二十一阶段/SASRec与BGE自匹配及历史惩罚对照_20261017.md'
MODULE = 'experiment.phase21.selfmatch_history_ablation'
DOMAINS = F.DOMAINS
JOBS = {f'{d}_{shard}': dict(domain=d, shard=shard) for shard in range(2) for d in DOMAINS}
CACHED = ('native', 'pair_raw', 'sas_only', 'bge_only')
METHODS = ('native', 'sas_only', 'bge_only', *C.ARMS)
CFG = dict(users_per_domain=1024, source_validation_slice=[1024, 2048],
    alpha=.3, beta_pair=[.5, .5], history_penalty_lambda=.05, score_scale=20.,
    beam_size=50, length_penalty=1., arms=list(C.ARMS), fresh_arms=list(C.FRESH_ARMS),
    history_filter=False, penalty_scope='full_history', penalty_location='fused expert scores before softmax',
    self_exclusion='candidate leave-all-own-occurrences-out, original minmax, clip [0,1], no extra old-score cap',
    generic_control='uniform vs uniform_penalty; both alpha=.3; separate native reference',
    fusion_network=False, training_steps=0, test_read=False,
    source=str(SOURCE.relative_to(ROOT)), shards_per_domain=2, max_parallel=2,
    avoid_active_source_gpus=True, total_lease_mib=16384, expected_workload_peak_mib=12288,
    minimum_free_mib=18432, allocation_limit_gib=11, worker_timeout_seconds=43200,
    resource_wait_seconds=86400, scientific_auto_retry=False, bootstrap_repeats=2000,
    stop_after='all four main arms and uniform controls on all1024/domain; no automatic next experiment')
read, write, sha, stamp = F.read, F.write, F.sha, F.stamp


def indices(shard):
    return list(range(shard, CFG['users_per_domain'], CFG['shards_per_domain']))


def snapshot_references(domain, selected):
    rows, provenance = {}, []
    for shard in range(2):
        path = SOURCE/'jobs'/f'{domain}_{shard}'/'predictions.jsonl'
        with path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                # Read only complete, already-written records before source index2048.
                row = json.loads(line)
                j = row['sample_index']
                if j >= 2048:
                    break
                if j < 1024:
                    continue
                local = j-1024
                assert local not in rows
                s = selected[local]
                assert all(row[k] == s[k] for k in ('user_id', 'target', 'history', 'evaluation_block'))
                rows[local] = dict(sample_index=local, source_validation_index=j,
                    user_id=s['user_id'], target=s['target'], history=s['history'],
                    evaluation_block=s['evaluation_block'], native=row['native'],
                    pair_raw=row['SASRec__BGE'], sas_only=row['SASRec__only'], bge_only=row['BGE__only'])
                provenance.append(dict(source=str(path.relative_to(ROOT)), line_number=line_number,
                    source_validation_index=j, source_line_sha256=hashlib.sha256(line.encode()).hexdigest()))
    assert sorted(rows) == list(range(1024)), 'Required source predictions are not yet available; do not launch'
    return [rows[j] for j in range(1024)], provenance


def prepare():
    torch.set_num_threads(2)
    checks = C.tests()
    OUT.mkdir(exist_ok=False)
    write(OUT/'status.json', dict(state='PREPARING', updated_at=stamp(), test_read=False))
    write(OUT/'config.json', CFG)
    write(OUT/'checks.json', checks)
    shutil.copy2(PLAN, OUT/'protocol.md')
    previous = read(SOURCE/'manifest.json')['sha256']
    manifest, domain_inputs = {}, {}

    def use(path, expected=None):
        rel = str(path.relative_to(ROOT))
        digest = manifest.get(rel) or sha(path)
        assert expected is None or digest == expected, rel
        manifest[rel] = digest
        return rel

    common = [use(ROOT/rel, digest) for rel, digest in previous.items()
              if rel.startswith(('experiment/', 'GRAM/src/', '.cache/huggingface/')) and
              (ROOT/rel).suffix in ('.py', '.sh', '.json', '.txt', '.model')]
    for path in (Path(__file__), ROOT/'experiment/phase21/selfmatch_history_core.py', PLAN,
                 OUT/'config.json', ROOT/'experiment/gpu_memory_lease.py'):
        rel = use(path)
        if rel not in common:
            common.append(rel)
    for d in DOMAINS:
        folder = OUT/d
        folder.mkdir()
        files = []
        write(OUT/'status.json', dict(state='PREPARING', stage='freeze_inputs_and_text_scores',
            domain=d, updated_at=stamp(), test_read=False))
        all_samples = read(SOURCE/d/'samples.json')
        selected = [dict(s, source_validation_index=j, validation_index=j-1024)
                    for j, s in enumerate(all_samples) if 1024 <= j < 2048]
        assert len(selected) == len({s['user_id'] for s in selected}) == 1024
        assert {s['user_id'] for s in selected}.isdisjoint(s['user_id'] for s in all_samples[:1024])
        fit_path = B.OUT/d/'fit_samples.json'
        fit = read(fit_path)
        assert {s['user_id'] for s in selected}.isdisjoint(s['user_id'] for s in fit)
        refs, provenance = snapshot_references(d, selected)
        write(folder/'samples.json', selected)
        write(folder/'references.json', refs)
        write(folder/'reference_provenance.json', provenance)
        for name in ('catalog.json', 'data_config.json', 'teacher_checks.json'):
            source = SOURCE/d/name
            files.append(use(source, previous[str(source.relative_to(ROOT))]))
            shutil.copy2(source, folder/name)
        dc = read(folder/'data_config.json')
        for name in ('parent_checkpoint', 'embedding', 'original_config'):
            path = ROOT/dc[name]
            files.append(use(path, dc['parent_sha256'] if name == 'parent_checkpoint' else previous.get(str(path.relative_to(ROOT)))))
        source = B.OUT/d/'features/validation/experts.npy'
        files.append(use(source, previous[str(source.relative_to(ROOT))]))
        files.append(use(SOURCE/d/'samples.json', previous[str((SOURCE/d/'samples.json').relative_to(ROOT))]))
        files.append(use(fit_path))
        raw = np.load(source, mmap_mode='r')[1536:2560, :, :]
        pair = np.asarray(raw[:, :, [B.EXPERTS.index('SASRec'), B.EXPERTS.index('BGE')]])
        assert pair.shape[:1] == (1024,) and pair.shape[2] == 2 and np.isfinite(pair).all()
        np.save(folder/'raw_experts.npy', pair)
        packed = np.load(ROOT/dc['embedding'])
        assert packed['item_ids'].tolist() == read(folder/'catalog.json')['items'][1:]
        embeddings = packed['embeddings']
        loo = np.lib.format.open_memmap(folder/'loo_text.npy', mode='w+', dtype='float32', shape=pair.shape[:2])
        for start in range(0, 1024, 32):
            stop = start+32
            loo[start:stop] = C.leave_self_out([s['full_history'] for s in selected[start:stop]],
                                            embeddings, pair[start:stop, :, 1])
            write(OUT/'status.json', dict(state='PREPARING', stage='text_leave_self_out',
                domain=d, users_completed=stop, users_total=1024, updated_at=stamp(), test_read=False))
        loo.flush()
        maximum_nonhistory_change = 0.
        upward_changes = 0
        for j, s in enumerate(selected):
            mask = np.isin(np.arange(1, pair.shape[1]+1), s['full_history'])
            np.testing.assert_array_equal(loo[j, ~mask], pair[j, ~mask, 1])
            upward_changes += int((loo[j, mask] > pair[j, mask, 1]+3e-6).sum())
            value = C.arm_scores(pair[j, :, 0], pair[j, :, 1], loo[j], s['full_history'])
            np.testing.assert_array_equal(value['pair_raw'], F.fused(raw[j], 'SASRec__BGE'))
        write(folder/'cohort_checks.json', dict(users=1024, source_indices=[1024, 2048],
            previous1024_disjoint=True, fitting_users_disjoint=True, target_based_selection=False,
            target_in_history=sum(s['target'] in s['full_history'] for s in selected),
            no_test_read=True, nonhistory_text_change=maximum_nonhistory_change,
            historical_text_upward_changes_over_tolerance=upward_changes,
            original_fusion_exact=True, block_counts=np.bincount([s['evaluation_block'] for s in selected], minlength=5).tolist()))
        for name in ('samples.json', 'references.json', 'reference_provenance.json', 'catalog.json',
                     'data_config.json', 'teacher_checks.json', 'raw_experts.npy', 'loo_text.npy', 'cohort_checks.json'):
            files.append(use(folder/name))
        domain_inputs[d] = list(dict.fromkeys(common+files))
    for job, spec in JOBS.items():
        (OUT/'jobs'/job).mkdir(parents=True)
        write(OUT/'jobs'/job/'status.json', dict(state='PREPARED', **spec, users_total=512,
            users_completed=0, updated_at=stamp()))
    for rel in manifest:
        path = ROOT/rel
        if path.suffix in ('.py', '.sh', '.md'):
            dest = OUT/'sources'/rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    write(OUT/'manifest.json', dict(sha256=manifest, domain_inputs=domain_inputs))
    write(OUT/'status.json', dict(state='PREPARED', users_total=2048, users_completed=0,
        fresh_generations_total=10240, updated_at=stamp(), test_read=False))
    REPORT.write_text('# SASRec+BGE自匹配及历史惩罚对照\n\n状态：PREPARED，尚无新结果。\n')
    print(json.dumps(dict(state='PREPARED', users_per_domain=1024, hashes=len(manifest), checks=checks)), flush=True)


class Generation(B.CLEAN.Worker):
    def __init__(self, job, lease_pid):
        self.job, self.spec = job, JOBS[job]
        self.jobfolder = OUT/'jobs'/job
        self.row_seconds = []
        self.users_completed = 0
        self.lease = B.S.LeaseHandle(lease_pid)
        B.CLEAN.OUT = OUT
        super().__init__(self.spec['domain'])
        self.truth = self.samples
        from experiment.phase18.core.diff_data import DiffDataset
        self.samples = [dict(s, target=1) for s in self.truth]
        self.data = DiffDataset(self.samples, self.catalog)
        self.refs = read(self.folder/'references.json')
        self.raw = np.load(self.folder/'raw_experts.npy', mmap_mode='r')
        self.loo = np.load(self.folder/'loo_text.npy', mmap_mode='r')
        assert self.initial_digest == read(self.folder/'teacher_checks.json')['model_digest']

    def context(self, j, override_index=None):
        return super().context(j, override_index=j)

    def scores(self, j):
        return C.arm_scores(self.raw[j, :, 0], self.raw[j, :, 1], self.loo[j],
                            self.truth[j]['full_history'], CFG['history_penalty_lambda'])

    def status(self, state='RUNNING', **extra):
        assert self.lease.poll() is None, 'GPU lease exited'
        done = self.users_completed
        rate = float(np.median(self.row_seconds[-32:])) if self.row_seconds else None
        write(self.jobfolder/'status.json', dict(state=state, stage=self.stage, job=self.job,
            domain=self.spec['domain'], shard=self.spec['shard'], pid=os.getpid(), updated_at=stamp(),
            gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'), users_completed=done, users_total=512,
            completed_decodes=self.completed, fresh_generations_completed=done*len(C.FRESH_ARMS),
            median_seconds_per_user=rate, elapsed_seconds=time.monotonic()-self.started,
            remaining_generation_seconds_estimate=(512-done)*rate if rate else None,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            alpha=.3, beta_pair=[.5, .5], history_penalty_lambda=.05, history_filter=False,
            training_steps=0, test_read=False, **extra))

    def run(self):
        ix = indices(self.spec['shard'])
        self.stage = 'native_pair_zero_target_smoke'
        self.status()
        checks = []
        for j in (ix[0], ix[len(ix)//2], ix[-1]):
            ctx = self.context(j)
            native = self.decode(ctx)
            B.Generation.equal(dict(ids=native['ids'], scores=native['scores'].tolist()), self.refs[j]['native'])
            q = self.index.conditional_logs(self.scores(j)['pair_raw'])
            pair = self.decode(ctx, q, .3)
            B.Generation.equal(dict(ids=pair['ids'], scores=pair['scores'].tolist()), self.refs[j]['pair_raw'])
            zero = self.decode(ctx, q, 0.)
            assert native['paths'] == zero['paths']
            np.testing.assert_array_equal(native['scores'], zero['scores'])
            plain = self.decode(ctx, native_entry=True, record=False)
            assert native['paths'] == plain['paths']
            np.testing.assert_array_equal(native['scores'], plain['scores'])
            self.samples[j]['target'] = 2
            changed = self.context(j)
            self.samples[j]['target'] = 1
            torch.testing.assert_close(ctx['ids'], changed['ids'], atol=0, rtol=0)
            torch.testing.assert_close(ctx['mask'], changed['mask'], atol=0, rtol=0)
            checks.append(dict(row=j, native_reference_matches=True, raw_pair_reference_matches=True,
                cached_score_atol=3e-6, cached_score_rtol=2e-5,
                alpha_zero_exact=True, native_entry_exact=True, target_placeholder_invariant=True))
        write(self.jobfolder/'smoke.json', dict(state='PASSED', checks=checks))
        self.stage = 'selfmatch_history_factorial_generation'
        self.status()
        with (self.jobfolder/'predictions.jsonl').open('x') as stream:
            for j in ix:
                begin = time.monotonic()
                ctx = self.context(j)
                s = self.truth[j]
                row = dict(sample_index=j, source_validation_index=s['source_validation_index'],
                    user_id=s['user_id'], target=s['target'], history=s['history'],
                    evaluation_block=s['evaluation_block'], standalone_ranks={}, diagnostics={})
                for arm in CACHED:
                    row[arm] = self.refs[j][arm]
                for arm, score in self.scores(j).items():
                    q = self.index.conditional_logs(score, temperature_scale=20.)
                    if arm in C.FRESH_ARMS:
                        prediction = self.decode(ctx, q, .3)
                        row[arm] = dict(ids=prediction['ids'], scores=prediction['scores'].tolist())
                    row['standalone_ranks'][arm] = F.full_rank(score, s['target'])
                    weights = np.exp(20*(score.astype(float)-float(score.max())))
                    mask = np.isin(np.arange(1, len(score)+1), s['full_history'])
                    row['diagnostics'][arm] = dict(expert_history_probability=float(weights[mask].sum()/weights.sum()))
                stream.write(json.dumps(row)+'\n')
                stream.flush()
                self.users_completed += 1
                self.row_seconds.append(time.monotonic()-begin)
                if self.users_completed == 1 or self.users_completed % 8 == 0 or j == ix[-1]:
                    os.fsync(stream.fileno())
                    self.status()
        assert self.digest() == self.initial_digest
        write(self.jobfolder/'runtime_checks.json', dict(state='PASSED', users=512,
            all_prescribed_arms_per_user=True, gram_digest_unchanged=True, history_filter=False,
            targets_only_for_metrics=True, fusion_network=False, training_steps=0))
        self.stage = 'complete'
        self.status('COMPLETED')


def worker(job):
    # Reuse only the established lease/process wrapper in this separate process.
    F.OUT, F.CFG, F.JOBS, F.MODULE = OUT, CFG, JOBS, MODULE
    return F.worker(job)


def effect_interval(values, seed=20261017):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    boot = np.empty(CFG['bootstrap_repeats'])
    for i in range(len(boot)):
        boot[i] = values[rng.integers(0, len(values), len(values))].mean()
    return dict(delta_N10=float(values.mean()), descriptive_95pct_N10=np.quantile(boot, [.025, .975]).tolist())


def summarize():
    result = {}
    pairs = [('pair_loo', 'pair_raw'), ('pair_loo_penalty', 'pair_raw_penalty'),
             ('pair_raw_penalty', 'pair_raw'), ('pair_loo_penalty', 'pair_loo'),
             ('uniform_penalty', 'uniform'), ('uniform', 'native')]
    pairs += [(a, 'native') for a in C.PAIR_ARMS]
    pairs += [(a, 'sas_only') for a in C.PAIR_ARMS]
    lines = ['# SASRec+BGE自匹配及历史惩罚对照', '', '## Material Passport', '',
        '- Origin Skill: academic-research-suite / experiment-agent', '- Origin Mode: run / validate',
        '- Origin Date: 2026-10-17', '- Verification Status: ANALYZED; saved predictions independently ranked',
        '- Version Label: selfmatch_history_1024_20261017_v1', '',
        '每域wide4096索引[1024,2048)的1024人；与旧前1024不重叠，仍是开发验证集。',
        'α=.3、β各.5，λ预先固定.05。历史商品全程合法；没有训练或测试集读取。',
        'uniform两组用无专家内容的均匀叶子先验和相同引导接口；单独保留原生GRAM，不能把uniform_penalty直接称为原生GRAM只加惩罚。', '']
    for d in DOMAINS:
        folder = OUT/d
        samples = read(folder/'samples.json')
        refs = read(folder/'references.json')
        raw = np.load(folder/'raw_experts.npy', mmap_mode='r')
        loo = np.load(folder/'loo_text.npy', mmap_mode='r')
        rows = {}
        for job, spec in JOBS.items():
            if spec['domain'] != d:
                continue
            p = OUT/'jobs'/job
            assert read(p/'runtime_checks.json')['state'] == 'PASSED'
            assert read(p/'lease_status.json')['state'] == 'released'
            shard_rows = [json.loads(x) for x in (p/'predictions.jsonl').read_text().splitlines()]
            assert [x['sample_index'] for x in shard_rows] == indices(spec['shard'])
            for row in shard_rows:
                assert row['sample_index'] not in rows
                rows[row['sample_index']] = row
        assert sorted(rows) == list(range(1024))
        rr = {a: [] for a in METHODS}
        pure = {a: [] for a in C.ARMS}
        history_top10 = {a: [] for a in METHODS}
        history_top50 = {a: [] for a in METHODS}
        history_probability = {a: [] for a in C.ARMS}
        with (folder/'predictions.jsonl').open('x') as stream:
            for j, s in enumerate(samples):
                row = rows[j]
                assert all(row[k] == s[k] for k in ('user_id', 'target', 'history', 'evaluation_block', 'source_validation_index'))
                for a in CACHED:
                    assert row[a] == refs[j][a]
                for a in METHODS:
                    entry = row[a]
                    assert len(entry['ids']) == len(set(entry['ids'])) == len(entry['scores']) == 50
                    assert np.isfinite(entry['scores']).all()
                    rr[a].append(B.rank(entry, s['target']))
                    ordered = sorted(zip(entry['ids'], entry['scores']), key=lambda x: (-x[1], x[0]))
                    items = [x[0] for x in ordered]
                    assert min(items) >= 1 and max(items) <= raw.shape[1]
                    history_top10[a].append(float(np.isin(items[:10], s['full_history']).mean()))
                    history_top50[a].append(float(np.isin(items, s['full_history']).mean()))
                scores = C.arm_scores(raw[j, :, 0], raw[j, :, 1], loo[j], s['full_history'])
                for a, score in scores.items():
                    measured = F.full_rank(score, s['target'])
                    assert measured == row['standalone_ranks'][a]
                    pure[a].append(measured)
                    weights = np.exp(20*(score.astype(float)-float(score.max())))
                    mask = np.isin(np.arange(1, len(score)+1), s['full_history'])
                    mass = float(weights[mask].sum()/weights.sum())
                    np.testing.assert_allclose(mass, row['diagnostics'][a]['expert_history_probability'], atol=1e-12)
                    history_probability[a].append(mass)
                stream.write(json.dumps(row)+'\n')
        rr = {k: np.asarray(v, np.int64) for k, v in rr.items()}
        pure = {k: np.asarray(v, np.int64) for k, v in pure.items()}
        group_ids = np.asarray([s['evaluation_block'] for s in samples])
        groups = dict(all1024=np.arange(1024), first512=np.arange(512), second512=np.arange(512, 1024))
        groups.update({f'block{b+1}': np.flatnonzero(group_ids == b) for b in range(5)})
        details = {}
        interaction = (utilities(rr['pair_loo_penalty'])-utilities(rr['pair_loo'])-
                       utilities(rr['pair_raw_penalty'])+utilities(rr['pair_raw']))
        for group, ix in groups.items():
            assert len(ix)
            details[group] = dict(users=len(ix), metrics={k: metrics(v[ix]) for k, v in rr.items()},
                standalone_metrics={k: metrics(v[ix]) for k, v in pure.items()},
                comparisons={a+'_vs_'+b: contrast(rr[a][ix], rr[b][ix], repeats=2000, seed=20261017) for a, b in pairs},
                interaction_loo_by_penalty=effect_interval(interaction[ix]),
                mean_history_fraction_top10={a: float(np.asarray(v)[ix].mean()) for a, v in history_top10.items()},
                mean_history_fraction_top50={a: float(np.asarray(v)[ix].mean()) for a, v in history_top50.items()},
                mean_expert_history_probability={a: float(np.asarray(v)[ix].mean()) for a, v in history_probability.items()})
        result[d] = details
        np.savez_compressed(folder/'ranks.npz', users=np.asarray([s['user_id'] for s in samples]), evaluation_block=group_ids, **rr)
        np.savez_compressed(folder/'standalone_ranks.npz', **pure)
        write(folder/'summary.json', details)
        table = details['all1024']['metrics']
        lines += [f'## {d}', '', '| 方法 | NDCG@10 | 对GRAM | HR@10 | 历史商品占Top10 |', '|---|---:|---:|---:|---:|']
        for a in METHODS:
            m = table[a]
            relative = 100*(m['N10']/table['native']['N10']-1) if table['native']['N10'] else float('nan')
            fraction = details['all1024']['mean_history_fraction_top10'][a]
            lines.append(f'| {a} | {m["N10"]:.6f} | {relative:+.2f}% | {m["H10"]:.6f} | {fraction:.3%} |')
        lines += ['', '| 条件比较 | N10绝对差 | 描述性95%区间 |', '|---|---:|---|']
        for a, b in pairs[:6]:
            item = details['all1024']['comparisons'][a+'_vs_'+b]
            ci = item['descriptive_95pct_N10']
            lines.append(f'| {a} − {b} | {item["delta_N10"]:+.6f} | [{ci[0]:+.6f}, {ci[1]:+.6f}] |')
        value = details['all1024']['interaction_loo_by_penalty']
        lines += ['', f'去自匹配×惩罚的N10交互差：{value["delta_N10"]:+.6f}，区间{value["descriptive_95pct_N10"]}。', '']
        files = [folder/n for n in ('predictions.jsonl', 'ranks.npz', 'standalone_ranks.npz', 'summary.json')]
        for job, spec in JOBS.items():
            if spec['domain'] == d:
                files += [OUT/'jobs'/job/n for n in ('predictions.jsonl', 'smoke.json', 'runtime_checks.json', 'lease_status.json')]
        write(folder/'output_manifest.json', dict(sha256={str(p.relative_to(ROOT)): sha(p) for p in files}))
    write(OUT/'summary.json', dict(domains=result, config=CFG, test_read=False,
        descriptive_comparisons=len(pairs)*8*2, interaction_intervals=16,
        independent_prediction_rank_recomputation=True, automatic_final_selection=False,
        statistical_review=dict(coverage='11/11',
            simpson='Domains, batches, and blocks separated; no cross-domain selected winner.',
            ecological='Per-user gains/losses retained; no universal user benefit claim.',
            berkson='Fixed eligible validation cohort, not raw population.',
            collider='User selection uses indices, never outcomes.',
            base_rate='Every prescribed user retained in every arm.',
            regression_to_mean='Repeated development exploration, no independent confirmation claim.',
            survivorship='All arms and failures retained; incomplete cannot summarize as complete.',
            look_elsewhere='All planned intervals saved; unadjusted and descriptive.',
            forking_paths='Lambda, cohort, alpha, beta, and scorer frozen before generation.',
            causation='Conditional ablation of this frozen system; uniform prior control is explicit, not pure native-only penalty.',
            reverse_causality='Targets absent from scoring API and model encoding; metrics only.')))
    lines += ['所有配对区间为未经多重比较校正的开发集描述性bootstrap，不包含训练种子、专家选择或未来融合器变化的不确定性。',
        '此轮只筛查两个开关，不能直接宣称对其他专家或最终方法同样有效；保留五块、两批、负结果与逐用户预测。',
        '纯专家排序与实际GRAM生成分开报告；不能用历史占比下降代替推荐效果改善，也不能把所有通用抑制收益归给专家。']
    REPORT.write_text('\n'.join(lines)+'\n')


def protected_gpus():
    status = read(SOURCE/'status.json')
    return {a['gpu_uuid'] for a in status.get('active', {}).values()} if status['state'] in ('RUNNING', 'FINALIZING') else set()


def supervise():
    assert read(OUT/'status.json')['state'] == 'PREPARED'
    pending, active, finished = list(JOBS), {}, {}
    attempts = {j: 0 for j in JOBS}
    began = time.monotonic()
    try:
        while pending or active:
            for job, task in list(active.items()):
                code = task['process'].poll()
                if code is None and time.monotonic()-task['started'] > CFG['worker_timeout_seconds']+120:
                    B.stop(task['process'])
                    code = task['process'].returncode
                if code is not None:
                    task['log'].close()
                    p = OUT/'jobs'/job
                    status = read(p/'status.json')
                    if code == 75 and status['state'] == 'ADMISSION_DEFERRED':
                        pending.append(job)
                    else:
                        released = read(p/'lease_status.json').get('state') == 'released' if (p/'lease_status.json').exists() else True
                        finished[job] = dict(exit_code=code, state=status['state'], lease_released=released)
                    del active[job]
            excluded = protected_gpus() | {t['gpu']['uuid'] for t in active.values()}
            available = sorted([g for g in B.S.OLD.gpu_inventory() if g['uuid'] not in excluded
                                and g['free_mib'] >= CFG['minimum_free_mib']],
                               key=lambda g: (g['utilization'], -g['free_mib'], g['index']))
            for job in list(pending):
                if not available or len(active) >= CFG['max_parallel']:
                    break
                gpu = available.pop(0)
                pending.remove(job)
                attempts[job] += 1
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['uuid'], OMP_NUM_THREADS='2',
                           MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
                log = (OUT/'jobs'/job/f'worker_attempt{attempts[job]}.log').open('x')
                write(OUT/'jobs'/job/'status.json', dict(state='STARTING', **JOBS[job],
                    users_total=512, users_completed=0, gpu_index=gpu['index'], updated_at=stamp()))
                process = subprocess.Popen([sys.executable, '-u', '-m', MODULE, 'worker', '--job', job],
                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active[job] = dict(process=process, log=log, gpu=gpu, started=time.monotonic())
            if pending and time.monotonic()-began > CFG['resource_wait_seconds']:
                for job in pending:
                    finished[job] = dict(exit_code=124, state='RESOURCE_WAIT_LIMIT', lease_released=True)
                pending = []
            statuses = {j: read(OUT/'jobs'/j/'status.json') for j in JOBS}
            count = sum(s.get('users_completed', 0) for s in statuses.values())
            estimates = [statuses[j].get('remaining_generation_seconds_estimate') for j in active]
            known_estimates = [e for e in estimates if e is not None]
            # Schedule pending full jobs onto the observed active-job finish times.
            rates = [s.get('median_seconds_per_user') for s in statuses.values()]
            rates = [r for r in rates if r is not None]
            remaining = None
            if known_estimates and len(known_estimates) == len(active) and rates:
                slots = sorted(known_estimates + [0.]*(CFG['max_parallel']-len(active)))
                for job in pending:
                    domain_rates = [statuses[j].get('median_seconds_per_user') for j in JOBS
                                    if JOBS[j]['domain'] == JOBS[job]['domain']]
                    domain_rates = [r for r in domain_rates if r is not None]
                    full_job = 512*float(np.median(domain_rates or rates))
                    slots[0] += full_job
                    slots.sort()
                remaining = max(slots)
            write(OUT/'status.json', dict(state='RUNNING' if active else 'WAITING_FOR_RESOURCES' if pending else 'FINALIZING',
                pid=os.getpid(), updated_at=stamp(), elapsed_seconds=time.monotonic()-began,
                users_total=2048, users_completed=count, fresh_generations_total=10240,
                fresh_generations_completed=count*len(C.FRESH_ARMS), history_filter=False, test_read=False,
                remaining_generation_seconds_estimate=remaining, eta_includes_pending_approximate=bool(pending),
                pending=pending, finished=finished, protected_source_gpu_uuids=sorted(protected_gpus()),
                active={j: dict(pid=t['process'].pid, gpu_index=t['gpu']['index'], gpu_uuid=t['gpu']['uuid'],
                    worker=statuses[j]) for j, t in active.items()}, jobs=statuses))
            if pending or active:
                time.sleep(10)
        complete = len(finished) == len(JOBS) and all(t['exit_code'] == 0 and t['state'] == 'COMPLETED'
                                                      and t['lease_released'] for t in finished.values())
        if complete:
            summarize()
        write(OUT/'status.json', dict(state='COMPLETED' if complete else 'INCOMPLETE', finished=finished,
            updated_at=stamp(), users_total=2048, users_completed=count, elapsed_seconds=time.monotonic()-began,
            test_read=False, report=str(REPORT.relative_to(ROOT)), next_experiment_started=False))
    finally:
        for task in active.values():
            B.stop(task['process'])
            task['log'].close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('test', 'prepare', 'supervise', 'worker', 'generate'))
    parser.add_argument('--job', choices=JOBS)
    parser.add_argument('--lease-pid', type=int)
    args = parser.parse_args()

    def interrupted(signum, frame):
        raise InterruptedError('Selfmatch/history ablation interrupted or hard timeout')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGALRM, interrupted)
    try:
        if args.mode == 'test':
            print(json.dumps(C.tests()), flush=True)
        elif args.mode == 'prepare':
            prepare()
        elif args.mode == 'supervise':
            supervise()
        elif args.mode == 'worker':
            return worker(args.job)
        else:
            Generation(args.job, args.lease_pid).run()
        return 0
    except BaseException as exc:
        if OUT.exists():
            p = OUT/'jobs'/args.job if args.job else OUT
            previous = read(p/'status.json') if (p/'status.json').exists() else {}
            value = dict(state='FAILED', error=repr(exc), traceback=traceback.format_exc(),
                         updated_at=stamp(), previous=previous, test_read=False)
            write(p/'failure.json', value)
            write(p/'status.json', value)
        raise


if __name__ == '__main__':
    sys.exit(main())
