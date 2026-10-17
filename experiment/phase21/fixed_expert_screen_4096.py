"""Frozen expert selection: 5 singles + 6 pairs, fixed alpha/beta, no gate.

All historical items remain eligible during scoring, generation and evaluation.
The five evaluation blocks are NOT five-fold retraining/cross-validation.
Existing LIME-derived scorers and AGRec-inspired guidance retain attribution.
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

from experiment.phase21 import old10_expert_selection as B
from experiment.phase20.catalog_scope_core import metrics, contrast

ROOT = B.ROOT
OUT = ROOT/'artifacts/phase21/fixed_expert_screen_4096_20261017_v1'
PLAN = ROOT/'plan/第二十一阶段/21.49_固定权重五专家4096用户生成筛选_20261017.md'
REPORT = ROOT/'report/第二十一阶段/固定权重五专家4096用户生成筛选_20261017.md'
MODULE = 'experiment.phase21.fixed_expert_screen_4096'
DOMAINS, EXPERTS = B.DOMAINS, B.EXPERTS
PAIRS = {a+'__'+b: (a, b) for a in EXPERTS[:3] for b in EXPERTS[3:]}
SINGLES = {a+'__only': (a,) for a in EXPERTS}
ARMS = dict(SINGLES, **PAIRS)
JOBS = {d+'_'+str(s): dict(domain=d, shard=s) for d in DOMAINS for s in range(2)}
CFG = dict(alpha=.3, beta_pair=[.5, .5], beta_single=[1.], arms=ARMS,
    experts=list(EXPERTS), users_per_domain=4096, evaluation_blocks=5,
    block_seed='fixed-expert-screen-20261017', block_policy='user hash order, round robin 5',
    shards_per_domain=2, max_parallel=4, score_scale=20., beam_size=50, length_penalty=1.,
    fusion_network=False, fusion_training_steps=0, expert_training_steps=0, gram_optimizer_steps=0,
    history_filter=False, history_penalty=False, text_self_exclusion=False,
    total_lease_mib=16384, expected_workload_peak_mib=12288, minimum_free_mib=18432,
    allocation_limit_gib=11, worker_timeout_seconds=43200, resource_wait_seconds=86400,
    scientific_auto_retry=False, test_read=False, evaluation_only=True,
    cached_expert_offset=512, cached_expert_source=str(B.OUT.relative_to(ROOT)),
    evaluation='existing wide4096 validation cohort; includes previous 1024; not independent test',
    baseline='same-user native predictions with fresh zero/native checks in every shard',
    standalone_ranking='full catalog fixed expert score rankings, separate from GRAM-guided generation',
    stop_after='all 11 arms for all 4096 users in both domains; no pruning or next experiment')
read, sha, stamp = B.read, B.sha, B.stamp


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def indices(shard):
    return list(range(shard, CFG['users_per_domain'], CFG['shards_per_domain']))


def blocks(samples, domain):
    order = sorted(range(len(samples)), key=lambda j: hashlib.sha256(
        f'{CFG["block_seed"]}|{domain}|{samples[j]["user_id"]}'.encode()).digest())
    assigned = np.empty(len(samples), np.int64)
    assigned[order] = np.arange(len(samples)) % CFG['evaluation_blocks']
    return assigned


def fused(experts, arm):
    columns = [EXPERTS.index(n) for n in ARMS[arm]]
    if len(columns) == 1:
        return np.asarray(experts[:, columns[0]])
    return np.float32(.5)*experts[:, columns[0]] + np.float32(.5)*experts[:, columns[1]]


def full_rank(scores, target):
    """Metric only; all items remain legal, ties by ascending item ID."""
    scores = np.asarray(scores)
    value = scores[target-1]
    return int(1 + np.count_nonzero(scores > value) + np.count_nonzero(scores[:target-1] == value))


def tests():
    from experiment.phase20.lime_decode_screen_core import ExpertTrie
    # Independent direct leaf summation checks every branch, not decoder output.
    paths = [[0, 2, 4, 1], [0, 2, 5, 1], [0, 3, 1], [0, 3, 7, 1]]
    # Terminal nodes cannot have descendants in an actual ID trie.
    paths[2] = [0, 3, 6, 1]
    trie = ExpertTrie(paths, range(1, 5))
    e = np.random.default_rng(2149).random((4, 5)).astype(np.float32)
    for arm in ARMS:
        score = fused(e, arm)
        leaves = np.exp(20*(score.astype(np.float64)-float(score.max())))
        logs = trie.conditional_logs(score)
        for prefix, values in logs.items():
            selected = [j for j, p in enumerate(paths) if tuple(p[:len(prefix)]) == prefix]
            for token, value in values.items():
                child = [j for j in selected if paths[j][len(prefix)] == token]
                np.testing.assert_allclose(np.exp(value), leaves[child].sum()/leaves[selected].sum(), rtol=1e-12)
        if len(ARMS[arm]) == 1:
            np.testing.assert_array_equal(score, e[:, EXPERTS.index(ARMS[arm][0])])
    assert full_rank(np.array([3., 3., 2., 1.]), 2) == 2
    assert full_rank(np.array([3., 3., 2., 1.]), 1) == 1
    # A previously seen target is not removed by the rank helper.
    assert B.rank(dict(ids=[1, 2, 3], scores=[3., 2., 1.]), 1, beam=3) == 1
    rows = [dict(user_id=str(j)) for j in range(4096)]
    assignment = blocks(rows, 'Toys')
    np.testing.assert_array_equal(np.bincount(assignment), [820, 819, 819, 819, 819])
    assert set(indices(0)).isdisjoint(indices(1)) and sorted(indices(0)+indices(1)) == list(range(4096))
    return dict(state='PASSED', all_11_fusions_checked=True, independent_prefix_mass_check=True,
        single_scores_exact=True, unfiltered_metric_and_tie_checks=True,
        disjoint_complete_shards=True, five_balanced_deterministic_blocks=True, no_learned_weights=True)


def prepare():
    checks = tests()
    OUT.mkdir(exist_ok=False)
    write(OUT/'status.json', dict(state='PREPARING', updated_at=stamp(), test_read=False))
    write(OUT/'config.json', CFG); write(OUT/'checks.json', checks)
    shutil.copy2(PLAN, OUT/'protocol.md')
    manifest, domain_inputs = {}, {}

    def use(path, expected=None):
        rel = str(path.relative_to(ROOT))
        digest = manifest.get(rel) or sha(path)
        assert expected is None or digest == expected, rel
        manifest[rel] = digest
        return rel

    previous = read(B.OUT/'manifest.json')['sha256']
    common = [use(ROOT/rel, digest) for rel, digest in previous.items()
              if rel.startswith(('experiment/', 'GRAM/src/', '.cache/huggingface/'))]
    for path in (Path(__file__), PLAN, OUT/'config.json', ROOT/'experiment/gpu_memory_lease.py'):
        rel = use(path)
        if rel not in common: common.append(rel)
    for d in DOMAINS:
        p = OUT/d; p.mkdir(); files = []
        for name in ('catalog.json', 'data_config.json', 'teacher_checks.json'):
            source = B.OUT/d/name
            files.append(use(source, previous[str(source.relative_to(ROOT))]))
            shutil.copy2(source, p/name); files.append(use(p/name))
        original = read(B.OUT/d/'samples.json')
        fit = read(B.OUT/d/'fit_samples.json')
        chosen = read(B.SCREEN/d/'wide4096_samples.json')
        reference_file = B.SCREEN/d/'wide4096_generation_references.jsonl'
        refs = [json.loads(line) for line in reference_file.read_text().splitlines()]
        assert len(chosen) == len(refs) == 4096
        assert len(set(s['user_id'] for s in chosen)) == 4096
        assert {s['user_id'] for s in chosen}.isdisjoint(s['user_id'] for s in fit)
        assigned = blocks(chosen, d)
        rows = []
        for j, (s, r) in enumerate(zip(chosen, refs)):
            assert s['training_holdout_role'] == 'none'
            assert all(s[k] == r[k] == original[j+512][k]
                       for k in ('user_id', 'target', 'history', 'full_history') if k in r)
            assert s['full_history'] == original[j+512]['full_history']
            assert s['history'] == s['full_history'][-20:]
            rows.append(dict(s, validation_index=j, evaluation_block=int(assigned[j]),
                             cohort='wide4096', full_validation_index=r['sample_index']))
        write(p/'samples.json', rows); write(p/'references.json', refs)
        write(p/'cohort_checks.json', dict(users=4096, same_cached_order=True,
            fit_users_disjoint=True, no_training_holdout_users=True, prior1024_included=True,
            target_used_for_user_selection=False, block_counts=np.bincount(assigned).tolist(),
            validation_target_in_full_history=sum(s['target'] in s['full_history'] for s in rows),
            no_test_read=True, frozen_expert_checkpoints=True))
        source = B.OUT/d/'features/validation/experts.npy'
        cache_manifest = read(B.OUT/d/'feature_manifest.json')['sha256']
        files.append(use(source, cache_manifest[str(source.relative_to(ROOT))]))
        meta = read(B.OUT/d/'features/validation/metadata.json')
        assert meta['expert_channels'] == list(EXPERTS)
        assert meta['users_order'][512:] == [s['user_id'] for s in rows]
        array = np.load(source, mmap_mode='r')
        assert array.shape == (4608, len(read(p/'catalog.json')['items'])-1, 5)
        # Every source value is audited; frozen scores already use original minmax normalization.
        for begin in range(512, len(array), 128):
            value = array[begin:begin+128]
            assert np.isfinite(value).all() and value.min() >= -1e-6 and value.max() <= 1+1e-6
        for path in (B.OUT/d/'features/validation/metadata.json', B.OUT/d/'feature_manifest.json',
                     B.OUT/d/'samples.json', B.OUT/d/'fit_samples.json', reference_file,
                     B.SCREEN/d/'wide4096_samples.json', B.SCREEN/d/'wide4096_metadata.json',
                     p/'samples.json', p/'references.json', p/'cohort_checks.json'):
            files.append(use(path))
        dc = read(p/'data_config.json')
        files.append(use(ROOT/dc['parent_checkpoint'], dc['parent_sha256']))
        domain_inputs[d] = common+files
        write(p/'status.json', dict(state='PREPARED', users_total=4096, users_completed=0, updated_at=stamp()))
    for job, spec in JOBS.items():
        (OUT/'jobs'/job).mkdir(parents=True)
        write(OUT/'jobs'/job/'status.json', dict(state='PREPARED', **spec, users_total=2048,
            users_completed=0, updated_at=stamp()))
    for rel in manifest:
        source = ROOT/rel
        if source.suffix in ('.py', '.sh', '.md'):
            dest = OUT/'sources'/rel; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
    write(OUT/'manifest.json', dict(sha256=manifest, domain_inputs=domain_inputs))
    write(OUT/'status.json', dict(state='PREPARED', updated_at=stamp(), test_read=False,
        users_total=8192, guided_generations_total=90112, training_steps=0))
    REPORT.write_text('# 固定权重五专家4096用户生成筛选\n\n状态：PREPARED，尚无本轮结果。\n')
    print(json.dumps(dict(state='PREPARED', hashes=len(manifest), checks=checks)), flush=True)


class Generation(B.CLEAN.Worker):
    def __init__(self, job, lease_pid):
        self.job, self.spec = job, JOBS[job]
        self.jobfolder = OUT/'jobs'/job
        self.row_seconds = []; self.lease = B.S.LeaseHandle(lease_pid)
        self.users_completed = 0
        B.CLEAN.OUT = OUT
        super().__init__(self.spec['domain'])
        self.truth = self.samples
        from experiment.phase18.core.diff_data import DiffDataset
        self.samples = [dict(s, target=1) for s in self.truth]
        self.data = DiffDataset(self.samples, self.catalog)
        self.refs = read(self.folder/'references.json')
        self.raw = np.load(B.OUT/self.domain/'features/validation/experts.npy', mmap_mode='r')[512:]
        assert self.initial_digest == read(self.folder/'teacher_checks.json')['model_digest']

    def context(self, j, override_index=None):
        return super().context(j, override_index=j)

    def status(self, state='RUNNING', **extra):
        assert self.lease.poll() is None, 'GPU lease exited'
        done = self.users_completed
        rate = float(np.median(self.row_seconds[-32:])) if self.row_seconds else None
        write(self.jobfolder/'status.json', dict(state=state, stage=self.stage, job=self.job,
            domain=self.spec['domain'], shard=self.spec['shard'], pid=os.getpid(), updated_at=stamp(),
            gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'), users_completed=done, users_total=2048,
            completed_decodes=self.completed, guided_generations_completed=done*11,
            median_seconds_per_user=rate, elapsed_seconds=time.monotonic()-self.started,
            remaining_generation_seconds_estimate=(2048-done)*rate if rate else None,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            history_filter=False, history_penalty=False, text_self_exclusion=False,
            alpha=.3, beta_pair=[.5, .5], training_steps=0, test_read=False, **extra))

    def run(self):
        ix = indices(self.spec['shard'])
        self.stage = 'native_zero_target_smoke'; self.status(); checks = []
        for j in (ix[0], ix[len(ix)//2], ix[-1]):
            ctx = self.context(j); native = self.decode(ctx)
            B.Generation.equal(dict(ids=native['ids'], scores=native['scores'].tolist()), self.refs[j]['native'])
            q = self.index.conditional_logs(fused(self.raw[j], 'SASRec__BGE'))
            zero = self.decode(ctx, q, 0.)
            assert native['paths'] == zero['paths']
            np.testing.assert_array_equal(native['scores'], zero['scores'])
            plain = self.decode(ctx, native_entry=True, record=False)
            assert native['paths'] == plain['paths']
            np.testing.assert_array_equal(native['scores'], plain['scores'])
            self.samples[j]['target'] = 2; changed = self.context(j); self.samples[j]['target'] = 1
            torch.testing.assert_close(ctx['ids'], changed['ids'], atol=0, rtol=0)
            torch.testing.assert_close(ctx['mask'], changed['mask'], atol=0, rtol=0)
            checks.append(dict(row=j, native_reference_matches=True, alpha_zero_exact=True,
                native_entry_exact=True, target_placeholder_invariant=True))
        write(self.jobfolder/'smoke.json', dict(state='PASSED', checks=checks))
        self.stage = 'fixed_weights_all_11_generation'; self.status()
        with (self.jobfolder/'predictions.jsonl').open('x') as stream:
            for j in ix:
                begin = time.monotonic(); ctx = self.context(j); s = self.truth[j]
                row = dict(sample_index=j, user_id=s['user_id'], target=s['target'], history=s['history'],
                    evaluation_block=s['evaluation_block'], native=self.refs[j]['native'], standalone_ranks={})
                for arm in ARMS:
                    score = fused(self.raw[j], arm)
                    q = self.index.conditional_logs(score, temperature_scale=CFG['score_scale'])
                    decoded = self.decode(ctx, q, CFG['alpha'])
                    row[arm] = dict(ids=decoded['ids'], scores=decoded['scores'].tolist())
                    row['standalone_ranks'][arm] = full_rank(score, s['target'])
                stream.write(json.dumps(row)+'\n'); stream.flush()
                self.users_completed += 1; self.row_seconds.append(time.monotonic()-begin)
                if self.users_completed == 1 or self.users_completed % 8 == 0 or j == ix[-1]:
                    os.fsync(stream.fileno()); self.status()
        assert self.digest() == self.initial_digest
        write(self.jobfolder/'runtime_checks.json', dict(state='PASSED', users=2048,
            all_11_arms_per_user=True, gram_digest_unchanged=True, history_filter=False,
            history_penalty=False, text_self_exclusion=False, fusion_network=False, training_steps=0))
        self.stage = 'complete'; self.status('COMPLETED')


def worker(job):
    spec = JOBS[job]; p = OUT/'jobs'/job; lease = child = None
    signal.alarm(CFG['worker_timeout_seconds'])
    try:
        assert read(OUT/'config.json') == json.loads(json.dumps(CFG))
        manifest = read(OUT/'manifest.json')
        for rel in manifest['domain_inputs'][spec['domain']]:
            assert sha(ROOT/rel) == manifest['sha256'][rel], rel
        gpu = os.environ['CUDA_VISIBLE_DEVICES']
        free = next(g['free_mib'] for g in B.S.OLD.gpu_inventory() if g['uuid'] == gpu)
        if free < CFG['minimum_free_mib']:
            write(p/'status.json', dict(state='ADMISSION_DEFERRED', free_mib=free, updated_at=stamp()))
            return 75
        with (p/'lease.log').open('a') as log:
            lease_path = p/'lease_status.json'
            lease = subprocess.Popen([sys.executable, str(ROOT/'experiment/gpu_memory_lease.py'), '--gpu', '0',
                '--total-lease-mib', str(CFG['total_lease_mib']), '--expected-workload-peak-mib',
                str(CFG['expected_workload_peak_mib']), '--status-path', str(lease_path)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write(p/'lease_process.json', dict(pid=lease.pid, gpu_uuid=gpu, started_at=stamp()))
            began = time.monotonic()
            while not lease_path.exists() or read(lease_path).get('state') != 'holding':
                assert lease.poll() is None, 'Lease failed'
                assert time.monotonic()-began < 180, 'Lease startup timed out'
                time.sleep(1)
            child = subprocess.Popen([sys.executable, '-u', '-m', MODULE, 'generate',
                '--job', job, '--lease-pid', str(lease.pid)], cwd=ROOT, start_new_session=True)
            assert child.wait() == 0, 'Generation failed; no scientific automatic retry'
        return 0
    finally:
        signal.alarm(0); B.stop(child); B.stop(lease)


def summarize():
    result = {}; lines = ['# 固定权重五专家4096用户生成筛选', '', '## Material Passport', '',
        '- Origin Skill: academic-research-suite / experiment-agent', '- Origin Mode: run / validate',
        '- Origin Date: 2026-10-17', '- Verification Status: ANALYZED; saved predictions independently ranked',
        '- Version Label: fixed_expert_screen_4096_v1', '',
        '固定α=0.3；双专家β各0.5，单专家权重1。全程允许历史商品，无历史惩罚和文本去自匹配。',
        '五个不重叠评估块用于观察同一批验证用户的稳定性，不是五折重训，也不是独立测试。',
        '每域4096包含此前1024，全部11组完整报告。单专家引导指GRAM加一个专家；纯专家排序另列。', '']
    comparisons = [(a, 'native') for a in ARMS]
    comparisons += [(a, n+'__only') for a, names in PAIRS.items() for n in names]
    # Match behavior/text replacements as well as contributions of both pair members.
    comparisons += [(a+'__'+t, b+'__'+t) for a, b in (('TiSASRec', 'SASRec'), ('ItemCF', 'SASRec'),
                    ('ItemCF', 'TiSASRec')) for t in EXPERTS[3:]]
    comparisons += [(a+'__Qwen4', a+'__BGE') for a in EXPERTS[:3]]
    for d in DOMAINS:
        p = OUT/d; samples = read(p/'samples.json'); rows = {}
        for job, spec in JOBS.items():
            if spec['domain'] != d: continue
            folder = OUT/'jobs'/job
            assert read(folder/'runtime_checks.json')['state'] == 'PASSED'
            assert read(folder/'lease_status.json')['state'] == 'released'
            items = [json.loads(x) for x in (folder/'predictions.jsonl').read_text().splitlines()]
            assert [r['sample_index'] for r in items] == indices(spec['shard'])
            for row in items:
                j = row['sample_index']; assert j not in rows; rows[j] = row
        assert sorted(rows) == list(range(4096))
        rr = {a: [] for a in ('native', *ARMS)}; pure = {a: [] for a in ARMS}
        raw = np.load(B.OUT/d/'features/validation/experts.npy', mmap_mode='r')[512:]
        with (p/'predictions.jsonl').open('x') as stream:
            for j, s in enumerate(samples):
                row = rows[j]
                assert all(row[k] == s[k] for k in ('user_id', 'target', 'history', 'evaluation_block'))
                for a in rr: rr[a].append(B.rank(row[a], s['target']))
                for a in ARMS:
                    measured = full_rank(fused(raw[j], a), s['target'])
                    assert measured == row['standalone_ranks'][a]
                    pure[a].append(measured)
                stream.write(json.dumps(row)+'\n')
        rr = {k: np.asarray(v, np.int64) for k, v in rr.items()}
        pure = {k: np.asarray(v, np.int64) for k, v in pure.items()}
        groups = {'all4096': np.arange(4096), 'previous1024': np.arange(1024),
                  'remaining3072': np.arange(1024, 4096)}
        group_ids = np.asarray([s['evaluation_block'] for s in samples])
        groups.update({'block'+str(b+1): np.flatnonzero(group_ids == b) for b in range(5)})
        np.savez_compressed(p/'ranks.npz', users=np.asarray([s['user_id'] for s in samples]),
                            evaluation_block=group_ids, **rr)
        np.savez_compressed(p/'standalone_ranks.npz', **pure)
        result[d] = {}
        for group, ix in groups.items():
            result[d][group] = dict(users=len(ix), metrics={k: metrics(v[ix]) for k, v in rr.items()},
                standalone_metrics={k: metrics(v[ix]) for k, v in pure.items()},
                comparisons={a+'_vs_'+b: contrast(rr[a][ix], rr[b][ix], repeats=2000, seed=20261017)
                             for a, b in comparisons})
        lines += [f'## {d} / all4096', '', '| 方法 | NDCG@10 | 对GRAM相对变化 | HR@10 |',
                  '|---|---:|---:|---:|']
        table = result[d]['all4096']['metrics']
        for arm, m in table.items():
            change = 100*(m['N10']/table['native']['N10']-1)
            lines.append(f'| {arm} | {m["N10"]:.6f} | {change:+.2f}% | {m["H10"]:.6f} |')
        lines += ['', '五个评估块、前1024/其余3072、纯专家排序与全部配对区间见summary.json。', '']
        write(p/'summary.json', result[d])
        files = [p/n for n in ('predictions.jsonl', 'ranks.npz', 'standalone_ranks.npz', 'summary.json')]
        for job, spec in JOBS.items():
            if spec['domain'] == d:
                files += [OUT/'jobs'/job/n for n in ('predictions.jsonl', 'smoke.json', 'runtime_checks.json')]
        write(p/'output_manifest.json', dict(sha256={str(f.relative_to(ROOT)): sha(f) for f in files}))
    n_pairs = len(comparisons)*8*2
    write(OUT/'summary.json', dict(domains=result, comparisons_per_group=len(comparisons),
        comparison_groups_per_domain=8, total_descriptive_comparisons=n_pairs, test_read=False,
        fixed_alpha=.3, fixed_pair_beta=[.5, .5], automatic_final_selection=False,
        statistical_review=dict(coverage='11/11',
            simpson='Domains and five blocks separated; pooled users also reported, no cross-domain pooled winner.',
            ecological='Mean gains do not imply all users benefit.',
            berkson='Eligibility excludes fitting holdouts; not raw interaction population.',
            collider='No outcome-based cohort or block selection.', base_rate='All prescribed users retained.',
            regression_to_mean='Larger development sample still not independent final confirmation.',
            survivorship='All11 arms required for every user; failed jobs cannot produce complete success.',
            look_elsewhere=f'All {n_pairs} comparisons saved; intervals descriptive, unadjusted.',
            forking_paths='Fixed alpha/beta predeclared; original checkpoints and score scale unchanged.',
            causation='Expert comparisons conditional on frozen equal fusion and generation protocol.',
            reverse_causality='Targets used only for metrics, never weights, scores or encoding.')))
    lines += [f'全部{n_pairs}项配对区间未经多重比较校正；不含训练种子、模型选择或未来融合器优化的不确定性。',
        '这些结果仅确定当前固定权重协议下的候选，不能证明各专家在所有融合权重下的最优潜力。',
        '完成全部4096才标记完成；不自动训练下一轮融合器、调整历史分数或读取测试集。']
    REPORT.write_text('\n'.join(lines)+'\n')


def supervise():
    assert read(OUT/'status.json')['state'] == 'PREPARED'
    pending = list(JOBS); active = {}; finished = {}; attempts = {j: 0 for j in JOBS}
    began = time.monotonic()
    try:
        while pending or active:
            for job, task in list(active.items()):
                code = task['process'].poll()
                if code is None and time.monotonic()-task['started'] > CFG['worker_timeout_seconds']+120:
                    B.stop(task['process']); code = task['process'].returncode
                if code is not None:
                    task['log'].close(); p = OUT/'jobs'/job; status = read(p/'status.json')
                    if code == 75 and status['state'] == 'ADMISSION_DEFERRED': pending.append(job)
                    else:
                        finished[job] = dict(exit_code=code, state=status['state'],
                            lease_released=read(p/'lease_status.json').get('state') == 'released'
                            if (p/'lease_status.json').exists() else True)
                    del active[job]
            inventory = B.S.OLD.gpu_inventory(); occupied = {j['gpu']['uuid'] for j in active.values()}
            available = sorted([g for g in inventory if g['uuid'] not in occupied
                and g['free_mib'] >= CFG['minimum_free_mib']],
                key=lambda g: (g['utilization'], -g['free_mib'], g['index']))
            for job in list(pending):
                if not available or len(active) >= CFG['max_parallel']: break
                gpu = available.pop(0); pending.remove(job); attempts[job] += 1
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['uuid'], OMP_NUM_THREADS='2',
                    MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
                log = (OUT/'jobs'/job/f'worker_attempt{attempts[job]}.log').open('x')
                write(OUT/'jobs'/job/'status.json', dict(state='STARTING', **JOBS[job],
                    users_total=2048, users_completed=0, gpu_index=gpu['index'], updated_at=stamp()))
                process = subprocess.Popen([sys.executable, '-u', '-m', MODULE, 'worker', '--job', job],
                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active[job] = dict(process=process, log=log, gpu=gpu, started=time.monotonic())
            if pending and time.monotonic()-began > CFG['resource_wait_seconds']:
                for job in pending: finished[job] = dict(exit_code=124, state='RESOURCE_WAIT_LIMIT')
                pending = []
            statuses = {j: read(OUT/'jobs'/j/'status.json') for j in JOBS}
            domain_status = {}
            for d in DOMAINS:
                members = [statuses[j] for j in JOBS if JOBS[j]['domain'] == d]
                count = sum(m.get('users_completed', 0) for m in members)
                state = 'COMPLETED' if all(m['state'] == 'COMPLETED' for m in members) else (
                    'INCOMPLETE' if any(m['state'] == 'FAILED' for m in members) else 'RUNNING')
                domain_status[d] = dict(state=state, users_completed=count, users_total=4096,
                    updated_at=stamp(), jobs={j: statuses[j] for j in JOBS if JOBS[j]['domain'] == d})
                write(OUT/d/'status.json', domain_status[d])
            estimates = [statuses[j].get('remaining_generation_seconds_estimate') for j in active]
            estimates = [x for x in estimates if x is not None]
            write(OUT/'status.json', dict(state='RUNNING' if active else 'WAITING_FOR_RESOURCES' if pending else 'FINALIZING',
                pid=os.getpid(), updated_at=stamp(), elapsed_seconds=time.monotonic()-began,
                users_total=8192, users_completed=sum(s['users_completed'] for s in domain_status.values()),
                guided_generations_total=90112, alpha=.3, beta_pair=[.5, .5], history_filter=False,
                history_penalty=False, text_self_exclusion=False, test_read=False,
                remaining_generation_seconds_estimate=max(estimates) if len(estimates) == len(active) and not pending and estimates else None,
                pending=pending, finished=finished, domains=domain_status,
                active={j: dict(pid=t['process'].pid, gpu_index=t['gpu']['index'], gpu_uuid=t['gpu']['uuid'],
                    worker=statuses[j]) for j, t in active.items()}))
            if pending or active: time.sleep(10)
        complete = len(finished) == len(JOBS) and all(f['exit_code'] == 0 and f['state'] == 'COMPLETED'
                                                     and f['lease_released'] for f in finished.values())
        if complete: summarize()
        write(OUT/'status.json', dict(state='COMPLETED' if complete else 'INCOMPLETE', finished=finished,
            updated_at=stamp(), elapsed_seconds=time.monotonic()-began, test_read=False,
            users_total=8192, users_completed=8192 if complete else sum(s['users_completed'] for s in domain_status.values()),
            report=str(REPORT.relative_to(ROOT)), history_filter=False, history_penalty=False,
            alpha=.3, beta_pair=[.5, .5], next_experiment_started=False))
    finally:
        for task in active.values(): B.stop(task['process']); task['log'].close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('test', 'prepare', 'supervise', 'worker', 'generate'))
    parser.add_argument('--job', choices=JOBS); parser.add_argument('--lease-pid', type=int)
    args = parser.parse_args()
    def interrupted(signum, frame): raise InterruptedError('Fixed expert screening interrupted or hard timeout')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGALRM, interrupted)
    try:
        if args.mode == 'test': print(json.dumps(tests()), flush=True)
        elif args.mode == 'prepare': prepare()
        elif args.mode == 'supervise': supervise()
        elif args.mode == 'worker': return worker(args.job)
        else: Generation(args.job, args.lease_pid).run()
        return 0
    except BaseException as exc:
        if OUT.exists():
            p = OUT/'jobs'/args.job if args.job else OUT
            previous = read(p/'status.json') if (p/'status.json').exists() else {}
            value = dict(state='FAILED', error=repr(exc), traceback=traceback.format_exc(),
                         updated_at=stamp(), previous=previous, test_read=False)
            write(p/'failure.json', value); write(p/'status.json', value)
        raise


if __name__ == '__main__': sys.exit(main())
