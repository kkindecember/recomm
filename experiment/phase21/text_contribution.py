"""Three parallel semantic-contribution controls on the same clean GRAM assets."""
import argparse
import gc
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

from experiment.phase20 import lime_decode_screen as base
from experiment.phase20 import lime_clean_benefit as clean
from experiment.phase20.catalog_scope_core import metrics, contrast, ranks
from experiment.phase21.text_contribution_core import ARMS, semantic_donors, transform_scores, fuse

ROOT = base.ROOT
SOURCE = ROOT / 'artifacts/phase20/lime_clean_benefit_20260922_v1'
OUT = ROOT / 'artifacts/phase21/text_contribution_20260923_v1'
PLAN = ROOT / 'plan/第二十一阶段/21.3B_文本专家贡献三组并行对照_20260923.md'
REPORT = ROOT / 'report/第二十一阶段/文本专家贡献三组并行对照_20260923.md'
MODULE = 'experiment.phase21.text_contribution'
DOMAINS = ('Toys', 'Beauty')
GPUS = dict(real='GPU-4e97077a-5bab-99a7-2fdc-598df6be74cc',
            without_text='GPU-cfd866c2-ccc1-8a3c-00b8-594d7a8cd059',
            shuffled='GPU-7876ca7a-7bb3-eb10-f22b-2ec3b17b9253')
read, write, sha, stamp = base.read, base.write, base.sha, base.stamp


def prepare():
    OUT.mkdir(exist_ok=False)
    write(OUT / 'status.json', dict(state='PREPARING', updated_at=stamp(), test_read=False))
    assert read(SOURCE / 'independent_verification.json').get('state') in ('VERIFIED', 'PASSED', 'COMPLETED')
    config = dict(arms=list(ARMS), domains=list(DOMAINS), gpus=GPUS, seed=0, epochs=4,
        batch_size=256, learning_rate=.02, weight_decay=.001, score_scale=20., alpha=.3,
        beam_size=50, development_users=512, minimum_free_mib=16384,
        allocation_limit_gib=11, worker_timeout_seconds=5400,
        resource_wait_seconds=600, supervisor_timeout_seconds=10800, test_read=False)
    write(OUT / 'config.json', config); shutil.copy2(PLAN, OUT / 'protocol.md')
    manifest = {}
    def use(path):
        manifest[str(path.relative_to(ROOT))] = sha(path)
    for domain in DOMAINS:
        src = SOURCE / domain
        samples = read(src / 'samples.json')
        chosen = [dict(s, source_sample_index=i) for i, s in enumerate(samples)
                  if s['role'] in ('gate_calibration', 'development')]
        for i, s in enumerate(chosen):
            s['validation_index'] = i
        assert sum(s['role'] == 'gate_calibration' for s in chosen) == dict(Toys=721, Beauty=775)[domain]
        assert sum(s['role'] == 'development' for s in chosen) == 512
        assert len({s['user_id'] for s in chosen}) == len(chosen)
        cfg = read(src / 'data_config.json'); catalog = read(src / 'catalog.json')
        donors, audit = semantic_donors(catalog['items'][1:], np.load(src / 'frequencies.npy')[1:], domain)
        assert audit['changed'] / audit['total'] > .95
        shared = OUT / 'shared' / domain; shared.mkdir(parents=True)
        np.save(shared / 'donors.npy', donors); write(shared / 'donor_audit.json', audit)
        for name in ('samples.json', 'experts.npy', 'lime.npy', 'gate.pt', 'gate_prediction.npz',
                     'catalog.json', 'data_config.json', 'frequencies.npy', 'candidates.jsonl',
                     'ranks.npz', 'cache_summary.json', 'analysis.json'):
            use(src / name)
        use(ROOT / cfg['parent_checkpoint'])
        assert manifest[cfg['parent_checkpoint']] == cfg['parent_sha256']
        use(ROOT / cfg['original_config'])
        for arm in ARMS:
            folder = OUT / arm / domain; folder.mkdir(parents=True)
            write(folder / 'samples.json', chosen); write(folder / 'data_config.json', cfg)
            (folder / 'catalog.json').symlink_to(src / 'catalog.json')
            write(folder / 'status.json', dict(state='PREPARED', arm=arm, domain=domain, updated_at=stamp()))
            for name in ('samples.json', 'data_config.json'): use(folder / name)
        use(shared / 'donors.npy'); use(shared / 'donor_audit.json')
    for arm in ARMS:
        write(OUT / arm / 'config.json', config); use(OUT / arm / 'config.json')
    paths = [PLAN, OUT / 'config.json', Path(__file__),
             ROOT / 'experiment/phase21/text_contribution_core.py',
             ROOT / 'experiment/phase21/test_text_contribution.py',
             ROOT / 'experiment/phase21/run_text_contribution.sh',
             ROOT / 'experiment/phase20/lime_clean_benefit.py',
             ROOT / 'experiment/phase20/lime_decode_screen.py',
             ROOT / 'experiment/phase20/lime_decode_screen_core.py',
             ROOT / 'experiment/phase20/lime_clean_upstream_core.py',
             ROOT / 'experiment/phase20/lime_clean_gram_core.py',
             ROOT / 'experiment/phase20/lime_parallel_core.py',
             ROOT / 'experiment/phase20/guided_decoding_core.py',
             ROOT / 'experiment/phase20/catalog_scope_core.py',
             ROOT / 'experiment/phase18/core/diff_data.py']
    paths += list((ROOT / 'GRAM/src').rglob('*.py'))
    paths += list((ROOT / 'third_party/LIME-Rec/lime_rec').rglob('*.py'))
    paths += [ROOT / 'third_party/LIME-Rec/scripts/evaluation/run_repeat_aware_gate.py']
    for path in paths:
        use(path)
        if path.suffix in ('.py', '.sh', '.md'):
            dest = OUT / 'sources' / path.relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    from experiment.phase20.lime_clean_gram_core import T5
    for name in ('tokenizer.json', 'tokenizer_config.json', 'spiece.model', 'config.json'):
        use(T5 / name)
    write(OUT / 'manifest.json', dict(sha256=manifest, created_at=stamp()))
    write(OUT / 'status.json', dict(state='PREPARED', updated_at=stamp(), test_read=False))
    REPORT.write_text('# 文本专家贡献三组并行对照\n\n状态：PREPARED。三组配置已固定，尚无效果结论。\n')
    print(json.dumps(dict(state='PREPARED', files=len(manifest))), flush=True)


class Worker(clean.Worker):
    def __init__(self, arm, domain):
        self.arm, self.fit_steps = arm, 0
        clean.OUT = OUT / arm  # Process-local adapter; never edit historical code/assets.
        super().__init__(domain)
        self.src = SOURCE / domain
        assert self.initial_digest == read(self.src / 'cache_summary.json')['model_digest']
        self.source_indices = np.asarray([s['source_sample_index'] for s in self.samples])
        self.raw = np.load(self.src / 'experts.npy', mmap_mode='r')
        self.donors = np.load(OUT / 'shared' / domain / 'donors.npy')
        self.development = [i for i, s in enumerate(self.samples) if s['role'] == 'development']
        self.total_queries = len(self.development)
        self.references = {}
        with (self.src / 'candidates.jsonl').open() as f:
            for line in f:
                r = json.loads(line)
                if r['role'] == 'development': self.references[r['user_id']] = r
        assert len(self.references) == self.total_queries == 512

    def status(self, state='RUNNING', **values):
        super().status(state, arm=self.arm, fusion_optimizer_steps=self.fit_steps, **values)

    def fit_signals(self):
        import torch
        from scripts.evaluation.run_repeat_aware_gate import RepeatAwareGate
        self.stage = 'fit_fusion'; self.status()
        n, v = len(self.samples), self.raw.shape[-1]
        history = np.zeros((n, v), dtype=bool)
        lengths = np.asarray([len(s['full_history']) for s in self.samples], dtype=np.float32)
        for i, s in enumerate(self.samples): history[i, np.asarray(s['full_history']) - 1] = True
        fit_indices = np.asarray(sorted([i for i, s in enumerate(self.samples) if s['role'] == 'gate_calibration'],
                                       key=lambda i: self.samples[i]['user_id']))
        assert set(fit_indices).isdisjoint(self.development)
        old_gate = torch.load(self.src / 'gate.pt', map_location='cpu')
        assert [self.samples[i]['user_id'] for i in fit_indices] == old_gate['fit_users']
        torch.manual_seed(0); gate = RepeatAwareGate(14, [.6, .15, .25], .1).to(self.device)
        optimizer = torch.optim.AdamW(gate.parameters(), lr=.02, weight_decay=.001)
        def batch(indices):
            scores = transform_scores(self.raw[self.source_indices[indices]], self.arm, self.donors)
            return (torch.from_numpy(scores).to(self.device), torch.from_numpy(lengths[indices]).to(self.device),
                    torch.from_numpy(history[indices]).to(self.device))
        curve = []
        for epoch in range(4):
            losses = []
            for start in range(0, len(fit_indices), 256):
                ix = fit_indices[start:start + 256]; s, l, h = batch(ix)
                predictions, _, _, _ = fuse(gate, s, l, h, self.arm)
                targets = torch.tensor([self.samples[i]['target'] - 1 for i in ix], device=self.device)
                loss = torch.nn.functional.cross_entropy(20 * predictions, targets)
                assert torch.isfinite(loss)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                self.fit_steps += 1; losses.append(float(loss.detach()))
            curve.append(losses); self.status(epoch=epoch + 1, loss=float(np.mean(losses)))
        assert self.fit_steps == dict(Toys=12, Beauty=16)[self.domain]
        gate.eval()
        gate_state = {k: value.detach().cpu() for k, value in gate.state_dict().items()}
        if self.arm == 'real':
            for k, value in gate_state.items():
                np.testing.assert_allclose(value.numpy(), old_gate['state_dict'][k].numpy(), atol=2e-6, rtol=1e-5)
        torch.save(dict(state_dict=gate_state, fit_users=[self.samples[i]['user_id'] for i in fit_indices],
                        curve=curve, optimizer_steps=self.fit_steps, arm=self.arm, development_labels_used=False), self.folder / 'gate.pt')
        self.fused = np.lib.format.open_memmap(self.folder / 'fused.npy', mode='w+', dtype='float32', shape=(n, v))
        all_features, all_weights, all_penalties = [], [], []
        with torch.no_grad():
            for start in range(0, n, 64):
                ix = np.arange(start, min(start + 64, n)); s, l, h = batch(ix)
                value, f, w, p = fuse(gate, s, l, h, self.arm)
                self.fused[ix] = value.cpu().numpy()
                all_features.append(f.cpu().numpy()); all_weights.append(w.cpu().numpy()); all_penalties.append(p.cpu().numpy())
        self.fused.flush()
        np.savez_compressed(self.folder / 'gate_prediction.npz', features=np.concatenate(all_features),
                            weights=np.concatenate(all_weights), penalty=np.concatenate(all_penalties))
        parity = dict(state='PASSED', fit_users_disjoint_from_development=True)
        if self.arm == 'real':
            old = np.load(self.src / 'lime.npy', mmap_mode='r')
            delta = 0.
            for start in range(0, n, 64):
                ix = np.arange(start, min(start + 64, n))
                expected = old[self.source_indices[ix]]
                np.testing.assert_allclose(self.fused[ix], expected, atol=2e-6, rtol=1e-5)
                delta = max(delta, float(np.max(np.abs(self.fused[ix] - expected))))
            parity['real_fusion_max_abs_difference'] = delta
        write(self.folder / 'fit_checks.json', parity)
        del gate, optimizer, old_gate, history, all_features, all_weights, all_penalties
        gc.collect(); torch.cuda.empty_cache()

    def smoke_control(self):
        self.stage = 'native_and_zero_smoke'; self.status()
        longest = max(self.development, key=lambda j: len(self.samples[j]['history']))
        js = list(dict.fromkeys([longest, *self.development]))[:3]
        checks, times = [], []
        for j in js:
            started = time.monotonic(); s = self.samples[j]
            ctx = self.context(j)
            plain = self.decode(ctx, native_entry=True, record=False)
            native = self.decode(ctx)
            assert plain['paths'] == native['paths']
            np.testing.assert_array_equal(plain['scores'], native['scores'])
            q = self.index.conditional_logs(self.fused[j]); zero = self.decode(ctx, q, 0.)
            assert zero['paths'] == native['paths']
            np.testing.assert_array_equal(zero['scores'], native['scores'])
            guided = self.decode(ctx, q, .3)
            ref = self.references[s['user_id']]
            assert native['paths'] == ref['paths']
            np.testing.assert_array_equal(native['scores'], ref['native'])
            if self.arm == 'real':
                assert guided['paths'] == ref['guided']['paths']
                np.testing.assert_allclose(guided['scores'], ref['guided']['scores'], atol=2e-6, rtol=1e-5)
            checks.append(dict(user=s['user_id'], native_and_zero_exact=True, old_native_exact=True))
            # Four decodes in smoke versus two in formal. Conservative full duration.
            times.append(time.monotonic() - started)
            del ctx, plain, native, zero, guided
        projected = float(np.median(times) * 512)
        write(self.folder / 'smoke.json', dict(state='PASSED', checks=checks,
              profile_seconds=times, conservative_formal_seconds=projected,
              longest_history_checked=True))
        if projected > 4500:
            self.status(state='RESOURCE_BUDGET_STOP', conservative_formal_seconds=projected)
            raise RuntimeError('RESOURCE_BUDGET_STOP: projected formal generation exceeds 75 minutes')

    def run(self):
        self.fit_signals(); self.smoke_control()
        self.stage = 'actual_generation'; self.status()
        with (self.folder / 'predictions.jsonl').open('x') as handle:
            for j in self.development:
                s = self.samples[j]; ctx = self.context(j)
                q = self.index.conditional_logs(self.fused[j]); native = self.decode(ctx)
                ref = self.references[s['user_id']]
                assert native['paths'] == ref['paths']
                np.testing.assert_array_equal(native['scores'], ref['native'])
                late = native['guide'].rescore(native['paths'], q, .3)
                guided = self.decode(ctx, q, .3)
                if self.arm == 'real':
                    assert guided['paths'] == ref['guided']['paths']
                    np.testing.assert_allclose(guided['scores'], ref['guided']['scores'], atol=2e-6, rtol=1e-5)
                    np.testing.assert_allclose(late, ref['mixed'], atol=2e-6, rtol=1e-5)
                record = dict(user_id=s['user_id'], target=s['target'], history=s['history'], sample_index=j,
                    native=dict(ids=native['ids'], scores=native['scores'].tolist()),
                    late=dict(ids=native['ids'], scores=late.tolist()),
                    guided=dict(ids=guided['ids'], scores=guided['scores'].tolist()))
                handle.write(json.dumps(record) + '\n'); handle.flush()
                self.users_completed += 1
                if self.users_completed % 8 == 0 or self.users_completed == 1: self.status()
                del ctx, native, guided
        assert self.digest() == self.initial_digest
        report = verify_domain(self.arm, self.domain)
        report.update(model_unchanged=True, fusion_optimizer_steps=self.fit_steps,
                      total_decodes=self.completed, seconds=time.monotonic() - self.started)
        write(self.folder / 'summary.json', report)
        self.stage = 'complete'; self.status(state='COMPLETED')


def verified_rank(ids, scores, target, history):
    ids, scores = np.asarray(ids), np.asarray(scores)
    masked = np.where(~np.isin(ids, history), scores, -np.inf)
    count_rank = int(ranks(masked[None], ids[None], np.asarray([target]))[0])
    valid = np.isfinite(masked)
    ordered = ids[valid][np.lexsort((ids[valid], -masked[valid]))]
    locations = np.flatnonzero(ordered == target)
    expected = int(locations[0] + 1) if len(locations) else 2**30
    assert count_rank == expected
    return count_rank


def verify_domain(arm, domain):
    folder = OUT / arm / domain; predictions = clean.rows(folder / 'predictions.jsonl')
    samples = read(folder / 'samples.json'); expected = [s for s in samples if s['role'] == 'development']
    assert [r['user_id'] for r in predictions] == [s['user_id'] for s in expected]
    assert len(predictions) == 512
    rr = {name: [] for name in ('GRAM', 'late', 'guided', 'LIME')}
    fused = np.load(folder / 'fused.npy', mmap_mode='r'); all_ids = np.arange(1, fused.shape[1] + 1)
    for r, s in zip(predictions, expected):
        assert r['target'] == s['target'] and r['history'] == s['history']
        for name, field in (('GRAM', 'native'), ('late', 'late'), ('guided', 'guided')):
            value = r[field]
            rr[name].append(verified_rank(value['ids'], value['scores'], r['target'], r['history']))
        rr['LIME'].append(verified_rank(all_ids, fused[r['sample_index']], r['target'], r['history']))
    old = base.arrays(SOURCE / domain / 'ranks.npz')
    np.testing.assert_array_equal(rr['GRAM'], old['dev_GRAM'])
    if arm == 'real':
        np.testing.assert_array_equal(rr['late'], old['dev_fixed_late'])
        np.testing.assert_array_equal(rr['guided'], old['dev_fixed_decode'])
    np.savez_compressed(folder / 'ranks.npz', users=np.asarray([s['user_id'] for s in expected]), **rr)
    return dict(state='COMPLETED', arm=arm, domain=domain, users=512,
                metrics={name: metrics(value) for name, value in rr.items()},
                guided_vs_late=contrast(rr['guided'], rr['late'], repeats=2000, seed=20260923),
                verified_by_sort_and_count=True, test_read=False)


def summarize():
    domains = {}; checks = {}
    for domain in DOMAINS:
        summaries = {arm: read(OUT / arm / domain / 'summary.json') for arm in ARMS}
        rr = {arm: base.arrays(OUT / arm / domain / 'ranks.npz') for arm in ARMS}
        for arm in ARMS: np.testing.assert_array_equal(rr[arm]['users'], rr['real']['users'])
        comparisons = {arm: {scope: contrast(rr['real'][scope], rr[arm][scope], repeats=2000, seed=20260923)
                              for scope in ('guided', 'late', 'LIME')} for arm in ('without_text', 'shuffled')}
        checks[domain] = all(summaries['real']['metrics']['guided']['N10'] > summaries[a]['metrics']['guided']['N10']
            and summaries['real']['metrics']['guided']['H10'] >= summaries[a]['metrics']['guided']['H10']
            for a in ('without_text', 'shuffled'))
        domains[domain] = dict(arms=summaries, real_minus_control=comparisons)
    result = dict(state='COMPLETED', domains=domains, directional_support=checks,
                  dual_domain_support=all(checks.values()), test_read=False,
                  caveat='Exploratory contribution controls, not validation of a new mechanism.')
    write(OUT / 'summary.json', result)
    lines = ['# 文本专家贡献三组并行对照', '', '状态：COMPLETED；三组、两域真实生成及排序/计数复核完成。', '',
        'Origin Skill：academic-research-suite。一个上游seed、一个错配映射；开发池反复使用，本轮不作新机制或确认性显著结论。', '',
        '| 域 | 组 | 真实生成 N10 | H10 | H50 | 同信号后重排 N10 |', '|---|---|---:|---:|---:|---:|']
    for domain in DOMAINS:
        for arm in ARMS:
            m = domains[domain]['arms'][arm]['metrics']; g = m['guided']
            lines.append(f'| {domain} | {arm} | {g["N10"]:.6f} | {g["H10"]:.6f} | {g["H50"]:.6f} | {m["late"]["N10"]:.6f} |')
    lines += ['', '| 域 | real 对照 | 真实生成 ΔN10 | 描述性95%区间 | 前十新增/丢失 |', '|---|---|---:|---|---|']
    for domain in DOMAINS:
        for arm in ('without_text', 'shuffled'):
            c = domains[domain]['real_minus_control'][arm]['guided']; interval = c['descriptive_95pct_N10']
            lines.append(f'| {domain} | {arm} | {c["delta_N10"]:+.6f} | [{interval[0]:+.6f}, {interval[1]:+.6f}] | {c["top10_gained"]}/{c["top10_lost"]} |')
    lines += ['', '预设方向性条件：'+json.dumps(checks, ensure_ascii=False)+'；双域支持：'+str(all(checks.values()))+'。', '',
        'True仅支持当前配方真实文本对应的进一步研究；False表示未满足本轮投入条件，不证明文本普遍无效。未按结果追加拟合、映射或alpha。', '',
        'real完整复现旧组合；去文本同时移除评分和四项统计输入；错配保留分数分布并在训练频次桶内打乱商品对应。各组融合器独立拟合4轮，生成器不训练。', '',
        '源输入与代码哈希、完整融合评分、每用户候选分数、两种排序复核及全部指标保存在产物目录。', '',
        '[冻结协议](../../plan/第二十一阶段/21.3B_文本专家贡献三组并行对照_20260923.md)；[完整结果](../../artifacts/phase21/text_contribution_20260923_v1/summary.json)。']
    REPORT.write_text('\n'.join(lines) + '\n')
    return result


def worker(arm, domain):
    folder = OUT / arm / domain
    try:
        for rel, digest in read(OUT / 'manifest.json')['sha256'].items():
            # Shared source hashes verified by supervisor before launch; check worker code/config/checkpoint here.
            if rel.startswith('experiment/') or rel.startswith('third_party/') or rel.startswith('GRAM/src/'):
                assert sha(ROOT / rel) == digest, rel
        cfg = read(folder / 'data_config.json')
        assert sha(ROOT / cfg['parent_checkpoint']) == cfg['parent_sha256']
        job = Worker(arm, domain); job.run()
    except BaseException as error:
        (folder / 'failure.txt').write_text(traceback.format_exc())
        state = 'RESOURCE_BUDGET_STOP' if 'RESOURCE_BUDGET_STOP' in str(error) else 'FAILED'
        write(folder / 'status.json', dict(state=state, arm=arm, domain=domain,
              error=repr(error), updated_at=stamp(), pid=os.getpid()))
        raise


def free_memory():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
    return {line.split(',')[0].strip(): int(line.split(',')[1]) for line in output.splitlines()}


def supervise():
    cfg = read(OUT / 'config.json'); start = time.monotonic()
    for rel, digest in read(OUT / 'manifest.json')['sha256'].items(): assert sha(ROOT / rel) == digest, rel
    processes, cursor, pending_since, failures, launches = {}, {a: 0 for a in ARMS}, {}, {}, []
    def stop_owned(process):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL); process.wait()
    try:
        while True:
            now = time.monotonic(); free = free_memory()
            if now - start > cfg['supervisor_timeout_seconds']: raise TimeoutError('Supervisor three-hour hard timeout')
            for arm in ARMS:
                if arm in processes:
                    process, domain, launched_at = processes[arm]
                    code = process.poll()
                    if code is None and now - launched_at > cfg['worker_timeout_seconds']:
                        stop_owned(process); code = process.returncode
                        write(OUT / arm / domain / 'status.json', dict(state='TIMEOUT', arm=arm, domain=domain, updated_at=stamp()))
                    if code is not None:
                        del processes[arm]
                        if code == 0: cursor[arm] += 1
                        else: failures[arm] = dict(domain=domain, exit_code=code)
                if arm in processes or arm in failures or cursor[arm] >= len(DOMAINS): continue
                domain = DOMAINS[cursor[arm]]
                if free[GPUS[arm]] < cfg['minimum_free_mib']:
                    pending_since.setdefault(arm, now)
                    if now - pending_since[arm] > cfg['resource_wait_seconds']:
                        failures[arm] = dict(domain=domain, error='insufficient_free_GPU_memory')
                        write(OUT / arm / domain / 'status.json', dict(state='RESOURCE_BUDGET_STOP', reason='insufficient_free_GPU_memory', updated_at=stamp()))
                    continue
                pending_since.pop(arm, None)
                env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=GPUS[arm], OMP_NUM_THREADS='2',
                    MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
                folder = OUT / arm / domain
                with (folder / 'worker.log').open('x') as handle:
                    process = subprocess.Popen([sys.executable, '-u', '-m', MODULE, 'worker', '--arm', arm, '--domain', domain],
                        cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                processes[arm] = (process, domain, now)
                launches.append(dict(arm=arm, domain=domain, pid=process.pid, gpu=GPUS[arm], launched_at=stamp()))
                write(OUT / 'launch.json', dict(supervisor_pid=os.getpid(), launches=launches))
            states = {arm: {domain: read(OUT / arm / domain / 'status.json') for domain in DOMAINS} for arm in ARMS}
            write(OUT / 'status.json', dict(state='RUNNING', pid=os.getpid(), updated_at=stamp(), arms=states,
                  active={a:dict(pid=p.pid, domain=d) for a, (p,d,_) in processes.items()},
                  failures=failures, elapsed_seconds=now-start, test_read=False))
            if not processes and all(a in failures or cursor[a] == len(DOMAINS) for a in ARMS): break
            time.sleep(15)
        if failures:
            write(OUT / 'status.json', dict(state='FAILED', failures=failures, arms=states, updated_at=stamp(), test_read=False))
            REPORT.write_text('# 文本专家贡献三组并行对照\n\n状态：部分任务失败或资源停止，不能做完整三组结论。\n\n'+json.dumps(failures, ensure_ascii=False, indent=2)+'\n')
        else:
            result = summarize()
            write(OUT / 'status.json', dict(state='COMPLETED', updated_at=stamp(), elapsed_seconds=time.monotonic()-start,
                  directional_support=result['directional_support'], dual_domain_support=result['dual_domain_support'], test_read=False))
    except BaseException as error:
        for process, _, _ in processes.values(): stop_owned(process)
        (OUT / 'failure.txt').write_text(traceback.format_exc())
        write(OUT / 'status.json', dict(state='FAILED', error=repr(error), updated_at=stamp(), test_read=False))
        raise


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['prepare', 'supervise', 'worker'])
    parser.add_argument('--arm', choices=ARMS); parser.add_argument('--domain', choices=DOMAINS)
    args = parser.parse_args()
    if args.action == 'prepare': prepare()
    elif args.action == 'supervise': supervise()
    else:
        assert args.arm and args.domain
        worker(args.arm, args.domain)


if __name__ == '__main__': main()
