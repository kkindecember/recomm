"""Minimal full-validation experiment; persistent background controller.

Frozen LIME-derived three-expert beta and AGRec-inspired alpha=.3. Fit only one
global repeat mass per domain on training-terminal events, never validation.
"""
import argparse
import fcntl
import hashlib
import json
import math
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
from experiment.phase20 import lime_clean_benefit as CLEAN
from experiment.phase20 import lime_decode_screen as BASE
from experiment.phase20.catalog_scope_core import metrics
from experiment.phase22 import repeat_split_core as C

ROOT = BASE.ROOT
OUT = ROOT/'artifacts/phase22/repeat_split_minimal_20261017_v1'
FULL = ROOT/'artifacts/phase21/full_validation_20260924_v1'
TRAIN = ROOT/'artifacts/phase20/lime_clean_benefit_20260922_v1'
TEACHER = ROOT/'artifacts/phase21/joint_fusion_20261015_v2'
PLAN = ROOT/'plan/第二十二阶段/22.1_探索重复拆分最小全量验证_20261017.md'
REPORT = ROOT/'report/第二十二阶段/探索重复拆分最小全量验证_20261017.md'
MODULE = 'experiment.phase22.repeat_split'
DOMAINS = ('Toys', 'Beauty')
NEW = ('raw', 'split_zero', 'split_learned', 'native_norepeat')
ARMS = ('native', 'bounded') + NEW
COUNTS = dict(Toys=19412, Beauty=22363)
CFG = dict(alpha=.3, tau=20., beam_size=50, length_penalty=1., domains=list(DOMAINS),
    counts=COUNTS, new_arms=list(NEW), arms=list(ARMS), seed=0,
    fit_steps=200, fit_lr=.1, initial_rho=.1, rho_parameters_per_domain=1,
    trainable_beta=False, trainable_alpha=False, history_filter=False, test_read=False,
    max_parallel=2, total_lease_mib=30720, expected_workload_peak_mib=12288,
    minimum_free_mib=32768, allocation_limit_gib=11,
    worker_timeout_seconds=48*3600, resource_wait_seconds=48*3600,
    bootstrap_repeats=2000, randomization_repeats=10000, statistics_seed=20261017,
    scientific_auto_retry=False, next_experiment=False)
read, sha, stamp = BASE.read, BASE.sha, BASE.stamp


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def status(folder, state, **kw):
    write(folder/'status.json', dict(state=state, updated_at=stamp(), pid=os.getpid(),
                                   test_read=False, history_filter=False, **kw))


def rows(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def recover(path, samples):
    """Only an incomplete final write is repairable; semantic errors fail."""
    complete = 0
    if not path.exists():
        return complete
    with path.open('r+b') as f:
        while True:
            start = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.endswith(b'\n'):
                f.truncate(start)
                f.flush(); os.fsync(f.fileno())
                break
            row = json.loads(line)
            assert complete < len(samples)
            sample = samples[complete]
            assert row['sample_index'] == complete
            assert all(row[k] == sample[k] for k in ('user_id', 'target', 'history'))
            assert set(row['methods']) == set(ARMS)
            for name in ARMS:
                C.ordered(row['methods'][name])
            complete += 1
    return complete


def prepare():
    from experiment.phase22.test_repeat_split import tests
    checks = tests()
    OUT.mkdir(parents=True, exist_ok=False)
    status(OUT, 'PREPARING', stage='freeze_inputs')
    write(OUT/'config.json', CFG)
    write(OUT/'tests.json', checks)
    shutil.copy2(PLAN, OUT/'protocol.md')
    frozen, domain_inputs = {}, {}

    def use(path, expected=None):
        path = Path(path)
        rel = str(path.relative_to(ROOT))
        assert path.name not in ('test.jsonl', 'test.json', 'user_sequence.txt')
        digest = frozen.get(rel) or sha(path)
        if expected:
            assert digest == expected, rel
        frozen[rel] = digest
        return rel

    sources = list((ROOT/'experiment/phase22').glob('*.py')) + list((ROOT/'experiment/phase22').glob('*.sh'))
    for prefix in ('GRAM/src', 'third_party/LIME-Rec/lime_rec'):
        sources += list((ROOT/prefix).rglob('*.py'))
    sources += [ROOT/p for p in (
        'experiment/phase20/lime_clean_benefit.py', 'experiment/phase20/lime_decode_screen.py',
        'experiment/phase20/lime_decode_screen_core.py', 'experiment/phase20/lime_clean_upstream_core.py',
        'experiment/phase20/lime_clean_gram_core.py', 'experiment/phase20/lime_comparison_core.py',
        'experiment/phase20/lime_clean_benefit_core.py', 'experiment/phase20/catalog_scope_core.py',
        'experiment/phase18/core/diff_data.py', 'experiment/gpu_memory_lease.py',
        'experiment/GPU_MEMORY_LEASE_POLICY.md',
        'third_party/LIME-Rec/scripts/evaluation/run_repeat_aware_gate.py')]
    # Include every repo module imported during the checks, plus intended model dependencies.
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', '') or ''
        p = Path(name).resolve()
        if p.suffix == '.py' and ROOT in p.parents:
            sources.append(p)
    previous = read(FULL/'manifest.json')['sha256']
    common = []
    for rel, digest in previous.items():
        if rel.startswith(('GRAM/src/', '.cache/huggingface/')):
            common.append(use(ROOT/rel, digest))
    common += [use(p) for p in set(sources+[PLAN, OUT/'config.json'])]
    design = {}
    for domain in DOMAINS:
        p = OUT/domain; p.mkdir()
        used = list(common)
        cache = read(FULL/domain/'score_cache_manifest.json')
        assert cache['input_manifest_sha256'] == sha(FULL/'manifest.json')
        for name, digest in cache['sha256'].items():
            used.append(use(FULL/domain/name, digest))
        for name in ('samples.json', 'catalog.json', 'data_config.json', 'predictions.jsonl', 'smoke.json', 'cache_summary.json'):
            source = FULL/domain/name
            if source.exists():
                used.append(use(source, previous.get(str(source.relative_to(ROOT)))))
        samples = read(FULL/domain/'samples.json')
        assert len(samples) == COUNTS[domain]
        assert len({s['user_id'] for s in samples}) == len(samples)
        for name in ('samples.json', 'catalog.json'):
            (p/name).symlink_to(FULL/domain/name)
        dc = read(FULL/domain/'data_config.json')
        write(p/'data_config.json', dict(dc, prepared=str(p.relative_to(ROOT))))
        used.append(use(ROOT/dc['parent_checkpoint'], dc['parent_sha256']))
        used.append(use(ROOT/dc['original_config']))
        used.append(use(ROOT/dc['frozen_gate']))
        for name in ('samples.json', 'experts.npy'):
            used.append(use(TRAIN/domain/name))
        for name in ('fit_samples.json', 'teacher.npz', 'teacher_checks.json'):
            used.append(use(TEACHER/domain/name))
        check = read(TEACHER/domain/'teacher_checks.json')
        assert check['state'] == 'PASSED' and check['gram_digest_unchanged']
        assert check['sha256'] == sha(TEACHER/domain/'teacher.npz')
        fit = read(TEACHER/domain/'fit_samples.json')
        assert len(fit) == dict(Toys=2773, Beauty=3310)[domain]
        bank = np.load(TEACHER/domain/'teacher.npz')
        np.testing.assert_array_equal(bank['users'], [s['user_id'] for s in fit])
        np.testing.assert_array_equal(bank['targets'], [s['target']-1 for s in fit])
        source_samples = read(TRAIN/domain/'samples.json')
        source_fit = [s for s in source_samples if s['role'] == 'ranker_fit']
        assert len(source_fit) == len(fit)
        for a, b in zip(source_fit, fit):
            assert all(a[k] == b[k] for k in ('user_id', 'target', 'full_history', 'history'))
        fit_users = {s['user_id'] for s in source_samples if s['role'] != 'development'}
        assert all(s['training_holdout_role'] == 'none' or s['user_id'] in fit_users for s in samples)
        design[domain] = dict(validation_users=len(samples), fit_events=len(fit),
            validation_targets_in_history=sum(s['target'] in s['full_history'] for s in samples),
            train_targets_in_history=sum(s['target'] in s['full_history'] for s in fit),
            previous512=sum(s['previously_evaluated'] for s in samples),
            fitting_user_overlap=sum(s['user_id'] in fit_users for s in samples),
            validation_labels_used_for_fit=False, gate_beta_frozen=True, alpha=.3)
        write(p/'data_audit.json', design[domain])
        used += [use(p/'data_config.json'), use(p/'data_audit.json')]
        domain_inputs[domain] = sorted(set(used))
        status(p, 'PREPARED', users_completed=0, users_total=len(samples))
    for rel in frozen:
        source = ROOT/rel
        if source.suffix in ('.py', '.sh', '.md'):
            dest = OUT/'sources'/rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
    write(OUT/'manifest.json', dict(sha256=frozen, domain_inputs=domain_inputs, created_at=stamp()))
    write(OUT/'data_audit.json', design)
    write(OUT/'environment.json', dict(python=sys.version, torch=torch.__version__, numpy=np.__version__))
    status(OUT, 'PREPARED', users_completed=0, users_total=sum(COUNTS.values()),
           new_generations_total=4*sum(COUNTS.values()), pending=list(DOMAINS))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text('# 探索／重复拆分最小全量验证\n\n状态：PREPARED。尚无本轮效果结论。\n')
    print(json.dumps(dict(state='PREPARED', inputs=len(frozen), data=design)), flush=True)


def verify(domain=None):
    assert read(OUT/'config.json') == CFG
    manifest = read(OUT/'manifest.json')
    paths = manifest['domain_inputs'][domain] if domain else manifest['sha256']
    for rel in paths:
        assert sha(ROOT/rel) == manifest['sha256'][rel], 'Frozen input changed: '+rel


class Generation(CLEAN.Worker):
    def __init__(self, domain):
        CLEAN.OUT = OUT
        self.row_times = []
        super().__init__(domain)
        from experiment.phase18.core.diff_data import DiffDataset
        self.truth = self.samples
        self.samples = [dict(s, target=1) for s in self.truth]
        self.data = DiffDataset(self.samples, self.catalog)
        self.total_queries = len(self.truth)
        self.references = rows(FULL/domain/'predictions.jsonl')
        assert len(self.references) == len(self.truth)
        for j, (r, s) in enumerate(zip(self.references, self.truth)):
            assert r['sample_index'] == j
            assert all(r[k] == s[k] for k in ('user_id', 'target', 'history'))
        self.raw = np.load(FULL/domain/'experts.npy', mmap_mode='r')
        self.bounded = np.load(FULL/domain/'fused.npy', mmap_mode='r')
        with np.load(FULL/domain/'gate_prediction.npz') as a:
            self.weights, self.penalties = a['weights'], a['penalty']
        assert self.initial_digest == read(TEACHER/domain/'teacher_checks.json')['model_digest']

    def context(self, j, override_index=None):
        return super().context(j, override_index=j)

    def status(self, state='RUNNING', **kw):
        total = COUNTS[self.domain]
        done = getattr(self, 'users_completed', 0)
        times = getattr(self, 'row_times', [])
        rate = float(np.median(times[-64:])) if times else None
        values = dict(users_total=total, users_completed=done, durable_users=done,
            rho=getattr(self, 'rho', None), gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),
            remaining_seconds_estimate=rate*(total-done) if rate is not None else None)
        values.update(kw)
        status(self.folder, state, stage=self.stage, elapsed_seconds=time.monotonic()-self.started,
               completed_decodes=self.completed, **values)

    def decode(self, ctx, q=None, alpha=0., counts=None):
        from transformers import LogitsProcessorList
        from model.gram import GRAM
        from model.gram_t5_outputs import BaseModelOutputWithPastAndCrossAttentions
        assert time.monotonic()-self.started < CFG['worker_timeout_seconds']
        guide = C.SplitGuide(self.index, q, alpha, counts)
        began = time.monotonic()
        with torch.no_grad():
            output = super(GRAM, self.model).generate(input_ids=ctx['ids'].reshape(1,-1),
                attention_mask=ctx['mask'].reshape(1,-1),
                encoder_outputs=BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=ctx['hidden']),
                max_length=self.max_length, prefix_allowed_tokens_fn=self.prefix_fn,
                num_beams=50, num_return_sequences=50, return_dict_in_generate=True,
                output_scores=True, length_penalty=1., logits_processor=LogitsProcessorList([guide]))
        paths = [p[:p.index(1)+1] for p in output.sequences.cpu().tolist()]
        ids = [self.index.path_to_item[tuple(p)] for p in paths]
        scores = output.sequences_scores.detach().cpu().double().numpy()
        assert len(ids) == len(set(ids)) == 50 and np.isfinite(scores).all()
        np.testing.assert_allclose(guide.rescore(paths, q, alpha), scores, atol=2e-6, rtol=1e-5)
        self.seconds.append(time.monotonic()-began); self.completed += 1
        return dict(ids=ids, paths=paths, scores=scores, guide=guide)

    def path_values(self, ctx, path):
        from model.gram_t5_outputs import BaseModelOutputWithPastAndCrossAttentions
        with torch.no_grad():
            output = self.model(attention_mask=ctx['mask'].reshape(1,-1),
                encoder_outputs=BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=ctx['hidden']),
                decoder_input_ids=torch.tensor([path[:-1]], device=self.device), use_cache=False, return_dict=True)
            lp = output.logits[0].log_softmax(-1)
            native, legal = [], []
            for d in range(1, len(path)):
                children = list(self.index.children[self.index.lookup[tuple(path[:d])]])
                native.append(float(lp[d-1, path[d]]))
                legal.append(float(torch.logsumexp(lp[d-1, children], 0)))
        return np.asarray(native), np.asarray(legal)

    def fit_rho(self):
        receipt = self.folder/'fit_manifest.json'
        if receipt.exists():
            for name, digest in read(receipt)['sha256'].items():
                assert sha(self.folder/name) == digest
            self.rho = read(self.folder/'rho.json')['rho']
            return
        sys.path.insert(0, str(ROOT/'third_party/LIME-Rec'))
        from scripts.evaluation.run_repeat_aware_gate import RepeatAwareGate, features
        self.stage = 'fit_global_rho'; self.status()
        raw_samples = read(TRAIN/self.domain/'samples.json')
        ix = [i for i, s in enumerate(raw_samples) if s['role'] == 'ranker_fit']
        fit = [raw_samples[i] for i in ix]
        bank = np.load(TEACHER/self.domain/'teacher.npz')
        np.testing.assert_array_equal(bank['users'], [s['user_id'] for s in fit])
        raw = np.load(TRAIN/self.domain/'experts.npy', mmap_mode='r')
        gate = RepeatAwareGate(14, [.6,.15,.25], .1)
        gate.load_state_dict(torch.load(ROOT/self.cfg['frozen_gate'], map_location='cpu')['state_dict'])
        gate.eval().requires_grad_(False)
        depth = bank['native'].shape[1]
        parent, child = np.empty((len(fit), depth, 2)), np.empty((len(fit), depth, 2))
        valid = np.empty((len(fit), depth), bool)
        R = np.empty(len(fit))
        with torch.no_grad():
            for start in range(0, len(fit), 64):
                block = fit[start:start+64]
                values = torch.from_numpy(np.array(raw[np.asarray(ix[start:start+64])]))
                mask = np.stack([C.consumed_mask(raw.shape[-1], s['full_history']) for s in block])
                lengths = torch.tensor([len(s['full_history']) for s in block], dtype=torch.float32)
                beta, _ = gate(features(values, lengths, torch.from_numpy(mask)))
                fused = (beta[:,:,None]*values).sum(1).numpy()
                for k, s in enumerate(block):
                    j = start+k
                    parent[j], child[j], valid[j], R[j] = C.path_components(self.index, fused[k],
                        s['full_history'], self.catalog['decoder_paths'][s['target']-1], depth)
                self.status(training_components_completed=min(start+64, len(fit)), training_events=len(fit))
        np.savez_compressed(self.folder/'training_components.npz', parent=parent, child=child,
            valid=valid, original_history_mass=R, users=bank['users'], targets=bank['targets'],
            native=bank['native'], legal=bank['legal'])
        tensors = [torch.from_numpy(a).double() for a in (bank['native'], bank['legal'], parent, child)]
        vt = torch.from_numpy(valid)
        theta = torch.tensor(math.log(CFG['initial_rho']/(1-CFG['initial_rho'])), dtype=torch.double, requires_grad=True)
        opt = torch.optim.Adam([theta], lr=CFG['fit_lr'])
        curve = []
        for step in range(CFG['fit_steps']+1):
            loss = C.training_nll(*tensors, vt, theta, CFG['alpha'])
            assert torch.isfinite(loss)
            curve.append(dict(step=step, rho=float(theta.sigmoid()), nll=float(loss.detach())))
            if step == CFG['fit_steps']:
                break
            opt.zero_grad(); loss.backward()
            assert torch.isfinite(theta.grad)
            opt.step()
        self.rho = float(theta.detach().sigmoid())
        saved = dict(rho=self.rho, theta=float(theta.detach()), training_events=len(fit),
            steps=CFG['fit_steps'], lr=CFG['fit_lr'], alpha=.3, trainable_parameters=1,
            validation_labels_used=False, selection='fixed final step', curve=curve)
        write(self.folder/'rho.json', saved)
        torch.save(dict(theta=theta.detach(), fit_users=bank['users'].tolist(),
                        validation_labels_used=False), self.folder/'rho.pt')
        write(receipt, dict(sha256={n:sha(self.folder/n) for n in ('rho.json', 'rho.pt', 'training_components.npz')}))
        self.status(training_steps=CFG['fit_steps'])

    @staticmethod
    def equal(actual, expected):
        assert actual['ids'] == expected['ids']
        np.testing.assert_allclose(actual['scores'], expected['scores'], atol=2e-6, rtol=1e-5)

    def smoke(self):
        from experiment.phase18.core.diff_data import DiffDataset
        self.stage = 'native_bounded_and_teacher_smoke'; self.status()
        checks = []
        for j in (0, len(self.truth)//2, len(self.truth)-1):
            ctx = self.context(j)
            native = self.decode(ctx)
            self.equal(native, self.references[j]['native'])
            qold = self.index.conditional_logs(self.bounded[j])
            bounded = self.decode(ctx, qold, .3)
            self.equal(bounded, self.references[j]['guided'])
            raw = (self.weights[j,:,None]*np.asarray(self.raw[j])).sum(0)
            R = C.components(raw, self.truth[j]['full_history'])[1]
            qraw = self.index.conditional_logs(raw)
            qr = C.conditional_logs(self.index, raw, self.truth[j]['full_history'], R)
            for prefix in qraw:
                assert qr[prefix] is not None
                np.testing.assert_allclose(list(qraw[prefix].values()), list(qr[prefix].values()), atol=1e-12)
            zero = self.decode(ctx, qr, 0.)
            assert zero['ids'] == native['ids']
            np.testing.assert_array_equal(zero['scores'], native['scores'])
            for q in (qraw, qr):
                self.equal(self.decode(ctx, q, .3), self.decode(ctx, qraw, .3))
            path = native['paths'][0]
            nt, mt = self.path_values(ctx, path)
            expected = [native['guide'].trace[tuple(path[:d])] for d in range(1, len(path))]
            np.testing.assert_allclose(nt, [a[1][a[0].index(path[d])] for d,a in enumerate(expected, 1)], atol=2e-5, rtol=1e-5)
            np.testing.assert_allclose(mt, [a[2] for a in expected], atol=2e-5, rtol=1e-5)
            self.samples[j]['target'] = 2
            changed = self.context(j)
            self.samples[j]['target'] = 1
            torch.testing.assert_close(ctx['ids'], changed['ids'], atol=0, rtol=0)
            torch.testing.assert_close(ctx['mask'], changed['mask'], atol=0, rtol=0)
            checks.append(dict(row=j, native_cache_matches=True, bounded_cache_matches=True,
                alpha_zero_exact=True, rho_R_matches_raw=True, target_placeholder_invariant=True,
                teacher_incremental_matches=True))
        # Confirm the reused training teacher against fresh target forcing too.
        original_samples, original_data = self.samples, self.data
        fit = read(TEACHER/self.domain/'fit_samples.json')
        bank = np.load(TEACHER/self.domain/'teacher.npz')
        self.samples = [dict(s, target=1) for s in fit]
        self.data = DiffDataset(self.samples, self.catalog)
        for j in (0, len(fit)//2, len(fit)-1):
            path = self.catalog['decoder_paths'][fit[j]['target']-1]
            nt, mt = self.path_values(self.context(j), path)
            np.testing.assert_allclose(nt, bank['native'][j,:len(nt)], atol=2e-5, rtol=1e-5)
            np.testing.assert_allclose(mt, bank['legal'][j,:len(mt)], atol=2e-5, rtol=1e-5)
        self.samples, self.data = original_samples, original_data
        write(self.folder/'smoke.json', dict(state='PASSED', checks=checks,
            training_teacher_replayed=True, model_digest=self.initial_digest))

    def run(self):
        self.fit_rho()
        self.smoke()
        self.users_completed = recover(self.folder/'predictions.jsonl', self.truth)
        self.stage = 'full_validation_generation'; self.status()
        with (self.folder/'predictions.jsonl').open('a') as stream:
            for j in range(self.users_completed, len(self.truth)):
                began = time.monotonic()
                s = self.truth[j]; ctx = self.context(j)
                raw = (self.weights[j,:,None]*np.asarray(self.raw[j])).sum(0)
                # Independent reconstruction check of the frozen bounded scores.
                bounded = raw.copy()
                bounded[np.unique(s['full_history'])-1] -= self.penalties[j]
                np.testing.assert_allclose(bounded, self.bounded[j], atol=2e-6, rtol=1e-5)
                h = s['full_history']
                qraw = self.index.conditional_logs(raw)
                qzero = C.conditional_logs(self.index, raw, h, 0.)
                qlearn = C.conditional_logs(self.index, raw, h, self.rho)
                methods = dict(native=self.references[j]['native'], bounded=self.references[j]['guided'])
                for arm, q, alpha, counts in (
                    ('raw', qraw, .3, None), ('split_zero', qzero, .3, None),
                    ('split_learned', qlearn, .3, None), ('native_norepeat', None, 0., C.novel_counts(self.index, h))):
                    result = self.decode(ctx, q, alpha, counts)
                    methods[arm] = dict(ids=result['ids'], scores=result['scores'].tolist())
                    if arm == 'native_norepeat':
                        assert set(result['ids']).isdisjoint(h)
                row = dict(sample_index=j, user_id=s['user_id'], target=s['target'], history=s['history'],
                    rho=self.rho, original_history_mass=C.components(raw,h)[1], methods=methods)
                stream.write(json.dumps(row, separators=(',',':'))+'\n'); stream.flush()
                os.fsync(stream.fileno())
                self.users_completed = j+1; self.row_times.append(time.monotonic()-began)
                if j == 0 or (j+1)%8 == 0 or j+1 == len(self.truth):
                    self.status(last_completed_user=s['user_id'])
                del ctx, methods, result
        assert self.digest() == self.initial_digest
        write(self.folder/'runtime_checks.json', dict(state='PASSED', users=len(self.truth),
            new_arms=list(NEW), gram_digest_unchanged=True, model_digest=self.initial_digest,
            history_filter=False, validation_labels_used_for_fit=False,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
        self.stage = 'generation_complete'; self.status('GENERATION_COMPLETED')


def stop(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait(timeout=30)


def worker(domain):
    folder = OUT/domain; lease = child = None; code = 1
    lock = (folder/'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        verify(domain)
        gpu = os.environ['CUDA_VISIBLE_DEVICES']
        free = {g['uuid']:g['free_mib'] for g in inventory()}
        if free.get(gpu,0) < CFG['minimum_free_mib']:
            status(folder, 'ADMISSION_DEFERRED', free_mib=free.get(gpu))
            return 75
        leasepath = folder/'gpu_lease_status.json'
        if leasepath.exists():
            leasepath.rename(folder/f'gpu_lease_previous_{time.time_ns()}.json')
        with (folder/'lease.log').open('a') as log:
            lease = subprocess.Popen([sys.executable, str(ROOT/'experiment/gpu_memory_lease.py'),
                '--gpu','0','--total-lease-mib',str(CFG['total_lease_mib']),
                '--expected-workload-peak-mib',str(CFG['expected_workload_peak_mib']),
                '--status-path',str(leasepath)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            began = time.monotonic()
            while not leasepath.exists() or read(leasepath).get('state') != 'holding':
                assert lease.poll() is None, 'GPU lease failed'
                assert time.monotonic()-began < 180, 'GPU lease startup timeout'
                time.sleep(1)
            child = subprocess.Popen([sys.executable, '-u', '-m', MODULE, 'generate', '--domain', domain], cwd=ROOT)
            began = time.monotonic()
            while child.poll() is None:
                assert lease.poll() is None, 'GPU lease exited during generation'
                if time.monotonic()-began > CFG['worker_timeout_seconds']:
                    raise TimeoutError('48h domain computation limit')
                time.sleep(5)
            assert child.returncode == 0, 'Generation failed; see worker log'
        code = 0
    except BaseException as error:
        (folder/f'failure_{os.getpid()}.txt').write_text(traceback.format_exc())
        previous = read(folder/'status.json')
        status(folder, 'FAILED', error=repr(error), traceback=f'failure_{os.getpid()}.txt',
               users_completed=previous.get('users_completed',0), users_total=COUNTS[domain])
    finally:
        stop(child); stop(lease)
        if code == 0:
            assert read(folder/'gpu_lease_status.json')['state'] == 'released'
            previous = read(folder/'status.json')
            status(folder, 'COMPLETED', users_completed=COUNTS[domain], users_total=COUNTS[domain],
                   rho=previous.get('rho'), stage='complete', lease_released=True)
    return code


def inventory():
    output = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],
        text=True, timeout=20)
    result = []
    for line in output.splitlines():
        index, uuid, free, util = [s.strip() for s in line.split(',')]
        result.append(dict(index=int(index), uuid=uuid, free_mib=int(free), utilization=int(util)))
    return result


def paired(a, b, seed):
    diff = (a<=10)/np.log2(a+1) - (b<=10)/np.log2(b+1)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(CFG['bootstrap_repeats']//20):
        boot.extend(diff[rng.integers(len(diff), size=(20,len(diff)))].mean(1))
    nonzero = diff[diff != 0]
    observed = abs(float(nonzero.sum()))
    count = 0
    for _ in range(CFG['randomization_repeats']//100):
        draws = rng.integers(0,2,size=(100,len(nonzero)),dtype=np.int8)*2-1
        count += int((np.abs(draws@nonzero) >= observed-1e-14).sum())
    base = float(((b<=10)/np.log2(b+1)).mean())
    return dict(delta_N10=float(diff.mean()), relative_N10=float(diff.mean()/base) if base else None,
        descriptive_95_interval=np.quantile(boot,[.025,.975]).tolist(),
        paired_sign_flip_p=(count+1)/(CFG['randomization_repeats']+1),
        gained_H10=int(((a<=10)&(b>10)).sum()), lost_H10=int(((a>10)&(b<=10)).sum()),
        gained_H50=int(((a<=50)&(b>50)).sum()), lost_H50=int(((a>50)&(b<=50)).sum()))


def summarize():
    result, primary = {}, []
    comparisons = [('split_zero','raw'), ('split_learned','raw'), ('split_learned','split_zero')]
    comparisons += [(a,b) for a in ('split_zero','split_learned') for b in ('native','bounded','native_norepeat')]
    for domain in DOMAINS:
        folder = OUT/domain
        assert read(folder/'runtime_checks.json')['state'] == 'PASSED'
        assert read(folder/'gpu_lease_status.json')['state'] == 'released'
        samples = read(folder/'samples.json')
        ranks = {a:[] for a in ARMS}; history10 = {a:[] for a in ARMS}; history50 = {a:[] for a in ARMS}
        number = 0
        with (folder/'predictions.jsonl').open() as f:
            for j, line in enumerate(f):
                row, s = json.loads(line), samples[j]
                assert row['sample_index'] == j and all(row[k] == s[k] for k in ('user_id','target','history'))
                assert set(row['methods']) == set(ARMS)
                seen = set(s['full_history'])
                for a in ARMS:
                    items = C.ordered(row['methods'][a])
                    ranks[a].append(C.rank(row['methods'][a],s['target']))
                    history10[a].append(sum(i in seen for i in items[:10]))
                    history50[a].append(sum(i in seen for i in items[:50]))
                assert history50['native_norepeat'][-1] == 0
                number += 1
        assert number == COUNTS[domain]
        rr = {a:np.asarray(v) for a,v in ranks.items()}
        np.savez_compressed(folder/'ranks.npz', users=[s['user_id'] for s in samples], **rr,
            **{a+'_history10':v for a,v in history10.items()}, **{a+'_history50':v for a,v in history50.items()})
        groups = dict(all=np.arange(number), previous512=np.flatnonzero([s['previously_evaluated'] for s in samples]),
            remaining=np.flatnonzero([not s['previously_evaluated'] for s in samples]),
            no_fitting_user_overlap=np.flatnonzero([s['training_holdout_role']=='none' for s in samples]),
            fitting_user_overlap=np.flatnonzero([s['training_holdout_role']!='none' for s in samples]))
        result[domain] = dict(rho=read(folder/'rho.json')['rho'], groups={})
        for name, ix in groups.items():
            result[domain]['groups'][name] = dict(users=len(ix), metrics={a:metrics(v[ix]) for a,v in rr.items()},
                history_at10={a:float(np.asarray(v)[ix].mean()) for a,v in history10.items()},
                history_at50={a:float(np.asarray(v)[ix].mean()) for a,v in history50.items()})
        result[domain]['comparisons'] = {}
        for a,b in comparisons:
            value = paired(rr[a], rr[b], CFG['statistics_seed'])
            result[domain]['comparisons'][a+'_vs_'+b] = value
            if b == 'raw':
                primary.append(value)
    assert len(primary) == 4
    ordered = sorted(primary,key=lambda p:p['paired_sign_flip_p'])
    running = 0.
    for i, value in enumerate(ordered):
        running = max(running, min(1.,(len(ordered)-i)*value['paired_sign_flip_p']))
        value['holm_adjusted_p_primary_family4'] = running
    lines = ['# 探索／重复拆分最小全量验证', '', '## Material Passport', '',
        '- Origin Skill: academic-research-suite / experiment-agent', '- Origin Mode: run / validate',
        '- Origin Date: 2026-10-17', '- Verification Status: ANALYZED; complete saved predictions independently ranked',
        '- Version Label: repeat_split_minimal_20261017_v1', '',
        '全部41,775名validation用户；无评测后历史过滤；固定三专家β和α=.3；每域只拟合一个全局ρ。', '']
    for d in DOMAINS:
        data = result[d]['groups']['all']
        lines += [f'## {d}（{COUNTS[d]}人，学习ρ={result[d]["rho"]:.6f}）', '',
            '| 方法 | NDCG@10 | HR@10 | HR@50 | 前10历史数 |', '|---|---:|---:|---:|---:|']
        for a in ARMS:
            m = data['metrics'][a]
            lines.append(f'| {a} | {m["N10"]:.6f} | {m["H10"]:.6f} | {m["H50"]:.6f} | {data["history_at10"][a]:.3f} |')
        lines += ['', '主比较：', '']
        for a in ('split_zero','split_learned'):
            v = result[d]['comparisons'][a+'_vs_raw']
            lines.append(f'- {a} 对 raw：ΔN10={v["delta_N10"]:+.6f}，95%描述区间={v["descriptive_95_interval"]}，Holm p={v["holm_adjusted_p_primary_family4"]:.4f}。')
        lines.append('')
        write(OUT/d/'summary.json', result[d])
        names = ('predictions.jsonl','ranks.npz','summary.json','rho.json','rho.pt','training_components.npz','smoke.json','runtime_checks.json')
        write(OUT/d/'output_manifest.json', dict(sha256={n:sha(OUT/d/n) for n in names}))
    lines += ['完整5/10/20/50指标、旧512／其余、拟合用户重叠分组和配对检验见summary.json。', '',
        '本轮是反复使用过的validation上的探索性验证，只有固定上游和门控版本；训练侧拟合事件与validation不同，但用户可能重叠。',
        'bootstrap区间是用户配对描述区间，不包含模型种子和研究选择；Holm只校正预设四项主比较。',
        'native_norepeat是生成策略控制。本轮没有完整3×2矩阵，不据其比较宣称已严格分离专家与不重复策略的净贡献。',
        '统计解释覆盖11类风险：域／分组混合、总体／个体混淆、样本选择、按结果选人、稀疏命中基率、回归均值、只留成功运行、重复比较、分析分叉、因果外推及标签反向泄漏。',
        '全部预设方法均保留；不按结果自动调参，不读test，不自动启动下一轮。']
    write(OUT/'summary.json', dict(state='COMPLETED', domains=result, test_read=False,
        history_filter=False, primary_comparisons=4, independent_test=False, upstream_retrained=False))
    REPORT.write_text('\n'.join(lines)+'\n')


def supervise(resume=False):
    lock = (OUT/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    initial = read(OUT/'status.json')['state']
    assert resume or initial == 'PREPARED', 'Use resume explicitly for interrupted runs'
    verify()
    active, finished, attempts = {}, {}, {d:0 for d in DOMAINS}
    for d in DOMAINS:
        # Fail before touching status or reserving memory if an orphan worker lives.
        with (OUT/d/'worker.lock').open('a') as probe:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        st = read(OUT/d/'status.json')
        if st['state'] == 'COMPLETED':
            finished[d] = dict(exit_code=0, state='COMPLETED')
        elif not resume:
            assert st['state'] == 'PREPARED'
    pending = [d for d in DOMAINS if d not in finished]
    began = time.monotonic()
    try:
        while pending or active:
            for d, task in list(active.items()):
                code = task['process'].poll()
                if code is None:
                    continue
                task['log'].close()
                st = read(OUT/d/'status.json')
                if code == 75 and st['state'] == 'ADMISSION_DEFERRED':
                    pending.append(d)
                else:
                    finished[d] = dict(exit_code=code, state=st['state'])
                del active[d]
            error = None
            try:
                gpus = inventory()
            except (OSError, subprocess.SubprocessError, ValueError) as e:
                gpus = []; error = repr(e)
            used = {v['gpu']['uuid'] for v in active.values()}
            available = sorted([g for g in gpus if g['uuid'] not in used and g['free_mib'] >= CFG['minimum_free_mib']],
                               key=lambda g:(-g['free_mib'],g['utilization'],g['index']))
            for d in list(pending):
                if not available or len(active) >= CFG['max_parallel']:
                    break
                gpu = available.pop(0); pending.remove(d); attempts[d] += 1
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['uuid'])
                logpath = OUT/d/f'worker_{time.time_ns()}.log'
                log = logpath.open('x')
                status(OUT/d, 'STARTING', users_completed=read(OUT/d/'status.json').get('users_completed',0),
                       users_total=COUNTS[d], gpu_uuid=gpu['uuid'], gpu_index=gpu['index'])
                proc = subprocess.Popen([sys.executable,'-u','-m',MODULE,'worker','--domain',d],
                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active[d] = dict(process=proc, log=log, gpu=gpu, started=time.monotonic())
                write(OUT/d/f'launch_{proc.pid}.json', dict(pid=proc.pid, gpu=gpu, log=str(logpath), started_at=stamp()))
            if pending and time.monotonic()-began > CFG['resource_wait_seconds']:
                for d in pending:
                    status(OUT/d, 'RESOURCE_WAIT_LIMIT', users_completed=0, users_total=COUNTS[d])
                    finished[d] = dict(exit_code=124,state='RESOURCE_WAIT_LIMIT')
                pending.clear()
            states = {d:read(OUT/d/'status.json') for d in DOMAINS}
            status(OUT, 'RUNNING' if active else 'WAITING_FOR_RESOURCES' if pending else 'FINALIZING',
                stage='background_experiment', users_completed=sum(s.get('users_completed',0) for s in states.values()),
                users_total=sum(COUNTS.values()), new_generations_total=4*sum(COUNTS.values()),
                active={d:dict(pid=t['process'].pid,gpu=t['gpu']) for d,t in active.items()},
                pending=pending, finished=finished, domains=states, resource_query_error=error,
                elapsed_seconds=time.monotonic()-began, report=str(REPORT.relative_to(ROOT)))
            if active or pending:
                time.sleep(15)
        ok = len(finished) == 2 and all(v['exit_code']==0 and v['state']=='COMPLETED' for v in finished.values())
        if ok:
            status(OUT, 'SUMMARIZING', domains={d:read(OUT/d/'status.json') for d in DOMAINS},
                   users_completed=sum(COUNTS.values()), users_total=sum(COUNTS.values()))
            summarize()
        status(OUT, 'COMPLETED' if ok else 'INCOMPLETE', finished=finished,
            domains={d:read(OUT/d/'status.json') for d in DOMAINS},
            users_completed=sum(read(OUT/d/'status.json').get('users_completed',0) for d in DOMAINS),
            users_total=sum(COUNTS.values()), elapsed_seconds=time.monotonic()-began,
            report=str(REPORT.relative_to(ROOT)), next_experiment_started=False)
    finally:
        for task in active.values():
            try:
                os.killpg(task['process'].pid,signal.SIGTERM)
                task['process'].wait(timeout=60)
            except (ProcessLookupError,subprocess.TimeoutExpired):
                pass
            task['log'].close()


def start(resume=False):
    assert (OUT/'manifest.json').exists(), 'Run prepare first'
    session = 'p22_repeat_split_20261017'
    # An existing session must not produce duplicate controllers or workers.
    found = subprocess.run(['tmux','has-session','-t',session], stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    assert found.returncode != 0, 'Background session already exists'
    import shlex
    command = ' '.join(map(shlex.quote,[str(ROOT/'experiment/phase22/run_repeat_split.sh'), 'resume-controller' if resume else 'supervise']))
    # Shell script is invoked through bash so its executable bit is immaterial.
    command = 'bash '+command+' >> '+shlex.quote(str(OUT/'controller.log'))+' 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',session,'bash','-lc',command], check=True, cwd=ROOT)
    write(OUT/f'background_launch_{time.time_ns()}.json', dict(session=session, launched_at=stamp(), resume=resume))
    print(json.dumps(dict(state='STARTED', tmux_session=session, status=str(OUT/'status.json'))),flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('test','prepare','start','status','resume','supervise','resume-controller','worker','generate','summarize'))
    p.add_argument('--domain',choices=DOMAINS)
    args = p.parse_args()
    def interrupt(signum, frame):
        raise InterruptedError(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM,interrupt)
    if args.action == 'test':
        from experiment.phase22.test_repeat_split import tests
        print(json.dumps(tests()),flush=True)
    elif args.action == 'prepare':
        prepare()
    elif args.action in ('start','resume'):
        start(args.action == 'resume')
    elif args.action == 'status':
        print((OUT/'status.json').read_text())
    elif args.action == 'worker':
        assert args.domain
        sys.exit(worker(args.domain))
    elif args.action == 'generate':
        assert args.domain
        Generation(args.domain).run()
    elif args.action == 'summarize':
        summarize()
    else:
        try:
            supervise(args.action == 'resume-controller')
        except BaseException as error:
            (OUT/f'controller_failure_{os.getpid()}.txt').write_text(traceback.format_exc())
            status(OUT,'FAILED',error=repr(error))
            raise


if __name__ == '__main__':
    main()
