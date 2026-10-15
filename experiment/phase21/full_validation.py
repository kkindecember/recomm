"""Frozen, resumable full-validation comparison of GRAM and three-expert guidance."""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

import numpy as np

from experiment.phase20 import lime_clean_benefit as clean
from experiment.phase20 import lime_decode_screen as base
from experiment.phase20.catalog_scope_core import metrics, contrast
from experiment.phase21.full_validation_core import build_samples
from experiment.phase21.night_resume_core import prediction_prefix, process_identity, identity_alive
from experiment.phase21.text_contribution import verified_rank

ROOT = base.ROOT
SOURCE = ROOT / 'artifacts/phase20/lime_clean_benefit_20260922_v1'
REFERENCE = ROOT / 'artifacts/phase21/text_contribution_20260923_v1/real'
OUT = ROOT / 'artifacts/phase21/full_validation_20260924_v1'
PLAN = ROOT / 'plan/第二十一阶段/21.5_两域全量验证_20260924.md'
REPORT = ROOT / 'report/第二十一阶段/两域全量验证_20260924.md'
MODULE = 'experiment.phase21.full_validation'
DOMAINS = ('Toys', 'Beauty')
COUNTS = dict(Toys=19412, Beauty=22363)
read, write, sha, stamp = base.read, base.write, base.sha, base.stamp


def status(folder, state, **kw):
    write(folder / 'status.json', dict(state=state, updated_at=stamp(), pid=os.getpid(),
                                     test_read=False, **kw))


def prepare():
    OUT.mkdir(exist_ok=False)
    status(OUT, 'PREPARING')
    cfg = dict(domains=list(DOMAINS), validation_users=COUNTS, alpha=.3, beam_size=50,
               length_penalty=1., max_parallel=2, upstream_seed=2023, gate_seed=0,
               upstream_optimizer_steps=0, fusion_optimizer_steps=0,
               allocation_limit_gib=11, expected_workload_peak_mib=12288,
               total_lease_mib=30720, minimum_free_mib=32768,
               worker_timeout_seconds=48 * 3600, supervisor_timeout_seconds=72 * 3600,
               save_batch_users=64, bootstrap_repeats=2000, bootstrap_seed=20260924,
               gpus=dict(Toys='GPU-4e97077a-5bab-99a7-2fdc-598df6be74cc',
                         Beauty='GPU-cfd866c2-ccc1-8a3c-00b8-594d7a8cd059'),
               test_read=False, validation_labels_used_for_fit=False,
               auto_start_after_reboot=False)
    assert shutil.disk_usage(OUT).free > 20 * 2**30, 'Need 20GiB available for frozen score caches'
    write(OUT / 'config.json', cfg)
    shutil.copy2(PLAN, OUT / 'protocol.md')
    frozen = {}

    def use(path, expected=None):
        path = Path(path)
        assert path.name not in ('test.jsonl', 'test.json', 'user_sequence.txt')
        digest = sha(path)
        if expected is not None:
            assert digest == expected, str(path)
        frozen[str(path.relative_to(ROOT))] = digest
        return path

    old_manifest = read(SOURCE / 'manifest.json')['sha256']
    counts = {}
    for domain in DOMAINS:
        p = OUT / domain
        p.mkdir()
        source = SOURCE / domain
        ref = REFERENCE / domain
        dcfg = read(use(source / 'data_config.json'))
        use(ROOT / dcfg['parent_checkpoint'], dcfg['parent_sha256'])
        for key in ('sasrec', 'itemcf', 'embedding'):
            use(ROOT / dcfg[key], old_manifest[dcfg[key]])
        for name in ('catalog.json', 'samples.json', 'experts.npy'):
            src = source / name
            use(src, old_manifest.get(str(src.relative_to(ROOT))))
        assert read(use(ref / 'summary.json'))['verified_by_sort_and_count']
        for name in ('gate.pt', 'fused.npy', 'samples.json', 'predictions.jsonl', 'ranks.npz'):
            use(ref / name)
        original_cfg = read(use(ROOT / f'experiment/phase18/config/s18_diff_gram_{domain.lower()}.json'))
        validation_path = use(ROOT / original_cfg['prepared'] / 'validation.jsonl')
        design = ROOT / 'artifacts/phase20/lime_clean_training_design_20260920_v1'
        receipt = read(use(design / 'summary.json'))['output_sha256']
        paths = [design / domain / name for name in ('upstream_sequences_without_terminal.jsonl', 'terminal_holdout.jsonl')]
        for path in paths:
            use(path, receipt[str(path.relative_to(ROOT))])
        upstream = {r['user_id']: r['history'] for r in clean.rows(paths[0])}
        held = clean.rows(paths[1])
        samples = build_samples(clean.rows(validation_path), upstream, held, read(source / 'samples.json'))
        assert len(samples) == COUNTS[domain]
        assert sum(s['previously_evaluated'] for s in samples) == 512
        write(p / 'samples.json', samples)
        write(p / 'data_config.json', dict(dcfg, prepared=str(p.relative_to(ROOT)),
              frozen_gate=str((ref / 'gate.pt').relative_to(ROOT))))
        (p / 'catalog.json').symlink_to(source / 'catalog.json')
        for name in ('samples.json', 'data_config.json'):
            use(p / name)
        counts[domain] = dict(validation_users=len(samples), previously_evaluated=512,
            newly_evaluated=len(samples) - 512,
            observed_training_terminal_restored=sum(s['training_holdout_role'] != 'none' for s in samples),
            targets_in_recent_history=sum(s['target'] in s['history'] for s in samples),
            exact_original_validation_history=True, no_validation_target_injected=True)
        write(p / 'input_audit.json', counts[domain])
        status(p, 'QUEUED', users_completed=0, users_total=len(samples))
    paths = [PLAN, OUT / 'config.json', ROOT / 'experiment/GPU_MEMORY_LEASE_POLICY.md',
             ROOT / 'experiment/gpu_memory_lease.py', Path(__file__),
             ROOT / 'experiment/phase21/full_validation_core.py',
             ROOT / 'experiment/phase21/test_full_validation.py',
             ROOT / 'experiment/phase21/run_full_validation.sh',
             ROOT / 'experiment/phase21/night_resume_core.py',
             ROOT / 'experiment/phase21/text_contribution.py',
             ROOT / 'experiment/phase21/text_contribution_core.py',
             ROOT / 'experiment/phase18/core/diff_data.py']
    paths += [ROOT / 'experiment/phase20' / name for name in (
        'lime_clean_benefit.py', 'lime_clean_benefit_core.py', 'lime_decode_screen.py',
        'lime_decode_screen_core.py', 'lime_clean_gram_core.py', 'lime_clean_upstream_core.py',
        'lime_comparison_core.py', 'catalog_scope_core.py', 'guided_decoding_core.py',
        'guided_decoding_job.py', 'guided_decoding_train_core.py') if (ROOT / 'experiment/phase20' / name).exists()]
    paths += list((ROOT / 'GRAM/src').rglob('*.py'))
    paths += list((ROOT / 'third_party/LIME-Rec/lime_rec').rglob('*.py'))
    paths += [ROOT / 'third_party/LIME-Rec/scripts/evaluation/run_repeat_aware_gate.py']
    t5 = ROOT / '.cache/huggingface/models--t5-small/snapshots/df1b051c49625cf57a3d0d8d3863ed4d13564fe4'
    for name in ('config.json', 'pytorch_model.bin', 'tokenizer.json', 'tokenizer_config.json', 'spiece.model'):
        paths.append(t5 / name)
    for path in paths:
        use(path)
        if path.suffix in ('.py', '.sh', '.md'):
            dest = OUT / 'sources' / path.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    write(OUT / 'manifest.json', dict(sha256=frozen, created_at=stamp()))
    write(OUT / 'input_audit.json', counts)
    status(OUT, 'PREPARED', domains={d: read(OUT / d / 'status.json') for d in DOMAINS})
    write_report()
    print(json.dumps(dict(state='PREPARED', domains=counts, frozen_files=len(frozen))), flush=True)


def verify_manifest():
    for rel, expected in read(OUT / 'manifest.json')['sha256'].items():
        assert sha(ROOT / rel) == expected, f'Frozen input changed: {rel}'


class FullWorker(clean.Worker):
    def __init__(self, domain, lease):
        self.lease = lease
        self.generated_users = 0
        self.reused_users = 0
        self.saved_users = 0
        clean.OUT = OUT
        super().__init__(domain)
        self.references = {r['user_id']: r for r in clean.rows(REFERENCE / domain / 'predictions.jsonl')}
        assert len(self.references) == 512

    def status(self, state='RUNNING', **kw):
        if self.lease.poll() is not None:
            raise RuntimeError('GPU lease sidecar exited')
        super().status(state, generated_users=self.generated_users, reused_users=self.reused_users,
                       durable_users=self.saved_users, fusion_optimizer_steps=0, **kw)

    def signals(self):
        import torch
        from lime_rec.models import SASRec
        from lime_rec.experts import ItemCFExpert, LLMSemanticExpert
        from lime_rec.evaluation import normalize_score_rows
        from scripts.evaluation.run_repeat_aware_gate import RepeatAwareGate, features
        from experiment.phase20.lime_comparison_core import left_pad, semantic_users
        self.stage = 'full_catalog_expert_scoring'
        self.status()
        cache_receipt = self.folder / 'score_cache_manifest.json'
        if cache_receipt.exists():
            receipt = read(cache_receipt)
            assert receipt['input_manifest_sha256'] == sha(OUT / 'manifest.json')
            for name, digest in receipt['sha256'].items():
                assert sha(self.folder / name) == digest, name
            self.fused = np.load(self.folder / 'fused.npy', mmap_mode='r')
            return
        n = len(self.samples)
        source = self.catalog['items'][1:]
        v = len(source)
        histories = [s['full_history'] for s in self.samples]
        e = base.arrays(ROOT / self.cfg['embedding'])
        assert e['item_ids'].tolist() == source
        semantic = LLMSemanticExpert(source, {}, embeddings=e['embeddings'])
        user_emb = semantic_users(histories, e['embeddings'])
        cf = ItemCFExpert(str(ROOT / self.cfg['itemcf']), source)
        checkpoint = torch.load(ROOT / self.cfg['sasrec'], map_location='cpu')
        assert checkpoint['train_complete'] and checkpoint['epoch'] == 200 and checkpoint['training_only']
        sas = SASRec(v, 64, 50, 2, 2, 0., architecture='legacy').to(self.device)
        sas.load_state_dict(checkpoint['state_dict'], strict=True)
        sas.eval().requires_grad_(False)
        del checkpoint
        saved = torch.load(ROOT / self.cfg['frozen_gate'], map_location='cpu')
        gate = RepeatAwareGate(14, [.6, .15, .25], .1).to(self.device)
        gate.load_state_dict(saved['state_dict'], strict=True)
        gate.eval().requires_grad_(False)
        old_samples = read(SOURCE / self.domain / 'samples.json')
        old_ix = {s['user_id']: i for i, s in enumerate(old_samples) if s['role'] == 'development'}
        old_raw = np.load(SOURCE / self.domain / 'experts.npy', mmap_mode='r')
        ref_samples = read(REFERENCE / self.domain / 'samples.json')
        ref_ix = {s['user_id']: i for i, s in enumerate(ref_samples) if s['role'] == 'development'}
        ref_fused = np.load(REFERENCE / self.domain / 'fused.npy', mmap_mode='r')
        assert set(old_ix) == set(ref_ix) == set(self.references)
        scores = np.lib.format.open_memmap(self.folder / 'experts.partial.npy', mode='w+', dtype='float32', shape=(n, 3, v))
        fused = np.lib.format.open_memmap(self.folder / 'fused.partial.npy', mode='w+', dtype='float32', shape=(n, v))
        weights, penalties = [], []
        raw_delta = fused_delta = 0.
        with torch.no_grad():
            for start in range(0, n, 64):
                stop = min(start + 64, n)
                hs = histories[start:stop]
                raw_s = sas.predict(torch.from_numpy(left_pad(hs)).to(self.device))[:, 1:].cpu().numpy()
                raw_c = np.stack([cf.score(self.samples[start + i]['user_id'], [source[x - 1] for x in h]) for i, h in enumerate(hs)])
                raw_b = user_emb[start:stop] @ semantic.item_embeddings_norm.T
                values = np.stack([normalize_score_rows(a) for a in (raw_s, raw_c, raw_b)], axis=1)
                for k in range(stop - start):
                    user = self.samples[start + k]['user_id']
                    if user in old_ix:
                        old = old_raw[old_ix[user]]
                        np.testing.assert_allclose(values[k], old, atol=5e-6, rtol=1e-5)
                        raw_delta = max(raw_delta, float(np.max(np.abs(values[k] - old))))
                        values[k] = old  # Preserve exact numerical inputs for reused rows.
                scores[start:stop] = values
                hmask = np.zeros((stop - start, v), dtype=bool)
                for k, h in enumerate(hs):
                    hmask[k, np.asarray(h) - 1] = True
                score_tensor = torch.from_numpy(values).to(self.device)
                lengths = torch.tensor(list(map(len, hs)), dtype=torch.float32, device=self.device)
                mask_tensor = torch.from_numpy(hmask).to(self.device)
                w, penalty = gate(features(score_tensor, lengths, mask_tensor))
                value = ((w[:, :, None] * score_tensor).sum(1) - penalty[:, None] * mask_tensor).cpu().numpy()
                for k in range(stop - start):
                    user = self.samples[start + k]['user_id']
                    if user in ref_ix:
                        old = ref_fused[ref_ix[user]]
                        np.testing.assert_allclose(value[k], old, atol=2e-6, rtol=1e-5)
                        fused_delta = max(fused_delta, float(np.max(np.abs(value[k] - old))))
                        value[k] = old
                fused[start:stop] = value
                weights.append(w.cpu().numpy())
                penalties.append(penalty.cpu().numpy())
                if start % 512 == 0:
                    self.status(score_rows=stop, score_total=n)
        for i in (0, n // 2, n - 1):
            expected = semantic.score(self.samples[i]['user_id'], [source[x - 1] for x in histories[i]])
            np.testing.assert_allclose(normalize_score_rows(expected[None])[0], scores[i, 2], atol=5e-6, rtol=1e-5)
        scores.flush()
        fused.flush()
        del scores, fused, sas, cf, gate, saved, score_tensor, mask_tensor, lengths, w, penalty
        gc.collect()
        torch.cuda.empty_cache()
        (self.folder / 'experts.partial.npy').replace(self.folder / 'experts.npy')
        (self.folder / 'fused.partial.npy').replace(self.folder / 'fused.npy')
        np.savez_compressed(self.folder / 'gate_prediction.npz', weights=np.concatenate(weights), penalty=np.concatenate(penalties))
        write(self.folder / 'score_checks.json', dict(state='PASSED', users=n,
              frozen_gate=True, optimizer_steps=0, existing512_raw_max_abs_delta=raw_delta,
              existing512_fused_max_abs_delta=fused_delta, existing512_exact_cache_reused=True,
              author_semantic_scorer_parity=True, validation_labels_used_for_fit=False))
        names = ('experts.npy', 'fused.npy', 'gate_prediction.npz', 'score_checks.json')
        write(cache_receipt, dict(input_manifest_sha256=sha(OUT / 'manifest.json'),
                                 sha256={name: sha(self.folder / name) for name in names}))
        self.fused = np.load(self.folder / 'fused.npy', mmap_mode='r')

    def smoke(self):
        self.stage = 'native_zero_and_prior_parity'
        self.status()
        old = [i for i, s in enumerate(self.samples) if s['previously_evaluated']]
        longest = max(range(len(self.samples)), key=lambda j: len(self.samples[j]['history']))
        indices = list(dict.fromkeys([old[0], old[len(old) // 2], old[-1], longest]))
        checks = []
        for j in indices:
            ctx = self.context(j)
            native_entry = self.decode(ctx, native_entry=True, record=False)
            native = self.decode(ctx)
            q = self.index.conditional_logs(self.fused[j])
            zero = self.decode(ctx, q, 0.)
            guided = self.decode(ctx, q, self.config['alpha'])
            assert native_entry['paths'] == native['paths'] == zero['paths']
            np.testing.assert_array_equal(native_entry['scores'], native['scores'])
            np.testing.assert_array_equal(zero['scores'], native['scores'])
            late = native['guide'].rescore(native['paths'], q, self.config['alpha'])
            user = self.samples[j]['user_id']
            if user in self.references:
                reference = self.references[user]
                for name, ids, values in (('native', native['ids'], native['scores']),
                                          ('late', native['ids'], late),
                                          ('guided', guided['ids'], guided['scores'])):
                    assert ids == reference[name]['ids'], (user, name)
                    np.testing.assert_allclose(values, reference[name]['scores'], atol=2e-6, rtol=1e-5)
            checks.append(dict(user_id=user, native_and_zero_exact=True,
                               prior_prediction_checked=user in self.references))
            del ctx, native_entry, native, zero, guided
        estimate = float(np.median(self.seconds) * 2 * (len(self.samples) - len(self.references)))
        write(self.folder / 'smoke.json', dict(state='PASSED', checks=checks,
              longest_history_checked=True, projected_generation_seconds_advisory=estimate,
              projections_do_not_stop_run=True))
        self.status(projected_generation_seconds_advisory=estimate)

    def run(self):
        self.signals()
        self.smoke()
        path = self.folder / 'predictions.jsonl'
        expected = list(enumerate(self.samples))
        records = prediction_prefix(path, expected, repair=True)
        self.users_completed = len(records)
        self.saved_users = len(records)
        self.reused_users = sum(r.get('source') == 'verified_existing512' for r in records)
        self.generated_users = len(records) - self.reused_users
        del records
        self.stage = 'full_validation_generation'
        self.status()
        batch = []
        last_status = time.monotonic()
        start_index = self.users_completed
        with path.open('a') as handle:
            for j in range(start_index, len(self.samples)):
                sample = self.samples[j]
                user = sample['user_id']
                if user in self.references:
                    record = dict(self.references[user], sample_index=j, source='verified_existing512')
                    self.reused_users += 1
                else:
                    ctx = self.context(j)
                    q = self.index.conditional_logs(self.fused[j])
                    native = self.decode(ctx)
                    late = native['guide'].rescore(native['paths'], q, self.config['alpha'])
                    guided = self.decode(ctx, q, self.config['alpha'])
                    record = dict(user_id=user, target=sample['target'], history=sample['history'],
                        sample_index=j, source='new_generation',
                        native=dict(ids=native['ids'], scores=native['scores'].tolist()),
                        late=dict(ids=native['ids'], scores=late.tolist()),
                        guided=dict(ids=guided['ids'], scores=guided['scores'].tolist()))
                    self.generated_users += 1
                    del ctx, native, guided
                batch.append(json.dumps(record) + '\n')
                self.users_completed += 1
                if len(batch) >= self.config['save_batch_users'] or j == len(self.samples) - 1:
                    handle.writelines(batch)
                    handle.flush()
                    os.fsync(handle.fileno())
                    batch.clear()
                    self.saved_users = self.users_completed
                if self.users_completed % 32 == 0 or time.monotonic() - last_status >= 30:
                    self.status()
                    last_status = time.monotonic()
        assert self.digest() == self.initial_digest
        self.stage = 'verify_all_rankings'
        self.status()
        result = verify_domain(self.domain)
        result.update(model_unchanged=True, model_digest=self.initial_digest,
                      elapsed_seconds_current_attempt=time.monotonic() - self.started,
                      current_attempt_decodes=self.completed, generated_users=self.generated_users,
                      reused_users=self.reused_users, optimizer_steps=0)
        write(self.folder / 'summary.json', result)
        self.stage = 'complete'
        self.status(state='COMPLETED')


def verify_domain(domain):
    p = OUT / domain
    samples = read(p / 'samples.json')
    records = prediction_prefix(p / 'predictions.jsonl', list(enumerate(samples)))
    assert len(records) == len(samples) == COUNTS[domain]
    rr = {name: [] for name in ('GRAM', 'late', 'guided')}
    for record, sample in zip(records, samples):
        for name, field in (('GRAM', 'native'), ('late', 'late'), ('guided', 'guided')):
            value = record[field]
            assert len(value['ids']) == len(value['scores']) == 50
            rr[name].append(verified_rank(value['ids'], value['scores'], sample['target'], sample['history']))
    rr = {k: np.asarray(v) for k, v in rr.items()}
    users = np.asarray([s['user_id'] for s in samples])
    existing = np.asarray([s['previously_evaluated'] for s in samples])
    source = base.arrays(REFERENCE / domain / 'ranks.npz')
    by_user = {str(u): i for i, u in enumerate(users)}
    indices = [by_user[str(u)] for u in source['users']]
    for name in rr:
        np.testing.assert_array_equal(rr[name][indices], source[name])
    np.savez_compressed(p / 'ranks.npz', users=users, previously_evaluated=existing, **rr)
    cfg = read(OUT / 'config.json')

    def describe(mask):
        r = {k: v[mask] for k, v in rr.items()}
        ms = {k: metrics(v) for k, v in r.items()}
        cs = {f'{a}_minus_{b}': contrast(r[a], r[b], repeats=cfg['bootstrap_repeats'], seed=cfg['bootstrap_seed'])
              for a, b in (('guided', 'GRAM'), ('late', 'GRAM'), ('guided', 'late'))}
        return dict(users=int(mask.sum()), metrics=ms, comparisons=cs,
                    relative_guided_vs_GRAM={k: ms['guided'][k] / ms['GRAM'][k] - 1
                        if ms['GRAM'][k] else None for k in ms['GRAM']})

    result = describe(np.ones(len(samples), dtype=bool))
    result.update(state='COMPLETED', domain=domain, updated_at=stamp(), test_read=False,
                  verified_by_sort_and_count=True, existing512_rank_parity=True,
                  subgroups=dict(previous512=describe(existing), remaining=describe(~existing)),
                  single_upstream_seed=True,
                  caveat='Full validation, not independent test; intervals conditional on fixed trained models.')
    return result


def worker(domain):
    p = OUT / domain
    stop = threading.Event()

    def heartbeat():
        while not stop.is_set():
            write(p / 'heartbeat.json', dict(pid=os.getpid(), updated_at=stamp(), liveness_only=True))
            stop.wait(30)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    lease = None
    lease_log = None
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(InterruptedError('Worker terminated')))
    try:
        status(p, 'RUNNING', stage='verify_frozen_inputs', users_total=COUNTS[domain])
        verify_manifest()
        cfg = read(OUT / 'config.json')
        lease_path = p / f'gpu_lease_status_{os.getpid()}.json'
        lease_log = (p / f'gpu_lease_{os.getpid()}.log').open('x')
        lease = subprocess.Popen([sys.executable, 'experiment/gpu_memory_lease.py', '--gpu', '0',
            '--total-lease-mib', str(cfg['total_lease_mib']), '--expected-workload-peak-mib',
            str(cfg['expected_workload_peak_mib']), '--status-path', str(lease_path)],
            cwd=ROOT, stdout=lease_log, stderr=subprocess.STDOUT)
        write(p / 'lease_launch.json', dict(identity=process_identity(lease.pid), status_path=str(lease_path),
              gpu=os.environ['CUDA_VISIBLE_DEVICES'], expected_workload_peak_mib=cfg['expected_workload_peak_mib']))
        deadline = time.monotonic() + 180
        while True:
            assert lease.poll() is None, 'GPU lease failed; see lease log'
            try:
                if lease_path.exists() and read(lease_path)['state'] == 'holding':
                    break
            except json.JSONDecodeError:
                pass
            assert time.monotonic() < deadline, 'GPU lease admission timed out'
            time.sleep(1)
        FullWorker(domain, lease).run()
    except BaseException as error:
        (p / f'failure_{os.getpid()}.txt').write_text(traceback.format_exc())
        state = 'RESOURCE_TIME_LIMIT' if isinstance(error, TimeoutError) else 'INTERRUPTED' if isinstance(error, InterruptedError) else 'FAILED'
        previous = read(p / 'status.json')
        status(p, state, error=repr(error), previous=previous)
        raise
    finally:
        if lease is not None and lease.poll() is None:
            lease.terminate()
            try:
                lease.wait(timeout=10)
            except subprocess.TimeoutExpired:
                lease.kill()
                lease.wait()
        if lease_log is not None:
            lease_log.close()
        stop.set()
        thread.join(timeout=2)


def complete(domain):
    p = OUT / domain
    return (p / 'summary.json').exists() and read(p / 'summary.json').get('state') == 'COMPLETED' and read(p / 'status.json')['state'] == 'COMPLETED'


def write_report():
    current = read(OUT / 'status.json')
    lines = ['# 两域全量验证', '', 'Origin Skill：academic-research-suite / experiment-agent。', '',
             f"状态：{current['state']}；更新时间：{stamp()}。", '',
             '冻结当前基础版，比较原始GRAM、相同三专家后重排、三专家生成引导。单上游种子，未读取test，没有新训练或调参。', '',
             '| 域 | 总用户 | 状态 | 阶段 | 已处理 | 已批量保存 |', '|---|---:|---|---|---:|---:|']
    for domain in DOMAINS:
        s = read(OUT / domain / 'status.json')
        lines.append(f"| {domain} | {COUNTS[domain]} | {s['state']} | {s.get('stage', '等待')} | {s.get('users_completed', 0)} | {s.get('durable_users', 0)} |")
    for domain in DOMAINS:
        path = OUT / domain / 'summary.json'
        if not path.exists():
            continue
        s = read(path)
        lines += ['', f"{domain}：{s['users']}人完整核验结果。", '',
                  '| 方法 | NDCG@10 | H10 | H50 |', '|---|---:|---:|---:|']
        for key, title in (('GRAM', '原始GRAM'), ('late', '三专家后重排'), ('guided', '三专家生成引导')):
            m = s['metrics'][key]
            lines.append(f"| {title} | {m['N10']:.6f} | {m['H10']:.6f} | {m['H50']:.6f} |")
        rel = s['relative_guided_vs_GRAM']['N10']
        c = s['comparisons']['guided_minus_GRAM']
        lines += ['', f"引导相对GRAM：N10相对变化{rel:+.2%}，绝对差{c['delta_N10']:+.6f}；前十新增/丢失{c['top10_gained']}/{c['top10_lost']}。",
                  f"固定预测条件下的描述性95%区间：{c['descriptive_95pct_N10']}。"]
    lines += ['', '原512人核对输入、评分与排名后复用；其余用户真实生成。全体用户均进入主指标分母，原512人与其余用户另存分组结果。',
              '曾用于融合拟合的用户只使用其后续validation事件评价；已发生的训练末次事件恢复到推理历史，不把该部分称为用户外泛化。',
              '完整validation仍是开发集；区间不含上游训练随机性及研究选择。不得将本批写成最终test或多种子结论。', '',
              '[实时状态](../../artifacts/phase21/full_validation_20260924_v1/status.json)；'
              '[冻结协议](../../plan/第二十一阶段/21.5_两域全量验证_20260924.md)。', '',
              '中断后：`bash experiment/phase21/run_full_validation.sh resume`；完成任务跳过，不自动重试FAILED。']
    temp = REPORT.with_name(REPORT.name + f'.{os.getpid()}.tmp')
    temp.write_text('\n'.join(lines) + '\n')
    temp.replace(REPORT)


def supervise():
    lock = (OUT / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status(OUT, 'VERIFYING_INPUTS')
    verify_manifest()
    cfg = read(OUT / 'config.json')
    active, children, blocked = {}, {}, {}
    started = time.monotonic()
    for path in OUT.glob('launch_*.json'):
        rec = read(path)
        if identity_alive(rec['identity']):
            domain = rec['domain']
            assert domain not in active, 'Duplicate full validation worker'
            active[domain] = rec
    for domain in DOMAINS:
        s = read(OUT / domain / 'status.json')
        if domain not in active and not complete(domain) and s['state'] in ('FAILED', 'RESOURCE_TIME_LIMIT'):
            blocked[domain] = s
    while True:
        now = time.time()
        for domain, rec in list(active.items()):
            if domain in children:
                children[domain].poll()
            if identity_alive(rec['identity']) and now - rec['started_wall'] > cfg['worker_timeout_seconds']:
                os.killpg(rec['identity']['pid'], signal.SIGKILL if rec.get('terminating') else signal.SIGTERM)
                rec['terminating'] = True
                blocked[domain] = dict(state='RESOURCE_TIME_LIMIT', reason='Actual 48h worker limit')
            if not identity_alive(rec['identity']):
                del active[domain]
                if not complete(domain):
                    blocked.setdefault(domain, read(OUT / domain / 'status.json'))
        if time.monotonic() - started > cfg['supervisor_timeout_seconds']:
            for rec in active.values():
                if identity_alive(rec['identity']):
                    os.killpg(rec['identity']['pid'], signal.SIGTERM)
            status(OUT, 'RESOURCE_TIME_LIMIT', reason='72h supervisor limit', active=active, blocked=blocked)
            write_report()
            return
        pending = [d for d in DOMAINS if d not in active and d not in blocked and not complete(d)]
        query_error = None
        if pending:
            try:
                output = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True, timeout=30)
                free = {row.split(',')[0].strip(): int(row.split(',')[1]) for row in output.splitlines()}
            except (subprocess.SubprocessError, ValueError) as error:
                free = {}
                query_error = repr(error)
            for domain in pending:
                gpu = cfg['gpus'][domain]
                if free.get(gpu, 0) < cfg['minimum_free_mib'] or len(active) >= cfg['max_parallel']:
                    continue
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                           PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
                attempt = str(int(time.time()))
                logpath = OUT / domain / f'worker_{attempt}.log'
                with logpath.open('x') as log:
                    child = subprocess.Popen([sys.executable, '-u', '-m', MODULE, 'worker', '--domain', domain],
                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                rec = dict(domain=domain, gpu=gpu, identity=process_identity(child.pid),
                           started_wall=time.time(), log=str(logpath.relative_to(ROOT)))
                assert rec['identity']
                write(OUT / f'launch_{domain}_{child.pid}.json', rec)
                children[domain] = child
                active[domain] = rec
        pending = [d for d in DOMAINS if d not in active and d not in blocked and not complete(d)]
        states = {d: read(OUT / d / 'status.json') for d in DOMAINS}
        if all(complete(d) for d in DOMAINS):
            write(OUT / 'summary.json', dict(state='COMPLETED', updated_at=stamp(),
                  domains={d: read(OUT / d / 'summary.json') for d in DOMAINS}, test_read=False, single_upstream_seed=True))
            state = 'COMPLETED'
        elif not active and not pending:
            state = 'FAILED'
        else:
            state = 'RUNNING' if active else 'WAITING_FOR_RESOURCES'
        status(OUT, state, active=active, pending=pending, blocked=blocked, domains=states,
               resource_query_error=query_error, elapsed_seconds=time.monotonic() - started,
               resume_command='bash experiment/phase21/run_full_validation.sh resume', auto_start_after_reboot=False)
        write_report()
        if state in ('COMPLETED', 'FAILED'):
            return
        time.sleep(15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'supervise', 'resume', 'worker'))
    parser.add_argument('--domain', choices=DOMAINS)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'worker':
        assert args.domain
        worker(args.domain)
    else:
        try:
            supervise()
        except Exception as error:
            status(OUT, 'FAILED', error=repr(error))
            (OUT / f'controller_failure_{os.getpid()}.txt').write_text(traceback.format_exc())
            raise


if __name__ == '__main__':
    main()
