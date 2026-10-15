"""Full-catalog single fusion -> existing GRAM probability guide.

One frozen protocol, seed0, two domains, with persistent status/timeout handling.
No original fusion model is used by the new fitter or feature construction.
"""
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
import torch

from experiment.phase21 import single_fusion_core as C
from experiment.phase21 import candidate_gate_generation as G
from experiment.phase21 import dual_candidate_generation as OLD
from experiment.phase21.candidate_gate_core import fold
from experiment.phase21.candidate_gate_screen import ROOT, SOURCE, FULL, write, sha
from experiment.phase20.catalog_scope_core import metrics, contrast

OUT = ROOT/'artifacts/phase21/single_fusion_20261014_v1'
PREVIOUS = ROOT/'artifacts/phase21/dual_candidate_generation_20261014_v1'
PLAN = ROOT/'plan/第二十一阶段/21.24_单融合器全目录双专家_20261014.md'
REPORT = ROOT/'report/第二十一阶段/单融合器全目录双专家_20261014.md'
MODULE = 'experiment.phase21.single_fusion'
DOMAINS, ARMS = ('Toys', 'Beauty'), ('mean', 'user', 'candidate')
read, stamp = OLD.read, OLD.stamp
ACTIVE_LEASE = None
CFG = dict(seed=0, epochs=20, checkpoints=[0, 4, 8, 12, 20], batch_size=32, lr=.002,
    weight_decay=.001, score_scale=20., alpha=.3, beam_size=50, length_penalty=1.,
    evaluation_users=512, worker_timeout_seconds=21600, resource_wait_seconds=86400,
    minimum_free_mib=32768, total_lease_mib=30720, expected_workload_peak_mib=12288,
    allocation_limit_gib=11, max_parallel=2, gpu_pool='all visible GPUs',
    experts=['SASRec', 'BGE'], arms=list(ARMS), user_features=C.USER_NAMES, item_features=C.ITEM_NAMES,
    hidden_width=32, head_seed_selection='fixed seed0', old_fusion_used_by_new_model=False,
    residual=False, candidate_shortlist=False, history_penalty=False, kl=0.,
    loss='exact full-catalog CE(20*f, target)', fit_role='ranker_fit',
    epoch_selection='internal user hash 80/20 CE; refit all ranker_fit',
    test_read=False, upstream_optimizer_steps=0, scientific_auto_retry=False,
    evaluation='same reused historical512 per domain; exploratory')


def progress(domain, stage, **values):
    if ACTIVE_LEASE is not None:
        assert ACTIVE_LEASE.poll() is None, 'GPU lease exited'
    write(OUT/domain/'status.json', dict(state='RUNNING', stage=stage, pid=os.getpid(),
        gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'), updated_at=stamp(), test_read=False, **values))
    print(json.dumps(dict(domain=domain, stage=stage, **values)), flush=True)


def prepare():
    from experiment.phase21.test_single_fusion import tests as integration_tests
    checks = integration_tests()
    assert read(PREVIOUS/'status.json')['state'] == 'COMPLETED'
    assert read(PREVIOUS/'review/verification.json')['state'] == 'VERIFIED_SAVED_PREDICTIONS'
    OUT.mkdir(exist_ok=False)
    write(OUT/'status.json', dict(state='PREPARING', updated_at=stamp()))
    write(OUT/'config.json', CFG); write(OUT/'tests.json', checks)
    shutil.copy2(PLAN, OUT/'protocol.md')
    manifest = {}

    def use(path, expected=None):
        digest = sha(path)
        if expected is not None:
            assert digest == expected, str(path)
        manifest[str(path.relative_to(ROOT))] = digest

    for d in DOMAINS:
        folder = OUT/d; folder.mkdir()
        for name in ('samples.json', 'data_config.json', 'catalog.json'):
            use(PREVIOUS/d/name)
            shutil.copy2(PREVIOUS/d/name, folder/name); use(folder/name)
        samples = read(folder/'samples.json')
        training = [s for s in read(SOURCE/d/'samples.json') if s['role'] == 'ranker_fit']
        assert {s['user_id'] for s in training}.isdisjoint(s['user_id'] for s in samples)
        assert len(samples) == len({s['user_id'] for s in samples}) == 512
        for origin in (SOURCE, FULL):
            for name in ('samples.json', 'experts.npy'):
                use(origin/d/name)
        use(SOURCE/d/'frequencies.npy')
        config = read(folder/'data_config.json')
        use(ROOT/config['parent_checkpoint'], config['parent_sha256'])
        for relative, digest in read(PREVIOUS/d/'output_manifest.json')['sha256'].items():
            use(ROOT/relative, digest)
        use(PREVIOUS/d/'signals.npz')
        old_rows = [json.loads(s) for s in (PREVIOUS/d/'predictions.jsonl').read_text().splitlines()]
        refs = []
        for sample, row in zip(samples, old_rows):
            assert (sample['user_id'], sample['target'], sample['history']) == (row['user_id'], row['target'], row['history'])
            refs.append(dict(user_id=row['user_id'], native=row['native'], baseline=row['baseline_guided'],
                three=row['three'], residual_candidate=row['candidate_guided']))
        assert len(refs) == 512
        write(folder/'references.json', refs); use(folder/'references.json')
        write(folder/'status.json', dict(state='PREPARED', users_total=512, updated_at=stamp()))

    # Pin the reused decoder, GRAM sources, tokenizer and T5 dependencies.
    prior_manifest = read(PREVIOUS/'manifest.json')['sha256']
    for relative, digest in prior_manifest.items():
        if relative.startswith(('GRAM/src/', '.cache/huggingface/')) or relative.startswith('experiment/'):
            use(ROOT/relative, digest)
    for p in (Path(__file__), ROOT/'experiment/phase21/single_fusion_core.py',
              ROOT/'experiment/phase21/test_single_fusion.py', PLAN,
              ROOT/'experiment/phase21/run_single_fusion.sh', OUT/'config.json'):
        use(p)
    for relative in manifest:
        p = ROOT/relative
        if p.suffix in ('.py', '.sh', '.md'):
            dest = OUT/'sources'/relative; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dest)
    write(OUT/'manifest.json', dict(sha256=manifest, created_at=stamp()))
    write(OUT/'status.json', dict(state='PREPARED', updated_at=stamp(), test_read=False))
    REPORT.write_text('# 单融合器全目录双专家探索\n\n状态：PREPARED。一个融合器覆盖全目录，尚无新效果结论。\n')
    print(json.dumps(dict(state='PREPARED', tests=checks, hashes=len(manifest))), flush=True)


class Data:
    def __init__(self, domain, training):
        self.domain = domain
        origin = (SOURCE if training else FULL)/domain
        raw_samples = read(origin/'samples.json')
        if training:
            ix = [j for j, s in enumerate(raw_samples) if s['role'] == 'ranker_fit']
            self.samples = [raw_samples[j] for j in ix]
        else:
            self.samples = read(OUT/domain/'samples.json')
            ix = [s['full_validation_index'] for s in self.samples]
            for sample, j in zip(self.samples, ix):
                for key in ('user_id', 'target', 'full_history', 'history'):
                    assert sample[key] == raw_samples[j][key]
        self.n = len(ix)
        experts = np.load(origin/'experts.npy', mmap_mode='r')
        self.v = experts.shape[-1]
        self.folder = OUT/domain/('train' if training else 'validation')
        self.folder.mkdir()
        self.user = np.lib.format.open_memmap(self.folder/'user.npy', mode='w+', dtype='float32', shape=(self.n, C.DU))
        self.item = np.lib.format.open_memmap(self.folder/'item.npy', mode='w+', dtype='float32', shape=(self.n, self.v, C.DC))
        self.m1, self.m2 = np.zeros((self.n, C.DC)), np.zeros((self.n, C.DC))
        counts = np.load(SOURCE/domain/'frequencies.npy')
        stage = 'features_train' if training else 'features_validation'
        begin = time.monotonic()
        progress(domain, stage, completed=0, total=self.n)
        for j, src in enumerate(ix):
            s = self.samples[j]
            assert s['history'] == s['full_history'][-20:]
            u, c = C.features(C.retained(experts[src]), s['full_history'], counts)
            assert c[:, :2].min() >= -1e-6 and c[:, :2].max() <= 1.00001
            self.user[j], self.item[j] = u, c
            self.m1[j] = c.mean(0, dtype=np.float64)
            self.m2[j] = np.square(c.astype(np.float64)).mean(0)
            if (j+1) % 256 == 0 or j+1 == self.n:
                progress(domain, stage, completed=j+1, total=self.n, elapsed_seconds=time.monotonic()-begin)
        self.user.flush(); self.item.flush()
        self.targets = np.asarray([s['target']-1 for s in self.samples], dtype=np.int64)
        self.folds = np.asarray([fold(s['user_id'], domain) for s in self.samples])
        write(self.folder/'users.json', [dict(user_id=s['user_id'], source_index=int(j), target=s['target']) for s, j in zip(self.samples, ix)])
        write(self.folder/'metadata.json', dict(users=self.n, catalog_items=self.v, training=training,
            old_fusion_accessed=False, target_used_in_features=False, candidate_shortlist=False))
        paths = list(self.folder.glob('*.npy')) + [self.folder/'users.json', self.folder/'metadata.json']
        write(self.folder/'manifest.json', dict(sha256={str(p.relative_to(ROOT)): sha(p) for p in paths}))

    def norm(self, ix):
        u = self.user[ix].astype(np.float64)
        mean = np.r_[u.mean(0), self.m1[ix].mean(0)]
        square = np.r_[np.square(u).mean(0), self.m2[ix].mean(0)]
        std = np.sqrt(np.maximum(square - mean**2, 0)); std[std < 1e-5] = 1.
        return mean.astype(np.float32), std.astype(np.float32)


def fit(data, arm, device):
    user = torch.from_numpy(np.array(data.user)).to(device)
    item = torch.from_numpy(np.array(data.item)).to(device)
    target = torch.from_numpy(data.targets).to(device)
    fi, hi = np.flatnonzero(data.folds != 0), np.flatnonzero(data.folds == 0)
    assert len(fi) and len(hi)

    def initialize():
        torch.manual_seed(CFG['seed']); torch.cuda.manual_seed_all(CFG['seed'])
        return C.Fusion().to(device)

    def loss(model, indices, norm):
        j = torch.as_tensor(indices, device=device)
        f, _ = C.score(model, user[j], item[j], *norm, arm)
        return torch.nn.functional.cross_entropy(CFG['score_scale']*f, target[j])

    def run(model, indices, norm, epochs, phase):
        opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
        rng = np.random.default_rng(CFG['seed'])
        curve = []

        def heldout(epoch):
            with torch.no_grad():
                value = sum(float(loss(model, hi[start:start+32], norm))*len(hi[start:start+32])
                            for start in range(0, len(hi), 32))/len(hi)
            curve.append(dict(epoch=epoch, loss=value))
            return value

        if phase == 'selection':
            heldout(0)
        begin = time.monotonic()
        for epoch in range(1, epochs+1):
            order = rng.permutation(indices); total = 0.
            for start in range(0, len(order), CFG['batch_size']):
                jj = order[start:start+CFG['batch_size']]
                opt.zero_grad(); value = loss(model, jj, norm)
                assert torch.isfinite(value), 'Nonfinite training loss'
                value.backward(); opt.step(); total += float(value)*len(jj)
            val = heldout(epoch) if phase == 'selection' and epoch in CFG['checkpoints'] else None
            progress(data.domain, phase, arm=arm, epoch=epoch, epochs=epochs,
                loss=total/len(order), internal_heldout_loss=val, elapsed_seconds=time.monotonic()-begin)
            write(OUT/data.domain/(arm+'_'+phase+'_curve.json'), dict(curve=curve, completed_epochs=epoch))
        return curve

    norm = data.norm(fi); nt = tuple(torch.from_numpy(x).to(device) for x in norm)
    model = initialize(); curve = run(model, fi, nt, CFG['epochs'], 'selection')
    selected = min(curve, key=lambda r: (r['loss'], r['epoch']))['epoch']
    del model
    norm = data.norm(np.arange(data.n)); nt = tuple(torch.from_numpy(x).to(device) for x in norm)
    model = initialize(); run(model, np.arange(data.n), nt, selected, 'refit')
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    info = dict(arm=arm, seed=CFG['seed'], selected_epochs=selected, selection_curve=curve,
        selection_fit_users=len(fi), selection_heldout_users=len(hi), refit_users=data.n,
        parameters=sum(p.numel() for p in model.parameters()), old_fusion_used=False)
    torch.save(dict(state_dict=state, mean=norm[0], std=norm[1], fit_users=[s['user_id'] for s in data.samples],
        config=CFG, **info), OUT/data.domain/(arm+'.pt'))
    del item, user, target
    return model, norm, info


def export(data, model, norm, arm, device):
    scores = np.empty((data.n, data.v), dtype=np.float32)
    nt = tuple(torch.from_numpy(x).to(device) for x in norm)
    checks, spreads = [], []
    with torch.no_grad():
        for start in range(0, data.n, CFG['batch_size']):
            stop = min(start+CFG['batch_size'], data.n)
            u = torch.from_numpy(np.array(data.user[start:stop])).to(device)
            c = torch.from_numpy(np.array(data.item[start:stop])).to(device)
            f, w = C.score(model, u, c, *nt, arm)
            assert torch.isfinite(f).all()
            torch.testing.assert_close(w.sum(-1), torch.ones_like(w[..., 0]))
            assert ((f >= c[..., :2].min(-1).values - 2e-6) & (f <= c[..., :2].max(-1).values + 2e-6)).all()
            scores[start:stop] = f.cpu().numpy()
            if arm == 'candidate':
                spreads.extend(w[:, :, 0].std(1).cpu().tolist())
            for j in (0, data.n//2, data.n-1):
                if start <= j < stop:
                    actual, _ = C.numpy_score(model.state_dict(), data.user[j], data.item[j], *norm, arm)
                    np.testing.assert_allclose(actual, scores[j], atol=3e-6, rtol=2e-5)
                    checks.append(j)
    return scores, dict(independent_numpy_full_catalog_users=checks,
        mean_item_weight_std=float(np.mean(spreads)) if spreads else 0., candidate_shortlist=False)


def train_and_export(domain):
    device = torch.device('cuda:0')
    training, validation = Data(domain, True), Data(domain, False)
    assert {s['user_id'] for s in training.samples}.isdisjoint(s['user_id'] for s in validation.samples)
    signals = dict(mean=np.asarray(validation.item[..., :2]).mean(-1))
    fits, checks = {}, {}
    for arm in ('user', 'candidate'):
        progress(domain, 'fit_start', arm=arm)
        model, norm, fits[arm] = fit(training, arm, device)
        signals[arm], checks[arm] = export(validation, model, norm, arm, device)
        # Reload the saved model independently; its exported scores must agree.
        ck = torch.load(OUT/domain/(arm+'.pt'), map_location='cpu')
        assert ck['config'] == CFG and ck['seed'] == 0
        for j in (0, 256, 511):
            f, _ = C.numpy_score(ck['state_dict'], validation.user[j], validation.item[j], ck['mean'], ck['std'], arm)
            np.testing.assert_allclose(f, signals[arm][j], atol=3e-6, rtol=2e-5)
        del model; gc.collect(); torch.cuda.empty_cache()
    pure = {a: np.asarray([C.target_rank(signals[a][j], s['target'], s['history'])
            for j, s in enumerate(validation.samples)]) for a in ARMS}
    with np.load(PREVIOUS/domain/'signals.npz') as old:
        signals['legacy'] = old['baseline']
    np.savez(OUT/domain/'signals.npz', **signals)
    np.savez_compressed(OUT/domain/'pure_ranks.npz', users=np.asarray([s['user_id'] for s in validation.samples]), **pure)
    write(OUT/domain/'training.json', dict(fits=fits, checks=checks, pure_metrics={a: metrics(r) for a, r in pure.items()},
        test_read=False, old_fusion_used_by_new_model=False, full_catalog=True))
    paths = [OUT/domain/name for name in ('user.pt', 'candidate.pt', 'signals.npz', 'pure_ranks.npz', 'training.json')]
    write(OUT/domain/'training_manifest.json', dict(sha256={str(p.relative_to(ROOT)): sha(p) for p in paths}))
    del training, validation, signals
    gc.collect(); torch.cuda.empty_cache()


class Generation(G.Worker):
    def __init__(self, domain, lease):
        G.OUT = OUT
        self.row_seconds = []
        super().__init__(domain, lease)

    def status(self, state='RUNNING', **values):
        assert self.lease.poll() is None, 'GPU lease exited'
        done = getattr(self, 'users_completed', 0)
        rate = float(np.median(self.row_seconds[-32:])) if self.row_seconds else None
        write(self.folder/'status.json', dict(state=state, stage=self.stage, pid=os.getpid(),
            domain=self.domain, gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'), updated_at=stamp(),
            users_completed=done, users_total=512, completed_decodes=self.completed,
            elapsed_seconds=time.monotonic()-self.started,
            remaining_generation_seconds_estimate=rate*(512-done) if rate else None,
            median_seconds_per_user=rate, gram_optimizer_steps=0, test_read=False,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20, **values))

    def run(self):
        self.stage = 'native_zero_legacy_smoke'; self.status()
        longest = max(range(len(self.samples)), key=lambda j: len(self.samples[j]['history']))
        indices = list(dict.fromkeys([longest, 0, 256, 511]))[:3]
        checks = []
        for j in indices:
            ctx = self.context(j)
            native = self.decode(ctx); plain = self.decode(ctx, native_entry=True, record=False)
            q = self.index.conditional_logs(self.signals['legacy'][j])
            zero = self.decode(ctx, q, 0.); old = self.decode(ctx, q, CFG['alpha'])
            for other in (plain, zero):
                assert native['paths'] == other['paths']
                np.testing.assert_array_equal(native['scores'], other['scores'])
            ref = self.reference[self.samples[j]['user_id']]
            for value, key in ((native, 'native'), (old, 'baseline')):
                assert value['ids'] == ref[key]['ids']
                np.testing.assert_allclose(value['scores'], ref[key]['scores'], atol=2e-6, rtol=1e-5)
            checks.append(dict(user_id=self.samples[j]['user_id'], alpha0_exact=True, prior_dual_reproduced=True))
            del ctx, native, plain, zero, old
        write(self.folder/'smoke.json', dict(state='PASSED', checks=checks))
        self.stage = 'matched_generation'; self.status()
        with (self.folder/'predictions.jsonl').open('x') as handle:
            for j, sample in enumerate(self.samples):
                begin = time.monotonic()
                ctx = self.context(j); native = self.decode(ctx)
                ref = self.reference[sample['user_id']]
                assert native['ids'] == ref['native']['ids']
                np.testing.assert_allclose(native['scores'], ref['native']['scores'], atol=2e-6, rtol=1e-5)
                row = dict(user_id=sample['user_id'], target=sample['target'], history=sample['history'],
                    native=dict(ids=native['ids'], scores=native['scores'].tolist()),
                    baseline=ref['baseline'], three=ref['three'], residual_candidate=ref['residual_candidate'])
                for arm in ARMS:
                    q = self.index.conditional_logs(self.signals[arm][j], temperature_scale=CFG['score_scale'])
                    late = native['guide'].rescore(native['paths'], q, CFG['alpha'])
                    guided = self.decode(ctx, q, CFG['alpha'])
                    row[arm+'_late'] = dict(ids=native['ids'], scores=late.tolist())
                    row[arm+'_guided'] = dict(ids=guided['ids'], scores=guided['scores'].tolist())
                    del guided
                handle.write(json.dumps(row)+'\n'); handle.flush(); os.fsync(handle.fileno())
                self.users_completed = j+1; self.row_seconds.append(time.monotonic()-begin)
                if j == 0 or (j+1) % 8 == 0 or j == 511:
                    self.status()
                del ctx, native
        assert self.digest() == self.initial_digest
        write(self.folder/'runtime_checks.json', dict(state='PASSED', native_reproduced_users=512,
            gram_model_digest_unchanged=True, model_digest=self.initial_digest, gram_optimizer_steps=0))
        self.stage = 'complete'; self.status('COMPLETED')


class LeaseHandle:
    """Read-only liveness handle shared with training/generation child processes."""
    def __init__(self, pid):
        self.pid = int(pid)

    def poll(self):
        try:
            os.kill(self.pid, 0)
            return None
        except ProcessLookupError:
            return 1


def worker(domain):
    folder = OUT/domain; lease = None

    def interrupted(signum, frame):
        raise InterruptedError('Single-fusion worker interrupted or timed out')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGALRM, interrupted)
    signal.alarm(CFG['worker_timeout_seconds'])
    try:
        assert read(OUT/'config.json') == CFG
        progress(domain, 'verify_inputs')
        for relative, digest in read(OUT/'manifest.json')['sha256'].items():
            assert sha(ROOT/relative) == digest, relative
        gpu = os.environ['CUDA_VISIBLE_DEVICES']
        free = next(g['free_mib'] for g in OLD.gpu_inventory() if g['uuid'] == gpu)
        if free < CFG['minimum_free_mib']:
            write(folder/'status.json', dict(state='ADMISSION_DEFERRED', gpu_uuid=gpu, free_mib=free,
                required_free_mib=CFG['minimum_free_mib'], experiment_started=False, updated_at=stamp()))
            return 75
        with (folder/'lease.log').open('x') as log:
            lease_path = folder/'lease_status.json'
            lease = subprocess.Popen([sys.executable, str(ROOT/'experiment/gpu_memory_lease.py'), '--gpu', '0',
                '--total-lease-mib', str(CFG['total_lease_mib']),
                '--expected-workload-peak-mib', str(CFG['expected_workload_peak_mib']),
                '--status-path', str(lease_path)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            write(folder/'lease_process.json', dict(pid=lease.pid, gpu_uuid=gpu, started_at=stamp()))
            begin = time.monotonic()
            while not lease_path.exists() or read(lease_path).get('state') != 'holding':
                assert lease.poll() is None, 'GPU lease startup failed'
                assert time.monotonic()-begin < 180, 'GPU lease startup timeout'
                time.sleep(1)
            # Separate CUDA lifetimes release training buffers completely and
            # preserve the existing decoder's one-shot interop-thread setup.
            for phase in ('train', 'generate'):
                assert lease.poll() is None, 'GPU lease exited'
                subprocess.run([sys.executable, '-u', '-m', MODULE, phase, '--domain', domain,
                    '--lease-pid', str(lease.pid)], cwd=ROOT, check=True)
        return 0
    except BaseException as exc:
        previous = read(folder/'status.json')
        write(folder/'status.json', dict(state='FAILED', error=repr(exc), traceback=traceback.format_exc(),
            previous=previous, updated_at=stamp(), test_read=False))
        raise
    finally:
        signal.alarm(0)
        if lease is not None and lease.poll() is None:
            lease.terminate()
            try:
                lease.wait(timeout=15)
            except subprocess.TimeoutExpired:
                lease.kill(); lease.wait()


def summarize():
    results = {}
    keys = ('native', 'baseline', 'three', 'residual_candidate') + tuple(a+s for a in ARMS for s in ('_late', '_guided'))
    lines = ['# 单融合器全目录双专家探索', '',
        '状态：COMPLETED。每域同批512开发用户，固定小头seed0，探索性结果。', '',
        '单一610参数融合器覆盖全目录；没有旧融合器、残差加分、候选集合外回退或历史惩罚项。GRAM及专家冻结。', '',
        '| 域 | GRAM N10 | 原双专家 N10 | 简单平均 N10 | 单用户融合 N10 | 单候选融合 N10 | 旧候选修正 N10 | 三专家 N10 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for d in DOMAINS:
        folder = OUT/d
        assert read(folder/'status.json')['state'] == 'COMPLETED'
        assert read(folder/'lease_status.json')['state'] == 'released'
        assert read(folder/'runtime_checks.json')['gram_model_digest_unchanged']
        rows = [json.loads(s) for s in (folder/'predictions.jsonl').read_text().splitlines()]
        samples = read(folder/'samples.json')
        assert len(rows) == len(samples) == 512
        rr = {key: [] for key in keys}
        for row, sample in zip(rows, samples):
            for key in ('user_id', 'target', 'history'):
                assert row[key] == sample[key]
            for key in keys:
                rr[key].append(OLD.record_rank(row[key], sample))
        rr = {key: np.asarray(value, dtype=np.int64) for key, value in rr.items()}
        prior = np.load(PREVIOUS/d/'ranks.npz')
        for new, old in (('native', 'native'), ('baseline', 'baseline_guided'), ('three', 'three'), ('residual_candidate', 'candidate_guided')):
            np.testing.assert_array_equal(rr[new], prior[old])
        np.savez_compressed(folder/'ranks.npz', users=np.asarray([s['user_id'] for s in samples]), **rr)
        pairs = [('candidate_guided', 'user_guided'), ('candidate_guided', 'residual_candidate')]
        pairs += [(a+'_guided', b) for a in ('user', 'candidate') for b in ('native', 'baseline', 'mean_guided')]
        pairs += [(a+'_guided', a+'_late') for a in ARMS]
        result = dict(users=512, seed=0, metrics={k: metrics(v) for k, v in rr.items()},
            comparisons={a+'_vs_'+b: contrast(rr[a], rr[b], repeats=2000, seed=20261014) for a, b in pairs},
            training=read(folder/'training.json'), test_read=False)
        results[d] = result; write(folder/'summary.json', result)
        write(folder/'verification.json', dict(state='VERIFIED_SAVED_PREDICTIONS', rows=512,
            rank_arrays=len(keys), prior_rank_arrays_matched=4, gram_unchanged=True, test_read=False))
        names = ('native', 'baseline', 'mean_guided', 'user_guided', 'candidate_guided', 'residual_candidate', 'three')
        lines.append('| '+d+' | '+' | '.join(f'{result["metrics"][k]["N10"]:.6f}' for k in names)+' |')
        paths = [folder/name for name in ('predictions.jsonl', 'ranks.npz', 'summary.json', 'verification.json',
                    'runtime_checks.json', 'smoke.json', 'training.json', 'training_manifest.json')]
        write(folder/'output_manifest.json', dict(sha256={str(p.relative_to(ROOT)): sha(p) for p in paths}))
    write(OUT/'summary.json', dict(domains=results, test_read=False, head_seed=0))
    lines += ['', '完整H10/H50、配对差值和描述性区间见summary.json。所有预定组均执行生成，没有依据纯排序效果筛掉组别。', '',
        '本轮是结构简化的整体探索，旧融合移除、全目录应用和特征同时改变，不作单因素归因。一个seed、反复使用的开发池、未经多重比较校正的区间不能代替正式确认。', '',
        '简单平均用于检验训练融合器的实际贡献；两专家在新融合器下是否各自必要仍需后续消融。训练仅ranker_fit，开发用户与拟合用户无重叠，没有读取test。', '',
        '完成后停止，不自动追加模型、参数、种子或论文实验。']
    REPORT.write_text('\n'.join(lines)+'\n')


def supervise():
    assert read(OUT/'status.json')['state'] == 'PREPARED'
    start = time.monotonic(); pending = list(DOMAINS); active = {}; finished = {}; attempts = {d: 0 for d in DOMAINS}
    try:
        while pending or active:
            for d, job in list(active.items()):
                code = job['process'].poll()
                if code is None and time.monotonic()-job['started'] > CFG['worker_timeout_seconds']+120:
                    job['process'].terminate()
                    try:
                        code = job['process'].wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        job['process'].kill(); code = job['process'].wait()
                    write(OUT/d/'status.json', dict(state='TIME_LIMIT', updated_at=stamp()))
                if code is not None:
                    job['log'].close(); state = read(OUT/d/'status.json')
                    if code == 75 and state['state'] == 'ADMISSION_DEFERRED':
                        pending.append(d)  # admission only; no scientific work has begun
                    else:
                        lease = OUT/d/'lease_status.json'
                        finished[d] = dict(exit_code=code, state=state['state'],
                            lease_released=lease.exists() and read(lease)['state'] == 'released')
                    del active[d]
            if pending and time.monotonic()-start > CFG['resource_wait_seconds']:
                for d in pending:
                    write(OUT/d/'status.json', dict(state='RESOURCE_WAIT_LIMIT', updated_at=stamp()))
                    finished[d] = dict(exit_code=124, state='RESOURCE_WAIT_LIMIT', lease_released=True)
                pending = []
            inventory = OLD.gpu_inventory()
            free = OLD.available_gpus(inventory, {j['gpu']['uuid'] for j in active.values()}, CFG['max_parallel']-len(active))
            for gpu in free:
                if not pending:
                    break
                d = pending.pop(0); attempts[d] += 1; folder = OUT/d
                log = (folder/f'worker_attempt{attempts[d]}.log').open('x')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['uuid'], OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                    OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
                write(folder/'status.json', dict(state='STARTING', gpu_index=gpu['index'], gpu_uuid=gpu['uuid'], updated_at=stamp()))
                process = subprocess.Popen([sys.executable, '-u', '-m', MODULE, 'worker', '--domain', d],
                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                active[d] = dict(process=process, log=log, started=time.monotonic(), gpu=gpu)
            write(OUT/'status.json', dict(state='RUNNING' if active else 'WAITING_FOR_RESOURCES' if pending else 'FINALIZING',
                pid=os.getpid(), updated_at=stamp(), elapsed_seconds=time.monotonic()-start, pending=pending, finished=finished,
                active={d: dict(pid=j['process'].pid, gpu_index=j['gpu']['index'], gpu_uuid=j['gpu']['uuid'],
                    worker=read(OUT/d/'status.json')) for d, j in active.items()},
                gpu_inventory=inventory, required_free_mib=CFG['minimum_free_mib'], test_read=False))
            if pending or active:
                time.sleep(10)
        complete = len(finished) == 2 and all(v['exit_code'] == 0 and v['state'] == 'COMPLETED' and v['lease_released'] for v in finished.values())
        if complete:
            summarize()
        write(OUT/'status.json', dict(state='COMPLETED' if complete else 'INCOMPLETE', finished=finished,
            elapsed_seconds=time.monotonic()-start, updated_at=stamp(), test_read=False, report=str(REPORT.relative_to(ROOT))))
    finally:
        for job in active.values():
            if job['process'].poll() is None:
                job['process'].terminate()
            job['log'].close()


def main():
    global ACTIVE_LEASE
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('test', 'prepare', 'worker', 'train', 'generate', 'supervise', 'summarize'))
    parser.add_argument('--domain', choices=DOMAINS)
    parser.add_argument('--lease-pid', type=int); args = parser.parse_args()

    def interrupted(signum, frame):
        raise InterruptedError('Single-fusion supervisor interrupted')

    signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.mode == 'test':
            print(json.dumps(C.tests()))
        elif args.mode == 'prepare':
            prepare()
        elif args.mode == 'worker':
            assert args.domain
            return worker(args.domain)
        elif args.mode in ('train', 'generate'):
            assert args.domain and args.lease_pid
            ACTIVE_LEASE = LeaseHandle(args.lease_pid)
            assert ACTIVE_LEASE.poll() is None
            torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
            if args.mode == 'train':
                torch.set_num_threads(2); torch.set_num_interop_threads(1)
                torch.cuda.set_per_process_memory_fraction(CFG['allocation_limit_gib']*2**30/torch.cuda.get_device_properties(0).total_memory)
                train_and_export(args.domain)
            else:
                Generation(args.domain, ACTIVE_LEASE).run()
        elif args.mode == 'supervise':
            supervise()
        else:
            summarize()
        return 0
    except BaseException as exc:
        if OUT.exists() and args.mode != 'worker':
            destination = OUT/args.domain if args.mode in ('train', 'generate') else OUT
            write(destination/'status.json', dict(state='FAILED', error=repr(exc), traceback=traceback.format_exc(), updated_at=stamp()))
        raise


if __name__ == '__main__':
    sys.exit(main())
