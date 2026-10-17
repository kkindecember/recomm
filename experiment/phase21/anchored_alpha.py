"""Reverse-KL anchored user guidance strength, SASRec + BGE expert pair.

Frozen CE expert mixture (beta) and the existing AGRec-inspired decoder rule are
reused unchanged. Only the 33 alpha parameters are trained, under four
prescribed objectives. The anchored arms add one pool-level term that penalises
removing mass from candidates GRAM already ranked well; the forward-KL arm is a
direction control, not a tuning choice. Lambda is fixed by a mechanical rule
computed before any ranking outcome is inspected.

Scientific scope is stated in the protocol copy under OUT/protocol.md.
"""
import argparse
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
from experiment.phase21 import joint_fusion_core as J
from experiment.phase21 import qwen4_alpha_rank_core as A
from experiment.phase21 import anchored_alpha_core as K
from experiment.phase21.candidate_gate_core import fold
from experiment.phase20 import lime_clean_benefit as CLEAN
from experiment.phase20.catalog_scope_core import metrics, contrast

ROOT = B.ROOT
INPUT = B.OUT                                                  # O10 samples/experts/predictions
BASE = ROOT/'artifacts/phase21/single_fusion_20261014_v1'      # bge parent heads + features
JF = ROOT/'artifacts/phase21/joint_fusion_20261015_v2'         # fit samples, catalog, fold parent
SRC = ROOT/'artifacts/phase20/lime_clean_benefit_20260922_v1'  # native candidates, frequencies
OUT = ROOT/'artifacts/phase21/anchored_alpha_20261017_v1'
PLAN = ROOT/'plan/第二十一阶段/21.49_SASRec与BGE的反向KL锚定用户干预强度_20261017.md'
REPORT = ROOT/'report/第二十一阶段/SASRec与BGE的反向KL锚定用户干预强度_20261017.md'
MODULE = 'experiment.phase21.anchored_alpha'
DOMAINS = ('Toys', 'Beauty')
ARMS = ('fixed', 'user_nll', 'user_kl', 'user_kl_fwd')
# Mechanical lambda: ceiling of the NLL/KL exchange rate to the grid argmin-NLL
# strength. Computed on ranker_fit before any ranking outcome was inspected;
# identical to two significant figures in both domains, so one value is used.
LAMBDA = 0.01
GRID = (0.3, 0.45, 0.6, 0.75, 0.9, 0.99)
CFG = dict(seed=0, epochs=20, checkpoints=[0, 4, 8, 12, 20], batch_size=32, lr=.002,
    weight_decay=.001, score_scale=20., alpha_initial=.3, beam_size=50, length_penalty=1.,
    arms=list(ARMS), expert_pair=['SASRec', 'BGE'], user_dimension=10, hidden_width=32,
    total_parameters=451, adapted_parameters=33, frozen_beta=True, frozen_shared_trunk=True,
    loss='target-path NLL; anchored arms add lambda * pool KL (reverse or forward)',
    lambda_value=LAMBDA, lambda_rule='ceiling of NLL/KL exchange rate at grid argmin NLL',
    lambda_sensitivity_grid=[0.001, 0.01, 0.1], lambda_sensitivity_is_selection=False,
    pool='target + native beam top10 + SASRec top10 + BGE top10, max 31',
    evaluation_users=512, input_validation_slice=[512, 1024], history_filter=False,
    history_penalty=False, test_read=False, expert_training=False, gram_optimizer_steps=0,
    total_lease_mib=16384, expected_workload_peak_mib=12288, minimum_free_mib=18432,
    allocation_limit_gib=11, max_parallel=2, worker_timeout_seconds=14400,
    resource_wait_seconds=86400, scientific_auto_retry=False,
    stop_after='four arms x fixed new512 per domain; no automatic next run')
read, write, sha, stamp = B.read, B.write, B.sha, B.stamp


def progress(domain, stage, **extra):
    # Callers report their own users_total (fit users during caching, dev users
    # during generation); nothing is assumed here.
    write(OUT/domain/'status.json', dict(state='RUNNING', stage=stage, pid=os.getpid(),
        updated_at=stamp(), gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'), test_read=False, **extra))
    print(json.dumps(dict(domain=domain, stage=stage, **extra)), flush=True)


def prepare():
    checks = K.tests()
    assert read(BASE/'status.json')['state'] == 'COMPLETED'
    assert read(JF/'status.json')['state'] == 'COMPLETED'
    assert read(INPUT/'status.json')['state'] in ('COMPLETED', 'INCOMPLETE', 'PREPARED', 'RUNNING')
    OUT.mkdir(exist_ok=False)
    write(OUT/'config.json', CFG); write(OUT/'tests.json', checks)
    shutil.copy2(PLAN, OUT/'protocol.md')
    manifest = {}

    def use(path, expected=None):
        rel = str(path.relative_to(ROOT))
        digest = manifest.get(rel) or sha(path)
        assert expected is None or digest == expected, rel
        manifest[rel] = digest
        return rel

    for domain in DOMAINS:
        p = OUT/domain; p.mkdir(); files = []
        for name in ('catalog.json', 'data_config.json', 'fit_samples.json',
                     'teacher.npz', 'teacher_checks.json'):
            src = JF/domain/name
            files.append(use(src)); shutil.copy2(src, p/name); files.append(use(p/name))
        assert sha(p/'teacher.npz') == read(p/'teacher_checks.json')['sha256']
        fit = read(p/'fit_samples.json')
        # Parents: full-fit bge head, and the internal-fold bge head.
        full = torch.load(BASE/domain/'user.pt', map_location='cpu')
        fp = torch.load(JF/domain/'parents'/'bge'/'fold.pt', map_location='cpu')
        torch.save(full, p/'parent_full.pt'); torch.save(fp, p/'parent_fold.pt')
        files += [use(BASE/domain/'user.pt'), use(JF/domain/'parents'/'bge'/'fold.pt'),
                  use(p/'parent_full.pt'), use(p/'parent_fold.pt')]
        assert full['fit_users'] == [s['user_id'] for s in fit]
        fi = [s['user_id'] for s in fit if fold(s['user_id'], domain) != 0]
        assert fp['fit_users'] == fi and len(set(fi)) == len(fi)
        assert set(fi).isdisjoint(s['user_id'] for s in fit if s['user_id'] not in fi)
        for name in ('user.npy', 'item.npy', 'metadata.json'):
            files.append(use(BASE/domain/'train'/name))
        shutil.copy2(BASE/domain/'train'/'user.npy', p/'train_user.npy')
        files.append(use(p/'train_user.npy'))
        # Development users: the first 512 of wide4096, disjoint from every fit user.
        samples = read(INPUT/domain/'samples.json')[512:1024]
        assert len(samples) == len({s['user_id'] for s in samples}) == 512
        assert all(s['cohort'] == 'wide4096' for s in samples)
        assert {s['user_id'] for s in samples}.isdisjoint(s['user_id'] for s in fit)
        write(p/'samples.json', samples)
        files += [use(INPUT/domain/'samples.json'), use(p/'samples.json')]
        users = np.load(INPUT/domain/'features'/'validation'/'SASRec__BGE.npy', mmap_mode='r')[512:1024]
        np.save(p/'dev_user.npy', users)
        source = np.load(INPUT/domain/'features'/'validation'/'experts.npy', mmap_mode='r')
        np.save(p/'dev_experts.npy', np.take(source[512:1024], (0, 3), axis=-1))
        files += [use(p/'dev_user.npy'), use(p/'dev_experts.npy')]
        # Reference rows: native plus the already-trained SASRec__BGE arm (a
        # non-matched reference; the anchored arms are trained here from scratch).
        refs = []
        with (INPUT/domain/'predictions.jsonl').open() as stream:
            for j, line in enumerate(stream):
                if j >= 1024: break
                if j < 512: continue
                refs.append(json.loads(line))
        assert len(refs) == 512
        for s, r in zip(samples, refs):
            assert all(s[k] == r[k] for k in ('user_id', 'target', 'history'))
        write(p/'references.json', refs); files.append(use(p/'references.json'))
        # Candidate pool: target first, then native beam and both expert top10.
        native = {}
        for line in (SRC/domain/'candidates.jsonl').read_text().splitlines():
            row = json.loads(line)
            if row['role'] == 'ranker_fit': native[row['user_id']] = row
        e = np.load(BASE/domain/'train'/'item.npy', mmap_mode='r')
        pool = np.zeros((len(fit), 31), np.int64); valid = np.zeros_like(pool, dtype=bool)
        for j, s in enumerate(fit):
            r = native[s['user_id']]; assert r['target'] == s['target']
            ids = np.asarray(r['ids']); scores = np.asarray(r['native'])
            top = ids[np.lexsort((ids, -scores))][:10].tolist()
            experts = [(np.lexsort((np.arange(e.shape[1]), -e[j, :, col]))[:10]+1).tolist() for col in (0, 1)]
            choices = [s['target']]
            for item in top+experts[0]+experts[1]:
                if item not in choices: choices.append(int(item))
            pool[j] = s['target']; pool[j, :len(choices)] = choices; valid[j, :len(choices)] = True
        np.savez(p/'pool.npz', ids=pool, valid=valid, users=np.asarray([s['user_id'] for s in fit]))
        files.append(use(p/'pool.npz'))
        assert np.load(p/'pool.npz')['users'].tolist() == np.load(p/'teacher.npz')['users'].tolist()
        cfg = read(p/'data_config.json')
        files.append(use(ROOT/cfg['parent_checkpoint'], cfg['parent_sha256']))
        write(p/'status.json', dict(state='PREPARED', updated_at=stamp(), fit_users=len(read(p/'fit_samples.json')),
            internal_fit_users=len(fi), evaluation_users=512, test_read=False))
    common = []
    for path in (Path(__file__), ROOT/'experiment/phase21/anchored_alpha_core.py',
                 ROOT/'experiment/phase21/qwen4_alpha_rank_core.py',
                 ROOT/'experiment/phase21/joint_fusion_core.py',
                 ROOT/'experiment/phase21/old10_expert_selection.py',
                 ROOT/'experiment/phase20/lime_decode_screen_core.py',
                 ROOT/'experiment/phase20/lime_clean_benefit.py',
                 PLAN, ROOT/'experiment/phase21/run_anchored_alpha.sh', OUT/'config.json'):
        common.append(use(path))
        dest = OUT/'sources'/path.relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    write(OUT/'manifest.json', dict(sha256=manifest,
        domain_inputs={d: common+[r for r in manifest if r.startswith(f'artifacts/phase21/anchored_alpha_20261017_v1/{d}/')] for d in DOMAINS}))
    write(OUT/'status.json', dict(state='PREPARED', updated_at=stamp(), test_read=False))
    print(json.dumps(dict(state='PREPARED', tests=checks, files=len(manifest))), flush=True)


class Teacher(CLEAN.Worker):
    """Batched teacher forcing over pool candidate paths; GRAM weights frozen."""

    def status(self, state='RUNNING', **extra):
        progress(self.domain, 'teacher_'+self.stage, **extra)

    def context(self, j, override_index=None):
        return super().context(j, override_index=j)

    def values(self, ctx, paths):
        from model.gram_t5_outputs import BaseModelOutputWithPastAndCrossAttentions
        depth = self.max_length-1; n = len(paths)
        tokens = torch.zeros(n, depth, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(tokens)
        for j, path in enumerate(paths):
            tokens[j, :len(path)-1] = torch.as_tensor(path[:-1], device=self.device)
            mask[j, :len(path)-1] = 1
        with torch.no_grad():
            output = self.model(attention_mask=ctx['mask'].reshape(1, -1).expand(n, -1),
                encoder_outputs=BaseModelOutputWithPastAndCrossAttentions(
                    last_hidden_state=ctx['hidden'].expand(n, -1, -1)),
                decoder_input_ids=tokens, decoder_attention_mask=mask, use_cache=False, return_dict=True)
            lp = output.logits.log_softmax(-1)
            native = np.zeros((n, depth), np.float32); legal = native.copy()
            for j, path in enumerate(paths):
                for d in range(1, len(path)):
                    children = list(self.index.children[self.index.lookup[tuple(path[:d])]])
                    native[j, d-1] = float(lp[j, d-1, path[d]])
                    legal[j, d-1] = float(torch.logsumexp(lp[j, d-1, children], 0))
        return native, legal


def cache(domain):
    CLEAN.OUT = OUT
    w = Teacher(domain)
    from experiment.phase18.core.diff_data import DiffDataset
    actual = read(w.folder/'fit_samples.json')
    w.samples = [dict(s, target=1) for s in actual]; w.data = DiffDataset(w.samples, w.catalog)
    pool = np.load(w.folder/'pool.npz'); ids, valid = pool['ids'], pool['valid']
    assert pool['users'].tolist() == [s['user_id'] for s in actual]
    n, k = ids.shape; depth = w.max_length-1
    native = np.zeros((n, k, depth), np.float32); legal = native.copy()
    checks = []; times = []; start = time.monotonic()
    old = np.load(w.folder/'teacher.npz')
    assert old['users'].tolist() == pool['users'].tolist()
    assert w.initial_digest == read(w.folder/'teacher_checks.json')['model_digest']
    for j, s in enumerate(actual):
        began = time.monotonic(); ctx = w.context(j)
        paths = [w.catalog['decoder_paths'][item-1] for item in ids[j, valid[j]]]
        nt, mt = w.values(ctx, paths); native[j, valid[j]] = nt; legal[j, valid[j]] = mt
        native[j, ~valid[j]] = nt[0]; legal[j, ~valid[j]] = mt[0]
        length = len(paths[0])-1
        np.testing.assert_allclose(nt[0, :length], old['native'][j, :length], atol=4e-5, rtol=2e-5)
        np.testing.assert_allclose(mt[0, :length], old['legal'][j, :length], atol=4e-5, rtol=2e-5)
        if j in (0, n//2, n-1):
            decoded = w.decode(ctx); expected = dict(zip(decoded['ids'], decoded['scores']))
            for col, item in enumerate(ids[j, valid[j]]):
                if item in expected:
                    np.testing.assert_allclose(float(nt[col, :len(paths[col])-1].mean()), expected[item],
                        atol=4e-5, rtol=2e-5)
            checks.append(dict(user=s['user_id'], batched_teacher_matches_beam=True))
        times.append(time.monotonic()-began)
        if j == 0 or (j+1) % 32 == 0 or j+1 == n:
            progress(domain, 'cache_competitor_paths', users_completed=j+1, users_total=n,
                elapsed_seconds=time.monotonic()-start,
                remaining_seconds_estimate=float(np.median(times[-64:])*(n-j-1)))
    assert w.digest() == w.initial_digest
    np.savez(w.folder/'path_teacher.npz', native=native, legal=legal)
    write(w.folder/'path_teacher_checks.json', dict(state='PASSED', all_gold_paths_match_existing=True,
        checks=checks, gram_digest_unchanged=True, users=n, max_paths=k,
        elapsed_seconds=time.monotonic()-start, model_digest=w.initial_digest,
        sha256=sha(w.folder/'path_teacher.npz')))


def fit(domain):
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(11*2**30/torch.cuda.get_device_properties(0).total_memory)
    device = torch.device('cuda:0'); p = OUT/domain
    samples = read(p/'fit_samples.json'); u = np.load(p/'train_user.npy'); v = np.load(p/'dev_user.npy')
    pool = np.load(p/'pool.npz'); items = torch.from_numpy(pool['ids']-1).to(device)
    cmask = torch.from_numpy(pool['valid']).to(device)
    teacher = np.load(p/'path_teacher.npz')
    assert sha(p/'path_teacher.npz') == read(p/'path_teacher_checks.json')['sha256']
    native, legal = [torch.from_numpy(teacher[key]).to(device) for key in ('native', 'legal')]
    tree = J.PathMass(read(p/'catalog.json')['decoder_paths'], device)
    v3 = tree.valid[items]
    e = np.load(BASE/domain/'train'/'item.npy', mmap_mode='r')
    fi = np.asarray([j for j, s in enumerate(samples) if fold(s['user_id'], domain) != 0])
    hi = np.asarray([j for j, s in enumerate(samples) if fold(s['user_id'], domain) == 0])
    parents = {name: torch.load(p/('parent_'+name+'.pt'), map_location='cpu') for name in ('fold', 'full')}
    tensors = {}; qcache = {}; features = {}
    for name, parent in parents.items():
        norm = [torch.from_numpy(np.asarray(parent[key][:10])).to(device) for key in ('mean', 'std')]
        features[name] = J.normalize(torch.from_numpy(u).to(device), *norm)
        model = A.AlphaOnly(parent['state_dict']).to(device)
        q = torch.zeros_like(native)
        for start in range(0, len(u), 32):
            stop = min(start+32, len(u)); ix = slice(start, stop)
            with torch.no_grad():
                beta, _ = model(features[name][ix])
                et = torch.from_numpy(np.array(e[ix, :, :2])).to(device)
                scores = (beta[:, None, :]*et).sum(-1)
                value, vm = A.pool_logs(tree, scores, items[ix]); q[ix] = value
                ref = tree.logs(scores, items[ix, 0])
                torch.testing.assert_close(value[:, 0][vm[:, 0]], ref[vm[:, 0]], atol=3e-5, rtol=2e-5)
        qcache[name] = q
        progress(domain, 'cache_expert_path_mass', parent=name, users_completed=len(u), users_total=len(u))
        del model
    # s_gram is the alpha=0 mixed score, which is exactly the native path mean and
    # therefore identical for both parents; computed once.
    s_gram = A.sequence_scores(native.double(), legal.double(), qcache['full'].double(),
        torch.zeros(len(u), dtype=torch.float64, device=device), v3)
    sensitivity = []
    for a in GRID:
        alpha = torch.full((len(u),), a, dtype=torch.float64, device=device)
        sm = A.sequence_scores(native.double(), legal.double(), qcache['full'].double(), alpha, v3)
        sensitivity.append(dict(alpha=a, target_nll=-float(sm[:, 0].mean()),
            reverse_kl=float(K.kl_divergence(s_gram, sm, cmask).mean())))
    write(p/'lambda_calibration.json', dict(grid=sensitivity, selected_lambda=LAMBDA,
        rule=CFG['lambda_rule'], used_for_selection=False,
        note='pre-registered constant; sensitivity values are descriptive only'))
    all_ix = np.arange(len(u)); info = {}
    alphas = {'fixed': np.full(len(v), .3)}
    full = parents['full']
    vn = J.normalize(torch.from_numpy(v).to(device),
        *[torch.from_numpy(np.asarray(full[k][:10])).to(device) for k in ('mean', 'std')])
    base = A.AlphaOnly(full['state_dict']).to(device)
    with torch.no_grad(): bn = base(vn)[0].cpu().numpy()
    ev = np.load(p/'dev_experts.npy'); scores = (bn[:, None, :]*ev).sum(-1)
    np.save(p/'signals.npy', scores); np.save(p/'betas.npy', bn)
    original_beta = {k: val.detach().cpu().clone() for k, val in base.state_dict().items()
                     if not k.startswith('strength.')}
    kinds = {'user_nll': 'nll', 'user_kl': 'kl', 'user_kl_fwd': 'kl_fwd'}
    for arm, kind in kinds.items():
        def run(parent_name, ix, epochs, select):
            torch.manual_seed(0); torch.cuda.manual_seed_all(0)
            model = A.AlphaOnly(parents[parent_name]['state_dict']).to(device)
            opt = torch.optim.AdamW(model.strength.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
            rng = np.random.default_rng(0); curve = []
            def loss(rows):
                _, aa = model(features[parent_name][rows])
                z = A.sequence_scores(native[rows], legal[rows], qcache[parent_name][rows], aa, v3[rows])
                value, penalty = K.anchored_objective(z, s_gram[rows], z, cmask[rows], kind, LAMBDA)
                return value
            for epoch in range(epochs+1):
                if epoch:
                    order = rng.permutation(ix)
                    for start in range(0, len(ix), 32):
                        opt.zero_grad(); value = loss(order[start:start+32]); assert torch.isfinite(value)
                        value.backward()
                        assert all(x.grad is None or torch.isfinite(x.grad).all() for x in model.parameters())
                        opt.step()
                value = None
                if select and epoch in CFG['checkpoints']:
                    with torch.no_grad():
                        value = sum(float(loss(hi[k:k+32]))*len(hi[k:k+32]) for k in range(0, len(hi), 32))/len(hi)
                    curve.append(dict(epoch=epoch, loss=value))
                if epoch in CFG['checkpoints']:
                    progress(domain, 'fit_alpha', arm=arm, phase='selection' if select else 'refit',
                        epoch=epoch, epochs=epochs, internal_heldout_loss=value)
            return model, curve
        selected, curve = run('fold', fi, 20, True)
        epoch = min(curve, key=lambda x: (x['loss'], x['epoch']))['epoch']; del selected
        model, _ = run('full', all_ix, epoch, False)
        for key, value in original_beta.items():
            torch.testing.assert_close(model.state_dict()[key].cpu(), value, rtol=0, atol=0)
        with torch.no_grad():
            beta, alpha = model(vn)
            np.testing.assert_array_equal(beta.cpu().numpy(), bn)
            alphas[arm] = alpha.cpu().double().numpy()
        aa = alphas[arm]
        info[arm] = dict(selected_epochs=epoch, selection_curve=curve, kind=kind, lambda_value=LAMBDA,
            alpha_mean=float(aa.mean()), alpha_quantiles=np.quantile(aa, [0, .1, .5, .9, 1]).tolist(),
            beta_exactly_unchanged=True, total_parameters=451, adapted_parameters=33)
        torch.save(dict(state_dict={k: val.detach().cpu().clone() for k, val in model.state_dict().items()},
            mean=full['mean'][:10], std=full['std'][:10], fit_users=[s['user_id'] for s in samples],
            config=CFG, **info[arm]), p/(arm+'.pt'))
        del model
    np.savez(p/'alphas.npz', **alphas)
    write(p/'training.json', dict(fits=info, test_read=False, evaluation_used_for_epoch_selection=False,
        internal_fit_users=len(fi), internal_heldout_users=len(hi), full_fit_users=len(u),
        internal_parent_excludes_heldout_labels=True, alpha_features_exclude_target=True))
    files = [p/n for n in (*[a+'.pt' for a in kinds], 'alphas.npz', 'betas.npy', 'signals.npy',
             'training.json', 'lambda_calibration.json', 'path_teacher.npz', 'path_teacher_checks.json')]
    write(p/'training_manifest.json', dict(sha256={str(f.relative_to(ROOT)): sha(f) for f in files}))


class Generation(CLEAN.Worker):
    def __init__(self, domain, lease_pid):
        self.lease = B.S.LeaseHandle(lease_pid); self.row_seconds = []; CLEAN.OUT = OUT
        super().__init__(domain)
        self.truth = self.samples
        from experiment.phase18.core.diff_data import DiffDataset
        self.samples = [dict(s, target=1) for s in self.truth]; self.data = DiffDataset(self.samples, self.catalog)
        self.refs = read(self.folder/'references.json')
        self.scores = np.load(self.folder/'signals.npy'); self.alpha = dict(np.load(self.folder/'alphas.npz'))

    def context(self, j, override_index=None):
        return super().context(j, override_index=j)

    def status(self, state='RUNNING', **extra):
        assert self.lease.poll() is None, 'GPU lease exited'
        done = getattr(self, 'users_completed', 0)
        rate = float(np.median(self.row_seconds[-32:])) if self.row_seconds else None
        write(OUT/self.domain/'status.json', dict(state=state, stage=self.stage, pid=os.getpid(),
            updated_at=stamp(), gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'), users_completed=done,
            users_total=512, completed_decodes=self.completed, elapsed_seconds=time.monotonic()-self.started,
            median_seconds_per_user=rate, remaining_generation_seconds_estimate=rate*(512-done) if rate else None,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20, test_read=False, **extra))

    def run(self):
        self.stage = 'native_zero_smoke'; self.status(); checks = []
        assert self.initial_digest == read(self.folder/'teacher_checks.json')['model_digest']
        for j in (0, 256, 511):
            ctx = self.context(j); native = self.decode(ctx)
            B.Generation.equal(dict(ids=native['ids'], scores=native['scores'].tolist()), self.refs[j]['native'])
            q = self.index.conditional_logs(self.scores[j]); zero = self.decode(ctx, q, 0.)
            assert zero['paths'] == native['paths']
            np.testing.assert_array_equal(zero['scores'], native['scores'])
            self.samples[j]['target'] = 2; changed = self.context(j); self.samples[j]['target'] = 1
            torch.testing.assert_close(ctx['ids'], changed['ids'], atol=0, rtol=0)
            torch.testing.assert_close(ctx['mask'], changed['mask'], atol=0, rtol=0)
            checks.append(dict(row=j, native_reproduced=True, alpha_zero_exact=True,
                target_placeholder_invariant=True))
        write(self.folder/'smoke.json', dict(state='PASSED', checks=checks))
        self.stage = 'four_arm_generation'; self.status()
        with (self.folder/'predictions.jsonl').open('x') as stream:
            for j, s in enumerate(self.truth):
                start = time.monotonic(); ctx = self.context(j)
                q = self.index.conditional_logs(self.scores[j])
                row = dict(user_id=s['user_id'], target=s['target'], history=s['history'],
                    native=self.refs[j]['native'], sas_bge_reference=self.refs[j]['SASRec__BGE'],
                    legacy_three_reference=self.refs[j]['legacy_three_guided_reference'])
                same = {}
                for arm in ARMS:
                    aa = float(self.alpha[arm][j])
                    if aa in same:
                        row[arm] = row[same[aa]]; continue
                    decoded = self.decode(ctx, q, aa)
                    row[arm] = dict(ids=decoded['ids'], scores=decoded['scores'].tolist()); same[aa] = arm
                stream.write(json.dumps(row)+'\n'); stream.flush()
                self.users_completed = j+1; self.row_seconds.append(time.monotonic()-start)
                if j == 0 or (j+1) % 16 == 0 or j+1 == len(self.truth): self.status()
        assert self.digest() == self.initial_digest
        write(self.folder/'runtime_checks.json', dict(state='PASSED', gram_model_digest_unchanged=True,
            gram_optimizer_steps=0, beta_frozen=True, history_filter=False, all_users_and_arms=True))
        self.stage = 'complete'; self.status('COMPLETED')


def worker(domain):
    p = OUT/domain; lease = child = None
    def interrupted(signum, frame): raise InterruptedError('Anchored alpha interrupted or timed out')
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGALRM, interrupted)
    signal.alarm(CFG['worker_timeout_seconds'])
    try:
        progress(domain, 'verify_inputs'); m = read(OUT/'manifest.json')
        assert read(OUT/'config.json') == CFG
        for rel in m['domain_inputs'][domain]: assert sha(ROOT/rel) == m['sha256'][rel], rel
        gpu = os.environ['CUDA_VISIBLE_DEVICES']
        free = next(g['free_mib'] for g in B.S.OLD.gpu_inventory() if g['uuid'] == gpu)
        if free < CFG['minimum_free_mib']:
            write(p/'status.json', dict(state='ADMISSION_DEFERRED', free_mib=free, updated_at=stamp())); return 75
        with (p/'lease.log').open('x') as log:
            path = p/'lease_status.json'
            lease = subprocess.Popen([sys.executable, str(ROOT/'experiment/gpu_memory_lease.py'), '--gpu', '0',
                '--total-lease-mib', str(CFG['total_lease_mib']),
                '--expected-workload-peak-mib', str(CFG['expected_workload_peak_mib']),
                '--status-path', str(path)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write(p/'lease_process.json', dict(pid=lease.pid, gpu_uuid=gpu, started_at=stamp()))
            start = time.monotonic()
            while not path.exists() or read(path).get('state') != 'holding':
                assert lease.poll() is None, 'Lease exited'
                assert time.monotonic()-start < 180, 'Lease timeout'; time.sleep(1)
            for phase in ('cache', 'fit', 'generate'):
                assert lease.poll() is None, 'Lease exited'
                child = subprocess.Popen([sys.executable, '-u', '-m', MODULE, phase, '--domain', domain,
                    '--lease-pid', str(lease.pid)], cwd=ROOT, start_new_session=True)
                assert child.wait() == 0, ('Scientific phase failed; no automatic retry', phase)
        return 0
    finally:
        signal.alarm(0); B.stop(child); B.stop(lease)


def summarize():
    keys = ('native', *ARMS, 'sas_bge_reference', 'legacy_three_reference'); result = {}
    lines = ['# SASRec＋BGE：反向KL锚定用户干预强度', '', '## Material Passport', '',
        '- Origin Skill: academic-research-suite / experiment-agent', '- Origin Mode: validate',
        '- Origin Date: 2026-10-17', '- Verification Status: ANALYZED；真实生成开发集，非独立test',
        '- Version Label: anchored_alpha_v1', '',
        '每域同批512名wide开发用户，排除反复使用的old512。冻结同一CE用户β与共享隐藏层，仅α的33参数可训练；',
        f'锚定臂在候选池上加入 λ={LAMBDA} 的 KL 项（反向为主，前向为方向对照），λ由机械规则在查看任何排序结果前确定。',
        '最终排名不排除历史商品。','']
    for domain in DOMAINS:
        p = OUT/domain; samples = read(p/'samples.json')
        rows = [json.loads(line) for line in (p/'predictions.jsonl').read_text().splitlines()]
        assert len(rows) == len(samples) == 512
        ranks = {key: [] for key in keys}
        for s, row in zip(samples, rows):
            assert all(row[k] == s[k] for k in ('user_id', 'target', 'history'))
            for key in keys: ranks[key].append(B.rank(row[key], s['target']))
        ranks = {k: np.asarray(v, np.int64) for k, v in ranks.items()}
        np.savez_compressed(p/'ranks.npz', **ranks)
        pairs = [(arm, 'native') for arm in ARMS] + [(a, 'fixed') for a in ARMS[1:]] \
            + [('user_kl', 'user_nll'), ('user_kl_fwd', 'user_kl')]
        result[domain] = dict(metrics={k: metrics(r) for k, r in ranks.items()},
            comparisons={a+'_vs_'+b: contrast(ranks[a], ranks[b], repeats=2000, seed=20261017) for a, b in pairs},
            training=read(p/'training.json'), lambda_calibration=read(p/'lambda_calibration.json'))
        write(p/'summary.json', result[domain])
        files = [p/n for n in ('predictions.jsonl', 'ranks.npz', 'summary.json', 'smoke.json', 'runtime_checks.json')]
        write(p/'output_manifest.json', dict(sha256={str(f.relative_to(ROOT)): sha(f) for f in files}))
    for key in keys:
        a, b = [result[d]['metrics'] for d in DOMAINS]
        lines.append(f'| {key} | {a[key]["N10"]:.6f} | {100*(a[key]["N10"]/a["native"]["N10"]-1):+.2f}% | '
                     f'{b[key]["N10"]:.6f} | {100*(b[key]["N10"]/b["native"]["N10"]-1):+.2f}% |')
    lines += ['', '| 数据集 | 比较 | N10绝对差 | 描述性95%区间 |', '|---|---|---:|---|']
    for domain in DOMAINS:
        for name, x in result[domain]['comparisons'].items():
            ci = x['descriptive_95pct_N10']
            lines.append(f'| {domain} | {name} | {x["delta_N10"]:+.6f} | [{ci[0]:+.6f}, {ci[1]:+.6f}] |')
    review = dict(coverage='11/11', simpson='Separate domains; no pooled winner.',
        ecological='Paired user gains and losses saved; no every-user claim.',
        berkson='Fixed eligible validation hash sample, not the general population.',
        collider='No outcome or hit based user selection.',
        base_rate='All 512 prescribed users retained per domain.',
        regression_to_mean='Old512 excluded, but new512 remains reused development.',
        survivorship='All four prescribed arms and failures retained; full denominator kept.',
        look_elsewhere='All eight paired comparisons per domain saved; unadjusted descriptive bootstrap.',
        forking_paths='Lambda fixed by a stated mechanical rule before outcomes; sensitivity grid is descriptive.',
        causation='Only the training objective differs between arms; expert necessity untested.',
        reverse_causality='Targets used for supervised losses and metrics only, never as inference features.')
    write(OUT/'summary.json', dict(domains=result, test_read=False, statistical_review=review,
        automatic_final_selection=False))
    lines += ['', '池上的 KL 是逐候选代理量，不是全目录量；代理目标下降不保证最终 NDCG 提升，以本轮真实生成为准。',
        'sas_bge_reference 与 legacy_three_reference 是不同已训练融合器，不是本轮的匹配消融。',
        '统计检查11/11；各域8项配对区间未做多重比较校正，不含训练种子与方法选择不确定性。区间跨0不表示等效。']
    REPORT.write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('test', 'prepare', 'supervise', 'worker', 'cache', 'fit', 'generate'))
    parser.add_argument('--domain', choices=DOMAINS); parser.add_argument('--lease-pid', type=int)
    args = parser.parse_args()
    try:
        if args.mode == 'test': print(json.dumps(K.tests()), flush=True)
        elif args.mode == 'prepare': prepare()
        elif args.mode == 'supervise':
            B.OUT, B.CFG, B.MODULE, B.REPORT, B.summarize = OUT, CFG, MODULE, REPORT, summarize
            B.supervise()
        elif args.mode == 'worker': return worker(args.domain)
        elif args.mode == 'cache': cache(args.domain)
        elif args.mode == 'fit': fit(args.domain)
        else: Generation(args.domain, args.lease_pid).run()
        return 0
    except BaseException as exc:
        if OUT.exists():
            destination = OUT/args.domain if args.domain else OUT
            write(destination/'failure.json', dict(error=repr(exc), traceback=traceback.format_exc(),
                updated_at=stamp()))
            write(destination/'status.json', dict(state='FAILED', error=repr(exc), updated_at=stamp(),
                test_read=False))
        raise


if __name__ == '__main__':
    sys.exit(main())
