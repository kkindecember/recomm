"""Matched expert replacement with learned beta and guidance alpha.

Frozen GRAM target-path likelihood is a training proxy; every prespecified arm
is evaluated with fresh beam search. Historical development users are reused.
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

from experiment.phase21 import single_fusion as S
from experiment.phase21 import single_fusion_core as C
from experiment.phase21 import joint_fusion_core as J
from experiment.phase21.candidate_gate_core import fold
from experiment.phase20.catalog_scope_core import metrics, contrast

ROOT, SOURCE = S.ROOT, S.SOURCE
BASE = ROOT/'artifacts/phase21/single_fusion_20261014_v1'
QWEN = ROOT/'artifacts/phase21/qwen_text_expert_20261015_v1'
TIME = ROOT/'artifacts/phase21/time_behavior_expert_20261015_v1'
OUT = ROOT/'artifacts/phase21/joint_fusion_20261015_v2'
PLAN = ROOT/'plan/第二十一阶段/21.35_专家比例与生成强度联合融合_20261015.md'
REPORT = ROOT/'report/第二十一阶段/专家比例与生成强度联合融合_20261015.md'
MODULE = 'experiment.phase21.joint_fusion'
DOMAINS = ('Toys', 'Beauty')
BACKBONES = ('bge', 'qwen06', 'qwen4', 'tisas', 'tisas_qwen06')
MODES = ('fixed', 'calibrate', 'global', 'user')
ARMS = tuple(b+'_'+m for b in BACKBONES for m in MODES)
CFG = dict(S.CFG, arms=list(ARMS), backbones=list(BACKBONES), adaptation_modes=list(MODES),
    loss='mean per-user length-normalized frozen-GRAM target-path negative log probability',
    alpha_learned=True, alpha_initial=.3, alpha_numerical_bounds=[1e-6, 1.-1e-6],
    alpha_controls='fixed .3; frozen beta + global alpha; joint beta/global alpha; joint beta/user alpha',
    old_fusion_used_by_new_model=True, warm_start='each matching original CE beta; internal-fold parent retrained on fit fold only',
    epoch_selection='hash 80/20 target-path NLL; all-ranker_fit refit from matching all-fit CE parent',
    pretraining_epoch_selection='original CE-selected epoch for existing parents; identical CE selection for new TiSAS+Qwen06',
    hidden_width=32, active_beta_parameters=418, joint_global_parameters=419, joint_user_parameters=451,
    trainable_calibration_parameters=1, expert_models_frozen=True,
    all_arms_generate=True, test_read=False, scientific_auto_retry=False,
    stop_after='five expert pairs x four controls x two reused development512 pools, seed0')
read, write, sha, stamp = S.read, S.write, S.sha, S.stamp
LEASE = None


def progress(domain, stage, **kw):
    if LEASE is not None:
        assert LEASE.poll() is None, 'GPU lease exited'
    value = dict(state='RUNNING', stage=stage, updated_at=stamp(), pid=os.getpid(),
                 gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'), test_read=False, **kw)
    write(OUT/domain/'status.json', value)
    print(json.dumps(dict(domain=domain, **value)), flush=True)


def bind():
    S.OUT, S.REPORT, S.MODULE, S.CFG = OUT, REPORT, MODULE, CFG
    S.progress, S.summarize = progress, summarize


def origin(domain, backbone):
    if backbone == 'bge':
        return BASE/domain, BASE/domain/'user.pt'
    base = TIME if backbone == 'tisas' else QWEN
    if backbone == 'tisas_qwen06':
        return OUT/domain/'features'/backbone, OUT/domain/'parents'/backbone/'full.pt'
    return base/domain/'features'/backbone, base/'models'/backbone/domain/'user.pt'


def prepare():
    checks = J.tests()
    for root in (BASE, QWEN, TIME):
        assert read(root/'status.json')['state'] == 'COMPLETED'
    OUT.mkdir(exist_ok=False)
    write(OUT/'status.json', dict(state='PREPARING', updated_at=stamp()))
    write(OUT/'config.json', CFG)
    write(OUT/'tests.json', checks)
    shutil.copy2(PLAN, OUT/'protocol.md')
    manifest = {}
    def use(path, expected=None):
        relative = str(path.relative_to(ROOT))
        digest = manifest.get(relative) or sha(path)
        assert expected is None or digest == expected, relative
        manifest[relative] = digest
    for root in (BASE, QWEN, TIME):
        for relative, digest in read(root/'manifest.json')['sha256'].items():
            if relative.startswith(('experiment/', 'GRAM/src/', '.cache/huggingface/')):
                use(ROOT/relative, digest)
    for domain in DOMAINS:
        p = OUT/domain
        p.mkdir()
        for name in ('samples.json', 'catalog.json', 'data_config.json'):
            use(BASE/domain/name)
            shutil.copy2(BASE/domain/name, p/name)
            use(p/name)
        fit = [s for s in read(SOURCE/domain/'samples.json') if s['role'] == 'ranker_fit']
        dev = read(p/'samples.json')
        assert len(dev) == 512 and len({s['user_id'] for s in dev}) == 512
        assert {s['user_id'] for s in fit}.isdisjoint(s['user_id'] for s in dev)
        assert [s['user_id'] for s in fit] == [s['user_id'] for s in read(BASE/domain/'train/users.json')]
        write(p/'fit_samples.json', fit); use(p/'fit_samples.json')
        use(SOURCE/domain/'samples.json'); use(SOURCE/domain/'frequencies.npy')
        refs = [dict(user_id=s['user_id']) for s in dev]
        for root in (BASE, QWEN, TIME):
            for manifest_name in ('training_manifest.json', 'output_manifest.json'):
                for rel, digest in read(root/domain/manifest_name)['sha256'].items():
                    use(ROOT/rel, digest)
            rows = [json.loads(x) for x in (root/domain/'predictions.jsonl').read_text().splitlines()]
            for j, (row, sample) in enumerate(zip(rows, dev)):
                assert all(row[k] == sample[k] for k in ('user_id', 'target', 'history'))
                if 'native' in refs[j]:
                    assert refs[j]['native']['ids'] == row['native']['ids']
                    np.testing.assert_allclose(refs[j]['native']['scores'], row['native']['scores'], atol=2e-6, rtol=1e-5)
                refs[j]['native'] = row['native']
                mapping = {'bge': 'user'} if root == BASE else {'qwen06':'qwen06', 'qwen4':'qwen4'} if root == QWEN else {'tisas':'tisas'}
                for b, key in mapping.items():
                    for suffix in ('_guided', '_late'):
                        refs[j][b+'_fixed'+suffix] = row[key+suffix]
        write(p/'references.json', refs); use(p/'references.json')
        for b in BACKBONES[:-1]:
            folder, checkpoint = origin(domain, b)
            ck = torch.load(checkpoint, map_location='cpu')
            assert ck['fit_users'] == [s['user_id'] for s in fit]
            use(checkpoint)
            for part in ('train', 'validation'):
                for name in ('user.npy', 'item.npy', 'metadata.json'):
                    use(folder/part/name)
                assert np.load(folder/part/'user.npy', mmap_mode='r').shape == (len(fit) if part == 'train' else 512, 10)
        config = read(p/'data_config.json')
        use(ROOT/config['parent_checkpoint'], config['parent_sha256'])
        write(p/'status.json', dict(state='PREPARED', updated_at=stamp()))
        write(OUT/'status.json', dict(state='PREPARING', completed_domain=domain, updated_at=stamp()))
    for path in (Path(__file__), ROOT/'experiment/phase21/joint_fusion_core.py',
                 ROOT/'experiment/phase21/run_joint_fusion.sh', PLAN, OUT/'config.json'):
        use(path)
    for relative in manifest:
        path = ROOT/relative
        if path.suffix in ('.py', '.sh', '.md'):
            dest = OUT/'sources'/relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    write(OUT/'manifest.json', dict(sha256=manifest, created_at=stamp()))
    write(OUT/'status.json', dict(state='PREPARED', updated_at=stamp(), test_read=False))
    REPORT.write_text('# 专家比例与生成强度联合融合\n\n状态：PREPARED。五种专家组合、四档强度对照，尚无效果结论。\n')
    print(json.dumps(dict(state='PREPARED', hashes=len(manifest), tests=checks)), flush=True)


class Teacher(S.G.clean.Worker):
    def status(self, state='RUNNING', **kw):
        progress(self.domain, 'cache_'+self.stage, **kw)

    def context(self, j, override_index=None):
        return super().context(j, override_index=j)

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


def cache(domain, limit=None):
    from experiment.phase18.core.diff_data import DiffDataset
    S.G.clean.OUT = OUT
    worker = Teacher(domain)
    checks = []
    # Match teacher-forced logits against the actual incremental beam decoder,
    # including its legal-token mass. These are implementation checks only.
    for j in (0, 256, 511):
        ctx = worker.context(j)
        native = worker.decode(ctx)
        path = native['paths'][0]
        nt, mt = worker.path_values(ctx, path)
        expected_n, expected_m = [], []
        for d in range(1, len(path)):
            tokens, logits, mass = native['guide'].trace[tuple(path[:d])]
            expected_n.append(logits[tokens.index(path[d])]); expected_m.append(mass)
        np.testing.assert_allclose(nt, expected_n, atol=2e-5, rtol=1e-5)
        np.testing.assert_allclose(mt, expected_m, atol=2e-5, rtol=1e-5)
        checks.append(dict(user=worker.samples[j]['user_id'], teacher_incremental_max_error=float(np.max(np.abs(nt-expected_n)))))
    samples = read(OUT/domain/'fit_samples.json')
    worker.samples = samples[:limit] if limit else samples
    worker.data = DiffDataset(worker.samples, worker.catalog)
    n, depth = len(worker.samples), worker.max_length-1
    nt, mt = np.zeros((n, depth), np.float32), np.zeros((n, depth), np.float32)
    begin = time.monotonic()
    times = []
    for j, sample in enumerate(worker.samples):
        start = time.monotonic()
        ctx = worker.context(j)
        path = worker.catalog['decoder_paths'][sample['target']-1]
        a, b = worker.path_values(ctx, path)
        nt[j, :len(a)], mt[j, :len(b)] = a, b
        times.append(time.monotonic()-start)
        if j == 0 or (j+1)%32 == 0 or j+1 == n:
            progress(domain, 'cache_target_paths', completed=j+1, total=n,
                     elapsed_seconds=time.monotonic()-begin,
                     remaining_seconds_estimate=float(np.median(times[-64:])*(n-j-1)))
        del ctx
    assert worker.digest() == worker.initial_digest
    prefix = 'teacher_smoke' if limit else 'teacher'
    path = OUT/domain/(prefix+'.npz')
    np.savez(path, native=nt, legal=mt, users=np.asarray([s['user_id'] for s in worker.samples]),
             targets=np.asarray([s['target']-1 for s in worker.samples]))
    write(OUT/domain/(prefix+'_checks.json'), dict(state='PASSED', checks=checks,
        target_only_used_as_teacher_forcing_label=True, gram_digest_unchanged=True,
        model_digest=worker.initial_digest, users=n, sha256=sha(path), elapsed_seconds=time.monotonic()-begin))


def user_norm(user, indices):
    u = user[indices].astype(np.float64)
    mean = u.mean(0)
    std = np.sqrt(np.maximum(np.square(u).mean(0)-mean**2, 0))
    std[std < 1e-5] = 1.
    return mean.astype(np.float32), std.astype(np.float32)


def combined_features(domain):
    p, _ = origin(domain, 'tisas_qwen06')
    counts = np.load(SOURCE/domain/'frequencies.npy')
    for part, sample_name in (('train','fit_samples.json'), ('validation','samples.json')):
        samples = read(OUT/domain/sample_name)
        behavior = np.load(origin(domain,'tisas')[0]/part/'item.npy', mmap_mode='r')
        text = np.load(origin(domain,'qwen06')[0]/part/'item.npy', mmap_mode='r')
        dest = p/part; dest.mkdir(parents=True)
        user = np.empty((len(samples), 10), np.float32)
        item = np.lib.format.open_memmap(dest/'item.npy', mode='w+', dtype='float32', shape=behavior.shape)
        for j, sample in enumerate(samples):
            e = np.stack((behavior[j,:,0], text[j,:,1]))
            user[j], item[j] = C.features(e, sample['full_history'], counts)
            if j == 0 or (j+1)%256 == 0 or j+1 == len(samples):
                progress(domain, 'combined_features', part=part, completed=j+1, total=len(samples))
        item.flush(); np.save(dest/'user.npy', user)
        write(dest/'metadata.json', dict(users_order=[s['user_id'] for s in samples],
            behavior_source='cached TiSASRec', text_source='cached Qwen0.6B', target_used_in_features=False))
        del item


def fit(domain):
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(11*2**30/torch.cuda.get_device_properties(0).total_memory)
    device = torch.device('cuda:0')
    combined_features(domain)
    p = OUT/domain
    samples = read(p/'fit_samples.json'); dev = read(p/'samples.json')
    targets = np.asarray([s['target']-1 for s in samples])
    teacher = np.load(p/'teacher.npz')
    assert read(p/'teacher_checks.json')['sha256'] == sha(p/'teacher.npz')
    np.testing.assert_array_equal(teacher['users'], [s['user_id'] for s in samples])
    np.testing.assert_array_equal(teacher['targets'], targets)
    nt, mt = [torch.from_numpy(teacher[name]).to(device) for name in ('native','legal')]
    yt = torch.from_numpy(targets).to(device)
    tree = J.PathMass(read(p/'catalog.json')['decoder_paths'], device)
    fi = np.asarray([j for j,s in enumerate(samples) if fold(s['user_id'],domain) != 0])
    hi = np.asarray([j for j,s in enumerate(samples) if fold(s['user_id'],domain) == 0])
    all_ix = np.arange(len(samples))
    exported, alphas, betas, info = {}, {}, {}, {}

    for backbone in BACKBONES:
        folder, ckpath = origin(domain, backbone)
        u = np.load(folder/'train/user.npy')
        e = np.ascontiguousarray(np.load(folder/'train/item.npy', mmap_mode='r')[..., :2])
        ut, et = torch.from_numpy(u).to(device), torch.from_numpy(e).to(device)
        vn = np.load(folder/'validation/user.npy')
        ve = np.ascontiguousarray(np.load(folder/'validation/item.npy', mmap_mode='r')[..., :2])
        fit_norm = user_norm(u, fi)
        fit_nt = tuple(torch.from_numpy(x).to(device) for x in fit_norm)
        full_norm = user_norm(u, all_ix)

        def ce_parent(indices, norm, epochs, select=False):
            torch.manual_seed(0); torch.cuda.manual_seed_all(0)
            model = C.Fusion().to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
            rng = np.random.default_rng(0)
            normalized_u = J.normalize(ut, *norm)
            def loss(ix):
                weights = model(normalized_u[ix], None, 'user')
                scores = (weights*et[ix]).sum(-1)
                return torch.nn.functional.cross_entropy(20*scores, yt[ix])
            curve, selected_state, best = [], None, float('inf')
            for epoch in range(epochs+1):
                if epoch:
                    order = rng.permutation(indices)
                    for start in range(0, len(order), 32):
                        opt.zero_grad(); value = loss(order[start:start+32]); value.backward(); opt.step()
                if select and epoch in CFG['checkpoints']:
                    with torch.no_grad():
                        val = sum(float(loss(hi[k:k+32]))*len(hi[k:k+32]) for k in range(0,len(hi),32))/len(hi)
                    curve.append(dict(epoch=epoch, loss=val))
                    if val < best:
                        best = val; selected_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                if epoch:
                    progress(domain, 'CE_parent_fit', backbone=backbone, epoch=epoch, epochs=epochs, selection=select)
            state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            return (selected_state if select else state), curve

        if backbone == 'tisas_qwen06':
            fold_state, curve = ce_parent(fi, fit_nt, 20, select=True)
            selected = min(curve, key=lambda x:(x['loss'],x['epoch']))['epoch']
            full_state, _ = ce_parent(all_ix, tuple(torch.from_numpy(x).to(device) for x in full_norm), selected)
            parent = dict(state_dict=full_state, mean=full_norm[0], std=full_norm[1], selected_epochs=selected,
                fit_users=[s['user_id'] for s in samples], selection_curve=curve, config=CFG)
            ckpath.parent.mkdir(parents=True); torch.save(parent, ckpath)
        else:
            parent = torch.load(ckpath, map_location='cpu')
            assert parent['fit_users'] == [s['user_id'] for s in samples]
            np.testing.assert_array_equal(full_norm[0], parent['mean'][:10])
            np.testing.assert_array_equal(full_norm[1], parent['std'][:10])
            fold_state, curve = ce_parent(fi, fit_nt, parent['selected_epochs'])
        parent_folder = p/'parents'/backbone; parent_folder.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state_dict=fold_state, mean=fit_norm[0], std=fit_norm[1],
            fit_users=[samples[j]['user_id'] for j in fi], parent_epochs=parent['selected_epochs']), parent_folder/'fold.pt')
        assert {samples[j]['user_id'] for j in fi}.isdisjoint(samples[j]['user_id'] for j in hi)

        def export(model, norm, key, fixed=False):
            with torch.no_grad():
                v = torch.from_numpy(vn).to(device)
                norm_t = tuple(torch.from_numpy(x[:10]).to(device) for x in norm)
                beta, alpha = model(J.normalize(v, *norm_t))
                bn = beta.cpu().numpy()
                scores = (bn[:,None,:]*ve).sum(-1)
                a = np.full(len(vn), .3) if fixed else alpha.cpu().double().numpy()
            exported[key], alphas[key], betas[key] = scores, a, bn
            assert np.isfinite(scores).all() and np.all((a>0)&(a<1))
            # Independent NumPy MLP, covering all exported users.
            sd = {k:v.detach().cpu().numpy() for k,v in model.state_dict().items()}
            x = np.clip((vn-norm[0][:10])/norm[1][:10], -10, 10)
            h = np.maximum(x@sd['input.weight'].T+sd['input.bias'], 0)
            z = h@sd['beta.weight'].T+sd['beta.bias']; w = np.exp(z-z.max(-1,keepdims=True)); w/=w.sum(-1,keepdims=True)
            np.testing.assert_allclose(bn,w,rtol=2e-5,atol=2e-6)
            if not fixed:
                z = (h@sd['strength.weight'].T+sd['strength.bias']).ravel() if model.mode=='user' else np.full(len(vn),sd['strength'])
                np.testing.assert_allclose(a,np.clip(1/(1+np.exp(-z)),1e-6,1-1e-6),rtol=2e-5,atol=2e-6)
            return dict(alpha_mean=float(a.mean()), alpha_quantiles=np.quantile(a,[0,.1,.5,.9,1]).tolist(),
                        behavior_weight_mean=float(bn[:,0].mean()), numpy_verified_users=len(vn))

        baseline = J.JointFusion('calibrate', parent['state_dict']).to(device)
        stats = export(baseline, full_norm, backbone+'_fixed', fixed=True)
        if backbone != 'tisas_qwen06':
            old_path = (BASE if backbone=='bge' else TIME if backbone=='tisas' else QWEN)/domain/'signals.npz'
            old_key = 'user' if backbone=='bge' else backbone
            expected = np.load(old_path)[old_key]
            np.testing.assert_allclose(exported[backbone+'_fixed'], expected, atol=3e-6,rtol=2e-5)
            # Keep historical scores bit-identical for the fixed decoder replay.
            exported[backbone+'_fixed'] = expected
        info[backbone+'_fixed'] = dict(parent_epochs=parent['selected_epochs'], export=stats)
        del baseline

        for mode in MODES[1:]:
            arm = backbone+'_'+mode
            def run(state, indices, norm, epochs, phase):
                torch.manual_seed(0); torch.cuda.manual_seed_all(0)
                model = J.JointFusion(mode, state).to(device)
                opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=CFG['lr'],weight_decay=CFG['weight_decay'])
                rng = np.random.default_rng(0)
                normal = J.normalize(ut, *norm)
                def loss(ix):
                    beta, alpha = model(normal[ix])
                    scores = (beta[:,None,:]*et[ix]).sum(-1)
                    q = tree.logs(scores, yt[ix])
                    return J.path_nll(nt[ix], mt[ix], q, alpha, tree.valid[yt[ix]]).mean()
                curve = []
                for epoch in range(epochs+1):
                    total = 0.
                    if epoch:
                        order = rng.permutation(indices)
                        for start in range(0,len(order),32):
                            ix = order[start:start+32]
                            opt.zero_grad(); value = loss(ix)
                            assert torch.isfinite(value)
                            value.backward()
                            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                            opt.step(); total += float(value)*len(ix)
                    heldout = None
                    if phase=='selection' and epoch in CFG['checkpoints']:
                        with torch.no_grad():
                            heldout = sum(float(loss(hi[k:k+32]))*len(hi[k:k+32]) for k in range(0,len(hi),32))/len(hi)
                        curve.append(dict(epoch=epoch, loss=heldout))
                    progress(domain, 'adapt_'+phase, arm=arm, epoch=epoch, epochs=epochs,
                        training_loss=total/len(indices) if epoch else None, internal_heldout_loss=heldout)
                    write(p/(arm+'_'+phase+'_curve.json'), dict(curve=curve, completed_epochs=epoch))
                return model, curve
            model, curve = run(fold_state, fi, fit_nt,20,'selection')
            selected = min(curve,key=lambda x:(x['loss'],x['epoch']))['epoch']
            del model
            model, _ = run(parent['state_dict'],all_ix,tuple(torch.from_numpy(x).to(device) for x in full_norm),selected,'refit')
            info[arm] = dict(selected_epochs=selected, selection_curve=curve, selection_fit_users=len(fi), heldout_users=len(hi),
                refit_users=len(samples), parameters=sum(p.numel() for p in model.parameters()),
                adapted_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                export=export(model,full_norm,arm))
            torch.save(dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                mean=full_norm[0],std=full_norm[1],parent=str(ckpath.relative_to(ROOT)),mode=mode,
                fit_users=[s['user_id'] for s in samples],config=CFG,**info[arm]),p/(arm+'.pt'))
            del model
        del et, ut, e, ve
        gc.collect(); torch.cuda.empty_cache()
    np.savez(p/'signals.npz', **exported)
    np.savez(p/'alphas.npz', **alphas)
    np.savez(p/'betas.npz', **betas)
    write(p/'training.json', dict(fits=info, test_read=False, dev_used_for_epoch_selection=False,
        teacher_cache=read(p/'teacher_checks.json'), fitting_users=len(samples),
        internal_parent_weights_exclude_heldout_labels=True, previous_parent_epoch_selection_reused=True))
    paths = [p/n for n in ('signals.npz','alphas.npz','betas.npz','training.json','teacher.npz','teacher_checks.json')]
    paths += list(p.glob('*.pt'))+list((p/'parents').glob('*/*.pt'))+list((p/'features').glob('*/*/*'))
    write(p/'training_manifest.json',dict(sha256={str(x.relative_to(ROOT)):sha(x) for x in paths}))


class Generation(S.Generation):
    def run(self):
        self.alpha = dict(np.load(self.folder/'alphas.npz'))
        self.stage = 'native_zero_fixed_smoke'; self.status()
        checks = []
        for j in (0,256,511):
            ctx = self.context(j)
            native = self.decode(ctx)
            plain = self.decode(ctx,native_entry=True,record=False)
            q = self.index.conditional_logs(self.signals['bge_fixed'][j])
            zero = self.decode(ctx,q,0.)
            for other in (plain,zero):
                assert native['paths']==other['paths']
                np.testing.assert_array_equal(native['scores'],other['scores'])
            checks.append(dict(user_id=self.samples[j]['user_id'],alpha0_exact=True,native_entry_exact=True))
        write(self.folder/'smoke.json',dict(state='PASSED',checks=checks))
        self.stage = 'matched_generation'; self.status()
        with (self.folder/'predictions.jsonl').open('x') as stream:
            for j,sample in enumerate(self.samples):
                begin = time.monotonic()
                ctx = self.context(j); native = self.decode(ctx)
                ref = self.reference[sample['user_id']]
                assert native['ids']==ref['native']['ids']
                np.testing.assert_allclose(native['scores'],ref['native']['scores'],atol=2e-6,rtol=1e-5)
                row = dict(user_id=sample['user_id'], target=sample['target'], history=sample['history'],
                    native=dict(ids=native['ids'],scores=native['scores'].tolist()))
                for arm in ARMS:
                    a = float(self.alpha[arm][j])
                    q = self.index.conditional_logs(self.signals[arm][j],temperature_scale=20.)
                    late = native['guide'].rescore(native['paths'],q,a)
                    guided = self.decode(ctx,q,a)
                    row[arm+'_late'] = dict(ids=native['ids'],scores=late.tolist())
                    row[arm+'_guided'] = dict(ids=guided['ids'],scores=guided['scores'].tolist())
                    for suffix in ('_late','_guided'):
                        key = arm+suffix
                        if key in ref:
                            assert row[key]['ids']==ref[key]['ids'],(j,key)
                            np.testing.assert_allclose(row[key]['scores'],ref[key]['scores'],atol=2e-6,rtol=1e-5)
                    del guided
                stream.write(json.dumps(row)+'\n');stream.flush();os.fsync(stream.fileno())
                self.users_completed=j+1;self.row_seconds.append(time.monotonic()-begin)
                if j==0 or (j+1)%8==0 or j==511:
                    self.status(arms_per_user=len(ARMS))
                del ctx,native
        assert self.digest()==self.initial_digest
        write(self.folder/'runtime_checks.json',dict(state='PASSED',native_reproduced_users=512,
            four_historical_fixed_arms_reproduced_users=512,gram_model_digest_unchanged=True,
            gram_optimizer_steps=0,all_arms_generated=True))
        self.stage='complete';self.status('COMPLETED')


def summarize():
    results={}
    keys=('native',)+tuple(a+s for a in ARMS for s in ('_late','_guided'))
    lines=['# 专家比例与生成强度联合融合','','状态：COMPLETED。两域各512名历史开发用户，seed0，探索性结果。',
        '', '| 数据域 | 专家组合 | 固定0.3 | 只调全局强度 | 联合全局强度 | 联合用户强度 |',
        '|---|---|---:|---:|---:|---:|']
    for domain in DOMAINS:
        p=OUT/domain
        assert read(p/'status.json')['state']=='COMPLETED'
        assert read(p/'lease_status.json')['state']=='released'
        for rel,digest in read(p/'training_manifest.json')['sha256'].items():
            assert sha(ROOT/rel)==digest,rel
        samples=read(p/'samples.json')
        rows=[json.loads(x) for x in (p/'predictions.jsonl').read_text().splitlines()]
        assert len(rows)==len(samples)==512
        rr={k:[] for k in keys}
        for row,sample in zip(rows,samples):
            assert all(row[k]==sample[k] for k in ('user_id','target','history'))
            for k in keys:
                r=S.OLD.record_rank(row[k],sample)
                ids=np.asarray(row[k]['ids']);scores=np.asarray(row[k]['scores'])
                keep=~np.isin(ids,sample['history']);ids,scores=ids[keep],scores[keep]
                order=ids[np.lexsort((ids,-scores))].tolist()
                assert r==(order.index(sample['target'])+1 if sample['target'] in order else 2**30)
                rr[k].append(r)
        rr={k:np.asarray(v) for k,v in rr.items()}
        np.savez_compressed(p/'ranks.npz',users=np.asarray([s['user_id'] for s in samples]),**rr)
        pairs=[]
        for b in BACKBONES:
            for mode in MODES:
                pairs.append((b+'_'+mode+'_guided','native'))
            pairs += [(b+'_calibrate_guided',b+'_fixed_guided'),(b+'_global_guided',b+'_calibrate_guided'),
                      (b+'_user_guided',b+'_global_guided')]
        for b in BACKBONES[1:]:
            for mode in MODES:
                pairs.append((b+'_'+mode+'_guided','bge_'+mode+'_guided'))
        for mode in MODES:
            for b in ('tisas','qwen06'):
                pairs.append(('tisas_qwen06_'+mode+'_guided',b+'_'+mode+'_guided'))
        result=dict(metrics={k:metrics(v) for k,v in rr.items()},
            comparisons={a+'_vs_'+b:contrast(rr[a],rr[b],repeats=2000,seed=20261015) for a,b in pairs},
            training=read(p/'training.json'),test_read=False)
        results[domain]=result;write(p/'summary.json',result)
        for b in BACKBONES:
            lines.append('| '+domain+' | '+b+' | '+' | '.join(f'{result["metrics"][b+"_"+m+"_guided"]["N10"]:.6f}' for m in MODES)+' |')
        write(p/'verification.json',dict(state='VERIFIED_SAVED_PREDICTIONS',users=512,
            independent_rank_arrays=len(keys),training_hashes_verified=True))
        paths=[p/n for n in ('predictions.jsonl','ranks.npz','summary.json','verification.json','training_manifest.json','runtime_checks.json','smoke.json')]
        write(p/'output_manifest.json',dict(sha256={str(x.relative_to(ROOT)):sha(x) for x in paths}))
    write(OUT/'summary.json',dict(domains=results,test_read=False,head_seed=0))
    lines += ['', '所有预定组均重新生成。主要报告NDCG@10，并保留H@10/H@50、纯后重排结果、配对差值和描述性区间。',
        '', '用户接受标准首先是相对原始GRAM的提升；相对当前双专家的变化另报。训练目标为路径似然代理，不保证NDCG提升。',
        '', '开发池被反复使用、只有一个种子、比较较多，区间未经多重校正，不能当作正式确认。没有读取test。',
        '', '专家评分沿用LIME-Rec衍生实现；概率引导沿用AGRec启发规则。模型替换本身不作为方法创新结论。',
        '', '本轮完成即停止，不自动追加论文实验。']
    REPORT.write_text('\n'.join(lines)+'\n')


def main():
    global LEASE
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('test','prepare','worker','train','cache','smoke','fit','generate','supervise','summarize'))
    parser.add_argument('--domain',choices=DOMAINS);parser.add_argument('--lease-pid',type=int)
    args=parser.parse_args();bind()
    def interrupted(signum,frame):
        raise InterruptedError('Joint fusion interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        if args.mode=='test': print(json.dumps(J.tests()))
        elif args.mode=='prepare': prepare()
        elif args.mode=='worker': return S.worker(args.domain)
        elif args.mode=='supervise': S.supervise()
        elif args.mode=='summarize': summarize()
        else:
            assert args.domain and args.lease_pid
            LEASE=S.LeaseHandle(args.lease_pid);assert LEASE.poll() is None
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            if args.mode=='train':
                for phase in ('cache','fit'):
                    subprocess.run([sys.executable,'-u','-m',MODULE,phase,'--domain',args.domain,
                                    '--lease-pid',str(args.lease_pid)],cwd=ROOT,check=True)
            elif args.mode in ('cache','smoke'): cache(args.domain,limit=3 if args.mode=='smoke' else None)
            elif args.mode=='fit': fit(args.domain)
            else: Generation(args.domain,LEASE).run()
        return 0
    except BaseException as exc:
        if OUT.exists() and args.mode!='worker':
            dest=OUT/args.domain if args.domain else OUT
            write(dest/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),updated_at=stamp()))
            write(dest/'status.json',dict(state='FAILED',error=repr(exc),updated_at=stamp(),test_read=False))
        raise


if __name__=='__main__':
    sys.exit(main())
