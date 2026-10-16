"""Select experts through actual GRAM generation with the original ten inputs.

Freeze the 451-parameter user beta/alpha architecture. Reuse verified heads
and old prediction rows only where the expert inputs and learned model match.
Expert scoring remains LIME-derived and probability guidance AGRec-inspired.
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
from experiment.phase21 import joint_fusion as OLD
from experiment.phase21 import joint_fusion_core as J
from experiment.phase21.candidate_gate_core import fold
from experiment.phase20 import lime_clean_benefit as CLEAN
from experiment.phase20.catalog_scope_core import metrics, contrast

ROOT = S.ROOT
OUT = ROOT/'artifacts/phase21/old10_expert_selection_20261016_v1'
SCREEN = ROOT/'artifacts/phase21/expert_reselection_20261016_v2'
PRIOR = ROOT/'artifacts/phase21/joint_fusion_20261015_v2'
PLAN = ROOT/'plan/第二十一阶段/21.43_固定原十维筛选专家的真实生成对照_20261016.md'
REPORT = ROOT/'report/第二十一阶段/固定原十维的专家组合真实生成对照_20261016.md'
MODULE = 'experiment.phase21.old10_expert_selection'
EXPERTS = ('SASRec', 'TiSASRec', 'ItemCF', 'BGE', 'Qwen4')
BEHAVIOR, CONTENT = EXPERTS[:3], EXPERTS[3:]
PAIRS = {a+'__'+b:(a,b) for a in BEHAVIOR for b in CONTENT}
SINGLES = {a+'__only':(a,a) for a in EXPERTS}
ARMS = dict(PAIRS, **SINGLES)
REUSE = {'SASRec__BGE':'bge', 'SASRec__Qwen4':'qwen4', 'TiSASRec__BGE':'tisas'}
DOMAINS = ('Toys', 'Beauty')
CFG = dict(seed=0, epochs=20, checkpoints=[0,4,8,12,20], batch_size=32, lr=.002,
    weight_decay=.001, score_scale=20., alpha_initial=.3, beam_size=50, length_penalty=1.,
    arms=ARMS, experts=list(EXPERTS), user_dimension=10, hidden_width=32, parameters=451,
    feature_names=[x.replace('_sas','_slot0').replace('_text','_slot1') for x in C.USER_NAMES],
    old10_definitions_unchanged=True, alpha_mode='user', evaluation_users=4608,
    cohorts={'old512':512, 'wide4096':4096}, primary_development_cohort='wide4096',
    loss='original full-catalog CE parent, then frozen-GRAM length-normalized target-path NLL',
    single_expert='duplicate the same expert in both slots; invariant to beta; degenerate CE parent epoch0',
    history_filter=False, history_penalty=False, text_self_exclusion=False,
    expert_training=False, gram_optimizer_steps=0, test_read=False,
    reused_heads=REUSE, reused_prediction_rows='old512 for matching frozen reused heads only; smoke-replayed',
    total_lease_mib=16384, expected_workload_peak_mib=12288, minimum_free_mib=18432,
    allocation_limit_gib=11, max_parallel=2, worker_timeout_seconds=43200,
    resource_wait_seconds=86400, gpu_pool='all GPUs, lowest utilization then most free memory',
    scientific_auto_retry=False, stop_after='expert comparison only; no feature redesign or automatic next run')
read, write, sha, stamp = S.read, S.write, S.sha, S.stamp


def progress(domain, stage, **extra):
    write(OUT/domain/'status.json', dict(state='RUNNING', stage=stage, pid=os.getpid(),
        gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'), updated_at=stamp(), test_read=False, **extra))
    print(json.dumps(dict(domain=domain, stage=stage, **extra)), flush=True)


def user_inputs(experts, history):
    """Same original ten statistics, computed once per frozen expert."""
    e = np.asarray(experts, np.float32)
    v = e.shape[1]
    mask = np.zeros(v, bool)
    mask[np.asarray(history, np.int64)-1] = True
    statistics = []
    for row in e:
        order = np.lexsort((np.arange(v), -row))
        statistics.append((row[order[0]]-row.mean(), row[order[0]]-row[order[1]],
            row.std(ddof=1), mask[order[:5]].mean()))
    result = {}
    for arm, names in ARMS.items():
        a, b = [EXPERTS.index(name) for name in names]
        result[arm] = np.asarray([np.log1p(len(history))/5,
            *[statistics[j][k] for k in range(4) for j in (a,b)], mask.mean()], np.float32)
    return result


def tests():
    rng = np.random.default_rng(2143)
    for history in ([1, 3, 3, 7], [2], []):
        experts = rng.random((5, 73)).astype(np.float32)
        users = user_inputs(experts, history)
        for arm, names in ARMS.items():
            e = experts[[EXPERTS.index(name) for name in names]]
            expected, _ = C.features(e, history, np.ones(74))
            np.testing.assert_array_equal(users[arm], expected)
    torch.manual_seed(0)
    model = J.JointFusion('user', C.Fusion().state_dict())
    assert sum(p.numel() for p in model.parameters()) == 451
    e = torch.from_numpy(rng.random((4, 73)).astype(np.float32))
    for weight in (0., .2, .7, 1.):
        torch.testing.assert_close(weight*e+(1-weight)*e, e, atol=1e-7, rtol=1e-6)
    # Independent tiny catalog check: no final historical-item exclusion.
    row = dict(ids=[1, 2, 3], scores=[3., 2., 1.])
    assert rank(row, 3, beam=3) == 3
    # End-to-end differentiable loss on a tiny trie, with real unequal experts.
    paths = [[0,2,4,1],[0,2,5,1],[0,3,1],[0,3,7,1]]
    tree = J.PathMass(paths,'cpu')
    inputs = torch.randn(3,10)
    experts = torch.rand(3,4,2)
    target = torch.tensor([0,1,3])
    beta, alpha = model(inputs)
    mixed = (beta[:,None,:]*experts).sum(-1)
    logs = tree.logs(mixed,target)
    loss = J.path_nll(torch.full_like(logs,-3.),torch.full_like(logs,-.4),logs,alpha,tree.valid[target]).mean()
    loss.backward()
    assert model.beta.weight.grad.abs().sum() > 0 and model.strength.weight.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    nb, na = numpy_forward(model.state_dict(),inputs.numpy(),np.zeros(10,np.float32),np.ones(10,np.float32))
    np.testing.assert_allclose(nb,beta.detach().numpy(),atol=2e-6,rtol=2e-5)
    np.testing.assert_allclose(na,alpha.detach().numpy(),atol=2e-6,rtol=2e-5)
    return dict(original10_exact_for_all_arms=True, single_beta_invariance=True,
        parameters=451, unfiltered_rank=True, no_target_argument_in_features=True,
        joint_loss_reaches_beta_and_alpha=True, numpy_joint_forward=True)


def rank(value, target, beam=50):
    ids, scores = np.asarray(value['ids']), np.asarray(value['scores'])
    assert len(ids) == len(set(ids.tolist())) == len(scores) == beam
    assert np.isfinite(scores).all()
    order = ids[np.lexsort((ids, -scores))].tolist()
    return order.index(target)+1 if target in order else 2**30


def prepare():
    checks = tests()
    assert read(SCREEN/'status.json')['state'] == read(PRIOR/'status.json')['state'] == 'COMPLETED'
    OUT.mkdir(exist_ok=False)
    write(OUT/'status.json', dict(state='PREPARING', updated_at=stamp(), test_read=False))
    write(OUT/'config.json', CFG); write(OUT/'tests.json', checks)
    shutil.copy2(PLAN, OUT/'protocol.md')
    manifest = {}; domain_inputs = {}
    def use(path, expected=None):
        rel = str(path.relative_to(ROOT))
        digest = manifest.get(rel) or sha(path)
        assert expected is None or digest == expected, rel
        manifest[rel] = digest
        return rel
    previous = read(PRIOR/'manifest.json')['sha256']
    for rel, digest in previous.items():
        if rel.startswith(('experiment/', 'GRAM/src/', '.cache/huggingface/')):
            use(ROOT/rel, digest)
    common = list(manifest)
    for domain in DOMAINS:
        p = OUT/domain; p.mkdir()
        checked = read(SCREEN/'manifest.json')['sha256']
        checked.update(read(SCREEN/domain/'output_manifest.json')['sha256'])
        files = []
        for name in ('catalog.json', 'data_config.json', 'teacher.npz', 'teacher_checks.json'):
            src = PRIOR/domain/name
            files.append(use(src)); shutil.copy2(src, p/name); files.append(use(p/name))
        fit = read(SCREEN/domain/'fit_samples.json')
        assert fit == read(PRIOR/domain/'fit_samples.json')
        assert len(fit) == dict(Toys=2773, Beauty=3310)[domain]
        chosen, refs = [], []
        for part in ('old512', 'wide4096'):
            rows = read(SCREEN/domain/(part+'_samples.json'))
            files.append(use(SCREEN/domain/(part+'_samples.json'), checked[str((SCREEN/domain/(part+'_samples.json')).relative_to(ROOT))]))
            for row in rows:
                chosen.append(dict(row, cohort=part, validation_index=len(chosen)))
            source = SCREEN/domain/(part+'_generation_references.jsonl')
            files.append(use(source))
            refs.extend(json.loads(line) for line in source.read_text().splitlines())
        assert len(chosen) == len(refs) == 4608
        assert len(set(s['user_id'] for s in chosen)) == 4608
        assert {s['user_id'] for s in fit}.isdisjoint(s['user_id'] for s in chosen)
        for sample, reference in zip(chosen, refs):
            assert all(sample[k] == reference[k] for k in ('user_id','target','history'))
            assert sample['history'] == sample['full_history'][-20:]
        for name, value in (('samples.json', chosen), ('fit_samples.json', fit), ('references.json', refs)):
            write(p/name, value); files.append(use(p/name))
        for part in ('fit','old512','wide4096'):
            src = SCREEN/domain/(part+'_metadata.json')
            files.append(use(src, checked[str(src.relative_to(ROOT))]))
        for origin in (S.SOURCE, S.FULL):
            src = origin/domain/'experts.npy'
            files.append(use(src, checked[str(src.relative_to(ROOT))]))
        for origin, variant in ((OLD.QWEN, 'qwen4'), (OLD.TIME, 'tisas')):
            for part in ('train','validation'):
                src = origin/domain/'features'/variant/part/'item.npy'
                files.append(use(src, checked[str(src.relative_to(ROOT))]))
        for name in ('TiSASRec','Qwen4'):
            src = SCREEN/domain/'features/wide4096'/(name+'.npy')
            files.append(use(src, checked[str(src.relative_to(ROOT))]))
        for arm, backbone in REUSE.items():
            source = PRIOR/domain/(backbone+'_user.pt')
            digest = read(PRIOR/domain/'training_manifest.json')['sha256'][str(source.relative_to(ROOT))]
            files.append(use(source, digest))
            folder, _ = OLD.origin(domain, backbone)
            for part in ('train','validation'):
                files.append(use(folder/part/'user.npy', previous[str((folder/part/'user.npy').relative_to(ROOT))]))
        for name in ('signals.npz','alphas.npz','betas.npz','predictions.jsonl','training.json'):
            files.append(use(PRIOR/domain/name))
        config = read(p/'data_config.json')
        files.append(use(ROOT/config['parent_checkpoint'], config['parent_sha256']))
        assert sha(p/'teacher.npz') == read(p/'teacher_checks.json')['sha256']
        domain_inputs[domain] = common+files
        write(p/'status.json', dict(state='PREPARED', users_total=4608, updated_at=stamp()))
    for path in (Path(__file__), PLAN, ROOT/'experiment/phase21/run_old10_expert_selection.sh', OUT/'config.json'):
        rel = use(path)
        for domain in DOMAINS: domain_inputs[domain].append(rel)
    for rel in manifest:
        source = ROOT/rel
        if source.suffix in ('.py','.sh','.md'):
            dest = OUT/'sources'/rel
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, dest)
    write(OUT/'manifest.json', dict(sha256=manifest, domain_inputs=domain_inputs))
    write(OUT/'status.json', dict(state='PREPARED', updated_at=stamp(), test_read=False))
    REPORT.write_text('# 固定原十维的专家组合真实生成对照\n\n状态：PREPARED，尚无新生成效果结论。\n')
    print(json.dumps(dict(state='PREPARED', files=len(manifest), tests=checks)), flush=True)


def raw_part(domain, part):
    meta = read(SCREEN/domain/(part+'_metadata.json'))
    indices = meta['source_indices']
    base = np.load((S.SOURCE if part == 'fit' else S.FULL)/domain/'experts.npy', mmap_mode='r')
    cache = {}
    for name, root, variant in (('TiSASRec', OLD.TIME, 'tisas'), ('Qwen4', OLD.QWEN, 'qwen4')):
        if part == 'wide4096':
            cache[name] = np.load(SCREEN/domain/'features/wide4096'/(name+'.npy'), mmap_mode='r')
        else:
            cache[name] = np.load(root/domain/'features'/variant/('train' if part == 'fit' else 'validation')/'item.npy', mmap_mode='r')
    for j, index in enumerate(indices):
        time_row = cache['TiSASRec'][j] if part == 'wide4096' else cache['TiSASRec'][j,:,0]
        text_row = cache['Qwen4'][j] if part == 'wide4096' else cache['Qwen4'][j,:,1]
        yield np.stack((base[index,0], time_row, base[index,1], base[index,2], text_row))


def cache(domain):
    p = OUT/domain; v = len(read(p/'catalog.json')['items'])-1
    for part, names, parts in (('train','fit_samples.json',('fit',)),
                                ('validation','samples.json',('old512','wide4096'))):
        samples = read(p/names); n = len(samples)
        dest = p/'features'/part; dest.mkdir(parents=True)
        experts = np.lib.format.open_memmap(dest/'experts.npy', mode='w+', dtype='float32', shape=(n,v,5))
        users = {arm:np.empty((n,10),np.float32) for arm in ARMS}
        j = 0
        for source_part in parts:
            for row in raw_part(domain, source_part):
                experts[j] = row.T
                extracted = user_inputs(row, samples[j]['full_history'])
                for arm in ARMS: users[arm][j] = extracted[arm]
                j += 1
                if j == 1 or j % 256 == 0 or j == n:
                    progress(domain, 'original10_features', part=part, completed=j, total=n)
        assert j == n
        experts.flush(); del experts
        for arm, array in users.items(): np.save(dest/(arm+'.npy'), array)
        for arm, backbone in REUSE.items():
            source, _ = OLD.origin(domain, backbone)
            old_user = np.load(source/part/'user.npy')
            np.testing.assert_array_equal(users[arm][:len(old_user)], old_user)
        write(dest/'metadata.json', dict(users_order=[s['user_id'] for s in samples],
            names=CFG['feature_names'], dimension=10, expert_channels=EXPERTS,
            original_feature_values_exact=True, target_used_in_features=False))
    paths = list((p/'features').glob('*/*'))
    write(p/'feature_manifest.json', dict(sha256={str(x.relative_to(ROOT)):sha(x) for x in paths}))


def ce_parent(domain, arm, user, expert, targets, fi, hi, fn, an, device):
    p = OUT/domain/'parents'/arm; p.mkdir(parents=True)
    def run(indices, norm, epochs, selection=False):
        torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        model = C.Fusion().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.001)
        normal = J.normalize(user, *[torch.from_numpy(x).to(device) for x in norm])
        rng = np.random.default_rng(0)
        curve = []
        def loss(ix):
            beta = model(normal[ix], None, 'user')
            return torch.nn.functional.cross_entropy(20*(beta*expert[ix]).sum(-1), targets[ix])
        for epoch in range(epochs+1):
            if epoch:
                order = rng.permutation(indices)
                for start in range(0,len(order),32):
                    opt.zero_grad(); value = loss(order[start:start+32])
                    assert torch.isfinite(value); value.backward(); opt.step()
            if selection and epoch in CFG['checkpoints']:
                with torch.no_grad():
                    val = sum(float(loss(hi[k:k+32]))*len(hi[k:k+32]) for k in range(0,len(hi),32))/len(hi)
                curve.append(dict(epoch=epoch, loss=val))
            progress(domain, 'CE_parent', arm=arm, selection=selection, epoch=epoch, epochs=epochs)
        return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}, curve
    if arm in SINGLES:
        selected = 0; curve = []; fold_state, _ = run(fi, fn, 0)
    else:
        _, curve = run(fi, fn, 20, True)
        selected = min(curve, key=lambda x:(x['loss'],x['epoch']))['epoch']
        fold_state, _ = run(fi, fn, selected)
    full_state, _ = run(np.arange(len(user)), an, selected)
    for name, state, norm, indices in (('fold',fold_state,fn,fi),('full',full_state,an,np.arange(len(user)))):
        torch.save(dict(state_dict=state, mean=norm[0], std=norm[1], selected_epochs=selected,
            fit_indices=indices.tolist(), selection_curve=curve), p/(name+'.pt'))
    return fold_state, full_state, selected, curve


def adapt(domain, arm, parent, user, expert, target, tree, native, legal, fi, hi, norm, epochs, selection, device):
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    model = J.JointFusion('user', parent).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.001)
    normal = J.normalize(user, *[torch.from_numpy(x).to(device) for x in norm])
    rng = np.random.default_rng(0)
    curve = []
    def loss(ix):
        beta, alpha = model(normal[ix])
        scores = (beta[:,None,:]*expert[ix]).sum(-1)
        return J.path_nll(native[ix], legal[ix], tree.logs(scores,target[ix]),alpha,tree.valid[target[ix]]).mean()
    for epoch in range(epochs+1):
        if epoch:
            order = rng.permutation(fi)
            for start in range(0,len(order),32):
                opt.zero_grad(); value = loss(order[start:start+32]); assert torch.isfinite(value)
                value.backward()
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                opt.step()
        if selection and epoch in CFG['checkpoints']:
            with torch.no_grad():
                val = sum(float(loss(hi[k:k+32]))*len(hi[k:k+32]) for k in range(0,len(hi),32))/len(hi)
            curve.append(dict(epoch=epoch,loss=val))
        progress(domain, 'joint_NLL', arm=arm, selection=selection, epoch=epoch, epochs=epochs)
    return model, curve


def numpy_forward(state, user, mean, std):
    s = {k:v.detach().cpu().numpy() for k,v in state.items()}
    u = np.clip((user-mean[:10])/std[:10],-10,10)
    h = np.maximum(u@s['input.weight'].T+s['input.bias'],0)
    z = h@s['beta.weight'].T+s['beta.bias']
    beta = np.exp(z-z.max(1,keepdims=True)); beta /= beta.sum(1,keepdims=True)
    z = (h@s['strength.weight'].T+s['strength.bias']).ravel()
    alpha = np.clip(1/(1+np.exp(-z)),1e-6,1-1e-6)
    return beta, alpha


def fit(domain):
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(11*2**30/torch.cuda.get_device_properties(0).total_memory)
    device = torch.device('cuda:0'); p = OUT/domain
    samples, dev = read(p/'fit_samples.json'), read(p/'samples.json')
    fi = np.asarray([j for j,s in enumerate(samples) if fold(s['user_id'],domain) != 0])
    hi = np.asarray([j for j,s in enumerate(samples) if fold(s['user_id'],domain) == 0])
    all_ix = np.arange(len(samples))
    source = np.load(p/'features/train/experts.npy',mmap_mode='r')
    validation = np.load(p/'features/validation/experts.npy',mmap_mode='r')
    teacher = np.load(p/'teacher.npz')
    np.testing.assert_array_equal(teacher['users'],[s['user_id'] for s in samples])
    np.testing.assert_array_equal(teacher['targets'],[s['target']-1 for s in samples])
    yt = torch.from_numpy(teacher['targets']).to(device)
    native, legal = [torch.from_numpy(teacher[key]).to(device) for key in ('native','legal')]
    tree = J.PathMass(read(p/'catalog.json')['decoder_paths'],device)
    (p/'models').mkdir(); (p/'signals').mkdir()
    alphas, betas, info = {}, {}, {}
    for arm, names in ARMS.items():
        columns = [EXPERTS.index(name) for name in names]
        u = np.load(p/'features/train'/(arm+'.npy'))
        v = np.load(p/'features/validation'/(arm+'.npy'))
        fn, an = OLD.user_norm(u,fi), OLD.user_norm(u,all_ix)
        ut = torch.from_numpy(u).to(device)
        if arm in REUSE:
            source_path = PRIOR/domain/(REUSE[arm]+'_user.pt')
            ck = torch.load(source_path,map_location='cpu')
            assert ck['fit_users'] == [s['user_id'] for s in samples]
            np.testing.assert_array_equal(an[0],ck['mean'][:10]); np.testing.assert_array_equal(an[1],ck['std'][:10])
            # Initialize using a correctly shaped parent, then load all trained values.
            model = J.JointFusion('user',C.Fusion().state_dict()).to(device)
            model.load_state_dict(ck['state_dict'])
            shutil.copy2(source_path,p/'models'/(arm+'.pt'))
            info[arm] = dict(reused=True,source=str(source_path.relative_to(ROOT)),selected_epochs=ck['selected_epochs'])
        else:
            e = np.ascontiguousarray(np.take(source,columns,axis=-1))
            et = torch.from_numpy(e).to(device)
            fold_state, full_state, ce_epoch, ce_curve = ce_parent(domain,arm,ut,et,yt,fi,hi,fn,an,device)
            model, curve = adapt(domain,arm,fold_state,ut,et,yt,tree,native,legal,fi,hi,fn,20,True,device)
            selected = min(curve,key=lambda x:(x['loss'],x['epoch']))['epoch']
            del model
            model, _ = adapt(domain,arm,full_state,ut,et,yt,tree,native,legal,all_ix,hi,an,selected,False,device)
            ck = dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},mean=an[0],std=an[1],
                mode='user',fit_users=[s['user_id'] for s in samples],config=CFG,selected_epochs=selected,
                parent_epochs=ce_epoch,parent_curve=ce_curve,selection_curve=curve,
                internal_fit_users=[samples[j]['user_id'] for j in fi],internal_heldout_users=[samples[j]['user_id'] for j in hi])
            torch.save(ck,p/'models'/(arm+'.pt'))
            info[arm] = dict(reused=False,parent_epochs=ce_epoch,selected_epochs=selected,
                parent_curve=ce_curve,selection_curve=curve)
            del et,e
        model.eval()
        with torch.no_grad():
            beta, alpha = model(J.normalize(torch.from_numpy(v).to(device),*[torch.from_numpy(x).to(device) for x in an]))
            bn, aa = beta.cpu().numpy(), alpha.cpu().double().numpy()
        nb, na = numpy_forward(model.state_dict(),v,*an)
        np.testing.assert_allclose(bn,nb,atol=2e-6,rtol=2e-5)
        np.testing.assert_allclose(aa,na,atol=2e-6,rtol=2e-5)
        signals = np.lib.format.open_memmap(p/'signals'/(arm+'.npy'),mode='w+',dtype='float32',shape=validation.shape[:2])
        for start in range(0,len(dev),32):
            stop = min(start+32,len(dev)); ev = np.take(validation[start:stop],columns,axis=-1)
            value = (bn[start:stop,None,:]*ev).sum(-1)
            np.testing.assert_allclose(value,(nb[start:stop,None,:]*ev).sum(-1),atol=3e-6,rtol=2e-5)
            if arm in SINGLES:
                np.testing.assert_allclose(value,ev[:,:,0],atol=2e-7,rtol=2e-6)
                value = ev[:,:,0]  # Exact duplicate-slot reduction, avoiding roundoff.
            signals[start:stop] = value
        if arm in REUSE:
            key = REUSE[arm]+'_user'
            old_scores = np.load(PRIOR/domain/'signals.npz')[key]
            old_alpha = np.load(PRIOR/domain/'alphas.npz')[key]
            old_beta = np.load(PRIOR/domain/'betas.npz')[key]
            np.testing.assert_allclose(signals[:512],old_scores,atol=3e-6,rtol=2e-5)
            np.testing.assert_allclose(aa[:512],old_alpha,atol=2e-6,rtol=2e-5)
            np.testing.assert_allclose(bn[:512],old_beta,atol=2e-6,rtol=2e-5)
            signals[:512], aa[:512], bn[:512] = old_scores, old_alpha, old_beta
        signals.flush(); del signals
        alphas[arm], betas[arm] = aa,bn
        info[arm].update(alpha_mean=float(aa.mean()),slot0_weight_mean=float(bn[:,0].mean()),
            selection_fit_users=len(fi),selection_heldout_users=len(hi),refit_users=len(samples),
            export_users=len(dev),parameters=451,numpy_forward_verified=True)
        write(p/'training_progress.json',dict(fits=info,test_read=False))
        del model,ut; gc.collect(); torch.cuda.empty_cache()
    np.savez(p/'alphas.npz',**alphas); np.savez(p/'betas.npz',**betas)
    write(p/'training.json',dict(fits=info,fit_dev_disjoint=True,test_read=False,dev_used_for_epoch_selection=False))
    paths = list((p/'models').glob('*.pt'))+list((p/'signals').glob('*.npy'))
    paths += list((p/'parents').glob('*/*.pt'))+[p/'alphas.npz',p/'betas.npz',p/'training.json']
    write(p/'training_manifest.json',dict(sha256={str(x.relative_to(ROOT)):sha(x) for x in paths}))


class Generation(CLEAN.Worker):
    def __init__(self,domain,lease_pid):
        self.lease = S.LeaseHandle(lease_pid); self.row_seconds = []
        CLEAN.OUT = OUT
        super().__init__(domain)
        self.reference = read(self.folder/'references.json')
        self.truth = self.samples
        from experiment.phase18.core.diff_data import DiffDataset
        self.samples = [dict(s,target=1) for s in self.truth]
        self.data = DiffDataset(self.samples,self.catalog)
        self.alpha = dict(np.load(self.folder/'alphas.npz'))
        self.signal = {a:np.load(self.folder/'signals'/(a+'.npy'),mmap_mode='r') for a in ARMS}
        assert self.initial_digest == read(self.folder/'teacher_checks.json')['model_digest']

    def context(self,j,override_index=None):
        return super().context(j,override_index=j)

    def status(self,state='RUNNING',**extra):
        assert self.lease.poll() is None,'GPU lease exited'
        done = getattr(self,'users_completed',0)
        rate = float(np.median(self.row_seconds[-32:])) if self.row_seconds else None
        write(OUT/self.domain/'status.json',dict(state=state,stage=self.stage,pid=os.getpid(),
            gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),updated_at=stamp(),
            users_completed=done,users_total=4608,completed_decodes=self.completed,
            elapsed_seconds=time.monotonic()-self.started,median_seconds_per_user=rate,
            remaining_generation_seconds_estimate=rate*(4608-done) if rate else None,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,test_read=False,**extra))

    def guided(self,ctx,j,arm):
        q = self.index.conditional_logs(self.signal[arm][j],temperature_scale=20.)
        result = self.decode(ctx,q,float(self.alpha[arm][j]))
        return dict(ids=result['ids'],scores=result['scores'].tolist())

    @staticmethod
    def equal(actual,expected):
        assert set(actual['ids']) == set(expected['ids'])
        by_id = dict(zip(expected['ids'],expected['scores']))
        np.testing.assert_allclose(actual['scores'],[by_id[i] for i in actual['ids']],atol=3e-6,rtol=2e-5)

    def run(self):
        old_rows = [json.loads(line) for line in (PRIOR/self.domain/'predictions.jsonl').read_text().splitlines()]
        self.stage = 'native_zero_reused_head_smoke'; self.status()
        checks = []
        for j in (0,256,511,512,2560,4607):
            ctx = self.context(j); native = self.decode(ctx)
            self.equal(dict(ids=native['ids'],scores=native['scores'].tolist()),self.reference[j]['native'])
            q = self.index.conditional_logs(self.signal['SASRec__BGE'][j])
            zero = self.decode(ctx,q,0.)
            assert native['paths'] == zero['paths']; np.testing.assert_array_equal(native['scores'],zero['scores'])
            plain = self.decode(ctx,native_entry=True,record=False)
            assert native['paths'] == plain['paths']; np.testing.assert_array_equal(native['scores'],plain['scores'])
            if j < 512:
                for arm,backbone in REUSE.items(): self.equal(self.guided(ctx,j,arm),old_rows[j][backbone+'_user_guided'])
            self.samples[j]['target'] = 2
            changed = self.context(j)
            torch.testing.assert_close(ctx['ids'],changed['ids'],atol=0,rtol=0)
            torch.testing.assert_close(ctx['mask'],changed['mask'],atol=0,rtol=0)
            self.samples[j]['target'] = 1
            checks.append(dict(row=j,baseline_matches=True,zero_exact=True,native_entry_exact=True,
                target_placeholder_invariant=True,reused_heads_replayed=j<512))
        write(self.folder/'smoke.json',dict(state='PASSED',checks=checks))
        self.stage = 'expert_combinations_generation'; self.status()
        with (self.folder/'predictions.jsonl').open('x') as stream:
            for j,sample in enumerate(self.truth):
                begin = time.monotonic(); ctx = self.context(j)
                row = dict(user_id=sample['user_id'],target=sample['target'],history=sample['history'],
                    cohort=sample['cohort'],native=self.reference[j]['native'],
                    legacy_three_guided_reference=self.reference[j]['guided'],reused_arms=[])
                for arm in ARMS:
                    if j < 512 and arm in REUSE:
                        row[arm] = old_rows[j][REUSE[arm]+'_user_guided']; row['reused_arms'].append(arm)
                    else:
                        row[arm] = self.guided(ctx,j,arm)
                stream.write(json.dumps(row)+'\n'); stream.flush()
                self.users_completed = j+1; self.row_seconds.append(time.monotonic()-begin)
                if j == 0 or (j+1)%8 == 0 or j+1 == len(self.truth):
                    os.fsync(stream.fileno()); self.status(arms_per_user=8 if j<512 else 11,cohort=sample['cohort'])
        assert self.digest() == self.initial_digest
        write(self.folder/'runtime_checks.json',dict(state='PASSED',gram_model_digest_unchanged=True,
            gram_optimizer_steps=0,all_prescribed_users_and_arms=True,
            baseline='saved same-user native with six deterministic fresh checks',
            reused_old512_arms=REUSE,new_generation_for_every_wide4096_arm=True))
        self.stage='complete'; self.status('COMPLETED')


def stop(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM)
        try: process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL); process.wait()


def worker(domain):
    p = OUT/domain; lease = child = None
    def interrupted(signum,frame): raise InterruptedError('Expert selection interrupted or timed out')
    signal.signal(signal.SIGTERM,interrupted); signal.signal(signal.SIGALRM,interrupted)
    signal.alarm(CFG['worker_timeout_seconds'])
    try:
        progress(domain,'verify_inputs')
        manifest = read(OUT/'manifest.json')
        for rel in manifest['domain_inputs'][domain]:
            assert sha(ROOT/rel) == manifest['sha256'][rel],rel
        gpu = os.environ['CUDA_VISIBLE_DEVICES']
        free = next(g['free_mib'] for g in S.OLD.gpu_inventory() if g['uuid'] == gpu)
        if free < CFG['minimum_free_mib']:
            write(p/'status.json',dict(state='ADMISSION_DEFERRED',free_mib=free,updated_at=stamp()))
            return 75
        with (p/'lease.log').open('x') as log:
            lease_path = p/'lease_status.json'
            lease = subprocess.Popen([sys.executable,str(ROOT/'experiment/gpu_memory_lease.py'),'--gpu','0',
                '--total-lease-mib','16384','--expected-workload-peak-mib','12288','--status-path',str(lease_path)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(p/'lease_process.json',dict(pid=lease.pid,gpu_uuid=gpu,started_at=stamp()))
            start = time.monotonic()
            while not lease_path.exists() or read(lease_path).get('state') != 'holding':
                assert lease.poll() is None,'Lease failed'; assert time.monotonic()-start < 180,'Lease timeout'
                time.sleep(1)
            for phase in ('cache','fit','generate'):
                assert lease.poll() is None,'Lease exited'
                child = subprocess.Popen([sys.executable,'-u','-m',MODULE,phase,'--domain',domain,
                    '--lease-pid',str(lease.pid)],cwd=ROOT,start_new_session=True)
                assert child.wait() == 0,('Phase failed',phase)
        return 0
    except BaseException as exc:
        previous = read(p/'status.json')
        write(p/'status.json',dict(state='FAILED',error=repr(exc),traceback=traceback.format_exc(),
            previous=previous,updated_at=stamp(),test_read=False))
        raise
    finally:
        signal.alarm(0); stop(child); stop(lease)


def summarize():
    result = {}
    lines = ['# 固定原十维的专家组合真实生成对照','','## Material Passport','',
        '- Origin Skill: academic-research-suite / experiment-agent','- Origin Mode: validate',
        '- Origin Date: 2026-10-16','- Verification Status: ANALYZED；保存的预测逐条重排，无独立测试',
        '- Version Label: old10_expert_selection_v1','',
        '冻结原10维、32隐藏单元、451参数联合用户beta/alpha，比较专家实际生成增量。最终不过滤历史。',
        'old512是反复使用的旧开发子集；wide4096是扩大开发子集，仍非独立test。单专家通过重复同一输入槽实现，CE阶段退化为0步，NLL采用同预算。',
        'ItemCF在原两个专家槽内作为行为候选；历史三专家不同融合器与历史惩罚，仅作非匹配参考。','']
    for domain in DOMAINS:
        p=OUT/domain; samples=read(p/'samples.json')
        keys=['native',*ARMS,'legacy_three_guided_reference']
        ranks={key:[] for key in keys}; count=0
        with (p/'predictions.jsonl').open() as stream:
            for j,line in enumerate(stream):
                row=json.loads(line); sample=samples[j]
                assert all(row[k]==sample[k] for k in ('user_id','target','history','cohort'))
                assert row['reused_arms'] == (list(REUSE) if j<512 else [])
                for key in keys:ranks[key].append(rank(row[key],sample['target']))
                count+=1
        assert count==len(samples)==4608
        ranks={k:np.asarray(v,np.int64) for k,v in ranks.items()}
        np.savez_compressed(p/'ranks.npz',users=np.asarray([s['user_id'] for s in samples]),**ranks)
        result[domain]={}
        for cohort,ix in (('old512',np.arange(512)),('wide4096',np.arange(512,4608))):
            measured={key:metrics(value[ix]) for key,value in ranks.items()}
            pairs=[(arm,'native') for arm in ARMS]
            pairs += [(arm,expert+'__only') for arm,names in PAIRS.items() for expert in names]
            comparisons={a+'_vs_'+b:contrast(ranks[a][ix],ranks[b][ix],repeats=2000,seed=20261016) for a,b in pairs}
            result[domain][cohort]=dict(users=len(ix),metrics=measured,comparisons=comparisons)
            lines += [f'## {domain} / {cohort}','','| 方法 | N10 | 对GRAM相对变化 | H10 |','|---|---:|---:|---:|']
            for key in keys:
                value=measured[key]; relative=100*(value['N10']/measured['native']['N10']-1)
                lines.append(f'| {key} | {value["N10"]:.6f} | {relative:+.2f}% | {value["H10"]:.6f} |')
        write(p/'summary.json',result[domain])
        files=[p/n for n in ('predictions.jsonl','ranks.npz','summary.json','smoke.json','runtime_checks.json')]
        write(p/'output_manifest.json',dict(sha256={str(x.relative_to(ROOT)):sha(x) for x in files}))
    write(OUT/'summary.json',dict(domains=result,test_read=False,automatic_final_selection=False,
        statistical_review=dict(coverage='11/11',
            simpson='Separate domains and old512/wide4096; no pooled winner.',
            ecological='Paired per-user gains/losses do not imply universal benefit.',
            berkson='Fixed role-restricted validation sample; not raw population.',
            collider='No score/outcome-based user selection.',base_rate='All prescribed users retained.',
            regression_to_mean='Repeated small development outcomes do not guarantee future improvement.',
            survivorship='All11 arms retained; incomplete runs are not summarized as success.',
            look_elsewhere='All23 paired comparisons/cohort/domain saved; unadjusted descriptive intervals.',
            forking_paths='Exploratory expert selection, fixed fuser; final test and seed confirmation deferred.',
            causation='Claims limited to expert intervention under this exact fuser and decoder.',
            reverse_causality='No target inputs at inference; target only in training labels and metrics.')))
    lines += ['', '所有对照均保留同一GRAM、目录、生成规则和评分尺度。配对区间描述当前固定训练模型下的用户抽样变化，未经多重比较校正，不包含训练种子和模型选择不确定性。',
        '单专家、配对相对GRAM和相对其单专家对照的全部92项比较保存在summary.json；统计解释覆盖11/11。',
        '本轮只筛专家；没有比较六维/七维，也没有自动确定最终方法、修改融合器或启动下一轮。']
    REPORT.write_text('\n'.join(lines)+'\n')


def supervise():
    assert read(OUT/'status.json')['state']=='PREPARED'
    pending=list(DOMAINS); active={}; finished={}; attempts={d:0 for d in DOMAINS}
    start=time.monotonic()
    try:
        while pending or active:
            for domain,job in list(active.items()):
                code=job['process'].poll()
                if code is None and time.monotonic()-job['started']>CFG['worker_timeout_seconds']+120:
                    stop(job['process']);code=job['process'].returncode
                if code is not None:
                    job['log'].close(); status=read(OUT/domain/'status.json')
                    if code==75 and status['state']=='ADMISSION_DEFERRED':pending.append(domain)
                    else:finished[domain]=dict(exit_code=code,state=status['state'],
                        lease_released=read(OUT/domain/'lease_status.json').get('state')=='released' if (OUT/domain/'lease_status.json').exists() else True)
                    del active[domain]
            inventory=S.OLD.gpu_inventory(); occupied={job['gpu']['uuid'] for job in active.values()}
            available=sorted([g for g in inventory if g['uuid'] not in occupied and g['free_mib']>=CFG['minimum_free_mib']],
                key=lambda g:(g['utilization'],-g['free_mib'],g['index']))
            for domain in list(pending):
                if not available or len(active)>=2:break
                gpu=available.pop(0);pending.remove(domain);attempts[domain]+=1
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
                log=(OUT/domain/f'worker_attempt{attempts[domain]}.log').open('x')
                write(OUT/domain/'status.json',dict(state='STARTING',gpu_index=gpu['index'],gpu_uuid=gpu['uuid'],updated_at=stamp()))
                process=subprocess.Popen([sys.executable,'-u','-m',MODULE,'worker','--domain',domain],cwd=ROOT,
                    env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[domain]=dict(process=process,log=log,gpu=gpu,started=time.monotonic())
            if pending and time.monotonic()-start>CFG['resource_wait_seconds']:
                for domain in pending:finished[domain]=dict(exit_code=124,state='RESOURCE_WAIT_LIMIT')
                pending=[]
            write(OUT/'status.json',dict(state='RUNNING' if active else 'WAITING_FOR_RESOURCES' if pending else 'FINALIZING',
                pid=os.getpid(),updated_at=stamp(),elapsed_seconds=time.monotonic()-start,pending=pending,finished=finished,
                active={d:dict(pid=j['process'].pid,gpu_index=j['gpu']['index'],gpu_uuid=j['gpu']['uuid'],
                    worker=read(OUT/d/'status.json')) for d,j in active.items()},gpu_inventory=inventory,test_read=False))
            if pending or active:time.sleep(10)
        complete=len(finished)==2 and all(v['exit_code']==0 and v['state']=='COMPLETED' and v.get('lease_released') for v in finished.values())
        if complete:summarize()
        write(OUT/'status.json',dict(state='COMPLETED' if complete else 'INCOMPLETE',finished=finished,
            updated_at=stamp(),elapsed_seconds=time.monotonic()-start,test_read=False,report=str(REPORT.relative_to(ROOT))))
    finally:
        for job in active.values():stop(job['process']);job['log'].close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('test','prepare','supervise','worker','cache','fit','generate'))
    parser.add_argument('--domain',choices=DOMAINS);parser.add_argument('--lease-pid',type=int)
    args=parser.parse_args()
    def interrupted(signum,frame):raise InterruptedError('Expert selection interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    try:
        if args.mode=='test':print(json.dumps(tests()),flush=True)
        elif args.mode=='prepare':prepare()
        elif args.mode=='supervise':supervise()
        elif args.mode=='worker':return worker(args.domain)
        elif args.mode=='cache':cache(args.domain)
        elif args.mode=='fit':fit(args.domain)
        else:Generation(args.domain,args.lease_pid).run()
        return 0
    except BaseException as exc:
        if OUT.exists():
            destination=OUT/args.domain if args.domain else OUT
            write(destination/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),updated_at=stamp()))
            write(destination/'status.json',dict(state='FAILED',error=repr(exc),updated_at=stamp(),test_read=False))
        raise


if __name__=='__main__':sys.exit(main())
