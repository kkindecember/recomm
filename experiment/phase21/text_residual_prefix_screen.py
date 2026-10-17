"""CPU continuation: user-balanced prefix augmentation x adapter capacity."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
import torch

from experiment.phase21 import text_residual_screen as R
from experiment.phase21 import text_residual_prefix_core as D
from experiment.phase20.catalog_scope_core import metrics, contrast, utilities

ROOT, SOURCE = R.ROOT, R.OUT
OUT = ROOT/'artifacts/phase21/text_residual_prefix_1024_20261017_v1'
PLAN = ROOT/'plan/第二十一阶段/21.54_文本残差前缀扩充与小容量对照_20261017.md'
REPORT = ROOT/'report/第二十一阶段/文本残差前缀扩充与小容量对照_20261017.md'
MODULE = 'experiment.phase21.text_residual_prefix_screen'
DOMAINS, PROPOSALS = R.DOMAINS, R.PROPOSALS
SPECS = dict(terminal_rank32=dict(rank=32, augmented=False, objective='residual_hard'),
             prefix_rank32=dict(rank=32, augmented=True, objective='residual_hard'),
             terminal_rank4=dict(rank=4, augmented=False, objective='residual_hard'),
             residual_hard=dict(rank=4, augmented=True, objective='residual_hard'),
             text_hard=dict(rank=4, augmented=True, objective='text_hard'),
             residual_random=dict(rank=4, augmented=True, objective='residual_random'))
METHODS = ('off', 'bge_cal', 'bge_fixed1') + tuple(
    arm+'_'+suffix for arm in (*SPECS, 'residual_hard_shuffled') for suffix in ('cal', 'fixed1'))
CFG = dict(R.CFG, train_arms=list(SPECS), methods=list(METHODS), specifications=SPECS,
           epochs=20, checkpoints=[0,1,2,3,4,5,10,20], max_prefixes_per_user=8,
           selection='stop terminal target-free SASRec50 fixed1 N10, H10, lower CE, earlier epoch; positive checkpoints only',
           data_schedule='one example per user per epoch; identical optimizer steps; terminal first then fixed prefix cycle',
           normalization='original terminal training folds2/3/4, same for every arm',
           rank=4, worker_timeout_seconds=3600, automatic_next_experiment=False)
read, write, sha, stamp = R.read, R.write, R.sha, R.stamp


def bind():
    # Process-local settings only; never edit modules used by running experiments.
    R.OUT, R.CFG, R.PROPOSALS, R.METHODS = OUT, CFG, PROPOSALS, METHODS


def progress(domain, stage, **extra):
    write(OUT/domain/'status.json', dict(state='RUNNING', domain=domain, stage=stage,
        pid=os.getpid(), updated_at=stamp(), gpu_used=False, test_read=False, **extra))


def prepare():
    checks = tests()
    OUT.mkdir(exist_ok=False)
    write(OUT/'status.json',dict(state='PREPARING',updated_at=stamp(),test_read=False,gpu_used=False))
    write(OUT/'config.json', CFG)
    write(OUT/'tests.json', checks)
    shutil.copy2(PLAN, OUT/'protocol.md')
    expected = read(SOURCE/'manifest.json')['sha256']
    manifest, inputs = {}, {}
    def use(path):
        rel = str(path.relative_to(ROOT))
        digest = manifest.get(rel) or sha(path)
        assert rel not in expected or expected[rel] == digest, rel
        manifest[rel] = digest
        return rel
    common = []
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name and name.endswith('.py'):
            path = Path(name).resolve()
            if ROOT in path.parents:
                common.append(use(path))
    for path in (PLAN, OUT/'config.json'):
        common.append(use(path))
    # The CPU teacher is loaded later; freeze its implementation now.
    for path in (ROOT/'third_party/LIME-Rec/lime_rec').rglob('*.py'):
        common.append(use(path))
    for domain in DOMAINS:
        folder = OUT/domain
        folder.mkdir()
        files = []
        for name in ('samples.json','references.json','fit_samples.json','folds.npy','fit_inputs.npy',
                     'evaluation_inputs.npy','item_vectors.npz','proposals.npz',
                     'calibration_proposals.npz','data_config.json'):
            source = SOURCE/domain/name
            files.append(use(source))
            (folder/name).symlink_to(source)
            files.append(use(folder/name))
        for name in ('training_pools.npz',):
            source = SOURCE/domain/name
            files.append(use(source))
            (folder/('terminal_'+name)).symlink_to(source)
            files.append(use(folder/('terminal_'+name)))
        dc = read(folder/'data_config.json')
        files.extend(use(ROOT/dc[key]) for key in ('sasrec','embedding'))
        ep = R.P.FIT/domain/'features/train/experts.npy'
        files.append(use(ep))
        samples, fit = read(folder/'samples.json'), read(folder/'fit_samples.json')
        assert len(samples) == 1024 and {s['user_id'] for s in samples}.isdisjoint(s['user_id'] for s in fit)
        inputs[domain] = sorted(set(common+files))
        write(folder/'status.json', dict(state='PREPARED', updated_at=stamp(), test_read=False))
    for rel in manifest:
        path = ROOT/rel
        if path.suffix in ('.py','.md'):
            dest = OUT/'sources'/rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    write(OUT/'manifest.json', dict(sha256=manifest, domain_inputs=inputs))
    write(OUT/'environment.json', dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
        cpu_threads=2, gpu_used=False))
    write(OUT/'status.json', dict(state='PREPARED', users_total=2048, updated_at=stamp(), test_read=False))
    REPORT.write_text('# 文本残差前缀扩充与小容量对照\n\n状态：PREPARED。\n')
    print(json.dumps(dict(state='PREPARED', hashes=len(manifest), tests=checks)), flush=True)


def make_prefix_data(domain, folder):
    from experiment.phase20.lime_comparison_core import left_pad
    sys.path.insert(0, str(ROOT/'third_party/LIME-Rec'))
    from lime_rec.models import SASRec
    from lime_rec.evaluation import normalize_score_rows
    samples, folds = read(folder/'fit_samples.json'), np.load(folder/'folds.npy')
    records, terminal, schedules = D.expand(samples, folds, domain, CFG['max_prefixes_per_user'])
    write(folder/'prefix_samples.json', records)
    write(folder/'prefix_schedules.json', schedules)
    np.save(folder/'terminal_indices.npy', terminal)
    dc = read(folder/'data_config.json')
    packed = np.load(ROOT/dc['embedding'])
    emb = packed['embeddings']
    np.testing.assert_array_equal(packed['item_ids'],np.load(folder/'item_vectors.npz')['item_ids'])
    x = R.P.C.history_inputs([row['full_history'] for row in records], emb)
    np.testing.assert_allclose(x[terminal], np.load(folder/'fit_inputs.npy'), atol=2e-7, rtol=2e-6)
    np.save(folder/'prefix_inputs.npy', x)
    saved = torch.load(ROOT/dc['sasrec'], map_location='cpu')
    assert saved['train_complete'] and saved['training_only'] and saved['epoch'] == 200
    sas = SASRec(len(emb),64,50,2,2,0.,architecture='legacy')
    sas.load_state_dict(saved['state_dict'], strict=True)
    sas.eval().requires_grad_(False)
    original = np.load(R.P.FIT/domain/'features/train/experts.npy', mmap_mode='r')
    parity = []
    for j in (0, len(samples)//2, len(samples)-1):
        with torch.no_grad():
            logits = sas.predict(torch.from_numpy(left_pad([samples[j]['full_history']]))).numpy()[:,1:]
        actual = normalize_score_rows(logits)[0]
        expected = original[j,:,0]
        np.testing.assert_allclose(actual, expected, atol=5e-6, rtol=1e-5)
        parity.append(dict(user=samples[j]['user_id'], max_absolute_error=float(np.abs(actual-expected).max())))
    write(folder/'teacher_parity.json', dict(state='PASSED', rows=parity, cpu_only=True,
        cached_terminal_candidates_unchanged=True))
    n = len(records)
    pools = {k: np.empty((n,51), np.int64 if k.endswith('ids') else bool if k.endswith('valid') else np.float32)
             for k in ('hard_ids','hard_valid','hard_sas','random_ids','random_valid','random_sas')}
    old = np.load(folder/'terminal_training_pools.npz')
    for k in pools:
        pools[k][terminal] = old[k]
    extra = [j for j,row in enumerate(records) if not row['terminal']]
    for start in range(0,len(extra),32):
        indices = extra[start:start+32]
        with torch.no_grad():
            raw = sas.predict(torch.from_numpy(left_pad([records[j]['full_history'] for j in indices]))).numpy()[:,1:]
        raw = normalize_score_rows(raw)
        for j, score in zip(indices,raw):
            row = records[j]
            hard, valid = R.C.training_pool(R.C.top50(score), row['target'])
            seed = int.from_bytes(hashlib.sha256(
                f'prefix-random|{domain}|{row["user_id"]}|{row["position"]}'.encode()).digest()[:8],'big')
            rng = np.random.default_rng(seed)
            alternatives = rng.choice(len(emb)-1,50,replace=False)+1
            alternatives += alternatives >= row['target']
            random = np.ones(51,np.int64)
            random[0] = row['target']
            random[1:int(valid.sum())] = alternatives[:int(valid.sum())-1]
            for kind, ids in (('hard',hard),('random',random)):
                assert ids[0] == row['target'] and len(set(ids[valid])) == valid.sum()
                pools[kind+'_ids'][j], pools[kind+'_valid'][j] = ids, valid
                pools[kind+'_sas'][j] = score[ids-1]
        progress(domain,'mine_prefix_SASRec_candidates',prefixes_completed=min(start+32,len(extra)),
                 prefixes_total=len(extra),training_examples=n)
    for j,row in enumerate(records):
        for kind in ('hard','random'):
            valid = pools[kind+'_valid'][j]
            assert pools[kind+'_ids'][j,0] == row['target']
            assert len(set(pools[kind+'_ids'][j,valid])) == valid.sum()
        assert np.array_equal(pools['hard_valid'][j],pools['random_valid'][j])
    np.savez_compressed(folder/'prefix_training_pools.npz', **pools)
    write(folder/'prefix_audit.json', dict(state='PASSED', users=len(samples), examples=n,
        training_users=int((folds>=2).sum()),
        train_terminal_examples=int((folds>=2).sum()),
        train_available_prefix_examples=sum(row['fold']>=2 for row in records),
        no_user_crosses_fold=True, no_future_history_features=True, calibration_users_not_augmented=True,
        equal_optimizer_steps_per_user=True, test_read=False))
    del sas, original, saved
    gc.collect()


def train_models(domain, folder, specifications=None):
    specs = specifications or SPECS
    terminal = np.load(folder/'terminal_indices.npy')
    schedules = read(folder/'prefix_schedules.json')
    x = np.load(folder/'prefix_inputs.npy')
    base_x, folds = np.load(folder/'fit_inputs.npy'), np.load(folder/'folds.npy')
    train, stop, refit, cal = [np.flatnonzero(v) for v in (folds>=2,folds==1,folds!=0,folds==0)]
    mean, std = base_x[train].mean(0), np.maximum(base_x[train].std(0), .01)
    xt = torch.from_numpy(((x-mean)/std).astype(np.float32))
    bank = np.load(folder/'item_vectors.npz')['bank']
    item = torch.from_numpy(bank)
    pools = dict(np.load(folder/'prefix_training_pools.npz'))
    stop_proposals = dict(np.load(folder/'calibration_proposals.npz'))
    (folder/'models').mkdir()
    models = {}
    for arm,spec in specs.items():
        kind = 'random' if spec['objective']=='residual_random' else 'hard'
        ids, valid, sas = [torch.from_numpy(pools[kind+'_'+name] - (1 if name=='ids' else 0))
                           if name!='valid' else torch.from_numpy(pools[kind+'_'+name])
                           for name in ('ids','valid','sas')]
        checkpoint_dir = folder/'selection_checkpoints'/arm
        checkpoint_dir.mkdir(parents=True)
        def initialize():
            torch.manual_seed(CFG['seed'])
            return R.C.TextAdapter(x.shape[1],bank.shape[1],spec['rank'],CFG['bound'])
        def run(model, users, epochs, selection):
            optimizer = torch.optim.AdamW(model.parameters(),lr=CFG['lr'],weight_decay=CFG['weight_decay'])
            rng = np.random.default_rng(CFG['seed'])
            curve, learning, example_orders = [], [], []
            def assess(epoch):
                model.eval()
                pieces = []
                with torch.no_grad():
                    for start in range(0,len(stop),64):
                        jj = terminal[stop[start:start+64]]
                        delta = model(xt[jj],item[ids[jj]])
                        pieces.append((len(jj),float(R.C.objective(delta,sas[jj],valid[jj],
                            spec['objective'],CFG['regularization']))))
                loss = sum(n*v for n,v in pieces)/sum(n for n,v in pieces)
                delta = R.predict(model,base_x[stop],bank,stop_proposals['ids'][stop],mean,std)/2
                score = R.P.C.corrected_scores(20*stop_proposals['sas'][stop],-delta,1.)
                ranks = R.P.C.ranks(stop_proposals['ids'][stop],score,stop_proposals['targets'][stop])
                row = dict(epoch=epoch,stopping_CE=loss,mean_abs_delta=float(np.abs(delta).mean()),**metrics(ranks))
                curve.append(row)
                np.save(checkpoint_dir/f'epoch_{epoch}_stop_ranks.npy',ranks)
                torch.save({k:v.detach().clone() for k,v in model.state_dict().items()},checkpoint_dir/f'epoch_{epoch}.pt')
                return row
            if selection:
                assess(0)
            for epoch in range(epochs):
                began = time.monotonic()
                examples = D.epoch_examples(users,terminal,schedules,epoch,spec['augmented'])
                order = examples[rng.permutation(len(examples))]
                assert len(order)==len(users)
                example_orders.append(order)
                model.train()
                total = 0.
                for start in range(0,len(order),CFG['batch_size']):
                    jj = order[start:start+CFG['batch_size']]
                    optimizer.zero_grad()
                    value = R.C.objective(model(xt[jj],item[ids[jj]]),sas[jj],valid[jj],
                                          spec['objective'],CFG['regularization'])
                    assert torch.isfinite(value)
                    value.backward()
                    optimizer.step()
                    total += float(value)*len(jj)
                validation = assess(epoch+1) if selection and epoch+1 in CFG['checkpoints'] else None
                learning.append(dict(epoch=epoch+1,loss=total/len(order),stopping=validation))
                progress(domain,'adapter_selection' if selection else 'adapter_refit',arm=arm,
                         epoch=epoch+1,epochs=epochs,seconds_per_epoch=time.monotonic()-began,stopping=validation)
            return curve,learning,np.asarray(example_orders,dtype=np.int64)
        model = initialize()
        curve,learning,orders = run(model,train,CFG['epochs'],True)
        candidates = [row for row in curve if row['epoch']>0]
        selected = max(candidates,key=lambda row:(row['N10'],row['H10'],-row['stopping_CE'],-row['epoch']))
        chosen = selected['epoch']
        model = initialize()
        _,refit_learning,refit_orders = run(model,refit,chosen,False)
        model.eval().requires_grad_(False)
        assert model.item.abs().sum()>0, 'Selected positive checkpoint must produce a trained projection'
        np.savez_compressed(folder/f'training_schedule_{arm}.npz',selection=orders,refit=refit_orders)
        info = dict(selected_epochs=chosen,selection_curve=curve,selection_learning=learning,
                    refit_learning=refit_learning,specification=spec,
                    parameters=sum(p.numel() for p in model.parameters()),train_users=len(train),
                    stop_users=len(stop),refit_users=len(refit),calibration_users=len(cal),
                    selection_steps=CFG['epochs']*int(np.ceil(len(train)/CFG['batch_size'])),
                    positive_checkpoint_selection=True,epoch0_diagnostic=curve[0])
        samples = read(folder/'fit_samples.json')
        torch.save(dict(state_dict=model.state_dict(),mean=mean,std=std,config=CFG,info=info,
                        fit_users=[samples[j]['user_id'] for j in refit],
                        calibration_users=[samples[j]['user_id'] for j in cal]),folder/'models'/(arm+'.pt'))
        write(folder/('training_'+arm+'.json'),info)
        models[arm] = model
    pairs = [('terminal_rank32','prefix_rank32'),('terminal_rank4','residual_hard')]
    checked = []
    for a,b in pairs:
        if a not in specs or b not in specs:
            continue
        sa,sb = [torch.load(folder/'selection_checkpoints'/name/'epoch_1.pt',map_location='cpu') for name in (a,b)]
        for key in sa:
            torch.testing.assert_close(sa[key],sb[key],rtol=0,atol=0)
        oa,ob = [np.load(folder/f'training_schedule_{name}.npz')['selection'] for name in (a,b)]
        assert oa.shape==ob.shape
        np.testing.assert_array_equal(oa[0],ob[0])
        checked.append([a,b])
    write(folder/'schedule_parity.json',dict(state='PASSED',first_epoch_weights_exact=checked,
        same_users_and_optimizer_steps=True))
    return models,mean,std,cal


def worker(domain):
    bind()
    folder = OUT/domain
    signal.alarm(CFG['worker_timeout_seconds'])
    try:
        progress(domain,'verify_inputs')
        assert read(OUT/'config.json')==CFG
        manifest = read(OUT/'manifest.json')
        for rel in manifest['domain_inputs'][domain]:
            assert sha(ROOT/rel)==manifest['sha256'][rel], rel
        make_prefix_data(domain,folder)
        # Freeze all generated scientific inputs before fitting.
        files = [folder/name for name in ('prefix_samples.json','prefix_schedules.json','terminal_indices.npy',
            'prefix_inputs.npy','prefix_training_pools.npz','teacher_parity.json','prefix_audit.json')]
        write(folder/'prefix_input_manifest.json',dict(sha256={str(p.relative_to(ROOT)):sha(p) for p in files}))
        models,mean,std,cal = train_models(domain,folder)
        bank = np.load(folder/'item_vectors.npz')['bank']
        strengths = R.calibrate(folder,models,mean,std,cal,bank)
        progress(domain,'frozen_calibration_evaluate',strengths=strengths)
        R.evaluate(domain,folder,models,mean,std,strengths,bank)
        outputs = list((folder/'models').glob('*.pt'))+list(folder.glob('training_*.json'))
        outputs += list(folder.glob('training_schedule_*.npz'))
        outputs += list((folder/'selection_checkpoints').rglob('*.*'))
        outputs += [folder/name for name in ('selection.json','predictions.jsonl','corrections.npz','ranks.npz',
                                            'diagnostics.json','runtime_checks.json','schedule_parity.json')]
        write(folder/'output_manifest.json',dict(sha256={str(p.relative_to(ROOT)):sha(p) for p in outputs}))
        write(folder/'status.json',dict(state='COMPLETED',users_completed=1024,users_total=1024,
            updated_at=stamp(),gpu_used=False,test_read=False))
    finally:
        signal.alarm(0)


def interaction(rr,pool,suffix,indices):
    names = ('residual_hard','terminal_rank4','prefix_rank32','terminal_rank32')
    ranks = [rr[pool+'__'+name+'_'+suffix][indices] for name in names]
    values = []
    for kind in ('N10','H10'):
        v = [utilities(r) if kind=='N10' else (r<=10).astype(float) for r in ranks]
        delta = v[0]-v[1]-v[2]+v[3]
        rng = np.random.RandomState(20261017)
        boot = [delta[rng.randint(0,len(delta),len(delta))].mean() for _ in range(2000)]
        values.append((kind,dict(delta=float(delta.mean()),descriptive_95pct=np.percentile(boot,[2.5,97.5]).tolist())))
    return dict(values)


def summarize():
    result = {}
    pairs = [(pool+'__residual_hard_'+suffix,pool+'__'+control)
             for pool in PROPOSALS for suffix in ('cal','fixed1')
             for control in ('off','bge_'+suffix,'terminal_rank32_'+suffix,'prefix_rank32_'+suffix,
                             'terminal_rank4_'+suffix,'text_hard_'+suffix,'residual_random_'+suffix,
                             'residual_hard_shuffled_'+suffix)]
    lines = ['# 文本残差前缀扩充与小容量对照','','## Material Passport','',
        '- Origin Skill: academic-research-suite / experiment-agent','- Origin Mode: run / validate',
        '- Origin Date: 2026-10-17','- Verification Status: ANALYZED, reused development users','',
        '固定候选重排，不是新生成。所有头选择实际训练后的非零轮次，校准力度仍允许0；固定1组保留实际干预。',
        '同容量的单样本/多前缀优化步数相同，第1轮权重须完全相同。主方案固定为多前缀rank4，不按开发结果改选其他组。','']
    for domain in DOMAINS:
        folder = OUT/domain
        for name in ('prefix_input_manifest.json','output_manifest.json'):
            for rel,digest in read(folder/name)['sha256'].items():
                assert sha(ROOT/rel)==digest,rel
        samples = read(folder/'samples.json')
        rows = [json.loads(line) for line in (folder/'predictions.jsonl').read_text().splitlines()]
        rr = dict(np.load(folder/'ranks.npz'))
        assert rr.pop('users').tolist()==[s['user_id'] for s in samples]
        assert len(rows)==len(samples)==1024
        for j,row in enumerate(rows):
            assert row['user_id']==samples[j]['user_id'] and row['target']==samples[j]['target']
            for key,ar in rr.items():
                ids = row[key]['ids'];target = row['target']
                assert (ids.index(target)+1 if target in ids else 2**30)==ar[j]
        groups = dict(all1024=np.arange(1024),first512=np.arange(512),second512=np.arange(512,1024))
        blocks = np.asarray([s['evaluation_block'] for s in samples])
        groups.update({f'block{b+1}':np.flatnonzero(blocks==b) for b in range(5)})
        result[domain] = {}
        for group,indices in groups.items():
            entry = dict(users=len(indices),metrics={key:metrics(ar[indices]) for key,ar in rr.items()})
            if group in ('all1024','first512','second512'):
                entry['comparisons'] = {a+'_vs_'+b:contrast(rr[a][indices],rr[b][indices],2000,20261017) for a,b in pairs}
                entry['capacity_data_interaction'] = {pool+'__'+suffix:interaction(rr,pool,suffix,indices)
                    for pool in PROPOSALS for suffix in ('cal','fixed1')}
            result[domain][group] = entry
        write(folder/'summary.json',result[domain])
        strengths = read(folder/'selection.json')['strengths']
        lines += [f'## {domain}', '',f'前缀数据审计：`{json.dumps(read(folder/"prefix_audit.json"),ensure_ascii=False)}`','',
                  '| 训练组 | 轮次 | 参数 | 校准力度 | GRAM+SAS N10校准 | N10固定1 |','|---|---:|---:|---:|---:|---:|']
        mm = result[domain]['all1024']['metrics']
        for arm in SPECS:
            info = read(folder/('training_'+arm+'.json'))
            lines.append(f'| {arm} | {info["selected_epochs"]} | {info["parameters"]} | {strengths[arm]} | {mm["gram_sas__"+arm+"_cal"]["N10"]:.6f} | {mm["gram_sas__"+arm+"_fixed1"]["N10"]:.6f} |')
        lines += ['',f'原排序N10：{mm["gram_sas__off"]["N10"]:.6f}；原BGE校准：{mm["gram_sas__bge_cal"]["N10"]:.6f}。','']
    write(OUT/'summary.json',dict(domains=result,config=CFG,paired_comparisons=len(pairs)*3*2,
        interaction_comparisons=36,test_read=False,next_experiment_started=False,
        statistical_review=read(SOURCE/'summary.json')['statistical_review']))
    lines += ['所有区间为未经多重比较校正的开发集描述，不含训练种子与反复探索的不确定性。',
              'SASRec校准池向GRAM候选迁移、前缀teacher已见训练交互、最多8个前缀、单种子等限制仍须保留。',
              '校准力度0代表关闭贡献，不能解释为活跃头与基线等效；不得用最高开发组事后替换预定主方案。']
    REPORT.write_text('\n'.join(lines)+'\n')


def supervise():
    assert read(OUT/'status.json')['state']=='PREPARED'
    began,finished = time.monotonic(),{}
    for domain in DOMAINS:
        with (OUT/domain/'worker.log').open('x') as stream:
            p = subprocess.Popen([sys.executable,'-u','-m',MODULE,'worker','--domain',domain],cwd=ROOT,
                env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES=''),
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            started = time.monotonic()
            while p.poll() is None:
                if time.monotonic()-started>CFG['worker_timeout_seconds']+30:
                    R.P.A.B.stop(p)
                    break
                write(OUT/'status.json',dict(state='RUNNING',domain=domain,pid=os.getpid(),worker_pid=p.pid,
                    updated_at=stamp(),elapsed_seconds=time.monotonic()-began,worker=read(OUT/domain/'status.json'),
                    finished=finished,gpu_used=False,test_read=False,users_total=2048))
                time.sleep(5)
        status = read(OUT/domain/'status.json')
        finished[domain] = dict(exit_code=p.returncode,state=status['state'])
    complete = all(v['exit_code']==0 and v['state']=='COMPLETED' for v in finished.values())
    if complete:
        write(OUT/'status.json',dict(state='FINALIZING',finished=finished,updated_at=stamp(),test_read=False))
        summarize()
    write(OUT/'status.json',dict(state='COMPLETED' if complete else 'INCOMPLETE',finished=finished,
        updated_at=stamp(),elapsed_seconds=time.monotonic()-began,users_total=2048,
        users_completed=sum(1024 for v in finished.values() if v['state']=='COMPLETED'),gpu_used=False,
        report=str(REPORT.relative_to(ROOT)),test_read=False,next_experiment_started=False))


def tests():
    result = dict(prefix=D.tests(),adapter=R.C.tests())
    # Small end-to-end fitting fixture exercises positive checkpoint selection,
    # saved schedules, refitting, and exact first-epoch factorial parity.
    global CFG
    saved_cfg = CFG
    try:
        CFG = dict(CFG,epochs=2,checkpoints=[0,1,2],batch_size=4)
        with tempfile.TemporaryDirectory(prefix='prefix-residual-test-') as tmp:
            folder = Path(tmp)
            rng = np.random.default_rng(20)
            samples = [dict(user_id=f'u{j}',full_history=[1,2,3],target=4+j%20) for j in range(15)]
            folds = np.arange(15)%5
            records,terminal,schedules = D.expand(samples,folds,'synthetic')
            x = rng.normal(size=(len(records),7)).astype(np.float32)
            np.save(folder/'prefix_inputs.npy',x)
            np.save(folder/'fit_inputs.npy',x[terminal])
            np.save(folder/'folds.npy',folds)
            np.save(folder/'terminal_indices.npy',terminal)
            write(folder/'prefix_schedules.json',schedules)
            write(folder/'fit_samples.json',samples)
            np.savez(folder/'item_vectors.npz',bank=rng.normal(size=(60,5)).astype(np.float32))
            ids,valid = zip(*(R.C.training_pool(np.arange(1,51),row['target']) for row in records))
            ids,valid = np.asarray(ids),np.asarray(valid)
            sas = rng.random(ids.shape).astype(np.float32)
            np.savez(folder/'prefix_training_pools.npz',hard_ids=ids,hard_valid=valid,hard_sas=sas,
                     random_ids=ids,random_valid=valid,random_sas=sas)
            np.savez(folder/'calibration_proposals.npz',ids=np.tile(np.arange(1,51),(15,1)),
                     sas=rng.random((15,50)).astype(np.float32),bge=rng.random((15,50)).astype(np.float32),
                     targets=np.array([r['target'] for r in samples]))
            # Test progress must stay in /tmp instead of the scientific run.
            old_progress = globals()['progress']
            globals()['progress'] = lambda *a,**kw: None
            try:
                models,mean,std,cal = train_models('synthetic',folder)
            finally:
                globals()['progress'] = old_progress
            assert len(models)==6 and all(m.item.abs().sum()>0 for m in models.values())
            assert all(read(folder/f'training_{arm}.json')['selected_epochs']>0 for arm in SPECS)
            result['end_to_end_training'] = 'PASSED'
            result['first_epoch_factorial_parity'] = read(folder/'schedule_parity.json')
            # Exercise every added method, including actual nonzero heads,
            # calibration, shuffled histories, NumPy and full-user rank replay.
            dev = [dict(user_id=f'dev{j}',full_history=[1,2,3],history=[1,2,3],target=1+j%60,
                        evaluation_block=j%5) for j in range(1024)]
            write(folder/'samples.json',dev)
            np.save(folder/'evaluation_inputs.npy',rng.normal(size=(1024,7)).astype(np.float32))
            props = {}
            ids = np.tile(np.arange(1,51),(1024,1))
            scores = np.tile(np.linspace(1,0,50),(1024,1))
            for pool in PROPOSALS:
                props.update({pool+'_ids':ids,pool+'_scores':scores,
                              pool+'_bge':rng.random((1024,50)).astype(np.float32)})
            np.savez(folder/'proposals.npz',**props)
            reference = dict(ids=ids[0].tolist(),scores=scores[0].tolist())
            write(folder/'references.json',[dict(native=reference,sas_only=reference,pair_raw=reference) for _ in dev])
            old_r = (R.CFG,R.METHODS,R.PROPOSALS,R.progress)
            R.CFG,R.METHODS,R.PROPOSALS,R.progress = CFG,METHODS,PROPOSALS,lambda *a,**kw:None
            try:
                bank = np.load(folder/'item_vectors.npz')['bank']
                strengths = R.calibrate(folder,models,mean,std,cal,bank)
                R.evaluate('synthetic',folder,models,mean,std,strengths,bank)
            finally:
                R.CFG,R.METHODS,R.PROPOSALS,R.progress = old_r
            assert read(folder/'runtime_checks.json')['state']=='PASSED'
            result['all_methods_1024_user_evaluation_fixture'] = 'PASSED'
    finally:
        CFG = saved_cfg
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=('test','prepare','supervise','worker'))
    parser.add_argument('--domain',choices=DOMAINS)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    def interrupted(signum,frame):
        raise InterruptedError('Prefix residual run interrupted or hard timeout')
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGALRM,interrupted)
    try:
        if args.mode=='test':
            print(json.dumps(tests()),flush=True)
        elif args.mode=='prepare':
            prepare()
        elif args.mode=='supervise':
            supervise()
        else:
            worker(args.domain)
    except BaseException as exc:
        if OUT.exists():
            folder = OUT/args.domain if args.domain else OUT
            failure = dict(state='FAILED',error=repr(exc),traceback=traceback.format_exc(),updated_at=stamp(),test_read=False)
            write(folder/'failure.json',failure)
            write(folder/'status.json',failure)
        raise


if __name__=='__main__':
    main()
