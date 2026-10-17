"""CPU-only feasibility screen: fixed behavioral proposals and soft attributes."""
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

from experiment.phase21 import selfmatch_history_ablation as A
from experiment.phase21 import proposal_attribute_core as C
from experiment.phase20.catalog_scope_core import metrics, contrast, utilities

ROOT = A.ROOT
SOURCE = A.OUT
FIT = A.B.OUT
NATIVE = ROOT/'artifacts/phase20/lime_clean_benefit_20260922_v1'
OUT = ROOT/'artifacts/phase21/proposal_attribute_1024_20261017_v1'
PLAN = ROOT/'plan/第二十一阶段/21.51_行为提案与软属性约束可行性筛查_20261017.md'
REPORT = ROOT/'report/第二十一阶段/行为提案与软属性约束可行性筛查_20261017.md'
MODULE = 'experiment.phase21.proposal_attribute_screen'
DOMAINS = ('Toys', 'Beauty')
CFG = dict(users_per_domain=1024, source_indices=[1024,2048], proposals=['native','sas'],
    candidate_count=50, rerank_arms=list(C.ARMS), hidden=64, seed=0, epochs=60,
    checkpoints=[0,5,10,20,40,60], batch_size=128, lr=.001, weight_decay=.01,
    minimum_attribute_support=3, temperatures=[.5,1.,2.,4.,8.], strengths=[0.,.1,.25,.5,1.,2.],
    attributes=list(C.FIELDS), fitting='hash folds2/3/4; stop fold1; refit folds1/2/3/4; calibrate fold0',
    calibration_proposal='cached native GRAM50, shared by every arm; transfer to SAS-guided proposal',
    input='BGE full-history recency mean plus last-item text vector',
    confidence='temperature-scaled posterior normalized entropy, not guaranteed reliability',
    score='per-user std-normalized generated score minus strength times bounded penalty',
    history_filter=False, history_penalty=False, text_self_exclusion=False,
    gram_training_steps=0, expert_training_steps=0, gpu_used=False, cpu_threads=2,
    max_parallel=1, worker_timeout_seconds=3600, scientific_auto_retry=False,
    bootstrap_repeats=2000, test_read=False, automatic_next_experiment=False)
read, write, sha, stamp = A.read, A.write, A.sha, A.stamp


def progress(domain, stage, **extra):
    write(OUT/domain/'status.json', dict(state='RUNNING', domain=domain, stage=stage,
        pid=os.getpid(), updated_at=stamp(), gpu_used=False, test_read=False, **extra))


def public_spec(spec):
    return dict(field=spec['field'], values=spec['values'], prior=spec['prior'].tolist(),
        selection_prior=spec['selection_prior'].tolist(),
        support_target_counts=spec['support_target_counts'].tolist(),
        fitting_target_counts=spec['fitting_target_counts'].tolist(),
        catalogue_supported=int((spec['labels'][1:]>=0).sum()))


def specs_for(folder):
    public = read(folder/'schema.json')['supported_fields']
    with np.load(folder/'labels.npz') as packed:
        return [dict(s, labels=packed[s['field']].copy(), prior=np.asarray(s['prior']),
                     selection_prior=np.asarray(s['selection_prior'])) for s in public]


def prepare():
    checks = C.tests()
    OUT.mkdir(exist_ok=False)
    write(OUT/'status.json', dict(state='PREPARING', updated_at=stamp(), test_read=False, gpu_used=False))
    write(OUT/'config.json', CFG)
    write(OUT/'tests.json', checks)
    shutil.copy2(PLAN, OUT/'protocol.md')
    expected = read(SOURCE/'manifest.json')['sha256']
    manifest, domain_inputs = {}, {}

    def use(path, digest=None):
        rel = str(path.relative_to(ROOT))
        actual = manifest.get(rel) or sha(path)
        reference = digest or expected.get(rel)
        assert reference is None or actual == reference, rel
        manifest[rel] = actual
        return rel

    common = [use(p) for p in (Path(__file__), ROOT/'experiment/phase21/proposal_attribute_core.py',
        ROOT/'experiment/phase21/attribute_retention_core.py', ROOT/'experiment/phase20/lime_comparison_core.py',
        ROOT/'experiment/phase20/catalog_scope_core.py', ROOT/'experiment/phase21/selfmatch_history_ablation.py',
        ROOT/'experiment/phase21/fixed_expert_screen_4096.py', ROOT/'experiment/phase21/old10_expert_selection.py',
        PLAN, OUT/'config.json')]
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name and name.endswith('.py'):
            path=Path(name).resolve()
            if ROOT in path.parents:
                rel=use(path)
                if rel not in common:common.append(rel)
    write(OUT/'environment.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,
        cpu_threads=CFG['cpu_threads'],gpu_used=False))
    common.append(use(OUT/'environment.json'))
    for d in DOMAINS:
        folder = OUT/d
        folder.mkdir()
        files = []
        progress(d, 'freeze_inputs')
        for name in ('samples.json','references.json','catalog.json','data_config.json','teacher_checks.json'):
            source = SOURCE/d/name
            files.append(use(source))
            shutil.copy2(source, folder/name)
        fit_path = FIT/d/'fit_samples.json'
        fit = read(fit_path)
        samples = read(folder/'samples.json')
        assert len(samples)==1024 and len({s['user_id'] for s in samples})==1024
        assert {s['user_id'] for s in samples}.isdisjoint(s['user_id'] for s in fit)
        assert len(fit)==dict(Toys=2773,Beauty=3310)[d]
        files.append(use(fit_path))
        write(folder/'fit_samples.json', fit)
        source = NATIVE/d/'candidates.jsonl'
        files.append(use(source))
        by_user = {}
        with source.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['role']=='ranker_fit':
                    assert row['user_id'] not in by_user
                    by_user[row['user_id']] = row
        ids, scores = [], []
        original_samples = read(NATIVE/d/'samples.json')
        original_fit = {s['user_id']:s for s in original_samples if s['role']=='ranker_fit'}
        files.append(use(NATIVE/d/'samples.json'))
        for sample in fit:
            row = by_user[sample['user_id']]
            original = original_fit[sample['user_id']]
            assert original['full_history']==sample['full_history']
            assert row['target']==original['target']==sample['target']
            assert len(row['ids'])==len(set(row['ids']))==len(row['native'])==50
            ids.append(row['ids'])
            scores.append(row['native'])
        ids, scores = np.asarray(ids,np.int64), np.asarray(scores,float)
        metadata = FIT/d/'features/train/metadata.json'
        files.append(use(metadata))
        assert read(metadata)['users_order']==[s['user_id'] for s in fit]
        epath = FIT/d/'features/train/experts.npy'
        old_manifest = read(FIT/d/'feature_manifest.json')['sha256']
        files.append(use(epath, old_manifest[str(epath.relative_to(ROOT))]))
        e = np.load(epath,mmap_mode='r')
        bge = np.stack([e[j,items-1,3] for j,items in enumerate(ids)])
        np.savez_compressed(folder/'fit_candidates.npz',ids=ids,scores=scores,bge=bge,
            targets=np.asarray([s['target'] for s in fit]),users=np.asarray([s['user_id'] for s in fit]))
        dc = read(folder/'data_config.json')
        embpath = ROOT/dc['embedding']
        files.append(use(embpath))
        packed = np.load(embpath)
        catalog = read(folder/'catalog.json')
        assert packed['item_ids'].tolist()==catalog['items'][1:]
        emb = packed['embeddings']
        xfit = C.history_inputs([s['full_history'] for s in fit],emb)
        xdev = C.history_inputs([s['full_history'] for s in samples],emb)
        assert np.isfinite(xfit).all() and np.isfinite(xdev).all()
        np.save(folder/'fit_inputs.npy',xfit)
        np.save(folder/'evaluation_inputs.npy',xdev)
        folds = np.asarray([C.fold(s['user_id'],d) for s in fit])
        np.save(folder/'folds.npy',folds)
        schema = C.catalogue_schema(catalog,d)
        target = np.asarray([s['target'] for s in fit])
        specs = C.supported_schema(schema,target[folds>=2],minimum=CFG['minimum_attribute_support'])
        for spec in specs:
            spec['selection_prior']=spec['prior'].copy()
            spec['support_target_counts']=spec['fitting_target_counts'].copy()
            observed=spec['labels'][target[folds!=0]]
            counts=np.bincount(observed[observed>=0],minlength=len(spec['values']))
            spec['fitting_target_counts']=counts
            spec['prior']=(counts+.5)/(counts.sum()+.5*len(counts))
        write(folder/'schema.json',dict(supported_fields=[public_spec(s) for s in specs],
            raw_coverage={k:sum(bool(x) for x in v[1:])/len(v[1:]) for k,v in schema.items()},
            raw_value_counts={k:len({x for x in v[1:] if x}) for k,v in schema.items()},
            limitations='coarse/fine categories are correlated; missing/unsupported neutral; no function/compatibility extraction'))
        np.savez_compressed(folder/'labels.npz',**{s['field']:s['labels'] for s in specs})
        rawpath = SOURCE/d/'raw_experts.npy'
        files.append(use(rawpath))
        raw = np.load(rawpath,mmap_mode='r')
        refs = read(folder/'references.json')
        eval_pools = {}
        for proposal,key in (('native','native'),('sas','sas_only')):
            pool_ids = np.asarray([r[key]['ids'] for r in refs],np.int64)
            eval_pools[proposal+'_ids']=pool_ids
            eval_pools[proposal+'_scores']=np.asarray([r[key]['scores'] for r in refs],float)
            eval_pools[proposal+'_bge']=np.stack([raw[j,items-1,1] for j,items in enumerate(pool_ids)])
        for sample,row in zip(samples,refs):
            assert all(sample[k]==row[k] for k in ('user_id','target','history'))
        np.savez_compressed(folder/'evaluation_candidates.npz',**eval_pools)
        checks_d=dict(fit_users=len(fit),evaluation_users=1024,users_disjoint=True,source_indices=[1024,2048],
            fit_head_train_users=int((folds>=2).sum()),stop_users=int((folds==1).sum()),
            calibration_users=int((folds==0).sum()),no_target_added_to_candidate_pool=True,
            history_filter=False,calibration_uses_native_proposal_only=True,no_test_read=True)
        write(folder/'cohort_checks.json',checks_d)
        for name in ('samples.json','references.json','catalog.json','data_config.json','teacher_checks.json',
                     'fit_samples.json','fit_candidates.npz','fit_inputs.npy','evaluation_inputs.npy','folds.npy',
                     'schema.json','labels.npz','evaluation_candidates.npz','cohort_checks.json'):
            files.append(use(folder/name))
        domain_inputs[d]=common+files
        write(folder/'status.json',dict(state='PREPARED',updated_at=stamp(),**checks_d))
    for rel in manifest:
        source=ROOT/rel
        if source.suffix in ('.py','.md'):
            dest=OUT/'sources'/rel
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,dest)
    write(OUT/'manifest.json',dict(sha256=manifest,domain_inputs=domain_inputs))
    write(OUT/'status.json',dict(state='PREPARED',users_total=2048,updated_at=stamp(),gpu_used=False,test_read=False))
    REPORT.write_text('# 行为提案与软属性约束可行性筛查\n\n状态：PREPARED，尚无新效果结果。\n')
    print(json.dumps(dict(state='PREPARED',hashes=len(manifest),tests=checks)),flush=True)


def train_head(domain,folder,specs):
    x=np.load(folder/'fit_inputs.npy')
    folds=np.load(folder/'folds.npy')
    samples=read(folder/'fit_samples.json')
    labels=np.stack([s['labels'][[r['target'] for r in samples]] for s in specs],axis=1)
    train,stop,refit,cal=[np.flatnonzero(mask) for mask in (folds>=2,folds==1,folds!=0,folds==0)]
    mean=x[train].mean(0)
    std=np.maximum(x[train].std(0),.01)
    xt=torch.from_numpy(((x-mean)/std).astype(np.float32))
    yt=torch.from_numpy(labels)

    def initialize(selection=False):
        torch.manual_seed(CFG['seed'])
        initialization=[dict(s,prior=s['selection_prior']) for s in specs] if selection else specs
        return C.AttributeHead(x.shape[1],initialization,CFG['hidden'])

    def run(model,ix,epochs,selection):
        rng=np.random.default_rng(CFG['seed'])
        opt=torch.optim.AdamW(model.parameters(),lr=CFG['lr'],weight_decay=CFG['weight_decay'])
        curve=[]
        learning=[]
        def assess(epoch):
            model.eval()
            with torch.no_grad():
                value=float(C.loss(model(xt[stop]),yt[stop]))
            curve.append(dict(epoch=epoch,attribute_nll=value))
            return value
        if selection:assess(0)
        for epoch in range(1,epochs+1):
            model.train()
            order=rng.permutation(ix)
            total=0.
            for start in range(0,len(order),CFG['batch_size']):
                jj=order[start:start+CFG['batch_size']]
                opt.zero_grad()
                value=C.loss(model(xt[jj]),yt[jj])
                assert torch.isfinite(value)
                value.backward()
                opt.step()
                total+=float(value)*len(jj)
            held=assess(epoch) if selection and epoch in CFG['checkpoints'] else None
            learning.append(dict(epoch=epoch,training_attribute_nll=total/len(order),stopping_attribute_nll=held))
            progress(domain,'head_selection' if selection else 'head_refit',epoch=epoch,epochs=epochs,
                training_attribute_nll=total/len(order),stopping_attribute_nll=held)
        return curve,learning

    model=initialize(selection=True)
    curve,selection_learning=run(model,train,CFG['epochs'],True)
    selected=min(curve,key=lambda r:(r['attribute_nll'],r['epoch']))['epoch']
    model=initialize()
    _,refit_learning=run(model,refit,selected,False)
    model.eval().requires_grad_(False)
    with torch.no_grad():
        logits=[v.numpy() for v in model(xt)]
    temperatures=[]
    temp_curves={}
    for spec,z in zip(specs,logits):
        y=spec['labels'][[samples[j]['target'] for j in cal]]
        known=y>=0
        grid=[]
        for t in CFG['temperatures']:
            q=C.softmax(z[cal],t)
            nll=float(-np.log(np.maximum(q[np.flatnonzero(known),y[known]],1e-300)).mean()) if known.any() else None
            grid.append(dict(temperature=t,nll=nll,known=int(known.sum())))
        chosen=min(grid,key=lambda r:(r['nll'],abs(np.log(r['temperature']))))['temperature'] if known.any() else 1.
        temperatures.append(chosen)
        temp_curves[spec['field']]=grid
    info=dict(selected_epochs=selected,selection_curve=curve,temperatures=temperatures,
        selection_training_curve=selection_learning,refit_training_curve=refit_learning,
        temperature_curves=temp_curves,parameters=sum(v.numel() for v in model.parameters()),
        train_users=len(train),stopping_users=len(stop),refit_users=len(refit),calibration_users=len(cal))
    torch.save(dict(state_dict=model.state_dict(),mean=mean,std=std,config=CFG,info=info,
        fit_users=[samples[j]['user_id'] for j in refit],calibration_users=[samples[j]['user_id'] for j in cal]),folder/'attribute_head.pt')
    write(folder/'training.json',info)
    return model,mean,std,temperatures,cal,logits


def model_predict(model,inputs,mean,std,temperatures):
    x=torch.from_numpy(((np.asarray(inputs)-mean)/std).astype(np.float32))
    with torch.no_grad():
        logits=[z.numpy() for z in model(x)]
    return [C.softmax(z,t) for z,t in zip(logits,temperatures)],logits


def head_diagnostics(probabilities,specs,targets):
    result={}
    for q,spec in zip(probabilities,specs):
        y=spec['labels'][targets]
        known=y>=0
        rows=np.flatnonzero(known)
        hit=q.argmax(1)==y
        certainty=q.max(1)
        bins=[]
        for low,high in zip(np.linspace(0,1,6)[:-1],np.linspace(0,1,6)[1:]):
            chosen=known&(certainty>=low)&(certainty<=high if high==1 else certainty<high)
            bins.append(dict(lower=float(low),upper=float(high),users=int(chosen.sum()),
                accuracy=float(hit[chosen].mean()) if chosen.any() else None,
                mean_top1_probability=float(certainty[chosen].mean()) if chosen.any() else None))
        result[spec['field']]=dict(known_targets=int(known.sum()),known_fraction=float(known.mean()),
            attribute_nll=float(-np.log(np.maximum(q[rows,y[known]],1e-300)).mean()) if len(rows) else None,
            accuracy=float(hit[known].mean()) if known.any() else None,
            mean_entropy_weight=float(C.confidence(q).mean()),reliability_bins=bins)
    return result


def select_strengths(folder,specs,temperatures,cal,logits,domain):
    samples=read(folder/'fit_samples.json')
    subset=[samples[j] for j in cal]
    pools=dict(np.load(folder/'fit_candidates.npz'))
    probabilities=[C.softmax(z[cal],t) for z,t in zip(logits,temperatures)]
    penalties=C.penalties(probabilities,specs,pools['ids'][cal],pools['bge'][cal],
                          C.history_donors(subset,domain),C.shuffled_labels(specs,domain))
    strengths={}
    grids={}
    for arm in C.ARMS:
        grid=[]
        for value in ([0.] if arm=='off' else CFG['strengths']):
            scores=C.corrected_scores(pools['scores'][cal],penalties[arm],value)
            rr=C.ranks(pools['ids'][cal],scores,pools['targets'][cal])
            grid.append(dict(strength=value,**metrics(rr)))
        strengths[arm]=max(grid,key=lambda r:(r['N10'],r['H10'],-r['strength']))['strength']
        grids[arm]=grid
    write(folder/'selection.json',dict(strengths=strengths,grid=grids,calibration_users=len(cal),
        proposal='native GRAM50 only',evaluation_labels_used=False,head_diagnostics=head_diagnostics(probabilities,specs,pools['targets'][cal])))
    return strengths


def independent_forward(model,inputs,mean,std,temperatures,actual):
    state={k:v.numpy() for k,v in model.state_dict().items()}
    x=((inputs-mean)/std).astype(np.float32)
    h=np.tanh(x@state['input.weight'].T+state['input.bias'])
    errors=[]
    for k,(temperature,expected) in enumerate(zip(temperatures,actual)):
        z=h@state[f'heads.{k}.weight'].T+state[f'heads.{k}.bias']
        q=C.softmax(z,temperature)
        np.testing.assert_allclose(q,expected,atol=2e-6,rtol=2e-5)
        errors.append(float(np.abs(q-expected).max()))
    return errors


def evaluate(domain,folder,specs,model,mean,std,temperatures,strengths):
    samples=read(folder/'samples.json')
    refs=read(folder/'references.json')
    inputs=np.load(folder/'evaluation_inputs.npy')
    probabilities,logits=model_predict(model,inputs,mean,std,temperatures)
    selected=np.asarray([0,127,511,767,1023])
    errors=independent_forward(model,inputs[selected],mean,std,temperatures,[q[selected] for q in probabilities])
    np.savez_compressed(folder/'attribute_posteriors.npz',**{s['field']:q for s,q in zip(specs,probabilities)})
    targets=np.asarray([s['target'] for s in samples])
    pools=dict(np.load(folder/'evaluation_candidates.npz'))
    donors=C.history_donors(samples,domain)
    placebo=C.shuffled_labels(specs,domain)
    np.savez_compressed(folder/'placebo_assignments.npz',history_donors=donors,
                        **{s['field']:v for s,v in zip(specs,placebo)})
    predicted,ranked,penalties={},{},{}
    diagnostics={}
    for proposal in CFG['proposals']:
        ids,base,bge=[pools[proposal+'_'+k] for k in ('ids','scores','bge')]
        ps=C.penalties(probabilities,specs,ids,bge,donors,placebo)
        penalties.update({proposal+'__'+a:p for a,p in ps.items()})
        for arm in C.ARMS:
            key=proposal+'__'+arm
            score=C.corrected_scores(base,ps[arm],strengths[arm])
            ranked[key]=C.ranks(ids,score,targets)
            # Independent Python sort for every user, with unchanged candidate set.
            rows=[]
            for j in range(len(samples)):
                order=sorted(range(50),key=lambda k:(-float(score[j,k]),int(ids[j,k])))
                entry=dict(ids=ids[j,order].tolist(),scores=score[j,order].tolist())
                assert set(entry['ids'])==set(ids[j]) and len(set(entry['ids']))==50
                rr=entry['ids'].index(int(targets[j]))+1 if targets[j] in entry['ids'] else 2**30
                assert rr==ranked[key][j]
                rows.append(entry)
            predicted[key]=rows
            np.testing.assert_array_equal(ranked[key]<=50,ranked[proposal+'__off']<=50)
            target_positions=ids==targets[:,None]
            known=target_positions.any(1)
            target_penalty=ps[arm][target_positions]
            diagnostics[key]=dict(mean_penalty=float(ps[arm].mean()),
                target_present=int(known.sum()),target_penalty_mean=float(target_penalty.mean()) if known.any() else None,
                modified_users=sum(r['ids']!=predicted[proposal+'__off'][j]['ids'] for j,r in enumerate(rows)),
                mean_history_fraction_top10=float(np.mean([np.isin(r['ids'][:10],samples[j]['full_history']).mean() for j,r in enumerate(rows)])),
                mean_history_fraction_top50=float(np.mean([np.isin(r['ids'],samples[j]['full_history']).mean() for j,r in enumerate(rows)])))
        original_key='native' if proposal=='native' else 'sas_only'
        np.testing.assert_array_equal(ranked[proposal+'__off'],[A.B.rank(r[original_key],s['target']) for r,s in zip(refs,samples)])
        progress(domain,'rerank_fixed_'+proposal,users_completed=1024,users_total=1024)
    ranked['fixed_pair_reference']=np.asarray([A.B.rank(r['pair_raw'],s['target']) for r,s in zip(refs,samples)])
    with (folder/'predictions.jsonl').open('x') as stream:
        for j,s in enumerate(samples):
            row=dict(sample_index=j,user_id=s['user_id'],target=s['target'],history=s['history'],
                     source_validation_index=s['source_validation_index'],evaluation_block=s['evaluation_block'])
            row.update({k:v[j] for k,v in predicted.items()})
            row['fixed_pair_reference']=refs[j]['pair_raw']
            stream.write(json.dumps(row)+'\n')
    np.savez_compressed(folder/'penalties.npz',**penalties)
    np.savez_compressed(folder/'ranks.npz',users=np.asarray([s['user_id'] for s in samples]),**ranked)
    write(folder/'attribute_diagnostics.json',dict(fields=head_diagnostics(probabilities,specs,targets),methods=diagnostics,
        proposal_coverage={p:dict(Recall50=metrics(ranked[p+'__off'])['H50'],
            oracle_N10_upper_bound=metrics(ranked[p+'__off'])['H50'],oracle_is_not_a_real_method=True) for p in CFG['proposals']},
        history_donor_self_fraction=float((donors==np.arange(len(donors))).mean()),
        candidate_attribute_coverage={s['field']:float((s['labels'][pools['sas_ids']]>=0).mean()) for s in specs}))
    write(folder/'runtime_checks.json',dict(state='PASSED',users=1024,numpy_forward_max_error=errors,
        independent_all_user_ranking=True,unchanged_candidate_sets_all_arms=True,unchanged_H50_all_arms=True,
        native_and_sas_baselines_replayed=True,outside_targets_retained=True,zero_strength_exact=True,
        gpu_used=False,gram_training_steps=0,history_filter=False,test_read=False))
    return ranked


def worker(domain):
    folder=OUT/domain
    signal.alarm(CFG['worker_timeout_seconds'])
    try:
        progress(domain,'verify_frozen_inputs')
        assert read(OUT/'config.json')==CFG
        manifest=read(OUT/'manifest.json')
        for rel in manifest['domain_inputs'][domain]:
            assert sha(ROOT/rel)==manifest['sha256'][rel],rel
        specs=specs_for(folder)
        model,mean,std,temps,cal,logits=train_head(domain,folder,specs)
        strengths=select_strengths(folder,specs,temps,cal,logits,domain)
        progress(domain,'frozen_selection_evaluate',strengths=strengths)
        evaluate(domain,folder,specs,model,mean,std,temps,strengths)
        outputs=[folder/n for n in ('attribute_head.pt','training.json','selection.json','attribute_posteriors.npz',
            'placebo_assignments.npz','predictions.jsonl','penalties.npz','ranks.npz','attribute_diagnostics.json','runtime_checks.json')]
        write(folder/'output_manifest.json',dict(sha256={str(p.relative_to(ROOT)):sha(p) for p in outputs}))
        write(folder/'status.json',dict(state='COMPLETED',domain=domain,updated_at=stamp(),users_completed=1024,
            users_total=1024,gpu_used=False,test_read=False))
    finally:
        signal.alarm(0)


def effect_interval(values):
    rng=np.random.default_rng(20261017)
    boot=np.empty(CFG['bootstrap_repeats'])
    for j in range(len(boot)):
        boot[j]=values[rng.integers(0,len(values),len(values))].mean()
    return dict(delta_N10=float(values.mean()),descriptive_95pct_N10=np.quantile(boot,[.025,.975]).tolist())


def summarize():
    lines=['# 行为提案与软属性约束可行性筛查','','## Material Passport','',
        '- Origin Skill: academic-research-suite / experiment-agent','- Origin Mode: run / validate',
        '- Origin Date: 2026-10-17','- Verification Status: ANALYZED; saved rankings independently checked',
        '- Version Label: proposal_attribute_1024_20261017_v1','',
        '固定原生GRAM50与GRAM+SASRec50，属性预测只训练于训练侧留出用户；两域各1024开发用户。',
        '所有组不滤历史；没有新增生成或GPU使用。校准仅使用训练侧原生GRAM池，冻结后迁移到SASRec引导池。',
        '此轮检验类别/品牌软相容性，不检验功能、兼容性或真实即时硬需求。粗细类相关；熵权重不等于正确率。','']
    result={}
    pairs=[]
    for proposal in CFG['proposals']:
        pairs.extend([(proposal+'__attribute',proposal+'__'+a) for a in C.ARMS if a!='attribute'])
        pairs.append((proposal+'__bge',proposal+'__off'))
    pairs.extend([('sas__attribute','native__attribute'),('sas__off','native__off'),
                  ('sas__attribute','native__off'),('fixed_pair_reference','sas__off')])
    for d in DOMAINS:
        folder=OUT/d
        assert read(folder/'status.json')['state']=='COMPLETED'
        for rel,digest in read(folder/'output_manifest.json')['sha256'].items():
            assert sha(ROOT/rel)==digest
        samples=read(folder/'samples.json')
        rr=dict(np.load(folder/'ranks.npz'))
        users=rr.pop('users')
        assert users.tolist()==[s['user_id'] for s in samples]
        # Re-read emitted JSON independently; no history filtering in the metric.
        rows=[json.loads(line) for line in (folder/'predictions.jsonl').read_text().splitlines()]
        assert len(rows)==len(samples)==1024 and len({r['user_id'] for r in rows})==1024
        for j,(row,sample) in enumerate(zip(rows,samples)):
            assert row['user_id']==sample['user_id'] and row['target']==sample['target']
            for arm in rr:
                assert A.B.rank(row[arm],sample['target'])==rr[arm][j]
        groups=dict(all1024=np.arange(1024),first512=np.arange(512),second512=np.arange(512,1024))
        blocks=np.asarray([s['evaluation_block'] for s in samples])
        groups.update({f'block{j+1}':np.flatnonzero(blocks==j) for j in range(5)})
        result[d]={}
        for name,ix in groups.items():
            interaction=utilities(rr['sas__attribute'][ix])-utilities(rr['sas__off'][ix])-utilities(rr['native__attribute'][ix])+utilities(rr['native__off'][ix])
            result[d][name]=dict(users=len(ix),metrics={a:metrics(v[ix]) for a,v in rr.items()},
                comparisons={a+'_vs_'+b:contrast(rr[a][ix],rr[b][ix],repeats=2000,seed=20261017) for a,b in pairs},
                proposal_by_attribute_interaction=effect_interval(interaction))
        write(folder/'summary.json',result[d])
        selection=read(folder/'selection.json')
        training=read(folder/'training.json')
        lines.extend([f'## {d}','',f'属性头选轮：{training["selected_epochs"]}；参数量：{training["parameters"]}。',
            '校准强度：`'+json.dumps(selection['strengths'],ensure_ascii=False)+'`。','',
            '| 方法 | NDCG@10 | HR@10 | HR@50 |','|---|---:|---:|---:|'])
        for arm,m in result[d]['all1024']['metrics'].items():
            lines.append(f'| {arm} | {m["N10"]:.6f} | {m["H10"]:.6f} | {m["H50"]:.6f} |')
        lines.extend(['','| 主要比较 | ΔN10 | 描述性95%区间 |','|---|---:|---|'])
        comparisons=result[d]['all1024']['comparisons']
        for a,b in [('sas__attribute','sas__off'),('sas__attribute','sas__bge'),
                    ('sas__attribute','sas__attribute_history_shuffled'),('sas__attribute','sas__attribute_item_shuffled'),
                    ('sas__attribute','native__attribute')]:
            value=comparisons[a+'_vs_'+b]
            lines.append(f'| {a} − {b} | {value["delta_N10"]:+.6f} | {value["descriptive_95pct_N10"]} |')
        lines.append('')
    review=dict(coverage='11/11',simpson='Domains and batches/blocks separate; no per-domain selected method.',
        ecological='Per-user gains/losses saved; no universal benefit.',berkson='Fixed development cohort only.',
        collider='Targets never select candidate pools; outside targets retained.',base_rate='All1024 per domain retained.',
        regression_to_mean='Repeated exploration; no independent confirmation.',survivorship='All planned arms retained.',
        look_elsewhere='All intervals descriptive and unadjusted; selection grid fixed beforehand.',
        forking_paths='Attribute fields, split, head, grid and methods frozen before evaluation.',
        causation='Fixed-pool reranking only; calibration-proposal shift disclosed; no generation improvement claim.',
        reverse_causality='Training-side target attributes are supervision only; evaluation targets used for diagnostics/metrics.')
    write(OUT/'summary.json',dict(domains=result,config=CFG,statistical_review=review,
        descriptive_comparisons=len(pairs)*8*2,interaction_intervals=16,test_read=False,
        independent_output_rank_verification=True,automatic_next_experiment=False))
    lines.extend(['强度0是合法结果：表示当前校准数据没有支持该干预，不强制非零或据开发结果改强度。',
        '主看SASRec提案中真实属性相对不干预、普通BGE及错配的条件增量；HR@50必须与原池一致。',
        '不能把固定池重排收益写成生成期收益，也不能因本版小头失败否定所有属性约束。',
        '属性分支新增训练与最后商品文本输入，超过原始BGE不单独证明任务解耦的贡献；若升级机制论断需补同预算评分头控制。',
        '全部区间为未经多重比较校正的探索性bootstrap，不含训练种子与研究选择不确定性。'])
    REPORT.write_text('\n'.join(lines)+'\n')
    write(OUT/'output_manifest.json',dict(sha256={str(p.relative_to(ROOT)):sha(p) for p in (OUT/'summary.json',REPORT)}))


def supervise():
    assert read(OUT/'status.json')['state']=='PREPARED'
    finished={}
    began=time.monotonic()
    for domain in DOMAINS:
        with (OUT/domain/'worker.log').open('x') as log:
            process=subprocess.Popen([sys.executable,'-u','-m',MODULE,'worker','--domain',domain],cwd=ROOT,
                env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES=''),
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            started=time.monotonic()
            while process.poll() is None:
                if time.monotonic()-started>CFG['worker_timeout_seconds']+30:
                    A.B.stop(process)
                    break
                write(OUT/'status.json',dict(state='RUNNING',domain=domain,pid=os.getpid(),worker_pid=process.pid,
                    updated_at=stamp(),elapsed_seconds=time.monotonic()-began,worker=read(OUT/domain/'status.json'),
                    finished=finished,gpu_used=False,test_read=False,users_total=2048))
                time.sleep(5)
        status=read(OUT/domain/'status.json')
        finished[domain]=dict(exit_code=process.returncode,state=status['state'])
    complete=all(s['exit_code']==0 and s['state']=='COMPLETED' for s in finished.values())
    if complete:
        write(OUT/'status.json',dict(state='FINALIZING',finished=finished,updated_at=stamp(),
            users_completed=2048,users_total=2048,gpu_used=False,test_read=False))
        summarize()
    write(OUT/'status.json',dict(state='COMPLETED' if complete else 'INCOMPLETE',finished=finished,
        updated_at=stamp(),elapsed_seconds=time.monotonic()-began,users_total=2048,
        users_completed=sum(1024 for s in finished.values() if s['state']=='COMPLETED'),gpu_used=False,
        report=str(REPORT.relative_to(ROOT)),test_read=False,next_experiment_started=False))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('test','prepare','supervise','worker'))
    parser.add_argument('--domain',choices=DOMAINS)
    args=parser.parse_args()
    torch.set_num_threads(CFG['cpu_threads'])
    torch.set_num_interop_threads(1)
    def interrupted(signum,frame):raise InterruptedError('Attribute feasibility screen interrupted or timed out')
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGALRM,interrupted)
    try:
        if args.mode=='test':print(json.dumps(C.tests()),flush=True)
        elif args.mode=='prepare':prepare()
        elif args.mode=='supervise':supervise()
        else:worker(args.domain)
    except BaseException as exc:
        if OUT.exists():
            p=OUT/args.domain if args.domain else OUT
            value=dict(state='FAILED',error=repr(exc),traceback=traceback.format_exc(),updated_at=stamp(),test_read=False)
            write(p/'failure.json',value)
            write(p/'status.json',value)
        raise


if __name__=='__main__':main()
