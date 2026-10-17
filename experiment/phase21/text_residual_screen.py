"""CPU text-adapter training on SASRec hard alternatives, with matched controls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback

import numpy as np
import torch

from experiment.phase21 import proposal_attribute_screen as P
from experiment.phase21 import text_residual_core as C
from experiment.phase20.catalog_scope_core import metrics,contrast

ROOT=P.ROOT
SOURCE=P.OUT
OUT=ROOT/'artifacts/phase21/text_residual_hard_1024_20261017_v1'
PLAN=ROOT/'plan/第二十一阶段/21.53_SASRec困难候选的文本残差学习_20261017.md'
REPORT=ROOT/'report/第二十一阶段/SASRec困难候选文本残差筛查_20261017.md'
MODULE='experiment.phase21.text_residual_screen'
DOMAINS=P.DOMAINS
PROPOSALS=('sas_catalog','gram_sas','gram_native')
METHODS=('off','bge_cal','bge_fixed1','residual_hard_cal','residual_hard_fixed1',
    'text_hard_cal','text_hard_fixed1','residual_random_cal','residual_random_fixed1',
    'residual_hard_shuffled_cal','residual_hard_shuffled_fixed1')
CFG=dict(users_per_domain=1024,source_indices=[1024,2048],train_arms=list(C.TRAIN_ARMS),
    proposals=list(PROPOSALS),methods=list(METHODS),rank=32,bound=2.,seed=0,epochs=40,
    checkpoints=[0,5,10,20,40],batch_size=128,lr=.001,weight_decay=.01,regularization=.01,
    strengths=[0.,.1,.25,.5,1.,2.],fixed_diagnostic_strength=1.,cpu_threads=2,
    worker_timeout_seconds=3600,max_parallel=1,gpu_used=False,test_read=False,
    history_filter=False,history_penalty=False,text_self_exclusion=False,gram_training_steps=0,
    bge_backbone_training_steps=0,scientific_auto_retry=False,bootstrap_repeats=2000,
    calibration='training-side fold0 SASRec full-catalog Top50, no target injection',
    evaluation='fixed50 postrerank; transfer to GRAM+SASRec and native GRAM',automatic_next_experiment=False)
read,write,sha,stamp=P.read,P.write,P.sha,P.stamp


def progress(domain,stage,**extra):
    write(OUT/domain/'status.json',dict(state='RUNNING',domain=domain,stage=stage,pid=os.getpid(),
        updated_at=stamp(),gpu_used=False,test_read=False,**extra))


def prepare():
    checks=C.tests()
    OUT.mkdir(exist_ok=False)
    write(OUT/'config.json',CFG)
    write(OUT/'tests.json',checks)
    shutil.copy2(PLAN,OUT/'protocol.md')
    expected=read(SOURCE/'manifest.json')['sha256']
    manifest={}
    def use(path):
        rel=str(path.relative_to(ROOT))
        digest=manifest.get(rel) or sha(path)
        assert rel not in expected or expected[rel]==digest,rel
        manifest[rel]=digest
        return rel
    common=[]
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name and name.endswith('.py'):
            path=Path(name).resolve()
            if ROOT in path.parents:
                rel=use(path)
                if rel not in common:common.append(rel)
    common.extend([use(PLAN),use(OUT/'config.json')])
    inputs={}
    for d in DOMAINS:
        folder=OUT/d
        folder.mkdir()
        progress(d,'freeze_text_inputs_and_hard_alternatives')
        files=[]
        for name in ('samples.json','references.json','fit_samples.json','folds.npy',
                     'fit_inputs.npy','evaluation_inputs.npy','evaluation_candidates.npz','data_config.json'):
            source=SOURCE/d/name
            files.append(use(source))
            if source.suffix=='.npy':(folder/name).symlink_to(source)
            else:shutil.copy2(source,folder/name)
            files.append(use(folder/name))
        dc=read(folder/'data_config.json')
        embpath=ROOT/dc['embedding']
        files.append(use(embpath))
        packed=np.load(embpath)
        catalog=read(SOURCE/d/'catalog.json')
        files.append(use(SOURCE/d/'catalog.json'))
        assert packed['item_ids'].tolist()==catalog['items'][1:]
        emb=packed['embeddings'].astype(np.float32)
        mean=emb.mean(0)
        std=np.maximum(emb.std(0),.01)
        bank=(emb-mean)/std
        np.savez_compressed(folder/'item_vectors.npz',bank=bank,mean=mean,std=std,item_ids=packed['item_ids'])
        samples=read(folder/'fit_samples.json')
        n,v=len(samples),len(emb)
        ep=P.FIT/d/'features/train/experts.npy'
        files.append(use(ep))
        raw=np.load(ep,mmap_mode='r')
        ids=np.empty((n,50),np.int64)
        sas=np.empty((n,50),np.float32)
        bge=np.empty_like(sas)
        training={k:np.empty((n,51),np.int64 if k.endswith('ids') else bool if k.endswith('valid') else np.float32)
                  for k in ('hard_ids','hard_valid','hard_sas','random_ids','random_valid','random_sas')}
        for j,s in enumerate(samples):
            choices=C.top50(raw[j,:,0])
            ids[j],sas[j],bge[j]=choices,raw[j,choices-1,0],raw[j,choices-1,3]
            hi,valid=C.training_pool(choices,s['target'])
            seed=int.from_bytes(hashlib.sha256(f'text-hard-random|{d}|{s["user_id"]}'.encode()).digest()[:8],'big')
            rng=np.random.default_rng(seed)
            negatives=rng.choice(v-1,50,replace=False)+1
            negatives+=negatives>=s['target']
            ri=np.ones(51,np.int64)
            ri[0]=s['target']
            ri[1:int(valid.sum())]=negatives[:int(valid.sum())-1]
            for kind,items in (('hard',hi),('random',ri)):
                training[kind+'_ids'][j]=items
                training[kind+'_valid'][j]=valid
                training[kind+'_sas'][j]=raw[j,items-1,0]
            assert len(set(ri[valid]))==int(valid.sum())
        np.savez_compressed(folder/'training_pools.npz',**training)
        np.savez_compressed(folder/'calibration_proposals.npz',ids=ids,sas=sas,bge=bge,
            targets=np.asarray([s['target'] for s in samples]))
        evalp=dict(np.load(folder/'evaluation_candidates.npz'))
        original=P.A.OUT/d/'raw_experts.npy'
        files.append(use(original))
        e=np.load(original,mmap_mode='r')
        ei=np.stack([C.top50(e[j,:,0]) for j in range(1024)])
        newp=dict(sas_catalog_ids=ei,sas_catalog_scores=np.stack([20*e[j,x-1,0] for j,x in enumerate(ei)]),
                  sas_catalog_bge=np.stack([e[j,x-1,1] for j,x in enumerate(ei)]))
        for dest,source in (('gram_sas','sas'),('gram_native','native')):
            for key in ('ids','scores','bge'):newp[dest+'_'+key]=evalp[source+'_'+key]
        np.savez_compressed(folder/'proposals.npz',**newp)
        dev=read(folder/'samples.json')
        assert len(dev)==len({s['user_id'] for s in dev})==1024
        assert {s['user_id'] for s in dev}.isdisjoint(s['user_id'] for s in samples)
        write(folder/'cohort_checks.json',dict(fit_users=n,evaluation_users=1024,users_disjoint=True,
            hard_and_random_counts_identical=True,training_only_target_injection=True,
            evaluation_and_calibration_pools_target_free=True,history_filter=False,test_read=False))
        for name in ('item_vectors.npz','training_pools.npz','calibration_proposals.npz','proposals.npz','cohort_checks.json'):
            files.append(use(folder/name))
        inputs[d]=common+files
        write(folder/'status.json',dict(state='PREPARED',updated_at=stamp(),gpu_used=False,test_read=False))
    for rel in manifest:
        source=ROOT/rel
        if source.suffix in ('.py','.md'):
            dest=OUT/'sources'/rel
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,dest)
    write(OUT/'manifest.json',dict(sha256=manifest,domain_inputs=inputs))
    write(OUT/'status.json',dict(state='PREPARED',updated_at=stamp(),users_total=2048,gpu_used=False,test_read=False))
    REPORT.write_text('# SASRec困难候选文本残差筛查\n\n状态：PREPARED。\n')
    print(json.dumps(dict(state='PREPARED',checks=checks,hashes=len(manifest))),flush=True)


def train(domain,folder,bank):
    x=np.load(folder/'fit_inputs.npy')
    folds=np.load(folder/'folds.npy')
    train,stop,refit,cal=[np.flatnonzero(v) for v in (folds>=2,folds==1,folds!=0,folds==0)]
    mean=x[train].mean(0)
    std=np.maximum(x[train].std(0),.01)
    xt=torch.from_numpy(((x-mean)/std).astype(np.float32))
    item=torch.from_numpy(bank)
    pools=dict(np.load(folder/'training_pools.npz'))
    models={}
    (folder/'models').mkdir()
    for arm in C.TRAIN_ARMS:
        kind='random' if arm=='residual_random' else 'hard'
        ids=torch.from_numpy(pools[kind+'_ids']-1)
        valid=torch.from_numpy(pools[kind+'_valid'])
        sas=torch.from_numpy(pools[kind+'_sas'])
        def initialize():
            torch.manual_seed(CFG['seed'])
            return C.TextAdapter(x.shape[1],bank.shape[1],CFG['rank'],CFG['bound'])
        def run(model,indices,epochs,selection):
            opt=torch.optim.AdamW(model.parameters(),lr=CFG['lr'],weight_decay=CFG['weight_decay'])
            rng=np.random.default_rng(CFG['seed'])
            curve=[]
            learning=[]
            def held(epoch):
                model.eval()
                values=[]
                with torch.no_grad():
                    for start in range(0,len(stop),128):
                        jj=stop[start:start+128]
                        values.append((len(jj),float(C.objective(model(xt[jj],item[ids[jj]]),sas[jj],valid[jj],arm,CFG['regularization']))))
                loss=sum(n*v for n,v in values)/sum(n for n,v in values)
                curve.append(dict(epoch=epoch,loss=loss))
                return loss
            if selection:held(0)
            for epoch in range(1,epochs+1):
                began=time.monotonic()
                model.train()
                order=rng.permutation(indices)
                total=0.
                for start in range(0,len(order),CFG['batch_size']):
                    jj=order[start:start+CFG['batch_size']]
                    opt.zero_grad()
                    value=C.objective(model(xt[jj],item[ids[jj]]),sas[jj],valid[jj],arm,CFG['regularization'])
                    assert torch.isfinite(value)
                    value.backward();opt.step()
                    total+=float(value)*len(jj)
                validation=held(epoch) if selection and epoch in CFG['checkpoints'] else None
                learning.append(dict(epoch=epoch,loss=total/len(order),stopping_loss=validation))
                progress(domain,'adapter_selection' if selection else 'adapter_refit',arm=arm,epoch=epoch,epochs=epochs,
                    seconds_per_epoch=time.monotonic()-began,stopping_loss=validation)
            return curve,learning
        model=initialize()
        curve,selection=run(model,train,CFG['epochs'],True)
        chosen=min(curve,key=lambda row:(row['loss'],row['epoch']))['epoch']
        model=initialize()
        _,learning=run(model,refit,chosen,False)
        model.eval().requires_grad_(False)
        info=dict(selected_epochs=chosen,selection_curve=curve,selection_learning=selection,refit_learning=learning,
                  parameters=sum(p.numel() for p in model.parameters()),train_users=len(train),stop_users=len(stop),
                  refit_users=len(refit),calibration_users=len(cal))
        samples=read(folder/'fit_samples.json')
        torch.save(dict(state_dict=model.state_dict(),mean=mean,std=std,config=CFG,info=info,
            fit_users=[samples[j]['user_id'] for j in refit],calibration_users=[samples[j]['user_id'] for j in cal]),folder/'models'/(arm+'.pt'))
        write(folder/('training_'+arm+'.json'),info)
        models[arm]=model
    return models,mean,std,cal


def predict(model,x,bank,ids,mean,std):
    values=[]
    for start in range(0,len(x),64):
        stop=min(start+64,len(x))
        with torch.no_grad():
            values.append(model(torch.from_numpy(((x[start:stop]-mean)/std).astype(np.float32)),
                torch.from_numpy(bank[ids[start:stop]-1])).numpy())
    return np.concatenate(values)


def calibrate(folder,models,mean,std,cal,bank):
    pools=dict(np.load(folder/'calibration_proposals.npz'))
    x=np.load(folder/'fit_inputs.npy')[cal]
    signals={'bge':2*pools['bge'][cal]-1}
    for arm,model in models.items():signals[arm]=predict(model,x,bank,pools['ids'][cal],mean,std)/2
    strengths={}
    grids={}
    for arm,delta in signals.items():
        grid=[]
        for value in CFG['strengths']:
            score=P.C.corrected_scores(20*pools['sas'][cal],-delta,value)
            rr=P.C.ranks(pools['ids'][cal],score,pools['targets'][cal])
            grid.append(dict(strength=value,**metrics(rr)))
        strengths[arm]=max(grid,key=lambda row:(row['N10'],row['H10'],-row['strength']))['strength']
        grids[arm]=grid
    write(folder/'selection.json',dict(strengths=strengths,grids=grids,calibration_users=len(cal),
        calibration_pool='SASRec catalog Top50; target never inserted',evaluation_targets_used=False))
    return strengths


def evaluate(domain,folder,models,mean,std,strengths,bank):
    samples=read(folder/'samples.json')
    targets=np.asarray([s['target'] for s in samples])
    x=np.load(folder/'evaluation_inputs.npy')
    proposals=dict(np.load(folder/'proposals.npz'))
    donors=P.C.history_donors(samples,domain)
    predictions={}
    ranks={}
    corrections={}
    diagnostics={}
    checks=[]
    for proposal in PROPOSALS:
        ids,base,bge=[proposals[proposal+'_'+k] for k in ('ids','scores','bge')]
        deltas={arm:predict(model,x,bank,ids,mean,std)/2 for arm,model in models.items()}
        # Wrong histories retain the recipient's candidate pool.
        deltas['residual_hard_shuffled']=predict(models['residual_hard'],x[donors],bank,ids,mean,std)/2
        deltas['bge']=2*bge-1
        for arm,model in models.items():
            js=np.asarray([0,127,511,767,1023])
            state={k:v.numpy() for k,v in model.state_dict().items()}
            independent=C.numpy_delta(state,((x[js]-mean)/std).astype(np.float32),bank[ids[js]-1],CFG['bound'])/2
            np.testing.assert_allclose(independent,deltas[arm][js],atol=2e-6,rtol=2e-5)
            checks.append(dict(proposal=proposal,arm=arm,max_numpy_error=float(np.abs(independent-deltas[arm][js]).max())))
        for method in METHODS:
            key=proposal+'__'+method
            if method=='off':delta=np.zeros(base.shape);strength=0.
            else:
                arm=method.rsplit('_',1)[0]
                delta=deltas[arm]
                strength=1. if method.endswith('fixed1') else strengths['residual_hard' if arm.endswith('shuffled') else arm]
            score=P.C.corrected_scores(base,-delta,strength)
            rr=P.C.ranks(ids,score,targets)
            rows=[]
            for j in range(len(samples)):
                order=sorted(range(50),key=lambda k:(-float(score[j,k]),int(ids[j,k])))
                entry=dict(ids=ids[j,order].tolist(),scores=score[j,order].tolist())
                assert set(entry['ids'])==set(ids[j]) and len(set(entry['ids']))==50
                measured=entry['ids'].index(int(targets[j]))+1 if targets[j] in entry['ids'] else 2**30
                assert measured==rr[j]
                rows.append(entry)
            ranks[key]=rr
            predictions[key]=rows
            corrections[key]=delta
            diagnostics[key]=dict(strength=strength,mean_absolute_correction=float(np.abs(delta).mean()),
                modified_user_orders=sum(r['ids']!=predictions[proposal+'__off'][j]['ids'] for j,r in enumerate(rows)))
            np.testing.assert_array_equal(rr<=50,ranks[proposal+'__off']<=50)
        progress(domain,'evaluate_'+proposal,users_completed=1024,users_total=1024)
    refs=read(folder/'references.json')
    for proposal,ref in [('gram_sas','sas_only'),('gram_native','native')]:
        np.testing.assert_array_equal(ranks[proposal+'__off'],[P.A.B.rank(r[ref],s['target']) for r,s in zip(refs,samples)])
    ranks['fixed_pair_reference']=np.asarray([P.A.B.rank(r['pair_raw'],s['target']) for r,s in zip(refs,samples)])
    with (folder/'predictions.jsonl').open('x') as stream:
        for j,s in enumerate(samples):
            row=dict(sample_index=j,user_id=s['user_id'],target=s['target'],history=s['history'],evaluation_block=s['evaluation_block'])
            row.update({k:v[j] for k,v in predictions.items()})
            row['fixed_pair_reference']=refs[j]['pair_raw']
            stream.write(json.dumps(row)+'\n')
    np.savez_compressed(folder/'corrections.npz',**corrections)
    np.savez_compressed(folder/'ranks.npz',users=np.asarray([s['user_id'] for s in samples]),**ranks)
    write(folder/'diagnostics.json',diagnostics)
    write(folder/'runtime_checks.json',dict(state='PASSED',users=1024,numpy_forward=checks,
        unchanged_candidate_sets=True,unchanged_H50=True,independent_all_user_ranks=True,
        native_and_sas_guided_baselines_exact=True,history_filter=False,targets_not_in_inference=True,test_read=False))


def worker(domain):
    folder=OUT/domain
    signal.alarm(CFG['worker_timeout_seconds'])
    try:
        progress(domain,'verify_inputs')
        assert read(OUT/'config.json')==CFG
        m=read(OUT/'manifest.json')
        for rel in m['domain_inputs'][domain]:assert sha(ROOT/rel)==m['sha256'][rel],rel
        bank=np.load(folder/'item_vectors.npz')['bank']
        models,mean,std,cal=train(domain,folder,bank)
        strengths=calibrate(folder,models,mean,std,cal,bank)
        progress(domain,'frozen_calibration_evaluate',strengths=strengths)
        evaluate(domain,folder,models,mean,std,strengths,bank)
        paths=list((folder/'models').glob('*.pt'))+list(folder.glob('training_*.json'))
        paths += [folder/n for n in ('selection.json','predictions.jsonl','corrections.npz','ranks.npz','diagnostics.json','runtime_checks.json')]
        write(folder/'output_manifest.json',dict(sha256={str(p.relative_to(ROOT)):sha(p) for p in paths}))
        write(folder/'status.json',dict(state='COMPLETED',users_completed=1024,users_total=1024,
            updated_at=stamp(),gpu_used=False,test_read=False))
    finally:signal.alarm(0)


def summarize():
    result={}
    pairs=[]
    for proposal in PROPOSALS:
        for suffix in ('cal','fixed1'):
            a=proposal+'__residual_hard_'+suffix
            pairs.extend((a,proposal+'__'+b) for b in ('off','bge_'+suffix,'text_hard_'+suffix,
                'residual_random_'+suffix,'residual_hard_shuffled_'+suffix))
    pairs.append(('gram_sas__residual_hard_cal','gram_native__residual_hard_cal'))
    lines=['# SASRec困难候选文本残差筛查','','## Material Passport','',
        '- Origin Skill: academic-research-suite / experiment-agent','- Origin Mode: run / validate',
        '- Origin Date: 2026-10-17','- Verification Status: ANALYZED, fixed development users','',
        'BGE主干冻结，只学习小型语义适配层；校准与固定非零版本全部保留。固定50候选后重排，不是新增生成。',
        '训练池可按监督加入目标；校准和评估池不加入目标，不过滤历史。SASRec目录池校准向GRAM候选迁移仍有分布差异。','']
    for d in DOMAINS:
        folder=OUT/d
        for rel,digest in read(folder/'output_manifest.json')['sha256'].items():assert sha(ROOT/rel)==digest
        samples=read(folder/'samples.json')
        rr=dict(np.load(folder/'ranks.npz'));users=rr.pop('users')
        assert users.tolist()==[s['user_id'] for s in samples]
        rows=[json.loads(line) for line in (folder/'predictions.jsonl').read_text().splitlines()]
        assert len(rows)==1024 and len({r['user_id'] for r in rows})==1024
        for j,(row,s) in enumerate(zip(rows,samples)):
            assert row['user_id']==s['user_id'] and row['target']==s['target']
            for key in rr:assert P.A.B.rank(row[key],s['target'])==rr[key][j]
        groups=dict(all1024=np.arange(1024),first512=np.arange(512),second512=np.arange(512,1024))
        blocks=np.asarray([s['evaluation_block'] for s in samples])
        groups.update({f'block{b+1}':np.flatnonzero(blocks==b) for b in range(5)})
        result[d]={}
        for group,ix in groups.items():
            result[d][group]=dict(users=len(ix),metrics={k:metrics(v[ix]) for k,v in rr.items()})
            if group in ('all1024','first512','second512'):
                result[d][group]['comparisons']={a+'_vs_'+b:contrast(rr[a][ix],rr[b][ix],repeats=2000,seed=20261017) for a,b in pairs}
        write(folder/'summary.json',result[d])
        lines += [f'## {d} / GRAM+SASRec固定候选','','| 方法 | NDCG@10 | HR@10 | HR@50 |','|---|---:|---:|---:|']
        for method in METHODS:
            m=result[d]['all1024']['metrics']['gram_sas__'+method]
            lines.append(f'| {method} | {m["N10"]:.6f} | {m["H10"]:.6f} | {m["H50"]:.6f} |')
        lines += ['', 'SASRec目录池、原生GRAM池、两批及五块和全部预定配对区间见summary.json。','']
    write(OUT/'summary.json',dict(domains=result,config=CFG,comparisons=len(pairs)*3*2,
        statistical_review=dict(coverage='11/11',simpson='Domains/groups separated.',ecological='No universal user benefit.',
            berkson='Fixed development users.',collider='Evaluation pools target-free; all absent targets retained.',
            base_rate='All1024 users per domain.',regression_to_mean='Reused development data, exploratory.',
            survivorship='All heads and calibrated/nonzero controls retained.',look_elsewhere='All planned descriptive intervals, unadjusted.',
            forking_paths='Model/negative pools/splits/budgets frozen.',causation='Postrerank only; identical-budget controls; transfer limitation.',
            reverse_causality='Training targets supervised only; no target in inference.'),test_read=False,next_experiment_started=False))
    lines += ['所有区间是未经多重比较校正的开发集描述，不包括训练种子和研究选择不确定性。',
        '若只有纯SASRec池改善，不当成已经改善生成推荐；固定非零组不能事后取代校准组充当预定主方案。']
    REPORT.write_text('\n'.join(lines)+'\n')


def supervise():
    P.OUT,P.CFG,P.MODULE,P.REPORT,P.summarize=OUT,CFG,MODULE,REPORT,summarize
    P.supervise()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('test','prepare','supervise','worker'))
    parser.add_argument('--domain',choices=DOMAINS)
    args=parser.parse_args()
    torch.set_num_threads(CFG['cpu_threads']);torch.set_num_interop_threads(1)
    def interrupted(signum,frame):raise InterruptedError('Text residual screen interrupted or hard timeout')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGALRM,interrupted)
    try:
        if args.mode=='test':print(json.dumps(C.tests()),flush=True)
        elif args.mode=='prepare':prepare()
        elif args.mode=='supervise':supervise()
        else:worker(args.domain)
    except BaseException as exc:
        if OUT.exists():
            p=OUT/args.domain if args.domain else OUT
            value=dict(state='FAILED',error=repr(exc),traceback=traceback.format_exc(),updated_at=stamp(),test_read=False)
            write(p/'failure.json',value);write(p/'status.json',value)
        raise


if __name__=='__main__':main()
