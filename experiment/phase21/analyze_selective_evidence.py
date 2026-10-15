"""Bounded CPU probe: which expert suggestions merit replacing fused item ten?"""
import csv
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

from selective_evidence_core import ARMS,BASE_NAMES,EVIDENCE_NAMES,Evidence,donors,fold,model_checks,predict,proposals,ridge_fit

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'artifacts/phase20/lime_clean_benefit_20260922_v1'
FULL=ROOT/'artifacts/phase21/full_validation_20260924_v1'
OUT=ROOT/'artifacts/phase21/selective_evidence_20260925_v1'
PROTOCOL=ROOT/'plan/第二十一阶段/21.11_专家替换建议的增量依据检查_20260925.md'
ABSENT=2**30


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def csv_write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)


def feature_data(domain,kind,engine,manifest):
    training=kind=='train';folder=(SOURCE if training else FULL)/domain
    out=OUT/domain/kind;out.mkdir(parents=True)
    samples=json.loads((folder/'samples.json').read_text())
    selected=[(i,s) for i,s in enumerate(samples) if not training or s['role']=='ranker_fit']
    source_indices=[i for i,s in selected];samples=[s for i,s in selected]
    ds=donors(samples,domain,training)
    experts=np.load(folder/'experts.npy',mmap_mode='r',allow_pickle=False)
    fused=np.load(folder/('lime.npy' if training else 'fused.npy'),mmap_mode='r',allow_pickle=False)
    with np.load(folder/'gate_prediction.npz',allow_pickle=False) as a:weights=a['weights'];penalties=a['penalty']
    counts=np.load(SOURCE/domain/'frequencies.npy',allow_pickle=False)
    reference=ROOT/('artifacts/phase21/expert_signal_loss_20260924_v1/per_user.csv' if training else 'artifacts/phase21/full_opportunity_counts_20260925_v1/per_user.csv')
    with reference.open() as f:reference={r['user_id']:int(r['LIME']) for r in csv.DictReader(f) if r['domain']==domain}
    if training:
        expected=json.loads((ROOT/'artifacts/phase21/expert_signal_loss_20260924_v1/manifest.json').read_text())
        if 'sha256' in expected:expected=expected['sha256']
    else:expected=json.loads((folder/'score_cache_manifest.json').read_text())['sha256']
    for name in ('samples.json','experts.npy','lime.npy' if training else 'fused.npy','gate_prediction.npz'):
        path=folder/name;digest=sha(path);rel=str(path.relative_to(ROOT));manifest[rel]=digest
        exp=expected.get(rel) if training else expected.get(name)
        if exp is not None:assert digest==exp,(domain,kind,name)
    capacity=30*len(samples)
    arrays={k:np.lib.format.open_memmap(out/(k+'.npy'),mode='w+',dtype='float32',shape=(capacity,dim))
            for k,dim in (('score',len(BASE_NAMES)),('evidence',len(EVIDENCE_NAMES)),('shuffled',len(EVIDENCE_NAMES)))}
    item_ids=np.zeros(capacity,np.int32);labels=np.zeros(capacity,np.int8);row_user=np.zeros(capacity,np.int32)
    train_weight=np.zeros(capacity,np.float64);offsets=[0];records=[];p=0
    for local,(source_i,s) in enumerate(selected):
        ids,b,x,rr,ten,er=proposals(experts[source_i],fused[source_i],s['full_history'],weights[source_i],penalties[source_i],counts)
        target=s['target'];oldrank=int(rr[target]);assert oldrank==reference[s['user_id']]
        n=len(ids);sl=slice(p,p+n)
        arrays['score'][sl]=x
        arrays['evidence'][sl]=engine.paired(ids,b,s['full_history'])
        arrays['shuffled'][sl]=engine.paired(ids,b,samples[int(ds[local])]['full_history'])
        item_ids[sl]=ids;labels[sl]=(ids==target).astype(np.int8)-int(b==target)
        row_user[sl]=local;train_weight[sl]=1/max(n,1)
        cohort='train' if training else ('previous512' if s['previously_evaluated'] else
            'primary' if s['training_holdout_role']=='none' else 'seen_fit_user_later_event')
        record=dict(user_id=s['user_id'],target=target,cohort=cohort,fold=fold(s['user_id'],domain),
            old_rank=oldrank,boundary=b,candidates=n,start=p,end=p+n,donor_user=samples[int(ds[local])]['user_id'],
            expert_target_top10=int((er[:,target]<=10).sum()),top9=' '.join(map(str,ten[:9])))
        records.append(record)
        if local in (0,len(samples)//2,len(samples)-1):
            engine.check(list(ids[:2])+[b],s['full_history'])
            # Target is absent from both callable signatures; swap outcomes only after computing features.
            check_ids,check_b,check_x,*_=proposals(experts[source_i],fused[source_i],s['full_history'],weights[source_i],penalties[source_i],counts)
            np.testing.assert_array_equal(ids,check_ids);assert b==check_b;np.testing.assert_array_equal(x,check_x)
        p+=n;offsets.append(p)
        if (local+1)%2048==0:
            print(domain,kind,'features',local+1,'/',len(samples),'pairs',p,flush=True)
            write(OUT/'status.json',dict(state='RUNNING',domain=domain,stage=kind+'_features',completed=local+1,total=len(samples),pid=os.getpid()))
    for a in arrays.values():a.flush()
    np.savez(out/'pairs.npz',items=item_ids[:p],labels=labels[:p],row_user=row_user[:p],weight=train_weight[:p],offsets=np.array(offsets))
    csv_write(out/'users.csv',records)
    info=dict(users=len(samples),pairs=p,capacity=capacity,positive_pairs=int((labels[:p]>0).sum()),
        negative_pairs=int((labels[:p]<0).sum()),neutral_pairs=int((labels[:p]==0).sum()),
        users_with_positive_proposal=sum(any(labels[r['start']:r['end']]>0) for r in records),
        boundary_target_users=sum(r['boundary']==r['target'] for r in records),
        donor_changed_users=sum(s['user_id']!=samples[int(ds[i])]['user_id'] for i,s in enumerate(samples)),
        unique_users=len({s['user_id'] for s in samples}),source_hash_checks_passed=True,original_ranks_exact=True,
        cohorts={c:sum(r['cohort']==c for r in records) for c in sorted({r['cohort'] for r in records})})
    write(out/'metadata.json',info)
    return dict(x={k:v[:p] for k,v in arrays.items()},items=item_ids[:p],labels=labels[:p],row_user=row_user[:p],weight=train_weight[:p],users=records,info=info)


def matrix(data,arm):
    return data['x']['score'] if arm=='score' else np.concatenate([data['x']['score'],data['x'][arm]],axis=1)


def save_model(path,model,fit_users):
    np.savez(path,**model,fit_users=np.array(fit_users))


def best_choices(data,predictions):
    out={}
    for arm,values in predictions.items():
        rows=[]
        for user in data['users']:
            start,end=user['start'],user['end']
            if start==end:rows.append((0,float('-inf')));continue
            assert np.isfinite(values[start:end]).all()
            # IDs are sorted; argmax's first occurrence implements ID tie break.
            ix=start+int(values[start:end].argmax());rows.append((int(data['items'][ix]),float(values[ix])))
        out[arm]=rows
    return out


def evaluate(data,predictions,cohort):
    indices=[i for i,r in enumerate(data['users']) if cohort=='all' or r['cohort']==cohort]
    rows=[data['users'][i] for i in indices];n=len(rows);old=np.array([r['old_rank'] for r in rows])
    choices=best_choices(data,predictions);available=np.array([r['candidates']>0 for r in rows]);results={};detail=[]
    utilities={'original':np.where(old<=10,1/np.log2(old+1),0.)}
    policies={}
    for arm in ARMS:
        items=np.array([choices[arm][i][0] for i in indices]);scores=np.array([choices[arm][i][1] for i in indices])
        for policy in ('positive','budget05','budget10','budget20'):
            action=available&(scores>0) if policy=='positive' else np.zeros(n,bool)
            if policy!='positive':
                count=int(int(policy[-2:])/100*int(available.sum()))
                ranked=sorted(np.flatnonzero(available),key=lambda i:(-scores[i],rows[i]['user_id']))
                action[ranked[:count]]=True;assert action.sum()==count
            new=old.copy()
            target=np.array([r['target'] for r in rows]);boundary=np.array([r['boundary'] for r in rows])
            new[action&(items==target)]=10
            new[action&(boundary==target)]=ABSENT
            gains=int(((old>10)&(new<=10)).sum());losses=int(((old<=10)&(new>10)).sum())
            key=arm+'_'+policy;utility=np.where(new<=10,1/np.log2(new+1),0.);utilities[key]=utility
            results[key]=dict(actions=int(action.sum()),gained10=gains,lost10=losses,neutral_actions=int(action.sum())-gains-losses,
                net_hits=gains-losses,H10=float((new<=10).mean()),N10=float(utility.mean()),delta_N10=float((utility-utilities['original']).mean()),
                delta_H10=(gains-losses)/n)
            policies[key]=(items,action,new,scores)
    # One common user bootstrap per cohort preserves paired covariance across policies.
    keys=list(utilities);matrix_u=np.stack([utilities[k] for k in keys],axis=1)
    rng=np.random.default_rng(20260925);boot=[]
    for _ in range(20):
        ix=rng.integers(0,n,size=(100,n));boot.append(matrix_u[ix].mean(1))
    boot=np.concatenate(boot);key_i={k:i for i,k in enumerate(keys)}
    for key in results:
        results[key]['delta_N10_ci95']=np.quantile(boot[:,key_i[key]]-boot[:,0],[.025,.975]).tolist()
    comparisons={}
    for policy in ('positive','budget05','budget10','budget20'):
        for control in ('score','shuffled'):
            a='evidence_'+policy;b=control+'_'+policy
            comparisons[a+'_vs_'+b]=dict(delta_N10=float((utilities[a]-utilities[b]).mean()),
                ci95=np.quantile(boot[:,key_i[a]]-boot[:,key_i[b]],[.025,.975]).tolist(),
                net_hit_difference=results[a]['net_hits']-results[b]['net_hits'])
    for i,r in enumerate(rows):
        record=dict(user_id=r['user_id'],target=r['target'],old_rank=r['old_rank'],boundary=r['boundary'],candidates=r['candidates'])
        for key,(items,action,new,scores) in policies.items():
            record[key+'_item']=int(items[i]);record[key+'_action']=bool(action[i]);record[key+'_rank']=int(new[i])
        detail.append(record)
    return dict(users=n,eligible_users=int(available.sum()),old_H10=float((old<=10).mean()),
        old_N10=float(utilities['original'].mean()),policies=results,comparisons=comparisons),detail


def domain_run(domain,manifest):
    folder=SOURCE/domain;cfg=json.loads((folder/'data_config.json').read_text())
    catalog=json.loads((folder/'catalog.json').read_text())
    paths=[folder/'catalog.json',folder/'data_config.json',folder/'frequencies.npy',ROOT/cfg['embedding'],ROOT/cfg['itemcf']]
    source_manifest=json.loads((SOURCE/'manifest.json').read_text())['sha256']
    for path in paths:
        rel=str(path.relative_to(ROOT));digest=sha(path);manifest[rel]=digest
        assert digest==source_manifest[rel],rel
    with np.load(ROOT/cfg['embedding'],allow_pickle=False) as a:
        assert a['item_ids'].tolist()==catalog['items'][1:];embedding=a['embeddings']
    engine=Evidence(catalog,embedding,json.loads((ROOT/cfg['itemcf']).read_text()))
    train=feature_data(domain,'train',engine,manifest)
    valid=feature_data(domain,'validation',engine,manifest)
    train_users={r['user_id'] for r in train['users']}
    assert not train_users&{r['user_id'] for r in valid['users'] if r['cohort']=='primary'}
    expected=15406 if domain=='Toys' else 17766
    assert valid['info']['cohorts']['primary']==expected
    fold_pairs=np.array([train['users'][i]['fold'] for i in train['row_user']])
    train_predictions={};valid_predictions={}
    for arm in ARMS:
        x=matrix(train,arm);oof=np.full(len(x),np.nan)
        for k in range(5):
            fit=fold_pairs!=k;held=~fit
            model=ridge_fit(x[fit],train['labels'][fit],train['weight'][fit])
            fit_users=[r['user_id'] for r in train['users'] if r['fold']!=k]
            assert not set(fit_users)&{r['user_id'] for r in train['users'] if r['fold']==k}
            save_model(OUT/domain/(arm+'_fold'+str(k)+'.npz'),model,fit_users)
            oof[held]=predict(model,x[held])
        assert np.isfinite(oof).all();train_predictions[arm]=oof
        model=ridge_fit(x,train['labels'],train['weight'])
        save_model(OUT/domain/(arm+'_final.npz'),model,sorted(train_users))
        valid_x=matrix(valid,arm);pred=np.zeros(len(valid_x))
        for start in range(0,len(pred),10000):pred[start:start+10000]=predict(model,valid_x[start:start+10000])
        valid_predictions[arm]=pred
        del x,valid_x
        print(domain,arm,'fit complete',flush=True)
    np.savez(OUT/domain/'train_oof_predictions.npz',**train_predictions)
    np.savez(OUT/domain/'validation_predictions.npz',**valid_predictions)
    output=dict(train_data=train['info'],validation_data=valid['info'],cohorts={},fitted_labels='ranker_fit_only',primary_users_disjoint=True)
    for name,data,pred,cohort in [('train_oof',train,train_predictions,'all')]+[(c,valid,valid_predictions,c) for c in ('primary','previous512','seen_fit_user_later_event','all')]:
        print(domain,'statistics',name,flush=True)
        metrics,rows=evaluate(data,pred,cohort);output['cohorts'][name]=metrics
        csv_write(OUT/domain/(name+'_decisions.csv'),rows)
    write(OUT/domain/'summary.json',output)
    return output


def main():
    started=time.monotonic();OUT.mkdir(exist_ok=False)
    (OUT/'protocol.md').write_text(PROTOCOL.read_text())
    paths=[Path(__file__),Path(__file__).with_name('selective_evidence_core.py'),Path(__file__).with_name('analyze_history_evidence.py'),Path(__file__).with_name('analyze_expert_signal_loss.py'),PROTOCOL]
    manifest={str(p.relative_to(ROOT)):sha(p) for p in paths}
    for p in paths[:4]:(OUT/p.name).write_text(p.read_text())
    write(OUT/'features.json',dict(score=BASE_NAMES,evidence=EVIDENCE_NAMES,lambda_l2=1.0))
    write(OUT/'status.json',dict(state='RUNNING',stage='checks',pid=os.getpid(),GRAM_optimizer_steps=0,expert_optimizer_steps=0,gpu_inferences=0))
    try:
        checks=model_checks();write(OUT/'numerical_checks.json',checks)
        domains={}
        for d in ('Toys','Beauty'):domains[d]=domain_run(d,manifest)
        supported=all(r['cohorts']['primary']['policies']['evidence_budget10']['net_hits']>0 and
            all(r['cohorts']['primary']['comparisons']['evidence_budget10_vs_'+c+'_budget10']['net_hit_difference']>0 for c in ('score','shuffled')) for r in domains.values())
        summary=dict(state='COMPLETED',seconds=time.monotonic()-started,domains=domains,
            primary_direction_consistent=supported,independent_confirmation=False,exploratory=True,
            GRAM_optimizer_steps=0,expert_optimizer_steps=0,gpu_inferences=0,statistical_ridge_fits=36,test_read=False,automatic_gpu_launch=False)
        write(OUT/'manifest.json',manifest);write(OUT/'summary.json',summary)
        write(OUT/'status.json',{k:v for k,v in summary.items() if k!='domains'})
        print(json.dumps({k:v for k,v in summary.items() if k!='domains'}),flush=True)
    except BaseException as e:
        (OUT/'failure.txt').write_text(traceback.format_exc());write(OUT/'status.json',dict(state='FAILED',error=repr(e),pid=os.getpid()));raise


if __name__=='__main__':main()
