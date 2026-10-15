"""CPU-only: fixed evidence ablations and near-boundary supervision expansion."""
import csv
import hashlib
import json
import os
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

from evidence_decomposition_core import GROUPS,SPECS,evaluate,feature_matrix,load_data
from selective_evidence_core import BASE_NAMES,EVIDENCE_NAMES,Evidence,predict,ridge_fit
from analyze_expert_signal_loss import full_ranking

ROOT=Path(__file__).resolve().parents[2]
PRIOR=ROOT/'artifacts/phase21/selective_evidence_20260925_v1'
RAW=ROOT/'artifacts/phase20/lime_clean_benefit_20260922_v1'
DESIGN=ROOT/'artifacts/phase20/lime_clean_training_design_20260920_v1'
OUT=ROOT/'artifacts/phase21/evidence_decomposition_20260925_v1'
PROTOCOL=ROOT/'plan/第二十一阶段/21.12_监督范围扩展与依据分组消融_20260925.md'


def write(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def use(path,manifest,expected=None):
    digest=sha(path);manifest[str(path.relative_to(ROOT))]=digest
    if expected is not None:assert digest==expected,str(path)


def lineage(domain,original,manifest):
    receipt=json.loads((DESIGN/'summary.json').read_text());values={}
    for name in ('terminal_holdout.jsonl','upstream_sequences_without_terminal.jsonl','upstream_train_without_terminal.jsonl'):
        p=DESIGN/domain/name;use(p,manifest,receipt['output_sha256'][str(p.relative_to(ROOT))])
        values[name]=[json.loads(line) for line in p.read_text().splitlines()]
    sequences={r['user_id']:r['history'] for r in values['upstream_sequences_without_terminal.jsonl']}
    held={r['user_id']:r for r in values['terminal_holdout.jsonl']}
    rebuilt={};counts=Counter()
    for row in values['upstream_train_without_terminal.jsonl']:
        u=row['user_id'];history=row['history']
        if u not in rebuilt:assert len(history)==1;rebuilt[u]=history.copy()
        assert rebuilt[u][-20:]==history;rebuilt[u].append(row['target']);counts[u]+=1
    assert all(rebuilt.get(u,s[:1])==s and counts[u]==len(s)-1 for u,s in sequences.items())
    samples=json.loads((RAW/domain/'samples.json').read_text());by_user={s['user_id']:s for s in samples}
    assert {r['user_id'] for r in original['users']}=={u for u,h in held.items() if h['role']=='ranker_fit'}
    for r in original['users']:
        u=r['user_id'];h=held[u];s=by_user[u]
        assert r['target']==h['target']==s['target'] and s['full_history']==sequences[u] and h['history']==sequences[u][-20:]
    rankcounts=Counter(r['old_rank'] for r in original['users'])
    return dict(held_users=len(held),fit_users=len(original['users']),gate_users=sum(h['role']=='gate_calibration' for h in held.values()),
        clean_upstream_rows=len(values['upstream_train_without_terminal.jsonl']),sequence_reconstruction_exact=True,held_event_boundary_verified=True,
        target_rank_counts_1_to_10={str(i):rankcounts[i] for i in range(1,11)},
        harm_users_rank10=rankcounts[10],harm_users_rank6_to_10=sum(rankcounts[i] for i in range(6,11)),
        harm_users_rank1_to_10=sum(rankcounts[i] for i in range(1,11)),new_independent_users=0,new_interaction_labels=0)


def expand(domain,old,manifest):
    folder=RAW/domain;dest=OUT/domain/'expanded';dest.mkdir()
    cfg=json.loads((folder/'data_config.json').read_text());catalog=json.loads((folder/'catalog.json').read_text())
    source_manifest=json.loads((RAW/'manifest.json').read_text())['sha256']
    for p in (folder/'catalog.json',folder/'data_config.json',folder/'samples.json',folder/'frequencies.npy',ROOT/cfg['embedding'],ROOT/cfg['itemcf']):
        use(p,manifest,source_manifest[str(p.relative_to(ROOT))])
    raw_receipt=json.loads((PRIOR/'manifest.json').read_text())
    for name in ('experts.npy','lime.npy'):use(folder/name,manifest,raw_receipt[str((folder/name).relative_to(ROOT))])
    samples=json.loads((folder/'samples.json').read_text());source_ix={s['user_id']:i for i,s in enumerate(samples)}
    with np.load(ROOT/cfg['embedding'],allow_pickle=False) as z:
        assert z['item_ids'].tolist()==catalog['items'][1:];embedding=z['embeddings']
    engine=Evidence(catalog,embedding,json.loads((ROOT/cfg['itemcf']).read_text()))
    scores=np.load(folder/'experts.npy',mmap_mode='r');fused=np.load(folder/'lime.npy',mmap_mode='r');frequency=np.load(folder/'frequencies.npy')
    n=5*len(old['items']);arrays={k:np.lib.format.open_memmap(dest/(k+'.npy'),mode='w+',dtype='float32',shape=(n,d)) for k,d in [('score',32),('evidence',57),('shuffled',57)]}
    items=np.zeros(n,np.int32);labels=np.zeros(n,np.int8);owner=np.zeros(n,np.int32);weight=np.zeros(n,float)
    b_items=np.zeros(n,np.int32);b_ranks=np.zeros(n,np.int8);offsets=[0];p=0;metadata=[]
    for user_i,r in enumerate(old['users']):
        i=source_ix[r['user_id']];s=samples[i];a,z=r['start'],r['end'];c=old['items'][a:z];count=z-a
        eligible=np.ones(fused.shape[1],bool);eligible[np.asarray(s['full_history'][-20:])-1]=False
        order,fr=full_ranking(fused[i],eligible);er=np.stack([full_ranking(v,eligible)[1] for v in scores[i]])
        bs=order[5:10];assert bs[-1]==r['boundary'] and int(fr[r['target']])==r['old_rank']
        assert not set(c)&set(order[:10]);donor=samples[source_ix[r['donor_user']]]
        ev={'evidence':engine.item_features(bs,s['full_history']),'shuffled':engine.item_features(bs,donor['full_history'])}
        start=p
        for j,b in enumerate(bs):
            rank_b=j+6;sl=slice(p,p+count);x=np.array(old['x']['score'][a:z],copy=True)
            x[:,3:6]=scores[i,:,b-1];x[:,6:9]=x[:,:3]-x[:,3:6];x[:,12:15]=1/er[:,b]
            x[:,19]=fused[i,b-1];x[:,20]=x[:,18]-x[:,19];x[:,22]=fused[i,b-1]-fused[i,order[rank_b]-1]
            x[:,23]=fused[i,order[0]-1]-fused[i,b-1];x[:,29]=np.log1p(frequency[b]);x[:,31]=1/rank_b
            arrays['score'][sl]=x
            for kind in ('evidence','shuffled'):
                av=old['x'][kind][a:z,:19];bv=np.broadcast_to(ev[kind][j],av.shape)
                arrays[kind][sl]=np.concatenate([av,bv,av-bv],axis=1)
            items[sl]=c;labels[sl]=(c==r['target']).astype(np.int8)-int(b==r['target']);owner[sl]=user_i
            weight[sl]=1/max(5*count,1);b_items[sl]=b;b_ranks[sl]=rank_b
            if rank_b==10:
                for kind in arrays:np.testing.assert_allclose(arrays[kind][sl],old['x'][kind][a:z],atol=1e-7,rtol=0.)
                np.testing.assert_array_equal(labels[sl],old['labels'][a:z]);np.testing.assert_allclose(weight[sl]*5,old['weight'][a:z],atol=1e-15)
            p+=count
        rr=dict(r,start=start,end=p,candidates=5*count);metadata.append(rr);offsets.append(p)
        if (user_i+1)%512==0:
            print(domain,'expanded training features',user_i+1,'/',len(old['users']),flush=True)
            write(OUT/'status.json',dict(state='RUNNING',domain=domain,stage='expand_training',completed=user_i+1,pid=os.getpid()))
    assert p==n
    for v in arrays.values():v.flush()
    np.savez(dest/'pairs.npz',items=items,labels=labels,row_user=owner,weight=weight,offsets=np.array(offsets),b_items=b_items,b_ranks=b_ranks)
    with (dest/'users.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,list(metadata[0]));w.writeheader();w.writerows(metadata)
    audit=dict(users=len(metadata),pairs=n,positive_pairs=int((labels==1).sum()),negative_pairs=int((labels==-1).sum()),
        neutral_pairs=int((labels==0).sum()),positive_users=len(set(owner[labels==1].tolist())),harm_users=len(set(owner[labels==-1].tolist())),
        rank10_features_agree=True,source_targets_unchanged=True,per_user_weight_one=True)
    for j in range(len(metadata)):assert abs(weight[offsets[j]:offsets[j+1]].sum()-(1 if offsets[j+1]>offsets[j] else 0))<1e-12
    write(dest/'audit.json',audit)
    return dict(users=metadata,x=arrays,items=items,labels=labels,row_user=owner,weight=weight,offsets=np.array(offsets)),audit


def run_domain(domain,manifest):
    folder=OUT/domain;folder.mkdir()
    for kind in ('train','validation'):
        for name in ('users.csv','pairs.npz','score.npy','evidence.npy','shuffled.npy'):
            use(PRIOR/domain/kind/name,manifest)
    train=load_data(PRIOR/domain/'train');valid=load_data(PRIOR/domain/'validation')
    audit=lineage(domain,train,manifest);expanded,exp_audit=expand(domain,train,manifest)
    assert exp_audit['harm_users']==audit['harm_users_rank6_to_10']
    fit_users={r['user_id'] for r in train['users']};assert not fit_users&{r['user_id'] for r in valid['users'] if r['cohort']=='primary'}
    original_folds=np.array([train['users'][i]['fold'] for i in train['row_user']]);train_pred={};valid_pred={};max_old_error=0.
    for arm,spec in SPECS.items():
        source=expanded if spec[0]=='expanded' else train
        x=feature_matrix(source,spec);folds=np.array([source['users'][i]['fold'] for i in source['row_user']])
        eval_x=feature_matrix(train,spec);oof=np.full(len(train['items']),np.nan)
        for k in range(5):
            fit=folds!=k;held=original_folds==k
            model=ridge_fit(x[fit],source['labels'][fit],source['weight'][fit])
            users=[r['user_id'] for r in train['users'] if r['fold']!=k]
            np.savez(folder/(arm+'_fold'+str(k)+'.npz'),**model,fit_users=np.array(users))
            oof[held]=predict(model,eval_x[held])
        model=ridge_fit(x,source['labels'],source['weight']);np.savez(folder/(arm+'_final.npz'),**model,fit_users=np.array(sorted(fit_users)))
        vx=feature_matrix(valid,spec);vp=np.zeros(len(valid['items']))
        for start in range(0,len(vp),10000):vp[start:start+10000]=predict(model,vx[start:start+10000])
        assert np.isfinite(vp).all() and np.isfinite(oof).all()
        train_pred[arm]=oof;valid_pred[arm]=vp
        old_key={'score':'score','full':'evidence','shuffled_full':'shuffled'}.get(arm)
        if old_key:
            for label,new in [('train_oof',oof),('validation',vp)]:
                with np.load(PRIOR/domain/(label+'_predictions.npz'),allow_pickle=False) as z:old=z[old_key]
                error=float(np.max(np.abs(new-old)));max_old_error=max(max_old_error,error)
                np.testing.assert_allclose(new,old,atol=1e-9,rtol=0.)
        print(domain,arm,'fit complete',flush=True)
        write(OUT/'status.json',dict(state='RUNNING',domain=domain,stage='fit',arm=arm,pid=os.getpid()))
        del x,eval_x,vx
    np.savez(folder/'train_oof_predictions.npz',**train_pred);np.savez(folder/'validation_predictions.npz',**valid_pred)
    cohorts={}
    for name,data,pred in [('train_oof',train,train_pred)]+[(n,valid,valid_pred) for n in ('primary','previous512','seen_fit_user_later_event','all')]:
        cohorts[name]=evaluate(data,pred,name,folder);print(domain,'statistics',name,flush=True)
    previous=json.loads((PRIOR/domain/'summary.json').read_text())['cohorts']
    for name,cohort in cohorts.items():
        assert cohort['users']==previous[name]['users']
        for arm,oldarm in [('score','score'),('full','evidence'),('shuffled_full','shuffled')]:
            for policy in ('positive','budget05','budget10','budget20'):
                a=cohort['policies'][arm+'_'+policy];b=previous[name]['policies'][oldarm+'_'+policy]
                assert all(a[k]==b[k] for k in ('actions','gained10','lost10','neutral_actions','net_hits'))
    result=dict(lineage_audit=audit,expanded_audit=exp_audit,cohorts=cohorts,max_reproduced_prediction_error=max_old_error,
        original_three_arms_exact_decisions=True,validation_labels_used_for_fit=False)
    write(folder/'summary.json',result);return result


def main():
    started=time.monotonic();OUT.mkdir(exist_ok=False)
    (OUT/'protocol.md').write_text(PROTOCOL.read_text())
    scripts=[Path(__file__),Path(__file__).with_name('evidence_decomposition_core.py'),Path(__file__).with_name('selective_evidence_core.py'),Path(__file__).with_name('analyze_history_evidence.py'),Path(__file__).with_name('analyze_expert_signal_loss.py')]
    manifest={}
    for p in scripts+[PROTOCOL]:use(p,manifest)
    for p in scripts:(OUT/p.name).write_text(p.read_text())
    assert sorted(sum(GROUPS.values(),[]))==list(range(19))
    write(OUT/'features.json',dict(base=BASE_NAMES+['b_fused_reciprocal_rank'],evidence=EVIDENCE_NAMES,groups=GROUPS,arms=SPECS,lambda_l2=1.0))
    write(OUT/'status.json',dict(state='RUNNING',stage='lineage',pid=os.getpid()))
    try:
        domains={}
        for d in ('Toys','Beauty'):domains[d]=run_domain(d,manifest)
        result=dict(state='COMPLETED',seconds=time.monotonic()-started,domains=domains,statistical_ridge_fits=144,
            new_independent_training_users=0,new_interaction_labels=0,GRAM_optimizer_steps=0,expert_optimizer_steps=0,
            gpu_inferences=0,test_read=False,automatic_gpu_launch=False,exploratory=True,independent_confirmation=False)
        write(OUT/'manifest.json',manifest);write(OUT/'summary.json',result)
        write(OUT/'status.json',{k:v for k,v in result.items() if k!='domains'})
        print(json.dumps({k:v for k,v in result.items() if k!='domains'}),flush=True)
    except BaseException as e:
        (OUT/'failure.txt').write_text(traceback.format_exc());write(OUT/'status.json',dict(state='FAILED',error=repr(e),pid=os.getpid()));raise


if __name__=='__main__':main()
