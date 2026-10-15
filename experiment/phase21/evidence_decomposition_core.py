"""Frozen feature groups and evaluation for supervision/evidence decomposition."""
import csv
import json
from pathlib import Path

import numpy as np

GROUPS={'lexical':list(range(6)),'cf':list(range(6,12)),'text':list(range(12,19))}
SPECS={
    'score':('original','score',[]),
    'full':('original','evidence',list(range(19))),
    'shuffled_full':('original','shuffled',list(range(19))),
    **{'only_'+g:('original','evidence',ix) for g,ix in GROUPS.items()},
    **{'without_'+g:('original','evidence',[i for i in range(19) if i not in ix]) for g,ix in GROUPS.items()},
    'expanded_score':('expanded','score',[]),
    'expanded_full':('expanded','evidence',list(range(19))),
    'expanded_shuffled_full':('expanded','shuffled',list(range(19))),
}
POLICIES=('budget05','budget10','budget20','positive')
CONTRASTS=([('only_'+g,'score') for g in GROUPS]+[('full','without_'+g) for g in GROUPS]
    +[('expanded_full','full'),('expanded_score','score'),('expanded_shuffled_full','shuffled_full'),
      ('expanded_full','expanded_score'),('expanded_full','expanded_shuffled_full'),('full','score'),('full','shuffled_full')])


def read_users(path):
    with Path(path).open() as f:rows=list(csv.DictReader(f))
    for r in rows:
        for k in ('target','fold','old_rank','boundary','candidates','start','end','expert_target_top10'):r[k]=int(r[k])
    return rows


def load_data(folder):
    users=read_users(folder/'users.csv')
    with np.load(folder/'pairs.npz',allow_pickle=False) as z:pairs={k:z[k] for k in z.files}
    n=len(pairs['items'])
    x={k:np.load(folder/(k+'.npy'),mmap_mode='r',allow_pickle=False)[:n] for k in ('score','evidence','shuffled')}
    if x['score'].shape[1]==31:
        x['score']=np.column_stack([x['score'],np.full(n,.1,np.float32)])
    assert x['score'].shape[1]==32
    return dict(users=users,x=x,**pairs)


def feature_matrix(data,spec):
    _,kind,group=spec
    if not group:return data['x']['score']
    columns=[i+19*j for j in range(3) for i in group]
    return np.concatenate([data['x']['score'],data['x'][kind][:,columns]],axis=1)


def evaluate(data,predictions,cohort,out):
    arms=list(predictions)
    ix=[i for i,r in enumerate(data['users']) if cohort in ('all','train_oof') or r['cohort']==cohort]
    rows=[data['users'][i] for i in ix];n=len(rows)
    old=np.array([r['old_rank'] for r in rows]);target=np.array([r['target'] for r in rows]);b=np.array([r['boundary'] for r in rows])
    users=np.array([r['user_id'] for r in rows]);available=np.array([r['candidates']>0 for r in rows])
    items=np.zeros((len(arms),n),int);scores=np.full((len(arms),n),-np.inf)
    for k,arm in enumerate(arms):
        pred=predictions[arm]
        for j,r in enumerate(rows):
            if r['start']==r['end']:continue
            a=r['start']+int(pred[r['start']:r['end']].argmax())
            items[k,j]=data['items'][a];scores[k,j]=pred[a]
    actions=np.zeros((len(POLICIES),len(arms),n),bool)
    delta=np.zeros(actions.shape,np.int8)
    for pi,policy in enumerate(POLICIES):
        for ai,arm in enumerate(arms):
            if policy=='positive':act=available&(scores[ai]>0)
            else:
                count=int(int(policy[-2:])/100*int(available.sum()))
                order=sorted(np.flatnonzero(available),key=lambda j:(-scores[ai,j],users[j]))
                act=np.zeros(n,bool);act[order[:count]]=True;assert act.sum()==count
            actions[pi,ai]=act
            delta[pi,ai]=act.astype(np.int8)*((items[ai]==target).astype(np.int8)-(b==target).astype(np.int8))
    old_u=np.where(old<=10,1/np.log2(old+1),0.);unit=1/np.log2(11)
    result=dict(users=n,eligible_users=int(available.sum()),old_H10=float((old<=10).mean()),old_N10=float(old_u.mean()),policies={},contrasts={})
    for ai,arm in enumerate(arms):
        for pi,policy in enumerate(POLICIES):
            d=delta[pi,ai];key=arm+'_'+policy;g=int((d==1).sum());l=int((d==-1).sum());a=int(actions[pi,ai].sum())
            result['policies'][key]=dict(actions=a,gained10=g,lost10=l,neutral_actions=a-g-l,net_hits=g-l,
                H10=result['old_H10']+float(d.mean()),N10=result['old_N10']+float(d.mean())*unit,delta_N10=float(d.mean())*unit)
    for policy in POLICIES:
        for a,barm in CONTRASTS:
            x=result['policies'][a+'_'+policy];y=result['policies'][barm+'_'+policy]
            result['contrasts'][a+'_vs_'+barm+'_'+policy]=dict(net_hit_difference=x['net_hits']-y['net_hits'],delta_N10=x['delta_N10']-y['delta_N10'])
    if cohort in ('primary','train_oof'):
        d=delta[POLICIES.index('budget10')].T
        rng=np.random.default_rng(20260925);boot=[]
        for _ in range(20):
            take=rng.integers(0,n,size=(100,n));boot.append(d[take].mean(1)*unit)
        boot=np.concatenate(boot)
        for ai,arm in enumerate(arms):result['policies'][arm+'_budget10']['ci95']=np.quantile(boot[:,ai],[.025,.975]).tolist()
        for a,barm in CONTRASTS:
            result['contrasts'][a+'_vs_'+barm+'_budget10']['ci95']=np.quantile(boot[:,arms.index(a)]-boot[:,arms.index(barm)],[.025,.975]).tolist()
    np.savez_compressed(out/(cohort+'_decisions.npz'),users=users,old_rank=old,target=target,boundary=b,
        arms=np.array(arms),policies=np.array(POLICIES),items=items,predicted_benefit=scores,actions=actions,delta=delta)
    with (out/(cohort+'_budget10.csv')).open('w',newline='') as f:
        names=['user_id','target','old_rank','boundary']+[arm+'_'+suffix for arm in arms for suffix in ('item','action','hit_change')]
        writer=csv.DictWriter(f,names);writer.writeheader()
        for j,r in enumerate(rows):
            row={k:r[k] for k in names[:4]}
            for ai,arm in enumerate(arms):
                row.update({arm+'_item':int(items[ai,j]),arm+'_action':bool(actions[1,ai,j]),arm+'_hit_change':int(delta[1,ai,j])})
            writer.writerow(row)
    return result
