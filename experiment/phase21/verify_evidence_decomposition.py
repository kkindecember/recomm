"""Independent accounting of expansion, heldout folds, prediction and decisions."""
import hashlib
import json
from pathlib import Path

import numpy as np

from evidence_decomposition_core import GROUPS,SPECS,CONTRASTS,load_data

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/phase21/evidence_decomposition_20260925_v1'
PRIOR=ROOT/'artifacts/phase21/selective_evidence_20260925_v1'


def check_domain(domain,summary):
    folder=OUT/domain;train=load_data(PRIOR/domain/'train');valid=load_data(PRIOR/domain/'validation');expanded=load_data(folder/'expanded')
    with np.load(folder/'expanded/pairs.npz',allow_pickle=False) as z:bs=z['b_items'];ranks=z['b_ranks']
    assert sorted(sum(GROUPS.values(),[]))==list(range(19))
    max_error=0.;exact=True
    for i,u in enumerate(train['users']):
        a,b=u['start'],u['end'];c=train['items'][a:b];n=b-a
        for offset,rank_b in enumerate(range(6,11)):
            lo=5*a+offset*n;hi=lo+n
            np.testing.assert_array_equal(expanded['items'][lo:hi],c)
            np.testing.assert_array_equal(ranks[lo:hi],np.full(n,rank_b))
            np.testing.assert_array_equal(expanded['row_user'][lo:hi],np.full(n,i))
            np.testing.assert_array_equal(expanded['labels'][lo:hi],(c==u['target']).astype(np.int8)-(bs[lo:hi]==u['target']).astype(np.int8))
            np.testing.assert_allclose(expanded['weight'][lo:hi]*5,train['weight'][a:b],atol=1e-15)
            if rank_b==10:
                np.testing.assert_array_equal(bs[lo:hi],np.full(n,u['boundary']))
                for kind in ('score','evidence','shuffled'):
                    x=expanded['x'][kind][lo:hi];y=train['x'][kind][a:b]
                    if x.size:
                        error=float(np.max(np.abs(x-y)));max_error=max(max_error,error);exact=exact and np.array_equal(x,y)
                    np.testing.assert_allclose(x,y,atol=1e-7,rtol=0.)
    for kind in ('evidence','shuffled'):
        x=expanded['x'][kind]
        np.testing.assert_array_equal(x[:,38:],x[:,:19]-x[:,19:38])
    harm_users=len(set(expanded['row_user'][expanded['labels']<0].tolist()))
    assert harm_users==summary['lineage_audit']['harm_users_rank6_to_10']
    fitted={u['user_id'] for u in train['users']};assert not fitted&{u['user_id'] for u in valid['users'] if u['cohort']=='primary'}
    manual_checks=0;decision_count=0
    for dataset,data,cohorts in [('train_oof',train,('train_oof',)),('validation',valid,('primary','previous512','seen_fit_user_later_event','all'))]:
        with np.load(folder/(dataset+'_predictions.npz'),allow_pickle=False) as z:pred={k:z[k] for k in z.files}
        assert set(pred)==set(SPECS)
        for arm,(regime,kind,group) in SPECS.items():
            assert np.isfinite(pred[arm]).all()
            columns=[i+19*j for j in range(3) for i in group]
            for k in (range(5) if dataset=='train_oof' else (None,)):
                name=arm+('_fold'+str(k) if k is not None else '_final')+'.npz'
                with np.load(folder/name,allow_pickle=False) as model:
                    expected=fitted if k is None else {u['user_id'] for u in train['users'] if u['fold']!=k}
                    assert set(model['fit_users'].tolist())==expected
                    options=[u for u in data['users'] if u['end']>u['start'] and (k is None or u['fold']==k)]
                    for u in (options[0],options[len(options)//2],options[-1]):
                        j=u['start'];x=data['x']['score'][j].tolist()
                        if columns:x+=data['x'][kind][j,columns].tolist()
                        value=float(model['intercept'])+sum((v-m)/sd*c for v,m,sd,c in zip(x,model['mean'],model['std'],model['coef']))
                        assert abs(value-pred[arm][j])<1e-9;manual_checks+=1
        for cohort in cohorts:
            rows=[u for u in data['users'] if cohort in ('all','train_oof') or u['cohort']==cohort]
            with np.load(folder/(cohort+'_decisions.npz'),allow_pickle=False) as z:saved={k:z[k] for k in z.files}
            assert saved['users'].tolist()==[u['user_id'] for u in rows]
            eligible=[j for j,u in enumerate(rows) if u['end']>u['start']]
            arms=saved['arms'].tolist();policies=saved['policies'].tolist();original=np.array([u['old_rank'] for u in rows])
            old_utility=np.where(original<=10,1/np.log2(original+1),0.)
            for ai,arm in enumerate(arms):
                best={};values={}
                for j in eligible:
                    u=rows[j];ix=min(range(u['start'],u['end']),key=lambda t:(-pred[arm][t],data['items'][t]))
                    best[j]=int(data['items'][ix]);values[j]=float(pred[arm][ix])
                    assert best[j]==saved['items'][ai,j] and values[j]==saved['predicted_benefit'][ai,j]
                for pi,policy in enumerate(policies):
                    if policy=='positive':acted={j for j in eligible if values[j]>0}
                    else:
                        count=int(len(eligible)*int(policy[-2:])/100)
                        acted=set(sorted(eligible,key=lambda j:(-values[j],rows[j]['user_id']))[:count])
                    changes=[];new=[]
                    for j,u in enumerate(rows):
                        assert saved['actions'][pi,ai,j]==(j in acted)
                        change=(int(best[j]==u['target'])-int(u['boundary']==u['target'])) if j in acted else 0
                        assert change==saved['delta'][pi,ai,j]
                        if change==1:assert u['old_rank']>10
                        if change==-1:assert u['old_rank']==10
                        rr=10 if change==1 else 2**30 if change==-1 else u['old_rank']
                        changes.append(change);new.append(1/np.log2(rr+1) if rr<=10 else 0.);decision_count+=1
                    expected=summary['cohorts'][cohort]['policies'][arm+'_'+policy]
                    assert (len(acted),changes.count(1),changes.count(-1),sum(changes))==(expected['actions'],expected['gained10'],expected['lost10'],expected['net_hits'])
                    assert abs(float(np.mean(new))-expected['N10'])<1e-12
                    assert abs(float((np.array(new)-old_utility).mean())-expected['delta_N10'])<1e-12
            for policy in policies:
                for a,b in CONTRASTS:
                    delta=summary['cohorts'][cohort]['policies'][a+'_'+policy]['net_hits']-summary['cohorts'][cohort]['policies'][b+'_'+policy]['net_hits']
                    assert delta==summary['cohorts'][cohort]['contrasts'][a+'_vs_'+b+'_'+policy]['net_hit_difference']
    return dict(expanded_users=len(train['users']),harm_users=harm_users,expanded_pairs=len(expanded['items']),
        original_boundary_max_feature_error=max_error,original_boundary_bitwise_identical=exact,
        pair_identity_labels_and_weights_verified=True,user_fold_integrity=True,primary_users_disjoint=True,
        manual_prediction_checks=manual_checks,independent_user_policy_decisions=decision_count,all_counts_and_metrics_exact=True)


def main():
    summary=json.loads((OUT/'summary.json').read_text());assert summary['state']=='COMPLETED';result={}
    for domain in ('Toys','Beauty'):
        result[domain]=check_domain(domain,summary['domains'][domain]);print(domain,result[domain],flush=True)
    result['verification_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/'independent_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
