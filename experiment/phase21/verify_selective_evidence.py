"""Independently rebuild policies from saved predictions, then account all outcomes."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/phase21/selective_evidence_20260925_v1'
SOURCE=ROOT/'artifacts/phase20/lime_clean_benefit_20260922_v1'
FULL=ROOT/'artifacts/phase21/full_validation_20260924_v1'
ARMS=('score','evidence','shuffled')
POLICIES=('positive','budget05','budget10','budget20')
ABSENT=2**30


def read_csv(path):
    with path.open() as f:return list(csv.DictReader(f))


def verify_domain(domain,summary):
    folder=OUT/domain
    catalog=json.loads((SOURCE/domain/'catalog.json').read_text())
    full_catalog=json.loads((FULL/domain/'catalog.json').read_text())
    assert catalog['items']==full_catalog['items']
    assert catalog['passages']==full_catalog['passages']
    train_users=read_csv(folder/'train/users.csv')
    validation_users=read_csv(folder/'validation/users.csv')
    fit_ids={r['user_id'] for r in train_users}
    primary={r['user_id'] for r in validation_users if r['cohort']=='primary'}
    assert not fit_ids&primary
    original_samples=json.loads((SOURCE/domain/'samples.json').read_text())
    assert fit_ids=={s['user_id'] for s in original_samples if s['role']=='ranker_fit'}
    gate_ids={s['user_id'] for s in original_samples if s['role']=='gate_calibration'}
    assert not gate_ids&primary
    full_samples=json.loads((FULL/domain/'samples.json').read_text())
    assert primary=={s['user_id'] for s in full_samples if s['training_holdout_role']=='none' and not s['previously_evaluated']}
    verified_decisions=0;checked_features=0
    for kind,users,pred_name,cohorts in [('train',train_users,'train_oof_predictions',('train_oof',)),
        ('validation',validation_users,'validation_predictions',('primary','previous512','seen_fit_user_later_event','all'))]:
        metadata=json.loads((folder/kind/'metadata.json').read_text())
        assert len({u['user_id'] for u in users})==len(users)
        with np.load(folder/kind/'pairs.npz',allow_pickle=False) as z:
            item=z['items'];y=z['labels'];owner=z['row_user'];weight=z['weight'];offsets=z['offsets']
        assert len(item)==metadata['pairs'] and offsets[-1]==len(item)
        for i,u in enumerate(users):
            a,b=int(u['start']),int(u['end']);ids=item[a:b]
            assert (a,b)==(offsets[i],offsets[i+1]) and int(u['candidates'])==b-a
            assert ids.tolist()==sorted(set(ids.tolist())) and int(u['boundary']) not in ids
            assert not set(ids)&set(map(int,u['top9'].split()))
            np.testing.assert_array_equal(owner[a:b],np.full(b-a,i))
            np.testing.assert_array_equal(y[a:b],(ids==int(u['target'])).astype(np.int8)-int(u['boundary']==u['target']))
            if b>a:assert abs(weight[a:b].sum()-1)<1e-12
        xbase=np.load(folder/kind/'score.npy',mmap_mode='r')[:len(item)]
        for arm in ARMS:
            if arm=='score':x=xbase
            else:
                evidence=np.load(folder/kind/(arm+'.npy'),mmap_mode='r')[:len(item)]
                width=evidence.shape[1]//3
                np.testing.assert_array_equal(evidence[:,2*width:],evidence[:,:width]-evidence[:,width:2*width])
                x=np.concatenate([xbase,evidence],axis=1)
            with np.load(folder/(pred_name+'.npz'),allow_pickle=False) as z:pred=z[arm]
            assert len(pred)==len(item) and np.isfinite(pred).all() and np.isfinite(x).all()
            for k in (range(5) if kind=='train' else (None,)):
                name=arm+('_fold'+str(k) if k is not None else '_final')+'.npz'
                with np.load(folder/name,allow_pickle=False) as model:
                    fitted=set(model['fit_users'].tolist())
                    if k is None:assert fitted==fit_ids
                    else:
                        held={u['user_id'] for u in train_users if int(u['fold'])==k}
                        assert fitted==fit_ids-held and not fitted&held
                    options=[i for i,u in enumerate(users) if int(u['end'])>int(u['start']) and (k is None or int(u['fold'])==k)]
                    for user_i in (options[0],options[len(options)//2],options[-1]):
                        j=int(users[user_i]['start'])
                        manual=sum((float(v)-m)/sd*c for v,m,sd,c in zip(x[j],model['mean'],model['std'],model['coef']))+float(model['intercept'])
                        assert abs(manual-pred[j])<1e-10;checked_features+=1
            for cohort in cohorts:
                selected=[i for i,u in enumerate(users) if cohort in ('all','train_oof') or u['cohort']==cohort]
                eligible=[i for i in selected if int(users[i]['end'])>int(users[i]['start'])]
                chosen={};best={}
                for i in eligible:
                    a,b=int(users[i]['start']),int(users[i]['end'])
                    best_row=min(range(a,b),key=lambda j:(-pred[j],int(item[j])))
                    chosen[i]=int(item[best_row]);best[i]=float(pred[best_row])
                decisions={r['user_id']:r for r in read_csv(folder/(cohort+'_decisions.csv'))}
                assert set(decisions)=={users[i]['user_id'] for i in selected}
                for policy in POLICIES:
                    key=arm+'_'+policy
                    if policy=='positive':acted={i for i in eligible if best[i]>0}
                    else:
                        count=int(len(eligible)*int(policy[-2:])/100)
                        acted=set(sorted(eligible,key=lambda i:(-best[i],users[i]['user_id']))[:count])
                    gains=losses=0;utilities=[];old_utilities=[];hits=0
                    for i in selected:
                        u=users[i];row=decisions[u['user_id']];old=int(u['old_rank']);target=int(u['target'])
                        rank=old
                        if i in acted and chosen[i]==target:rank=10;gains+=1
                        if i in acted and int(u['boundary'])==target:rank=ABSENT;losses+=1
                        assert row[key+'_action']==str(i in acted)
                        assert int(row[key+'_item'])==chosen.get(i,0)
                        assert int(row[key+'_rank'])==rank
                        utilities.append(1/np.log2(rank+1) if rank<=10 else 0.)
                        old_utilities.append(1/np.log2(old+1) if old<=10 else 0.);hits+=rank<=10
                        verified_decisions+=1
                    expected=summary['cohorts'][cohort]['policies'][key]
                    assert (len(acted),gains,losses)==(expected['actions'],expected['gained10'],expected['lost10'])
                    assert abs(float(np.mean(utilities))-expected['N10'])<1e-12
                    assert abs(hits/len(selected)-expected['H10'])<1e-12
                    assert abs(float(np.mean(np.array(utilities)-old_utilities))-expected['delta_N10'])<1e-12
            del x
    return dict(unique_users=True,fit_and_primary_users_disjoint=True,primary_excludes_gate_fit=True,
        candidate_labels_and_weights_exact=True,models_fit_correct_users_only=True,manual_prediction_checks=checked_features,
        independent_policy_decisions_verified=verified_decisions,all_counts_H10_N10_exact=True)


def main():
    summary=json.loads((OUT/'summary.json').read_text());assert summary['state']=='COMPLETED'
    result={}
    for domain in ('Toys','Beauty'):
        result[domain]=verify_domain(domain,summary['domains'][domain]);print(domain,result[domain],flush=True)
    result['verification_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/'independent_verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')


if __name__=='__main__':main()
