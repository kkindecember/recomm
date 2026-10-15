"""Fixed fits, OOF predictions and development evaluation for clean benefit probe."""
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from scipy.stats import rankdata
from experiment.phase20.lime_decode_screen import read,write,sha,stamp,arrays
from experiment.phase20.lime_clean_benefit import OUT,ROOT,rows
from experiment.phase20.lime_clean_benefit_core import (
    fold_ids,ranking,benefit_labels,fit_ridge,predict_ridge,fit_head,predict_head)
from experiment.phase20.catalog_scope_core import metrics,contrast,ranks,utilities

ARMS=('GRAM','fixed_late','constant_gate','benefit_ridge','linear_calibration',
      'mlp_seed0','mlp_seed1','mlp_seed2')
HEADS=(('linear_calibration','linear',0),('mlp_seed0','mlp',0),('mlp_seed1','mlp',1),('mlp_seed2','mlp',2))


def auc(y,score):
    y=np.asarray(y,dtype=bool);n=int(y.sum());m=len(y)-n
    return None if not n or not m else float((rankdata(score)[y].sum()-n*(n+1)/2)/(n*m))


def subset(data,ix):return {k:v[ix] for k,v in data.items()}


def evaluate(data,pred,benefit_pred,cohort):
    rr={k:ranking(data,v) for k,v in pred.items()}
    table={k:metrics(v) for k,v in rr.items()}
    y=benefit_labels(data);nonzero=abs(y)>1e-12;chosen=benefit_pred>0
    oracle=np.where(y>0,rr['fixed_late'],rr['GRAM'])
    return dict(users=len(y),metrics=table,
        contrasts={k:{ref:contrast(rr[k],rr[ref],repeats=1000,seed=20260922) for ref in ('GRAM','fixed_late')} for k in rr if k not in ('GRAM','fixed_late')},
        benefit=dict(positive=int((y>1e-12).sum()),negative=int((y < -1e-12).sum()),zero=int((~nonzero).sum()),
            mean=float(y.mean()),predicted_mean=float(benefit_pred.mean()),intervention_rate=float(chosen.mean()),
            mse=float(np.mean((benefit_pred-y)**2)),
            auc_help_vs_rest=auc(y>1e-12,benefit_pred),
            auc_help_vs_harm=auc(y[nonzero]>0,benefit_pred[nonzero]),
            chosen_actual_mean=float(y[chosen].mean()) if chosen.any() else None,
            rejected_actual_mean=float(y[~chosen].mean()) if (~chosen).any() else None,
            oracle_only_descriptive=metrics(oracle)),
        cohort=cohort),rr


def fit_predict(train,fit_ix,predict_data,folder,tag):
    folder.mkdir(exist_ok=True)
    tr_users=set(train['users'][fit_ix]);assert not tr_users & set(predict_data['users'])
    y=benefit_labels(subset(train,fit_ix))
    ridge=fit_ridge(train['query_x'][fit_ix],y)
    torch.save(dict(model=ridge,fit_users=train['users'][fit_ix],fit_indices=fit_ix),folder/f'{tag}.ridge.pt')
    bp=predict_ridge(ridge,predict_data['query_x'])
    predictions=dict(GRAM=predict_data['native'],fixed_late=predict_data['mixed'],
        constant_gate=predict_data['mixed'] if y.mean()>0 else predict_data['native'],
        benefit_ridge=np.where((bp>0)[:,None],predict_data['mixed'],predict_data['native']))
    meta={}
    for name,kind,seed in HEADS:
        fitted=fit_head(train,fit_ix,kind,seed)
        fitted['fit_users']=train['users'][fit_ix];fitted['used_users']=train['users'][fitted['used']]
        path=folder/f'{tag}.{name}.pt';torch.save(fitted,path)
        saved=torch.load(path,map_location='cpu')
        pred=predict_head(saved,predict_data['x'],predict_data['mixed'])
        replay=predict_head(saved,predict_data['x'],predict_data['mixed'],chunk=97)
        np.testing.assert_allclose(pred,replay,atol=2e-6,rtol=1e-6)
        np.testing.assert_array_equal(ranking(predict_data,pred),ranking(predict_data,replay))
        predictions[name]=pred
        meta[name]=dict(fit_users=len(fit_ix),covered_fit_users=len(fitted['used']),curve=fitted['curve'],
                        optimizer_steps=30,checkpoint_sha256=sha(path))
    return predictions,bp,meta


def run_domain(domain):
    folder=OUT/domain;data=arrays(folder/'data.npz');users=data['users'];assert len(set(users))==len(users)
    train=subset(data,np.flatnonzero(data['roles']=='ranker_fit'))
    dev=subset(data,np.flatnonzero(data['roles']=='development'));assert len(dev['users'])==512
    samples=read(folder/'samples.json');gate=torch.load(folder/'gate.pt',map_location='cpu')
    assert not set(gate['fit_users']) & set(users)
    assert set(gate['fit_users'])=={s['user_id'] for s in samples if s['role']=='gate_calibration'}
    folds=fold_ids(train['users'],domain);oof={a:np.zeros_like(train['native']) for a in ARMS};bp=np.zeros(len(train['users']))
    coverage=np.zeros(len(train['users']),dtype=int);foldreports={};training={}
    for f in range(5):
        write(OUT/'status.json',dict(state='RUNNING',stage='fit_small_heads',domain=domain,fold=f,pid=os.getpid(),updated_at=stamp()))
        fit=np.flatnonzero(folds!=f);predict=np.flatnonzero(folds==f)
        pred,adv,meta=fit_predict(train,fit,subset(train,predict),folder/'heads',f'fold{f}')
        for k,v in pred.items():oof[k][predict]=v
        bp[predict]=adv;coverage[predict]+=1;training[f'fold{f}']=meta
    assert (coverage==1).all()
    # Save predictions BEFORE evaluation; no fit/epoch/threshold chosen from OOF.
    np.savez_compressed(folder/'oof_predictions.npz',users=train['users'],folds=folds,benefit=bp,**oof)
    oofreport,oofr=evaluate(train,oof,bp,'training_terminal_holdout_user_OOF')
    for f in range(5):
        ix=np.flatnonzero(folds==f)
        foldreports[str(f)]={k:metrics(v[ix]) for k,v in oofr.items()}
    write(OUT/'status.json',dict(state='RUNNING',stage='fit_final_small_heads',domain=domain,pid=os.getpid(),updated_at=stamp()))
    pred,adv,meta=fit_predict(train,np.arange(len(train['users'])),dev,folder/'heads','final');training['final']=meta
    np.savez_compressed(folder/'development_predictions.npz',users=dev['users'],benefit=adv,**pred)
    devreport,devr=evaluate(dev,pred,adv,'development_user_disjoint_from_all_new_heads')
    guided=[]
    for r in rows(folder/'candidates.jsonl'):
        if r['role']!='development':continue
        s=samples[r['sample_index']];g=r['guided'];ids=np.asarray(g['ids']);score=np.asarray(g['scores'])
        score=np.where(~np.isin(ids,s['history']),score,-np.inf)
        guided.append(int(ranks(score[None],ids[None],np.array([r['target']]))[0]))
    devr['fixed_decode']=np.array(guided);devreport['metrics']['fixed_decode']=metrics(guided)
    devreport['fixed_decode_vs_same_rule_late']=contrast(devr['fixed_decode'],devr['fixed_late'],repeats=1000,seed=20260922)
    np.savez_compressed(folder/'ranks.npz',train_users=train['users'],development_users=dev['users'],
        **{f'oof_{k}':v for k,v in oofr.items()},**{f'dev_{k}':v for k,v in devr.items()})
    checks={}
    for name in ARMS[2:]:
        good_folds=sum(all(foldreports[str(f)][name]['N10']>foldreports[str(f)][ref]['N10'] for ref in ('GRAM','fixed_late')) for f in range(5))
        dv=devreport['metrics'][name];oo=oofreport['metrics'][name]
        dev_pass=all(dv['N10']>=1.01*devreport['metrics'][ref]['N10'] and dv['H10']>=devreport['metrics'][ref]['H10'] for ref in ('GRAM','fixed_late'))
        oof_pass=all(oo['N10']>oofreport['metrics'][ref]['N10'] for ref in ('GRAM','fixed_late')) and good_folds>=4
        checks[name]=dict(development_pass=dev_pass,oof_pass=oof_pass,positive_folds_vs_both=good_folds,passed=dev_pass and oof_pass)
    output=dict(state='COMPLETED',domain=domain,oof=oofreport,development=devreport,folds=foldreports,
        checks=checks,training=training,test_read=False,development_labels_used_for_fit=False,
        repeated_development_research=True,upstream_seeds=1,not_a_new_mechanism=True)
    write(folder/'analysis.json',output);return output


def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);started=time.monotonic()
    assert read(OUT/'status.json')['state']=='CACHE_COMPLETED'
    result={domain:run_domain(domain) for domain in ('Toys','Beauty')}
    passed={name:all(result[d]['checks'][name]['passed'] for d in result) for name in ARMS[2:]}
    passed['mlp_all_three_seeds']=all(passed[f'mlp_seed{i}'] for i in range(3))
    receipt=dict(state='ANALYZED',domains=result,decision=passed,seconds=time.monotonic()-started,
                 independent_verification_pending=True,test_read=False,development_labels_used_for_fit=False)
    write(OUT/'summary.json',receipt)
    write(OUT/'status.json',dict(state='ANALYZED',updated_at=stamp(),decision=passed,independent_verification_pending=True))
    print(json.dumps(dict(state='ANALYZED',decision=passed,seconds=time.monotonic()-started)),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as e:
        (OUT/'analysis_failure.txt').write_text(traceback.format_exc());write(OUT/'status.json',dict(state='FAILED',error=repr(e),updated_at=stamp()));raise
