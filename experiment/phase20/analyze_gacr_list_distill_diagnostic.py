"""CPU summaries for Q0; preserve prefix and user-level units explicitly."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def read(path):return json.loads(path.read_text())


def z(x):return (x-x.mean(1,keepdims=True))/np.maximum(x.std(1,keepdims=True),1e-6)


def logprob(scores):
    normalized=z(scores.astype(np.float64))
    return normalized-np.logaddexp.reduce(normalized,axis=1)[:,None]


def rank(scores,pos):
    order=np.argsort(-scores,axis=1,kind='stable')
    inverse=np.argsort(order,axis=1,kind='stable')+1
    result=np.full(len(pos),51,dtype=np.int64);covered=pos>=0
    result[covered]=inverse[np.flatnonzero(covered),pos[covered]]
    return result


def ndcg(r,k=10):return np.where(r<=k,1/np.log2(r+1),0.)


def mean_by_user(values,users,mask):
    selected=set(users[mask])
    return float(np.mean([values[mask & (users==u)].mean() for u in selected])) if selected else None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='artifacts/phase20/gacr_list_distill_toys_q0_v1');args=parser.parse_args()
    out=ROOT/args.output;config=read(out/'config.json');panel=read(out/'panel.json')
    run=ROOT/config['source_output'];status=read(out/'summary.json')
    assert status['state']=='COMPLETED'
    indices=np.asarray(panel['fit_indices']+panel['calibration_indices'])
    users=np.asarray([r['user_id'] for r in panel['samples']])
    with np.load(run/'cache/distillation_cache.npz') as cache:
        pos=cache['positions'][indices];q=cache['true_probability'][indices];parent=cache['parent_scores'][indices]
        covered=pos>=0;fit=cache['fit'][indices]
    teacher_rank=rank(q,pos);parent_rank=rank(parent,pos)
    parent_log=logprob(parent);teacher_log=np.log(q)
    masks={}
    for part,m in (('fit',fit),('calibration',~fit)):
        for suffix,s in (('all',np.ones(len(indices),dtype=bool)),('covered',covered),('uncovered',~covered)):
            masks[part+'_'+suffix]=m&s
    parent_order=np.argsort(-parent,axis=1,kind='stable')
    teacher_order=np.argsort(-q,axis=1,kind='stable')
    allpairs=[];disagreements=[]
    for p,t in zip(parent_order,teacher_order):
        allpairs.append([(int(t[j]),int(t[k])) for j in range(10) for k in range(j+1,50)])
        up=set(t[:10])-set(p[:10]);down=set(p[:10])-set(t[:10])
        disagreements.append([(int(i),int(j)) for i in up for j in down])
    groups={};per_model={};csv_rows=[]
    for model in ('C1','CE_anchor','true_teacher','shuffled_teacher'):
        with np.load(out/f'{model}.scores.npz') as data:
            assert np.array_equal(indices,data['indices'])
            scores=data['scores'].astype(np.float64);ce=data['ce']
        logs=logprob(scores);ranks=rank(scores,pos)
        # Count a tied student pair as 0.5 rather than depending on item index.
        def satisfaction(pairs):
            return np.asarray([np.mean([float(row[i]>row[j])+0.5*float(row[i]==row[j]) for i,j in pair])
                               if pair else np.nan for row,pair in zip(scores,pairs)])
        values=dict(ce=ce,teacher_kl=np.sum(q*(teacher_log-logs),1),
            anchor_kl=np.sum(np.exp(parent_log)*(parent_log-logs),1),
            ndcg10=ndcg(ranks),hit10=(ranks<=10).astype(float),
            teacher_top10_pair_satisfaction=satisfaction(allpairs),
            promoted_demoted_satisfaction=satisfaction(disagreements),
            score_change_l2=np.linalg.norm(scores-parent,axis=1))
        per_model[model]=dict(values=values,ranks=ranks)
        groups[model]={}
        for group,mask in masks.items():
            result=dict(prefixes=int(mask.sum()),users=len(set(users[mask])))
            for key,value in values.items():
                valid=mask & np.isfinite(value)
                result[key]=float(value[valid].mean()) if valid.any() else None
                result[key+'_user_mean']=mean_by_user(value,users,valid)
            result['teacher_ndcg10']=float(ndcg(teacher_rank)[mask].mean())
            result['parent_ndcg10']=float(ndcg(parent_rank)[mask].mean())
            result['teacher_hit10']=float((teacher_rank[mask]<=10).mean())
            result['parent_hit10']=float((parent_rank[mask]<=10).mean())
            groups[model][group]=result;csv_rows.append(dict(model=model,group=group,**result))
    comparisons={}
    for other in ('C1','CE_anchor','shuffled_teacher'):
        comparisons[other]={}
        for group,mask in masks.items():
            a,b=per_model['true_teacher'],per_model[other]
            result={}
            for key,value in a['values'].items():
                delta=value-b['values'][key];valid=mask & np.isfinite(delta)
                result[key+'_delta']=float(delta[valid].mean()) if valid.any() else None
                result[key+'_delta_user_mean']=mean_by_user(delta,users,valid)
            result['hit10_gained']=int(((a['ranks']<=10)&(b['ranks']>10)&mask).sum())
            result['hit10_lost']=int(((a['ranks']>10)&(b['ranks']<=10)&mask).sum())
            result['target_rank_changed']=int(((a['ranks']!=b['ranks'])&mask).sum())
            result['teacher_kl_relative_reduction']=1-a['values']['teacher_kl'][mask].mean()/b['values']['teacher_kl'][mask].mean()
            comparisons[other][group]=result
    gradients={}
    for model,result in status['gradients'].items():
        gradients[model]=dict(mean_gradient=result['mean_gradient'])
        for key in ('teacher_over_ce','anchor_over_teacher','teacher_alignment_with_total'):
            vals=np.array([s[key] for s in result['samples']])
            gradients[model][key+'_sample_quantiles']=dict(zip(('min','q25','median','q75','max'),np.quantile(vals,[0,.25,.5,.75,1]).tolist()))
        for pair in ('ce__teacher','ce__anchor','anchor__teacher'):
            vals=[s['cosines'][pair] for s in result['samples'] if s['cosines'][pair] is not None]
            gradients[model][pair]=dict(nonzero_samples=len(vals),negative_samples=sum(x<0 for x in vals),
                median=float(np.median(vals)) if vals else None)
    report=dict(state='ANALYZED',groups=groups,true_teacher_minus=comparisons,gradients=gradients,
        fit_prefixes=int(fit.sum()),fit_users=len(set(users[fit])),calibration_prefixes=int((~fit).sum()),
        calibration_users=len(set(users[~fit])),new_optimizer_steps=0,test_read=False,
        caveats=['FIT is stratified 192 covered/64 uncovered; not a population estimate.',
                 'Calibration contains repeated users and is historical development data; user means reported.',
                 'Gradient panel contains only covered FIT examples; not the full epoch mixture.',
                 'Teacher KL applies diagnostically to all rows; original training applies it only to covered rows.',
                 'No fresh search or new training; fixed list metrics do not prove generation improvements.'])
    (out/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    with (out/'metrics.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    print(json.dumps(dict(groups={k:{g:v for g,v in d.items() if g.endswith('_covered')} for k,d in groups.items()},gradients=gradients),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
