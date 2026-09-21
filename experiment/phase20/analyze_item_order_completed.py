"""Read-only scientific closeout of the frozen item-order transfer experiment.

No model forward, optimizer updates, new beam, or test split reads.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'artifacts/phase20/item_order_toys_chunk16_v3'
OUT=RUN/'review_20260917'
ARMS=('CE_anchor','order_teacher')


def read(path): return json.loads(path.read_text())


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def stats(ranks):
    return {f'{name}@{k}':sum((int(r<=k) if name=='hit' else (1/math.log2(r+1) if r<=k else 0))
                            for r in ranks)/len(ranks)
            for k in (5,10,20,50) for name in ('hit','ndcg')}


def paired(a,b):
    return dict(ndcg10_delta=stats(a)['ndcg@10']-stats(b)['ndcg@10'],
                relative_ndcg10=stats(a)['ndcg@10']/stats(b)['ndcg@10']-1,
                hit10_gained=sum(x<=10 and y>10 for x,y in zip(a,b)),
                hit10_lost=sum(x>10 and y<=10 for x,y in zip(a,b)),
                hit50_net=sum(x<=50 for x in a)-sum(y<=50 for y in b))


def panel_metrics(scores,parent,probabilities,positions,users):
    rank=np.argsort(np.argsort(-scores,axis=1,kind='stable'),axis=1,kind='stable')
    targets=np.where(positions>=0,rank[np.arange(len(scores)),np.maximum(positions,0)]+1,51)
    teacher=np.argsort(-probabilities,axis=1,kind='stable')
    prior=np.argsort(-parent,axis=1,kind='stable')
    i,j=np.triu_indices(50,1)
    sorted_scores=np.take_along_axis(scores,teacher,axis=1)
    hinge=np.maximum(sorted_scores[:,j]-sorted_scores[:,i]+(j-i)*.01,0).mean(1)
    satisfaction=[]
    for s,t,p in zip(scores,teacher,prior):
        up=sorted(set(t[:10])-set(p[:10])); down=sorted(set(p[:10])-set(t[:10]))
        satisfaction.append(float(np.mean(s[up,None]>s[None,down])) if up and down else np.nan)
    covered=positions>=0
    groups={}
    for name,mask in [('all',np.ones(len(scores),bool)),('covered',covered)]:
        selected=targets[mask].tolist(); identifiers=np.array(users)[mask].tolist()
        per_user={}
        for u,r in zip(identifiers,selected): per_user.setdefault(u,[]).append(r)
        n10=np.mean([stats(rs)['ndcg@10'] for rs in per_user.values()])
        h10=np.mean([stats(rs)['hit@10'] for rs in per_user.values()])
        groups[name]=dict(prefixes=len(selected),users=len(per_user),prefix_metrics=stats(selected),
            user_equal_ndcg10=float(n10),user_equal_hit10=float(h10),
            mean_order_hinge=float(hinge[mask].mean()),
            mean_up_down_satisfaction=float(np.nanmean(np.array(satisfaction)[mask])))
    return groups


def main():
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=False)
    config=read(RUN/'config.json'); summary=read(RUN/'summary.json'); status=read(RUN/'status.json')
    assert status['state']==summary['state']=='COMPLETED' and status['new_optimizer_steps']==342
    hashes={**config['source_sha256'],**config['input_sha256']}
    for p,h in hashes.items(): assert digest(ROOT/p)==h,('changed_frozen_source_or_input',p)
    ranks={}; model_checks={}; prediction_rows=0; metric_checks=0
    for arm in ARMS:
        train=read(RUN/arm/'train/summary.json')
        curve=[json.loads(line) for line in (RUN/arm/'train/training_curve.jsonl').read_text().splitlines()]
        assert curve==train['curve'] and [r['step'] for r in curve]==list(range(1,172))
        assert curve[-1]['examples']==21839 and curve[0]['learning_rate']==0
        assert sum(r['learning_rate']>0 for r in curve)==170
        path=RUN/arm/'train/last.pt'; assert digest(path)==train['checkpoint_sha256']
        ckpt=torch.load(path,map_location='cpu'); optim=torch.load(RUN/arm/'train/optimizer.pt',map_location='cpu')
        assert ckpt['step']==optim['step']==171 and ckpt['config_sha256']==digest(RUN/'config.json')
        assert set(ckpt['decoder_parameters'])==set(ckpt['trainable_names'])
        assert all(torch.isfinite(p).all() for p in ckpt['decoder_parameters'].values())
        assert all(float(s['step'])==171 for s in optim['optimizer']['state'].values())
        model_checks[arm]=dict(checkpoint_sha256=train['checkpoint_sha256'],optimizer_calls=171,
            nonzero_lr_steps=170,clipped_updates=sum(r['gradient_norm_before_clip']>1 for r in curve),
            maximum_preclip_gradient=max(r['gradient_norm_before_clip'] for r in curve),
            parameters_finite=True,trainable_parameter_count=sum(p.numel() for p in ckpt['decoder_parameters'].values()),
            initial_decoder_sha256=train['initial_decoder_sha256'],training_order_sha256=train['training_order_sha256'])
        del ckpt,optim
        with (RUN/arm/'evaluate/per_user.tsv').open() as f: rows=list(csv.DictReader(f,delimiter='\t'))
        table={r['user_id']:{k:int(v) for k,v in r.items() if k!='user_id'} for r in rows}
        assert len(rows)==len(table)==19412
        users=sorted(table)
        for col in rows[0]:
            if col=='user_id': continue
            key=arm+'_'+col if col in ('raw','pcrf') else col
            ranks[key]=[table[u][col] for u in users]
            for name,v in stats(ranks[key]).items():
                assert abs(v-summary['all_metrics'][key][name])<1e-12
                metric_checks+=1
        seen=set()
        for line in (RUN/arm/'evaluate/predictions.jsonl').open():
            r=json.loads(line); ids=r['ranked_item_ids']; u=r['user_id']
            assert u not in seen and r['split']=='validation' and len(ids)==len(set(ids))==50
            assert np.isfinite(r['scores']).all()
            actual=ids.index(r['gold_item_id'])+1 if r['gold_item_id'] in ids else 51
            assert actual==table[u]['raw'];seen.add(u);prediction_rows+=1
        assert seen==set(users)
    a,b=(model_checks[arm] for arm in ARMS)
    assert a['initial_decoder_sha256']==b['initial_decoder_sha256'] and a['training_order_sha256']==b['training_order_sha256']
    comparisons={f'{x}_minus_{y}':paired(ranks[x],ranks[y]) for x,y in [
        ('order_teacher_raw','c1'),('order_teacher_raw','CE_anchor_raw'),('order_teacher_raw','original_gram'),
        ('order_teacher_pcrf','CE_anchor_pcrf'),('order_teacher_pcrf','original_pcrf'),('order_teacher_pcrf','B1')]}
    for kind in ('raw','pcrf'):
        row=comparisons[f'order_teacher_{kind}_minus_CE_anchor_{kind}']
        saved=summary['matched_comparisons'][kind]['all_validation']
        assert abs(row['ndcg10_delta']-saved['ndcg@10']['delta'])<1e-12
        assert row['hit10_gained']==saved['hit10_gained'] and row['hit10_lost']==saved['hit10_lost']
    records=[json.loads(line) for line in (ROOT/config['train_candidate_jsonl']).read_text().splitlines()]
    panel=read(RUN/'panel.json'); analyses={}; first=None
    for arm in ARMS:
        with np.load(RUN/arm/'train/fixed_panel_scores.npz') as z: data={k:z[k] for k in z.files}
        assert data['indices'].tolist()==panel['fit_indices']+panel['calibration_indices']
        if first is None: first=data
        for key in ('indices','parent_scores','teacher_probability','positions'): assert np.array_equal(first[key],data[key])
        for part,slc in [('FIT',slice(0,256)),('calibration',slice(256,None))]:
            ids=data['indices'][slc]; u=[records[i]['user_id'] for i in ids]
            values={arm:data['scores'][slc]}
            if arm==ARMS[0]: values['C1']=data['parent_scores'][slc]
            for name,scores in values.items():
                analyses.setdefault(part,{})[name]=panel_metrics(scores,data['parent_scores'][slc],data['teacher_probability'][slc],data['positions'][slc],u)
    fit_users={records[i]['user_id'] for i in panel['fit_indices']}
    calibration_users={records[i]['user_id'] for i in panel['calibration_indices']}
    assert not fit_users & calibration_users
    result=dict(state='VERIFIED_SAVED_RESULTS_NOT_TRAINING_RERUN',frozen_hashes_checked=len(hashes),
        all_user_metric_values_recounted=metric_checks,fresh_predictions_recounted=prediction_rows,
        model_checks=model_checks,comparisons=comparisons,fixed_panel=analyses,
        fixed_panel_user_overlap=0,decision=summary['decision'],
        config_sha256=digest(RUN/'config.json'),analysis_source_sha256=digest(Path(__file__)),
        wall_seconds=status['elapsed_seconds'],task_seconds=sum(t['elapsed_seconds'] for t in status['tasks'].values()),
        new_optimizer_steps=0,new_model_forward=False,new_beam=False,test_read=False,
        limitations='One seed; repeated development validation; fixed panel is historical training prefixes, not unseen future supervision.')
    (OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'state':result['state'],'hashes':len(hashes),'comparisons':comparisons,'fixed_panel':analyses},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
