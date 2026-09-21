"""Training-lineage feasibility and bounded user-OOF development-label probe.

This is not a train-only recommendation result or an upstream cross-fit run.
Never read test; do not refit or select any upstream checkpoint.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from experiment.phase20.conditional_reference_core import Correction, ranks
from experiment.phase20.conditional_reference_screen import sha, read, rows, write, metrics, paired
from experiment.phase20.gacr_fusion_core import standardize_tensor
from experiment.phase20.history_interaction_core import dense_pairwise_losses
from experiment.phase20.personal_popularity_core import reconstruct_training

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'experiment/phase20/unseen_interaction_probe_toys_v1.json'


def folds(users, seed, count):
    return np.asarray([int.from_bytes(hashlib.sha256(
        f'{seed}|unseen-interaction-probe|{u}'.encode()).digest()[:8], 'big') % count for u in users])


def outer_split(users, cohort_users, user_folds, fold):
    train = np.asarray([u in cohort_users and f != fold for u,f in zip(users,user_folds)])
    evaluation = user_folds == fold
    assert not (train & evaluation).any()
    return np.flatnonzero(train), np.flatnonzero(evaluation)


def train_model(features, base, positions, config):
    """Accept only fold-training rows, no outer labels or validation callback."""
    covered = positions >= 0
    assert covered.any()
    x, b, p = [torch.as_tensor(z[covered]) for z in (features, base, positions)]
    torch.manual_seed(config['seed'])
    model = Correction(config['bound'])
    assert sum(v.numel() for v in model.parameters()) == 129
    opt = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    losses = []
    for step in range(config['steps']):
        model.train(); opt.zero_grad(set_to_none=True)
        total = 0.
        for start in range(0, len(p), config['chunk_size']):
            sl = slice(start,start+config['chunk_size'])
            scores = b[sl] + model(x[sl]).double()
            loss = dense_pairwise_losses(standardize_tensor(scores),p[sl]).sum()/len(p)
            assert torch.isfinite(loss)
            loss.backward(); total += float(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),config['clip'])
        assert torch.isfinite(norm)
        opt.step(); losses.append(total)
    model.eval()
    return model, losses


def predict(model, features, base, chunk=256):
    result = []
    model.eval()
    with torch.no_grad():
        for start in range(0,len(base),chunk):
            sl=slice(start,start+chunk)
            result.append(base[sl]+model(torch.from_numpy(features[sl])).numpy().astype(np.float64))
    return np.concatenate(result)


def count_ranks(scores, positions):
    result=np.full(len(scores),51,dtype=np.int64)
    ix=np.flatnonzero(positions>=0); p=positions[ix]
    target=scores[ix,p][:,None]
    result[ix]=1+(scores[ix]>target).sum(1)+((scores[ix]==target)&(np.arange(50)[None]<p[:,None])).sum(1)
    return result


class Job:
    def __init__(self, mode):
        self.c=read(CONFIG); self.mode=mode; self.started=time.monotonic()
        self.root=ROOT/self.c['output']; self.path=self.root/mode
        self.path.mkdir(parents=True,exist_ok=False)
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        for name,digest in {**self.c['input_sha256'],**self.c['source_sha256']}.items():
            assert sha(ROOT/name)==digest,name
        write(self.path/'protocol.json',self.c)
        self.status('initializing')

    def status(self, stage, state='RUNNING', **extra):
        elapsed=time.monotonic()-self.started
        if state=='RUNNING': assert elapsed<self.c['budget_seconds'], 'Predeclared CPU budget exceeded'
        value=dict(state=state,stage=stage,pid=os.getpid(),elapsed_seconds=elapsed,
                   config_sha256=sha(CONFIG),updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**extra)
        write(self.path/'status.json',value); print(json.dumps(value),flush=True)

    def finish(self, summary):
        write(self.path/'summary.json',{**summary,'elapsed_seconds':time.monotonic()-self.started,
            'config_sha256':sha(CONFIG),'test_read':False,'upstream_updates':0,
            'new_beam':False,'evidence_scope':'User-OOF diagnostic using development validation labels, not train-only or independent test performance.'})
        write(self.path/'manifest.json',{p.name:sha(p) for p in self.path.iterdir()
              if p.is_file() and p.name not in ('status.json','manifest.json')})
        self.status('finished','COMPLETED')

    def load_mode(self, mode):
        path=self.root/mode
        assert read(path/'status.json')['state']=='COMPLETED'
        assert read(path/'summary.json')['config_sha256']==sha(CONFIG)
        for name,h in read(path/'manifest.json').items():assert sha(path/name)==h
        return path

    def data(self):
        p=self.load_mode('prepare')
        with np.load(p/'probe_queries.npz') as z:return {k:z[k] for k in z.files}

    def prepare(self):
        d=self.c['data']
        with np.load(ROOT/d['queries']) as z:a={k:z[k] for k in z.files}
        train=list(rows(ROOT/d['train'])); val=list(rows(ROOT/d['validation']))
        _,sequences,_,_=reconstruct_training(train)
        n=int(a['ntrain']); assert len(train)==109361 and len(val)==19412
        cohort=set(a['users'][:n][a['fit']]); assert len(cohort)==3882
        terminal={r['user_id']:i for i,r in enumerate(train)}
        cached={}; original=list(rows(ROOT/d['training_candidates']))
        for j,r in enumerate(original):
            source=train[r['train_index']]
            assert all(source[k]==r[k] for k in ('user_id','history','target'))
            assert a['users'][j]==r['user_id']
            if a['fit'][j] and r['train_index']==terminal[r['user_id']]:cached[r['user_id']]=j
        assert set(cached)==cohort
        users=np.asarray([r['user_id'] for r in val]); np.testing.assert_array_equal(users,a['users'][n:])
        lookup={u:i for i,u in enumerate(users)}
        event_indices={terminal[u] for u in cohort}
        cleaned=[r for i,r in enumerate(train) if i not in event_indices]
        _,clean_sequences,_,_=reconstruct_training(cleaned)
        for u in sequences: assert clean_sequences[u]==(sequences[u][:-1] if u in cohort else sequences[u])
        heldout=[]
        for u in sorted(cohort):
            r=train[terminal[u]]
            assert r['history']==clean_sequences[u][-20:]
            heldout.append(dict(r,original_train_index=terminal[u],cache_index=cached[u]))
            v=val[lookup[u]]; assert v['history']==sequences[u][-20:]
        for filename,records in [('upstream_train_without_terminal.jsonl',cleaned),('terminal_holdout.jsonl',heldout)]:
            with (self.path/filename).open('x') as f:
                for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        # Missing historical rows for users outside original FIT never enter fitting.
        selected=np.asarray([cached.get(u,-1) for u in users])
        present=selected>=0; safe=np.maximum(selected,0)
        arrays={'users':users,'fold':folds(users,self.c['seed'],self.c['folds']),
            'cohort':present,'seen_features':a['features'][safe], 'seen_base':a['base_scores'][safe],
            'seen_positions':np.where(present,a['positions'][safe],-1),
            'future_features':a['features'][n:],'future_base':a['base_scores'][n:],
            'future_positions':a['positions'][n:],'target_frequencies':a['target_frequencies'],
            'known_ranks':a['known_ranks']}
        for k in ('seen_features','seen_base'):arrays[k][~present]=0
        np.testing.assert_array_equal(count_ranks(arrays['future_base'],arrays['future_positions']),arrays['known_ranks'])
        np.savez(self.path/'probe_queries.npz',**arrays)
        fold_counts=[]
        for f in range(self.c['folds']):
            ti,ev=outer_split(users,cohort,arrays['fold'],f)
            fold_counts.append(dict(fold=f,training_users=len(ti),evaluation_users=len(ev),
                seen_covered=int((arrays['seen_positions'][ti]>=0).sum()),
                future_covered=int((arrays['future_positions'][ti]>=0).sum()),user_overlap=0))
        with (ROOT/d['epoch_timing']).open() as f:timing=list(csv.DictReader(f))
        training_seconds=sum(float(r['epoch_train_time_seconds']) for r in timing)
        head=read(ROOT/d['c1_training_summary']); oldcache=read(ROOT/d['fusion_summary'])
        self.finish({'state':'PREPARED','train_rows':len(train),'clean_train_rows':len(cleaned),
            'heldout_events':len(heldout),'heldout_has_later_training_prefix':False,
            'validation_users':len(users),'cohort_users':len(cohort),'fold_counts':fold_counts,
            'seen_coverage':float((arrays['seen_positions'][present]>=0).mean()),
            'future_coverage_same_users':float((arrays['future_positions'][present]>=0).mean()),
            'full_validation_coverage':float((arrays['future_positions']>=0).mean()),
            'historical_gram_train_seconds_30_epochs':training_seconds,
            'historical_c1_train_seconds':head['wall_time_seconds'],
            'historical_candidate_job_seconds':oldcache['elapsed_seconds'],
            'historical_candidate_job_rows':oldcache['cache']['rows'],
            'new_upstream_throughput_measured':False,'existing_trained_parents_safe_for_terminal_holdout':False,
            'clean_text_T5_initialization_available':True,
            'similar_item_prompt_cache_requires_new_lineage_or_reconstruction':True,
            'automatic_upstream_retraining':False})

    def run(self):
        a=self.data(); cohort=set(a['users'][a['cohort']]); stats=[]
        output={'A_frozen':a['future_base'].copy(), 'B_seen':np.empty_like(a['future_base']),
                'C_future':np.empty_like(a['future_base'])}
        for f in range(self.c['folds']):
            ti,ev=outer_split(a['users'],cohort,a['fold'],f)
            np.savez(self.path/f'fold{f}.indices.npz',training=ti,evaluation=ev)
            for arm,prefix in [('B_seen','seen'),('C_future','future')]:
                self.status('fit',fold=f,arm=arm)
                started=time.monotonic()
                model,losses=train_model(a[prefix+'_features'][ti],a[prefix+'_base'][ti],a[prefix+'_positions'][ti],self.c)
                torch.save(model.state_dict(),self.path/f'fold{f}.{arm}.pt')
                output[arm][ev]=predict(model,a['future_features'][ev],a['future_base'][ev])
                stats.append(dict(fold=f,arm=arm,training_users=len(ti),evaluation_users=len(ev),
                    covered_training=int((a[prefix+'_positions'][ti]>=0).sum()),optimizer_steps=self.c['steps'],
                    fit_loss_curve=losses,seconds=time.monotonic()-started))
        assert all(np.isfinite(v).all() for v in output.values())
        np.savez(self.path/'oof_scores.npz',**output)
        rr={k:ranks(v,a['future_positions']) for k,v in output.items()}
        masks={'all':np.ones(len(a['users']),bool),'low_frequency':a['target_frequencies']<=self.c['q1'],
               'other_frequency':a['target_frequencies']>self.c['q1']}
        groups={g:{k:metrics(r[m]) for k,r in rr.items()} for g,m in masks.items()}
        comparisons={}
        for arm,ref in [('B_seen','A_frozen'),('C_future','A_frozen'),('C_future','B_seen')]:
            comparisons[arm+'_minus_'+ref]={g:paired(rr[arm][m],rr[ref][m],self.c) for g,m in masks.items()}
        fold_metrics={str(f):{k:metrics(r[a['fold']==f]) for k,r in rr.items()} for f in range(self.c['folds'])}
        positive=sum(v['C_future']['ndcg@10']>v['A_frozen']['ndcg@10'] for v in fold_metrics.values())
        g=groups['all']; gain=g['C_future']['ndcg@10']/g['A_frozen']['ndcg@10']-1
        eligible=(gain>=self.c['minimum_relative_ndcg_gain'] and g['C_future']['hit@10']>=g['A_frozen']['hit@10']
            and g['C_future']['ndcg@10']>g['B_seen']['ndcg@10'] and positive>=self.c['minimum_positive_folds'])
        with (self.path/'per_user.tsv').open('x') as f:
            w=csv.writer(f,delimiter='\t'); w.writerow(['user_id','fold','target_frequency',*rr])
            for i,u in enumerate(a['users']):w.writerow([u,int(a['fold'][i]),int(a['target_frequencies'][i]),*[int(r[i]) for r in rr.values()]])
        self.finish({'state':'ANALYZED','models':stats,'groups':groups,'comparisons':comparisons,
            'fold_metrics':fold_metrics,'positive_folds':positive,'optimizer_steps':len(stats)*self.c['steps'],
            'decision':'REVIEW_CLEAN_UPSTREAM_PILOT_COST' if eligible else 'DO_NOT_ESCALATE_THIS_SIMPLE_FUSER_TO_UPSTREAM_RETRAINING',
            'new_paper_method_confirmed':False,'causal_exposure_effect_identified':False,
            'development_labels_used_for_fitting':True,'selection':'fixed step30; no outer-label selection'})

    def audit(self):
        a=self.data(); path=self.load_mode('run'); report=read(path/'summary.json')
        with np.load(path/'oof_scores.npz') as z:scores={k:z[k] for k in z.files}
        np.testing.assert_array_equal(scores['A_frozen'],a['future_base'])
        visits=np.zeros(len(a['users']),dtype=int)
        for f in range(self.c['folds']):
            with np.load(path/f'fold{f}.indices.npz') as z:ti,ev=z['training'],z['evaluation']
            assert set(a['users'][ti]).isdisjoint(set(a['users'][ev]))
            assert (a['fold'][ti]!=f).all() and (a['fold'][ev]==f).all() and a['cohort'][ti].all()
            visits[ev]+=1
            for arm in ('B_seen','C_future'):
                model=Correction(self.c['bound']); model.load_state_dict(torch.load(path/f'fold{f}.{arm}.pt',map_location='cpu'))
                # Different batch size also checks that predictions are query-independent.
                actual=predict(model,a['future_features'][ev],a['future_base'][ev],chunk=113)
                np.testing.assert_allclose(actual,scores[arm][ev],rtol=0,atol=1e-6)
                np.testing.assert_array_equal(count_ranks(actual,a['future_positions'][ev]),count_ranks(scores[arm][ev],a['future_positions'][ev]))
        assert (visits==1).all()
        rr={k:count_ranks(v,a['future_positions']) for k,v in scores.items()}
        np.testing.assert_array_equal(rr['A_frozen'],a['known_ranks'])
        masks={'all':np.ones(len(visits),bool),'low_frequency':a['target_frequencies']<=self.c['q1'],'other_frequency':a['target_frequencies']>self.c['q1']}
        for group,m in masks.items():
            for arm,r in rr.items():
                for k in (5,10,20,50):
                    x=r[m]; hit=x<=k; gain=np.zeros(len(x)); gain[hit]=np.log(2)/np.log(x[hit]+1)
                    assert abs(gain.mean()-report['groups'][group][arm][f'ndcg@{k}'])<1e-14
                    assert abs(hit.mean()-report['groups'][group][arm][f'hit@{k}'])<1e-14
        self.finish({'state':'VERIFIED','models_reloaded':10,'each_user_predicted_by_one_nonfitting_model':True,
            'metric_values_recomputed':72,'user_overlap':0,'all_replayed_ranks_exact':True,
            'decision':report['decision'],'original_validation_is_reused_development_data':True})


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','audit']);args=p.parse_args()
    job=None
    try:
        job=Job(args.mode);getattr(job,args.mode)()
    except Exception as e:
        if job is not None:job.status('error','FAILED',error=str(e),traceback=traceback.format_exc())
        raise


if __name__=='__main__':main()
