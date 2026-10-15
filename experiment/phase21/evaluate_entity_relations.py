"""Frozen six-arm rank-10 intervention on existing score caches; no fitting."""
import csv
import json
import os
import time
import traceback
from pathlib import Path
import numpy as np
from analyze_selective_evidence import sha, write, csv_write
from evidence_decomposition_core import read_users
from evaluate_explicit_relations import maps, support_counts, choose, metrics, UNIT
from extract_entity_relations import ROOT, OUT, RAW, OLD

PRIOR = ROOT/'artifacts/phase21/selective_evidence_20260925_v1'
FULL = ROOT/'artifacts/phase21/full_validation_20260924_v1'
POLICIES = ['old', 'old_known', 'normalized', 'normalized_known', 'score_matched', 'shuffled_known']


def evaluate(domain, kind, old_map, new_map, manifest, enabled):
    folder = PRIOR/domain/kind;dest = OUT/domain;dest.mkdir(exist_ok=True)
    paths = [folder/'users.csv', folder/'pairs.npz', folder/'score.npy',
             (RAW if kind=='train' else FULL)/domain/'samples.json']
    source_manifest = json.loads((ROOT/'artifacts/phase21/category_relations_20260925_v1/manifest.json').read_text())
    for p in paths:
        key=str(p.relative_to(ROOT));manifest[key]=sha(p);assert manifest[key]==source_manifest[key]
    users=read_users(paths[0]);samples={s['user_id']:s for s in json.loads(paths[-1].read_text())}
    with np.load(paths[1]) as z:pairs={k:z[k] for k in z.files}
    scores=np.load(paths[2],mmap_mode='r')[:len(pairs['items'])]
    policies=POLICIES if enabled else POLICIES[:2]
    supports=np.zeros((len(pairs['items']),6),np.int32)
    choices=np.zeros((len(policies),len(users)),np.int32);records=[]
    old_rows={r['user_id']:r for r in csv.DictReader((OLD/domain/(kind+'_series_decisions.csv')).open())}
    manifest[str((OLD/domain/(kind+'_series_decisions.csv')).relative_to(ROOT))]=sha(OLD/domain/(kind+'_series_decisions.csv'))
    for i,r in enumerate(users):
        sl=slice(r['start'],r['end']);items=pairs['items'][sl];b=r['boundary'];fused=scores[sl,18]
        s=samples[r['user_id']];donor=samples[r['donor_user']]
        assert len(s['full_history'])==len(donor['full_history'])
        old=support_counts(old_map,list(items)+[b],s['full_history'])
        new=support_counts(new_map,list(items)+[b],s['full_history'])
        wrong=support_counts(new_map,list(items)+[b],donor['full_history'])
        for j,counts in enumerate((old,new,wrong)):
            supports[sl,2*j]=counts[:-1];supports[sl,2*j+1]=counts[-1]
        choices[0,i]=choose(items,fused,old[:-1],old[-1])
        choices[1,i]=choices[0,i] if b in old_map else 0
        assert choices[0,i]==int(old_rows[r['user_id']]['relation_item'])
        if enabled:
            choices[2,i]=choose(items,fused,new[:-1],new[-1])
            choices[3,i]=choices[2,i] if b in new_map else 0
            if choices[3,i]:
                choices[4,i]=int(items[min(range(len(items)),key=lambda j:(-float(fused[j]),int(items[j])))])
            choices[5,i]=choose(items,fused,wrong[:-1],wrong[-1]) if b in new_map else 0
        record=dict(user_id=r['user_id'],cohort=r['cohort'],target=r['target'],old_rank=r['old_rank'],
            boundary=b,donor_user=r['donor_user'],self_donor=int(r['donor_user']==r['user_id']),
            old_b_known=int(b in old_map),new_b_known=int(b in new_map),
            old_b_support=int(old[-1]),new_b_support=int(new[-1]),shuffled_b_support=int(wrong[-1]),
            candidates=len(items),old_a_supported=int((old[:-1]>0).sum()),new_a_supported=int((new[:-1]>0).sum()))
        for pi,p in enumerate(policies):
            item=int(choices[pi,i]);delta=int(bool(item))*(int(item==r['target'])-int(b==r['target']))
            record[p+'_item']=item;record[p+'_delta']=delta
        assert record['old_delta']==int(old_rows[r['user_id']]['relation_delta'])
        records.append(record)
    csv_write(dest/(kind+'_decisions.csv'),records)
    np.savez_compressed(dest/(kind+'_support.npz'),counts=supports)
    result={}
    cohorts=('train_holdout',) if kind=='train' else ('primary','previous512','seen_fit_user_later_event','all')
    for cohort in cohorts:
        indices=[i for i,r in enumerate(users) if cohort in ('all','train_holdout') or r['cohort']==cohort]
        rows=[records[i] for i in indices];n=len(rows);old=np.array([r['old_rank'] for r in rows])
        deltas=np.column_stack([[r[p+'_delta'] for r in rows] for p in policies]).astype(np.int8)
        part=dict(users=n,self_donors=sum(r['self_donor'] for r in rows),
            old_b_known=sum(r['old_b_known'] for r in rows),new_b_known=sum(r['new_b_known'] for r in rows),
            candidates=sum(r['candidates'] for r in rows),old_a_supported=sum(r['old_a_supported'] for r in rows),
            new_a_supported=sum(r['new_a_supported'] for r in rows),policies={},contrasts={},transitions={})
        for pi,p in enumerate(policies):part['policies'][p]=metrics(old,deltas[:,pi],choices[pi,indices]>0)
        primary='normalized_known' if enabled else 'old_known';pj=policies.index(primary)
        for j,p in enumerate(policies):
            if j==pj:continue
            part['contrasts'][primary+'_vs_'+p]=dict(net_hit_difference=int((deltas[:,pj]-deltas[:,j]).sum()),
                delta_N10=float((deltas[:,pj]-deltas[:,j]).mean())*UNIT)
        for p,base in [('old_known','old')]+([('normalized','old'),('normalized_known','old'),('normalized_known','normalized')] if enabled else []):
            j,k=policies.index(p),policies.index(base);v,w=deltas[:,j],deltas[:,k]
            part['transitions'][p+'_vs_'+base]=dict(retained_gains=int(((v==1)&(w==1)).sum()),
                foregone_gains=int(((v!=1)&(w==1)).sum()),avoided_losses=int(((v!=-1)&(w==-1)).sum()),
                retained_losses=int(((v==-1)&(w==-1)).sum()),new_gains=int(((v==1)&(w!=1)).sum()),
                new_losses=int(((v==-1)&(w!=-1)).sum()))
        if enabled:
            assert part['policies']['normalized_known']['actions']==part['policies']['score_matched']['actions']
            part['shuffled_self_donor_actions']=sum(r['self_donor'] and bool(r['shuffled_known_item']) for r in rows)
        if cohort in ('primary','train_holdout'):
            rng=np.random.default_rng(20260925);boot=[]
            for _ in range(20):
                ix=rng.integers(0,n,size=(100,n));boot.append(deltas[ix].mean(1)*UNIT)
            boot=np.concatenate(boot)
            for pi,p in enumerate(policies):part['policies'][p]['ci95']=np.quantile(boot[:,pi],[.025,.975]).tolist()
            for j,p in enumerate(policies):
                if j!=pj:part['contrasts'][primary+'_vs_'+p]['ci95']=np.quantile(boot[:,pj]-boot[:,j],[.025,.975]).tolist()
        result[cohort]=part
    return result


def main():
    started=time.monotonic();manifest=json.loads((OUT/'quality_manifest.json').read_text())
    for path,digest in manifest.items():assert sha(ROOT/path)==digest
    quality=json.loads((OUT/'quality_gate.json').read_text());enabled=quality['eligible_new_policies']
    for p in [Path(__file__),Path(__file__).with_name('evaluate_explicit_relations.py')]:
        manifest[str(p.relative_to(ROOT))]=sha(p)
    write(OUT/'status.json',dict(state='RUNNING',pid=os.getpid(),stage='fixed_policy_evaluation'))
    try:
        domains={}
        for domain in ('Toys','Beauty'):
            old=maps(json.loads((OLD/(domain+'_relations.json')).read_text()))['series']
            value=json.loads((OUT/(domain+'_relations.json')).read_text());new={}
            for ids in value['families'].values():
                for item in ids:new[item]=set(ids)-{item}
            domains[domain]={}
            for kind in ('train','validation'):
                domains[domain].update(evaluate(domain,kind,old,new,manifest,enabled))
                print(domain,kind,'completed',flush=True)
        result=dict(state='COMPLETED',domains=domains,policies=POLICIES if enabled else POLICIES[:2],
            seconds=time.monotonic()-started,statistical_fits=0,GRAM_optimizer_steps=0,expert_optimizer_steps=0,
            gpu_inferences=0,test_read=False,exploratory=True,independent_confirmation=False,automatic_gpu_launch=False)
        write(OUT/'summary.json',result);write(OUT/'evaluation_manifest.json',manifest)
        write(OUT/'status.json',{k:v for k,v in result.items() if k!='domains'})
        print(json.dumps({d:v['primary']['policies'] for d,v in domains.items()}),flush=True)
    except BaseException as error:
        write(OUT/'status.json',dict(state='FAILED',error=repr(error)))
        (OUT/'failure.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
