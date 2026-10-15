"""Only quality-passed source relations enter these frozen, no-fit cache checks."""
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

from analyze_selective_evidence import sha,write,csv_write
from evidence_decomposition_core import read_users

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/phase21/explicit_relations_20260925_v1'
PRIOR=ROOT/'artifacts/phase21/selective_evidence_20260925_v1'
RAW=ROOT/'artifacts/phase20/lime_clean_benefit_20260922_v1'
FULL=ROOT/'artifacts/phase21/full_validation_20260924_v1'
UNIT=1/np.log2(11)


def maps(value):
    series={};forward={};reverse={}
    for ids in value['series_groups'].values():
        for a in ids:series[a]=set(ids)-{a}
    for edge in value['explicit_edges']:
        a,b=edge['source'],edge['target'];forward.setdefault(a,set()).add(b);reverse.setdefault(b,set()).add(a)
    return dict(series=series,explicit=forward,reverse=reverse)


def support_counts(mapping,items,history):
    h=set(history)
    return np.array([len(mapping.get(int(i),set())&h) for i in items],dtype=int)


def choose(items,scores,a_support,b_support):
    eligible=np.flatnonzero((a_support>=1)&(a_support>b_support))
    if not len(eligible):return 0
    j=min(eligible,key=lambda j:(-a_support[j],-float(scores[j]),int(items[j])))
    return int(items[j])


def metrics(old,delta,actions):
    return dict(users=len(old),actions=int(actions.sum()),gained10=int((delta==1).sum()),lost10=int((delta==-1).sum()),
        neutral_actions=int(actions.sum())-int((delta!=0).sum()),net_hits=int(delta.sum()),
        N10=float(np.where(old<=10,1/np.log2(old+1),0.).mean())+float(delta.mean())*UNIT,
        H10=float((old<=10).mean())+float(delta.mean()),delta_N10=float(delta.mean())*UNIT)


def evaluate(domain,kind,relation,mapping,manifest):
    folder=PRIOR/domain/kind;dest=OUT/domain;dest.mkdir(exist_ok=True)
    paths=[folder/'users.csv',folder/'pairs.npz',folder/'score.npy',(RAW if kind=='train' else FULL)/domain/'samples.json']
    old_manifest=json.loads((ROOT/'artifacts/phase21/category_relations_20260925_v1/manifest.json').read_text())
    for p in paths:
        rel=str(p.relative_to(ROOT));manifest[rel]=sha(p);assert manifest[rel]==old_manifest[rel]
    users=read_users(folder/'users.csv')
    with np.load(folder/'pairs.npz') as z:pairs={k:z[k] for k in z.files}
    score=np.load(folder/'score.npy',mmap_mode='r')[:len(pairs['items'])]
    samples={s['user_id']:s for s in json.loads(paths[-1].read_text())}
    policies=['relation','score_matched','shuffled']+(['reverse'] if relation=='explicit' else [])
    choices=np.zeros((len(policies),len(users)),int)
    support=np.zeros((len(pairs['items']),3),int) # real A, real B, shuffled A
    q=np.zeros(len(pairs['items']),bool);rec=[]
    for i,r in enumerate(users):
        sl=slice(r['start'],r['end']);items=pairs['items'][sl];b=r['boundary'];fused=score[sl,18]
        s=samples[r['user_id']];donor=samples[r['donor_user']]
        assert len(s['full_history'])==len(donor['full_history'])
        counts=support_counts(mapping[relation],list(items)+[b],s['full_history'])
        wrong=support_counts(mapping[relation],list(items)+[b],donor['full_history'])
        a=choose(items,fused,counts[:-1],counts[-1]);choices[0,i]=a
        if a:
            j=min(range(len(items)),key=lambda j:(-float(fused[j]),int(items[j])))
            choices[1,i]=int(items[j])
        choices[2,i]=choose(items,fused,wrong[:-1],wrong[-1])
        if relation=='explicit':
            reverse=support_counts(mapping['reverse'],list(items)+[b],s['full_history'])
            choices[3,i]=choose(items,fused,reverse[:-1],reverse[-1])
        support[sl,0]=counts[:-1];support[sl,1]=counts[-1];support[sl,2]=wrong[:-1]
        q[sl]=(counts[:-1]>=1)&(counts[:-1]>counts[-1])
        record=dict(user_id=r['user_id'],cohort=r['cohort'],target=r['target'],old_rank=r['old_rank'],boundary=b,
            candidates=len(items),a_relation_candidates=int((counts[:-1]>0).sum()),b_support=int(counts[-1]),
            qualifying_candidates=int(q[sl].sum()))
        for pi,p in enumerate(policies):
            item=int(choices[pi,i]);delta=int(bool(item))*(int(item==r['target'])-int(b==r['target']))
            record[p+'_item']=item;record[p+'_delta']=delta
        rec.append(record)
    csv_write(dest/(kind+'_'+relation+'_decisions.csv'),rec)
    np.savez_compressed(dest/(kind+'_'+relation+'_relations.npz'),support=support,qualifying=q)
    result={}
    cohorts=('train_holdout',) if kind=='train' else ('primary','previous512','seen_fit_user_later_event','all')
    for cohort in cohorts:
        indices=[i for i,r in enumerate(users) if cohort in ('all','train_holdout') or r['cohort']==cohort]
        rows=[rec[i] for i in indices];old=np.array([r['old_rank'] for r in rows]);n=len(rows)
        deltas=np.column_stack([[r[p+'_delta'] for r in rows] for p in policies]).astype(np.int8)
        part=dict(users=n,users_with_A_relation=sum(r['a_relation_candidates']>0 for r in rows),
                  users_with_B_relation=sum(r['b_support']>0 for r in rows),
                  users_with_qualifying_proposal=sum(r['qualifying_candidates']>0 for r in rows),
                  candidates=sum(r['candidates'] for r in rows),relation_candidates=sum(r['a_relation_candidates'] for r in rows),
                  policies={},contrasts={},label_groups={})
        for pi,p in enumerate(policies):part['policies'][p]=metrics(old,deltas[:,pi],choices[pi,indices]>0)
        assert part['policies']['relation']['actions']==part['policies']['score_matched']['actions']
        # Clustering is by user. These posthoc labels are never provided to choose().
        for label_name,label in [('helpful',1),('harmful',-1),('neutral',0)]:
            total=linked=qualified=total_users=linked_users=qualified_users=0
            for i in indices:
                r=users[i];sl=slice(r['start'],r['end']);mask=pairs['labels'][sl]==label
                has=(support[sl,0]>0)&mask;qual=q[sl]&mask
                total+=int(mask.sum());linked+=int(has.sum());qualified+=int(qual.sum())
                total_users+=int(mask.any());linked_users+=int(has.any());qualified_users+=int(qual.any())
            part['label_groups'][label_name]=dict(pairs=total,linked_pairs=linked,qualifying_pairs=qualified,
                users=total_users,linked_users=linked_users,qualifying_users=qualified_users)
        for control in policies[1:]:
            j=policies.index(control);part['contrasts']['relation_vs_'+control]=dict(
                net_hit_difference=int((deltas[:,0]-deltas[:,j]).sum()),delta_N10=float((deltas[:,0]-deltas[:,j]).mean())*UNIT)
        if cohort in ('primary','train_holdout'):
            rng=np.random.default_rng(20260925);boot=[]
            for _ in range(20):
                ix=rng.integers(0,n,size=(100,n));boot.append(deltas[ix].mean(1)*UNIT)
            boot=np.concatenate(boot)
            for pi,p in enumerate(policies):part['policies'][p]['ci95']=np.quantile(boot[:,pi],[.025,.975]).tolist()
            for control in policies[1:]:
                j=policies.index(control);part['contrasts']['relation_vs_'+control]['ci95']=np.quantile(boot[:,0]-boot[:,j],[.025,.975]).tolist()
        result[cohort]=part
    return result


def main():
    started=time.monotonic();quality=json.loads((OUT/'quality_gate.json').read_text())
    manifest=json.loads((OUT/'quality_audit_manifest.json').read_text())
    for rel,digest in manifest.items():assert sha(ROOT/rel)==digest
    manifest[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    allowed=quality['eligible_relation_types'];domains={}
    write(OUT/'status.json',dict(state='RUNNING',stage='fixed_rule_evaluation',eligible_relations=allowed,pid=os.getpid()))
    try:
        for d in ('Toys','Beauty'):
            mapping=maps(json.loads((OUT/(d+'_relations.json')).read_text()));domains[d]={}
            for relation in allowed:
                domains[d][relation]={}
                for kind in ('train','validation'):
                    domains[d][relation].update(evaluate(d,kind,relation,mapping,manifest))
                    print(d,relation,kind,'completed',flush=True)
        summary=dict(state='COMPLETED',domains=domains,eligible_relations=allowed,seconds=time.monotonic()-started,
            statistical_fits=0,GRAM_optimizer_steps=0,expert_optimizer_steps=0,gpu_inferences=0,test_read=False,
            exploratory=True,independent_confirmation=False,automatic_gpu_launch=False,
            excluded_relations=[k for k in ('series','explicit') if k not in allowed])
        write(OUT/'summary.json',summary);write(OUT/'evaluation_manifest.json',manifest)
        write(OUT/'status.json',{k:v for k,v in summary.items() if k!='domains'})
        print(json.dumps({k:v for k,v in summary.items() if k!='domains'}),flush=True)
    except BaseException as e:
        write(OUT/'status.json',dict(state='FAILED',error=repr(e)));(OUT/'failure.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
