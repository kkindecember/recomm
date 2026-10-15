"""Six fixed policies, no fitting or new GRAM generation."""
from collections import defaultdict
import csv
import json
import os
import time
import traceback
from pathlib import Path
import numpy as np
from analyze_selective_evidence import sha,write,csv_write
from evidence_decomposition_core import read_users
from evaluate_explicit_relations import choose,metrics,UNIT
from extract_typed_relations import ROOT,OUT,RAW,PREVIOUS

PRIOR=ROOT/'artifacts/phase21/selective_evidence_20260925_v1'
FULL=ROOT/'artifacts/phase21/full_validation_20260924_v1'
POLICIES=['previous','typed_parent','hierarchy','fine_only','score_matched','shuffled']


class Counts:
    def __init__(self,data,previous):
        self.records={r['item']:r for r in data['items']};self.required=data['requirements']
        self.old={r['item']:r['family'] for r in previous['items'] if r['family']}
        self.parent={r['item']:r['own_family'] for r in data['items'] if r['own_family']}
        self.fine={};self.neighbors={};groups=defaultdict(set)
        for i,family in self.parent.items():
            r=self.records[i];dims=self.required[family]
            if dims and all(r['versions'][k]['status']=='known' for k in dims):
                self.fine[i]=(family,)+tuple(r['versions'][k]['value'] for k in dims)
        for name,label in [('old',self.old),('parent',self.parent),('fine',self.fine)]:
            groups=defaultdict(set)
            for i,key in label.items():groups[key].add(i)
            self.neighbors[name]={i:groups[key]-{i} for i,key in label.items()}

    def support(self,items,b,history):
        h=set(history);all_items=list(items)+[b]
        counts={name:np.array([len(mapping.get(int(i),set())&h) for i in all_items],np.int32)
                for name,mapping in self.neighbors.items()}
        old=np.full((len(items),2),-1,np.int32);parent=old.copy();hierarchy=old.copy();level=np.zeros(len(items),np.int8)
        for j,a in enumerate(items):
            if a in self.old and b in self.old:old[j]=[counts['old'][j],counts['old'][-1]]
            if a not in self.parent or b not in self.parent:continue
            parent[j]=[counts['parent'][j],counts['parent'][-1]]
            if self.parent[a]!=self.parent[b]:hierarchy[j]=parent[j];level[j]=1
            elif a in self.fine and b in self.fine and self.fine[a]!=self.fine[b]:
                hierarchy[j]=[counts['fine'][j],counts['fine'][-1]];level[j]=2
        return old,parent,hierarchy,level


def select(items,scores,pair_support):
    return choose(items,scores,pair_support[:,0],pair_support[:,1])


def evaluate(domain,kind,engine,manifest):
    folder=PRIOR/domain/kind;dest=OUT/domain;dest.mkdir(exist_ok=True)
    paths=[folder/'users.csv',folder/'pairs.npz',folder/'score.npy',(RAW if kind=='train' else FULL)/domain/'samples.json',
           PREVIOUS/domain/(kind+'_decisions.csv')]
    old_manifest=json.loads((PREVIOUS/'evaluation_manifest.json').read_text())
    for p in paths:
        rel=str(p.relative_to(ROOT));manifest[rel]=sha(p)
        reference=old_manifest if p in paths[:4] else json.loads((PREVIOUS/'verification_manifest.json').read_text())
        assert manifest[rel]==reference[rel]
    users=read_users(paths[0]);samples={s['user_id']:s for s in json.loads(paths[3].read_text())}
    previous={r['user_id']:r for r in csv.DictReader(paths[4].open())}
    with np.load(paths[1]) as z:pairs={k:z[k] for k in z.files}
    scores=np.load(paths[2],mmap_mode='r')[:len(pairs['items'])]
    support=np.full((len(pairs['items']),8),-1,np.int32);levels=np.zeros(len(pairs['items']),np.int8)
    actions=np.zeros((len(POLICIES),len(users)),np.int32);records=[]
    for i,r in enumerate(users):
        sl=slice(r['start'],r['end']);items=pairs['items'][sl];b=r['boundary'];score=scores[sl,18]
        s=samples[r['user_id']];donor=samples[r['donor_user']]
        assert len(s['full_history'])==len(donor['full_history'])
        old,parent,hier,level=engine.support(items,b,s['full_history'])
        _,_,wrong,_=engine.support(items,b,donor['full_history'])
        support[sl]=np.column_stack([old,parent,hier,wrong]);levels[sl]=level
        for j,v in enumerate((old,parent,hier)):actions[j,i]=select(items,score,v)
        fine=hier.copy();fine[level!=2]=-1;actions[3,i]=select(items,score,fine)
        if actions[2,i]:actions[4,i]=int(items[min(range(len(items)),key=lambda j:(-float(score[j]),int(items[j])))])
        actions[5,i]=select(items,score,wrong)
        assert actions[0,i]==int(previous[r['user_id']]['normalized_known_item'])
        record=dict(user_id=r['user_id'],cohort=r['cohort'],target=r['target'],old_rank=r['old_rank'],boundary=b,
            donor_user=r['donor_user'],self_donor=int(r['user_id']==r['donor_user']),
            candidates=len(items),known_boundary=int(b in engine.parent),
            parent_comparable=int((parent[:,0]>=0).sum()),hierarchy_comparable=int((hier[:,0]>=0).sum()),
            version_comparable=int((level==2).sum()),
            hierarchy_qualifying=int(((hier[:,0]>=1)&(hier[:,0]>hier[:,1])).sum()),
            version_qualifying=int(((level==2)&(hier[:,0]>=1)&(hier[:,0]>hier[:,1])).sum()))
        for pi,p in enumerate(POLICIES):
            item=int(actions[pi,i]);record[p+'_item']=item
            record[p+'_delta']=int(bool(item))*(int(item==r['target'])-int(b==r['target']))
        chosen=record['hierarchy_item'];record['hierarchy_level']=int(level[np.flatnonzero(items==chosen)[0]]) if chosen else 0
        assert record['previous_delta']==int(previous[r['user_id']]['normalized_known_delta'])
        records.append(record)
    csv_write(dest/(kind+'_decisions.csv'),records)
    np.savez_compressed(dest/(kind+'_support.npz'),counts=support,levels=levels)
    result={}
    cohorts=['train_holdout'] if kind=='train' else ['primary','previous512','seen_fit_user_later_event','all']
    for cohort in cohorts:
        indices=[i for i,r in enumerate(users) if cohort in ('all','train_holdout') or r['cohort']==cohort]
        rows=[records[i] for i in indices];n=len(rows);old=np.array([r['old_rank'] for r in rows])
        deltas=np.column_stack([[r[p+'_delta'] for r in rows] for p in POLICIES]).astype(np.int8)
        part=dict(users=n,policies={},contrasts={},label_groups={})
        for k in ('self_donor','candidates','known_boundary','parent_comparable','hierarchy_comparable','version_comparable',
                  'hierarchy_qualifying','version_qualifying'):part[k]=sum(r[k] for r in rows)
        part['users_with_version_comparison']=sum(r['version_comparable']>0 for r in rows)
        part['users_with_version_qualification']=sum(r['version_qualifying']>0 for r in rows)
        part['hierarchy_parent_actions']=sum(r['hierarchy_level']==1 for r in rows)
        part['hierarchy_version_actions']=sum(r['hierarchy_level']==2 for r in rows)
        part['shuffled_self_donor_actions']=sum(r['self_donor'] and r['shuffled_item']>0 for r in rows)
        for pi,p in enumerate(POLICIES):part['policies'][p]=metrics(old,deltas[:,pi],actions[pi,indices]>0)
        assert part['policies']['hierarchy']['actions']==part['policies']['score_matched']['actions']
        for label_name,label in [('helpful',1),('harmful',-1),('neutral',0)]:
            counts=defaultdict(int)
            for i in indices:
                r=users[i];sl=slice(r['start'],r['end']);mask=pairs['labels'][sl]==label
                comparable=mask&(support[sl,4]>=0);version=mask&(levels[sl]==2)
                qualifies=comparable&(support[sl,4]>=1)&(support[sl,4]>support[sl,5])
                for key,val in [('all',mask),('comparable',comparable),('version',version),('qualifying',qualifies)]:
                    counts[key+'_pairs']+=int(val.sum());counts[key+'_users']+=int(val.any())
            part['label_groups'][label_name]=dict(counts)
        for j,p in enumerate(POLICIES):
            if p!='hierarchy':part['contrasts']['hierarchy_vs_'+p]=dict(net_hit_difference=int((deltas[:,2]-deltas[:,j]).sum()),
                delta_N10=float((deltas[:,2]-deltas[:,j]).mean())*UNIT)
        if cohort in ('primary','train_holdout'):
            rng=np.random.default_rng(20260925);boot=[]
            for _ in range(20):
                ix=rng.integers(0,n,size=(100,n));boot.append(deltas[ix].mean(1)*UNIT)
            boot=np.concatenate(boot)
            for pi,p in enumerate(POLICIES):part['policies'][p]['ci95']=np.quantile(boot[:,pi],[.025,.975]).tolist()
            for j,p in enumerate(POLICIES):
                if p!='hierarchy':part['contrasts']['hierarchy_vs_'+p]['ci95']=np.quantile(boot[:,2]-boot[:,j],[.025,.975]).tolist()
        result[cohort]=part
    return result


def main():
    started=time.monotonic();manifest=json.loads((OUT/'quality_manifest.json').read_text())
    for p,h in manifest.items():assert sha(ROOT/p)==h
    quality=json.loads((OUT/'quality_gate.json').read_text());assert quality['eligible_new_policies'],'Source gate not passed'
    for p in [Path(__file__),Path(__file__).with_name('evaluate_explicit_relations.py')]:manifest[str(p.relative_to(ROOT))]=sha(p)
    write(OUT/'status.json',dict(state='RUNNING',stage='fixed_six_policies',pid=os.getpid(),semantic_training_state='USER_PAUSED'))
    try:
        domains={}
        for domain in ('Toys','Beauty'):
            engine=Counts(json.loads((OUT/(domain+'_relations.json')).read_text()),json.loads((PREVIOUS/(domain+'_relations.json')).read_text()))
            domains[domain]={}
            for kind in ('train','validation'):
                domains[domain].update(evaluate(domain,kind,engine,manifest));print(domain,kind,'completed',flush=True)
        result=dict(state='COMPLETED',domains=domains,policies=POLICIES,seconds=time.monotonic()-started,
            statistical_fits=0,GRAM_optimizer_steps=0,expert_optimizer_steps=0,gpu_inferences=0,test_read=False,
            exploratory=True,independent_confirmation=False,automatic_gpu_launch=False)
        write(OUT/'summary.json',result);write(OUT/'evaluation_manifest.json',manifest)
        write(OUT/'status.json',{k:v for k,v in result.items() if k!='domains'})
        print(json.dumps({d:r['primary'] for d,r in domains.items()}),flush=True)
    except BaseException as e:
        write(OUT/'status.json',dict(state='FAILED',error=repr(e)))
        (OUT/'failure.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
