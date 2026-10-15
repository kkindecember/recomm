"""Fixed target-free A/B first; retrospective enriched sampling; hide identities and outcomes."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from analyze_history_evidence import fields
from analyze_selective_evidence import sha,write
from evidence_decomposition_core import read_users

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/phase21/blind_disagreements_20260925_v1'
PRIOR=ROOT/'artifacts/phase21/selective_evidence_20260925_v1'
RAW=ROOT/'artifacts/phase20/lime_clean_benefit_20260922_v1'
FULL=ROOT/'artifacts/phase21/full_validation_20260924_v1'
PROTOCOL=ROOT/'plan/第二十一阶段/21.17_扩大分歧样本的目标隐藏证据审查_20260925.md'


def key(s):return hashlib.sha256(('20260925|blind-disagreement|'+s).encode()).hexdigest()


def select_a(items,scores):
    return int(items[min(range(len(items)),key=lambda j:(-float(scores[j]),int(items[j])))]) if len(items) else 0


def gather_users(value):
    if isinstance(value,dict):
        found={value['user_id']} if 'user_id' in value else set()
        for v in value.values():found.update(gather_users(v))
        return found
    if isinstance(value,list):return set().union(*(gather_users(v) for v in value)) if value else set()
    return set()


def main():
    started=time.monotonic();OUT.mkdir(exist_ok=False);manifest={};excluded={};cards=[];sealed={};all_sources={};population={}
    for p in (Path(__file__),PROTOCOL,Path(__file__).with_name('analyze_history_evidence.py')):
        manifest[str(p.relative_to(ROOT))]=sha(p);(OUT/p.name).write_text(p.read_text())
    for domain in ('Toys','Beauty'):
        seen=set()
        for folder in ('explicit_relations_20260925_v1','entity_relations_20260925_v1','typed_relations_20260925_v1'):
            p=ROOT/'artifacts/phase21'/folder/'all_primary_non_neutral_cases.json'
            manifest[str(p.relative_to(ROOT))]=sha(p);v=json.loads(p.read_text())
            seen.update(gather_users(v.get(domain,v)))
        excluded[domain]=sorted(seen)
        folder=PRIOR/domain/'validation';paths=[folder/'users.csv',folder/'pairs.npz',folder/'score.npy',FULL/domain/'samples.json',RAW/domain/'catalog.json']
        prior_manifest=json.loads((ROOT/'artifacts/phase21/typed_relations_20260925_v1/evaluation_manifest.json').read_text())
        for p in paths:
            relative=str(p.relative_to(ROOT));manifest[relative]=sha(p);assert manifest[relative]==prior_manifest[relative]
        users=read_users(paths[0]);items=np.load(paths[1])['items'];score=np.load(paths[2],mmap_mode='r')[:len(items)]
        samples={r['user_id']:r for r in json.loads(paths[3].read_text())};catalog=json.loads(paths[4].read_text());parsed=[fields(p) for p in catalog['passages']]
        groups={k:[] for k in ('a_hit','b_hit','neither')};frame=[]
        for r in users:
            if r['cohort']!='primary' or r['user_id'] in seen:continue
            a=select_a(items[r['start']:r['end']],score[r['start']:r['end'],18]);b=r['boundary']
            if not a:continue
            # Outcomes used only after A has been fixed; never choose the target as A.
            label='a_hit' if a==r['target'] else 'b_hit' if b==r['target'] else 'neither'
            record=dict(user_id=r['user_id'],a=a,b=b,target=r['target'],stratum=label,start=r['start'],end=r['end'])
            groups[label].append(record);frame.append(record)
        selected=[]
        for label,count in [('a_hit',24),('b_hit',24),('neither',16)]:
            selected.extend(sorted(groups[label],key=lambda r:key(domain+'|sample|'+label+'|'+r['user_id']))[:count])
        population[domain]=dict(primary_after_exclusions=len(frame),strata={k:dict(population=len(v),sample=min(len(v),24 if k!='neither' else 16)) for k,v in groups.items()})
        selected.sort(key=lambda r:key(domain+'|display|'+r['user_id']))
        for number,r in enumerate(selected,1):
            cid=domain[0]+f'{number:03d}';swap=int(key(domain+'|orientation|'+r['user_id'])[:8],16)%2
            x,y=(r['b'],r['a']) if swap else (r['a'],r['b'])
            history=samples[r['user_id']]['full_history'];unique=list(dict.fromkeys(history))
            def public(item):
                f=parsed[item]
                return dict(title=f.get('title',''),category=f.get('categories',''),brand=f.get('brand',''))
            hcards=[]
            for j,h in enumerate(unique,1):
                hcards.append(dict(ref='H'+str(j),positions=[i+1 for i,v in enumerate(history) if v==h],
                    same_catalog_as=[side for side,identity in [('X',x),('Y',y)] if identity==h],**public(h)))
            card=dict(card_id=cid,domain=domain,history_events=len(history),X=public(x),Y=public(y),history=hcards)
            cards.append(card);all_sources[cid]=dict(X=parsed[x],Y=parsed[y],history={f'H{j}':parsed[h] for j,h in enumerate(unique,1)})
            sealed[cid]=dict(**r,X=x,Y=y,a_side='Y' if swap else 'X',b_side='X' if swap else 'Y',history=history)
        print(domain,'blind cards',len(selected),flush=True)
    write(OUT/'blind_cards.json',cards);write(OUT/'source_details.json',all_sources)
    write(OUT/'sealed_labels.json',dict(population=population,excluded_previously_shown=excluded,cards=sealed))
    for p in (OUT/'blind_cards.json',OUT/'source_details.json',OUT/'sealed_labels.json'):manifest[str(p.relative_to(ROOT))]=sha(p)
    write(OUT/'preparation_manifest.json',manifest)
    write(OUT/'status.json',dict(state='AWAITING_BLIND_REVIEW',cards=len(cards),seconds=time.monotonic()-started,
        labels_exposed_to_reviewer=False,semantic_training_state='USER_PAUSED',statistical_fits=0,gpu_inferences=0))


if __name__=='__main__':main()
