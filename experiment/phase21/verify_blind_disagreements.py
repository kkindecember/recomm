"""Recompute sampling, full source cards, reveal mapping and every aggregate independently."""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
from analyze_history_evidence import fields
from analyze_selective_evidence import sha,write
from evidence_decomposition_core import read_users
from prepare_blind_disagreements import ROOT,OUT,PRIOR,RAW,FULL


def hashkey(s):return hashlib.sha256(('20260925|blind-disagreement|'+s).encode()).hexdigest()


def main():
    manifest=json.loads((OUT/'analysis_manifest.json').read_text())
    for p,h in manifest.items():assert sha(ROOT/p)==h
    frozen=json.loads((OUT/'review_freeze.json').read_text());summary=json.loads((OUT/'summary.json').read_text())
    assert sha(OUT/'frozen_review.json')==frozen['frozen_review_sha256']==summary['review_sha256']
    assert sha(OUT/'review_entries.json')==frozen['raw_review_sha256']
    assert datetime.fromisoformat(summary['frozen_at_utc'])<datetime.fromisoformat(summary['unblinded_at_utc'])
    sealed=json.loads((OUT/'sealed_labels.json').read_text())
    cards={r['card_id']:r for r in json.loads((OUT/'blind_cards.json').read_text())}
    source=json.loads((OUT/'source_details.json').read_text())
    review={r['card_id']:r for r in json.loads((OUT/'frozen_review.json').read_text())['cards']}
    revealed={r['card_id']:r for r in json.loads((OUT/'unblinded_reviews.json').read_text())};assert len(cards)==len(review)==len(revealed)==128
    checks=Counter();independent=[]
    def public(f):return dict(title=f.get('title',''),category=f.get('categories',''),brand=f.get('brand',''))
    def absent_keys(obj):
        if isinstance(obj,dict):
            assert not (set(obj)&{'target','user_id','a','b','stratum','score','label','chosen_role'})
            for v in obj.values():absent_keys(v)
        elif isinstance(obj,list):
            for v in obj:absent_keys(v)
    absent_keys(list(cards.values()))
    for domain in ('Toys','Beauty'):
        folder=PRIOR/domain/'validation';users=read_users(folder/'users.csv')
        with np.load(folder/'pairs.npz') as z:items=z['items']
        score=np.load(folder/'score.npy',mmap_mode='r')[:len(items)]
        samples={r['user_id']:r for r in json.loads((FULL/domain/'samples.json').read_text())}
        catalog=json.loads((RAW/domain/'catalog.json').read_text());parsed=[fields(p) for p in catalog['passages']]
        excluded=set(sealed['excluded_previously_shown'][domain]);pools={k:[] for k in ('a_hit','b_hit','neither')};fixed={}
        for r in users:
            if r['cohort']!='primary' or r['user_id'] in excluded:continue
            local=items[r['start']:r['end']].tolist()
            if not local:continue
            ranked=sorted(zip(local,score[r['start']:r['end'],18]),key=lambda p:(-float(p[1]),p[0]))
            a=ranked[0][0];b=r['boundary'];label='a_hit' if r['target']==a else 'b_hit' if r['target']==b else 'neither'
            pools[label].append(r['user_id']);fixed[r['user_id']]=(a,b,r['target'],label)
            checks['target_free_population_choices']+=1
        selected=[]
        for label,n in [('a_hit',24),('b_hit',24),('neither',16)]:
            expected=sorted(pools[label],key=lambda u:hashkey(domain+'|sample|'+label+'|'+u))[:n]
            selected+=expected
            assert sealed['population'][domain]['strata'][label]==dict(population=len(pools[label]),sample=len(expected))
        selected.sort(key=lambda u:hashkey(domain+'|display|'+u))
        for number,u in enumerate(selected,1):
            cid=domain[0]+f'{number:03d}';s=sealed['cards'][cid];c=cards[cid];r=review[cid];out=revealed[cid]
            assert s['user_id']==u and (s['a'],s['b'],s['target'],s['stratum'])==fixed[u]
            swap=int(hashkey(domain+'|orientation|'+u)[:8],16)%2
            x,y=(s['b'],s['a']) if swap else (s['a'],s['b'])
            assert (s['X'],s['Y'])==(x,y)
            assert (s['a_side'],s['b_side'])==(('Y','X') if swap else ('X','Y'))
            assert c['X']==public(parsed[x]) and c['Y']==public(parsed[y])
            assert source[cid]['X']==parsed[x] and source[cid]['Y']==parsed[y]
            history=samples[u]['full_history'];assert history==s['history'] and len(history)==c['history_events']
            unique=list(dict.fromkeys(history));assert len(unique)==len(c['history'])
            positions=[]
            for j,(item,h) in enumerate(zip(unique,c['history']),1):
                assert h['ref']==f'H{j}' and source[cid]['history'][f'H{j}']==parsed[item]
                assert {k:h[k] for k in ('title','category','brand')}==public(parsed[item])
                assert h['positions']==[i+1 for i,v in enumerate(history) if v==item]
                assert h['same_catalog_as']==[side for side,value in [('X',x),('Y',y)] if value==item]
                positions+=h['positions'];checks['history_item_records']+=1
            assert sorted(positions)==list(range(1,len(history)+1))
            for key,value in r.items():assert out[key]==value
            assert set(r['X_history']+r['Y_history'])<=set(source[cid]['history'])
            chosen={'X':x,'Y':y,'abstain':0}[r['preference']]
            role='abstain' if chosen==0 else 'A' if chosen==s['a'] else 'B'
            agrees=None if not chosen or s['stratum']=='neither' else chosen==s['target']
            assert out['chosen_item']==chosen and out['chosen_role']==role and out['agrees_with_observed_target']==agrees
            independent.append(dict(domain=domain,stratum=s['stratum'],r=r,chosen_role=role,agrees=agrees));checks['review_decisions']+=1
    def tally(rows):
        t=Counter()
        for v in rows:
            r=v['r'];specific={'named','product_type','attribute','compatibility','repeat'}
            xs=bool(set(r['X_relations'])&specific);ys=bool(set(r['Y_relations'])&specific);answer=r['preference']!='abstain'
            t['cards']+=1;t['X_specific']+=xs;t['Y_specific']+=ys;t['both_specific']+=xs and ys;t['either_specific']+=xs or ys
            t['contrast']+=r['contrast'];t['preferences']+=answer;t['abstentions']+=not answer
            t['contrast_but_abstain']+=r['contrast'] and not answer
            t['choose_a']+=v['chosen_role']=='A';t['choose_b']+=v['chosen_role']=='B'
            informative=v['stratum']!='neither';t['informative']+=informative;t['informative_answered']+=informative and answer
            t['informative_correct']+=v['agrees'] is True;t['informative_wrong']+=v['agrees'] is False
            t['neither_answered']+=not informative and answer
        return t
    sections=[(independent,summary['pooled_sample'])]
    for domain in ('Toys','Beauty'):
        subset=[r for r in independent if r['domain']==domain]
        sections.append((subset,summary['domains'][domain]['review']))
        for label in ('a_hit','b_hit','neither'):
            sections.append(([r for r in subset if r['stratum']==label],summary['domains'][domain]['strata'][label]))
    for rows,part in sections:
        for key,n in tally(rows).items():assert part[key]==n,(key,part[key],n)
        for rel,m in part['relation_counts'].items():
            subset=[v for v in rows if rel in v['r']['X_relations']+v['r']['Y_relations']]
            tt=tally(subset);assert m['cards']==len(subset)
            assert m['both_sides']==sum(rel in v['r']['X_relations'] and rel in v['r']['Y_relations'] for v in subset)
            for key in ('preferences','informative_correct','informative_wrong','neither_answered'):assert m[key]==tt[key]
            checks['relation_aggregate_rows']+=1
        checks['summary_sections']+=1
    assert json.loads((ROOT/'artifacts/phase21/expert_feedback_20260924_v1/status.json').read_text())['state']=='USER_PAUSED'
    result=dict(state='VERIFIED',source_hashes=len(manifest),checks=dict(checks),
        semantic_annotation_independently_verified=False,frozen_review_unchanged=True,
        description='Source provenance, selection, all histories, blindness fields and numerical summaries recomputed; subjective semantic judgments not independently validated.')
    write(OUT/'verification.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
