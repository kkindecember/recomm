"""Independent label-count and decision recomputation; semantic labels are not independently verified."""
from collections import Counter, defaultdict
import csv
import json
import re
from pathlib import Path
import numpy as np
from analyze_selective_evidence import sha, write
from analyze_history_evidence import fields, tokens
from extract_entity_relations import ROOT, OUT, OLD, RAW
from entity_relation_core import EntityRelations, metadata

POLICIES=['old','old_known','normalized','normalized_known','score_matched','shuffled_known']


def main():
    manifest=json.loads((OUT/'evaluation_manifest.json').read_text())
    for p,digest in manifest.items():assert sha(ROOT/p)==digest
    summary=json.loads((OUT/'summary.json').read_text());checks=Counter();cases={};diagnostics={}
    for domain in ('Toys','Beauty'):
        value=json.loads((OUT/(domain+'_relations.json')).read_text())
        previous=json.loads((OLD/(domain+'_relations.json')).read_text())
        catalog=json.loads((RAW/domain/'catalog.json').read_text());parsed=[fields(p) for p in catalog['passages']]
        old_label={i:k for k,items in previous['series_groups'].items() for i in items}
        new_label={r['item']:r['family'] for r in value['items'] if r['family']}
        assert new_label=={i:k for k,items in value['families'].items() for i in items}
        assert len(value['items'])==len(catalog['items'])-1
        for r in value['items']:
            assert r['item'] not in new_label or r['status'] in ('seed','alias_added')
            assert r['status'] not in ('unmatched','ambiguous') or not r['family']
            if r['size_after_series']:assert not r['batch']
        for name in value['seeds']:
            for seed in value['seeds'][name]:assert seed['items']==previous['series_groups'][seed['label']]
        # Independently construct the substring lookup and recompute the longest prefix result.
        index=defaultdict(set)
        from explicit_relation_core import GENERIC, NEGATION
        def informative(part):return any(len(t)>=3 and not t.isdigit() and t not in GENERIC for t in part)
        for i,f in enumerate(parsed[1:],1):
            ts=tokens(f.get('title',''))
            for start in range(len(ts)):
                for end in range(start+1,min(len(ts),start+6)+1):
                    part=ts[start:end]
                    if informative(part) and (len(part)>1 or len(part[0])>=5):index[''.join(part)].add(i)
        for lookup in value['compatibility_lookups']:
            raw=parsed[lookup['source']].get(lookup['field'],'');assert lookup['source_span'] in raw
            status=lookup['status'];ids=lookup['possible_items']
            assert ids==sorted(set(ids));assert lookup['source'] not in ids
            if status=='negated':assert not ids
            else:
                ts=lookup['object_prefix'].split();expected=[];alias=''
                for length in range(len(ts),0,-1):
                    part=ts[:length]
                    if any(t in NEGATION for t in part) or not informative(part) or (length==1 and len(part[0])<5):continue
                    possible=sorted(index.get(''.join(part),set())-{lookup['source']})
                    if possible:expected=possible;alias=' '.join(part);break
                assert ids==expected and lookup['alias']==alias
                expected_status=('unmatched' if not ids else 'too_broad' if len(ids)>.02*(len(parsed)-1) else
                    'single_catalog_match' if len(ids)==1 else 'multiple_catalog_matches')
                assert status==expected_status
            checks['compatibility_lookups']+=1
        checks['catalog_items']+=len(value['items']);checks['known_family_items']+=len(new_label)
        cases[domain]=[];diagnostics[domain]={}
        for kind in ('train','validation'):
            folder=ROOT/'artifacts/phase21/selective_evidence_20260925_v1'/domain/kind
            input_users=list(csv.DictReader((folder/'users.csv').open()))
            output_users=list(csv.DictReader((OUT/domain/(kind+'_decisions.csv')).open()))
            samples_path=(RAW if kind=='train' else ROOT/'artifacts/phase21/full_validation_20260924_v1')/domain/'samples.json'
            samples={r['user_id']:r for r in json.loads(samples_path.read_text())}
            with np.load(folder/'pairs.npz') as z:items=z['items'];labels=z['labels']
            score=np.load(folder/'score.npy',mmap_mode='r')[:len(items)]
            support=np.load(OUT/domain/(kind+'_support.npz'))['counts']
            prior_rows=list(csv.DictReader((OLD/domain/(kind+'_series_decisions.csv')).open()))
            assert len(input_users)==len(output_users)==len(prior_rows)
            group_counts=Counter()
            for r,saved,prior in zip(input_users,output_users,prior_rows):
                assert r['user_id']==saved['user_id']==prior['user_id']
                b=int(r['boundary']);target=int(r['target']);start,end=int(r['start']),int(r['end'])
                candidate=items[start:end].tolist();history=set(samples[r['user_id']]['full_history'])
                donor=set(samples[r['donor_user']]['full_history'])
                def count(label,h):
                    totals=Counter(label[x] for x in h if x in label)
                    return [totals.get(label.get(i,''),0)-int(i in h and i in label) for i in candidate+[b]]
                oc,nc,sc=count(old_label,history),count(new_label,history),count(new_label,donor)
                expected=np.column_stack([oc[:-1],[oc[-1]]*len(candidate),nc[:-1],[nc[-1]]*len(candidate),sc[:-1],[sc[-1]]*len(candidate)])
                assert np.array_equal(expected,support[start:end]);checks['candidate_support_rows']+=len(candidate)
                assert np.array_equal(labels[start:end],[int(i==target)-int(b==target) for i in candidate])
                def select(counts):
                    eligible=[j for j,c in enumerate(counts[:-1]) if c>0 and c>counts[-1]]
                    eligible.sort(key=lambda j:(-counts[j],-float(score[start+j,18]),candidate[j]))
                    return candidate[eligible[0]] if eligible else 0
                old=select(oc);new=select(nc);known=new if b in new_label else 0
                matched=candidate[min(range(len(candidate)),key=lambda j:(-float(score[start+j,18]),candidate[j]))] if known else 0
                choices=[old,old if b in old_label else 0,new,known,matched,select(sc) if b in new_label else 0]
                assert old==int(prior['relation_item'])
                for p,item in zip(POLICIES,choices):
                    assert item==int(saved[p+'_item'])
                    assert int(saved[p+'_delta'])==int(bool(item))*(int(item==target)-int(b==target))
                    if item:
                        assert item in candidate and item!=b
                        # The cached top9 are untouched; proposals must lie outside them.
                        top9=[int(v) for v in re.findall(r'\d+',r['top9'])]
                        assert item not in top9
                    checks['policy_decisions']+=1
                if kind=='validation' and saved['cohort']=='primary':
                    for p,item in zip(POLICIES,choices):
                        if item:group_counts[p+'_same_family_as_B']+=int(new_label.get(item,'!')==new_label.get(b,'?'))
                    if any(int(saved[p+'_delta']) for p in POLICIES):
                        involved=sorted({int(saved[p+'_item']) for p in POLICIES if int(saved[p+'_item'])}|{b,target})
                        cases[domain].append(dict(decision=saved,items=[dict(item=i,title=parsed[i].get('title',''),
                            old_label=old_label.get(i,''),new_label=new_label.get(i,''),
                            old_history_support=sorted(x for x in history if x!=i and old_label.get(i) and old_label.get(i)==old_label.get(x)),
                            new_history_support=sorted(x for x in history if x!=i and new_label.get(i) and new_label.get(i)==new_label.get(x))) for i in involved]))
            diagnostics[domain][kind]=dict(group_counts)
            cohorts=['train_holdout'] if kind=='train' else ['primary','previous512','seen_fit_user_later_event','all']
            for cohort in cohorts:
                rows=[r for r in output_users if cohort in ('all','train_holdout') or r['cohort']==cohort]
                result=summary['domains'][domain][cohort];assert result['users']==len(rows)
                for p in POLICIES:
                    values=[int(r[p+'_delta']) for r in rows];m=result['policies'][p]
                    assert m['actions']==sum(int(r[p+'_item'])!=0 for r in rows)
                    assert m['gained10']==values.count(1) and m['lost10']==values.count(-1) and m['net_hits']==sum(values)
                    ndcg=sum((1/np.log2(int(r['old_rank'])+1) if int(r['old_rank'])<=10 else 0)+int(r[p+'_delta'])/np.log2(11) for r in rows)/len(rows)
                    assert abs(ndcg-m['N10'])<1e-12
                    checks['aggregate_rows']+=1
    # Small safety regressions exercise genuine extraction hazards, not policy performance.
    assert metadata('Vidal Sassoon Pro Series 1 inch flat iron')['batch']==[]
    assert metadata('Vidal Sassoon Pro Series 1.5 inch flat iron')['batch']==[]
    assert metadata('Walking Dead TV Series 4')['batch']==['4']
    assert metadata('Walking Dead TV Series 4')['scope']==['tv']
    synthetic={'items':['PAD','a','b','c','d','e','f'], 'passages':['',
        'title: alpha beta series red','title: alpha beta series blue',
        'title: gamma delta series red','title: gamma delta series blue',
        'title: alpha beta gamma delta accessory','title: alpha beta gamma delta series green']}
    fake={'series_groups':{'alpha beta series':[1,2],'gamma delta series':[3,4]}}
    engine=EntityRelations(synthetic,fake,'regression')
    assert engine.items[5]['status']=='ambiguous' and not engine.items[5]['family']
    checks['regression_assertions']=5
    quality=json.loads((OUT/'quality_gate.json').read_text());review=json.loads((OUT/'source_review.json').read_text())
    for domain in ('Toys','Beauty'):
        cards=review['domains'][domain]['family_alias'];supported=sum(r['verdict']=='supported' for r in cards)
        assert quality['domains'][domain]['supported']==supported
        assert quality['domains'][domain]['passed']==(len(cards)>=10 and supported/len(cards)>=.9)
    paused=json.loads((ROOT/'artifacts/phase21/expert_feedback_20260924_v1/status.json').read_text())
    assert paused['state']=='USER_PAUSED'
    write(OUT/'all_primary_non_neutral_cases.json',cases)
    write(OUT/'posthoc_action_relations.json',diagnostics)
    write(OUT/'verification.json',dict(state='VERIFIED',source_hashes=len(manifest),checks=dict(checks),
        semantic_annotations_independently_verified=False,semantic_training_state=paused['state']))
    print(json.dumps(dict(state='VERIFIED',source_hashes=len(manifest),checks=dict(checks))),flush=True)


if __name__=='__main__':main()
