"""Independent direct-history recomputation of all six policies and source invariants."""
from collections import Counter
import csv
import json
import re
from pathlib import Path
import numpy as np
from analyze_selective_evidence import sha,write
from analyze_history_evidence import fields,tokens
from entity_relation_core import normalize
from typed_relation_core import version_fields,TypedRelations,comparable_support
from extract_typed_relations import ROOT,OUT,RAW,PREVIOUS

POLICIES=['previous','typed_parent','hierarchy','fine_only','score_matched','shuffled']


def main():
    manifest=json.loads((OUT/'evaluation_manifest.json').read_text())
    for p,h in manifest.items():assert sha(ROOT/p)==h
    summary=json.loads((OUT/'summary.json').read_text());definitions=json.loads((OUT/'definitions.json').read_text())
    counter=Counter();cases={};comparability={}
    for domain in ('Toys','Beauty'):
        data=json.loads((OUT/(domain+'_relations.json')).read_text());previous=json.loads((PREVIOUS/(domain+'_relations.json')).read_text())
        catalog=json.loads((RAW/domain/'catalog.json').read_text());records={r['item']:r for r in data['items']}
        oldlabels={r['item']:r['family'] for r in previous['items'] if r['family']}
        names={n:tuple(n.split()) for n in data['names']}
        families={};finekeys={}
        for i,r in records.items():
            raw=tokens(fields(catalog['passages'][i]).get('title',''));cut=len(raw)
            for k,t in enumerate(raw):
                if t in definitions['boundaries']:cut=k;break
            assert r['subject_tokens']==raw[:cut] and r['boundary_index']==cut
            nt=tuple(normalize(raw[:cut]));matches=[n for n,p in names.items() if any(nt[j:j+len(p)]==p for j in range(len(nt)-len(p)+1))]
            assert sorted(matches)==r['own_matches']
            if r['own_family']:
                assert r['own_family'] in matches;families[i]=r['own_family']
            for kind,field in r['versions'].items():
                assert field['values']==sorted(set(e['value'] for e in field['evidence']))
                assert field['status']==('known' if len(field['values'])==1 else 'absent' if not field['values'] else 'ambiguous')
                assert field['value']==(field['values'][0] if field['status']=='known' else '')
                for e in field['evidence']:
                    assert ' '.join(raw[e['start']:e['end']])==e['span'] and e['end']<=cut
                    if kind=='scope':assert e['value'] in ('tv','comic','movie')
                    if kind=='year':
                        assert 1900<=int(e['value'])<=2099
                        assert any(t in set(definitions['seasons'])|{'collection'} for t in raw[e['start']:e['end']])
                    if kind=='batch':assert e['span'].startswith('series ')
                    counter['version_evidence_spans']+=1
            counter['catalog_source_records']+=1
        assert families=={i:f for f,ids in data['families'].items() for i in ids}
        for f,ids in data['families'].items():
            required=[k for k in ('scope','batch','year') if any(records[i]['versions'][k]['values'] for i in ids)]
            assert required==data['requirements'][f]
            for i in ids:
                if required and all(records[i]['versions'][k]['status']=='known' for k in required):
                    finekeys[i]=(f,)+tuple(records[i]['versions'][k]['value'] for k in required)
        cases[domain]=[];comparability[domain]={}
        for kind in ('train','validation'):
            folder=ROOT/'artifacts/phase21/selective_evidence_20260925_v1'/domain/kind
            users=list(csv.DictReader((folder/'users.csv').open()));saved=list(csv.DictReader((OUT/domain/(kind+'_decisions.csv')).open()))
            prior=list(csv.DictReader((PREVIOUS/domain/(kind+'_decisions.csv')).open()))
            sample_path=(RAW if kind=='train' else ROOT/'artifacts/phase21/full_validation_20260924_v1')/domain/'samples.json'
            samples={s['user_id']:s for s in json.loads(sample_path.read_text())}
            with np.load(folder/'pairs.npz') as z:items=z['items'];labels=z['labels']
            score=np.load(folder/'score.npy',mmap_mode='r')[:len(items)]
            with np.load(OUT/domain/(kind+'_support.npz')) as z:support=z['counts'];levels=z['levels']
            assert len(users)==len(saved)==len(prior)
            labeled_pairs=Counter()
            for r,out,old in zip(users,saved,prior):
                assert r['user_id']==out['user_id']==old['user_id']
                a0,a1=int(r['start']),int(r['end']);candidate=items[a0:a1].tolist();b=int(r['boundary']);target=int(r['target'])
                real=set(samples[r['user_id']]['full_history']);wrong=set(samples[r['donor_user']]['full_history'])
                def counts(mapping,history):
                    counted=Counter(mapping[h] for h in history if h in mapping)
                    return {i:counted.get(mapping.get(i,''),0)-int(i in history and i in mapping) for i in candidate+[b]}
                op=counts(oldlabels,real);pp=counts(families,real);fp=counts(finekeys,real)
                wp=counts(families,wrong);wf=counts(finekeys,wrong)
                full=[];lev=[]
                for a in candidate:
                    old_pair=[op[a],op[b]] if a in oldlabels and b in oldlabels else [-1,-1]
                    par=[pp[a],pp[b]] if a in families and b in families else [-1,-1]
                    pair=[-1,-1];donor_pair=[-1,-1];level=0
                    if a in families and b in families:
                        if families[a]!=families[b]:pair=par;donor_pair=[wp[a],wp[b]];level=1
                        elif a in finekeys and b in finekeys and finekeys[a]!=finekeys[b]:
                            pair=[fp[a],fp[b]];donor_pair=[wf[a],wf[b]];level=2
                    full.append(old_pair+par+pair+donor_pair);lev.append(level)
                full=np.array(full);lev=np.array(lev);assert np.array_equal(full,support[a0:a1])
                assert np.array_equal(lev,levels[a0:a1]);counter['candidate_support_rows']+=len(candidate)
                assert np.array_equal(labels[a0:a1],[int(a==target)-int(b==target) for a in candidate])
                def pick(col,versions=False):
                    allowed=[j for j in range(len(candidate)) if full[j,col]>0 and full[j,col]>full[j,col+1] and (not versions or lev[j]==2)]
                    if not allowed:return 0
                    allowed.sort(key=lambda j:(-full[j,col],-float(score[a0+j,18]),candidate[j]))
                    return candidate[allowed[0]]
                expected=[pick(0),pick(2),pick(4),pick(4,True),0,pick(6)]
                if expected[2]:expected[4]=candidate[min(range(len(candidate)),key=lambda j:(-float(score[a0+j,18]),candidate[j]))]
                assert expected[0]==int(old['normalized_known_item'])
                for policy,choice in zip(POLICIES,expected):
                    assert int(out[policy+'_item'])==choice
                    assert int(out[policy+'_delta'])==int(bool(choice))*(int(choice==target)-int(b==target))
                    if choice:assert choice in candidate and choice not in [int(n) for n in re.findall(r'\d+',r['top9'])]+[b]
                    counter['policy_decisions']+=1
                if kind=='validation' and r['cohort']=='primary':
                    for j,a in enumerate(candidate):
                        label='helpful' if a==target else 'harmful' if b==target else 'neutral'
                        labeled_pairs[label+'_all']+=1
                        if lev[j]:labeled_pairs[label+'_comparable']+=1
                        if lev[j]==2:labeled_pairs[label+'_version']+=1
                    if any(int(out[p+'_delta']) for p in POLICIES):
                        included=sorted({b,target}|{int(out[p+'_item']) for p in POLICIES if int(out[p+'_item'])})
                        cases[domain].append(dict(decision=out,items=[dict(item=i,title=records[i]['title'],
                            family=families.get(i,''),fine_key=finekeys.get(i),
                            parent_history=sorted(h for h in real if h!=i and i in families and families.get(h)==families[i]),
                            version_history=sorted(h for h in real if h!=i and i in finekeys and finekeys.get(h)==finekeys[i])) for i in included]))
            comparability[domain][kind]=dict(labeled_pairs)
            cohorts=['train_holdout'] if kind=='train' else ['primary','previous512','seen_fit_user_later_event','all']
            for cohort in cohorts:
                rows=[r for r in saved if cohort in ('all','train_holdout') or r['cohort']==cohort];part=summary['domains'][domain][cohort]
                assert len(rows)==part['users']
                for p in POLICIES:
                    m=part['policies'][p];delta=[int(r[p+'_delta']) for r in rows]
                    assert m['gained10']==delta.count(1) and m['lost10']==delta.count(-1) and m['net_hits']==sum(delta)
                    assert m['actions']==sum(int(r[p+'_item'])>0 for r in rows)
                    ndcg=sum((1/np.log2(int(r['old_rank'])+1) if int(r['old_rank'])<=10 else 0)+int(r[p+'_delta'])/np.log2(11) for r in rows)/len(rows)
                    assert abs(ndcg-m['N10'])<1e-12;counter['aggregate_rows']+=1
                for name,c in part['contrasts'].items():
                    control=name.removeprefix('hierarchy_vs_')
                    assert c['net_hit_difference']==sum(int(r['hierarchy_delta'])-int(r[control+'_delta']) for r in rows)
                if cohort=='primary':
                    for label in ('helpful','harmful','neutral'):
                        for short,long in [('all','all_pairs'),('comparable','comparable_pairs'),('version','version_pairs')]:
                            assert part['label_groups'][label][long]==labeled_pairs[label+'_'+short]
    # Frozen extraction hazards and same-family comparability, independent of user outcomes.
    for sentence in ('vidal sassoon pro series 1 inch','vidal sassoon pro series 1.5 inch'):
        f,sizes=version_fields(tokens(sentence));assert f['batch']['status']=='absent' and sizes
    assert version_fields(tokens('model 2013 device'))[0]['year']['status']=='absent'
    assert version_fields(tokens('summer 2013 collection'))[0]['year']['value']=='2013'
    assert version_fields(tokens('series fourth'))[0]['batch']['value']=='4'
    assert version_fields(tokens('summer 2012 collection 2013'))[0]['year']['status']=='ambiguous'
    cat={'passages':['','title: alpha beta series 3','title: alpha beta series 4','title: device for alpha beta series 4',
                     'title: alpha beta figure','title: alpha beta series 4 blue','title: alpha beta series 4 red'],
         'items':['pad','a','b','c','d','e','f']}
    prior={'seeds':{'alpha beta':[]},'items':[dict(item=i,family='alpha beta') for i in range(1,7)]}
    engine=TypedRelations(cat,prior,'regression')
    assert engine.items[3]['identity_status']=='mention_only' and engine.items[3]['versions']['batch']['status']=='absent'
    assert comparable_support(engine.items,engine.requirements,2,1,[5,6,4],mode='hierarchy')['a_support']==2
    assert comparable_support(engine.items,engine.requirements,2,4,[5,6],mode='hierarchy') is None
    assert comparable_support(engine.items,engine.requirements,2,5,[6],mode='hierarchy') is None
    counter['regression_assertions']=10
    review=json.loads((OUT/'source_review.json').read_text());quality=json.loads((OUT/'quality_gate.json').read_text())
    for domain,r in review['domains'].items():
        identities={x['item']:x for x in r['identity']};identities.update({x['item']:x for x in r['role_changes']})
        for kind,rows in [('identity',list(identities.values())),('version',r['version'])]:
            q=quality['domains'][domain][kind];good=sum(x['verdict']=='supported' for x in rows)
            assert q['units']==len(rows) and q['supported']==good and q['passed']==(len(rows)>=10 and good/len(rows)>=.9)
    paused=json.loads((ROOT/'artifacts/phase21/expert_feedback_20260924_v1/status.json').read_text())
    assert paused['state']=='USER_PAUSED'
    write(OUT/'all_primary_non_neutral_cases.json',cases)
    write(OUT/'comparability_verification.json',comparability)
    write(OUT/'verification.json',dict(state='VERIFIED',source_hashes=len(manifest),checks=dict(counter),
        semantic_annotation_independently_verified=False,semantic_training_state=paused['state']))
    print(json.dumps(dict(state='VERIFIED',source_hashes=len(manifest),checks=dict(counter))),flush=True)


if __name__=='__main__':main()
