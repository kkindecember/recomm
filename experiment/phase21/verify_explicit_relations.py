"""Independent source-span, mapping, selection and accounting checks for 21.14."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_history_evidence import fields,tokens
from analyze_selective_evidence import sha,write
from evidence_decomposition_core import read_users

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/phase21/explicit_relations_20260925_v1'
PRIOR=ROOT/'artifacts/phase21/selective_evidence_20260925_v1'
RAW=ROOT/'artifacts/phase20/lime_clean_benefit_20260922_v1'
FULL=ROOT/'artifacts/phase21/full_validation_20260924_v1'


def check_domain(domain,summary):
    relations=json.loads((OUT/(domain+'_relations.json')).read_text())
    catalog=json.loads((RAW/domain/'catalog.json').read_text());meta=[fields(p) for p in catalog['passages']]
    group_of={};groups={k:set(v) for k,v in relations['series_groups'].items()};definitions=json.loads((OUT/'definitions.json').read_text())
    for key,ids in groups.items():
        assert len(ids)>=2
        prefix=tuple(key.split())
        for i in ids:
            title=tokens(meta[i]['title']);assert tuple(title[:len(prefix)])==prefix
            assert 'series' in prefix or 'collection' in prefix;assert len(prefix)<=12
            assert i not in group_of;group_of[i]=key
    evidence_checks=0
    for edge in relations['explicit_edges']:
        a,b=edge['source'],edge['target'];assert a!=b
        for evidence in edge['evidence']:
            source=meta[a][evidence['field']]
            assert evidence['source_sentence'] in source
            target=tokens(meta[b]['title']);assert ' '.join(target)==evidence['matched_title']
            assert 2<=len(target)<=12
            if evidence['method']=='title_expansion':
                title=tokens(source);pre=title[:title.index('expansion')]
                while pre and (pre[-1].isdigit() or pre[-1] in definitions['ordinals']):pre.pop()
                assert pre==target
            else:
                sentence=evidence['source_sentence'];cue=evidence['cue'];start=sentence.lower().find(cue)
                # The same cue can occur twice; at least one recorded occurrence must reproduce its tail span.
                matches=False
                while start>=0:
                    tail=tokens(sentence[start+len(cue):])[:40]
                    lo,hi=evidence['tail_start'],evidence['tail_end']
                    matches=matches or tail[lo:hi]==target
                    start=sentence.lower().find(cue,start+1)
                assert matches
            evidence_checks+=1
    result=dict(series_items_checked=len(group_of),source_evidence_spans_checked=evidence_checks,
        candidate_relation_counts_checked=0,policy_user_decisions=0,all_metrics_exact=True)
    for relation,cohorts in summary.items():
        assert relation=='series' # This batch's quality decision allowed only series; fail closed on a change.
        for kind in ('train','validation'):
            users=read_users(PRIOR/domain/kind/'users.csv')
            with np.load(PRIOR/domain/kind/'pairs.npz') as z:items=z['items']
            score=np.load(PRIOR/domain/kind/'score.npy',mmap_mode='r')[:len(items)]
            samples={r['user_id']:r for r in json.loads(((RAW if kind=='train' else FULL)/domain/'samples.json').read_text())}
            with (OUT/domain/(kind+'_series_decisions.csv')).open() as f:rows=list(csv.DictReader(f))
            with np.load(OUT/domain/(kind+'_series_relations.npz')) as z:support=z['support'];qualifying=z['qualifying']
            assert len(users)==len(rows)
            for u,r in zip(users,rows):
                assert r['user_id']==u['user_id'];lo,hi=u['start'],u['end'];a=items[lo:hi];b=u['boundary']
                hist=set(samples[u['user_id']]['full_history']);wrong=set(samples[u['donor_user']]['full_history'])
                assert len(samples[u['user_id']]['full_history'])==len(samples[u['donor_user']]['full_history'])
                if kind=='train':
                    def fold(uid):return int.from_bytes(hashlib.sha256(f'20260925|selective-evidence|{domain}|{uid}'.encode()).digest()[:8],'big')%5
                    assert fold(u['user_id'])==fold(u['donor_user'])==u['fold']
                else:
                    s,t=samples[u['user_id']],samples[u['donor_user']]
                    assert (s['training_holdout_role'],s['previously_evaluated'])==(t['training_holdout_role'],t['previously_evaluated'])
                def number(item,history):
                    group=group_of.get(int(item))
                    return sum(h!=item and h in groups[group] for h in history) if group else 0
                counts=[number(int(i),hist) for i in a];bcount=number(b,hist);bad=[number(int(i),wrong) for i in a]
                np.testing.assert_array_equal(support[lo:hi],np.column_stack([counts,[bcount]*len(a),bad]))
                np.testing.assert_array_equal(qualifying[lo:hi],[c>=1 and c>bcount for c in counts])
                result['candidate_relation_counts_checked']+=len(a)
                def select(counts,bcount):
                    options=[(-counts[j],-float(score[lo+j,18]),int(a[j])) for j in range(len(a)) if counts[j]>=1 and counts[j]>bcount]
                    return min(options)[2] if options else 0
                real=select(counts,bcount);shuffled=select(bad,number(b,wrong))
                baseline=min((-float(score[lo+j,18]),int(a[j])) for j in range(len(a)))[1] if real else 0
                for policy,choice in [('relation',real),('score_matched',baseline),('shuffled',shuffled)]:
                    change=(int(choice==u['target'])-int(b==u['target'])) if choice else 0
                    assert choice==int(r[policy+'_item']) and change==int(r[policy+'_delta'])
                    result['policy_user_decisions']+=1
            for cohort in (('train_holdout',) if kind=='train' else ('primary','previous512','seen_fit_user_later_event','all')):
                take=[r for r in rows if cohort in ('all','train_holdout') or r['cohort']==cohort]
                for policy in ('relation','score_matched','shuffled'):
                    deltas=[int(r[policy+'_delta']) for r in take];new=[]
                    for r,d in zip(take,deltas):
                        old=int(r['old_rank']);rank=10 if d==1 else 2**30 if d==-1 else old
                        new.append(1/np.log2(rank+1) if rank<=10 else 0.)
                    m=cohorts[cohort]['policies'][policy]
                    assert (len(take),sum(int(r[policy+'_item'])>0 for r in take),deltas.count(1),deltas.count(-1),sum(deltas))==(m['users'],m['actions'],m['gained10'],m['lost10'],m['net_hits'])
                    assert abs(float(np.mean(new))-m['N10'])<1e-12
                real=np.array([int(r['relation_delta']) for r in take])
                for policy in ('score_matched','shuffled'):
                    control=np.array([int(r[policy+'_delta']) for r in take]);m=cohorts[cohort]['contrasts']['relation_vs_'+policy]
                    assert int((real-control).sum())==m['net_hit_difference']
    return result


def main():
    summary=json.loads((OUT/'summary.json').read_text());assert summary['state']=='COMPLETED'
    manifests=[json.loads((OUT/name).read_text()) for name in ('manifest.json','quality_audit_manifest.json','evaluation_manifest.json')]
    hashes={}
    for m in manifests:hashes.update(m)
    for rel,digest in hashes.items():assert sha(ROOT/rel)==digest,rel
    labels=json.loads((OUT/'source_audit_labels.json').read_text())['labels'];quality=json.loads((OUT/'quality_gate.json').read_text())
    for d in ('Toys','Beauty'):
        cards=json.loads((OUT/(d+'_relations.json')).read_text())['blind_audit_cards']
        assert {r['audit_id'] for r in cards}=={r['audit_id'] for r in labels if r['domain']==d}
        for kind in ('series','explicit'):
            rows=[r for r in labels if r['domain']==d and r['kind']==kind];supported=sum(r['judgment']=='supported' for r in rows)
            q=quality['domains'][d][kind];assert (q['audited'],q['supported'])==(len(rows),supported)
            assert q['gate_pass']==(len(rows)>=10 and supported/len(rows)>=.9)
    assert quality['eligible_relation_types']==summary['eligible_relations']==['series']
    result={d:check_domain(d,summary['domains'][d]) for d in ('Toys','Beauty')}
    result.update(source_hashes_checked=len(hashes),quality_gate_recomputed=True,
        semantic_judgments_not_independently_validated=True,verification_script_sha256=sha(Path(__file__)))
    write(OUT/'independent_verification.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
