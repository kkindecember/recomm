"""Rebuild evidence counts and cached decisions from saved inputs."""
import csv
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/phase21/history_evidence_20260924_v1'
SOURCE=ROOT/'artifacts/phase20/lime_clean_benefit_20260922_v1'
PRIOR=ROOT/'artifacts/phase21/expert_signal_loss_20260924_v1'
spec=importlib.util.spec_from_file_location('audit',ROOT/'experiment/phase21/analyze_history_evidence.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def main():
    results={}
    summary=json.loads((OUT/'summary.json').read_text())
    for domain in ('Toys','Beauty'):
        users=json.loads((OUT/(domain+'_users.json')).read_text());by_user={r['user_id']:r for r in users}
        samples={s['user_id']:s for s in json.loads((SOURCE/domain/'samples.json').read_text())
                 if s['role'] in ('ranker_fit','development')}
        engine=mod.CatalogEvidence(json.loads((SOURCE/domain/'catalog.json').read_text()))
        with (OUT/(domain+'_evidence.csv')).open() as f:
            rows=list(csv.DictReader(f))
        with (PRIOR/(domain+'_candidates.csv')).open() as f:
            prior=list(csv.DictReader(f))
        expected={(r['user_id'],int(r['item'])) for r in prior if int(r['semantic_rank'])<=10 and int(r['LIME_rank'])>10}
        observed={(r['user_id'],int(r['item'])) for r in rows}
        assert len(observed)==len(rows) and observed==expected
        groups={u:[] for u in by_user}
        checked=set();evidence_checks=0
        for row in rows:
            u=row['user_id'];groups[u].append(row)
            assert int(row['target_hit'])==int(int(row['item'])==samples[u]['target'])
            for prefix,donor,item in [('real',u,int(row['item'])),('shuffled',row['donor_user'],int(row['item'])),
                    ('boundary_real',u,by_user[u]['original10']),('boundary_shuffled',row['donor_user'],by_user[u]['original10'])]:
                key=(u,prefix,item)
                if key in checked:continue
                checked.add(key)
                hs=sorted(set(samples[donor]['full_history']))
                brand=engine.brands[item]
                actual_b=[h for h in hs if brand and engine.brands[h]==brand]
                assert actual_b==json.loads(row[prefix+'_brand_support_items'])
                assert len(actual_b)==int(row[prefix+'_brand_count'])
                for label,features,bank in [('word',engine.rare_words[item],engine.words),('phrase',engine.rare_phrases[item],engine.phrases)]:
                    # Recompute by scanning each history item, independent of aggregated history maps.
                    pairs=[(feature,[h for h in hs if feature in bank[h]]) for feature in features]
                    pairs=[p for p in pairs if p[1]]
                    if pairs:
                        feat,support=min(pairs,key=lambda p:(-len(p[1]),p[0]))
                        name=' '.join(feat) if isinstance(feat,tuple) else feat
                    else:name,support='',[]
                    assert name==row[prefix+'_'+label]
                    assert support==json.loads(row[prefix+'_'+label+'_support_items'])
                    assert len(support)==int(row[prefix+'_'+label+'_count'])
                assert int(row[prefix+'_older_exact_item_repeat'])==int(item in hs)
                evidence_checks+=1
        actual_changed=0;user_decisions=0
        for u,data in by_user.items():
            donor=data['donor_user']
            assert samples[u]['role']==samples[donor]['role']
            assert len(samples[u]['full_history'])==len(samples[donor]['full_history'])
            actual_changed+=int(samples[u]['full_history']!=samples[donor]['full_history'])
            selected=groups[u];boundary=data['original10']
            for name,feature in mod.RULE_FEATURES.items():
                for shuffled in (False,True):
                    prefix='shuffled' if shuffled else 'real';rule=('shuffled_' if shuffled else '')+name
                    valid=[r for r in selected if int(r[prefix+'_'+feature])>=2
                           and int(r[prefix+'_'+feature])>int(r['boundary_'+prefix+'_'+feature])]
                    chosen=int(sorted(valid,key=lambda r:(-int(r[prefix+'_'+feature]),int(r['semantic_rank']),int(r['item'])))[0]['item']) if valid else boundary
                    assert chosen==data['choices'][rule];user_decisions+=1
        for role in ('ranker_fit','development'):
            selected=[r for r in rows if r['role']==role]
            for typ in ('real','shuffled'):
                for flag,(feature,threshold) in mod.FEATURES.items():
                    subset=[r for r in selected if int(r[typ+'_'+feature])>=threshold]
                    expected_stats=summary['domains'][domain][role]['associations'][typ][flag]['with_evidence']
                    assert len(subset)==expected_stats['candidates']
                    assert sum(int(r['target_hit']) for r in subset)==expected_stats['targets']
            urows=[u for u in users if u['role']==role]
            for name,rule in summary['domains'][domain][role]['rules'].items():
                gained=sum(u['ranks'][name]<=10 and u['ranks']['original']>10 for u in urows)
                lost=sum(u['ranks'][name]>10 and u['ranks']['original']<=10 for u in urows)
                assert gained==rule['versus_original']['gained10'] and lost==rule['versus_original']['lost10']
        results[domain]=dict(users=len(users),dissent_candidates=len(rows),candidate_set_and_labels_exact=True,
            title_and_brand_evidence_rebuilt=evidence_checks,rule_decisions_rebuilt=user_decisions,
            all_association_counts_and_rule_hit_changes_exact=True,
            actual_history_content_changed_users=actual_changed,
            donor_user_changed_users=sum(u['donor_user']!=u['user_id'] for u in users))
    write=dict(state='VERIFIED_NUMERICAL_OUTPUTS_NOT_MECHANISM',domains=results,
               source='experiment/phase21/verify_history_evidence.py',uses_same_token_normalizer=True,
               independent_history_scan_and_decision_reconstruction=True)
    (OUT/'verification.json').write_text(json.dumps(write,indent=2)+'\n')
    (OUT/'verification_source.py').write_text(Path(__file__).read_text())
    print(json.dumps(write,indent=2))


if __name__=='__main__':main()
