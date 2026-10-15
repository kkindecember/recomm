"""Reveal outcomes only after review hashes are frozen; descriptive case-control analysis."""
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
from analyze_selective_evidence import sha,write,csv_write
from prepare_blind_disagreements import ROOT,OUT

RELATIONS=['broad','product_type','named','attribute','compatibility','repeat','none']
SPECIFIC={'product_type','named','attribute','compatibility','repeat'}


def stats(rows):
    result=dict(cards=len(rows),X_specific=sum(r['X_specific'] for r in rows),Y_specific=sum(r['Y_specific'] for r in rows),
        both_specific=sum(r['X_specific'] and r['Y_specific'] for r in rows),
        either_specific=sum(r['X_specific'] or r['Y_specific'] for r in rows),
        contrast=sum(r['contrast'] for r in rows),preferences=sum(r['preference']!='abstain' for r in rows),
        abstentions=sum(r['preference']=='abstain' for r in rows),
        contrast_but_abstain=sum(r['contrast'] and r['preference']=='abstain' for r in rows),
        choose_a=sum(r['chosen_role']=='A' for r in rows),choose_b=sum(r['chosen_role']=='B' for r in rows),
        informative=sum(r['stratum']!='neither' for r in rows),
        informative_answered=sum(r['stratum']!='neither' and r['preference']!='abstain' for r in rows),
        informative_correct=sum(r['stratum']!='neither' and r['agrees_with_observed_target'] is True for r in rows),
        informative_wrong=sum(r['stratum']!='neither' and r['agrees_with_observed_target'] is False for r in rows),
        neither_answered=sum(r['stratum']=='neither' and r['preference']!='abstain' for r in rows),
        relation_counts={})
    for rel in RELATIONS:
        group=[r for r in rows if rel in r['X_relations'] or rel in r['Y_relations']]
        result['relation_counts'][rel]=dict(cards=len(group),both_sides=sum(rel in r['X_relations'] and rel in r['Y_relations'] for r in group),
            preferences=sum(r['preference']!='abstain' for r in group),
            informative_correct=sum(r['agrees_with_observed_target'] is True for r in group if r['stratum']!='neither'),
            informative_wrong=sum(r['agrees_with_observed_target'] is False for r in group if r['stratum']!='neither'),
            neither_answered=sum(r['stratum']=='neither' and r['preference']!='abstain' for r in group))
    return result


def main():
    assert not (OUT/'summary.json').exists(),'Do not overwrite an unblinded run'
    freeze=json.loads((OUT/'review_freeze.json').read_text());assert freeze['state']=='FROZEN_BEFORE_UNBLINDING'
    assert sha(OUT/'frozen_review.json')==freeze['frozen_review_sha256']
    assert sha(OUT/'review_entries.json')==freeze['raw_review_sha256']
    assert sha(OUT/'sealed_labels.json')==freeze['sealed_labels_sha256']
    manifest=json.loads((OUT/'review_manifest.json').read_text())
    for p,h in manifest.items():assert sha(ROOT/p)==h
    revealed=datetime.now(timezone.utc).isoformat();assert revealed>freeze['utc']
    review=json.loads((OUT/'frozen_review.json').read_text());sealed=json.loads((OUT/'sealed_labels.json').read_text())
    cards={r['card_id']:r for r in json.loads((OUT/'blind_cards.json').read_text())};rows=[]
    for r in review['cards']:
        s=sealed['cards'][r['card_id']];chosen=0 if r['preference']=='abstain' else s[r['preference']]
        row=dict(**r,user_id=s['user_id'],stratum=s['stratum'],a_side=s['a_side'],b_side=s['b_side'],a=s['a'],b=s['b'],
            target=s['target'],chosen_item=chosen,chosen_role='abstain' if not chosen else 'A' if chosen==s['a'] else 'B',
            agrees_with_observed_target=None if not chosen or s['stratum']=='neither' else chosen==s['target'],
            X_specific=bool(set(r['X_relations'])&SPECIFIC),Y_specific=bool(set(r['Y_relations'])&SPECIFIC),
            history_events=cards[r['card_id']]['history_events'])
        rows.append(row)
    domains={}
    for domain in ('Toys','Beauty'):
        subset=[r for r in rows if r['domain']==domain]
        domains[domain]=dict(population=sealed['population'][domain],review=stats(subset),
            strata={label:stats([r for r in subset if r['stratum']==label]) for label in ('a_hit','b_hit','neither')})
    result=dict(state='COMPLETED',domains=domains,pooled_sample=stats(rows),cards=len(rows),
        frozen_at_utc=freeze['utc'],unblinded_at_utc=revealed,review_sha256=freeze['frozen_review_sha256'],
        exploratory=True,independent_confirmation=False,natural_prevalence_accuracy_estimated=False,
        new_recommendation_policy=False,statistical_fits=0,GRAM_optimizer_steps=0,expert_optimizer_steps=0,
        gpu_inferences=0,test_read=False,semantic_training_state='USER_PAUSED')
    write(OUT/'unblinded_reviews.json',rows);write(OUT/'summary.json',result)
    write(OUT/'all_definite_judgments.json',[dict(review=r,source_card=cards[r['card_id']]) for r in rows if r['preference']!='abstain'])
    for p in [Path(__file__),OUT/'review_freeze.json']:manifest[str(p.relative_to(ROOT))]=sha(p)
    write(OUT/'analysis_manifest.json',manifest)
    write(OUT/'status.json',{k:v for k,v in result.items() if k not in ('domains','pooled_sample')})
    print(json.dumps({d:dict(population=v['population'],review={k:x for k,x in v['review'].items() if k!='relation_counts'},
        strata={k:{f:r[f] for f in ('cards','contrast','preferences','choose_a','choose_b','informative_correct','informative_wrong','neither_answered')} for k,r in v['strata'].items()}) for d,v in domains.items()}),flush=True)


if __name__=='__main__':main()
