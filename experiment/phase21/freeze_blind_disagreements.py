"""Validate and hash every source-only review before revealing the outcome mapping."""
from datetime import datetime,timezone
import json
from pathlib import Path
from analyze_selective_evidence import sha,write
from prepare_blind_disagreements import ROOT,OUT

TYPES={'broad','product_type','named','attribute','compatibility','repeat','none'}
DESCRIPTION_READS={'T011':['X','Y'],'T018':['X','Y'],'T047':['X','Y'],'T058':['Y'],
    'T061':['X','Y','H4','H7'],'T064':['X','Y'],'B011':['X','Y'],'B025':['X'],
    'B031':['Y','H3'],'B034':['Y'],'B043':['Y'],'B045':['X','Y'],'B054':['Y']}


def main():
    assert not (OUT/'review_freeze.json').exists(),'Review already frozen; do not overwrite'
    manifest=json.loads((OUT/'preparation_manifest.json').read_text())
    for p,h in manifest.items():assert sha(ROOT/p)==h
    cards={r['card_id']:r for r in json.loads((OUT/'blind_cards.json').read_text())}
    rows=json.loads((OUT/'review_entries.json').read_text());assert len(rows)==len(cards)==128
    assert len({r[0] for r in rows})==128 and {r[0] for r in rows}==set(cards)
    reviewed=[]
    for r in rows:
        assert len(r)==8
        cid,xt,yt,xh,yh,contrast,pref,note=r
        assert set(xt+yt)<=TYPES and xt and yt and note
        assert pref in ('X','Y','abstain') and isinstance(contrast,bool)
        assert pref=='abstain' or contrast
        valid={h['ref'] for h in cards[cid]['history']};assert set(xh+yh)<=valid
        assert 'none' not in xt or xt==['none'];assert 'none' not in yt or yt==['none']
        reviewed.append(dict(card_id=cid,domain=cards[cid]['domain'],X_relations=xt,Y_relations=yt,
            X_history=xh,Y_history=yh,contrast=contrast,preference=pref,reason=note,
            description_reads=DESCRIPTION_READS.get(cid,[])))
    write(OUT/'frozen_review.json',dict(reviewer='same assistant; outcomes and candidate roles hidden, not independent or double blind',
        relation_types=sorted(TYPES),cards=reviewed,source_description_reads=DESCRIPTION_READS,
        limitations=['retrospective outcome-enriched case-control sample','no population accuracy or recommendation lift claim',
            'most judgments from titles and complete histories; description checks targeted, not exhaustive',
            'source-relation tags and preference are subjective single-reviewer hypotheses']))
    for p in [Path(__file__),OUT/'review_entries.json',OUT/'frozen_review.json']:
        manifest[str(p.relative_to(ROOT))]=sha(p)
    write(OUT/'review_manifest.json',manifest)
    freeze=dict(state='FROZEN_BEFORE_UNBLINDING',utc=datetime.now(timezone.utc).isoformat(),cards=128,
        frozen_review_sha256=sha(OUT/'frozen_review.json'),raw_review_sha256=sha(OUT/'review_entries.json'),
        sealed_labels_sha256=sha(OUT/'sealed_labels.json'),review_manifest_sha256=sha(OUT/'review_manifest.json'))
    write(OUT/'review_freeze.json',freeze)
    write(OUT/'status.json',dict(state='REVIEW_FROZEN',cards=128,labels_exposed_to_reviewer=False,
        semantic_training_state='USER_PAUSED',statistical_fits=0,gpu_inferences=0))
    print(json.dumps(freeze),flush=True)


if __name__=='__main__':main()
