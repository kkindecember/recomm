"""Descriptive gain/loss location; outcome-defined strata are not causal tests."""
import csv
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/phase21/candidate_gate_20261010_v1'


def main():
    with (ROOT/'artifacts/phase21/full_opportunity_counts_20260925_v1/per_user.csv').open() as f:old=list(csv.DictReader(f))
    result={}
    for d in ('Toys','Beauty'):
        users=json.loads((OUT/d/'validation/users.json').read_text());r=np.load(OUT/d/'ranks.npz')
        previous=[v for v in old if v['domain']==d]
        assert [v['user_id'] for v in previous]==[v['user_id'] for v in users]
        expert=np.array([[int(v[k]) for k in ('SASRec','ItemCF','semantic')] for v in previous])
        result[d]={}
        for co in ('primary','all'):
            mask=np.ones(len(users),bool) if co=='all' else np.array([u['cohort']=='primary' for u in users])
            rows=[];old_hit=r['original']<=10;one_expert=expert.min(1)<=10
            for seed in (0,1,2):
                new_hit=r[f'score_{seed}']<=10
                gained=mask&~old_hit&new_hit;lost=mask&old_hit&~new_hit
                user_hit=r[f'user_{seed}']<=10
                rows.append(dict(seed=seed,gained_vs_original=int(gained.sum()),lost_vs_original=int(lost.sum()),
                    gained_from_expert_top10=int((gained&one_expert).sum()),
                    gained_without_any_expert_top10=int((gained&~one_expert).sum()),
                    lost_original_synergy=int((lost&~one_expert).sum()),
                    gained_vs_user=int((mask&~user_hit&new_hit).sum()),lost_vs_user=int((mask&user_hit&~new_hit).sum())))
            result[d][co]=dict(users=int(mask.sum()),per_seed=rows)
    (OUT/'gain_loss.json').write_text(json.dumps(dict(results=result,
        interpretation='Descriptive target-rank strata; not causal attribution or achievable oracle opportunity.'),indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
