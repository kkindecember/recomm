"""All frozen results, source-grounded cases and explicitly posthoc missingness audit."""
import csv
import hashlib
import json
import re
from pathlib import Path

from analyze_history_evidence import fields
from analyze_selective_evidence import write

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/phase21/explicit_relations_20260925_v1'
RAW=ROOT/'artifacts/phase20/lime_clean_benefit_20260922_v1'
FULL=ROOT/'artifacts/phase21/full_validation_20260924_v1'


def main():
    summary=json.loads((OUT/'summary.json').read_text())
    lines=['# 21.14 全部预设结果','',
        '仅series通过两域来源初筛；explicit适配／扩展未进行效果评价。表内不是新的GRAM生成成绩。','',
        '|域|集合|政策|用户|修改|新增|丢失|净值|N10|ΔN10|描述性95%区间|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for domain,relations in summary['domains'].items():
        for relation,cohorts in relations.items():
            for cohort,c in cohorts.items():
                for policy,m in c['policies'].items():
                    ci=m.get('ci95');interval='—' if ci is None else f'[{ci[0]:+.8f}, {ci[1]:+.8f}]'
                    lines.append(f'|{domain}|{cohort}|{relation}/{policy}|{m["users"]}|{m["actions"]}|{m["gained10"]}|{m["lost10"]}|{m["net_hits"]:+d}|{m["N10"]:.9f}|{m["delta_N10"]:+.9f}|{interval}|')
    lines+=['','|域|集合|对照|净命中差|ΔN10|95%区间|','|---|---|---|---:|---:|---|']
    for domain,relations in summary['domains'].items():
        for relation,cohorts in relations.items():
            for cohort,c in cohorts.items():
                for name,m in c['contrasts'].items():
                    ci=m.get('ci95');interval='—' if ci is None else f'[{ci[0]:+.8f}, {ci[1]:+.8f}]'
                    lines.append(f'|{domain}|{cohort}|{name}|{m["net_hit_difference"]:+d}|{m["delta_N10"]:+.9f}|{interval}|')
    (OUT/'all_tables.md').write_text('\n'.join(lines)+'\n')
    missing={};examples={};donor_audit={}
    for domain in ('Toys','Beauty'):
        relations=json.loads((OUT/(domain+'_relations.json')).read_text())
        group_of={i:(k,set(v)) for k,v in relations['series_groups'].items() for i in v}
        catalog=json.loads((RAW/domain/'catalog.json').read_text());meta=[fields(p) for p in catalog['passages']]
        samples={s['user_id']:s for s in json.loads((FULL/domain/'samples.json').read_text())}
        with (OUT/domain/'validation_series_decisions.csv').open() as f:rows=list(csv.DictReader(f))
        donor_audit[domain]={}
        for kind in ('train','validation'):
            prior=ROOT/'artifacts/phase21/selective_evidence_20260925_v1'/domain/kind/'users.csv'
            with prior.open() as f:old={r['user_id']:r for r in csv.DictReader(f)}
            with (OUT/domain/(kind+'_series_decisions.csv')).open() as f:dec=list(csv.DictReader(f))
            for cohort in (('train_holdout',) if kind=='train' else ('primary','all')):
                action=[r for r in dec if int(r['shuffled_item'])>0 and (cohort in ('all','train_holdout') or r['cohort']==cohort)]
                donor_audit[domain][cohort]=dict(shuffled_actions=len(action),unchanged_history_donor_actions=
                    sum(old[r['user_id']]['donor_user']==r['user_id'] for r in action))
        missing[domain]={};examples[domain]=[]
        for cohort in ('primary','all'):
            take=[r for r in rows if int(r['relation_item'])>0 and (cohort=='all' or r['cohort']==cohort)]
            missing[domain][cohort]={}
            for known in (True,False):
                sub=[r for r in take if (int(r['boundary']) in group_of)==known];delta=[int(r['relation_delta']) for r in sub]
                missing[domain][cohort]['boundary_series_'+('known' if known else 'unknown')]=dict(
                    actions=len(sub),gained10=delta.count(1),lost10=delta.count(-1))
        selected=[r for r in rows if r['cohort']=='primary' and int(r['relation_delta'])!=0]
        for r in sorted(selected,key=lambda r:hashlib.sha256(r['user_id'].encode()).hexdigest()):
            a,b=int(r['relation_item']),int(r['boundary']);h=set(samples[r['user_id']]['full_history']);key,members=group_of[a]
            def card(i):return dict(item=i,title=meta[i].get('title',''),series=group_of.get(i,('',set()))[0])
            examples[domain].append(dict(selection='all_non_neutral_primary_cases_sorted_by_user_hash',
                user_id=r['user_id'],target=int(r['target']),hit_change=int(r['relation_delta']),
                proposal=card(a),boundary=card(b),supporting_history=[card(i) for i in sorted((h&members)-{a})]))
    write(OUT/'posthoc_boundary_missingness.json',dict(posthoc_descriptive_only=True,new_policy_evaluated=False,domains=missing))
    write(OUT/'all_primary_non_neutral_cases.json',examples)
    write(OUT/'shuffled_donor_action_audit.json',donor_audit)
    # Posthoc source-only illustration of the strict full-title matcher's limited recall.
    beauty=json.loads((RAW/'Beauty/catalog.json').read_text());snippets=[]
    for i,p in enumerate(beauty['passages'][1:],1):
        f=fields(p)
        for sentence in re.split(r'[.!?;\n]',f.get('description','')):
            if re.search(r'\b(?:compatible with|for use with|designed for use with)\b',sentence):
                snippets.append(dict(item=i,title=f.get('title',''),source_sentence=sentence.strip()))
    snippets=sorted(snippets,key=lambda r:hashlib.sha256(str((r['item'],r['title'],r['source_sentence'])).encode()).digest())[:18]
    write(OUT/'posthoc_unlinked_beauty_source_sample.json',dict(posthoc_source_audit=True,relation_edges_not_added=True,
        selection='SHA256 of item/title/sentence tuple, first 18',examples=snippets))
    write(OUT/'interpretation.json',dict(series_mechanism_confirmed=False,explicit_relation_effect_evaluated=False,
        low_coverage_does_not_prove_absence_of_semantic_relations=True,unknown_is_not_negative_evidence=True,
        missingness_subgroups_posthoc_only=True,no_new_unknown_boundary_policy_tested=True,
        GRAM_reading_semantic_feedback_evaluated=False,semantic_feedback_training_still_paused=True))
    print(json.dumps(dict(non_neutral_primary_cases={d:len(v) for d,v in examples.items()},missingness=missing),ensure_ascii=False))


if __name__=='__main__':main()
