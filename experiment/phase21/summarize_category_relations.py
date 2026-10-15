"""Publish every frozen policy/contrast and the limited interpretation of 21.13."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/phase21/category_relations_20260925_v1'


def main():
    summary=json.loads((OUT/'summary.json').read_text())
    lines=['# 21.13 全部固定政策与对照','',
           '六组×四个政策×五个评价集合×两域。主比较为primary / budget10；未根据效果选择预算。',
           'N10区间为主比较或训练折外的描述性用户bootstrap区间，未校正多重比较。','',
           '|域|集合|组／政策|人数|修改|新增|丢失|净值|N10|ΔN10|95%区间|',
           '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    policies=contrasts=0
    for domain,s in summary['domains'].items():
        for cohort,c in s['cohorts'].items():
            for arm,m in c['policies'].items():
                ci=m.get('ci95');interval='—' if ci is None else f'[{ci[0]:+.8f}, {ci[1]:+.8f}]'
                lines.append(f'|{domain}|{cohort}|{arm}|{m["users"]}|{m["actions"]}|{m["gained10"]}|{m["lost10"]}|{m["net_hits"]:+d}|{m["N10"]:.9f}|{m["delta_N10"]:+.9f}|{interval}|');policies+=1
    lines+=['','|域|集合|固定对照／政策|净命中差|ΔN10|95%区间|','|---|---|---|---:|---:|---|']
    for domain,s in summary['domains'].items():
        for cohort,c in s['cohorts'].items():
            for name,m in c['contrasts'].items():
                ci=m.get('ci95');interval='—' if ci is None else f'[{ci[0]:+.8f}, {ci[1]:+.8f}]'
                lines.append(f'|{domain}|{cohort}|{name}|{m["net_hit_difference"]:+d}|{m["delta_N10"]:+.9f}|{interval}|');contrasts+=1
    lines+=['','## 不拟合的固定规则','','|域|原列表|集合|依据|人数|修改|新增|丢失|净值|ΔN10|',
            '|---|---|---|---|---:|---:|---:|---:|---:|---:|']
    for domain,s in summary['domains'].items():
        for context,source in [('fused',s['direct']),('guided',s['guided']['cohorts'])]:
            for cohort,c in source.items():
                for arm in ('category','shuffled_category'):
                    m=c[arm]['rule'] if context=='fused' else c[arm]
                    lines.append(f'|{domain}|{context}|{cohort}|{arm}|{m["users"]}|{m["actions"]}|{m["gained10"]}|{m["lost10"]}|{m["net_hits"]:+d}|{m["delta_N10"]:+.9f}|')
    (OUT/'all_tables.md').write_text('\n'.join(lines)+'\n')
    interpretation=dict(category_incremental_value_confirmed=False,dual_domain_rule_supported=False,
        category_not_identical_to_function_or_series=True,new_GRAM_generation_result=False,
        semantic_feedback_training_still_paused=True,automatic_gpu_launch=False,
        exploratory=True,independent_confirmation=False,policy_rows=policies,contrast_rows=contrasts,
        scope='Current catalogue-category definitions, fixed ridge probe, and one-position cached-list rule only.',
        quality_review=dict(blind_catalog_items=64,reviewer='same assistant session; not an independent human semantic annotation',
            examples_of_granularity_ambiguity=[{'domain':'Toys','item':5952,'issue':'Storage accessory title; card-game category.'},
                {'domain':'Beauty','item':4504,'issue':'Wig-stand title; wig category.'},
                {'domain':'Beauty','item':1587,'issue':'Microcurrent device title; toners/astringents category.'}],
            no_semantic_ground_truth_accuracy_claim=True))
    (OUT/'interpretation.json').write_text(json.dumps(interpretation,ensure_ascii=False,indent=2)+'\n')
    locations={}
    for domain in summary['domains']:
        with (OUT/domain/'guided_rule.csv').open() as f:rows=list(csv.DictReader(f))
        locations[domain]={}
        for cohort in ('primary','all'):
            gains=[r for r in rows if r['category_delta']=='1' and (cohort=='all' or r['cohort']==cohort)]
            locations[domain][cohort]=dict(gains_original_rank11_50=sum(10<int(r['old_rank'])<=50 for r in gains),
                gains_original_outside50=sum(int(r['old_rank'])>50 for r in gains))
    (OUT/'guided_gain_locations.json').write_text(json.dumps(locations,indent=2)+'\n')
    print(json.dumps(interpretation,ensure_ascii=False))


if __name__=='__main__':main()
