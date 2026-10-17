"""CPU-only diagnosis from completed predictions; never use an active run's outcomes."""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/phase21'
LONG=BASE/'fixed_expert_screen_4096_20261017_v1'
PAIR=BASE/'selfmatch_history_1024_20261017_v1'
SINGLE=BASE/'single_switch_controls_1024_20261017_v1'
OUT=BASE/'expert_complementarity_diagnostic_20261017_v1'
PLAN=ROOT/'plan/第二十一阶段/21.57_专家互补性与候选覆盖CPU诊断_20261017.md'
REPORT=ROOT/'report/第二十一阶段/专家互补性与候选覆盖CPU诊断_20261017.md'
from experiment.phase20.catalog_scope_core import metrics,contrast


def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def utility(r):return (r<=10)/np.log2(r+1)
def rows(p):return [json.loads(line) for line in p.read_text().splitlines()]


def order(p):
    return np.array([i for i,_ in sorted(zip(p['ids'],p['scores']),key=lambda z:(-z[1],z[0]))])


def top50(s):
    threshold=np.partition(s,len(s)-50)[len(s)-50]
    above=np.flatnonzero(s>threshold)
    ties=np.flatnonzero(s==threshold)[:50-len(above)]
    ix=np.r_[above,ties]
    return ix[np.lexsort((ix,-s[ix]))]+1


def gaps(scores):
    top=np.sort(np.partition(scores,len(scores)-11)[-11:])[::-1]
    return np.array([top[0]-top[1],top[9]-top[10]],float)


def score_gaps(cache,samples,penalty):
    out=np.empty((len(samples),2))
    for j,s in enumerate(samples):
        score=np.asarray(cache[j,:,0]).copy()
        if penalty:score[np.unique(s['full_history'])-1]-=np.float32(.05)
        out[j]=gaps(score)
    return out


def case(domain,name,samples,sas,pair,sas_ids,text_scores,fit_gaps,eval_gaps):
    n=len(samples);us,up=utility(sas),utility(pair);delta=up-us
    categories=dict(rescue=(sas>10)&(pair<=10),lose=(sas<=10)&(pair>10),
        both_hit_rank_better=(sas<=10)&(pair<=10)&(pair<sas),
        both_hit_rank_worse=(sas<=10)&(pair<=10)&(pair>sas),unchanged=delta==0)
    parts={k:dict(users=int(mask.sum()),N10_contribution=float(delta[mask].sum()/n)) for k,mask in categories.items()}
    assert sum(v['users'] for v in parts.values())==n
    assert abs(sum(v['N10_contribution'] for v in parts.values())-delta.mean())<1e-12
    text_ids=np.array([top50(score) for score in text_scores])
    targets=np.array([s['target'] for s in samples]);sas_hit=(sas<=50)
    assert np.array_equal((sas_ids==targets[:,None]).any(1),sas_hit)
    union_hit=sas_hit|(text_ids==targets[:,None]).any(1)
    mixed=[]
    for s,t in zip(sas_ids,text_ids):
        result=[];seen=set()
        for item in np.r_[s[:25],t[:25],s,t]:
            if int(item) not in seen:result.append(int(item));seen.add(int(item))
            if len(result)==50:break
        assert len(result)==len(seen)==50
        mixed.append(result)
    mixed=np.array(mixed);mix_hit=(mixed==targets[:,None]).any(1)
    packed=dict(users=np.array([s['user_id'] for s in samples]),sas_ranks=sas,pair_ranks=pair,
        oracle_N10_utility=np.maximum(us,up),fixed50_ids=mixed,text50_ids=text_ids,
        fixed50_target_covered=mix_hit,union_target_covered=union_hit,score_gaps=eval_gaps)
    cuts=np.quantile(fit_gaps,[.25,.5,.75],axis=0)
    buckets,routes={},{}
    for column,feature in enumerate(('top1_top2_gap','top10_top11_gap')):
        bucket=np.searchsorted(cuts[:,column],eval_gaps[:,column],side='right')
        buckets[feature]=[]
        for k in range(4):
            mask=bucket==k
            buckets[feature].append(dict(bucket=k,users=int(mask.sum()),
                sas_N10=float(us[mask].mean()) if mask.any() else None,
                pair_N10=float(up[mask].mean()) if mask.any() else None,
                delta_N10=float(delta[mask].mean()) if mask.any() else None,
                rescue_users=int((mask&categories['rescue']).sum()),lost_users=int((mask&categories['lose']).sum())))
        for quantile,cut in zip((25,50),cuts[:2,column]):
            mask=eval_gaps[:,column]<=cut
            key=f'{feature}_low{quantile}'
            rr=np.where(mask,pair,sas)
            random_order=sorted(range(n),key=lambda j:hashlib.sha256(
                f'complementarity-random-20261017|{domain}|{name}|{key}|{samples[j]["user_id"]}'.encode()).digest())
            random_mask=np.zeros(n,bool);random_mask[random_order[:int(mask.sum())]]=True
            random_rr=np.where(random_mask,pair,sas)
            assert mask.sum()==random_mask.sum()
            routes[key]=dict(threshold=float(cut),intervened_users=int(mask.sum()),intervened_fraction=float(mask.mean()),
                metrics=metrics(rr),matched_random_metrics=metrics(random_rr),
                vs_sas=contrast(rr,sas,2000,20261017),vs_matched_random=contrast(rr,random_rr,2000,20261017))
            packed[key+'_ranks']=rr;packed[key+'_mask']=mask;packed[key+'_random_mask']=random_mask
    result=dict(users=n,baseline_sas=metrics(sas),fixed_pair=metrics(pair),pair_vs_sas=contrast(pair,sas,2000,20261017),
        N10_decomposition=parts,oracle_choose_best_of_two_N10=float(np.maximum(us,up).mean()),
        oracle_relative_N10_percent=100*(np.maximum(us,up).mean()/us.mean()-1),
        oracle_warning='Uses target labels; diagnostic upper bound only, not a deployable method or achieved result.',
        rescued_targets_already_in_sas50=int((categories['rescue']&(sas<=50)).sum()),
        rescued_targets_outside_sas50=int((categories['rescue']&(sas>50)).sum()),
        sas_pool_reranking_opportunity_users=int(((sas>10)&(sas<=50)).sum()),
        candidate_coverage=dict(sas50=float(sas_hit.mean()),union_up_to100=float(union_hit.mean()),
            union_new_targets=int((union_hit&~sas_hit).sum()),mixed_fixed50=float(mix_hit.mean()),
            mixed_fixed50_gained=int((mix_hit&~sas_hit).sum()),mixed_fixed50_lost=int((~mix_hit&sas_hit).sum())),
        fit_feature_quantile_thresholds=cuts.tolist(),feature_buckets=buckets,routes=routes,
        no_claim_of_ranking_gain_from_candidate_coverage=True)
    np.savez_compressed(OUT/f'{domain}_{name}_predictions.npz',**packed)
    write(OUT/f'{domain}_{name}.json',result)
    print(domain,name,json.dumps({k:result[k] for k in ('N10_decomposition','oracle_relative_N10_percent','candidate_coverage')}),flush=True)
    return result


def main():
    OUT.mkdir(exist_ok=False);began=time.monotonic()
    write(OUT/'status.json',dict(state='RUNNING',pid=os.getpid(),test_read=False))
    source=ROOT/read(LONG/'config.json')['cached_expert_source']
    inputs=[Path(__file__).resolve(),PLAN]
    result={}
    for domain in ('Toys','Beauty'):
        folder=LONG/domain;samples=read(folder/'samples.json');pred=rows(folder/'predictions.jsonl')
        ranks=dict(np.load(folder/'ranks.npz'));raw=np.load(source/domain/'features/validation/experts.npy',mmap_mode='r')[512:]
        fit=read(source/domain/'fit_samples.json');fit_scores=np.load(source/domain/'features/train/experts.npy',mmap_mode='r')
        inputs += [folder/f for f in ('samples.json','predictions.jsonl','ranks.npz')]
        inputs += [source/domain/f for f in ('features/validation/experts.npy','features/train/experts.npy','fit_samples.json')]
        fit_raw_gaps=score_gaps(fit_scores,fit,False);eval_raw_gaps=score_gaps(raw,samples,False)
        sas_ids=np.array([order(r['SASRec__only']) for r in pred])
        result[domain]={}
        for text,col in (('BGE',3),('Qwen4',4)):
            name='raw_'+text
            result[domain][name]=case(domain,name,samples,ranks['SASRec__only'],ranks['SASRec__'+text],
                sas_ids,raw[:,:,col],fit_raw_gaps,eval_raw_gaps)
        corrected_samples=read(PAIR/domain/'samples.json');corrected_rows=rows(SINGLE/domain/'predictions.jsonl')
        sr=np.load(SINGLE/domain/'ranks.npz')['sas_penalty'];pr=np.load(PAIR/domain/'ranks.npz')['pair_loo_penalty']
        corrected_raw=np.load(PAIR/domain/'raw_experts.npy',mmap_mode='r')
        text=np.load(PAIR/domain/'loo_text.npy').copy()
        for j,s in enumerate(corrected_samples):text[j,np.unique(s['full_history'])-1]-=np.float32(.05)
        fit_penalty_gaps=score_gaps(fit_scores,fit,True)
        eval_penalty_gaps=score_gaps(corrected_raw,corrected_samples,True)
        corrected_sas_ids=np.array([order(r['sas_penalty']) for r in corrected_rows])
        for src in (PAIR,SINGLE):inputs += [src/domain/f for f in ('samples.json','predictions.jsonl','ranks.npz')]
        inputs += [PAIR/domain/f for f in ('raw_experts.npy','loo_text.npy')]
        result[domain]['corrected_BGE']=case(domain,'corrected_BGE',corrected_samples,sr,pr,
            corrected_sas_ids,text,fit_penalty_gaps,eval_penalty_gaps)
    hashes={}
    for path in dict.fromkeys(inputs):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8388608),b''):h.update(block)
        hashes[str(path.relative_to(ROOT))]=h.hexdigest()
    write(OUT/'manifest.json',dict(sha256=hashes))
    write(OUT/'summary.json',dict(domains=result,test_read=False,using_only_completed_predictions=True,
        no_gpu=True,no_training=True,predeclared_routing_policies=True,oracle_is_not_a_method=True,
        statistical_review=read(LONG/'summary.json')['statistical_review'],
        intervals='Unadjusted descriptive intervals; same reused development users; multiple exploratory policies retained.'))
    lines=['# 专家互补性与候选覆盖CPU诊断','','## Material Passport','',
        '- Origin Skill: academic-research-suite / experiment-agent','- Origin Mode: validate',
        '- Verification Status: ANALYZED; no training or GPU generation','',
        '只读已完成结果，未使用正在运行的Qwen4修正预测。raw为4096/域，corrected_BGE为1024/域，分别解释。Oracle使用目标标签，只是诊断上限，不是实际方法成绩。','',
        '| 数据集 | 协议 | SAS N10 | 双专家 N10 | 救回前10人数 | 丢失前10人数 | 两者命中但目标后移人数 |','|---|---|---:|---:|---:|---:|---:|']
    for d,cases in result.items():
        for name,a in cases.items():
            z=a['N10_decomposition'];lines.append(f'| {d} | {name} | {a["baseline_sas"]["N10"]:.6f} | {a["fixed_pair"]["N10"]:.6f} | {z["rescue"]["users"]} | {z["lose"]["users"]} | {z["both_hit_rank_worse"]["users"]} |')
    lines += ['', '## 固定50项候选覆盖','',
        '| 数据集 | 协议 | SAS前50覆盖 | 混合50覆盖 | 新覆盖人数 | 丢失覆盖人数 |','|---|---|---:|---:|---:|---:|']
    for d,cases in result.items():
        for name,a in cases.items():
            c=a['candidate_coverage'];lines.append(f'| {d} | {name} | {c["sas50"]:.6f} | {c["mixed_fixed50"]:.6f} | {c["mixed_fixed50_gained"]} | {c["mixed_fixed50_lost"]} |')
    lines += ['', '## 预定低间隔路由','',
        '门槛仅由不重叠fit用户的SAS分数分位点决定。分数间隔不是校准正确率；四种策略全部报告，不按本批标签挑门槛。随机路由保持相同介入人数。','',
        '| 数据集 | 协议 | 策略 | 路由 N10 | 对SAS绝对差 | 对匹配随机绝对差 |','|---|---|---|---:|---:|---:|']
    for d,cases in result.items():
        for name,a in cases.items():
            for key,r in a['routes'].items():lines.append(f'| {d} | {name} | {key} | {r["metrics"]["N10"]:.6f} | {r["vs_sas"]["delta_N10"]:+.6f} | {r["vs_matched_random"]["delta_N10"]:+.6f} |')
    lines += ['', '完整用户级数组、阈值、四分位桶、N10变化分解、区间和候选ID见同目录JSON/NPZ。池构造未加入目标，保存的目标覆盖只作度量。',
        '所有历史商品合法。候选覆盖提升不等于排序效果提升，最多100项的并集有预算增加，不能代替固定50的比较。最佳二选一Oracle不可部署，不用它宣称方法已超过单专家。',
        '11/11统计谬误复核沿用固定筛选审计，并保留分域/分协议解释、配对单位、固定全样本、不按命中选人、全部策略和负结果、未校正多重比较、重复开发与训练种子未覆盖限制。结果用于选择下一步假设，不自动启动新训练或GPU实验。']
    REPORT.write_text('\n'.join(lines)+'\n')
    write(OUT/'status.json',dict(state='COMPLETED',elapsed_seconds=time.monotonic()-began,no_gpu=True,
        no_training=True,test_read=False,report=str(REPORT.relative_to(ROOT)),new_gpu_experiment_started=False))


if __name__=='__main__':main()
