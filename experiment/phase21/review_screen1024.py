"""Paired, same-user synthesis of all11 experts and frozen-beta alpha arms.

No fitting, resampling of the evaluation population, or automatic next run.
Any alpha/global checkpoint comparisons remain exploratory development results.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from experiment.phase21 import qwen4_alpha_rank_1024 as X
from experiment.phase20.catalog_scope_core import metrics, contrast, utilities

ROOT=X.ROOT
EXPERT=X.SCREEN
ALPHA=X.OUT
GLOBAL=ROOT/'artifacts/phase21/qwen4_global_alpha_1024_20261017_v1'
OUT=ROOT/'artifacts/phase21/screen1024_review_20261017_v1'
REPORT=ROOT/'report/第二十一阶段/1024用户专家与干预强度综合筛选结论_20261017.md'
read,write,sha,stamp=X.read,X.write,X.sha,X.stamp
DOMAINS=X.DOMAINS
LABELS={
 'native':'GRAM',
 'SASRec__BGE':'SASRec＋BGE（联合β/α）','SASRec__Qwen4':'SASRec＋Qwen4（联合β/α）',
 'TiSASRec__BGE':'TiSASRec＋BGE','TiSASRec__Qwen4':'TiSASRec＋Qwen4',
 'ItemCF__BGE':'ItemCF＋BGE','ItemCF__Qwen4':'ItemCF＋Qwen4',
 'SASRec__only':'SASRec单专家','TiSASRec__only':'TiSASRec单专家','ItemCF__only':'ItemCF单专家',
 'BGE__only':'BGE单专家','Qwen4__only':'Qwen4单专家',
 'qwen_fixed':'SASRec＋Qwen4（CEβ，α=.3）',
 'qwen_user_nll':'SASRec＋Qwen4（冻结CEβ，用户α/NLL）',
 'qwen_user_rank':'SASRec＋Qwen4（冻结CEβ，用户α/排序）',
 'qwen_global':'SASRec＋Qwen4（冻结CEβ，统一学习α/NLL）'}
EXPERT_KEYS=list(X.R.B.ARMS)
GROUPS=[('batch1_512',slice(0,512)),('batch2_512',slice(512,1024)),('all1024',slice(0,1024))]


def changes(a,b):
    a,b=np.asarray(a),np.asarray(b)
    ua,ub=utilities(a),utilities(b)
    gained=(a<=10)&(b>10);lost=(a>10)&(b<=10);both=(a<=10)&(b<=10)
    components=dict(recovered_top10=float(ua[gained].sum()/len(a)),
        lost_top10=float(-ub[lost].sum()/len(a)),reranked_existing_hits=float((ua[both]-ub[both]).sum()/len(a)))
    assert abs(sum(components.values())-float((ua-ub).mean()))<1e-12
    return dict(users=len(a),top10_hits=int((a<=10).sum()),top10_gained=int(gained.sum()),
        top10_lost=int(lost.sum()),both_top10_rank_improved=int((both&(a<b)).sum()),
        both_top10_rank_worsened=int((both&(a>b)).sum()),
        generated_top50_recovered=int(((a<=50)&(b>50)).sum()),
        generated_top50_lost=int(((a>50)&(b<=50)).sum()),delta_N10_components=components)


def run(global_ready=False):
    OUT.mkdir(parents=True,exist_ok=True)
    write(OUT/'status.json',dict(state='ANALYZING',updated_at=stamp(),test_read=False))
    assert read(EXPERT/'status.json')['state']==read(ALPHA/'status.json')['state']=='COMPLETED'
    has_global=(GLOBAL/'summary.json').exists() and (read(GLOBAL/'status.json')['state']=='COMPLETED' or global_ready)
    keys=['native',*EXPERT_KEYS,'qwen_fixed','qwen_user_nll','qwen_user_rank']
    if has_global:keys.append('qwen_global')
    pairs=[(k,'native') for k in keys if k!='native']
    for arm,(a,b) in X.R.B.PAIRS.items():pairs.extend([(arm,a+'__only'),(arm,b+'__only')])
    pairs += [('SASRec__only','TiSASRec__only'),('SASRec__only','ItemCF__only'),
        ('SASRec__BGE','SASRec__Qwen4'),('TiSASRec__BGE','TiSASRec__Qwen4'),('ItemCF__BGE','ItemCF__Qwen4'),
        ('qwen_user_nll','qwen_fixed'),('qwen_user_rank','qwen_fixed'),
        ('qwen_user_nll','SASRec__Qwen4'),('qwen_user_nll','SASRec__only'),
        ('qwen_user_nll','SASRec__BGE'),('qwen_fixed','SASRec__only')]
    if has_global:pairs += [('qwen_user_nll','qwen_global'),('qwen_global','qwen_fixed')]
    pairs=list(dict.fromkeys(pairs));results={};manifest={}
    for domain in DOMAINS:
        epath=EXPERT/domain/'stage1024_ranks.npz';apath=ALPHA/domain/'ranks.npz'
        e=np.load(epath);a=np.load(apath)
        np.testing.assert_array_equal(e['users'][512:1536],a['users'])
        ranks={k:np.asarray(e[k][512:1536]) for k in ['native',*EXPERT_KEYS]}
        np.testing.assert_array_equal(ranks['native'],a['native'])
        np.testing.assert_array_equal(ranks['SASRec__Qwen4'],a['joint_user_reference'])
        np.testing.assert_array_equal(ranks['SASRec__only'],a['sas_only_reference'])
        for src,dst in [('fixed','qwen_fixed'),('user_nll','qwen_user_nll'),('user_rank','qwen_user_rank')]:ranks[dst]=a[src]
        files=[epath,apath,EXPERT/domain/'stage1024_snapshot.json',ALPHA/domain/'output_manifest.json',
               ALPHA/domain/'export_checks.json',ALPHA/domain/'training.json',ALPHA/domain/'alphas.npz']
        if has_global:
            gpath=GLOBAL/domain/'ranks.npz';g=np.load(gpath)
            np.testing.assert_array_equal(g['users'],a['users']);np.testing.assert_array_equal(g['native'],ranks['native'])
            ranks['qwen_global']=g['frozen_global'];files += [gpath,GLOBAL/domain/'output_manifest.json']
        assert all(len(v)==1024 for v in ranks.values())
        np.savez_compressed(OUT/(domain+'_ranks.npz'),users=a['users'],**ranks)
        results[domain]={}
        for name,ix in GROUPS:
            measured={k:metrics(v[ix]) for k,v in ranks.items()}
            baseline=measured['native']['N10']
            results[domain][name]=dict(users=len(ranks['native'][ix]),metrics=measured,
                relative_to_gram_pct={k:100*(v['N10']/baseline-1) for k,v in measured.items()},
                changes_vs_gram={k:changes(v[ix],ranks['native'][ix]) for k,v in ranks.items() if k!='native'},
                paired={aa+'_vs_'+bb:contrast(ranks[aa][ix],ranks[bb][ix],repeats=2000,seed=20261017) for aa,bb in pairs})
        coverage={}
        for behavior,content in [('SASRec','BGE'),('SASRec','Qwen4'),('SASRec','ItemCF')]:
            br,cr=ranks[behavior+'__only'],ranks[content+'__only']
            coverage[behavior+'_vs_'+content]={str(k):dict(behavior_hits=int((br<=k).sum()),content_hits=int((cr<=k).sum()),
                behavior_only_hits=int(((br<=k)&(cr>k)).sum()),content_only_hits=int(((cr<=k)&(br>k)).sum()),
                overlap_hits=int(((cr<=k)&(br<=k)).sum())) for k in (10,50)}
        results[domain]['descriptive_single_expert_coverage']=coverage
        aa=np.load(ALPHA/domain/'alphas.npz')['user_nll']
        # Post-hoc descriptive bins; never training labels or policy selection.
        order=np.argsort(aa,kind='stable');bins=[]
        for j,ix in enumerate(np.array_split(order,4)):
            bins.append(dict(bin=j+1,users=len(ix),alpha_mean=float(aa[ix].mean()),
                alpha_range=[float(aa[ix].min()),float(aa[ix].max())],
                delta_N10_vs_gram=float((utilities(ranks['qwen_user_nll'][ix])-utilities(ranks['native'][ix])).mean()),
                delta_N10_vs_fixed=float((utilities(ranks['qwen_user_nll'][ix])-utilities(ranks['qwen_fixed'][ix])).mean())))
        results[domain]['descriptive_alpha_bins']=bins
        write(OUT/(domain+'_summary.json'),results[domain])
        manifest.update({str(f.relative_to(ROOT)):sha(f) for f in files})
        print(domain,'paired comparisons',len(pairs),'per group; same1024 users verified',flush=True)
    pareto=[]
    for k in keys[1:]:
        value=np.array([results[d]['all1024']['metrics'][k]['N10'] for d in DOMAINS])
        dominated=False
        for other in keys[1:]:
            ov=np.array([results[d]['all1024']['metrics'][other]['N10'] for d in DOMAINS])
            if (ov>=value).all() and (ov>value).any():dominated=True
        if not dominated:pareto.append(k)
    review=dict(coverage='11/11',confidence='CAUTION / exploratory development',
        simpson='Domains and both512 batches retained separately; mean Pareto is descriptive only.',
        ecological='Per-user gains/losses and coverage are reported; no universal user benefit claim.',
        berkson='Fixed eligible validation sample; not raw-population or independent test evidence.',
        collider='No hit-based user inclusion or removal; coverage bins are descriptive, not policy training.',
        base_rate='Every method evaluated on all1024 same users; top10 hit counts saved.',
        regression_to_mean='Extension requested after first512; both batches remain exploratory.',
        survivorship='All11 expert arms and all negative alpha arms retained; recovery documented.',
        look_elsewhere=f'{len(pairs)*6} paired descriptive intervals, unadjusted; all saved, no confirmatory p-value claim.',
        forking_paths='Synthesis follows many prior explorations; priorities are not preregistered winner selection.',
        causation='Cross-fuser pairings compare whole pipelines; they do not isolate text necessity or unique failure causes.',
        reverse_causality='Targets used only for metrics and posthoc coverage; no model fitting or new target-derived features.')
    summary=dict(domains=results,mean_pareto_front=pareto,global_control_present=has_global,
        pair_count_per_group=len(pairs),statistical_review=review,test_read=False,
        final_method_selected=False,next_experiment_automatically_started=False)
    write(OUT/'summary.json',summary)
    manifest[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    write(OUT/'input_manifest.json',dict(sha256=manifest))
    lines=['# 1024用户：专家与干预强度综合筛选结论','','## Material Passport','',
        '- Origin Skill: academic-research-suite / experiment-agent','- Origin Mode: validate',
        '- Origin Date: 2026-10-17','- Verification Status: ANALYZED；保存排名独立复核、同用户配对分析，非独立test',
        '- Version Label: screen1024_review_v1','- Overall Confidence: CAUTION，探索性候选筛选','',
        '用户已选择以完成的新增1024人筛选，不恢复原4096扩样。全部11组均保留；原计划4096尾部未完成。',
        '专家筛选与alpha扩样的两域用户名单、GRAM排名、旧Qwen联合组和SASRec单专家排名逐项一致。',
        '原11组使用联合beta/alpha；新Qwen组固定CE beta并只训练alpha。跨这两套训练的比较是完整方案对照，不能当作单独文本专家的严格消融。','',
        '## 合计1024','',
        '| 方案 | Toys N10 | 对GRAM | Beauty N10 | 对GRAM |','|---|---:|---:|---:|---:|']
    for k in keys:
        a,b=[results[d]['all1024'] for d in DOMAINS]
        lines.append(f'| {LABELS[k]} | {a["metrics"][k]["N10"]:.6f} | {a["relative_to_gram_pct"][k]:+.2f}% | '
            f'{b["metrics"][k]["N10"]:.6f} | {b["relative_to_gram_pct"][k]:+.2f}% |')
    lines += ['','均值Pareto前沿：'+ '、'.join(LABELS[k] for k in pareto)+'。这只是点估计上的取舍，不是统计上胜出或淘汰证据。','',
        '## 前后两批的方向与命中数','',
        '| 域 | 方案 | 第一批对GRAM | 第二批对GRAM | 第一批H10人数 | 第二批H10人数 |','|---|---|---|---:|---:|']
    focus=['SASRec__only','SASRec__BGE','SASRec__Qwen4','qwen_fixed','qwen_user_nll','qwen_user_rank']
    if has_global:focus.append('qwen_global')
    for d in DOMAINS:
        for k in focus:
            a,b=results[d]['batch1_512'],results[d]['batch2_512']
            lines.append(f'| {d} | {LABELS[k]} | {a["relative_to_gram_pct"][k]:+.2f}% | {b["relative_to_gram_pct"][k]:+.2f}% | '
                f'{a["changes_vs_gram"][k]["top10_hits"]} | {b["changes_vs_gram"][k]["top10_hits"]} |')
    lines += ['','这些是固定模型在不同用户上的差异，首批预测原始字节完全保留，不能将符号翻转直接解释为训练随机性。每批只有几十名top10命中，少量高位命中变化就会影响NDCG均值。',
        'Qwen用户alpha/NLL在Toys首批为负、第二批为正，合计转正；不应继续沿用512时“两域不能同时超过GRAM”的点估计判断，也不能隐藏首批为负的事实。','',
        '## 重点配对差值','',
        '| 域 | 比较 | N10绝对差 | 描述性95%区间 | 新增/丢失H10 |','|---|---|---:|---|---|']
    selected=[('qwen_user_nll','native'),('qwen_user_nll','qwen_fixed'),('qwen_user_nll','SASRec__Qwen4'),
        ('qwen_user_nll','SASRec__only'),('SASRec__BGE','SASRec__only'),('ItemCF__BGE','ItemCF__only'),
        ('ItemCF__Qwen4','ItemCF__only')]
    if has_global:selected += [('qwen_user_nll','qwen_global'),('qwen_global','qwen_fixed')]
    for d in DOMAINS:
        for aa,bb in selected:
            # Some reverse pairs are already present under their direct names.
            k=aa+'_vs_'+bb
            if k in results[d]['all1024']['paired']:v=results[d]['all1024']['paired'][k]
            else:
                rr=np.load(OUT/(d+'_ranks.npz'));v=contrast(rr[aa],rr[bb],repeats=2000,seed=20261017)
            ci=v['descriptive_95pct_N10']
            lines.append(f'| {d} | {LABELS[aa]} vs {LABELS[bb]} | {v["delta_N10"]:+.6f} | [{ci[0]:+.6f}, {ci[1]:+.6f}] | {v["top10_gained"]}/{v["top10_lost"]} |')
    lines += ['','## 专家意见能补回什么','',
        '| 域 | 方案 | GRAM前50未出现而引导出现 | 引导丢失GRAM前50命中 | 新增/丢失H10 |','|---|---|---:|---:|---|']
    for d in DOMAINS:
        for k in ['SASRec__only','SASRec__BGE','qwen_user_nll']:
            c=results[d]['all1024']['changes_vs_gram'][k]
            lines.append(f'| {d} | {LABELS[k]} | {c["generated_top50_recovered"]} | {c["generated_top50_lost"]} | {c["top10_gained"]}/{c["top10_lost"]} |')
    lines += ['','这是实际生成覆盖与排序的改变，不能据此认定全部属于某一层beam剪枝错误。进一步的N10增量分解和单专家独有命中数保存在summary.json。',
        '单专家独有命中只说明输出覆盖有差异，不能证明融合器在推理时能够识别这些用户，也不能替代融合后的增量消融。','',
        '## 本轮筛选落点','',
        '1. 保留SASRec单专家为必要强对照。它在两域、两批均为正均值；所有双专家仍需证明额外成本和内容贡献。',
        '2. SASRec＋BGE保留为性能参照；SASRec＋Qwen4＋冻结CE beta的用户alpha/NLL保留为主要自适应候选。二者各有域内优势，尚不能宣称统一胜出。',
        '3. TiSASRec相关组保持备选记录；当前没有足够证据支持替换SASRec。ItemCF两种替代行为搭配均未稳定优于ItemCF单专家，不优先投入，但不删除结果或宣称ItemCF普遍无用。',
        '4. 本版排序损失alpha不作为下一轮主推方向：Toys两批均差，Beauty选epoch0而实际退回固定alpha。',
        '5. 新Qwen用户alpha/NLL点估计两域均高于旧Qwen联合组，但Toys仍低于固定alpha、Beauty高于固定alpha；这不是“动态alpha已被证明更优”。']
    if has_global:
        lines += ['', '已补同一冻结CE beta、相同NLL、同fit数据与预算下的统一学习alpha。以表中配对差判断校准与用户差异，不按域选不同方法作为统一方案。']
    else:lines += ['', '学习统一alpha的1024同用户对照尚未完成，因此当前用户级alpha的额外价值仍待该对照。']
    lines += ['','## 尚缺的最小证据','',
        '- 确定Qwen替换是否更合适，需在同一冻结beta/用户alpha训练方式下比较BGE与Qwen；当前BGE联合组不能隔离文本编码器效应。',
        '- 最终保留内容专家前，需与强SASRec单专家作匹配或更保守的贡献确认；目前尚未在两域证明其稳定额外收益。',
        '- 本轮ItemCF是替代行为专家或单独专家；没有在当前融合规则下完成SASRec＋ItemCF＋文本三专家匹配实验，不能据此证明第三专家冗余。',
        '- 这些缺口不等同于现在就启动全部论文实验。最终独立test与多种子验证仍在方法定型以后，当前不自动扩样或训练。','',
        '## 统计检查（11/11）','']
    for k,v in review.items():lines.append(f'- {k}: {v}')
    lines += ['', '单次bootstrap只描述固定模型的用户抽样不确定性，不包含训练种子或研究选择。区间跨0不表示等效；未调整多重比较的区间不作确认性显著性结论。']
    REPORT.write_text('\n'.join(lines)+'\n')
    write(OUT/'status.json',dict(state='COMPLETED' if has_global else 'CURRENT_RESULTS_ANALYZED_GLOBAL_PENDING',
        updated_at=stamp(),global_control_present=has_global,test_read=False,
        report=str(REPORT.relative_to(ROOT)),scope='all11 plus fixed frozen-beta alpha arms on same1024'))
    handoff=ROOT/'artifacts/phase21/session_handoff_20261017.json'
    if handoff.exists():
        h=read(handoff);h['combined_screen_review']=dict(status=str((OUT/'status.json').relative_to(ROOT)),
            report=str(REPORT.relative_to(ROOT)),global_control_present=has_global,
            final_method_selected=False,all11_retained=True,restore4096=False)
        write(handoff,h)
    print(str(REPORT),flush=True)


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args();run()
