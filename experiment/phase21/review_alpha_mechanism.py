"""Independent saved-prediction audit; does not run or modify experiments."""
import json
from pathlib import Path
import numpy as np
from experiment.phase21 import alpha_mechanism as M


def independent_rank(prediction, target):
    ids=np.asarray(prediction['ids']);scores=np.asarray(prediction['scores'])
    assert ids.shape==scores.shape==(50,) and len(set(ids))==50 and np.isfinite(scores).all()
    match=np.flatnonzero(ids==target)
    if not len(match):return 2**30
    score=scores[match[0]]
    return int(1+(scores>score).sum()+((scores==score)&(ids<target)).sum())


def utility(ranks,k=10):
    ranks=np.asarray(ranks)
    return np.where(ranks<=k,1/np.log2(ranks+1),0.)


def decomposition(a,b):
    a,b=np.asarray(a),np.asarray(b);ua,ub=utility(a),utility(b)
    added=(a<=10)&(b>10);lost=(a>10)&(b<=10);shared=(a<=10)&(b<=10)
    return dict(delta_N10=float((ua-ub).mean()),added_top10=int(added.sum()),lost_top10=int(lost.sum()),
        rescued_from_outside_reference50_to_top10=int(((b>50)&(a<=10)).sum()),
        reference_top50_lost=int(((b<=50)&(a>50)).sum()),
        new_hit_utility=float(ua[added].sum()/len(a)),lost_hit_utility=-float(ub[lost].sum()/len(a)),
        shared_hit_rank_utility=float((ua[shared]-ub[shared]).sum()/len(a)),
        shared_hits_improved=int((shared&(a<b)).sum()),shared_hits_worsened=int((shared&(a>b)).sum()))


def main():
    root=M.read(M.OUT/'status.json');assert root['state']=='COMPLETED'
    assert all(v['exit_code']==0 and v['lease_released'] for v in root['finished'].values())
    result={};metrics_checked=0;inputs={}
    for d in M.DOMAINS:
        p=M.OUT/d;summary=M.read(p/'summary.json');samples=M.read(p/'samples.json')
        for rel,digest in M.read(p/'output_manifest.json')['sha256'].items():
            assert M.sha(M.ROOT/rel)==digest,rel;inputs[rel]=digest
        assert M.read(p/'runtime_checks.json')['state']==M.read(p/'smoke.json')['state']=='PASSED'
        fit=M.read(p/'fit_samples.json');targets=np.asarray([s['target']-1 for s in fit])
        np.testing.assert_array_equal(targets,np.load(p/'teacher.npz')['targets'])
        np.testing.assert_array_equal(targets,np.load(p/'pool.npz')['ids'][:,0]-1)
        ranks={k:[] for k in ('native',*M.ARMS)};stats={k:[] for k in M.ARMS};users=[]
        for j,line in enumerate((p/'predictions.jsonl').read_text().splitlines()):
            row=json.loads(line);sample=samples[j]
            assert all(row[k]==sample[k] for k in ('user_id','target','history'))
            users.append(row['user_id'])
            for k in ranks:ranks[k].append(independent_rank(row[k],sample['target']))
            for k in stats:stats[k].append(row['guidance_statistics'][k]['alpha_mean'])
        assert len(users)==len(set(users))==len(samples)==512
        assert set(users).isdisjoint(s['user_id'] for s in fit)
        saved=np.load(p/'ranks.npz');metrics={};alpha={}
        for k,rr in ranks.items():
            rr=np.asarray(rr);ranks[k]=rr;np.testing.assert_array_equal(rr,saved[k])
            measured={}
            for cut in (5,10,20,50):
                measured['N'+str(cut)]=float(utility(rr,cut).mean())
                measured['H'+str(cut)]=float((rr<=cut).mean())
            for name,value in measured.items():
                assert abs(value-summary['metrics'][k][name])<1e-14,(d,k,name)
                metrics_checked+=1
            metrics[k]=measured
            if k in stats:
                alpha[k]=dict(mean_across_user_branch_means=float(np.mean(stats[k])),
                    user_mean_quantiles=np.quantile(stats[k],[0,.1,.5,.9,1]).tolist())
        pairs=[(a,b) for a in M.ARMS for b in ('native','fixed03') if a!=b]
        pairs += [('mass_raw','mass_mean'),('evidence_kl01','user_kl01')]
        decomposed={a+'_vs_'+b:decomposition(ranks[a],ranks[b]) for a,b in pairs}
        for key,c in summary['comparisons'].items():
            a,b=key.split('_vs_');diff=utility(ranks[a])-utility(ranks[b])
            assert abs(float(diff.mean())-c['delta_N10'])<1e-14
            assert int(((ranks[a]<=10)&(ranks[b]>10)).sum())==c['top10_gained']
            assert int(((ranks[a]>10)&(ranks[b]<=10)).sum())==c['top10_lost']
        result[d]=dict(metrics=metrics,alpha=alpha,decompositions=decomposed,
            comparisons=summary['comparisons'],training=summary['training'],
            mass_calibration=summary['mass_calibration'],users=512)
    review=dict(coverage='11/11',simpson='Separate domains; no pooled winner.',
        ecological='Mean changes do not imply each user benefits; gains/losses retained.',
        berkson='Processed development cohort; not a population claim.',
        collider='No outcome-based user inclusion or subgroup conditioning.',
        base_rate='All512 users and baseline hit counts included per domain.',
        regression_to_mean='Single seed, reused development set; expand independently of desirable sign.',
        survivorship='All ten arms and all prescribed users completed.',
        look_elsewhere='All preregistered contrasts retained; descriptive intervals unadjusted; no formal p claims.',
        forking_paths='Frozen run plus logged pre-fit dtype amendment; standalone preflight index correction disclosed.',
        causation='Claims limited to these model variants; matched mean calibration shifted on generated prefixes.',
        reverse_causality='Targets only labels/metrics; no target inference features; disjoint fit/development users.')
    output=dict(state='VERIFIED_SAVED_OUTPUTS',updated_at=M.stamp(),independent_metrics_checked=metrics_checked,
        domains=result,statistical_review=review,output_hashes_verified=True,test_read=False,
        new_gpu_jobs_started=False,method_selected=False,training_reproducibility_rerun=False)
    M.write(M.OUT/'review.json',output)
    report=M.ROOT/'report/第二十一阶段/SASRec_BGE干预强度的锚定位置与历史证据_结果复核_20261017.md'
    lines=['# SASRec+BGE α探索：两域512完整结果复核','','## Material Passport','',
        '- Origin Skill: academic-research-suite / experiment-agent，内联执行','- Origin Mode: validate',
        '- Origin Date: 2026-10-17','- Verification Status: VERIFIED_SAVED_OUTPUTS；独立重算保存输出，未重训复现',
        '- Version Label: alpha_mechanism_20261017_v1_review','',
        '两域各512、十个α方案全部完成，显存租约全部释放。固定等权SASRec+BGE，全程保留历史、不加惩罚，不读test。',
        f'独立用分数比较/ID打破平局计算目标排名，全部{metrics_checked}项NDCG/Hit与原汇总一致，输出hash均通过；正确目标索引与teacher、pool逐条一致。','',
        '## 完整结果','','下表为NDCG@10相对同批原生GRAM的变化；Toys原生.069207，Beauty原生.085215。','',
        '| α方案 | Toys | Beauty |','|---|---:|---:|']
    labels={'fixed03':'固定.3','global_nll':'全局α/NLL','user_nll':'用户α/NLL',
        'user_kl001':'用户α/KL=.001','user_kl01':'用户α/KL=.01','user_kl1':'用户α/KL=.1',
        'mass_raw':'.3(1-m)','mass_mean':'小常数α均值对照','mass_matched':'按m调节/训练均值校准',
        'evidence_kl01':'前缀历史证据+KL=.01'}
    for arm in M.ARMS:
        values=[100*(result[d]['metrics'][arm]['N10']/result[d]['metrics']['native']['N10']-1) for d in M.DOMAINS]
        lines.append(f'| {labels[arm]} | {values[0]:+.2f}% | {values[1]:+.2f}% |')
    lines += ['', '## KL：修正NLL退化，未证明动态α胜过固定强度','',
        'NLL学习的全局α、用户α在两域均低于固定.3。增加KL权重使平均α下降，λ=.1两域均高于GRAM，但相对固定.3方向不同，配对区间均跨0。',
        '', '| 数据集 | 用户NLL平均α | 用户KL=.1平均α | KL=.1对固定.3相对N10 | 绝对N10差95%描述区间 |','|---|---:|---:|---:|---|']
    for d,v in result.items():
        rel=100*(v['metrics']['user_kl1']['N10']/v['metrics']['fixed03']['N10']-1)
        ci=v['comparisons']['user_kl1_vs_fixed03']['descriptive_95pct_N10']
        lines.append(f"| {d} | {v['alpha']['user_nll']['mean_across_user_branch_means']:.4f} | {v['alpha']['user_kl1']['mean_across_user_branch_means']:.4f} | {rel:+.2f}% | [{ci[0]:+.6f}, {ci[1]:+.6f}] |")
    lines += ['', '相比固定.3，KL=.1的Toys前10命中66→63，Beauty75→73；Beauty的NDCG上涨与更少Hit@10并存。具体分解：','',
        '| 域 | 新增前10 | 丢失前10 | 新增命中N10贡献 | 丢失命中N10贡献 | 共同命中排序N10贡献 |','|---|---:|---:|---:|---:|---:|']
    for d,v in result.items():
        z=v['decompositions']['user_kl1_vs_fixed03']
        lines.append(f"| {d} | {z['added_top10']} | {z['lost_top10']} | {z['new_hit_utility']:+.6f} | {z['lost_hit_utility']:+.6f} | {z['shared_hit_rank_utility']:+.6f} |")
    lines += ['', '因此目前只能说KL损失修正了这批NLL方案的退化，尚不能说用户级特征比合适的统一α更有用。若继续，需同一λ的全局α对照；单纯击败默认.3也不等于击败公平校准的固定强度。','',
        '## m：当前实现没有双域增量支持','',
        '原始.3(1-m)接近弱干预；Toys与其小常数对照几乎相同，Beauty略高于该对照（新增3个前10、未丢失），但仍未超过固定.3。训练均值校准版本两域低于GRAM。',
        '校准只对训练池前缀均值=.3成立，实际生成前缀的均值发生漂移：','']
    for d,v in result.items():
        lines.append(f"- {d}：mass_raw实际均值{v['alpha']['mass_raw']['mean_across_user_branch_means']:.4f}；mass_matched实际均值{v['alpha']['mass_matched']['mean_across_user_branch_means']:.4f}。")
    lines += ['', '因此校准版本的下降不能干净地归因于“位置相关无效”：平均强度和访问路径也改变了。结论限于当前门控实现尚未奏效，不推广为m完全无信号。','',
        '## 历史贡献：没有两域一致增量','',
        '证据臂相对同λ=.01的用户α，Toys略好、Beauty略差；两域配对区间均跨0。它不是稳定的新增收益来源。',
        '条件于其余输入，top1集中度和证据最近性的线性权重在两域均为负，与“越集中越近期就越强干预”的原设想符号相反；这只是当前代理损失下的拟合，不构成对真实机制的因果解释。','']
    for d,v in result.items():
        lines.append(f"- {d}：top1/top3/最近性权重{v['training']['fits']['evidence_kl01']['evidence_coefficients']}。")
    lines += ['', '## 后续判断与边界','',
        '当前最值得继续验证的是用户KL=.1，同时保留原生GRAM、固定.3，并补同λ的全局α强对照。建议不把全部十组直接扩大；本报告未启动任何新实验。',
        '两域各512是重复使用的开发用户，单训练种子；所有区间为2000次配对bootstrap的未校正描述区间，不包括训练种子和方案选择不确定性。跨0不代表等效。十组全部保留，无按域拼接不同方法。',
        '统计检查11/11的逐项说明保存在review.json；未将小样本赢家当最终方法。',
        '此前独立训练侧m/NLL预检的0/1基索引错误已更正并归档，正式训练/生成不读取这些预检输出；此次再次核对了teacher目标与pool ID的一致性。']
    report.write_text('\n'.join(lines)+'\n')
    M.write(M.OUT/'review_manifest.json',dict(sha256=inputs,review_source_sha256=M.sha(Path(__file__)),
        review_json_sha256=M.sha(M.OUT/'review.json'),review_report_sha256=M.sha(report)))
    print(json.dumps(dict(state=output['state'],metrics_checked=metrics_checked,report=str(report)),ensure_ascii=False))
    for d,v in result.items():
        print(d,'alpha_means',{k:round(x['mean_across_user_branch_means'],5) for k,x in v['alpha'].items()})
        print(d,'KL_vs_fixed_decomposition',v['decompositions']['user_kl1_vs_fixed03'])


if __name__=='__main__':main()
