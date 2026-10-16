"""Independent CPU audit of joint-fusion predictions, heads and resume history.

No expert/GRAM/head fitting, generation, test evaluation, or parameter selection.
"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from experiment.phase21 import review_expert_replacements as V
from experiment.phase21 import joint_fusion as J
from experiment.phase20.catalog_scope_core import contrast

ROOT, RUN = J.ROOT, J.OUT
OUT = ROOT/'artifacts/phase21/joint_fusion_review_20261016_v1'
REPORT = ROOT/'report/第二十一阶段/联合融合与专家替换结果复核_20261016.md'
read, write, digest = V.read, V.write, V.digest


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    write(OUT/'status.json',dict(state='RUNNING',stage='independent_saved_outputs_audit',test_read=False))
    checked={}
    def verify_manifest(path):
        for relative,expected in read(path)['sha256'].items():
            if relative not in checked:checked[relative]=digest(ROOT/relative)
            assert checked[relative]==expected,relative
    status=read(RUN/'status.json')
    assert status['state']=='COMPLETED'
    assert all(v['exit_code']==0 and v['lease_released'] for v in status['finished'].values())
    verify_manifest(RUN/'manifest.json');verify_manifest(RUN/'resource_resume/manifest.json')
    snapshot=read(RUN/'resource_resume/snapshot.json')
    domains={}
    for d in J.DOMAINS:
        p=RUN/d
        for name in ('output_manifest.json','training_manifest.json'):verify_manifest(p/name)
        assert read(p/'lease_status.json')['state']=='released'
        assert read(p/'runtime_checks.json')['gram_model_digest_unchanged']
        raw=(p/'predictions.jsonl').read_bytes()
        ss=snapshot[d]
        assert hashlib.sha256(raw[:ss['preserved_prefix_bytes']]).hexdigest()==ss['preserved_prefix_sha256']
        assert read(p/'smoke.json')['last_saved_row_all_20_arms_replayed']
        samples=read(p/'samples.json');fit=read(p/'fit_samples.json')
        assert samples==read(J.BASE/d/'samples.json')
        assert {s['user_id'] for s in samples}.isdisjoint(s['user_id'] for s in fit)
        rows=[json.loads(x) for x in raw.splitlines()]
        assert len(rows)==len(samples)==len({s['user_id'] for s in samples})==512
        summary=read(p/'summary.json');archive=np.load(p/'ranks.npz')
        ranks={k:[] for k in summary['metrics']}
        for row,sample in zip(rows,samples):
            assert all(row[k]==sample[k] for k in ('user_id','target','history'))
            for key in ranks:
                v=row[key]
                assert len(v['ids'])==len(v['scores'])==len(set(v['ids']))==50
                assert np.isfinite(v['scores']).all()
                ranks[key].append(V.sorted_rank(v['ids'],v['scores'],sample,2**30))
        ranks={k:np.asarray(v) for k,v in ranks.items()}
        for key,rr in ranks.items():
            np.testing.assert_array_equal(rr,archive[key])
            for metric,value in V.metrics(rr).items():assert abs(value-summary['metrics'][key][metric])<1e-12
        original=np.load(J.BASE/d/'ranks.npz')
        np.testing.assert_array_equal(ranks['native'],original['native'])
        for suffix in ('guided','late'):
            np.testing.assert_array_equal(ranks['bge_fixed_'+suffix],original['user_'+suffix])
            for b in ('qwen06','qwen4','tisas'):
                old=np.load((J.TIME if b=='tisas' else J.QWEN)/d/'ranks.npz')
                np.testing.assert_array_equal(ranks[b+'_fixed_'+suffix],old[b+'_'+suffix])
        for key,value in summary['comparisons'].items():
            a,b=key.split('_vs_')
            delta=V.metrics(ranks[a])['N10']-V.metrics(ranks[b])['N10']
            assert abs(delta-value['delta_N10'])<1e-12
            assert value['top10_gained']==int(((ranks[a]<=10)&(ranks[b]>10)).sum())
            assert value['top10_lost']==int(((ranks[a]>10)&(ranks[b]<=10)).sum())

        signals,alphas,betas=[np.load(p/name) for name in ('signals.npz','alphas.npz','betas.npz')]
        head_checks={}
        for backbone in J.BACKBONES:
            folder,parent_path=J.origin(d,backbone)
            u=np.load(folder/'validation/user.npy')
            item=np.load(folder/'validation/item.npy',mmap_mode='r')
            parent=torch.load(parent_path,map_location='cpu')
            assert parent['fit_users']==[s['user_id'] for s in fit]
            fold_ck=torch.load(p/'parents'/backbone/'fold.pt',map_location='cpu')
            expected=[s['user_id'] for s in fit if J.fold(s['user_id'],d)!=0]
            assert fold_ck['fit_users']==expected
            for mode in J.MODES:
                arm=backbone+'_'+mode
                if mode=='fixed':
                    w=V.head_weights(parent,u);a=np.full(512,.3)
                else:
                    ck=torch.load(p/(arm+'.pt'),map_location='cpu')
                    assert ck['fit_users']==parent['fit_users']
                    sd={k:v.numpy() for k,v in ck['state_dict'].items()}
                    x=np.clip((u-ck['mean'])/ck['std'],-10,10)
                    h=np.maximum(x@sd['input.weight'].T+sd['input.bias'],0)
                    z=h@sd['beta.weight'].T+sd['beta.bias']
                    w=np.exp(z-z.max(1,keepdims=True));w/=w.sum(1,keepdims=True)
                    z=(h@sd['strength.weight'].T+sd['strength.bias']).ravel() if mode=='user' else np.full(512,sd['strength'])
                    a=np.clip(1/(1+np.exp(-z)),1e-6,1.-1e-6)
                    if mode=='calibrate':
                        for k,old in (('input.weight','input.weight'),('input.bias','input.bias'),('beta.weight','output.weight'),('beta.bias','output.bias')):
                            wanted=parent['state_dict'][old].numpy()
                            if k=='input.weight':wanted=wanted[:,:10]
                            np.testing.assert_array_equal(sd[k],wanted)
                predicted=(w[:,None,:]*item[:,:,:2]).sum(2)
                np.testing.assert_allclose(predicted,signals[arm],atol=3e-6,rtol=2e-5)
                np.testing.assert_allclose(w,betas[arm],atol=2e-6,rtol=2e-5)
                np.testing.assert_allclose(a,alphas[arm],atol=2e-6,rtol=2e-5)
                head_checks[arm]=dict(all512_scores_beta_alpha_recomputed=True,alpha_mean=float(a.mean()),
                    alpha_quantiles=np.quantile(a,[0,.1,.5,.9,1]).tolist(),behavior_weight_mean=float(w[:,0].mean()))
        comparison=dict(summary['comparisons'])
        for b in J.BACKBONES:
            for first,second in ((b+'_user_guided',b+'_fixed_guided'),(b+'_user_guided',b+'_user_late'),
                                 (b+'_user_guided','bge_fixed_guided'),(b+'_user_guided','bge_user_guided')):
                if first!=second:
                    comparison[first+'_vs_'+second]=contrast(ranks[first],ranks[second],repeats=2000,seed=20261016)
        derived={}
        gram=summary['metrics']['native']['N10'];old=summary['metrics']['bge_fixed_guided']['N10']
        for b in J.BACKBONES:
            for mode in J.MODES:
                arm=b+'_'+mode;value=summary['metrics'][arm+'_guided']['N10']
                derived[arm]=dict(N10=value,relative_to_gram_pct=100*(value/gram-1),relative_to_old_dual_pct=100*(value/old-1))
        domains[d]=dict(metrics=summary['metrics'],derived=derived,comparisons=comparison,heads=head_checks,
            resumed_prefix_verified_users=ss['completed_users'],saved_rank_arrays_verified=len(ranks),
            paired_comparisons=len(comparison),training=summary['training'])
        write(OUT/(d+'.json'),domains[d])
        print(d,'VERIFIED: 41 rank arrays, 20 full-catalog heads, historical controls and resume prefix',flush=True)
    audit=dict(state='VERIFIED_SAVED_OUTPUTS',scope='independent CPU recomputation; no fresh stochastic run',
        domains=domains,hashes_checked=len(checked),test_read=False,new_training=False,
        statistical_review=dict(coverage='11/11',
            simpson='Keep Toys/Beauty separate; no pooled average used to declare a winner.',
            ecological='Per-domain means do not imply all users benefit.',
            berkson='Historical development512 samples reused; no population extrapolation.',
            collider='No subgroup conditioning or causal adjustment introduced.',
            base_rate='All512 users retained; gained/lost top10 counts preserved.',
            regression_to_mean='Adaptation is matched, but following disappointing replacements can invite winner selection; remains exploratory.',
            survivorship='Every prespecified arm and all512 users completed; saved prefixes retained.',
            look_elsewhere='Many comparisons, uncorrected descriptive intervals; no confirmatory significance claim.',
            forking_paths='Scientific configuration frozen; resource changes audited; same dev pool used repeatedly.',
            causation='This controlled pipeline comparison does not establish time-information utility or alpha as sole cause.',
            reverse_causality='Inputs remain history-only; target appears only in training objective/evaluation.'))
    write(OUT/'summary.json',audit);write(OUT/'verified_hashes.json',checked)
    write_report(domains)
    write(OUT/'status.json',dict(state='COMPLETED',verification='VERIFIED_SAVED_OUTPUTS',test_read=False,
        new_training=False,updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),report=str(REPORT.relative_to(ROOT))))


def write_report(domains):
    labels=dict(bge='SASRec+BGE',qwen06='SASRec+Qwen0.6B',qwen4='SASRec+Qwen4B',tisas='TiSASRec+BGE',tisas_qwen06='TiSASRec+Qwen0.6B')
    lines=['# 联合融合与专家替换结果复核', '', '## Material Passport', '',
        '- Origin Skill: academic-research-suite / experiment-agent', '- Origin Mode: validate',
        '- Origin Date: 2026-10-16', '- Verification Status: VERIFIED_SAVED_OUTPUTS；统计解释ANALYZED，未重跑随机训练',
        '- Version Label: joint_fusion_review_v1', '',
        '两域各512名历史开发用户、seed0、五种专家组合×四种强度方式全部完成。所有训练及输出清单哈希核对通过，显存租约已释放。',
        '独立重算82份排名数组及NDCG/H，40个小头的全512用户融合分数、beta和alpha；固定强度历史基线全部复现。'
        '16 GiB资源切换保留Toys82/Beauty1名用户的预测字节前缀，并完成断点全20组回放。没有新的训练、生成或test访问。', '',
        '当前最值得保留为整体性能候选的是TiSASRec+BGE+联合用户融合；若优先更换文本专家，Qwen0.6B和4B都有保留价值，存在两域取舍。'
        '用户级强度并非所有组合都优于全局强度，不能把它写成已经普遍成立的核心优势。TiSASRec与Qwen同时替换没有叠加获益依据。', '',
        '| 数据域 | 专家组合 | 固定0.3 | 仅全局强度校准 | 联合全局强度 | 联合用户强度 |',
        '|---|---|---:|---:|---:|---:|']
    for d in J.DOMAINS:
        for b in J.BACKBONES:
            lines.append('| '+d+' | '+labels[b]+' | '+' | '.join(f'{domains[d]["derived"][b+"_"+m]["N10"]:.6f}' for m in J.MODES)+' |')
    lines+=['','固定0.3和仅全局强度校准均保留原专家CE训练的用户beta；两个联合版本按同一GRAM路径NLL适配beta/alpha。'
        '“全局强度”仅指alpha全局共享，beta仍因用户变化。NLL是训练代理，不等于NDCG。', '',
        '| 联合用户组合 | Toys相对GRAM | Beauty相对GRAM | Toys相对旧双专家 | Beauty相对旧双专家 |',
        '|---|---:|---:|---:|---:|']
    for b in J.BACKBONES:
        t=domains['Toys']['derived'][b+'_user'];v=domains['Beauty']['derived'][b+'_user']
        lines.append(f'| {labels[b]} | {t["relative_to_gram_pct"]:+.2f}% | {v["relative_to_gram_pct"]:+.2f}% | {t["relative_to_old_dual_pct"]:+.2f}% | {v["relative_to_old_dual_pct"]:+.2f}% |')
    lines+=['',
        '按用户“超过原始GRAM即可接受”的标准，以上五个联合用户组合的两域点估计均满足；这不是统计确认。'
        'TiSASRec+BGE的Beauty从固定强度0.075260回升到0.081100，对旧双专家的下降由8.83%缩小到1.76%，Toys仍上升8.67%。',
        'SASRec+Qwen0.6B的Beauty从0.077882回升到0.081896，距旧双专家仅−0.79%，但Toys从0.066967降至0.065038。'
        'Qwen4B联合用户方案相对GRAM为+9.96%/+11.03%，比0.6B更高的Toys伴随较低的Beauty。不能宣称4B全面更好。',
        'TiSASRec+Qwen0.6B联合用户方案在两个域都低于TiSASRec+BGE联合用户方案，Beauty相对旧双专家仍−9.42%；不建议作为当前优先组合。',
        '公平强化后的原SASRec+BGE也有收益：联合全局强度为0.066991/0.085298，Beauty仍高于上述替换方案。不要只与未适配的旧基线比较。', '',
        '| 组合 | 用户强度相对全局强度：Toys | Beauty |', '|---|---:|---:|']
    for b in J.BACKBONES:
        vals=[]
        for d in J.DOMAINS:
            m=domains[d]['metrics'];vals.append(100*(m[b+'_user_guided']['N10']/m[b+'_global_guided']['N10']-1))
        lines.append(f'| {labels[b]} | {vals[0]:+.2f}% | {vals[1]:+.2f}% |')
    lines+=['',
        'TiSASRec+BGE用户强度相对全局强度为+1.62%/+2.64%；Beauty的未校正描述性N10差值区间[0.000513,0.004047]，Toys区间跨0。'
        'Beauty净增加2个Top10命中用户。单seed、重复开发池和多重比较不支持正式显著性或稳定泛化结论。',
        'Qwen4B用户强度虽然两域都高于联合全局强度，但与更简单的“固定beta只校准全局alpha”比较，Toys更低、Beauty更高；复杂度必要性尚未成立。', '',
        '| 联合用户组合 | Toys生成N10 | Toys后重排N10 | Beauty生成N10 | Beauty后重排N10 |',
        '|---|---:|---:|---:|---:|']
    for b in J.BACKBONES:
        vals=[domains[d]['metrics'][b+'_user_'+s]['N10'] for d in J.DOMAINS for s in ('guided','late')]
        lines.append('| '+labels[b]+' | '+' | '.join(f'{v:.6f}' for v in vals)+' |')
    lines+=['',
        'TiSASRec+BGE在两个域的生成点估计都高于同规则后重排；Qwen两种大小在Toys仍是后重排更高，Beauty为生成更高。'
        '不能将此前全量三专家的生成优势不加验证地迁移到新双专家与新融合器。',
        '小头的参数、权重和alpha独立前向复核正常，能学习不等于NDCG必然提高；当前不据此归因为网络太小，也不自动扩宽网络。',
        '若沿TiSASRec+BGE方向定型，仍不能把本轮提升解释为时间间隔本身的贡献：需要同融合方式的真实时间/等间隔对照才能分离。'
        '若把两个专家都作为必要组件，仍需在最终融合方式下分别去掉行为/文本的匹配消融。本轮没有补这些实验，也没有启动整套论文实验。',
        '全部比较与描述性区间（包括对旧双专家、固定强度、全局强度和后重排）见复核summary.json。统计解释检查11/11完成。']
    REPORT.write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
