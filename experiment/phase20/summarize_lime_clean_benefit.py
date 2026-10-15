"""Render verified fixed-protocol results; no model or threshold selection."""
import json
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/phase20/lime_clean_benefit_20260922_v1'
REPORT=ROOT/'report/第二十阶段/LIME干净监督_增量收益可学性检查_20260922.md'


def read(p):return json.loads(Path(p).read_text())


def main():
    s=read(OUT/'summary.json');audit=read(OUT/'independent_verification.json')
    assert audit['state']=='VERIFIED' and read(OUT/'status.json')['state']=='COMPLETED'
    domains=s['domains'];passed=s['decision'];methods=[k for k in ('constant_gate','benefit_ridge','linear_calibration','mlp_all_three_seeds') if passed[k]]
    names={'GRAM':'原 GRAM','fixed_late':'固定引导，同原候选重排','fixed_decode':'固定引导，实际生成',
        'constant_gate':'训练均值决定全部开/关','benefit_ridge':'预测实际收益的请求门控',
        'linear_calibration':'普通线性校准','mlp_seed0':'普通 MLP seed0','mlp_seed1':'普通 MLP seed1','mlp_seed2':'普通 MLP seed2'}
    text=['# LIME 干净监督：增量收益可学性检查','',
        '2026-09-22。Origin Skill: academic-research-suite，内联执行。','',
        '## Material Passport','',
        '- 状态：双域缓存、固定小头拟合、用户外预测和独立脚本复算完成。',
        '- 类型：探索性开发检查；不读取 test，不声称独立测试或新论文机制。',
        '- 问题：原候选上的专家加分收益是否可预测，普通校准是否可改善排序。','',
        '## 冻结规则下的结论','',
        ('通过预定双域投入条件：'+', '.join(methods)+'。仍须另行检验生成过程必要性。') if methods else
        '**没有方法通过预先规定的双域投入条件。当前结果不支持直接训练更复杂的生成控制器。** 这只限制本轮信号、样本和固定小头；不证明所有可学习机制都无效。','',
        '“通过”要求两域开发 N10 都比原 GRAM 和 fixed_late 各高至少1%，H10均不降；两域训练留出OOF N10高于两参照，且至少4/5折同向。MLP还要求三个seed全部满足开发条件。所有固定配置均报告，没有按开发结果挑头、轮数或阈值。','',
        '**未过双域门槛不等于没有可学信号。** 普通MLP在两域的训练末次交互五折外预测中，三个seed均提高N10，且均至少4/5折胜过两参照。相对fixed_late，Toys为+4.46%～+4.63%，Beauty为+3.76%～+5.59%。转到另外512名开发用户后，Beauty仍为+5.00%～+8.58%，Toys则为−1.39%～+0.16%。这是有价值但尚未跨场景稳定的校准信号，不能把两域分别挑选不同头拼成统一成功。','',
        '整请求收益门控的开发变化仅为Toys +0.37%、Beauty −0.42%，两域前十命中人数均未改变；它对98.05%/91.99%的开发用户仍选择固定引导。训练留出OOF中帮助vs损害AUC为0.513/0.503。这支持降低这版请求级开关的优先级；普通预测误差略小于常数并不足以构成推荐收益。','',
        '## 后续取舍','',
        '本轮按冻结条件收尾，不将单域成绩或最好seed当成推进复杂控制器的依据。保留固定引导、普通线性/MLP作为后续对照。','',
        '**下一项建议：在已有缓存上，检查去掉“生成后才知道”的信息后，普通MLP的收益是否还存在。** 当前10维特征中，完整路径分数、完整引导更新、原候选名次和第10名边界依赖生成完成；专家分数、历史长度、商品频次可以更早得到。应以训练标签重新拟合严格匹配的少量消融，冻结参数并完整报告用户外结果；单纯把已拟合网络的输入置零会有分布变化，不能替代匹配消融。','',
        '这是判断生成中可用性的必要信息检查，不是已发现新机制。若收益只依赖完成后的排序信息，当前证据主要支持后重排；若提前可得信息仍有收益，再讨论前缀状态如何使用它，并保留同信号后重排对照。本轮尚未运行这项后续消融，也没有追加开发样本或搜索阈值。','',
        '## 数据与比较范围','',
        '| 域 | 仅拟合 LIME 融合器 | 小头五折/最终拟合 | 用户外开发 |','|---|---:|---:|---:|',
        '| Toys | 721 | 2773 | 512 |','| Beauty | 775 | 3310 | 512 |','',
        '复用刚完成的截断训练 GRAM30、SASRec200、ItemCF与静态BGE向量；没有重新训练上游，未使用C1。融合器只见 gate_calibration 的末次训练交互，小头只见 ranker_fit 的末次训练交互。两组用户互斥，开发用户与两组均互斥；不是上游从未见过的新用户。GRAM/SASRec/协同统计均未学习被留出的末次事件。','',
        '评价分母包含所有预选用户；排序CE只能从目标在候选内且未被历史排除的训练行学习，门控则使用全部训练行（包括零收益）。所有标准化只使用当前训练折。开发名单采用目标无关SHA256规则，来自原开发评价池；这个池此前已反复用于研究，因此仍是开发证据。','',
        '**本轮小头只学习原 GRAM50 上的评分。** fixed_decode 是额外的真实引导生成对照，不是小头已经接入生成的成绩。本轮也不是9月20日“GRAM/LIME候选并集+MLP”的同配置重跑，不能把差异单独归因于训练泄漏修复。','',
        '## 开发集完整结果','',
        '| 方法 | Toys N10 | Beauty N10 | Toys H10 | Beauty H10 |','|---|---:|---:|---:|---:|']
    for name in names:
        a=domains['Toys']['development']['metrics'][name];b=domains['Beauty']['development']['metrics'][name]
        text.append(f"| {names[name]} | {a['N10']:.6f} | {b['N10']:.6f} | {a['H10']:.6f} | {b['H10']:.6f} |")
    text+=['','候选覆盖决定普通排序头能获得多少有效正例。未覆盖查询仍保留在全部评价分母中。','',
        '| 域/集合 | 查询数 | 原候选中有可推荐目标 | 目标在最近20条历史中 |','|---|---:|---:|---:|']
    for d in domains:
        with np.load(OUT/d/'data.npz',allow_pickle=False) as f:
            roles=f['roles'];present=((f['ids']==f['targets'][:,None]) & f['valid']).any(1)
            samples=read(OUT/d/'samples.json');repeated=np.array([int(t) in samples[int(j)]['history'] for t,j in zip(f['targets'],f['sample_indices'])])
            for role in ('ranker_fit','development'):
                mask=roles==role
                text.append(f"| {d}/{role} | {mask.sum()} | {present[mask].sum()} | {repeated[mask].sum()} |")
    text+=['','## 训练末次交互的用户外结果','','| 方法 | Toys OOF N10 | Beauty OOF N10 | Toys胜过两参照的折数 | Beauty胜过两参照的折数 |','|---|---:|---:|---:|---:|']
    for name in names:
        if name=='fixed_decode':continue
        a=domains['Toys']['oof']['metrics'][name];b=domains['Beauty']['oof']['metrics'][name]
        wins=[str(domains[d]['checks'][name]['positive_folds_vs_both'])+'/5' if name in domains[d]['checks'] else '参照' for d in domains]
        text.append(f"| {names[name]} | {a['N10']:.6f} | {b['N10']:.6f} | {wins[0]} | {wins[1]} |")
    text+=['','## 门控是否识别实际增量收益','','ΔN10指相同原候选上 fixed_late−GRAM。0阈值直接比较预测收益的正负；没有沿用旧0.70分类阈值，也没有在开发结果上搜索阈值。这是普通回退对照，不是旧第七阶段门控的精确复现。','',
        '| 域/集合 | 有帮助/有损害/零收益人数 | 干预率 | 帮助vs损害 AUC | 被采用用户实际平均ΔN10 | 被拒绝用户实际平均ΔN10 |',
        '|---|---:|---:|---:|---:|---:|']
    def fmt(v):return 'NA' if v is None else f'{v:.6f}'
    for d in domains:
        for group in ('oof','development'):
            b=domains[d][group]['benefit']
            text.append(f"| {d}/{group} | {b['positive']}/{b['negative']}/{b['zero']} | {b['intervention_rate']:.2%} | {fmt(b['auc_help_vs_harm'])} | {fmt(b['chosen_actual_mean'])} | {fmt(b['rejected_actual_mean'])} |")
    text+=['','补充预测误差对照：比较同一训练折的平均收益常数。此项只作诊断，不用于新增拟合、选阈值或改变投入判定。','',
        '| 域/集合 | 岭回归 MSE | 训练均值常数 MSE |','|---|---:|---:|']
    forecast={}
    for d in domains:
        with np.load(OUT/d/'ranks.npz',allow_pickle=False) as ranks:
            forecast[d]={}
            for group in ('oof','development'):
                prefix='oof' if group=='oof' else 'dev'
                def util(r):return (r<=10)/np.log2(r+1)
                y=util(ranks[prefix+'_fixed_late'])-util(ranks[prefix+'_GRAM'])
                with np.load(OUT/d/(group+'_predictions.npz'),allow_pickle=False) as pred:
                    constant=np.zeros(len(y))
                    if group=='oof':
                        for f in range(5):
                            c=torch.load(OUT/d/'heads'/f'fold{f}.ridge.pt',map_location='cpu')
                            constant[pred['folds']==f]=c['model']['intercept']
                    else:constant[:]=torch.load(OUT/d/'heads/final.ridge.pt',map_location='cpu')['model']['intercept']
                    learned=float(np.mean((pred['benefit']-y)**2));ordinary=float(np.mean((constant-y)**2))
                    forecast[d][group]=dict(ridge_mse=learned,training_constant_mse=ordinary)
                    text.append(f'| {d}/{group} | {learned:.8f} | {ordinary:.8f} |')
    (OUT/'forecast_diagnostic.json').write_text(json.dumps(forecast,indent=2)+'\n')
    text+=['','AUC只描述区分能力；干预率、实际推荐收益以及与训练常数回退的比较共同决定是否有用。知道答案后逐用户选最佳的oracle只存入诊断JSON，不作为可实现方法成绩。','',
        '## 开发增益/误伤与描述性区间','','下表相对 fixed_late。区间是固定预测的用户重采样描述，不包含上游训练随机性、五折共享训练、研究选择或多方法比较的不确定性。没有确认性显著性声明。','',
        '| 域/方法 | ΔN10 | 描述性95%区间 | 新增/丢失前十命中 |','|---|---:|---|---:|']
    for d in domains:
        for name in ('benefit_ridge','linear_calibration','mlp_seed0','mlp_seed1','mlp_seed2'):
            c=domains[d]['development']['contrasts'][name]['fixed_late'];lo,hi=c['descriptive_95pct_N10']
            text.append(f"| {d}/{names[name]} | {c['delta_N10']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {c['top10_gained']}/{c['top10_lost']} |")
    text+=['','## 实际生成与同规则后重排','','| 域 | fixed_decode−fixed_late ΔN10 | 新增/丢失前十命中 |','|---|---:|---:|']
    for d in domains:
        c=domains[d]['development']['fixed_decode_vs_same_rule_late']
        text.append(f"| {d} | {c['delta_N10']:+.6f} | {c['top10_gained']}/{c['top10_lost']} |")
    text+=['','这是新上游和新融合器的匹配对照，不能拿旧模型的最佳提升幅度直接套用。','',
        '## 工程和独立复核','','- 6项CPU检查通过：标签/历史掩码、目标无关特征、评价标签与特征改变不影响拟合、预测分块重放、正负收益和按用户固定分折。',
        '- 首次预检查发现跨注意力缓存复用造成最高1.43e−6的分数差异，未通过精确一致。失败记录完整保留，关闭该提速后重新执行；未生成正式候选、未查看开发成绩、未改变科学配置。',
        '- 修复后两域各3人原生/包装入口与零强度逐值一致；每条完成路径分数在生成时重构核对；上游模型前后digest一致。',
        '- 独立复核脚本未导入本轮拟合/评价运行器，重新排序计算指标、重算特征、核对用户名单/掩码/角色/哈希、复算岭回归并重放全部48个校准头。此处独立指代码路径，不是外部独立研究者复核。','']
    for d in domains:
        c=read(OUT/d/'cache_summary.json');v=audit['domains'][d]
        text.append(f"- {d}：{c['users']}名用户，{c['decodes']}次成功批次生成，缓存阶段{c['seconds']/60:.2f}分钟；复核{v['checked_ranks']}个排名。")
    text+=['','成功批次LIME融合器优化Toys 12次、Beauty 16次；校准头共1440次更新；上游GRAM/SASRec更新为0。逐域运行status继承的optimizer_steps=0指冻结生成器，不能解释为融合器和小头完全没训练。首次失败尝试的融合器更新另保存在工程归档。','',
        '## 统计解释检查','',
        '11/11检查：两域/五折分开报告；不用群体均值推断每个用户；训练覆盖筛选明确但评价不筛；不按目标覆盖做因果控制；完整报告收益基率；不挑极端失败者称恢复；全名单无掉样；全部固定方法和三seed报告；冻结规则与工程修复留痕；评分控制不能识别唯一现实因果机制；特征只来自推荐时可得信息。仅一个上游随机种子、每域512人开发，负结果不证明所有方法无效。','',
        '产物：[冻结协议](../../plan/第二十阶段/LIME干净监督_增量收益可学性检查_20260922.md)、[完整结果](../../artifacts/phase20/lime_clean_benefit_20260922_v1/summary.json)、[独立复核](../../artifacts/phase20/lime_clean_benefit_20260922_v1/independent_verification.json)、[运行状态](../../artifacts/phase20/lime_clean_benefit_20260922_v1/status.json)。','']
    REPORT.write_text('\n'.join(text));print(REPORT)


if __name__=='__main__':main()
