"""Report all fixed comparisons, source errors and the conditional coverage ceiling."""
import json
from pathlib import Path
import numpy as np
from analyze_selective_evidence import sha,write
from extract_typed_relations import ROOT,OUT

REPORT=ROOT/'report/第二十一阶段/实体角色与版本分层检查_20260925.md'
LABELS={'previous':'21.15主规则','typed_parent':'区分自身／提及，仅父家族',
        'hierarchy':'父家族＋细分版本（主）','fine_only':'只用细分版本',
        'score_matched':'同用户分数对照','shuffled':'错配历史'}


def table(summary,cohort):
    lines=['|域／政策|修改人数|救回|误伤|净增|N10|','|---|---:|---:|---:|---:|---:|']
    for domain in ('Toys','Beauty'):
        for policy,r in summary['domains'][domain][cohort]['policies'].items():
            lines.append(f"|{domain}／{LABELS[policy]}|{r['actions']}|{r['gained10']}|{r['lost10']}|{r['net_hits']:+d}|{r['N10']:.9f}|")
    return '\n'.join(lines)


def main():
    summary=json.loads((OUT/'summary.json').read_text());extraction=json.loads((OUT/'extraction_summary.json').read_text())
    verification=json.loads((OUT/'verification.json').read_text());assert verification['state']=='VERIFIED'
    tables=[];ceiling={}
    for cohort in ('primary','train_holdout','previous512','seen_fit_user_later_event','all'):
        tables.extend(['## '+cohort,'',table(summary,cohort),''])
        lines=['|域／主政策对照|净命中差|ΔN10|95%用户配对区间|','|---|---:|---:|---|']
        for domain in ('Toys','Beauty'):
            for name,r in summary['domains'][domain][cohort]['contrasts'].items():
                ci=r.get('ci95');interval='未计算' if ci is None else f'[{ci[0]:+.9f}, {ci[1]:+.9f}]'
                lines.append(f"|{domain}／{name}|{r['net_hit_difference']:+d}|{r['delta_N10']:+.9f}|{interval}|")
        tables.extend(['\n'.join(lines),''])
    (OUT/'all_tables.md').write_text('\n'.join(tables))
    for domain in ('Toys','Beauty'):
        part=summary['domains'][domain]['primary'];helpful=part['label_groups']['helpful']
        ceiling[domain]=dict(users=part['users'],expert_rescuable_targets=helpful['all_users'],
            comparable_targets=helpful['comparable_users'],version_comparable_targets=helpful['version_users'],
            admissible_targets=helpful['qualifying_users'],max_extra_hits=helpful['qualifying_users'],
            max_extra_N10=helpful['qualifying_users']/part['users']/np.log2(11),
            fraction_of_expert_rescuable=helpful['qualifying_users']/helpful['all_users'])
    write(OUT/'conditional_coverage_ceiling.json',dict(domains=ceiling,
        interpretation='Hindsight bound using labels only for diagnosis; conditioned on this candidate set, metadata, known-state requirement and admissibility rule. Not a bound on all semantic methods or GRAM.',
        deployable=False))
    prefix='../../artifacts/phase21/typed_relations_20260925_v1/'
    text=f'''# 实体角色与版本分层检查

## Material Passport

- 日期2026-09-25；Origin Skill：academic-research-suite，内联执行。
- 状态COMPLETED；来源与数值Verification Status：VERIFIED。来源意义由同一助手审核，不是独立标注。
- [执行前固定协议](../../plan/第二十一阶段/21.16_实体角色与版本分层固定检查_20260925.md)。用户授权继续解决21.15暴露的主体／提及混淆和父主题合并问题。
- 提取{extraction['seconds']:.2f}秒，六政策评价{summary['seconds']:.2f}秒；无新拟合、无GRAM／专家训练、无GPU推理、无test读取。21.6保持USER_PAUSED。
- 提取和来源审查先于本轮效果评价。整个问题来自已探索的validation及已知案例，不能当作独立确认，尤其新增救回仍是此前已看过的案例。

## 结论

**版本分层恢复了1个已知案例，但没有形成覆盖充分的新机制。** 主要验证Toys15406／Beauty17766人中，主政策修改25／9人，救回2／0、误伤0／0；上一轮是修改16／9人，救回1／0、误伤0／0。同动作用户按原分数选候选，救回1／0、误伤0／0。

新增的Toys救回就是此前已发现的Walking Dead TV Series 4对Series 3案例。这能验证修复确实保留了该区别，不能当成一条独立新证据。Beauty没有产生版本替换，不能说两个域都支持这个机制。

更关键的是当前方法的上限：原专家候选中可救回的目标有735／862个，但这份关系表示与可采纳条件只覆盖其中2／0个。**在保持当前候选、关系和采纳条件的前提下，即使每次都选对且零误伤，也最多增加2／0个命中。** 该样本内上限已被达到，继续调整候选选择器并不能补出缺失的比较信息。

这不是全体语义方法的上限。它约束的是当前66个种子家族、标题提取、双方已知、固定第10名替换这一整套做法。该上限用目标标签事后计算，只用于诊断，不是可部署方法。

## 1. 本轮修了什么

沿用21.15的66个名字。标题首次出现for／with／works／compatible等固定界限之前，才可作为商品自身身份；之后的家族名只保留为“提及”，不自动建立兼容关系。媒体scope、series数字批次和套系年度也只从主体窗口读取，缺失保留未知。

年度仅取季节词或collection紧邻的1900–2099年份，不把任意四位商品编号当年份；1 inch／1.5 inch不会被当成数字批次；多个冲突字段值不选第一个。

|来源覆盖|Toys|Beauty|
|---|---:|---:|
|目录商品|11924|12101|
|可识别自身父家族商品|154（1.29%）|140（1.16%）|
|至少一个明确细分字段|75|15|
|明确媒体scope|33|0|
|明确series批次|65|0|
|明确套系／季节年度|0|15|
|主体身份审核支持|32/32|33/35|
|细分字段审核支持|32/32|15/15|

主体审核包括每域哈希样本以及全部身份改变商品，按item去重；细分字段样本不足32则全审。未知字段没有被补成默认版本。≥10个单位、支持≥90%只是事前工程初筛，不代表全目录精确率或漏提取率。

Beauty身份改变3件：

- 9993是烘干灯，标题在for之后提到OPI GelColor；本轮正确删除“商品自身属于该指甲油家族”的判断，保留提及。
- 4145和4907原本属于Bath & Body Works的Japanese Cherry Blossom Signature Collection；单词works被误当语法界限，身份被错误截断。这两次漏识别明确计入质量分母，没有为通过筛查而隐藏，也没有在看效果后补品牌例外。

Toys154件身份未变化。父家族仍可能跨生产方、功能或商品形态；同Walking Dead主题不能自动当成同功能产品。保存的“提及”不是已经验证的配套边，本轮没有将适配边用于推荐。

## 2. 如何公平比较两个候选

候选A仍来自三专家前10并集中、原融合前10之外；B仍是原融合第10名，前9名不变。A和B必须都有自身家族标签。

- 家族不同：比较去重历史中各自家族的商品数。
- 家族相同：检查该家族在目录中出现过的scope／batch／year维度。A、B必须都有全部要求字段，再比较各自具体版本的历史支持；任意字段缺失则该A/B不可比较，不退回成“缺失=没有支持”。
- 没有细分维度、或两边版本相同，不据此替换。候选自身重复不算另一件历史支持商品。
- A支持至少1且大于B才可采纳；多个A按支持数、原融合分和ID选取，没有搜索阈值。

这仍只是冻结的可检验启发式，没有训练一个懂语义的模型。

## 3. 六组主要验证结果

{table(summary,'primary')}

表内救回／误伤均**相对原融合列表**，不代表相对原始GRAM的新成绩。原融合N10为0.081115060／0.071651869；主政策为0.081152586／0.071651869，即ΔN10 +0.000037526／0（Toys相对变化约+0.046%）。

只修正身份角色、仍按父家族比较，主要组结果与21.15相同。加入版本后，Toys新增9次修改，其中1次救回、8次中性；Beauty新增0次。主政策Toys的25次动作由16次不同父家族比较、9次同家族版本比较组成。

主政策比同动作用户分数对照多1个净命中，ΔN10 +0.000018763，2000次用户配对bootstrap的95%区间[−0.000037526,+0.000075053]，包含0。主政策比上一轮多1人的区间为[0,+0.000056289]；事件少、案例已知、验证集反复探索，不能解释为稳定增量。各个对照全部保留，没有多重比较校正。

更早21.14的“旧系列＋未知不改”也已经在Toys救回2人、误伤0。当前两次救回对应同样的两位用户，不能把恢复到这一结果称为超过所有旧机制。

## 4. 唯一恢复的版本案例

用户A2KNSFX7T24ET7，目标A10778“Walking Dead TV Series 4 Riot Gear Gas Mask Zombie”，原第10名B10096“Walking Dead TV Series 3 Michonne”。

|比较依据|A历史支持|B历史支持|决定|
|---|---:|---:|---|
|上一轮只看Walking Dead父主题|3|3|不改|
|本轮要求TV媒体＋具体批次|1|0|采纳A，恢复命中|

支持A的是同属TV Series 4的Dixon Brother商品10773。另两件同父主题历史不会再被无条件算成两边的具体批次支持。

这说明**在该例中，关系要带上层级才有区别能力**。但同批次偏好不是已验证的用户心理原因，也没有证明所有系列都应按批次推荐。全部主要非中性案例3个（Toys两个主政策救回和一个分数对照救回；Beauty无）均保存，未只报这个有利例子。

## 5. 为什么不值得继续在这份关系上调选择器

|主要验证覆盖与条件上限|Toys|Beauty|
|---|---:|---:|
|全部候选建议对|342874|395644|
|满足本轮可比较条件的建议对|336|84|
|其中可比较具体版本的建议对|89|1|
|存在具体版本比较的用户|33|1|
|专家建议中存在可救回目标的用户|735|862|
|其中目标A/B可比较|2|0|
|其中目标还满足采纳条件|2|0|
|其中依靠细分版本的目标|1|0|
|当前条件下最多可新增命中|2|0|

两域合计1597个候选可救回目标，只有2个进入当前可采纳关系，覆盖约0.125%。主政策已经选对这两个，限制主要在**看不到其他目标的可比较依据**，而非这两例上还缺一个更强的打分器。

此外，原第10名正确、专家建议可能造成误伤的用户有87／90人；本轮可比较关系仅触及其中4／0人，最终均未触发替换。零误伤是在极窄动作范围下得到的结果，不能推广为能保护所有原正确判断。

全量用户不少，但有用事件仍少。Beauty本轮细分维度只有少量套系年份；无法据此判断更完整的功能、适配或具体用途关系有没有价值，也不能据此说GRAM读不懂语言。

## 6. 其余固定分组与运行决定

全部validation共19412／22363人：主政策修改31／10人，救回2／0、误伤0／0；同用户分数对照救回1／0、误伤0／0。旧512与旧拟合用户后续事件没有新增救回／误伤。

训练留出2773／3310人中，主政策修改3／1人，救回和误伤均0；没有新拟合，不能称为新模型折外结果。所有分组见[完整表格]({prefix}all_tables.md)。

错配历史在主要组只触发Toys1次、Beauty0次，均中性；主要组无法换人的单例18／17人，本轮错配动作没有来自这些单例。动作人数不同，不把差值解释为纯信息因果效应。

**本轮完成后停止扩展这套窄覆盖的字符串采纳规则，不升级生成训练。** 现有GRAM与三专家方案保持原状态。下一步若继续，需要先面向更广的专家分歧建立不展示推荐目标的来源审查样本，确认可提取依据的关系类型、可靠性以及双方覆盖，再决定是否存在值得接入GRAM的信号。仅在已知Walking Dead案例附近加规则，不能补足机制证据。

这项建议不表示其他关系必然有效，也不预设需要再加模型。未自动启动后续实验，21.6仍暂停。

## 7. 核验与统计解释

- 36个来源哈希核验通过；24025件标题主体记录及308条字段证据跨度核验通过（字段输出包括尚未归入家族的商品）。
- 1063607条候选支持记录、287148次用户—政策决定和60行分组指标独立重算一致；旧政策逐用户复现21.15。
- 10项有意义的回归检查覆盖尺寸、序号、年度上下文、冲突年度、提及对象、缺失版本、同版本不得替换；质量门槛从114个审核单位复算。
- 数值核验不替代独立语义审查。标题界限仍有误伤品牌、丢失身份的局限；没有估计全目录召回率。
- 60个政策结果、50个对照全部报告。仅主要／训练留出提供2000次配对区间，不包含上游种子随机性。稀疏或全零样本产生的[0,0]区间不能被解释成总体无不确定性。

统计解释11/11：两域和固定分组分别报告（Simpson）；按用户而非候选对重采样（生态谬误）；明确种子、主体截断和已知条件限制（选择偏差）；条件上限只对当前可采纳集合成立（碰撞偏差）；报告1597个潜在救回中只有2个可用（基率）；新增案例已知，不能当独立确认（均值回归）；两次身份漏识别及所有对照保留（幸存者偏差）；六政策全报不筛有利域（多重探索）；协议和审核先于当前效果、不补例外（分叉路径）；同批次支持不等于真实购买原因（相关因果）；目标仅用于统计及事后上限，提取和选择不接受目标（反向因果／未来信息）。

## 文件

- [完整数值]({prefix}summary.json)、[条件覆盖上限]({prefix}conditional_coverage_ceiling.json)、[全部主要非中性案例]({prefix}all_primary_non_neutral_cases.json)
- [来源审核]({prefix}source_review.json)、[质量门槛]({prefix}quality_gate.json)、[Toys提取]({prefix}Toys_relations.json)、[Beauty提取]({prefix}Beauty_relations.json)
- [独立数值核验]({prefix}verification.json)、[状态]({prefix}status.json)
- [提取和比较规则](../../experiment/phase21/typed_relation_core.py)、[固定评价](../../experiment/phase21/evaluate_typed_relations.py)、[核验代码](../../experiment/phase21/verify_typed_relations.py)
'''
    REPORT.write_text(text)
    dashboard=ROOT/'report/第二十一阶段/在跑实验总览_20260923.md';marker='<!-- typed_relations_20260925 -->'
    if marker not in dashboard.read_text():
        intro=marker+'\n**9月25日实体角色／版本分层检查完成：主要组Toys15406／Beauty17766人，主政策修改25／9人，救回2／0、误伤0／0；比21.15仅恢复1个已知批次案例，同用户分数对照净增1／0。1597个专家候选可救回目标中，当前关系及采纳条件仅覆盖2个；该条件上限已达到，停止扩展这套窄覆盖字符串规则。主体／版本来源审核保留2个品牌截断错误，未包装成完整语义理解。见[完整结果与覆盖上限](实体角色与版本分层检查_20260925.md)。来源和106万候选记录复核通过；无新拟合／训练／GPU推理，21.6仍暂停；本轮已结束，未自动启动下一轮。**\n\n'
        dashboard.write_text(intro+dashboard.read_text())
    for name in ('audit_typed_relations.py','evaluate_typed_relations.py','verify_typed_relations.py','summarize_typed_relations.py'):
        p=Path(__file__).with_name(name);(OUT/name).write_text(p.read_text())
    status=json.loads((OUT/'status.json').read_text());status.update(verification='VERIFIED',report=str(REPORT.relative_to(ROOT)),semantic_training_state='USER_PAUSED')
    write(OUT/'status.json',status)
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='verification_manifest.json'}
    hashes[str(REPORT.relative_to(ROOT))]=sha(REPORT)
    write(OUT/'verification_manifest.json',hashes)
    print(str(REPORT),flush=True)


if __name__=='__main__':main()
