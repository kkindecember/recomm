# PCRF机制对照与研究收敛计划

**2026-09-12终态核查：** Toys七臂已完成；泛化CE＋教师建议仍撤回；GACR组合已出现额外正结果，当前建议见新计划。 下文保留原协议；后续见[收尾后验证与方法决策计划](第20阶段收尾后的验证与方法决策计划.md)，尚未启动新的实验。

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-11
- Verification Status: PLANNED；执行结果另存，不把设计写成已证实方法。
- Version Label: pcrf_component_audit_v1 / historical_teacher_correction_20260911

## 研究问题与本次范围

用户要求复盘长期未形成方法的原因，并选择下一步值得运行的实验。沿用第20阶段最近明确的目标：先获得可靠正常推荐增益，允许融合与模型内部更新；创新性需另行核对。

**最新目标澄清（2026-09-11）：用户要求形成有实质贡献的方法，不接受仅将PCRF补齐实验后当作论文主方法。** PCRF已有效的结论保留；“足以独立支撑方法论文”尚未建立。下一步优先完成方法问题与机制设计、历史去重及最接近工作比较，Beauty组件消融降为辅助工作，不自动启动。见[贡献边界复核](../../report/第二十阶段/PCRF有效性与论文贡献边界复核.md)。这一目标修正不改变已运行实验的原协议。

先回答一个成本低且直接影响主线取舍的问题：**在相同父模型和候选上，PCRF的校准与可靠性权重，相比同权重的普通分数融合，究竟提供了多少新增收益？** 本次是固定参数的描述性消融，不拟合新参数、不训练骨干、不重新生成beam，不作为新的独立确认。

已知前提：PCRF整体相对原GRAM的收益已经有Toys/Beauty各三个item-head seed的validation正结果及两域各一次冻结test正结果支持。组件对照研究额外机制贡献，不将PCRF整体降回“尚待证明有效”；后续Beauty消融也不是保留PCRF的前提。跨域沿用既有训练频次计算规则与该域参数记录，历史Beauty q1=6，不能机械复制Toys q1=5。

SPRec、已保存SFT评估及GACR/PCRF组合继续按既定任务运行。本次不启动新的长训练，也不恢复已经停止的分支。

## 固定输入与比较

Toys全量19,412名validation用户；第20阶段同精度父模型beam50及其逐用户参考，固定seed2023 item-head。只读取prepared train/validation，训练频次按原实现计算；不打开原始全序列文件或test。

预先固定七个臂：原GRAM、候选内CF独排、普通融合lambda0.75、普通融合lambda1、仅流行度校准、仅可靠性权重、完整PCRF。所有候选集合完全相同，排序使用原稳定排序规则。主比较是完整PCRF相对lambda1普通融合的NDCG@10；lambda0.75是既有第9阶段参照，不从两个普通融合中挑赢家。

完整PCRF固定lambda1/beta0.5/gamma1/q1=5。两个单组件臂仅将另一个组件设为identity，保留其余条件。组件之间可能存在交互，不能把消融差简单相加当作贡献百分比。

报告Hit/NDCG@5/10/20/50、总体与目标频次≤5/>5分组、Top10命中得失。主比较及消融使用2,000次配对用户bootstrap，seed20260911；所有区间仅描述已反复开发validation上的用户变化，不表示训练seed不确定性或独立显著性。不得根据结果修改这些权重。

## 工程与预算

入口：`python -m experiment.phase20.audit_pcrf_components`，配置为[固定配置](../../experiment/phase20/pcrf_component_audit.json)。CPU最多4线程，GPU不可见，外部`timeout`硬上限600秒。当前输出目录：`artifacts/phase20/pcrf_component_audit_20260911_v2/`，产出status、summary和逐用户rank；拒绝覆盖已有目录。

先验证用户集合、target、频次、候选合法唯一、分数有限；完整PCRF与原GRAM逐用户rank必须和已有同精度父参考精确一致。所有输入在前后核对SHA256。工程失败必须记录，不发布不完整科学结论。

首次工程检查在模型评分前停止：`matched_parent_reference.json`仅更新过raw/PCRF rank，保留的`tail_mass`元数据来自历史beam，不能要求它与新FP32候选完全一致。首次目录`pcrf_component_audit_20260911`保留失败状态、配置哈希和源码；v2依据实际候选重算tail_mass并记录元数据差异数，继续严格要求最终raw/PCRF rank逐用户精确一致。没有读取新消融指标或改动科学参数。

## 结果怎样影响下一步

- 若PCRF相对普通融合有可辨认的总体或分层收益，保留这项机制作为方法基础；进一步跨域组件验证和相关工作核对，不能仅凭一个消融声称原创。
- 若其主要收益可由普通融合解释，就收窄PCRF贡献表述，不再靠增加公式复杂度扩写结论。
- GACR组合若超过原PCRF，再补冻结参数的跨域、独立训练seed验证；若组合不增，不自动扩GACR参数网格。
- 撤回把“保留CE、用PCRF教师指导原GRAM”列为优先新方向的建议。第18阶段PCPS已实际执行CE与CF/PCRF引导的前缀、完整路径辅助损失，并设置普通机制与打乱CF对照。该宽泛思路不是尚未尝试的空白；本轮不因更换教师名称而启动重复训练。

## 历史教师核验与修正（2026-09-11）

用户指出以前做过PCRF机制扩展后，本轮核对了实际损失、优化器与执行结果，不能只凭旧计划中残留的“未实施”字段判断执行状态。

| 已执行路线 | 教师或协同信息怎样进入 | 结果与范围 |
|---|---|---|
| 第9阶段P9-2B安全融合 | 冻结GRAM和item encoder，将协同user state经零初始adapter注入coarse prompt；属于模型内部协同注入，并非PCRF排序蒸馏 | 109,361个训练样本、2 epochs；4,096条验证NLL变差，完整beam未跑。 |
| 第10阶段CF1-C2 | PCRF分数作排序锚点，加Top10非gold相邻对的顺序保持损失、监督排序和流行度平衡；训练外部残差排序器 | Toys 19,412人五折OOF完成；NDCG@10相对PCRF约-0.000017，Hit@10 -0.000567。 |
| 第18阶段PCPS / S18-2 | 固定协同教师/PCRF指导困难前缀负例，校准CF与可靠性改变完整路径负例权重；CE与两项辅助损失共同更新GRAM | Toys I0 1,000人四臂完成；PCPS NDCG@10 0.042276，低于普通机制0.043918、打乱CF 0.042834及frozen parent 0.051310。 |

PCPS不是完整beam50教师概率分布的KL蒸馏，但已经覆盖了“CE＋协同教师指导生成”的主要设计方向。此次没有找到该完整分布蒸馏版本的已完成记录；这只是具体实现边界，不能据此把宽泛教师设计说成全新方案，也不自动构成值得训练的理由。

若以后重新提出训练候选，必须先说明相对PCPS的实质差异，以及它处理了哪项旧证据：所有继续训练臂低于parent、局部sibling代理不等于全局beam剪枝边界、真实CF尚未优于普通/打乱对照。小样本、训练数据分布、优化器重置和batch变化的影响尚未被分离，不能把其中一个写成确定根因。

证据：[第9阶段结果](../../report/第九阶段/GRAM_第九阶段_CF0-B2_P9-2B零初始安全融合结果与验证报告.md)、[第10阶段结果](../../report/第十阶段/GRAM_第十阶段_CF1-C2锚定PCRF正式结果报告.md)、[第18阶段复核](../../report/第十八阶段/Stage18_S18-2_失败后机制复核与文献研究.md)、[PCPS实际损失](../../experiment/phase18/core/pcps_loss.py)、[PCPS实际训练](../../experiment/phase18/protocol/s18_s2_mechanism_probe.py)。

## 本轮文献边界

截至2026-09-11的定向检索已确认：[UniGRF](https://arxiv.org/abs/2504.16454)包含ranking-driven generation enhancement；[SmartGR](https://arxiv.org/abs/2608.02048)包含beam-aware ranking distillation，后者为2026-08预印本。因此“用排序指导生成”或“beam感知蒸馏”本身不能作为本项目的新颖性声明。当前只做相关工作边界核对，不是完整系统综述，也不据论文摘要承诺本项目提点。
