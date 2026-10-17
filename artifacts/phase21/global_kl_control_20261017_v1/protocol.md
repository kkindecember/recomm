# 全局α＋同KL损失的匹配对照

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent，内联执行
- Origin Mode: run；用户授权补充全局α＋KL对照
- Origin Date: 2026-10-17
- Verification Status: PLANNED；21.50两域512结果已知后的探索性补充
- Version Label: global_kl_control_20261017_v1

## 唯一新增生成组

固定SASRec+BGE，β=(.5,.5)，学习全域统一α=clamp(sigmoid(b),1e-6,1-1e-6)，只适配一个参数。目标为原方案完全相同的目标路径NLL + .1 KL(p_GRAM||p_mix)。不增加λ网格、不改专家评分、无历史过滤/惩罚/去自匹配，beam50、score scale20、长度惩罚1。不读test。

主要比较global_kl1与已完成的user_kl1（33参数）；参照原生GRAM、fixed03、global_nll、user_nll。所有参照复用同用户保存预测，新增组真实生成。关键问题是用户级α能否超过同损失下的一参数全局α，而非仅超过默认固定.3。

## 数据与优化

复用alpha_mechanism_20261017_v1的完整候选池、冻结GRAM逐路径native/legal、等权专家逐路径q、fit样本及hash折。Toys2773、Beauty3310；fold!=0拟合，fold==0选epoch，随后全fit重新拟合。评估沿用同一512用户，完整名单与fit隔离，不换用户、不按此前效果选人。

与原训练保持一致：seed0，初始α=.3，AdamW lr=.002，weight_decay=.001，batch32，最多20轮，epoch候选0/4/8/12/20，以内部留出NLL+.1KL最小选epoch。重新从.3初始化，不从已知较好的用户头热启动。

当前GPU均未达到18GiB空闲准入，故单参数拟合在CPU float32上完成。先以λ=0重放历史global_nll训练，要求选中epoch一致、各checkpoint留出损失与最终α对历史GPU结果误差≤2e-5，记录具体误差。此重放只核验数值流程，不新增生成组、不参与λ选择。若不通过则停止，不能静默放宽阈值。然后用同一流程拟合唯一新增λ=.1组。CPU/GPU后端差异显式披露，不称逐位复现。

在正确目标索引上核验teacher.targets=sample.target−1=pool首列−1，目标路径q与缓存一致；α=.3的完整训练NLL与原训练审计一致。缓存及参照输入均校验原manifest哈希，不新建专家或GRAM训练缓存，不覆盖原目录。

## 生成与报告

每域512用户，仅新增global_kl1生成。smoke在第0/256/511行核验原生重放、α=0精确退化、固定.3重放、用户KL=.1重放、target占位输入不变；新α的逐路径重评分与真实beam分数核对。输出保存所有参照与新组预测，并逐条校验参照未改变、目标排名/指标独立重算。

报告两域NDCG/Hit@5/10/20/50、全局α、主要global_kl1 vs user_kl1配对差、救回/丢失及共同命中排序变化。配对bootstrap2000次，全部预定参照保留，区间未校正多重比较；没有独立test、没有多种子，不因某域胜出选不同方法。

已知21.50结果后才补此对照，属于事后探索；不把“增加了强对照”称为新的独立确认。完成两域各512即停止，不自动扩大4096或追加下一轮。

## 资源和记录

CPU阶段两线程；生成最多两张卡，每任务16GiB总租约（12GiB工作预算+4GiB侧车）、18GiB空闲准入、11GiB模型分配上限。GPU不足时后台排队，不改准入线、不停止其他任务。单域生成30分钟硬超时，资源等待24小时；科学失败不自动重试。

- 输出：artifacts/phase21/global_kl_control_20261017_v1/
- 报告：report/第二十一阶段/全局alpha同KL损失匹配对照_20261017.md
- 入口：experiment/phase21/global_kl_control.py
- tmux：s21_global_kl_control_20261017_v1
