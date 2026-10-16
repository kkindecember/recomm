# 生成决策统计特征匹配对照

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-16
- Verification Status: PLANNED
- Version Label: decision_features_v1

用户授权试验7维特征。范围仅为TiSASRec+BGE、用户级联合融合器、Toys/Beauty既有512开发用户、seed0；不扩宽网络、不重训专家或GRAM、不启动完整论文实验。

## 假设与干预

原10维为历史长度、两专家各4项分数统计、目录历史占比；新7维为log1p(完整可见历史长度)/5、三方首层分支分布的归一化熵、行为/内容、行为/GRAM、内容/GRAM三对JS散度除log(2)。熵除log(分支数)。所有分布在同一排序后的合法首层token空间；Toys30分支，Beauty108分支。GRAM取仅decoder起始token0的前向结果并在合法token上归一化；专家分别按既有exp(20*score)商品质量汇总到首层分支，保留现有历史处理。无新温度、无topK截断、无融合后分数作为输入。

新特征不读取目标token、正确路径概率、目标日期或未来信息。特征提取数据集将target替换为固定占位商品；检查改占位目标不改变模型输入/输出。训练损失可使用真实标签与此前冻结的teacher缓存，缓存不作为输入特征。

两组均为32维ReLU隐藏层、softmax双专家比例与sigmoid用户强度，共享隐藏层。旧10维451参数，新7维355参数；相同隐藏宽度但不宣称严格等参数量。每用户一次计算权重，整次生成共享；专家前缀条件分布仍在每一步影响生成。

## 对照与训练

- old10：复用已验证的TiSASRec旧CE父融合器及内部fit-only父融合器，重新执行相同联合适配；重放历史输出检查。
- decision7：相同初始化规则（现有16输入原型的前7列，保持初始化尺度）、CE父训练规则及联合适配规则。父CE20轮、候选0/4/8/12/20，随后全ranker_fit重拟合所选轮数；内部选轮用fit-only父参数。
- 两组联合适配：seed0，batch32，AdamW lr.002/wd.001，20轮、候选0/4/8/12/20；内部验证正确路径长度归一化NLL选轮数，随后全ranker_fit refit。
- 两组标准化均仅拟合当前训练部分；相同用户hash80/20切分；Toys2773、Beauty3310训练用户，与开发512不重叠。
- 旧CE父轮数已按同一内部验证规则确定；内部留出跨阶段复用，非独立确认集。不用开发表现选轮数/参数，不读test。
- 三方分布不是独立证据，熵不是正确率，JS不是纠错收益。首层统计不代表后续所有分支。不声称各特征必要或最优。

## 生成与核验

两组均执行fresh beam50完整生成，同时记录相同规则native50后重排。原GRAM和旧联合融合结果作参照。概率混合、legal mass、score_scale20、历史过滤、length penalty1全部不变。拟合后独立NumPy重算所有512用户的beta/alpha/商品分数。首层GRAM分布与真实beam trace对照，专家首层汇总与原ExpertTrie对照，熵/JS边界、交换对称性及直接KL公式核验。保存输入/代码/训练/输出哈希，重读所有预测独立排序、重算H/NDCG。

主要比较decision7_guided与old10_guided的NDCG@10，另报原GRAM与各自后重排，保留H@5/10/20/50、N@5/10/20/50及配对描述性区间。现有开发池反复使用、单seed、未经多重校正，结果仅作方法探索。两域结果全部报告，不按效果中止或换特征。

## 资源与交付

按用户此前减少30GiB预留的要求，每域预算16GiB（工作峰值12GiB、sidecar4GiB），启动空闲显存18GiB，工作进程分配上限11GiB；最多两域并行，优先低负载且显存满足的GPU。冻结历史资源策略文件不修改。每域6小时硬上限、资源等待24小时；科学失败不自动重试，只有未开始科学计算的资源准入失败可重新排队。后台tmux，状态每10秒写入根status.json及分域文件；完成自动汇总并释放GPU。

输出：artifacts/phase21/decision_features_20261016_v1；报告：report/第二十一阶段/生成决策统计特征匹配对照_20261016.md。完成启动核验后交还用户，不持续人工监看。
