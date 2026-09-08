# 第十八阶段：第十七阶段失败复盘与大改选型依据

## Material Passport

- 日期：2026-09-08
- 工作：本地实验回顾、已保存预测的 CPU 重算、后续架构选型修订
- 来源：academic-research-suite / experiment-agent；inline
- 状态：ANALYZED；S17-4 指标已从冻结预测重算，未重训或重新推理
- 用户要求：已有 GRAM / PCRF 数据优先；当前重点找到提点结构；接受大幅改动；方法确定后统一重跑

## 1. 主结论

第十七阶段已经尝试过顶会机制迁移，而且包含完整架构迁移。不能把失败全部归结为“只做了 lite”，也不能把所有登记过的论文都视为已经充分验证失败。

失败经验应改变两个方面：

1. 选型不能只看论文名气或模块大小，要看其在本地数据上依赖什么能力，以及这次实现与旧失败相比究竟改变了什么。
2. 允许重构 encoder、decoder、identifier、交互方式和训练范式，必要时从头训练。词法 ID 和 FiD 是已经有价值的资产，但不是用户规定必须保留的硬约束。替换它们需要说明被拿走的能力如何得到补偿，并尽早看绝对效果。

当前 PCPS 仍暂停。本文也撤回“DIFF / DIF-SR 已经应当稳居第一优先”的过早排序；它们留在候选池，与有实质差异的大改路径一起比较。

## 2. 实际试过什么，负结果能说明什么

以下数值只在各自实验内部比较，不能跨 S3 official validation、S4 shadow D0、FP fresh D0 直接相减。这里引用的 FP G0 不是用来替换第一、第二阶段历史基线。

| 方向 / 阶段 | 实际实现与预算 | 已有结果 | 对新选型的影响 |
| --- | --- | --- | --- |
| BEAR / PrefixCurr，S3 | full-vocabulary survival proxy / 深度课程；从 epoch-30 parent 续训 1 epoch | BEAR 相对续训对照 ΔNDCG@10 +0.000165；课程 -0.000086；均未超过历史 parent | 没有稳定结构收益。仅能评价当前 proxy / 课程，不等于所有前缀结构失败 |
| MVI / LATTE-lite，S3 | 第二个词法路径 / 确定性 hash root；非原论文完整机制 | MVI -0.002635；LATTE-lite -0.000023，且 Hit 下降 | 不重复“额外路径或 root + 聚合”这种同构改法 |
| BiFlow / TED，S3、S4 | pooled 全局 / 历史状态之间门控注入；转移 teacher + recent/long 注入 | S3 BiFlow -0.000588，TED -0.000724；S4 单向版仍无稳定正信号 | 原始机制未被完整复现；也没有证据支持继续给均值池化门控换名字 |
| Shortcut-FiD，S3 | 语义历史选择；有全历史、同规模随机选择对照 | semantic NDCG 比随机低 0.000698、比全历史低 0.000204 | 删除历史比例和“过滤噪声”的故事不能代替推荐效果；类似新筛选器要避免重复此机制 |
| Gryphon-item / DiffGRM，S17-2R | 独立缩放实现，3,000 用户；约 5.36M–8.12M 参数；各自在 5 epoch 结束 | Gryphon ΔNDCG -0.001990；DiffGRM -0.002883 | 负证据针对当前缩放映射；不应写成官方完整模型已失败 |
| LATTE，FP1 / FP2 | Native 使用 pinned official backend；GRAM 全量 fresh 迁移；56,421 训练转移 | Native 弱正但不确定；GRAM-LATTE 明显低于 GRAM 与同 SID 对照 | 当前完整迁移有较充分负证据，降低继续投入优先级 |
| SETRec，FP3 | 五维连续 identifier、CF / semantic tokenizer、query、attention、全目录 grounding；四臂均 30 epoch / 3,330 步 | GRAM-SETRec 对 ordered control 弱正，对强 GRAM 大幅负向 | 不再以超过较弱新对照作为提点成功；当前完整路径暂不重启 |
| 其余 P1 / DIGER | 部分只有卡片、CPU 契约或静态审查 | LS-FiD / MHM、GraphMAE / DCRec、SPRINT 等未进入 S4 正式屏；DIGER 为静态候选 | 这些不能计为完成有效性实验后的负结果；也不能据“没充分试过”就直接排第一 |

本地证据：

- [S2 七方向探针](../第十七阶段/Stage17_S2_P0七方向机制探针汇总报告.md)
- [S3 正式筛选](../第十七阶段/Stage17_S3_P0独立正式筛选报告.md)
- [S4 原报告](../第十七阶段/Stage17_S4_P1定向迁移筛选报告.md)（配对指标标注问题见下节）
- [S17-2R 大改筛选](../第十七阶段/Stage17_S2R_架构级候选重选与大改筛选报告.md)
- [Native LATTE](../第十七阶段/Stage17_FP1_FullLATTE_NativeParity报告.md)
- [GRAM-LATTE Full](../第十七阶段/Stage17_FP2_GRAM_LATTE_Full正式结果报告.md)
- [SETRec Full](../第十七阶段/Stage17_FP3_FullSETRec正式结果报告.md)

## 3. 两个完整迁移的关键教训

同一 FP external D0 的 12,833 用户上：

| 模型 | NDCG@10 |
| --- | ---: |
| GRAM-B0-Fresh | 0.061700 |
| GRAM-PSID-Full | 0.028915 |
| GRAM-LATTE-Full | 0.025044 |
| SETRec ordered control | 0.014518 |
| GRAM-SETRec-Paper-Full | 0.015663 |

**LATTE：** 大幅下降已经发生在替换 identifier / linking 的 G1 上；latent forest 在此基础上进一步下降。这支持“替换原输入—输出联结损失了重要能力”的工作解释，但不是把唯一原因单独识别出来。G0/G1/G2 分别完成 35/50/40 epoch；因此不能用“只续训了一个 epoch”解释这组失败。新大改可以换 ID，但不能把冻结 semantic code 当作天然比词法 ID 更好的起点。

**SETRec：** S2 相对自身 ordered control 的 +0.001145 不足以弥补相对强 GRAM 的 -0.046037。完整集合恢复率 0.000468 提示学习 / grounding 仍弱，但这一指标不是唯一原因的证明。四臂都完成 30 epoch，S2 best epoch 22；部分对照 best 接近预算末端，不能进一步宣称所有模型都充分收敛、整个集合生成范式被否定。

**共同教训：** 机制被激活、路径合法、速度更快、超过一个较弱对照，均不等于给原模型提点。探索阶段尽早使用已有强基线判断绝对差距；统一训练与完整消融留到候选有希望、方法确定之后。

## 4. S17-4 的一个已确认指标问题

本轮发现旧 TSV 的表头只有四个指标，而实际每行包含十二个指标。`s4_targeted_p1_runtime.py::prediction_rows` 用表头 `DictReader` 读取，使 H@3 被标作 H@10、H@10 被标作 NDCG@10。因此旧配对差值 / CI 不对应其标签；报告中的日志总体均值本身可以复现。

只读取四个已保存 prediction 文件，从末尾 gold / 排名字段重新计算。四个 SHA 与历史 summary 一致，每臂 12,833 用户完全对齐，逐行十二项指标与排名一致，所有总体均值与日志 / 文件 footer 的最大差异小于 1.5e-15。使用固定 seed 2023、2,000 次用户配对 bootstrap，得到：

| Treatment vs GRAM-Continue | 正确 ΔNDCG@10 | 95% CI | 正确 ΔHit@10 |
| --- | ---: | --- | ---: |
| PAWA-lite | +0.000489 | [-0.000577, +0.001560] | +0.000234 |
| LATTE-sethead | -0.013671 | [-0.016037, -0.011038] | -0.027118 |
| BiFlow-s2g | -0.000273 | [-0.001384, +0.000870] | -0.000156 |

校正后仍未产生稳健 winner；这是有限的报表纠正，不构成重启 PAWA / PCPS 的新理由。旧报告与原数据保留，未来引用配对值应使用本补充。

- [CPU 重算脚本](../../experiment/phase18/analysis/s17_s4_saved_prediction_review.py)
- [重算结果与文件哈希](../../artifacts/phase18/s17_failure_review/s4_saved_prediction_review.json)

## 5. 放开大改后，候选如何重新排序

不按“改动越小越优先”或“改动越大越新颖”排序。接下来对最多三条路线，比较来源机制、旧失败差异、绝对提点可能性和可运行成本，再选一条实现。

1. **完整的协同—语义交互结构：保留在候选池。** 重点看真实的注意力 / 表示计算变化。DIFF / DIF-SR 可作为来源候选，但若迁移后仍只是旧 CF0 / HI-GRAM / BiFlow 的池化注入，就淘汰该映射。是否保留原 FiD 由结构需要决定。
2. **完整非自回归 / 扩散解码：允许重新评估，但不是直接重跑。** 旧 DiffGRM 的本地 hidden=256、3k 用户、5 epoch、不同 tokenizer，不能代表官方完整路径。当前作者仓库提供 Toys / Beauty 命令；Toys 例子 hidden=1024、batch=1024、OPQ / SentenceT5，起始评估 epoch=10。来源设置与旧试验差异很大，必须保留核心机制后再判断投入。[官方仓库](https://github.com/liuzhao09/DiffGRM)
3. **推荐任务与 identifier 联合学习：可考察，尚未充分试过。** 它与 LATTE 的冻结 SID 替换不是同一假设。DIGER 是其中一个来源候选，但作者仍提示历史 `main` 的 recommendation gradient 被意外截断，建议新实验使用 `gradient-fix`；不能用旧论文表格证明修正路径已经有效。[作者说明](https://github.com/junchen-fu/DIGER)

DIFF（属性融合）和 DiffGRM（扩散生成）是不同模型。以上是待比较的路线，不是“三个方向一定有效”的承诺，也不要求三个全部训练。

HSTU 暂留备选：仅靠长序列 / 大规模的优势不足以支持本地优先级。已有 FP D0 中 7,610 / 12,833 用户历史长度不超过 3；该比例只描述这个 fold，不能冒充全项目分布。

## 6. 当前执行规则

- 大改允许：encoder、decoder、identifier、交互结构、训练目标与从头训练均可纳入选型。相同任务、数据边界与评价定义仍需可比较。
- 当前复用第一、第二阶段 GRAM 和已有 PCRF 结果，训练预算主要用于新候选。方法确定后再统一重跑。
- 合理缩放可用于显存 / 接口检查；如果改变论文核心机制，就明确称为新迁移假设。不能把它的失败归到完整论文名下。
- 小样本 smoke 负责检查能否训练；候选效果筛选应获得与其结构相适应的训练预算。初始弱正允许有限适配，不强制在一次小屏中完成最终统计确认。
- 在开发集选择模型；独立确认与测试的既有边界保留，历史读取不能伪装成新的确认。
- PCRF 组合使用新候选自己的 item 输出与分数。新模型输出形式变更时明确分数接口，不能默认把任意分数硬套进 PCRF 后仍等价于旧算法。
- 已有完整负证据的 LATTE / SETRec 当前映射，以及当前 PCPS，继续降低优先级；若重开必须有与旧失败实质不同的机制依据。

本轮未训练模型、未重跑基线。新增的是这一份复盘、一个 CPU 重算脚本及其结果；现行目标计划同步放开大改范围。
