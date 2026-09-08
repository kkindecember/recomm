# Stage18 S18-2 失败后机制复核与文献研究

## Material Passport

- 日期：2026-09-08；研究者授权：“那你继续研究吧”。
- Origin Skill：academic-research-suite；定向研究与既有产物复核，角色分析在同一会话内执行。
- Verification Status：`ANALYZED`；CPU 缓存分析已执行，未复现训练，未执行新 GPU 诊断。
- Scientific State：原 S18-2 `FAILED_SCIENTIFIC_GATE` 保留。本报告是看过结果后的探索性分析，不是预注册成功检验。
- 数据：原 S18-2 Toys I0 四臂及 parent 缓存、两域已存在训练缓存、S18-1R 已消费诊断、冻结 lexical catalog。
- I1/I2、D1/D2、official validation/test、Sports 未读取；模型参数未载入或更新。
- 产物：[主审计 JSON](../../artifacts/phase18/postmortem_research/run-0001/summary.json)、[parent 对照 JSON](../../artifacts/phase18/postmortem_research/run-0001/parent_comparison.json)。

## 1. 结论与需要修正的解释

S18-2 的门槛失败真实存在，逐用户重新汇总可以重建原指标。继续研究发现两项比“调大辅助损失”更值得优先处理的问题：

1. 原 `actual_pruner` 指标只识别最终 beam50 内的同父前缀 sibling items，并不记录剪枝当步的全部竞争前缀、累计分数或第 50 名边界。高覆盖率证明了对这个代理集合的覆盖，不能直接解释为覆盖了真实剪枝边界。
2. 四个继续训练分支均低于同一批用户上的 frozen parent。PCPS 相对 C0 的 +0.005 Hit@50，没有恢复 parent 原有的覆盖率。

前次“没有发现足以推翻结论的实现错误”仍适用于程序执行与已冻结合同；本轮进一步发现的是测量定义与机制解释之间的缺口，以及原简报没有展示的 parent 退化。它们不支持把旧 Gate 改判为通过，也不证明修正测量后新方法必然有效。

## 2. 新发现一：继续训练分支共同退化

下表全部来自同一 1,000 名 Toys I0 用户，parent 的候选已保存在 evaluation anchors 中，本轮未重新生成候选。

| Arm | Hit@50 | Hit@10 | NDCG@10 | Prefix survival |
|---|---:|---:|---:|---:|
| Frozen parent | 0.166 | 0.089 | 0.051310 | 0.430600 |
| C0 continuation | 0.128 | 0.065 | 0.041045 | 0.405533 |
| A0 generic | 0.131 | 0.066 | 0.043918 | 0.403933 |
| M0 PCPS | 0.133 | 0.064 | 0.042276 | 0.405533 |
| S0 shuffled CF | 0.128 | 0.064 | 0.042834 | 0.404633 |

C0 相对 parent：新增命中 16 人、丢失 54 人，净减 38 人；M0：新增 16 人、丢失 49 人，净减 33 人。M0−C0 的比较仍是合法的 matched comparison，但它只能描述两个继续训练分支的相对差异，不能称为相对原基座的实际改善。

本地代码揭示了训练条件的改变：parent 使用全部 eligible 用户的可见历史 transition，训练 10 epochs，有效 batch 16×8=128；S18-2 每域只选 1,000 用户，每用户最后一个可见 transition，重新初始化 AdamW/学习率调度，有效 batch 16。它们是执行补遗预先规定的选择，不是本轮发现的未授权改动。数据分布变窄、优化器重置和 batch 改变均是退化的候选解释；现有实验没有分离它们，不能选定一个为已证实原因，更不能在旧 I0 结果上扫学习率修复。

来源：[parent 训练](../../experiment/phase18/protocol/s18_s1_runtime.py)、[原训练样本构造](../../GRAM/src/data/multi_task_dataset_gram.py)、[S18-2 训练](../../experiment/phase18/protocol/s18_s2_mechanism_probe.py)。

## 3. 新发现二：局部 sibling 排名与全局 beam 保留不是同一个条件

原 `actual_pruner_items()` 的输入是最终返回 item paths；它保留共享 `target[:drop_depth-1]` 且下一 token 不同的 item。它没有输入剪枝当步的分数。`mine_prefix_nodes()` 同样要求与目标共享父前缀；`prefix_loss()` 比较这些 token 的单步 logit。完整路径损失则比较最终长度归一化分数。两者都没有直接使用中间 beam 的全局保留边界。

本机 pinned Transformers 的 beam search 先做词表 log-softmax、Trie mask，再加各父 beam 的累计分数；随后跨 beam 展平排序，由 BeamSearchScorer 处理 EOS/active beams。分支之间的累计分数会影响下一步保留，不能由一个父节点内的 token rank 代替。

一个可验证反例，beam width=2：

| 父前缀概率 | child | 条件概率 | 累计概率 |
|---|---|---:|---:|
| A：0.4 | 目标 | 0.51 | 0.204 |
| A：0.4 | sibling | 0.49 | 0.196 |
| B：0.6 | child 1 | 0.50 | 0.300 |
| B：0.6 | child 2 | 0.50 | 0.300 |

目标已经是 A 下局部第 1 名，仍被 B 的两个 child 挤出 beam。这个反例证明两个条件不等价；它本身不是对本地失败原因比例的测量。

基于冻结 catalog 和已保存 survival 的回顾性结构审计：

| 既有样本 | 非 EOS 前缀丢失用户 | 丢失处合法 child 数 ≤50 | 占比 |
|---|---:|---:|---:|
| Toys 训练缓存，1,000 用户 | 720 | 609 | 84.58% |
| Toys I0 evaluation anchors，1,000 用户 | 828 | 691 | 83.45% |
| Beauty 训练缓存，仅已完成的 100 用户 | 61 | 11 | 18.03% |

在这些 child 数不超过 50 的节点，合法 sibling 内的 top50 必要条件自动满足，不能辨别实际 beam 是否会丢失目标。PCPS 的 soft loss 仍可能影响绝对分数，不能由此断言它没有作用。Beauty 与 Toys 的比例明显不同，也不能把 Toys 的解释推广成双域统一结论。

这里的 first non-EOS drop 利用“目标前缀一旦退出 active beam 就不能在更深处重新出现”的性质，从保存的 survival 和路径长度恢复；另列 Toys 6 个、Beauty 训练样本 1 个“全部非 EOS 前缀存活但最终 miss”，没有把 EOS/finalization 混进上表。缓存不含 live frontier，无法计算真正的跨父分支挤出比例或最近边界竞争者。

来源：[代理集合定义](../../experiment/phase18/core/s1_contracts.py)、[prefix mining/loss](../../experiment/phase18/core/pcps_loss.py)、[可复算审计](../../experiment/phase18/analysis/s18_postmortem_research.py)。

## 4. CF 的作用确实进入了实现，但没有足够的机制收益

Toys 的 3,490 个训练 loss 节点中，M0 与 A0 的 negative child 集合只有 1,098 个完全相同；M0 与 S0 只有 919 个相同。因此不能解释为“CF 根本没接上”。

完整路径负例则按照原合同固定为同一批 parent top8 wrong paths。CF 只改变归一化权重：M0 与 A0 的平均权重 total variation 为 0.01827，与 S0 为 0.05181。这是权重分布差异，不是有效参数梯度的差异；目前不能断言 CF 梯度太小或者没有信息。

额外数值检查发现，缓存中同一 parent wrong path 的 teacher-forced 分数与 generation 分数存在小残差：Toys 训练最大约 0.000801，Toys evaluation 最大约 0.001015，Beauty 训练最大约 0.000402。其来源尚未定位，不能宣称 exact identity，也不能仅凭此认定它造成了失败。后续边界诊断应记录原生生成分数，并单独审计 teacher-forcing 残差，不能直接用后者代替近边界排序。

## 5. 定向文献研究与新意边界

检索日期：2026-09-08。围绕 generative recommendation/retrieval、beam-search optimization、prefix survival、global prefix ranking 检索，重点 2025–2026，并回溯直接相关的 BSO。采用 arXiv 作者全文、ACL Anthology 和作者官方代码库核验。检索为定向扫描，不声称系统综述完整性；没有调用外部模型或上传本地材料。来源与版本记录见 [literature_sources.json](../../artifacts/phase18/postmortem_research/run-0001/literature_sources.json)。

| 工作与核验范围 | 对本任务的直接含义 | 本轮决定 |
|---|---|---|
| Wiseman & Rush，BSO，EMNLP 2016，全文 §4 | 已有对 gold prefix 与 beam 边界候选进行训练比较的思路，也讨论 constrained search | 真实 beam 边界优化属于已有方法基础，不能作为首次提出的贡献。[原文](https://aclanthology.org/D16-1137.pdf) |
| Yang et al.，BEAR，SIGIR 2026，v3 §3 | 区分完整 beam 保留条件与便宜的局部 token 必要条件；直接模拟搜索有成本 | 必须先检查局部条件在 GRAM Trie 上是否具有区分力。[原文](https://arxiv.org/html/2601.22925v3) |
| Yu et al.，APAO，KDD 2026，v4 §4 | 已有累计前缀的 pointwise/pairwise 目标及最弱前缀自适应加权 | 仅换成累计 prefix loss 或最弱层加权不足以构成新方法；pairwise 版本应作为相关对照。[原文](https://arxiv.org/html/2603.02730v4) |
| Wu et al.，CARD，SIGIR 2025，全文摘要与分析/限制节 | 从约束和边际分布分析 beam retrieval 的局限，有明确理论假设 | 支持检查生成目标与搜索行为的差距，不提供对本地失败的因果证明。[原文](https://arxiv.org/html/2504.09935v1) |
| Li et al.，TAAL，2026-08-29 预印本，v1 §3–4 | 使用历史 transition 的联合早期前缀分布监督；PMI 是最终候选重排 | 只作为“监督信号来源”的备选。其 SID/TIGER、单 seed 和验证选 alpha 与本项目不同，效果不能直接迁移。[原文](https://arxiv.org/html/2608.29179v1) |
| Lee et al.，GRAM，ACL 2025，本轮核验官方摘要/元数据 | lexical identifier 和 multi-granular late fusion 是本地基座边界 | 保持原基座；本轮不宣称重新核验全部原论文实验。[官方页](https://aclanthology.org/2025.acl-long.1596/) |
| Chen et al.，PRO，2026 预印本，v3 摘要筛查 | 同时涉及 quantized ID、prefix distillation、vocabulary schedule、搜索分数融合 | 排除整套迁移；与保留 lexical ID 和固定 inference 的边界不符。[作者摘要](https://arxiv.org/abs/2606.09241v3) |

以上只支持“下一步需要哪些测量与对照”，没有证明在 GRAM 上的 efficacy。尤其 TAAL 的正结果与本地 PCPS 失败并不冲突：监督来源、identifier、训练预算、选参协议均不同。

## 6. 推荐的下一步：先做不更新模型的两类诊断

具体执行设计见[真实 Beam 边界与继续训练退化诊断计划](../../plan/第十八阶段/GRAM_第十八阶段_后续研究_真实Beam边界诊断计划v0.1.md)。

- 第一优先：在 frozen parent 上记录完整的剪枝过程，验证局部 rank、真实全局边界与旧 final-sibling proxy 的差异。
- 第二优先：检查训练样本分布和优化设置相对 parent 的改变，并用只读梯度方向诊断区分 CE 更新方向与 PCPS 增量；不根据旧 I0 NDCG 选择新的学习率或 epoch。
- 若真实边界约束与现有 PCPS/通用对照无法形成可测的区别，结束这条机制重设计；若只发现 continuation 损伤，也不能把“恢复 parent”写成新方法收益。
- 只有测量可信、机制不同且有解释力后，再另立训练计划。新计划需要 frozen parent、matched CE、通用机制、真实 CF、shuffled CF 五个参照；保留 PCRF 作为最终评价锚点。

## 7. 反证与限制检查

三次 inline 反证检查结论：研究问题可检验；“错配已证明是根因”和“frontier loss 本身具有新意”两项过强解释均撤回；最终只保留诊断优先级。

最强替代解释是：预注册的小样本 continuation 本身破坏了 parent，掩盖了辅助损失的潜在作用。另一个替代解释是：CF 本身不足以指导更好的生成。现有数据无法在二者和目标错配之间给出唯一因果裁决。

11/11 类统计误读检查覆盖：Simpson/生态推断/选择偏差/collider/基率/回归均值/幸存者偏差/多指标挑选/分析自由度/相关即因果/反向因果。适用项处理：保留四臂及 parent、明确 1,000/100 不同分母、区分训练与评估、标注回顾性，不给探索性分析套确认性 CI/p 值，不把单域结论推广为双域成功。无适用数据的类型标为不适用，不声称排除所有偏差。

本轮新增两份 CPU 分析脚本，均运行成功并保存输入 SHA 与分析源码 SHA；新分析中的断言覆盖用户配对、标签一致性、合法负例、目录唯一性和汇总重建。此前完成的 25 项 S18 合同测试不是新 GPU 诊断的验收结果。
