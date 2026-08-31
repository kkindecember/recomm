# GRAM 第十七阶段：顶会优秀机制迁移、效果搜索与可叠加方向开发计划 v0.5

> 文件名为兼容既有索引保留 `CLaF-GRAM...v0.1.md`；本文件正文已升级为 v0.5，CLaF 只是候选轨道之一，不再是第十七阶段的唯一主线。

## Material Passport

- Origin Skill：`academic-research-suite / deep-research + experiment-agent`
- Origin Mode：research → repository audit → transferable-mechanism portfolio → experiment plan
- Created：2026-08-28
- Revised：2026-08-30
- Verification Status：`S17-0_TO_S17-4_COMPLETED / S17-2R_AUTHORIZED / S17-5_HOLD`
- Version Label：`phase17_gram_architecture_reselection_v0.5`
- Scope：标准 GRAM sequential recommendation；**不是 cold50、cold50ft 或伪冷启动实验**
- Development Domains：Beauty、Toys 的 train-prefix rolling-origin folds
- Confirmation Domain：Sports，继续封存；只有方法冻结且研究者单独授权后才可读取
- Core Objective：把顶会/强相关工作的优秀机制迁移到 GRAM，寻找能提高效果、可继续优化、可正交叠加的方向
- Non-objective：不要求 1:1 复现论文，不以当前创新性或立即写论文为阶段约束
- Authorization：研究者以“开始/继续某一步”授权该步实现、短 smoke 与计划内实验；**不自动授权读取 Sports 或重读 Beauty/Toys 官方 test 调参**
- Resource Policy：Stage17 **不设固定 GPU 数量硬上限**。小实验使用当时空闲且满足显存准入的卡；大实验根据独立 arm 数、预计时长和并行收益提出所需 GPU 数量及每卡显存，得到研究者分配后启动。当前通常只能规划 1–2 张卡、每个单卡 job 约 30 GiB，但这是可用性预估，不是永久上限；若 3–4 张卡确有并行价值，可以如实申请，资源不足时再降级为分波串行。

## 0.0 2026-08-30 路线修订：先做 S17-2R

S17-2～S17-4 已证明局部 hook、lite loss、确定性 root 和轻量 rerank 尚未形成稳健增益。研究者现已授权架构级大改，因此旧顺序中的 S17-5 暂停，先执行 `S17-2R` 架构级候选重选。允许重做 identifier/tokenizer、decoder、training objective、candidate-generation/ranking architecture 和 backbone；继续严格保护 shadow-fold、no-future-read、统一 item evaluator、official test/Sports 封存与 GPU1 重复轮。

S17-2R 的详细候选、native control、准入门槛和资源策略以 `plan/第十七阶段/GRAM_第十七阶段_S17-2R架构级候选重选与大改实验计划v0.1.md` 为准；该修订不追溯改变 S17-0～S17-4 的实验事实。

## 0.0.1 2026-08-31 路线修订：S17-FP 完整论文机制迁移

S17-2R 已终态收口为 `COMPLETED_NO_R3_CANDIDATE`。研究者进一步明确：Stage17 只负责正常 GRAM 推荐提点，后续不再用轻量/缩放 proxy 代替论文大机制，允许 full-data、完整训练预算和多卡并行的大实验。新的权威执行计划为：

`plan/第十七阶段/GRAM_第十七阶段_S17-FP完整论文机制迁移与架构级大实验计划v0.1.md`

该计划以 Full LATTE native parity、`GRAM-LATTE-Full`、Full SETRec 和条件式架构融合为主线，并覆盖本文中过时的 lite-first 后续顺序。资源规则同步修订：大于 10 分钟一律后台、用户通过 `artifacts/phase17/status/` 观察且无需 agent 实时监看；小实验自行选择安全空闲的非 GPU1 卡；大实验和任何已占用 GPU handoff 必须逐次申请；GPU1 默认不释放，获批使用后任意科学终态都必须恢复交接前重复轮。只有 GPU1 具有强制重复轮恢复要求，其他卡在科学终态后正常释放。

## 0. 阶段定位：不是复现赛，而是 GRAM 的机制迁移与效果搜索

第十七阶段的唯一硬目标是：

> **广泛吸收顶会与高相关工作的有效机制，在 GRAM 上做低成本、可归因的迁移实验；先找到真实增益和可继续优化的方向，再考虑如何组合、包装创新或写论文。**

因此本阶段采用以下原则：

1. **效果优先**：评价一个方向首先看 GRAM 上的 NDCG/Hit、稳定性、开销和失败边界，不看是否完整复现原论文数字。
2. **机制迁移，不做 1:1 复现**：允许只迁移一个 loss、一个 attention pattern、一种辅助视图、一种训练课程或一种解码约束。
3. **允许为 GRAM 改造**：可以替换论文里的 backbone、tokenizer、任务头、负采样或数据接口；必须写清“借鉴了什么、改了什么、为什么适合 GRAM”。
4. **多找、多试、分轨推进**：一个方向失败只关闭该轨道，不终止第十七阶段，也不能因为 Latte/CLaF 失败就停止搜索。
5. **先单模块，后叠加**：每个模块先独立和 GRAM 对比；只有独立正增益或明确互补信号的模块才进入二元、三元组合。
6. **不为论文故事限制实验**：本阶段可以保留多个并列有效模块，不强行统一成一个“新方法”。
7. **可追溯**：所有尝试都进 attempt ledger；每个计划步骤完成时只写一份汇总报告，不为步骤内每次试错单独写报告。
8. **探索期复用历史 GRAM**：第一、二阶段已完成且 checkpoint 仍在的 Beauty/Toys GRAM 不重复做 30-epoch 基线训练；先把它们作为探索初始化和历史参照。只有胜出方向进入独立确认或论文准备时，才重跑 fresh fold-specific baseline 与多 seed 对照。

### 0.1 “迁移成功”的定义

论文原模型是否被完整复刻，不是本阶段的通过条件。对候选机制 `m`，本阶段只问：

1. `m` 能否在不破坏 GRAM lexical target、FiD 输入和合法 Trie 的条件下接入；
2. 相同数据、主要训练预算和 beam 条件下，是否改善 GRAM 排序效果或关键机制指标；
3. 改善是否能在独立 fold、另一数据域或另一 seed 上重复；
4. 它与已有正模块是否作用于不同瓶颈，组合后能否超过最强单模块；
5. 性能增益是否值得显存、训练时长和推理时延。

本文件中出现的 `*-GRAM`、`*-lite`、`*-inspired` 都表示 **GRAM 适配实现**，不得在报告中写成 faithful reproduction，除非另有逐项 fidelity contract。

## 1. GRAM 可迁移接口与已有证据

### 1.1 当前代码提供的接口

仓库静态审计确认：

- `GRAM/src/data/multi_task_dataset_gram.py` 同时产出 history、逐历史商品文本、`history_item_ids` 和 `target_item_id`；
- `GRAM/src/utils/indexing.py` 可选择 title、brand、category、description 等 metadata，并生成 native lexical ID；
- `GRAM/src/processor/Collator.py` 形成 `B × (N+1) × L` 的多 passage 输入，同时保留协同 item id；
- `GRAM/src/model/gram.py` 使用 FiD-style encoder wrapper，可在 encoder 表示、辅助 loss、decoder loss 和 generation 之间插入模块；
- `GRAM/src/utils/generation_trie.py` 与生成接口允许实现 prefix/beam 约束和 item-level 聚合。

这意味着可迁移方向不限于 tokenizer：至少可以从 **输入/历史表示、全局协同视图、encoder 交互、训练目标、identifier/path、decoder/beam、校准** 七个位置改进 GRAM。

### 1.2 历史结果给出的边界

- 第九阶段 PCRF 在 Toys/Beauty 上稳定提高 top-10 排序，但没有增加 Hit@50，说明已有 beam 内仍存在排序空间；
- 第四阶段观察到 lexical tree close pairs 的预测相关性异常高，说明路径耦合问题真实存在，但旧 decorrelation loss 没修好；
- 第十至十一阶段增加候选和后验 gate 有 oracle headroom，却难以稳定把 target 排进前列；
- 第十二阶段 HI-GRAM 有弱信号但状态与 test-read 记录矛盾，只能视为 `HISTORICAL_UNVERIFIED_SIGNAL`；
- 第十三至十六阶段属于 cold 方向，只能提供工程和完整性经验，收益不得混入本阶段 normal-setting 结论。

因此 Phase 17 不只押注“路径森林”，还要同时试：训练—解码错配、历史兴趣噪声、顺序与非顺序交互、全局转移图、输出路径结构、校准与流行度偏置。

## 2. 文献检索与候选纳入规则（截至 2026-08-29）

### 2.1 检索策略

检索范围以 2022–2026 年 ACL/EMNLP、SIGIR、KDD、WWW、WSDM、CIKM、ICML、NeurIPS 的论文页、arXiv 原文与作者/官方 GitHub 为主，关键词覆盖：

- generative recommendation / generative retrieval；
- semantic identifier / lexical identifier / multi-view identifier；
- beam-search-aware training / prefix-aware decoder；
- sequential and non-sequential feature interaction / bidirectional information flow；
- global item transition / graph-enhanced sequential recommendation；
- denoising / long-short interest / popularity bias / uncertainty。

纳入优先级：

1. 机制能被隔离，并可合理落在 GRAM 的现有接口；
2. 顶会已发表优先，高相关的新近预印本也可进入想法池；
3. 有官方代码优先，且单独核验 license、commit 和关键实现；
4. 无代码不等于排除，只要机制足够清晰且能从论文独立实现；
5. 已在本仓库反复失败的 candidate union/gate 只有出现本质不同机制时才重开。

限制：这是一轮目标导向检索，不是 PRISMA 系统综述；2026 年新论文的接收、代码和版本可能变化。每个方法正式实现前必须重新确认论文版本、仓库 commit 与许可。**公开源码不自动等于允许复制源码**；无明确许可时只能阅读和独立实现思想。

### 2.2 证据等级

| 等级 | 含义 | 使用方式 |
|---|---|---|
| A | 顶会正式论文 + 官方代码 + 许可清晰 | 可读论文和代码，按许可借鉴实现 |
| B | 顶会正式论文 + 官方代码，但许可不清晰/代码不完整 | 重点借鉴机制；默认独立实现 |
| C | 顶会正式论文但无可用代码，或高相关近期预印本 + 代码 | 做低成本机制 probe，结论保持克制 |
| D | 近期预印本/第三方实现/概念相关 | idea backlog；不能用第三方结果替代本地证据 |

## 3. 可迁移方法池

### 3.1 Track A：训练目标与 beam/search 错配

| 方法来源 | 可借鉴机制 | GRAM 迁移版本 | 代码状态 | 优先级 |
|---|---|---|---|---:|
| [BEAR](https://arxiv.org/abs/2601.22925) | 正样本 target token 在每个解码步都应存活于 beam top-B，训练时显式惩罚早期被剪枝 | `BEAR-GRAM`：在原 teacher-forcing CE 外加 prefix survival/rank margin regularizer；不改 lexical ID | [官方代码，MIT](https://github.com/Tiny-Snow/BEAR-SIGIR-2026) | **P0** |
| [GenRet](https://proceedings.neurips.cc/paper_files/paper/2023/hash/91228b942a4528cdae031c1b68b127e8-Abstract-Conference.html) | progressive training 逐步学习自回归 identifier 各位置 | `PrefixCurr-GRAM`：先训练短前缀/浅层 lexical 路径，再逐步开放完整 suffix；不重学 tokenizer | [作者代码](https://github.com/sunnweiwei/GenRet)，根目录许可需再核验 | **P0** |
| [NCI](https://arxiv.org/abs/2206.02743) | Prefix-Aware Weight-Adaptive decoder、前缀一致性 | `PAWA-lite-GRAM`：按 lexical depth/prefix 选择轻量 adapter 或 output bias；先做两档深度 | 论文机制清晰，正式实现前再核验官方代码 | P1 |
| [SEATER](https://arxiv.org/abs/2309.13375) | tree-structured identifier 上的层次对比与 ranking loss | `TreeCL-GRAM`：按 lexical prefix overlap 构造正负层次关系，辅助原 CE | [作者代码](https://github.com/ethan00si/seater_generative_retrieval)，许可待核验 | P1 |

第一优先做 BEAR-GRAM，因为它直接作用于 GRAM 已存在的自回归 beam，改动小、归因清楚，且不依赖重训 identifier。

### 3.2 Track B：identifier、路径与 item-level 生成

| 方法来源 | 可借鉴机制 | GRAM 迁移版本 | 代码状态 | 优先级 |
|---|---|---|---|---:|
| [Latte](https://arxiv.org/abs/2605.06331) | latent root、多树路径、同 item 多路径聚合 | `Latte-GRAM` 与 `CLaF-GRAM`：保留原 lexical suffix，只加随机或上下文 root | [官方代码，MIT](https://github.com/hyp1231/Latte) | P0/P1 |
| [Pctx](https://arxiv.org/abs/2510.21276) | 同一 item 随用户 context 获得不同 tokenization | `Pctx-Root-GRAM`：只借鉴 context-dependent route，不复制完整 RQ-VAE/ensemble | [官方代码，MIT](https://github.com/YoungZ365/Pctx) | P1 |
| [MINDER](https://aclanthology.org/2023.acl-long.366/) | title、substring、synthetic query 等 multi-view identifiers，并在 item 层融合 | `MVI-GRAM`：为同一商品构造 2–3 个 native lexical view，生成后按 item 聚合 | [官方代码](https://github.com/liyongqi67/MINDER)，许可需再核验 | **P0** |
| [SETRec](https://arxiv.org/abs/2502.10833) | 把 identifier 看成 order-agnostic token set，减少顺序依赖与 beam 局部错误 | `SetHead-GRAM`：先加并行 token-set 辅助头/重排分数，再决定是否替换 AR 解码 | [官方代码](https://github.com/Linxyhaha/SETRec)，未见标准 license 时只独立实现 | P1 |
| [ActionPiece](https://proceedings.mlr.press/v267/hou25f.html) | 依据 action 内与相邻 action 共现合并 feature patterns | `ActionPiece-lite-GRAM`：仅在 lexical path/历史 token grouping 上做离线合并对照 | [官方代码，Apache-2.0](https://github.com/google-deepmind/action_piece) | P2 |
| [LMIndexer](https://proceedings.mlr.press/v235/jin24h.html) | reconstruction、contrastive learning 与 progressive semantic ID | 只迁移 lexical prefix 的 reconstruction/contrastive 辅助目标，不替换 native ID | [作者代码](https://github.com/PeterGriffinJin/LMIndexer)，许可待核验 | P2 |
| [VaLiDRec](https://arxiv.org/abs/2607.25209) | variable-length、LLM-aligned ID、graph-aware soft prompt、token-set scoring | GRAM 已有 variable-length lexical ID，优先借鉴 graph soft prompt 与 item-level token-set score | 暂未核验到官方代码 | P2 |

Latte/CLaF 不再拥有“失败即结束 Phase 17”的地位。它们只是路径轨；即使全部失败，其他轨道照常推进。

### 3.3 Track C：序列与非序列信息的双向交互

目前检索中，最接近“序列数据与非序列数据进行双向信息学习”描述的是 [OneTrans](https://arxiv.org/abs/2510.26104)：它将 sequential features 与 non-sequential features token 化到同一主干，并让 sequence modeling 与 feature interaction 在层间共同进行，而非单向的“先编码序列、再做特征交互”。

这里必须区分两种“bidirectional”：

- **允许**：历史序列表征与非序列/全局特征表征双向交换；同一已观测 prefix 内的 encoder 双向注意力；
- **禁止**：使用预测 cutoff 之后的未来行为、目标 item 信息或官方 test label。

| 方法来源 | 可借鉴机制 | GRAM 迁移版本 | 代码状态 | 优先级 |
|---|---|---|---|---:|
| [OneTrans](https://arxiv.org/abs/2510.26104) | 统一 tokenization、layer-wise sequence/feature interaction、双向信息流 | `BiFlow-GRAM-lite`：历史 passage bus 与 global/field token bus 做 1–2 层 gated cross-exchange；`BiFlow-GRAM-full` 再尝试逐层融合 | 暂无已核验的官方完整仓库；第三方 PyTorch 端口只能辅助理解，不能当官方复现 | **P0/P2** |
| [UniDot](https://arxiv.org/abs/2608.16797) | sequence/feature 双总线交换与 mutual learning | 只借鉴 two-bus consistency loss 或 exchange block，不追求原工业模型规模 | 新近预印本，暂未核验官方代码 | P2 |
| [BlossomRec](https://doi.org/10.1145/3774904.3792408) | long-term 与 short-term interest 的稀疏选择、门控融合 | `LS-FiD-GRAM`：最近窗口与全历史语义块分两路 FiD，再学习 gate | [作者代码](https://github.com/Applied-Machine-Learning-Lab/WWW2026_BlossomRec)，许可待核验 | P1 |
| BERT4Rec-style masked history | 双向 masked item modeling | `MHM-GRAM`：仅在 train prefix 内随机 mask 历史 passage/lexical item，辅助恢复；正式预测仍因果 cutoff | 机制成熟，可独立实现 | P1 |

`BiFlow-GRAM-lite` 的非序列侧只允许使用推理时真实可得的信息：coarse user prompt、历史商品 metadata 汇总、train-only 频率/图统计和历史 item id 表示。**不得把未知 target 的 title/category/embedding 放入输入。**

### 3.4 Track D：全局转移、图与协同信息注入

| 方法来源 | 可借鉴机制 | GRAM 迁移版本 | 代码状态 | 优先级 |
|---|---|---|---|---:|
| [MQSA-TED](https://arxiv.org/abs/2311.01056) | multi-query self-attention 学协同偏好；Transition-aware Embedding Distillation 注入全局 item transition | `TED-GRAM`：用 fold-train transition teacher 蒸馏 history/item slot；`MQPool-GRAM`：多时间尺度 query pooling | [官方代码](https://github.com/zhuty16/MQSA-TED)，许可待核验 | **P0** |
| [MAERec](https://arxiv.org/abs/2305.04619) | item-item graph masked autoencoder，提炼全局转移结构 | `GraphMAE-Prompt-GRAM`：train-only item 图生成 soft prompt 或辅助 reconstruction target | [官方代码](https://github.com/HKUDS/MAERec)，许可待核验 | P1 |
| [DCRec](https://arxiv.org/abs/2303.11780) | sequence/global graph 双视图与 debiased contrastive learning | `DCRec-CL-GRAM`：对 GRAM history state 和 train-only graph state 做频率感知对比 | [官方代码，MIT](https://github.com/hkuds/dcrec) | P1 |
| [TASTE](https://arxiv.org/abs/2308.14029) | text matching 缓解 popularity bias | `TextMatch-GRAM`：复用 lexical/text 表示构造 item-text matching 辅助 loss | [官方代码](https://github.com/OpenMatch/TASTE)，许可需核验 | P2 |

这一轨道与 OneTrans 类方向可以交叉：图/转移向量可以作为非序列 bus，但必须先分别证明 `TED/Graph` 和 `BiFlow` 的独立价值，不能一开始混在一起。

### 3.5 Track E：兴趣去噪、校准与偏置控制

| 方法来源 | 可借鉴机制 | GRAM 迁移版本 | 代码状态 | 优先级 |
|---|---|---|---|---:|
| [LISRec](https://arxiv.org/abs/2505.22130) | 从历史中抽取稳定的 learned item shortcuts，过滤噪声点击 | `Shortcut-FiD-GRAM`：用现有 text/lexical embedding 选语义连通子集，作为额外 FiD branch；先不做完整预训练 | [官方代码，MIT](https://github.com/NEUIR/LISRec) | **P0** |
| [SPRINT](https://arxiv.org/abs/2606.21911) | attention column-sum 与 FFN spectral regularization，抑制规模化 popularity amplification | `SPRINT-GRAM`：只接入可对应的注意力/FFN 正则，并报告 head/tail | [GenRec 代码库](https://github.com/Tiny-Snow/GenRec)，正式使用前核验对应目录和许可 | P1 |
| [UGR](https://arxiv.org/abs/2602.11719) | confidence token 与 uncertainty-aware optimization | `ConfHead-GRAM`：先做置信度 token/校准辅助目标，不直接上 RL；只有 lite 有信号才考虑完整优化 | [官方代码](https://github.com/cxfann/UGR)，许可待核验 | P2 |
| [MemGen-GR](https://github.com/Jamesding000/MemGen-GR) | 区分 memorization/generalization 样本并做自适应组合 | 先迁移其诊断切分，找出 GRAM 增益来自哪里；本仓库已多次 gate 失败，不先做 ensemble | 官方仓库标注 Apache-2.0，仍需固定 commit | Diagnostic |

### 3.6 已有方法与暂不优先方向

- PCRF 保留为强 incumbent 和最终正交后处理，不作为新迁移方向；
- COBRA/BeamFusion 类稀疏—稠密候选融合与本仓库第 9–11 阶段高度重叠，除非出现新的可验证机制，不列 P0；
- DIGER、完整 RL/GRPO、巨型 LLM、多 rollout 方法成本高且可能替换 lexical ID，先放 P2；
- 多模态/时间特征方法若当前 Amazon 数据缺对应输入，不为了复现而伪造字段；
- HI-GRAM 先完成历史 forensic audit；若重跑，作为独立 architecture baseline，不和新模块默认绑定。

## 4. 首轮优先级与最低试验承诺

### 4.1 P0：必须各做至少一次机制 probe

| ID | 迁移模块 | 预期改动位置 | 首个低成本问题 |
|---|---|---|---|
| A0 | `BEAR-GRAM` | decoder loss | target 是否更少在 lexical 前缀早期掉出 beam？ |
| A1 | `PrefixCurr-GRAM` | training schedule | shallow/deep token accuracy 与 NDCG 是否改善？ |
| B0 | `MVI-GRAM` | target path + item aggregation | 多 lexical view 是否补充合法 item 覆盖？ |
| B1 | `Latte-GRAM-lite` | target root + Trie | 随机多树是否减弱 lexical coupling？ |
| C0 | `BiFlow-GRAM-lite` | FiD encoder | 序列 bus 与 metadata/global bus 双向交换是否优于 concat/单向？ |
| D0 | `TED-GRAM-lite` | train-only transition teacher + auxiliary loss | 全局转移知识是否改善短历史与中频 item？ |
| E0 | `Shortcut-FiD-GRAM` | history selection/extra branch | 去除噪声行为后是否优于 last-k 和全历史？ |

P0 的每个方向至少获得：contract test、100-sample overfit/smoke、一个固定预算 D0 probe。不能因为前一个方向失败而取消后续方向。

### 4.2 P1：至少做静态映射和轻量可行性检查

`PAWA-lite-GRAM`、`TreeCL-GRAM`、`Pctx-Root-GRAM`、`SetHead-GRAM`、`LS-FiD-GRAM`、`MHM-GRAM`、`GraphMAE-Prompt-GRAM`、`DCRec-CL-GRAM`、`SPRINT-GRAM`。

其中满足以下任一条件就升级为正式 D0 screen：

- P0 诊断暴露了它直接针对的瓶颈；
- 它的 smoke 机制指标明显改善且训练稳定；
- 它与当前 P0 winner 作用位置正交、成本可控；
- 论文代码接口与 GRAM 的 T5/FiD/graph 数据高度相容。

### 4.3 P2：重型或不确定方向

`CLaF balanced hard-EM`、`BiFlow-GRAM-full`、完整 `SETRec`、完整 `UGR`、ActionPiece tokenizer、完整 graph pretraining、完整 LISRec pretraining。

P2 不因创新性不足被排除，但必须先有下列之一：lite 版本正信号、明确的失败归因、或研究者批准的大实验资源。

### 4.4 候选池是开放的

Phase 17 期间新发现的顶会 idea 可以追加到 `idea_registry.yaml`。新增项必须写：论文、机制、GRAM 接口、最小 probe、预期成本、代码与许可、与已有尝试的差异；不需要等到计划再次升版本才进入低成本静态审计。

## 5. 统一迁移模板：不追求 1:1，但必须可归因

每个候选方法实现前填写一页 `migration_card`：

```yaml
track_id: C0
source_paper: OneTrans
borrowed_mechanism:
  - layer-wise bidirectional sequence/feature interaction
explicitly_not_reproduced:
  - industrial feature schema
  - original ranking head and training scale
gram_insertion_point:
  - FiD encoder wrapper
unchanged_contracts:
  - native lexical target
  - legal trie and item mapping
  - fold and evaluator
controls:
  - GRAM continuation
  - concat-only
  - one-way fusion
primary_mechanism_metric:
  - seq_to_field_and_field_to_seq_gate_norm
estimated_gpu:
  count: TBD_AFTER_S17_0_PROFILE
```

最低归因对照：

1. `GRAM-Historical`：探索期复用第一/二阶段 checkpoint 与 validation 指标；确认期再切换为 fresh fold-specific `GRAM-B0`；
2. `GRAM-Continue`：从同一历史 checkpoint 多训练 1 epoch，与所有迁移 arm 严格匹配，用于排除“只是多训练”；
3. `Capacity-Control`：尽量匹配新增参数/层/额外 token；
4. `Method-Lite`：只加目标机制；
5. `Shuffled/Blocked-Control`：打乱输入关系或阻断一侧信息流，证明真实机制而非容量；
6. `Method-Full`：只有 lite 有信号或成本合理时运行。

我们不要求还原论文全部 ablation；只要求本地结论能回答“哪一个被迁移的机制在 GRAM 上起作用”。

## 6. 数据协议与防泄漏

### 6.1 Rolling-origin shadow datasets（胜出方向确认协议）

Beauty/Toys 官方 `[-2]` validation 与 `[-1]` test 已在历史阶段暴露，因此不能承担论文级独立确认。胜出方向进入准入、跨 fold 或论文准备时，只允许一次性 projection job 打开原始序列，并按下式生成三个 shadow datasets：

`shadow_seq = train_prefix + [shadow_validation_target] + [guard_item]`

其中 `guard_item` 固定取 `train_prefix[0]`，只用于占据 loader 的 `[-1]` test 槽，禁止评估；下游训练/评估 job 只能打开 shadow 文件，不能再打开原始 monolithic sequence。

| Fold | `train_prefix` | Shadow validation target | Loader 序列 | 用途 |
|---|---|---|---|---|
| D0-discovery | `original[:-5]` | 原序列 `[-5]` | `prefix + [target, guard]` | 机制 probe、成本估计、有限超参选择 |
| D1-admission | `original[:-4]` | 原序列 `[-4]` | `prefix + [target, guard]` | 独立准入，禁止调参 |
| D2-fresh | `original[:-3]` | 原序列 `[-3]` | `prefix + [target, guard]` | 冻结配置、多 seed 确认 |

GRAM loader 对 `shadow_seq` 使用 `[:-2]` 训练、`[-2]` validation，恰好恢复上述 `train_prefix/target`；`[-1]` 只是已观察训练 item。生成脚本必须通过 unit test 证明 shadow target 不进入训练 interaction、官方 `[-2:]` 从未序列化、guard 不参与评估。D0 中没有 target 前历史的用户必须排除并在 manifest 计数。projection job 不打印 heldout item 值，所有下游日志和配置也不得包含这些值。

### 6.2 Normal-setting 定义

- 不人工构造 cold50，不屏蔽 target 商品的已有 transductive 信息；
- 不引入未来行为；“双向信息流”只发生在已观察历史 prefix 内；
- graph、transition、popularity、cluster、shortcut、negative pool 都按 fold train-only 重建；
- overall 为主结果，同时报告 head/mid/tail、history length、memorization/generalization 分层；
- S17-3 广搜允许复用 Beauty/Toys official validation 作为明确标注的 `exploration-only` 选择信号，从而避免先重训 GRAM；不得把它包装成独立确认结果；
- Beauty/Toys official test 不为广搜重跑，也不能用于选择方法、epoch、beam 或权重；
- 胜出方向进入确认时改用 D0/D1/D2 shadow folds、fresh matched baseline 和多 seed，届时不能拿探索期 official-validation 数值代替；
- Sports 只允许做文件存在性与 schema 的 target-free preflight；label、prediction、metric 必须另行授权。

### 6.3 公平性

- 主比较固定 backbone、初始化来源、fold、seed、训练 examples、evaluator 和主要 beam；
- 新方法需要更多训练步时，必须有 matched continuation；
- 改变 beam 的方法同时报告 compute-matched beam 与 method-native beam；
- 每个方法记录参数量、峰值显存、GPU-hours、训练/推理 wall time、合法 unique item 数；
- 外部代码只提供实现提示，不直接复用其处理后的 label、split 或 test artifact。

## 7. 实验漏斗与步骤报告

Phase 17 是“广搜—独立验证—组合”的漏斗，而不是一条单方法闯关线。

### S17-0：证据、历史结果、源码与资源审计

任务：

1. 冻结本文件候选论文的 source manifest：URL、版本/commit、license、关键文件、可借鉴机制；
2. 完成 Phase 12 HI-GRAM forensic audit，修正状态与 test-read 矛盾；
3. 生成 D0/D1/D2 data manifest、SHA256 与 leakage tests；
4. 复核 lexical ID、EOS、variable length、Trie、item aggregation；
5. 在服务器测 GRAM baseline 的 100/1k sample 显存和 wall time，给每个 track 估算 GPU 数量；
6. 建立 `idea_registry.yaml`、`migration_cards/`、status schema 和 attempt ledger schema。

完成报告：`report/第十七阶段/Stage17_S0_证据源码数据与资源审计报告.md`

### S17-1：公共迁移框架、状态系统与 contract tests

任务：

- 统一 feature hook、auxiliary loss hook、decoder loss hook、item aggregation hook；
- 实现 module registry，禁止各 track 复制并漂移 evaluator；
- 建立 legal target、no-future-read、fold isolation、K=1 equivalence、score reconstruction 测试；
- 实现本文件第 10 节的后台/status/占卡重复协议；
- 做 CPU tests 与 100-sample GPU smoke，不比较正式效果。

完成报告：`report/第十七阶段/Stage17_S1_公共迁移框架与运行合约报告.md`

### S17-2：P0 七方向固定预算机制 probe

对 A0/A1/B0/B1/C0/D0/E0 逐一运行：

1. static shape/gradient test；
2. 100-sample overfit；
3. Toys D0 的统一短预算 probe；
4. 每个方向至少一个专属机制指标；
5. 记录失败原因：接口失败、不可学习、有效但太贵、指标无变化或 accuracy 负向。

短 probe 只用于排明显 bug 和极弱方向，不用小样本 NDCG 做论文结论。任何 P0 失败都不取消其他 P0。

完成报告：`report/第十七阶段/Stage17_S2_P0七方向机制探针汇总报告.md`

### S17-3：P0 standalone 正式筛选

根据 S17-2 修正一次实现 bug 后，复用第二阶段 Toys seed 2023 epoch-30 checkpoint，在 official validation 上运行明确标注为探索期的固定预算 standalone：GRAM-Continue、七个 P0 lite 及必要机制 control。所有 arm 从同一 checkpoint、fresh matched optimizer 继续 1 epoch；不先重训 30-epoch GRAM-B0，也不重跑 official test。

历史零额外步参照为第二阶段 validation `Hit@10=0.119411`、`NDCG@10=0.0762745143`；主要公平对照是同样多训练 1 epoch 的 `GRAM-Continue`。本轮只用于快速方向筛选，若方向胜出，再增加训练预算并按 6.1 节在 fresh shadow folds 重跑基线、GRAM-Continue 与方法。

保留进入 D1 的方向满足任一：

- NDCG@10 明确为正且 Hit@10 不出现不可接受下降；
- accuracy 尚弱，但论文对应机制指标明显改善，且存在一个预注册、低成本的二次实现版本；
- 对某个预先定义子群有大增益、overall 不显著受损，并能解释为明确 boundary condition。

每个方向最多允许一次基于机制诊断的修订版；不能无上限换 loss、gate、beam 或特征追结果。

完成报告：`report/第十七阶段/Stage17_S3_P0独立正式筛选报告.md`

### S17-4：P1 定向轻量迁移

无论 P0 是否全胜，都完成 P1 的静态卡片和 smoke；对符合 4.2 升级条件的方向运行 Toys D0 standalone screen。重点是：

- P0 暴露早期 prefix pruning → PAWA/TreeCL；
- 多路径有覆盖但排序差 → SetHead/MVI item scoring；
- 长短历史差异大 → LS-FiD/MHM；
- 短历史或协同稀疏明显 → GraphMAE/DCRec；
- tail 变差或 popularity amplification → SPRINT。

完成报告：`report/第十七阶段/Stage17_S4_P1定向迁移筛选报告.md`

### S17-5：独立 D1 双域准入

将 S17-3/S17-4 的 positive/diagnostically-promising modules 冻结，在 Toys D1 + Beauty D1、seed 2023 上独立运行。

分级：

- `WINNER`：双域 macro NDCG@10 为正，至少一域明确为正，另一域不出现实质负迁移；
- `DOMAIN_SPECIALIST`：一域稳定强正、另一域中性/可解释，后续只在对应边界使用；
- `MECHANISM_ONLY`：机制指标成立但 accuracy 未转化，可保留一次结构优化机会；
- `REJECTED`：独立 fold 负向或增益来自错误/额外预算；
- `INCONCLUSIVE_RESOURCE`：只因资源/数值问题未完成，不假装科学失败。

完成报告：`report/第十七阶段/Stage17_S5_D1双域独立准入报告.md`

### S17-6：正交性分析与二元/三元叠加

先按作用层分类：

- 输入/历史：Shortcut、LS-FiD、MHM；
- encoder/全局视图：BiFlow、TED、GraphMAE/DCRec；
- identifier/path：MVI、Latte/CLaF、SetHead；
- decoder/train-search：BEAR、PrefixCurr、PAWA/TreeCL；
- 后处理/校准：PCRF、SPRINT/ConfHead。

叠加规则：

1. 只组合 `WINNER`，或一个 WINNER + 一个有强机制证据的正交模块；
2. 优先不同作用层的二元组合；同层方法先做替代比较；
3. 组合必须同时跑两个 parent，判断是否超过 `max(parent_A, parent_B)`；
4. 若二元组合不能超过最强 parent，不进入三元；
5. 三元组合最多包含三个经过独立验证的模块；
6. 不允许用临时 gate 把失败模块藏起来；
7. PCRF 只在 standalone winner 冻结后统一应用，不能救活失败主模块。

完成报告：`report/第十七阶段/Stage17_S6_正交性与模块叠加报告.md`

### S17-7：D2 多 seed 稳健性确认

在 Beauty/Toys D2 跑冻结的：

- GRAM-B0 / matched continuation；
- 最强 2–4 个 standalone modules；
- 最强 1–2 个组合；
- seeds `2023/2024/2025`。

此处不是为投稿设硬阈值，而是决定后续研发资源：平均增益、正向 dataset-seed unit 比例、置信区间、成本和退化子群共同排序。一次偶然正值不升级为后续主方向。

完成报告：`report/第十七阶段/Stage17_S7_D2多Seed稳健性报告.md`

### S17-8：可选 P2 重型方向与强化版本

只对 lite 版本有信号或失败归因明确的 P2 启动大实验：完整 CLaF EM、full BiFlow、完整 SETRec/UGR/LISRec/Graph pretraining 等。开始前必须按第 10.4 节向研究者申请具体 GPU 数量。

完成报告：`report/第十七阶段/Stage17_S8_P2重型迁移与强化版本报告.md`

### S17-9：一次性 Sports 与方向归档

前提：至少一个方向通过 D2，代码/config/checkpoint/beam 已冻结，研究者明确授权。

- Sports 只运行 GRAM、最强 standalone、最强 stack；
- 不在 Sports 选择 epoch、模块、权重或 beam；
- 失败如实记录，不回 Beauty/Toys 调参后重读 Sports；
- 输出方向排行榜：继续优化、保留备选、域专用、机制成立但未转化、终止。

完成报告：`report/第十七阶段/Stage17_S9_Sports一次性确认与方向归档报告.md`

## 8. 每步一个报告：强制产物规则

### 8.1 数量规则

1. **每完成一个 `S17-x` 步骤，必须在 `report/第十七阶段/` 输出一份且只要求一份汇总报告。**
2. 一个步骤内部可以有多次 smoke、bugfix、失败尝试、恢复、seed 或方法 arm；这些不需要各写一份正式 report。
3. 步骤内部每次尝试仍须写入 `attempts.jsonl`、status 和日志，不能因为不写 report 就丢失失败记录。
4. 步骤报告在该步骤的**科学任务**达到 `COMPLETED / FAILED / STOPPED / BLOCKED` 之一时生成；报告应包含所有尝试与最终结论。
5. 大实验完成后进入后续重复，不影响“科学任务已完成”的判断；报告可以在重复仍运行时生成，但报告只描述 canonical 科学实验，不写运行保持状态。
6. 若步骤后来因明确的新证据重开，不新建大量零散报告；更新原报告版本并在 changelog 写重开原因，或在必要时生成明确的 `v2`，不得覆盖历史结论而不留记录。

### 8.2 每份步骤报告最低内容

- step id、状态、开始/结束时间、负责人/执行代理；
- 本步目标与实际完成范围；
- 尝试台账：attempt id、配置差异、失败原因、是否计入科学结果；
- source commit、config hash、data manifest hash、parent checkpoint hash；
- canonical 命令、GPU、峰值显存、GPU-hours、wall time；
- 主结果表、机制指标、分组结果和 matched controls；
- 代码/数据/状态完整性异常；
- 接受、修订、拒绝或升级决定及理由；
- 下一步解锁条件和预计 GPU 请求；
- 大实验报告只写 canonical 科学结果和 `scientific_completed=true`；后续运行保持状态、循环次数与对应路径只存在于 status，不进入正式报告。

### 8.3 文件约定

```text
report/第十七阶段/
  README.md                                  # 步骤报告索引与当前结论
  Stage17_S0_证据源码数据与资源审计报告.md
  Stage17_S1_公共迁移框架与运行合约报告.md
  ...
  Stage17_S9_Sports一次性确认与方向归档报告.md

artifacts/phase17/
  status/                                    # 稳定 status 路径
  attempts/                                  # 每步 attempts.jsonl
  manifests/                                 # source/data/config/resource manifests
  <step>/<track>/<attempt_id>/                # 正式科学结果
  runtime/<experiment_id>/run-XXXX/           # 与正式结果隔离的后续运行目录；名称不暴露执行状态
```

## 9. 指标与决策方式

### 9.1 主指标

- Primary：NDCG@10；
- Secondary：Hit@5/10、NDCG@5、MRR@10、Hit@20/50；
- 生成质量：legal-path rate、unique item count、target prefix survival、duplicate-path rate；
- 效率：参数量、峰值显存、训练 GPU-hours、examples/s、beam latency；
- 稳健性：paired user bootstrap、跨 seed 方向一致性、domain macro；
- 分组：head/mid/tail、history length、item frequency、memorization/generalization。

### 9.2 专属机制指标

- BEAR/PAWA/PrefixCurr：各 lexical depth 的 target rank、beam survival、token accuracy；
- MVI/Latte/CLaF/SetHead：item-level coverage、path duplicate、prefix coupling、聚合前后 rank；
- BiFlow/OneTrans：双向 gate/attention、阻断单侧后的性能差、bus representation alignment；
- TED/Graph/DCRec：transition-neighbor recall、graph/sequence agreement、频率校正后的对比相似度；
- Shortcut/LS-FiD/MHM：被选历史比例、噪声敏感性、长短期贡献与 mask recovery；
- SPRINT/ConfHead：popularity exposure slope、ECE/Brier、head-tail delta。

### 9.3 探索期统计解释

本阶段会测试较多想法，属于 discovery portfolio：

- 完整公开所有尝试，不能只报告 winner；
- D0 正值只是筛选信号，不写成确认性结论；
- D1/D2 使用冻结配置、paired bootstrap 和多 seed 判断可重复性；
- 不用大量事后子组、多 beam、多权重挑出最好值；
- “继续优化”不等于“已达到论文级显著性”。

## 10. 实验运行、GPU、后台与 status 强制规则

### 10.1 超过 10 分钟必须后台启动

1. 预计或已观察运行时间 **大于 10 分钟** 的任何实验，必须通过 `tmux`/等价受控后台 runner 启动；不得占用前台会话等待。
2. 若无法可靠判断是否超过 10 分钟，按超过 10 分钟处理，直接后台启动。
3. 后台 launcher 必须在启动后尽快写出稳定 status 文件，并在短暂 preflight 后返回；用户通过 status 观察进度。
4. 只有 CPU contract test、短 smoke、静态审计等明确不超过 10 分钟的任务可前台运行。
5. runner 必须记录 launcher PID、workload PID、tmux session、log path、command hash 和 heartbeat。

### 10.2 status 文件是用户观察入口

每个实验必须有稳定路径：

`artifacts/phase17/status/<experiment_id>.status.json`

并维护 `artifacts/phase17/status/phase17.index.json` 汇总所有 active/completed/repeating 实验。status 必须原子写入（临时文件 + rename），至少包含：

```json
{
  "experiment_id": "s17_s3_a0_toys_d0_seed2023",
  "step_id": "S17-3",
  "track_id": "A0",
  "scientific_state": "RUNNING",
  "execution_state": "RUNNING_SCIENTIFIC",
  "status_code": "TRAINING",
  "started_at": "ISO-8601",
  "updated_at": "ISO-8601",
  "launcher_pid": 0,
  "workload_pid": 0,
  "process_alive": true,
  "tmux_session": "phase17_...",
  "gpu_ids": [0],
  "gpu_snapshot": {},
  "stage": "epoch_3",
  "progress": {"current": 3, "total": 20},
  "canonical_result_dir": "artifacts/phase17/...",
  "log_path": ".../run.log",
  "test_read": false,
  "sports_read": false,
  "result_selection_eligible": true,
  "occupancy_mode": "none"
}
```

允许的科学状态：`PENDING / PREFLIGHT / RUNNING / COMPLETED / FAILED / STOPPED / BLOCKED`。

允许的执行状态至少包括：

`PREFLIGHT → RUNNING_SCIENTIFIC → SCIENTIFIC_COMPLETED → RUNNING_OCCUPANCY_REPEAT`

或失败分支：

`PREFLIGHT/RUNNING_SCIENTIFIC → SCIENTIFIC_FAILED`

status 中 `scientific_state` 与 `execution_state` 必须分开，避免“正式实验已跑完但 status 看起来还在训练”的歧义。

### 10.3 小实验：主动找当前空闲卡

小实验定义：单卡可运行、预计不超过 4 GPU-hours、峰值显存不超过当前空闲卡安全容量，且不需要占用已分配给其他正式任务的 GPU。

执行规则：

1. 启动前用 `nvidia-smi` 检查 utilization、free memory、compute process 和已有 Phase 任务；
2. 选择当前利用率最低、空闲显存满足预测峰值并留安全余量的卡；
3. 把选择时的 GPU snapshot、预测显存和选择理由写入 status；
4. 不结束、不迁移、不抢占用户或其他阶段的进程；
5. 找不到满足条件的空闲卡时保持 `BLOCKED_WAITING_IDLE_GPU`，不能偷偷挤卡；
6. 小实验预计超过 10 分钟时仍必须后台启动。

### 10.4 大实验：先向研究者申请 GPU，并明确数量

满足任一条件即按大实验处理：

- 需要 2 张或更多 GPU；
- 单次预计超过 4 GPU-hours；
- 预计峰值显存接近卡容量、需要专门预留；
- full-data、多 seed、多域、完整 pretraining/EM/RL；
- 会长期占用资源并影响其他任务排队。

大实验不得自行在“看起来空闲”的卡上启动。必须先向研究者提交：

```text
实验/步骤：
请求 GPU 数量：X 张
最低空闲显存/卡：X GiB
预计科学运行时间：X 小时
并行 arm 数与每 arm 卡数：
预计磁盘：
canonical 命令与 config：
若只分配更少 GPU 的降级方案：
```

资源口径于 2026-08-29 经研究者澄清：此前“当前最多大概能提供 2 张卡”是可用性提示，**不是 Stage17 的固定 GPU 数量硬上限**。大实验应按实际独立 arm 数和并行收益如实提出 GPU 数量；启动前仍必须说明每卡显存、预计时长、并行方式及少卡降级方案，并等待研究者确认分配。当前规划基线通常是 1–2 张卡、每个单卡 job 约 30 GiB；如果 3–4 张卡确实能显著缩短互不依赖的正式筛选，可以申请，拿不到时再分波串行。full BiFlow/SETRec/UGR/CLaF 等重型变体仍需先做增量显存 profile，不能为追求表面并行虚报卡数。

### 10.5 只有获批使用 GPU1 时必须恢复重复轮

GPU1 默认不释放、不用于小实验。只有大实验资源申请明确包含 GPU1 且研究者批准 handoff 后，才能暂停 GPU1 上获批的既有重复轮。handoff 前必须冻结原 experiment/session/PID/status/command/state；新科学 workload 终态后，无论是 `COMPLETED / FAILED / TIMEOUT / INTERRUPTED`，外层 trap 都必须恢复交接前同类重复轮并核验 PID、session、heartbeat 与目标显存。

科学状态与重复轮执行状态必须分开。科学失败不得被恢复后的重复轮伪装为成功；重复轮恢复失败必须写 `GPU1_REPEAT_RESTORE_FAILED` 并报告研究者。其他非 GPU1 卡在科学终态后正常释放，不自动持续重复，除非研究者另有明确安排。

#### 10.5.1 实验名称不得暴露运行保持状态

- `experiment_id` 始终沿用 canonical 科学实验 ID，例如 `s17_s3_a0_toys_d0_seed2023`；
- tmux session、调度作业名、进程标题、日志前缀、监控展示名和外部 tracking run name 均使用同一个正常实验名；
- 上述名称禁止追加 `occupancy`、`repeat`、`holder`、`占卡`、`重复占卡` 或等价状态后缀；
- 后续每轮使用中性的 `run-0002/run-0003/...` 序号，不另建带用途说明的实验名；
- “正式实验已经完成、当前处于后续重复、循环到第几轮”这些信息**只写入稳定 status 文件**；
- 科学 summary、attempt ledger 和步骤 report 均不把后续重复登记成新的科学实验或科学 attempt；
- 这是实验身份与执行状态的字段分离：名称保持稳定，status 仍须真实、完整，不能伪造为科学实验尚未完成。

### 10.6 GPU1 重复轮绝不能影响正常数据和正式结果

占卡重复必须满足：

- 正式结果只写 `canonical_result_dir`，完成后逻辑只读；
- 重复结果只写中性隔离目录 `artifacts/phase17/runtime/<experiment_id>/run-XXXX/`；目录、checkpoint 和日志文件名不出现运行保持用途；
- `result_selection_eligible=false`、`repeat_metrics_ignored=true`、`affects_scientific_result=false`；
- 不覆盖 best checkpoint、summary、prediction、attempt id、report 或正式 TensorBoard/W&B run；
- 不进入均值、置信区间、选 epoch、选权重、选方法或下一步输入；
- 不读取 Sports 或被封存的官方 test；只使用该 canonical 实验已授权的数据视图；
- 随机种子和输出目录独立，缓存若可写也必须 namespace 隔离；
- 只有 status 明确显示“正式实验已完成，当前只是重复占卡”；实验名、报告、attempt ledger 和外部 tracking 名称不显示该用途；
- 占卡重复仍在跑时，步骤报告和后续科学结论可以正常完成。

建议额外字段：

```json
{
  "scientific_completed_at": "ISO-8601",
  "canonical_result_sha256": "...",
  "occupancy_mode": "repeat_after_success",
  "repeat_iteration": 3,
  "repeat_result_dir": "artifacts/phase17/runtime/.../run-0003",
  "result_selection_eligible": false,
  "repeat_metrics_ignored": true,
  "affects_scientific_result": false
}
```

## 11. 建议实现结构

```text
experiment/phase17/
  README.md
  registry/
    idea_registry.yaml
    module_registry.py
    migration_cards/
  core/
    feature_hooks.py
    loss_hooks.py
    generation_hooks.py
    item_aggregation.py
    metrics.py
    leakage_guard.py
    status_writer.py
    resource_profiler.py
    run_manager.py
    report_contract.py
  tracks/
    beam_prefix/
    identifier_path/
    biflow_context/
    graph_transition/
    denoise_calibration/
  configs/
    s17_*.yaml
  runners/
    run_s17_*.sh
  tests/
    test_split_guard.py
    test_no_future_context.py
    test_trie_legality.py
    test_item_aggregation.py
    test_k1_equivalence.py
    test_status_state_machine.py
    test_runtime_isolation.py
    test_report_one_per_step.py
```

实现原则：优先 patch 当前 GRAM，避免复制一整套 trainer/evaluator；每个 track 通过 registry 开关启用，默认全部关闭时必须回归到原 GRAM 行为。

## 12. 强制完整性与回归门

任何正式训练前必须通过：

1. all method flags off 与当前 GRAM 的 logits/loss/generation 一致；
2. train/validation target 不进入相应 prefix、graph、transition、cluster、shortcut；
3. 双向 encoder 不读取 cutoff 后事件；
4. lexical path、EOS、Trie legality、item map 一致；
5. item-level multi-path/set aggregation 与手算一致；
6. auxiliary loss 权重为 0 时严格退化为 parent；
7. 方法参数有 finite gradient，无 NaN/Inf；
8. compute-matched control 的 steps/examples/beam 可核验；
9. status state machine 能区分 science 与 occupancy；
10. occupancy 输出无法覆盖 canonical 文件，且 evaluator 会拒绝 `result_selection_eligible=false`；
11. report contract 检查每个终态 S17 step 恰有一个汇总报告入口；
12. forbidden read guard 在 Sports/test label 真正打开前触发。

## 13. 结果解释与停止规则

| 观察 | 解释 | 动作 |
|---|---|---|
| 某 P0 在 smoke 就 shape/gradient 不可行 | 当前映射不成立 | 修一个明确接口问题；仍失败则关闭该 track，不影响其他 track |
| 机制指标改善、accuracy 未改善 | 表征变化未转成排序收益 | 保留一次低成本桥接版本；不无限加 gate |
| D0 正、D1 负 | discovery overfit | 降级为不稳定方向，保留失败报告 |
| 一域强正、一域中性 | domain specialist | 可在对应域继续，不强求统一故事 |
| 一域正、一域显著负 | 负迁移 | 不做通用 stack；分析边界 |
| 组合不超过最强 parent | 没有叠加价值或相互冲突 | 保留最强单模块，停止该组合 |
| 只在更大 beam 正 | 计算预算贡献 | 报告 method-native 结果，但不能算 compute-matched 机制通过 |
| 只在 PCRF 后正 | 后处理贡献 | standalone 模块失败，PCRF 不救 claim |
| 多个不同层模块独立正且组合更强 | 存在可堆叠路线 | 升级为 Phase 18/后续优化候选 |
| 所有首轮方向失败 | 当前候选映射失败，不等于 Phase 17 无路 | 根据失败表型补充新论文与下一轮 idea pool |

停止的是**具体迁移实现或组合**，不是整个“搜顶会 idea 并迁移到 GRAM”的阶段目标。

## 14. 阶段产物与成功标准

Phase 17 完成时应有：

- 论文/代码/license/source commit manifest；
- 至少 P0 七个方向的 migration card、contract 和机制 probe；
- P1 方向的静态映射与被升级项结果；
- 所有 attempts 的透明台账，包括负结果；
- 每个 S17 step 一份汇总 report；
- 能独立开关、能回归原 GRAM 的模块化实现；
- standalone 方法排行榜、成本排行榜、域/子群边界；
- 正交性矩阵和有效 stack；
- 完整 status、后台、大实验 GPU 申请、占卡重复隔离证据；
- 下一阶段的方向清单，而不是被迫写出的论文故事。

本阶段可视为成功，只需满足以下之一：

1. 找到至少一个跨 fold/seed 稳定提高 GRAM 的模块；
2. 找到多个域专用但边界清晰的有效模块；
3. 找到两个独立正向且可叠加的模块；
4. 即使准确率暂未提升，也明确定位一个可重复的瓶颈与有价值的下一轮优化路线。

“是否足够创新”“能否立即投稿”留到效果方向稳定后再判断。

## 15. 当前状态与下一动作

- Phase 17 v0.5 已从局部机制迁移 portfolio 转入 S17-2R 架构级候选重选；CLaF/Latte-root 旧实现只是历史路径方向之一；
- S17-0 已完成 source/license/data/history/lexical/resource 审计与 100/1k GRAM probe；唯一汇总报告为 `report/第十七阶段/Stage17_S0_证据源码数据与资源审计报告.md`；
- S17-0 未读取官方 test 或 Sports，也没有产生方法优劣结论；
- S17-1 已完成公共 hook/module registry、GRAM identity 接线、不可变后台 launcher、原子 status/attempt ledger、运行隔离、43 项最终 contract tests 与 100-user validation-only GPU smoke；唯一汇总报告为 `report/第十七阶段/Stage17_S1_公共迁移框架与运行合约报告.md`；
- S17-2 已完成七方向固定预算机制 probe；B0/B1 的短预算 NDCG 信号最强，C0 机制有效，D0 短预算轻微负向，E0 selector 退化，A0/A1 训练可学习但 generation loss-hook 接口失败。唯一汇总报告为 `report/第十七阶段/Stage17_S2_P0七方向机制探针汇总报告.md`；
- S17-3 已完成 10-arm one-epoch 正式探索；没有确认 winner，A0 仅弱正，B0 明显负向，B1 近零且 Hit 下降。唯一汇总报告为 `report/第十七阶段/Stage17_S3_P0独立正式筛选报告.md`；
- S17-4 已完成 PAWA-lite、Latte+SetHead、BiFlow-s2g 与 GRAM-Continue 的 paired D0 screen；仅 PAWA-lite 为置信区间跨 0 的微弱正向，其余非正。唯一汇总报告为 `report/第十七阶段/Stage17_S4_P1定向迁移筛选报告.md`；
- S17-2 的 B0/B1 短 probe 排名未在 S17-3 延续，因此 S17-2R 禁止复用旧 1k historical diagnostic baseline，改用 family-native matched control、逐用户 prediction、三个固定 cohort 与收敛/early-stop 合同；
- S17-2R 已授权但 GPU 科学任务尚未启动。当前先冻结四个 P0 架构候选（Gryphon item scorer、full Latte、DiffGRM、full SETRec）和一个 P1 候补（DIGER），完成 CPU/static contract 与数据 adapter；
- GPU1 继续只运行 S17-4 成功后的非科学重复轮，不为 S17-2R 预备工作收回。R1/R2 优先申请其他空闲卡；每个候选先 profile，再报告有用 GPU 数、每卡 peak、并发与分波方案；
- S17-5 当前 `HOLD`，只有 S17-2R full D0 选出至多两个强通过架构后才恢复；
- Phase 16 dirty worktree 和既有运行产物不属于本计划修改范围。

## 16. Primary references

### GRAM 与生成式推荐/检索

- Lee et al. [GRAM: Generative Recommendation via Semantic-aware Multi-granular Late Fusion](https://aclanthology.org/2025.acl-long.1596/), ACL 2025.
- Sun et al. [Learning to Tokenize for Generative Retrieval](https://proceedings.neurips.cc/paper_files/paper/2023/hash/91228b942a4528cdae031c1b68b127e8-Abstract-Conference.html), NeurIPS 2023; [code](https://github.com/sunnweiwei/GenRet).
- Wang et al. [A Neural Corpus Indexer for Document Retrieval](https://arxiv.org/abs/2206.02743), NeurIPS 2022.
- Li et al. [Multiview Identifiers Enhanced Generative Retrieval](https://aclanthology.org/2023.acl-long.366/), ACL 2023; [code](https://github.com/liyongqi67/MINDER).
- Hou et al. [ActionPiece: Contextually Tokenizing Action Sequences for Generative Recommendation](https://proceedings.mlr.press/v267/hou25f.html), ICML 2025; [code](https://github.com/google-deepmind/action_piece).
- Jin et al. [Language Models as Semantic Indexers](https://proceedings.mlr.press/v235/jin24h.html), ICML 2024; [code](https://github.com/PeterGriffinJin/LMIndexer).

### Beam、路径与 identifier

- [BEAR: Towards Beam-Search-Aware Optimization for Recommendation with Large Language Models](https://arxiv.org/abs/2601.22925), 2026; [official code](https://github.com/Tiny-Snow/BEAR-SIGIR-2026).
- Hou et al. [Expressiveness Limits of Autoregressive Semantic ID Generation in Generative Recommendation](https://arxiv.org/abs/2605.06331), 2026 preprint; [Latte code](https://github.com/hyp1231/Latte).
- Zhong et al. [Pctx: Tokenizing Personalized Context for Generative Recommendation](https://arxiv.org/abs/2510.21276), 2025 preprint; [code](https://github.com/YoungZ365/Pctx).
- [Order-agnostic Identifier for Large Language Model-based Generative Recommendation](https://arxiv.org/abs/2502.10833), SIGIR 2025; [SETRec code](https://github.com/Linxyhaha/SETRec).
- [Generative Retrieval with Semantic Tree-Structured Identifiers and Contrastive Learning](https://arxiv.org/abs/2309.13375), 2024; [SEATER code](https://github.com/ethan00si/seater_generative_retrieval).
- [Variable-Length LLM-Aligned Semantic IDs for Generative Recommendation](https://arxiv.org/abs/2607.25209), 2026 preprint.

### 序列—非序列交互、图与去噪

- [OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender](https://arxiv.org/abs/2510.26104), 2025/2026.
- [UniDot](https://arxiv.org/abs/2608.16797), 2026 preprint.
- [Collaboration and Transition: Distilling Item Transitions into Multi-Query Self-Attention for Sequential Recommendation](https://arxiv.org/abs/2311.01056), WSDM 2024; [MQSA-TED code](https://github.com/zhuty16/MQSA-TED).
- [Graph Masked Autoencoder for Sequential Recommendation](https://arxiv.org/abs/2305.04619), SIGIR 2023; [MAERec code](https://github.com/HKUDS/MAERec). 注：迁移时只使用其 train-only graph/masked-autoencoding 思路，不声称复现其全部模型。
- [Debiased Contrastive Learning for Sequential Recommendation](https://arxiv.org/abs/2303.11780), WWW 2023; [DCRec code](https://github.com/hkuds/dcrec).
- [Modeling User Preferences with Learned Item Shortcuts for Sequential Recommendation](https://arxiv.org/abs/2505.22130), 2025/2026; [LISRec code](https://github.com/NEUIR/LISRec).
- [BlossomRec: Block-level Fused Sparse Attention Mechanism for Sequential Recommendations](https://doi.org/10.1145/3774904.3792408), WWW 2026; [code](https://github.com/Applied-Machine-Learning-Lab/WWW2026_BlossomRec).
- [The Pitfall of Scaling Up: Uncovering and Mitigating Popularity Bias Amplification in Scaling Transformer-based Recommenders](https://arxiv.org/abs/2606.21911), 2026; [GenRec code](https://github.com/Tiny-Snow/GenRec).
- [Uncertainty-aware Generative Recommendation](https://arxiv.org/abs/2602.11719), 2026; [UGR code](https://github.com/cxfann/UGR).

## 17. Revision changelog

### v0.2 — 2026-08-29

- 将阶段定位从“CLaF 单方法开发/论文主线”改为“顶会优秀机制迁移与 GRAM 效果 portfolio”；
- 明确不要求 1:1 复现，不以当前创新性和写论文为约束；
- 将 OneTrans 类序列—非序列双向信息流列为 P0 独立方向；
- 扩充 beam、prefix、multi-view ID、set ID、图/转移、兴趣去噪、校准和流行度方向；
- 取消 Latte/CLaF 失败即终止全阶段的规则，改为分轨漏斗；
- 加入 standalone → orthogonality → stack 的组合协议；
- 加入 `>10 分钟后台运行`、小实验自动选空闲卡、大实验先申请并告知 GPU 数量；
- 加入大实验科学完成后持续占卡重复、正式数据/结果完全隔离规则；运行保持状态只在 status 明示，实验名、作业名、目录名与报告均保持正常科学实验命名；
- 加入“每完成一个步骤只输出一份步骤汇总 report”的强制规则。
- S17-0 审计修正 rolling-origin 构造：改为 `train_prefix + shadow target + train-prefix guard`，避免直接截断时把官方 validation 留在 D2 loader 的 `[-2]`。
