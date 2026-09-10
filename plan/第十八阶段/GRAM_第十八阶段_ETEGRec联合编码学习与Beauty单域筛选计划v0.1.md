# 第十八阶段：ETEGRec 联合编码学习与 Beauty 单域筛选计划 v0.1

## Material Passport

- 日期：2026-09-09。
- 来源：academic-research-suite / experiment-agent，plan，inline。
- 用户目标：找到能超过现有 GRAM、后续可能与 PCRF 组合的新模型机制；允许大改，优先复用历史基线。
- 原计划请求：“那你继续计划”；随后用户“那继续吧”，接续实施、核验和预算内正式筛选。
- 当前状态：`COMPLETED`；2026-09-09 16:08:45 完成。RQ 预训练 5,850 商品 pass、交替训练 350 epoch、固定编码微调 65 epoch，均按平台停止；最终选择 `joint_290`，全量 NDCG@10 = 0.04912738797，相对 GRAM -24.39%。完整结果与预测复核见 [Beauty 结果报告](../../report/第十八阶段/Stage18_ETEGRec_Beauty完整结果与投入结论报告.md)。当前 Beauty 配置归档，不追加同设置训练或多 seed；下文为原计划与实施记录。
- 实验标识：`S18-ETEG-CF512-Beauty-S2023`。这是作者核心机制的本地适配，不是官方 Beauty 配置复现，也尚不是 GRAM 内部模块的完成稿。
- 本轮输入检查：[input_review_20260909.json](../../artifacts/phase18/etegrec_planning/input_review_20260909.json)。记录数据哈希、协同 checkpoint、基线及旧任务状态快照。

## 1. 本轮只回答一个问题

在相同 Beauty 训练边界与验证用户上，采用受推荐任务约束的可学习商品编码，是否能产生超过已有强 GRAM 的候选模型？

假设：tokenizer 与推荐器分开学习可能形成不适合序列预测的编码划分；通过序列—商品对齐、偏好—语义对齐及交替更新，有机会改善生成推荐。这是待检验假设，本地尚未证明编码就是主要瓶颈。

本轮优先检验完整候选的绝对效果。只有出现正信号，才进一步判断收益能否归因于联合编码、怎样接入 GRAM、是否与 PCRF 互补。独立候选的成功不自动等于 GRAM 迁移成功；失败也不能否定所有联合编码方案。

依据是[第十七阶段失败复盘](../../report/第十八阶段/Stage18_第十七阶段失败复盘与大改选型依据.md)：不再把新的论文名映射成相似的池化门控；不以超过较弱新对照替代超过强 GRAM；不为了排除所有失败解释持续追加预算。

## 2. 来源机制与本地改动

[ETEGRec 论文 v3](https://arxiv.org/abs/2409.05546v3) 已被 SIGIR 2025 接收。[作者仓库](https://github.com/BishopLiu/ETEGRec) 提供 RQ-VAE、生成推荐器及交替训练。论文实验不能直接证明它能超过本项目的 GRAM。

| 部分 | 首轮处理 | 与旧尝试的区别或限制 |
| --- | --- | --- |
| 商品 tokenizer | 保留 RQ-VAE 的编码、重建、量化与可训练码本 | 不把固定语义 ID 替换当作联合学习 |
| 推荐器 | 保留作者生成模型、历史编码与逐 token 解码 | 首轮独立运行，暂不同时实施 GRAM 结构迁移 |
| 对齐 | 保留 sequence-item alignment 与 preference-semantic alignment | 对齐损失必须实际影响 tokenizer 参数；不要求离散索引上的 CE 直接可导 |
| 训练 | 先预训练 RQ-VAE，再交替训练 tokenizer / 推荐器，最后固定编码训练推荐器 | 不以几轮 smoke 代替完整机制筛选 |
| 商品输入向量 | 复用本地 Beauty item-head 的 512 维商品向量，按其评分方式逐行 L2 归一化并冻结 | 作者提供 SASRec 向量；本地改用已有协同向量，需明确记载来源差异 |
| 模型维度 | 推荐器维度沿用作者 Scientific 配置；商品输入与重建输出改为 512 维 | 不额外做 PCA 或新增文本融合分支 |
| 数据 / 解码 | 本地历史最多 20 商品；全目录合法编码约束的 beam 50 | 属于本地评估适配，不能称作者默认 beam 20 的原样复现 |

作者配置入口：[Scientific 配置](https://raw.githubusercontent.com/BishopLiu/ETEGRec/main/config/scientific.yaml)、[运行参数](https://raw.githubusercontent.com/BishopLiu/ETEGRec/main/run.sh)、[模型](https://raw.githubusercontent.com/BishopLiu/ETEGRec/main/model.py)。实施已固定 commit `58d9736afc28e03e190a107a5fa22f5241be6088`，24 个源码 / 资源文件的哈希见 [source_manifest.json](../../artifacts/phase18/etegrec/sources/source_manifest.json)，训练不使用浮动 main。

首轮不新训 SASRec：已确认存在可复用、覆盖本地目录的协同向量。若本轮失败，不自动增加“换 SASRec 向量再试”的第二套实验；必须另有实质依据才能重开。复用的协同 checkpoint 曾用历史 validation 选模，本轮及后续单 seed 比较均属于探索证据。

## 3. 已确认可复用的输入

以下路径均相对项目根目录。

| 输入 | 路径 | 已确认内容 |
| --- | --- | --- |
| 训练样本 | `artifacts/phase18/diff_gram/beauty/prepared_v1/train.jsonl` | 131,413 条 rolling next-item 样本，历史最多 20 |
| 验证样本 | 同目录 `validation.jsonl` | 22,363 个用户，每人一个目标 |
| 商品映射 | 同目录 `catalog.json` | 12,101 件商品；训练可见商品 12,068 件 |
| 协同 checkpoint | `artifacts/phase9/p9x_beauty_item_head/best_item_head.pt` | epoch 7；商品矩阵 `[12102, 512]`，含 PAD；CPU 检查有限值及商品顺序一致 |
| 固定趋势 cohort | `artifacts/phase18/diffgrm_native/beauty/run_v1/screen_r1/trend_cohort.json` | 已有 2,000 用户，不重新挑选有利子集 |
| GRAM / PCRF 逐用户参照 | `artifacts/phase9/p9s_multiseed/Beauty/seed2023/validation/per_user.tsv` | 同用户 rank，可直接 CPU 配对重算 |
| 历史参照汇总 | `artifacts/phase18/diffgrm_native/beauty/run_v1/full_r2/historical_references.json` | 已有完整集和趋势子集指标 |

协同 checkpoint SHA256：`7f6efd92e92a681cad7eeae77a80c71bc7d28d405f34c685232b68b89fb6180a`。其 embedding 的行号来自按原始商品 ID 排序的目录，本轮已核对与 phase18 的目录一致。

训练只读取已物化 train；validation 只用于选模和报告。无需重新打开包含 test 目标的原始用户序列。RQ-VAE 拟合使用训练可见商品向量，编码映射仍覆盖完整 12,101 商品目录；33 件未出现在训练前缀中的商品不能从评估分母或目录中删除。

作者 `data.py` 默认同时载入 train / valid / test；本地使用仅含 train / validation 的 adapter，不能直接运行原入口后再声称没有读取 test。[数据入口](https://raw.githubusercontent.com/BishopLiu/ETEGRec/main/data.py)

| 参照与范围 | NDCG@10 | Hit@10 | Hit@50 |
| --- | ---: | ---: | ---: |
| GRAM，完整 22,363 用户 | 0.064973703 | 0.108751062 | 0.208111613 |
| GRAM + PCRF，完整用户 | 0.067907682 | 0.113848768 | 0.208111613 |
| GRAM，趋势 2,000 用户 | 0.076351501 | 0.124000000 | 0.215000000 |
| GRAM + PCRF，趋势用户 | 0.076601994 | 0.126500000 | 0.215000000 |

完整集与趋势子集不能混用。首轮只训练一个新候选，不重训、不重新推理 GRAM / PCRF 基线。

## 4. 实施顺序与训练预算

### P0：一次工程检查，不承担效果结论

固定作者源码、实现数据与 item 输出适配后，用少量真实训练样本完成以下四项：

1. 同一商品的向量、编码、目标及输出 item ID 对齐；合法四 token 编码唯一，PAD 不进入推荐列表。
2. 分别检查 ID 阶段与 REC 阶段。单独反传两项对齐损失时，tokenizer 对应参数有有限、非零梯度，并在其更新阶段实际改变；被冻结的模型保持不变。只看到重建 loss 下降不算通过。
3. 保存 / 恢复推荐器、RQ-VAE、编码表、两套 optimizer / scheduler 和 RNG；恢复前后下一步学习率及固定输入输出一致。每个 best checkpoint 必须与当时编码表成套保存。
4. 实测 RQ-VAE、ID 更新、REC 更新和 beam 50 生成吞吐及显存；检查单卡、最后一个 batch 和历史截断。记录更新步数，不能只打印 epoch。

这些检查仅验证本地适配的正确性；不扩展为一套新诊断研究。CPU 资产检查已完成，GPU smoke 尚未执行。

### P1：单域、单 seed、单配置

| 项目 | 首轮计划值 |
| --- | --- |
| 数据 / seed | Beauty / 2023；全量训练转移 |
| 历史 | 最多 20 商品，按真实时间顺序 |
| 推荐器 | encoder 6 层、decoder 6 层、`d_model=128`、`d_ff=512`、4 heads，沿用作者配置 |
| 编码 | 3 层 RQ，每层 256 code，latent 128；第 4 token 按目录顺序消解同前缀冲突 |
| RQ 输入 | 冻结 512 维协同商品向量；MLP hidden `[512,256]`，重建输出 512 |
| 实际训练 batch | 优先 512，梯度累积 1；131,413 样本对应 257 updates / pass |
| 联合训练 | 最多 400 个交替 epoch：200 个 ID pass、200 个 REC pass |
| 前期对齐 | 前 10 个交替 epoch 不启用对齐，随后按作者损失权重启用 |
| 学习率 | REC 峰值 0.005；ID 峰值 0.0001；AdamW，weight decay 0.05 |
| 对齐权重 | 两阶段 KL 均 0.0001、contrastive 均 0.0003；量化 / 重建与 CE 按各自阶段启用 |
| 固定编码阶段 | 从联合阶段选定的成套最佳模型开始，最多 100 个 REC epoch；LR 0.0005，无 warmup，cosine；只用 CE |
| 生成 | 目录约束 beam 50，输出 50 个不同商品及其模型分数 |

batch 512 是否可行由 P0 实测决定。若必须减小实际 batch，先在正式训练前记录对比学习负例数的变化，并重算训练步数；不能仅靠梯度累积就宣称与 batch 512 的 InfoNCE 等价。首轮不并行搜索多个 batch 或学习率。

RQ-VAE 预训练从作者设置出发：batch 1024、LR 0.001、weight decay 0.0001、最多 10,000 个商品 pass、k-means 初始化。它只遍历约 1.2 万商品，不等于 10,000 个推荐训练 epoch。可在至少 200 个 pass 后，每 50 个 pass 检查一次训练商品重建误差；连续 10 次相对最佳改善不足 0.1% 时结束，保留最低重建误差模型。达到上限或时间预算仍未稳定时标记预算截断，不声称 tokenizer 充分收敛。[作者预训练入口](https://raw.githubusercontent.com/BishopLiu/ETEGRec/main/RQVAE/run_pretrain.sh)

### 学习率必须按实际更新计算

静态阅读发现，作者联合训练的外层循环轮流更新 ID 与 REC；但 REC scheduler 的总步数按全部外层 epoch 计算。照搬计数会使 REC 走到外层 400 时仍未到该 scheduler 的终点。这是实现计数事实，不据此宣称论文结果错误。[训练实现](https://raw.githubusercontent.com/BishopLiu/ETEGRec/main/trainer.py)

本地明确采用一次预先冻结的适配：ID 与 REC 分别按实际最大更新数设置 cosine 总步数。batch 512 时，两者各为 `200 × 257 = 51,400`；warmup 分别为 ID 4,000、REC 8,000。固定编码阶段另设 `100 × 257 = 25,700` 步日程。这属于与作者源码的日程差异，结果报告必须保留。

不在 warmup 内累计用于停止的 patience。联合阶段至少运行到外层 epoch 160，再开始平台判断：每 10 个外层 epoch、且在 REC 更新后验证一次；之后连续 6 次 NDCG@10 未比阶段最佳改善至少 0.0001，结束联合阶段并进入固定编码阶段。这里结束联合阶段不是淘汰候选。

固定编码阶段每 5 epoch 验证一次；至少运行 40 epoch 后清零并启动 patience，连续 5 次未改善 0.0001 时停止，最多 100 epoch。严格按最大 NDCG@10 选 best，`0.0001` 只用于平台计数；并保留联合阶段的 best，避免后续训练退化后丢掉更好模型。早停表示当前日程下平台，不能声称已经走到学习率终点。

### 时间边界

首轮总上限为 **12 GPU 小时**，覆盖 smoke、RQ 预训练、联合训练、固定编码训练及一次完整验证；只使用一张卡。该数字是投入上限；执行实测的最大日程预计 6.93 小时（已含 25% 余量），当前使用 GPU 0 的空余资源，其他进程保持运行。

P0 后计算保守预计耗时：`1.25 ×（RQ 预训练 + 200 次 ID pass + 200 次 REC pass + 100 次固定编码 REC pass + 预定趋势验证 + 一次完整验证）`。用实测耗时计算，并单独列出固定编码阶段和完整验证所需时间。

如果最大计划日程不能落入该上限，先返回成本结论、修订预算；不偷偷压缩成几轮试验后作科学淘汰。正式启动前保存实际 batch、步数、ETA、最大时限和日程配置。训练过程遇到预算用尽时保存完整状态，结论写“预算截断”；不自动接长跑、增加第二 seed 或启用另一向量来源。

## 5. 评估、保留与归档

主指标 NDCG@10，辅以 Hit@10；Hit@50 用于判断候选覆盖。使用相同用户、目标、目录和历史边界，不用采样负例指标替代全目录生成推荐指标。

固定趋势 cohort 用于选模。训练结束后只对选定的一个 checkpoint 做一次完整 validation，并保存每个用户的 item 排序及分数。完整集包含趋势用户，且历史 validation 已反复用于探索，不能称为新的独立确认集。

复用历史 rank，报告绝对差、相对差、Hit 净增用户数和 2,000 次用户配对 bootstrap 的 95% CI。CI 描述用户层面的不确定性，不能消除选模偏差，也不代表跨 seed 稳健性；首轮不强制要求 CI 下界大于零才允许保留候选。

| 观察 | 本轮决策 |
| --- | --- |
| 完整 validation 的 NDCG@10 高于 GRAM，Hit@10 没有明显退化 | 保留为探索正信号；相对增益达到约 3% 时优先级更高，但 3% 不是显著性阈值，也不否定更小的正信号 |
| NDCG 小幅正，但 Hit@10 下降超过 0.001 或 tail Hit@10 下降超过 0.002 | 标记收益与损失并存，先做保存预测的 CPU 解释，不直接宣告胜出；上述数值是投入判断阈值，不是统计定理 |
| NDCG@10 不超过 GRAM，只有 Hit@50 上升 | 记录覆盖线索，不称主指标提点，不默认进入 PCRF 组合或扩域 |
| 两个主要训练阶段都已获得规定预算，主指标仍无正信号、趋势进入平台 | 归档当前适配；不以“也许还能训”追加训练 |
| 到时间上限仍持续上升，或关键训练阶段未完成 | 标记预算不足 / 未确认收敛；不证明机制无效，但当前不自动追加投入 |
| 映射、梯度、编码冲突或输出契约失败 | 工程失败，不能记作 ETEGRec 科学负结果 |

Hit@50 增益不是所有内部结构必须满足的硬门槛。候选先超过原 GRAM，并不意味着已经超过 GRAM + PCRF，更不等于获得第二个可发表贡献。

评估只接受真实 item：第四个 token 与对应 checkpoint 的目录映射共同决定商品；禁止任意共享 SID 命中、用目标补全、用旧 GRAM 补足新候选。编码前缀冲突超过第四 token 的容量时作为工程问题报告，不能截断目录。目录约束搜索与原始生成逻辑的差异要在报告记录。

## 6. 正信号之后才展开的工作

1. 先比较候选自身与候选 + 冻结 PCRF，以及历史 GRAM + PCRF。使用新候选自己的 50 件商品和四 token 对数概率；CF 评分器复用已有 checkpoint，必要时只计算新候选商品的分数。冻结 `lambda=1.0, beta=0.5, gamma=1.0, q1=6`，不重新调 PCRF。历史 PCRF rank 不能直接搬到新候选上。
2. 复用同一协同来源可能减少 PCRF 的互补性，因此组合收益需实测；不得从候选主指标上涨推导组合一定上涨。
3. 若值得进入 GRAM 路线，再设计保留已有文本 / 历史能力的迁移。迁移必须维持任务驱动的 tokenizer 更新；不能又变成冻结编码替换或池化门控。独立原型本身不是 GRAM 迁移结果。
4. 最终候选及迁移确定后，才补冻结 tokenizer 对照、匹配训练预算、Toys、多个 seed 和 A/B/C/D 消融，区分编码学习、协同输入、容量及额外训练带来的收益。既有 test 继续封存。

这些是后续决策顺序，不是首轮同时运行的实验矩阵。

## 7. 产物与接续入口

以下实现、配置和启动入口已落盘；源码与配置在启动时快照。结果报告在训练结束后补充指标：

- `experiment/phase18/core/etegrec_adapter.py`：作者核心与本地数据、编码、item 输出适配。
- `experiment/phase18/protocol/s18_etegrec.py`：预训练、交替训练、固定编码训练、评估与恢复。
- `experiment/phase18/config/s18_etegrec_beauty.json`：源码 commit、输入哈希、最终训练步数、预算和评估配置。
- `experiment/phase18/run_stage18_etegrec_beauty.sh`：单卡入口，固定配置和输出目录。
- `artifacts/phase18/etegrec/beauty/run_v1/`：源码 / 配置快照、profile、checkpoint、编码版本、status、逐用户预测及结果。
- [Stage18_ETEGRec_Beauty实现核验与启动报告](../../report/第十八阶段/Stage18_ETEGRec_Beauty实现核验与启动报告.md)：已确认的实现、检查、预算及启动证据；效果尚待验证。

runner 需在阶段、epoch、optimizer updates、两个 LR、验证进度和 ETA 上持续写状态；每个完整 epoch 保存可恢复 checkpoint。超时策略和进程归属仅针对本次新任务，不操作其他实验。

当前旧任务仅作状态备注：输入检查时 DIFF→GRAM Beauty 仍在完整验证；完成后按其结果收尾。DiffGRM 的额外 finish 代码已准备过，但未启动，本计划不触发该续训。旧 full_r2 配置、结果与 checkpoint 均保留。

2026-09-09 执行接续：正式命令为 `bash experiment/phase18/run_stage18_etegrec_beauty.sh --mode run`，tmux=`s18_etegrec_beauty`，PID=`1854985`。12:31 的完整 checkpoint 已保存 RQ epoch 786 / Adam 9,432 步，见 [startup_verification.json](../../artifacts/phase18/etegrec/beauty/run_v1/startup_verification.json)。RQ 的轮次描述商品向量预训练，不代表推荐器已训练 786 epoch。

执行补充：测速最初只预热了 25 次 RQ 更新，未训练成熟的前三位编码发生容量溢出；该失败原样保留于 `profile_v1`。将工程测速的 RQ 预热补足 50 个商品 pass 后，`profile_v2` 最大同前缀冲突为 24，四位完整编码唯一；未扩大码本、删除商品或改用人工编码。所有测速参数均丢弃，正式训练从固定 seed 重新开始；两次测速耗时一起计入 12 小时预算。
