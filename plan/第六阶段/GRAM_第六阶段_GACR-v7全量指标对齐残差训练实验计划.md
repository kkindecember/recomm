# GRAM 第六阶段：GACR 最后两轮增长与失败后主线计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-02
- Verification Status: IMPLEMENTED_PREREGISTERED_READY_TO_RUN
- Version Label: `phase6_gacr_last_two_growth_rounds_v2`
- Supersedes: `phase6_gacr_v7_full_fit_metric_aligned_loss_v1`
- Development Domains: Toys、Beauty
- Confirmation Domain: Sports（继续封存）
- Test: 继续封存
- Device: 物理 GPU0，30 GiB 总显存租约

## 0. 修订记录与阶段边界

本文件在原 GACR-v7 尚未实现、未启动、未生成新结果时修订。修订不改变 v6 已有结果，主要改变：

1. 保留原 v7 的“全量 fit + 指标对齐损失”作为第一轮机会；
2. 将零容忍 point-estimate safety gate 改为运行前冻结的非劣界，避免把一两个用户的离散交换
   直接解释为方法退化；
3. 预先定义第二轮 `GACR-v8`：真实生成 path score + 可辨识的 listwise residual calibration；
4. 预先定义 v7/v8 都未形成可增长信号后的主线：多源候选 drafting + GRAM item-level verification；
5. 关闭 GACR-v3 之后已经连续选择 identity 的 gate/attenuation/soft-weighting 修改族，也不回到
   ST-GCGD 的固定 token-logit 图融合。

“最后两轮”仅指当前 **固定候选集上的 GACR 排序校准主线**。它不是项目总实验次数，也不授权
自动实现、启动下一轮或读取 Sports/test。每轮完成后仍须由研究者明确要求分析并决定是否进入下一门。

## 1. 现有证据与总研究问题

冻结 GACR 核心在 v3/v4/v5 三批互斥 fresh development cohort 上累计 18/18 个域-seed
overall NDCG@10 点估计为正；v6 全量 fit 相对 GRAM 的 Toys/Beauty mean NDCG@10 分别为
`+2.469%/+2.972%`，six-cell macro 为 `+2.720%`，相对 v3 macro 增量为 `+0.559%`。

v6 未替换 v3 的原因是 Beauty mean tail NDCG@10、overall Recall@50、tail Recall@50 相对 v3
分别轻微下降。逐用户复算显示 Beauty overall Recall@50 的 `-0.0977pp` 相当于每 1024 用户净少
1 个命中，配对不确定区间跨零；因此它是需要保护的风险信号，但不足以证明 v6 科学上更差。

当前总研究问题是：

> **生成推荐优化 token likelihood，而最终评价发生在 item-level top-K ranking。能否在冻结 GRAM
> 的条件下，通过指标对齐、真实生成路径分数和轻量 listwise residual calibration，稳定放大当前
> 2%–4% 的开发收益；若固定候选集已到上限，能否通过多源候选 drafting 与 GRAM verification
> 进一步突破 coverage ceiling？**

## 2. 两轮总体顺序与决策

| 轮次 | 唯一核心问题 | 主要改变 | 无增量时的决定 |
|---|---|---|---|
| GACR-v7 | 现有 hinge 是否与 top-K 指标错位 | 只换截断敏感 pairwise loss | 保留 incumbent，但仍进入 v8；不搜索相邻 loss |
| GACR-v8 | rank-only、逐 item 打分是否丢失生成路径和列表关系 | path-score interface + nested listwise ablation | 关闭固定候选 GACR 增长主线，进入候选 drafting/verification |

探索期的健康目标是最终相对 GRAM 在双域形成约 `+4%–6%` 的稳定 NDCG@10 信号，但该范围只作
研究目标，不作结果后选择门，也不得为了达到该数字反复扫描配置。最终是否有论文价值由跨域稳定性、
效应区间、强基线、消融、效率和跨 backbone 泛化共同决定。

---

## 3. 第一轮：GACR-v7 全量 fit 指标对齐损失

### 3.1 研究假设

在 v6 全量 fit records 与其它因素不变时，仅以 NDCG@10/Recall@50 截断敏感的加权 pairwise
logistic loss 替换 highest-negative hinge，可保留 v6 的 Toys 增量，并减少 Beauty 在 top-10、
top-50 边界附近的错误交换。

### 3.2 唯一改动因素

对每个 covered fit record，按冻结 GRAM base score 的 stable rank 得到 target 位置 `r_t` 与负例
位置 `r_j`：

`D10(r)=1/log2(r+1)`（仅 `r<=10`，否则 0）；`R50(r)=1`（仅 `r<=50`，否则 0）。

`w_j=|D10(r_t)-D10(r_j)|+0.25*|R50(r_t)-R50(r_j)|`。

令 `s_i=base_i+residual_i`，每 record loss 为：

`sum_j w_j*softplus(s_j-s_t)/sum_j w_j`。

零权重 record 不进入 group 均值；head/tail 两个 group 的有效 record 均值等权平均。`0.25`、
loss 形式和截断位置运行前冻结，不做网格搜索。

### 3.3 严格冻结项

- GRAM C1 checkpoint、候选构造、stable tie-break、原 6 维特征；
- `BoundedResidualRanker(6,16,bound=0.2)`、identity initialization、residual scale=`1.0`；
- v6 全量 fit split、80/20 fit/calibration 用户隔离、calibration 128 head + 128 tail；
- AdamW、lr=`0.01`、weight decay=`0.01`、gradient clip=`10`、30 个 full-batch steps；
- seeds 2023/2024/2025、GRAM backbone optimizer steps=`0`；
- 不使用 v4/v5 gate 或 multiplier，不改模型容量、候选、训练步数或 loss 权重。

### 3.4 数据与对照

- Toys、Beauty；每域 1024 位新 fresh development 用户；salt：
  `phase6-gacr-v7-full-fit-metric-loss-development-v2`；
- 排除 GCDH train/validation 与 GACR-P0、v2、v3、v4、v5、v6 的全部 historical cohort；
- 对照：原始 GRAM、冻结 GACR-v3、全量 hinge GACR-v6、GACR-v7；
- Sports/test 禁读；fresh validation label 不得进入训练、特征、loss 权重或配置选择。

### 3.5 校准资格门

v7 没有可调参数。calibration 只检查：

- loss、gradient、checkpoint 和输出全部 finite；
- parent checkpoint SHA 不变、backbone optimizer steps=`0`；
- 每个域-seed broad harm `<=1%`；
- 相对 GRAM 的 overall Recall@10/50 不低于 `-0.2pp`；
- 相对 GRAM 的 tail Recall@50 不低于 `-0.4pp`，tail NDCG@10 绝对差不低于 `-0.0005`。

这里的非劣界约对应 overall 每 1024 用户最多净损失 2 个命中、tail 子群最多约 2 个命中。它们在
新 cohort 运行前冻结，只用于避免明显伤害，不等价于证明等效。任一域-seed越界则 v7 不进入 fresh
validation；不得调 `0.25`、步数、margin 或重新抽 calibration 用户救援。

### 3.6 Fresh-validation 双层决定

**增长资格（决定是否出现 v3 以上信号）**须全部满足：

1. six-cell macro NDCG@10 严格高于 v3；
2. 至少 4/6 个 cell 的 v7-v3 NDCG@10 为正；
3. Toys、Beauty 域均值中不得有一个低于 v3 超过 `0.5%` 相对值；
4. 相对 v3 的 overall Recall@10/50 不低于 `-0.2pp`，tail Recall@50 不低于 `-0.4pp`，
   tail NDCG@10 绝对差不低于 `-0.0005`；
5. 每 cell broad harm `<=1%`，完整性门全部通过。

**incumbent 替换资格**在增长资格之外，还要求：

- v7-v3 的跨 seed 用户级配对 bootstrap macro NDCG@10 95% CI 下界大于 0，或两域各自至少
  2/3 seed 为正且两域均值均为正；
- 两域相对 GRAM 的 mean NDCG@10 均至少 `+1%`；
- 各 guardrail delta 的配对 95% CI 下界不得低于对应非劣界。

通过增长资格但未通过替换资格：v3 仍为部署 incumbent，v7 作为 v8 的研究 parent；通过两层资格：
v7 成为新 incumbent 和 v8 parent；未通过增长资格：保留 v3，但仍执行预先定义、机制不同的 v8。

### 3.7 v7 停止规则

- v7 失败后不调 `0.25`，不尝试 LambdaRank/ListMLE/SoftNDCG 的结果后邻近搜索；
- 不回到 gate、attenuation、soft multiplier；
- 不因 v7 失败取消 v8，因为 v8 检验的是不同的 score interface/list interaction 假设；
- 非完整性错误不自动重跑；任何重跑必须保持科学配置并由研究者明确授权。

---

## 4. 第二轮：GACR-v8 Path-aware Listwise Residual Calibration

### 4.1 研究假设

当前 GACR 将 GRAM 生成分数压缩为 `1/beam_rank`，丢失真实路径概率幅度；同时逐 item MLP 看不到
候选之间的相对结构。若在固定候选 union 上恢复真实 item-path likelihood 和 prefix uncertainty，
再用小型 listwise residual ranker 学习截断边界交换，应比 rank-only residual 更稳定地改善 top-10。

### 4.2 预先冻结的方法边界

v8 必须保留以下接口，详细数值超参数在 v7 分析完成后写入独立 v8 预注册文件，并在运行任何 v8
fresh cohort 前冻结：

1. **真实 item-path score**：对 GRAM beam item 使用生成 sequence score；对 catalog-only item
   使用冻结 GRAM teacher forcing 计算其完整 lexical-ID log likelihood；统一采用预冻结的长度归一化；
2. **prefix 统计**：至少保存逐层 log-probability、最弱 prefix margin、prefix entropy、完整 path
   score 与 beam 内 score gap；所有统计只来自目标发生前的输入和冻结 GRAM；
3. **item-space normalization**：在同一用户候选列表内分别标准化 generative path score 与 catalog
   score，不把 catalog 标量直接注入 token logit；
4. **有界 listwise residual**：identity initialization，最终 residual 有界；候选间最多使用 1–2 层
   小型 self-attention/set encoder，参数预算必须在预注册中给出，并报告推理开销；
5. **指标对齐训练**：沿用 v7 的 top-10/top-50 截断意识与 head/tail 平衡，不在 v8 cohort 上搜索
   loss family；
6. **候选集合固定**：v8 仍使用与 incumbent 对齐的 GRAM beam + catalog candidates，不在同一轮扩大
   candidate source，以保证“排序接口”与“候选覆盖”可辨识。

### 4.3 必须同轮运行的 nested ablation

| 臂 | 生成分数接口 | 排序器 | 目的 |
|---|---|---|---|
| A | 原始 GRAM | 无 residual | 基线 |
| B | rank-only | 当前 incumbent | 冻结 GACR 对照 |
| C | rank-only | v7 loss/model | 区分 loss 收益 |
| D | path-aware | 与 C 同容量的逐 item residual | 隔离真实 path score 收益 |
| E | path-aware | 有界 listwise residual | 隔离列表交互收益 |

若 v7 未通过校准或没有有效 checkpoint，C 使用冻结 v3 架构重新按 v7 loss 在 v8 的 train-only fit
split 训练，但不得读取 v7 fresh outcome 调参。A/B/D/E 仍必须保留。D 与 E 共享同一 path features、
候选、fit records、优化预算和 loss；E 相对 D 的唯一结构变化是 candidate interaction。

### 4.4 v8 数据与统计

- 使用 Toys、Beauty 的新 fresh development cohort；排除至 v7 为止全部历史开发用户；
- 每域至少 1024 用户、3 seeds；若基于历史方差的运行前 power audit 显示无法识别预期增量，则在
  不读取新标签的前提下提高样本量，而不是先跑小样本再追加；
- 主要比较 E-vs-B；D-vs-C 检验 path score，E-vs-D 检验 list interaction；
- 主要指标为 NDCG@10；Recall@10/50、tail NDCG@10/tail Recall@50、changed coverage、broad
  harm、union coverage、延迟和显存为预注册次要指标；
- 使用逐用户配对 bootstrap；同一用户的三个 seed 先取均值，再按用户重采样，不能把 3072 行当作
  独立用户扩大显著性。

v8 沿用 v7 的非劣界作为最低安全门。只有 E-vs-B 的双域 macro NDCG@10 为正、至少 4/6 cell
为正、两域无超过非劣界的伤害，才称为“固定候选 GACR 仍可增长”；替换 incumbent 仍要求更强的
配对区间或两域多 seed 一致性证据。

### 4.5 v8 终止判定

- **成功**：冻结 E 的全部方法、超参数和 checkpoint；停止继续在 Toys/Beauty 上做相邻模型搜索，
  进入论文规模验证准备；
- **仅 D 成功**：保留 path-aware pointwise residual，停止 listwise 容量增长；
- **仅 E-vs-D 成功但 E 不超过 incumbent**：列表交互有机制信号但不足以保留，固定候选主线仍关闭；
- **D/E 均无增量或产生超界伤害**：正式关闭固定候选 GACR 增长主线，执行第 5 节 fallback；
- 不允许以增加层数、hidden size、loss 网格、gate 或结果后用户分群开启“第三次最后机会”。

---

## 5. 两轮不成功后的主线：Candidate Drafting + GRAM Verification

该主线的目标不是继续重排同一个 union，而是突破候选覆盖上限。既有 RPCD/FCRD 证明外部序列
模型可提供约 3pp 的互补 Recall@50 coverage，CCRR calibration 证明候选级条件排序具有较强拟合
信号；旧失败主要来自固定融合和 tail harm，而不是候选完全没有互补性。

### 5.1 F0：无训练 coverage/oracle 审计

只读取开发数据和已有候选产物，不读取 Sports/test，回答：

- target 不在当前 GRAM+catalog union 的比例；
- target 在 union 但未进入 top-10 的比例；
- GRAM、catalog、SASRec/sequence drafter 各自独占命中和交集；
- 固定候选预算下 union oracle NDCG@10/Recall@10/50 上限；
- head/tail、短/长历史、低/高置信用户的 coverage 缺口。

若额外候选源在两个域都没有可重复的独占 coverage，停止 candidate drafting，转向重新训练生成
backbone/identifier alignment 的独立研究计划；不得强行进入融合实验。

### 5.2 F1：多源 drafter 资格实验

候选源至少包括冻结 GRAM beam、GRAM catalog projection 和一个独立 sequence/full-catalog drafter。
本阶段只评价候选，不训练最终排序器：

- 在同一候选预算下报告 union Recall@50、独占 coverage、重复率和延迟；
- 资格目标：相对当前 union，两域 overall Recall@50 均至少 `+2pp`，且至少一域达到 `+3pp`；
- 两域 tail Recall@50 均至少 `+1pp`，不得通过移除原 GRAM candidates 换取；
- 所有候选构造必须 target-free，未知/重复 item 和历史 item 过滤必须可审计。

资格失败则不训练 verifier；优先诊断 drafter 表示与召回，而不是在同一 calibration 上搜索更多融合权重。

### 5.3 F2：GRAM item-level verifier

通过 F1 后才训练 verifier：

- 对所有 drafted candidates 计算冻结 GRAM teacher-forced lexical-ID likelihood；
- 联合 path score、drafter score、catalog score和用户/item表示，在 item space 做有界 listwise ranking；
- 训练目标对齐 NDCG@10/Recall@50，并保持 head/tail 平衡；
- 必须并列 `GRAM`、最强单 drafter、naive score fusion、当前 GACR incumbent、draft+verifier；
- 必须拆分“coverage gain”和“ranking realization rate”，不能把候选 oracle 当最终收益；
- 主要保留条件是 draft+verifier 在双域超过 incumbent，同时兑现一部分新增 coverage 且不产生超过
  非劣界的 head/tail harm。

### 5.4 fallback 的创新与比较边界

该方向应表述为“多源候选 drafting 后由冻结生成器作 item-level semantic verification，并以指标对齐
listwise calibration 兑现 coverage”，不能仅声称“把两个推荐列表取 union”。正式论文实验至少需要
加入与 graph-logit augmentation、direct item projection、generative ranking loss、draft-and-verify
范式相对应的强基线；具体文献表在 F1 实现前单独更新并冻结。

---

## 6. 论文规模验证门

v7/v8 或 fallback 方法只有在 Toys、Beauty 上冻结后才能进入论文规模验证：

1. Sports 作为一次性 confirmation domain，只允许在方法、checkpoint 选择规则、主要指标和统计分析
   全部冻结后读取一次；
2. test split 继续封存到开发方法完全冻结；
3. 最好补充 Yelp 或另一个公开数据集，并至少在第二个 generative recommendation backbone 上验证
   plug-and-play 性；
4. 报告 3–5 seeds、逐用户配对区间、绝对值与相对值、效率、显存、参数量和完整消融；
5. `+4%–6%` 是健康的开发目标，不是发表保证；跨数据集稳定的 `+2%–4%` 加上低开销和跨 backbone
   泛化，可能比单域较大但不稳定的收益更有价值；
6. 不把 v3 attenuation、v4 gate 或 v5 soft weighting 作为有效贡献；论文中的 incumbent 应称为
   “GACR core / frozen GACR configuration”，因为这些后处理因素最终都选择了 identity。

## 7. 产物、执行与治理

### 7.1 v7 预期产物

- implementation: `experiment/phase6/gacr_v7.py`
- tests: `experiment/phase6/test_gacr_v7.py`
- runner: `experiment/phase6/run_phase6_gacr_v7.sh`
- config: `artifacts/phase6/configs/gacr_v7_preregistered.json`
- outputs: `artifacts/phase6/gacr_v7/`
- report: `report/第六阶段/GRAM_第六阶段_GACR-v7结果与验证报告.md`

v8 与 fallback 必须在各自启动前另写详细、可执行的子计划与冻结配置，本文件只定义不得结果后改变的
研究问题、方法边界、对照和阶段门。

### 7.2 资源与运行规则

- GPU0 总租约=`30,720 MiB`；每个子实验在实现后按实际 smoke peak 预注册 workload/sidecar 分配；
- 单元测试至少覆盖 loss、非劣界、identity、path-score teacher forcing、nested ablation 对齐、cohort
  排除、Sports/test 禁读和逐用户输出；
- 运行前冻结 implementation/test/runner/config/input checkpoint SHA256；
- 停止 CodeLlama、通过显存门后在具名 tmux 启动；长实验 hard timeout 在子计划中按 smoke 实测冻结；
- 所有退出路径恢复 CodeLlama，scientific exit 与 resource restoration 分开记录；
- 非零 scientific exit、timeout 或完整性失败不自动重试；
- 每轮完成后只保存结果。未经研究者明确请求，不自动分析、不实现或启动后继、不读取 Sports/test。

## 8. 当前唯一获准的下一步

当前只允许实现和运行 **GACR-v7 全量 fit 指标对齐损失**。GACR-v8 和 fallback 已预先定义，但
尚未获准实现或启动。v7 结束后，由研究者根据本文件第 3.6 节要求分析，再明确决定后续动作。

### 8.1 v7 实现冻结记录

- 冻结时间：`2026-08-02T19:11:14+08:00`；
- v7 implementation、tests、runner 与 preregistered config 已按第 3 节实现；
- 启动前必须通过 CPU 单测、语法检查、GRAM C1/v3/v6 checkpoint SHA256 与本计划 SHA256 门；
- 本次授权仅覆盖 v7；运行完成后不得自动分析、实现 v8、启动 v8 或读取 Sports/test。
