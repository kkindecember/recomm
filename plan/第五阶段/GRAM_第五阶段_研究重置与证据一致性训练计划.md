# GRAM 第五阶段：研究重置与证据一致性训练计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate + plan
- Created: 2026-07-29
- Verification Status: ANALYZED（M0、Rank-R0、Rank-R0G、Rank-R1）；PLANNED（后续研究）
- Version Label: `phase5_reset_cet_v1`
- Upstream Audit: `artifacts/phase5/M0_search_process_audit.md`
- Development Domains: Toys、Beauty
- Confirmation Domain: Sports（继续封存）

## 0. 这次重置什么

第三、第四阶段最大的优点是证据链干净，最大的缺点是把研究流程做成了“高维串行
否决器”。第五阶段不降低完整性要求，但改变科学筛选方式：

1. 不再用一个 proxy deficit 是否超过任意阈值决定方法有没有资格训练；
2. 不再要求每个早期 pilot 同时证明 overall、tail、机制、校准、成本和跨域确认；
3. 不再一天换一个方向；本阶段只开发一个方法族；
4. 先用足够分辨率的真实 recommendation effect 决定去留，再用机制分析解释；
5. 论文目标从“必须找到绝对首创组件”改为“清楚的问题、可归因的方法、稳定效果和
   新的实证结论”。

## 1. 对过去失败的正式解释

过去不是二十个完整方法都训练失败，而是二十多个候选假设中，大多数在以下位置结束：

- 新颖性或 premise threshold；
- frozen checkpoint 的 teacher-forced proxy；
- 小样本 learnability/calibration；
- 双域多条件合取；
- 512-user 左右的低功效 effect gate。

真正改变可学习参数并读取 recommendation effect 的方法数量远少于方向数量。
因此第五阶段不继续堆 N0/N1 deficit audit，而进行一次有预算的 constructive method
development。

## 2. 唯一主方向：CET

方法暂名：

> **CET：Counterfactual Evidence-consistency Training**
> （反事实证据一致性训练）

### 2.1 研究问题

GRAM 的 coarse history passage 与多个 fine item passages 在 decoder 中共同融合。
既有 FPUG-N1 只说明“单独移除某个 passage 时 gold-path utility 有异质性”，但
FPUG-P0 进一步说明：在推理期预测并执行 per-user gate 会发生跨域迁移失败。

CET 不预测“该删哪一条”，也不改变推理输入。它把 passage variation 当作训练期
结构化扰动，要求模型在部分 detailed evidence 缺失时仍保持与 clean full-evidence
view 一致的合法 Trie-child 分布：

```text
clean view x:
  coarse passage + all current fine passages

perturbed view x~:
  coarse passage always kept
  newest fine passage always kept
  remaining fine passages independently masked with fixed probability q

p_t = softmax(z_clean[C_t] / tau)
r_t = softmax(z_perturbed[C_t] / tau)

L = CE_clean
  + alpha * CE_perturbed
  + beta * KL(stopgrad(p_t) || r_t)
```

`C_t` 是当前 gold prefix 下的合法 Trie children。正式推理只运行原始 clean view，
因此不增加推理参数、token、beam、reranker 或第二次解码。

### 2.2 为什么这不是 FPUG rescue

- FPUG 学的是 target-free deterministic gate，并在推理期改变 evidence；
- CET 不学习 gate，不使用 FPUG 的 Toys/Beauty validation 差异选动作；
- CET 的独立依据来自 consistency regularization、structured dropout 与 RAG
  context-perturbation training；
- CET 的目标是让同一个 generator 对合法 catalog continuations 在证据子采样下稳定，
  而不是预测某 passage 的正负 utility。

相关工作边界：

- [R-Drop](https://arxiv.org/abs/2106.14448) 已提出 dropout 子模型之间的输出一致性；
- [Passage-Mask](https://aclanthology.org/2022.emnlp-main.260/) 已研究 passage
  masking 作为 reader regularization；
- [CORD](https://aclanthology.org/2025.naacl-short.66/) 已研究 RAG context
  perturbation 下的一致性与 rank preservation；
- CET 不声称上述组件首创。候选新增点仅限于：
  **GRAM 类 multi-granular generative recommendation 中，clean-anchored structured
  passage subsampling 与 Trie-local legal-child consistency 的组合及其跨域机制证据。**

### 2.3 为什么它比继续做 selector 更有成功先验

1. 不需要 target-free features 准确预测每个 passage 的 utility sign；
2. 不在推理期做可能伤害 Beauty 的 hard routing；
3. clean CE 始终存在，正式输入与 baseline 完全相同；
4. consistency 是训练期正则，允许从所有样本学习，而不是只覆盖小 active subset；
5. 最小方法只有一个扰动机制和一个 Trie-local KL，容易做 matched ablation。

## 3. 新的证据门分层

### 3.1 Integrity gate

只判断执行是否有效：

- coarse passage 永不被 mask；
- newest fine passage永不被 mask；
- target 不进入扰动策略；
- clean view 与原 GRAM input 完全一致；
- `beta=alpha=0` 精确复现 matched CE；
- legal-child membership、finite、gradient、checkpoint reload 全通过；
- validation/test/Sports 不用于训练或配置选择。

Integrity 失败只能修实现，不能作科学 STOP。

### 3.2 Optimization gate

只判断 CET 是否真的被优化：

- 在 fit-disjoint training-prefix calibration 上，clean/perturbed legal-child
  symmetric KL 相对 matched C0 下降；
- clean lexical CE 不发生预注册的灾难性恶化；
- 不要求先达到某个“baseline defect prevalence”。

Optimization 失败才关闭当前实现。

### 3.3 Development effect gate

Toys/Beauty 是 development evidence，不再伪装成两个独立确认实验。主判据为：

1. 两域 NDCG@10 的等权平均相对变化 `>= +1%`；
2. 两域中至少一域 NDCG@10 `>= +1%`，另一域不得低于 `-0.5%`；
3. 两域 Recall@10 的绝对变化均不得低于 `-0.2pp`；
4. 合并的 tail NDCG@10 不得下降超过 `0.5% relative`；
5. broad-harm rate 相对 matched C0 增加不得超过 `0.5pp`。

不再要求两个数据集、overall、tail 的每个置信区间下界全部同时大于 0。置信区间继续
完整报告，用于决定证据强度和下一阶段样本量，不作为低功效 pilot 的连环否决器。

### 3.4 Confirmation gate

只有 development effect 通过才解封 Sports。Sports 使用冻结方法、超参数和 seed
规则；此时才要求：

- primary NDCG@10 提升的 95% CI 下界不低于 0；
- 至少 3 seeds 或由 pilot 方差得到的更大 seed 数；
- matched compute baseline 与关键消融完整；
- test 只在 checkpoint selection rule 冻结后读取一次。

## 4. M0：测量系统审计（已完成）

结果见 `artifacts/phase5/M0_search_process_audit.md`。最关键的定量结论是：

- FPUG Toys overall 在 512 users 上点估计 `+2.37% relative`，但只有 29 个用户
  NDCG@10 发生变化；
- 按观测配对差方差，约需 1,196 users 才能在相同点效应下使近似 95% 区间下界高于 0；
- 因此第五阶段 effect cohort 不再固定为 512。

M0 固定决定：
**`RESET_SEARCH_PROCESS_AND_START_ONE_METHOD_CYCLE`**。

## 5. CET 渐进实验

### C0：文献与代码边界

只做一次有界审查，不设“零重叠才通过”：

- 确认没有工作已经同时覆盖 GRAM multi-granular passages、structured evidence
  subsampling、gold-prefix legal-child consistency 和 clean-only inference；
- 若组件已有，收窄主张而不是自动停止；
- 只有整体问题、方法和实验协议均被实质覆盖才停止。

### C1：correctness smoke

- 每域 32 个 training-prefix users；
- 从 baseline checkpoint 出发，只更新 decoder 最后一层；
- 固定 `q=0.25, tau=1, alpha=1, beta=0.1`，不扫描；
- 5 optimizer steps；
- 检查 clean identity、mask localization、finite gradient、KL 下降和 reload。

C1 不读取 validation，不作效果结论。

### C2：高信息量 effect pilot

训练：

- C0：matched lexical-CE continuation；
- C1：CE clean + CE perturbed（控制 augmentation）；
- C2：完整 CET；
- 每域同一 baseline checkpoint、同一 25% training users、同一 batch/step；
- 5 epochs，seed 2023；
- 超参数只用 fit-disjoint training-prefix calibration 锁定；
- 不增加第四个候选，不做网格搜索。

评测：

- validation cohort 每域至少 2,048 users；若推理资源允许，直接使用完整 validation；
- cohort salt 在读取 candidate target 前冻结；
- baseline/candidate paired full-catalog Trie beam-50；
- 报告 overall、head/tail、history bins、broad harm、latency、显存与逐用户差；
- primary bootstrap 单位为 user，训练稳定性由后续 seeds 单独回答。

决定：

- 完整 CET 通过 3.3 节：`CET_FREEZE_FOR_CONFIRMATION`；
- CET 不优于 augmentation control：`STOP_CET_NO_CONSISTENCY_VALUE`；
- 两者都无效：`STOP_CET_NO_REGULARIZATION_EFFECT`；
- 工程无效：`INVALID_RUN_FIX_AND_EXACT_RERUN`。

### C2-O：optimization evidence audit

C2 run2 未通过原 development effect gate，但双域点估计均为正、完整 CET 高于
augmentation control，且除两个收益阈值外的安全门均通过。因此允许一次不改变 C2
结论、不读取新 validation target、不训练新模型的 optimization audit：

- 只使用现有 C0/C1/C2 checkpoint 与 fit-disjoint training-prefix calibration；
- 使用与正式训练相同的 `q=0.25, tau=1` 和冻结 mask seed 规则；
- 分别报告 clean/perturbed legal-child symmetric KL、clean lexical CE、
  competitive legal-child coverage、实际 masked-passage coverage；
- 主比较为 C2 相对 C1 的 symmetric KL 降幅；C0 作为 matched clean-CE 参照；
- 按 Toys、Beauty 分域报告，并报告等权平均；不使用 subgroup 或单个 batch 决定；
- validation/test/Sports 均继续封存，现有 C2 validation 结果不得用于选择 calibration
  样本、阈值或超参数。

C2-O 决定：

- 两域 C2 symmetric KL 均低于 C1，且等权平均相对下降 `>= 5%`，同时 clean CE
  相对 C1 恶化不超过 `1%`：`CET_C2O_OPTIMIZATION_PASS`；
- KL 有限、实现有效但未满足上述门：`STOP_CET_WEAK_OPTIMIZATION_SIGNAL`；
- checkpoint、样本隔离、mask、legal-child 或 finite 检查失败：
  `INVALID_C2O_FIX_AND_EXACT_RERUN`。

`5%` 是在启动 C2-O、读取其 calibration 输出前冻结的机制分辨率阈值；它只决定是否
值得付出多 seed development 成本，不追溯改变 C2 的 effect 结论。

### C2-V2：增强一致性强度的独立方法周期

2026-07-30 协议修订：C2-O 仍保持
`STOP_CET_WEAK_OPTIMIZATION_SIGNAL`，不追溯改判。鉴于 CET-v1 是既有方法探索中
唯一同时满足双域 recommendation 点估计为正、C2 高于 augmentation control、全部
安全门通过且跨域 symmetric KL 同向下降的方法，研究者明确选择继续同一方法族，
建立 **CET-v2**。CET-v2 是新的 development 方法，不是放宽 CET-v1 的原门槛。

CET-v2 只改变一个因素：

```text
CET-v1: beta = 0.1
CET-v2: beta = 0.3
```

选择 `beta=0.3` 的依据是训练日志中 `beta=0.1` 的 KL 加权项仅占总 loss 约
2.5%--2.8%，而 C2-O 的跨样本 KL 改善方向一致但幅度弱。三倍权重预计仍明显小于
clean CE 与 perturbed CE 两个主项。该值在任何 CET-v2 calibration/validation 输出
产生前冻结；不扫描其他 beta，不改变 `q=0.25, tau=1, alpha=1`、clean anchor、
Trie-local support、训练预算或正式推理。

#### V2-A：calibration-B optimization smoke

- 从尚未用于 C2 training、C2 validation、C2-O 或 GCDH frozen validation 的
  training-prefix users 中，按新 salt 冻结每域 256 users；
- 每域固定拆为 128 fit users 与 128 evaluation users，二者 user-disjoint；
- V1/V2 都从 matched source checkpoint 出发，只更新 decoder 最后一层；
- 同一 fit users、batch order、mask signature、50 optimizer steps；
- V1 使用 `beta=0.1`，V2 使用 `beta=0.3`，其余配置完全相同；
- optimization gate 只在未参与 50-step 更新的 evaluation users 上读取；
- validation recommendation target、test、Sports 全部禁止读取。

V2-A gate：

1. 两域 V2 symmetric KL 均低于 V1；
2. V2 相对 V1 的双域等权平均 symmetric KL 降幅 `>=5%`；
3. 两域 V2 clean lexical CE 相对 V1 恶化均不超过 `1%`；
4. coarse/newest identity、target-free mask、matched updates、finite/nonzero
   gradient、checkpoint reload、fit/evaluation disjoint 与 source SHA 全部通过。

决定：

- 全部通过：`CET_V2A_OPTIMIZATION_PASS`；
- 工程有效但 optimization/safety 门失败：`STOP_CET_V2_STRENGTHENING_FAILED`；
- 工程或隔离失败：`INVALID_V2A_FIX_AND_EXACT_RERUN`。

#### V2-B：全训练与全新 development effect

只在 V2-A 通过后执行：

- 使用 C2 相同的 seed 2023、25% training users、5 epochs、batch/order/update、
  `q/tau/alpha`、mask seed 与 final-epoch checkpoint rule；
- 只新增 Toys/Beauty 两个 CET-v2 (`beta=0.3`) 训练臂；已有 C0/C1/CET-v1
  checkpoint 保持冻结，不重训；
- 在训练启动前按新 salt 冻结每域 2,048 个 fresh development users；必须排除
  C2 train users、C2 原 2,048 validation users、C2-O users、V2-A users、全部
  GCDH 4,096 validation users、test 与 Sports；
- 在 fresh cohort 上统一评估 C0、C1、CET-v1、CET-v2，使用相同的 full-catalog
  Trie beam-50；cohort selection 不读取 candidate target；
- 旧 C2 validation 只保留为 CET-v1 历史证据，不参与 V2-B 判定。

V2-B development gate：

1. CET-v2 双域等权平均 NDCG@10 相对 C0 变化 `>=+1%`；
2. CET-v2 的 macro NDCG@10 同时高于 C1 与 CET-v1；
3. 至少一域 NDCG@10 `>=+1%`，另一域不得低于 `-0.5%`；
4. 两域 Recall@10 绝对变化均不得低于 `-0.2pp`；
5. pooled-tail NDCG@10 相对变化不得低于 `-0.5%`；
6. 每域 broad-harm rate 不得超过 `0.5pp`；
7. integrity、资源和 target-sealing 检查全部有效。

决定：

- 全部通过：`CET_V2_FREEZE_FOR_MULTISEED`；
- CET-v2 不高于 CET-v1：`STOP_CET_V2_NO_INCREMENTAL_VALUE`；
- 其他 effect/safety 门失败：`STOP_CET_V2_NO_DEVELOPMENT_EFFECT`；
- 工程或隔离失败：`INVALID_V2B_FIX_AND_EXACT_RERUN`。

V2-B 通过后才新增 seed 2024/2025；三个 seeds 稳定通过后才进入 C3 与 Sports。
CET-v2 开发不改变 CET-v1/C2-O 的既有负向记录。

### CET-Rank：beam-ranking 对齐的一致性机制

2026-07-30 第二次协议修订：CET-v1 的 recommendation 正向点估计继续保留，
C2-O 与 V2-A 的负向决定继续有效。V2-A 表明简单放大同一个 gold-prefix KL 权重
不能实质增强跨样本一致性；下一假设改为：

> CET 的主要瓶颈可能不是 loss 权重，而是 gold-prefix local-child KL 与正式
> full-sequence Trie beam ranking 错位。

候选新机制暂名 **CET-Rank**。对 training-prefix input，由冻结 source checkpoint
分别在 clean/perturbed view 生成 top-4 candidates，冻结两者 union `S(x)`（最多
8 items），在完整 identifier sequence score 上构造一致性：

```text
S(x) = top4_beam(x_clean) union top4_beam(x_perturbed)

p = softmax(length_normalized_sequence_score_clean[S(x)] / tau_r)
r = softmax(length_normalized_sequence_score_perturbed[S(x)] / tau_r)

L = CE_clean + CE_perturbed + gamma * JS(p || r)
```

候选生成只使用 training-prefix input，不读取 recommendation validation target；
正式推理仍只运行原始 clean view。`gamma` 不通过 validation 扫描，而在
training-prefix fit cohort 上按梯度比例冻结，使 rank-JS 初始梯度约为 CE 梯度的
10%。候选集、梯度规则与所有用户 hashes 必须在 effect evaluation 前冻结。

#### Rank-R0：surrogate-alignment audit

R0 是只读测量，不训练新模型：

- 每域冻结 128 个全新 training-prefix users，排除 C2 train/validation、C2-O、
  V2-A、全部 GCDH frozen validation、test 与 Sports；
- 审计现有 C0/C1/C2 checkpoints，三者使用同一 users、`q=0.25`、mask seed 与
  clean/perturbed inputs；
- 对每个 user 报告 gold-prefix symmetric KL、beam-20 clean/perturbed top-10
  overlap、union rank displacement、target rank shift、是否实际 mask；
- 无 mask 时 clean/perturbed ranking 必须精确一致；candidate mapping 必须 100%；
- 主分析只在实际 mask users 上进行，分别报告 rank-instability prevalence，以及
  gold-prefix KL 与 rank displacement 的 Spearman correlation；
- R0 不把相关性当作效果结论，只用于选择下一 surrogate。

R0 路由：

- 两域 rank-instability prevalence 均 `>=10%`，且 `|Spearman rho|<0.2`：
  `CET_R0_RANK_SURROGATE_MISMATCH`，允许进入 Rank-R1；
- 两域 instability prevalence 均 `<5%`：
  `STOP_CET_NO_MEANINGFUL_RANK_INSTABILITY`；
- 其他情况：`CET_R0_MIXED_ALIGNMENT_REVIEW_REQUIRED`，先人工审查，不自动训练；
- 工程、隔离、mapping 或 no-mask identity 失败：
  `INVALID_R0_FIX_AND_EXACT_RERUN`。

#### Rank-R0G：local-KL / direct-rank 梯度对齐审计

2026-07-30 根据 Rank-R0 的 mixed 结果新增该计划。R0G 是新的只读机制审计，
**不训练、不更新参数，也不事后改变 Rank-R0 的机器决定**。它回答：

> gold-prefix legal-child KL 与 direct sequence-rank JS 是否实际推动 decoder
> 最后一层朝相同方向更新？

如果两种 loss 的梯度高度同向，则 direct-rank 很可能只是重复同一信号，继续
Rank-R1 缺乏依据；如果 direct-rank 梯度稳定非零且与 local-KL 低对齐或冲突，
才说明它提供了当前 CET 没有利用的新优化方向。

样本与隔离：

- Toys/Beauty 各冻结 64 个全新 training-prefix users，按独立 salt 的 user hash
  排序选取；
- `target=sequence[-3]`，`history=sequence[:-3][-20:]`，minimum history 为 2；
- 排除 GCDH train/validation、C2 validation、C2-O、V2-A、Rank-R0 及所有此前
  development/evaluation users；
- 只读取 training-prefix target；recommendation validation、test 与 Sports
  target 继续封存；
- 仅审计 C1 checkpoint。C1 是 Rank-R0 的预注册 routing control，避免再把
  checkpoint 选择变成研究者自由度。

view 与 candidate construction：

- 使用与 Rank-R0 相同的 structured passage mask：`q=0.25`，coarse passage 与
  newest fine passage 强制保留，mask seed 在机器预注册中冻结；
- 主分析只包含至少一个旧 fine passage 实际被 mask 的 users，要求每域至少 24
  个，否则决定为 insufficient coverage；
- 在 `torch.no_grad()` 下分别生成 clean/perturbed top-4 Trie beams，取去重 union，
  最多 8 个 candidates；candidate union 随即 detach 并固定；
- union 不注入 positive target，不依据 loss、梯度或结果替换 candidate；
- full-sequence score 使用 teacher forcing，累加 identifier token 至 EOS 的
  log-probability，再按有效预测 token 数做 length normalization；
- 对 clean/perturbed 的 union scores 以冻结温度 `tau_r=1.0` softmax，计算
  symmetric Jensen--Shannon divergence，记为 `L_rank`。

梯度测量：

- `L_local` 沿用 Rank-R0 的 gold-prefix legal-child symmetric KL；
- 分别对 `L_local` 与 `L_rank` 调用 `autograd.grad`，只提取 decoder 最后一层
  trainable tensors；每个 loss 均从相同 C1 参数、相同 user、相同 views 独立求导，
  不调用 `optimizer.step()`；
- 每个 masked user 记录 `L_local`、`L_rank`、两者梯度 L2 norm、norm ratio、
  cosine similarity、dot-product sign、candidate union size 与 mapping 状态；
- 每域报告 nonzero-gradient coverage、median cosine、mean cosine、cosine
  bootstrap 95% CI、negative-cosine prevalence、median norm ratio；
- bootstrap 固定 5,000 次 user-level resamples，仅量化不确定性，不扫描门槛。

R0G 完整性门：

1. checkpoint、代码、配置、split、user/file hashes 在 GPU 审计前冻结；
2. 两域 candidate mapping 为 100%，union size 在 `[2,8]`；
3. no-mask clean/perturbed candidate、score、loss 与梯度 identity 检查通过；
4. mask 决策 target-independent，同一 user 的两种 loss 使用完全相同 views；
5. 所有 loss/gradient/cosine finite，审计前后 checkpoint SHA 不变；
6. validation/test/Sports target read flags 全为 false；
7. 不得因结果改 `top-4/tau_r/q/layer/sample size`，失败后不得自动重跑。

R0G 机器路由：

- 任一域 masked users `<24`，或 `L_rank>1e-6` / direct-rank nonzero-gradient
  coverage 任一项 `<90%`：
  `STOP_CET_RANK_NO_USABLE_GRADIENT`；
- signal 与 integrity 门全部通过，且两域 median cosine 均 `<0.2`、两域
  user-bootstrap 95% CI upper 均 `<0.3`：
  `CET_R0G_DIRECT_RANK_GRADIENT_DISTINCT`，允许另行预注册 Rank-R1；
- 两域 median cosine 均 `>=0.5` 且 negative-cosine prevalence 均 `<10%`：
  `STOP_CET_RANK_GRADIENT_REDUNDANT`，关闭 CET consistency 方法族；
- 其余有效结果：
  `CET_R0G_MIXED_GRADIENT_REVIEW_REQUIRED`，不得自动进入训练；
- 工程、隔离、SHA、mapping、identity 或 sealing 失败：
  `INVALID_R0G_FIX_AND_EXACT_RERUN`。

R0G 预计只运行 C1 的双域各一个审计臂，原计划 GPU3，2026-07-30 最新资源修订后
改用物理 GPU6，预算上限 2 小时。实施时沿用
tmux、状态 JSON、5 秒 GPU telemetry、单臂 hard timeout、CodeLlama
stop/restore 与“崩溃不自动重跑”规则。明日顺序为：审查本节门槛 → 实现与单元测试
→ 冻结机器配置和 SHA → 展示唯一启动命令 → 获得明确确认后运行。

#### Rank-R1：CET-Rank correctness smoke

只在 R0 直接支持 surrogate mismatch，或 R0G 得到
`CET_R0G_DIRECT_RANK_GRADIENT_DISTINCT` 后另行预注册执行：

- 每域 64 fit + 64 evaluation users，user-disjoint；
- frozen clean/perturbed top-4 union，最多 8 candidates；
- 只更新 decoder 最后一层，20--50 steps；
- 在 fit users 上按预注册梯度比例冻结 `gamma`，不读取 evaluation 结果调参；
- evaluation rank-JS 相对初始化下降 `>=10%`；
- clean lexical CE 恶化不超过 `1%`；
- top-10 overlap 提升，finite/nonzero gradient、candidate mapping、reload 与
  target sealing 全通过。

Rank-R1 机器预注册补充（在读取任何 Rank-R1 fit/evaluation 输出前冻结）：

- 每域按新 salt 冻结 64 fit + 64 evaluation users，排除至 Rank-R0G 为止的全部既有
  development/audit users，fit/evaluation user-disjoint；
- 固定 32 optimizer steps、batch size 2，seed-2023 顺序下每个 fit user 恰好使用一次；
- fit/evaluation 共用 source C1 上预先生成并写盘冻结的 clean/perturbed top-4 union，
  candidate bank 在 gamma calibration 和 optimizer update 前冻结；
- `gamma = 0.1 * median(||g_(CE_clean+CE_perturbed)|| / ||g_rank-JS||)`，中位数只在
  实际 mask 的 fit users 上计算，不裁剪、不扫描；
- evaluation primary 为实际 mask users 的 mean rank-JS；两域分别要求相对初始下降
  `>=10%`，all-user clean lexical CE 相对恶化 `<=1%`，masked-user mean top-10
  set-overlap absolute change 严格 `>0`；
- fit/evaluation 实际 mask users 各域均须 `>=24`；所有门逐域通过才允许 R2；
- 物理 GPU6、每域 hard timeout 2 小时、30,720 MiB 最低空闲显存门、tmux/status/
  telemetry、CodeLlama stop/restore 与不自动 retry 规则保持不变。

决定：

- 全部通过：`CET_R1_RANK_CONSISTENCY_PASS`；
- 优化/安全门失败：`STOP_CET_RANK_NOT_OPTIMIZABLE`；
- 工程失败：`INVALID_R1_FIX_AND_EXACT_RERUN`。

#### Rank-R2：fresh development effect

只在 R1 通过后执行：

- C0、C1、CET-v1 checkpoint 保持冻结；只新增 CET-Rank；
- CE 作用于全部 training samples；rank-JS 只作用于 hash 冻结的 25% samples；
- 每个选中样本最多 8 candidates，每域总预算不超过 8 GPU 小时；
- 使用全新双域各 2,048-user development cohort，排除所有既有 development users；
- primary gate 沿用双域 macro NDCG@10 `>=+1%`；
- CET-Rank macro 必须高于 C1 与 CET-v1；
- Recall、pooled tail、broad harm 与 integrity 沿用原安全门；
- changed-user coverage 必须高于 CET-v1 的 `29/2048`，否则标记为仍然过度稀疏。

R2 通过后才运行 seed 2024/2025、机制消融与 Sports；否则关闭 consistency 方法族。

### C2b：多 seed 稳定性裁决

只在 C2-O 通过后执行。C2b 是新的 development 实验，不是把原 C2 事后改判通过：

- 方法与 `q/tau/alpha/beta`、5 epochs、batch、学习率、checkpoint rule 全部不变；
- 新增两个训练 seed，与 seed 2023 合计 3 seeds；
- 每个新增 seed 在 Toys/Beauty 上完整运行 matched C0/C1/C2，保持同 seed 的训练
  users、顺序、更新数和 C1/C2 mask signature 可比；
- 优先从原 2,048-user cohort 之外冻结新的、target 尚未读取的 validation users；
  若剩余合法用户不足，则原 cohort 只能用于 training-seed robustness，不能宣称为
  新的独立 holdout；
- 不扫描 `q/beta/tau`，不增加 epoch，不依据 seed 2023 或当前 Beauty 结果选择配置；
- user bootstrap 描述给定 checkpoint 的用户不确定性，晋级以 seed-level 稳定性为
  核心，不把用户数当作训练重复。

C2b development gate：

1. 三个 seeds 的双域等权平均 NDCG@10 相对 C0 变化 `>= +1%`；
2. 至少 `2/3` seeds 的双域 macro NDCG@10 满足 C2 `>` C1；
3. seed-mean 中至少一域 NDCG@10 `>= +1%`，另一域不得低于 `-0.5%`；
4. 两域 seed-mean Recall@10 绝对变化均不得低于 `-0.2pp`；
5. pooled-tail NDCG@10 相对变化不得低于 `-0.5%`；
6. 每域每 seed broad-harm rate 不得超过 `0.5pp`；
7. integrity、C2-O optimization 与资源记录全部有效。

C2b 决定：

- 全部门通过：`CET_C2B_FREEZE_FOR_MECHANISM_AND_CONFIRMATION`；
- 工程有效但任一 development 门失败：`STOP_CET_AFTER_MULTISEED_ADJUDICATION`；
- 工程或隔离失败：`INVALID_C2B_FIX_AND_EXACT_RERUN`。

只有 C2b 通过，才允许进入 C3；Sports 与 Toys/Beauty test 在 C3 完成并冻结最终方法
前继续封存。

### C3：机制消融

只在原 C2 直接通过，或本次预注册的 C2-O 与 C2b 均通过后执行：

1. 去掉 Trie-local restriction，改全词表 KL；
2. 去掉 clean anchor，只做双扰动一致性；
3. uniform token dropout control；
4. fixed oldest-passage dropout control。

论文机制必须由完整 CET 超过这些对照支持；否则把贡献降级为实证 regularization
结果，不夸大机制。

### C4：Sports 与多 seed 确认

- 方法、`q/tau/alpha/beta`、训练预算、checkpoint rule 全冻结；
- Sports 为主要确认域；
- seed 数由 C2 的 run-level 方差估计，最低 3；
- 通过后才运行 Toys/Beauty test，并按冻结配置回报所有正负结果。

## 6. 统计与报告原则

1. development 阶段看效应大小、方向一致性和伤害上界，不用 `p<.05` 代替判断；
2. 不把用户数当作训练重复；user bootstrap 与 seed variance 分开；
3. 不再用任意 subgroup 的轻微负值否掉 overall 有意义且安全的方法；
4. tail 是重要 secondary endpoint，但只有明确以 tail 为主张时才作为 primary；
5. 所有候选和失败保留，但第五阶段在 CET 关闭前不另起方法 M/N/O。

## 7. 资源与执行协议

- 当前物理 GPU6，`CUDA_VISIBLE_DEVICES=6`；
- GPU 任务继续使用 tmux、status JSON、5 秒 GPU telemetry；
- 启动前停止 CodeLlama reservation，退出路径恢复；
- 不静默 retry，不删除 invalid run；
- C1 预计低于 20 分钟；C2 每域/配置预算上限 8 GPU 小时；
- C2 总预算在启动前锁定，禁止因中间 loss 好看追加 epoch。

## 8. 当前状态

- M0：**完成**
- CET-C0：**`CET_C0_PASS_WITH_TRANSFER_NARROWING`**，机制说明见
  `artifacts/phase5/cet_c0/mechanism_brief.md`
- CET-C1：机器预注册已冻结为
  `artifacts/phase5/configs/cet_c1_preregistered.json`
- CET-C1 实现与预检：5/5 单元测试、真实双域 Collator/mask 集成检查、Toys
  真实模型 CPU forward/backward micro-smoke 均通过；记录见
  `artifacts/phase5/cet_c1/preflight_summary.json`
- CET-C1 GPU correctness smoke：**`CET_C1_CORRECTNESS_PASS`**。Toys/Beauty
  分别使用 32 个 training-prefix users 和 5 optimizer steps；Trie-local KL
  分别下降 5.79%/13.56%，clean CE 分别下降 8.62%/8.56%。clean replay、coarse/
  newest passage identity、target-free mask、zero-weight identity、legal-child、
  finite/nonzero gradient、source SHA、reload 与 validation/test/Sports exclusion
  全部通过。结果见 `artifacts/phase5/cet_c1/summary.json`。
- CET-C2 已冻结：机器预注册见
  `artifacts/phase5/configs/cet_c2_preregistered.json`；双域各 2,048 个配对
  validation users，split 文件与 user-set SHA 均已冻结；三臂固定为 C0 clean CE、
  C1 clean+perturbed CE、C2 完整 CET，5 epochs、seed 2023、无 checkpoint
  selection、无超参数扫描。
- CET-C2 初始工程运行：**`INVALID_RUN_FIX_AND_EXACT_RERUN`**。原因是 smoke 抽到
  不含可遮蔽旧 fine passage 的短历史，`masked_passages=0` 却错误标为 PASS。
  在读取 validation/test/Sports 前已终止，证据原样保存在
  `artifacts/phase5/cet_c2/invalid_run.md`。
- CET-C2 run2：修正版 smoke 强制覆盖遮蔽路径，Toys/Beauty 均实际 mask 5 个
  passages 且 finite/Trie competitive-step 检查通过；正式训练的冻结 `q=0.25`
  未改变。独立输出根为 `artifacts/phase5/cet_c2_run2`。
- CET-C2 run2 已于 2026-07-29 完成六个训练臂与双域各 2,048-user 配对
  validation。C2 相对 C0 的 Toys/Beauty NDCG@10 分别为 `+0.8193%` /
  `+0.9983%`，双域等权平均 `+0.9088%`；C1 为 `+0.3621%`。Recall、pooled
  tail、broad harm 与全部 integrity checks 通过，但 macro `>= +1%` 和至少一域
  `>= +1%` 两个冻结收益门未通过。原机器决定保持为
  **`STOP_CET_NO_REGULARIZATION_EFFECT`**，科学解释为“安全、双域正向但效应小且
  未达到预注册晋级标准”，不得追溯改判。
- C2-O/C2b：基于 C2 的近门槛、C2 高于 C1 与安全性证据，已在读取 C2-O 输出和启动
  新训练前冻结一次有界延续。它不修改原 C2 结论；C2-O 不通过即关闭 CET，只有
  C2-O 通过才启动两个新增 seeds 的 C2b。
- CET-C2-O：**`STOP_CET_WEAK_OPTIMIZATION_SIGNAL`**。Toys/Beauty 各使用 256 个
  fit-disjoint training-prefix calibration users，且与 C2 training users 和全部
  4,096 个冻结 validation users 隔离。C2 相对 C1 的 legal-child symmetric KL
  分别下降 `2.0675%` / `2.8243%`，双域等权平均下降 `2.4459%`，未达到冻结的
  `5%` optimization 门；clean CE 相对 C1 分别变化 `+0.2821%` / `+0.1946%`，
  通过 `<=1%` 安全门。两域 finite、checkpoint SHA、相同 mask、相同 calibration
  users、target-free 与 validation/test/Sports 封存检查全部通过。结果见
  `artifacts/phase5/cet_c2o/summary.json`。
- C2-O wrapper 在科学汇总写出后发生 EXIT-trap 局部变量作用域错误；未影响六臂
  数值或科学决定。状态记录已校正，CodeLlama reservation 已恢复，脚本退出清理已
  修复供未来使用；未重跑任何审计臂。
- CET-v2：研究者在完整保留 CET-v1 与 C2-O 负向决定的前提下，授权一个单变量
  `beta=0.3` 的独立方法周期。V2-A 使用全新 fit/evaluation-disjoint
  calibration-B，只有通过 5% KL 改善与 1% clean-CE 安全门才启动 V2-B；V2-B
  只新增两个训练臂，并在全新、未读 target 的双域 cohort 上比较
  C0/C1/CET-v1/CET-v2。
- CET-V2-A：**`STOP_CET_V2_STRENGTHENING_FAILED`**。Toys/Beauty 各使用全新的
  128 fit + 128 evaluation training-prefix users，排除 C2 training、C2
  validation、C2-O 与全部 GCDH frozen validation users。50-step matched smoke
  中，`beta=0.3` 相对 `beta=0.1` 的 evaluation symmetric KL 分别下降
  `0.3817%` / `0.7450%`，双域等权平均仅下降 `0.5634%`，未达到冻结的 `5%`
  optimization 门；clean CE 分别变化 `+0.0832%` / `+0.0215%`，安全门通过。
  两域 V2 KL 均低于 V1，但增强幅度过弱。finite/nonzero gradient、parameter
  change、checkpoint reload、source SHA、matched fit/evaluation users、matched
  fit/evaluation masks 与 target sealing 全部通过。结果见
  `artifacts/phase5/cet_v2a/summary.json`。
- CET-Rank：已授权先执行只读 Rank-R0。R0 只判断 gold-prefix KL 是否与
  beam-ranking instability 错位，不训练新方法；只有 R0 双域支持 surrogate
  mismatch 才实现 Rank-R1。
- Rank-R0 机器预注册已写入
  `artifacts/phase5/configs/cet_rank_r0_preregistered.json`；实现与运行入口分别为
  `experiment/phase5/cet_rank_r0.py` 和
  `experiment/phase5/run_phase5_cet_rank_r0.sh`。冻结代码 SHA256 为
  `5f4c4880618095f37c26bb533906b1333297346e58995f57a08136f5fd0071f4`；
  六个输入 checkpoint SHA 已逐一核对，静态检查与 Rank-R0 指标单元测试通过。
- 新计划：**已建立**
- 当前运行状态：**`CET_RANK_R0_COMPLETED`**。运行时间为 2026-07-30
  00:35--00:50（Asia/Shanghai），六个审计臂成功完成，CodeLlama reservation
  已恢复。
- Rank-R0 机器决定：**`CET_R0_MIXED_ALIGNMENT_REVIEW_REQUIRED`**。路由控制 C1
  在 Toys/Beauty 的 masked-user rank-instability prevalence 均为 `100%`，但
  gold-prefix KL 与 union rank displacement 的 Spearman rho 分别为
  `0.5298` / `0.5053`，未满足预注册的 `|rho|<0.2` mismatch 条件；事后配对
  bootstrap 的 95% 区间分别为 `[0.3263, 0.6869]` /
  `[0.3058, 0.6692]`，说明未过门并非单纯样本波动。
- C2 相对 C1 的 gold-prefix KL 分别下降 `2.6395%` / `3.8136%`，但 top-10
  overlap 分别变化 `-0.0057` / `-0.0015`，union rank displacement 分别变化
  `+0.0503` / `+0.0750`；后两项配对 bootstrap 区间均跨 0。当前证据支持
  “local KL 与 ranking instability 有信息关联，但现有训练对 beam ranking 的
  干预传递弱”，不支持把 local KL 判为完全错误的 surrogate。
- integrity：双域 same-users、candidate mapping、no-mask identity、
  target-independent mask 与 target sealing 全部通过；validation/test/Sports
  未读。结果见 `artifacts/phase5/cet_rank_r0/summary.json`。
- Rank-R1 当前保持 **未授权/未启动**。若继续 CET，下一候选应先预注册一个小型
  **Rank-R0G gradient-alignment audit**：在全新 training-prefix users 上比较
  gold-prefix KL 与 direct sequence-rank JS 对同一 decoder block 的梯度 cosine、
  norm ratio 和非零覆盖，不做 optimizer update。只有双域 direct-rank 梯度有效，
  且与 local-KL 梯度低对齐或冲突，才允许把 direct-rank objective 作为独立方法
  周期进入 correctness smoke；若梯度高度同向，则关闭 CET consistency 方法族，
  避免重复优化同一信号。
- 当前科学决定：
  **`CET_R0_MIXED_ALIGNMENT_REVIEW_REQUIRED_NO_R1_AUTO_START`**
- Rank-R0G 门槛审查已完成：计划中的 signal、distinct、redundant 与 integrity 门槛
  保持不变；`usable gradient` 已在机器配置中机械化为 masked users 中
  `L_rank>1e-6` 覆盖率与 decoder-last-layer gradient norm `>1e-12` 覆盖率均
  `>=90%`，避免运行后解释自由度。
- Rank-R0G 计划状态：**`R0G_PREFLIGHT_FROZEN_AWAITING_EXPLICIT_START`**。实现、
  运行包装器与单元测试已创建；12 项 R0G/Rank-R0/C2-O 纯函数及回归检查通过，Shell
  与 Python 静态检查通过。机器预注册见
  `artifacts/phase5/configs/cet_rank_r0g_preregistered.json`，冻结 code SHA256 为
  `80056fa8327bf811e67fd8bde670c8219b834fbe9acb0dea20643e4580bf8616`，config
  SHA256 已随 GPU6 执行修订重新冻结为
  `fc57f260df8d29d1d31481ea08cad4c6e68a77718ab4285bbb6f3564388b4c26`；
  双域 C1 checkpoint SHA 已核对。
- Rank-R0G 新用户已按冻结 salt 生成：Toys/Beauty 各 64 人，分别与 10,995 / 11,842
  个既有排除用户零重叠；user-set SHA256 分别为
  `39bbb3fe36e7369b28d6518f67fcbf7143b23ed7b93831075a6c9285977e775a` /
  `c14dfb4ff1eeadb88acdaab4ca438238c333f45567ca3955a9de2c40879ed54d`。
  validation/test/Sports target 均未读，GPU 未占用。
- 下一动作：展示并由研究者明确确认唯一启动命令
  `bash experiment/phase5/run_phase5_cet_rank_r0g.sh start`。确认前不得运行；启动后
  不自动重试，完成后按冻结机器门生成终局路由。
- 2026-07-30 15:28 的已确认启动在进入任何审计臂前被 GPU3 资源门阻止：停止
  CodeLlama 后仍未达到冻结的 `30,720 MiB` 最低空闲显存，包装器以 exit 3 退出、
  未自动重试并执行恢复路径。事后只读快照显示 GPU3 有外部 Python 进程占用约
  `30,808 MiB`，空闲仅 `17,752 MiB`；该外部进程未被修改或终止。事件记为
  **`R0G_START_BLOCKED_GPU3_RESOURCE_GATE_NO_AUDIT`**，不是科学 STOP/INVALID
  决定；无 per-user 审计输出，validation/test/Sports 仍未读。记录见
  `artifacts/phase5/cet_rank_r0g/blocked_start_20260730.md`。下一次精确启动必须先确认
  GPU3 满足原资源门，并重新获得研究者明确确认。
- 2026-07-30 GPU 资源修订：研究者先指定 GPU7，随后以最新指令覆盖为物理
  **GPU1**；Rank-R0G 及后续 CodeLlama reservation 均以 GPU1 为最终目标。该修订
  只改变执行设备，不改变用户、checkpoint、loss、梯度、bootstrap、路由门槛或
  30,720 MiB 最低空闲显存门；GPU3 阻塞事件保持为历史记录。机器配置与包装器已
  相应更新并须重新冻结 config SHA。
- 2026-07-30 最新 GPU 资源修订：研究者以最新指令将物理 **GPU1** 覆盖为
  **GPU6**；Rank-R0G 与退出后 CodeLlama reservation 均转至 GPU6。本次仍只改执行设备，
  不改变用户、checkpoint、loss、梯度、bootstrap、路由门槛或 30,720 MiB 资源门；
  既有 GPU3 阻塞事件及 GPU1 修订均保留为历史记录。
- Rank-R0G 已于 2026-07-30 21:07--21:11（Asia/Shanghai）在物理 GPU6 完成，退出后
  CodeLlama reservation 已恢复到 GPU6。机器决定为
  **`CET_R0G_DIRECT_RANK_GRADIENT_DISTINCT`**：Toys/Beauty 的 masked users 为
  `34/33`，`L_rank>1e-6` 覆盖率与 direct-rank 非零梯度覆盖率均为 `100%`；
  median gradient cosine 分别为 `0.0882` / `0.1614`，mean-cosine user-bootstrap
  95% 区间分别为 `[0.0515, 0.2186]` / `[0.0299, 0.2741]`，双域均通过
  distinct 门。candidate mapping、union size、no-mask identity、same views、target-free mask、
  finite、checkpoint SHA 不变与 target sealing 全部通过；validation/test/Sports 未读。
  结果见 `artifacts/phase5/cet_rank_r0g/summary.json`。该结果当时只授予 Rank-R1
  “另行预注册”许可，未自动进入训练。
- Rank-R1 已完成独立预注册、实现与 CPU preflight，状态为
  **`R1_PREFLIGHT_FROZEN_AWAITING_EXPLICIT_START`**，尚未运行 GPU。代码/config/split
  manifest SHA256 分别为
  `7ea46d35eeb36e51310672fc0d759d804523379fbebcbc7bc5d39cf66b3d4f95` /
  `b48a5ee3324941531d1ae106b5273ffa41e85d865f797cc9846aced4771f29c9` /
  `aece5f0e3ed81491d46c760044337eb2382a5f14ca4d60daf4a7ba1525c3b128`。
  Toys/Beauty 各 64 fit + 64 evaluation users 与八类既有排除文件重叠均为 0，域内
  fit/evaluation 重叠为 0；双域 C1 checkpoint SHA 已核对。12 项历史回归测试与
  2 项 Rank-R1 测试通过，Python/Shell/JSON 静态检查通过。预注册记录见
  `artifacts/phase5/cet_rank_r1/preregistration.md`。唯一启动命令为
  `bash experiment/phase5/run_phase5_cet_rank_r1.sh start`；启动前必须获得研究者
  对该精确命令的明确确认，失败后不得自动重试。
- Rank-R1 已于 2026-07-30 21:29--21:43（Asia/Shanghai）在物理 GPU6 完整执行，
  CodeLlama reservation 已在退出后恢复到 GPU6。执行状态成功，机器科学决定为
  **`STOP_CET_RANK_NOT_OPTIMIZABLE`**。Toys/Beauty 的冻结 gamma 分别为
  `33.0465` / `137.2267`；masked fit users 为 `38/39`，masked evaluation users
  为 `28/33`。evaluation masked-user rank-JS 相对下降仅 `0.1938%` / `1.9307%`，
  均低于 `10%` 门；clean CE 相对变化为 `+0.0806%` / `+0.0231%`，均通过 `<=1%`
  安全门；masked-user top-10 overlap absolute change 为 `0` / `-1.5152pp`，均未过
  严格提升门。双域 candidate mapping、union size、candidate/gamma freeze、
  no-mask identity、finite/nonzero gradient、parameter change、reload、source SHA
  与 target sealing 全部通过；validation/test/Sports 未读。结果见
  `artifacts/phase5/cet_rank_r1/summary.json`。
- 按预注册路由，Rank-R2 **未授权且不得启动**；CET consistency 方法族在当前协议下
  关闭，不追加 step、不调整 gamma、不更换 cohort，也不以新 validation 结果 rescue。

本计划中的 CET-v1 与单变量 CET-v2 周期均已完成且不得改判。CET-Rank 是机制层面的
新假设，不是继续调大 `beta`。Rank-R0/R0G/R1/R2 给出终局决定前不新增其他方向，
Sports/test 保持未读。
