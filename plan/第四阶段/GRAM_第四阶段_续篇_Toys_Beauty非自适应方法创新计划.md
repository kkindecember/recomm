# GRAM 第四阶段续篇：Toys/Beauty 非自适应方法创新计划

## 0. 文档状态

- Version Label: `phase4_continuation_v1`
- Created: 2026-07-28
- Current Status: **`NEXT_N0_MECHANISM_REVIEW`**
- Development Domains: Toys、Beauty
- Confirmation Domain: Sports（继续封存）
- 历史实验账本：
  `plan/GRAM_第四阶段_方法创新与渐进实验计划.md`
- 治理协议：
  `artifacts/phase4/toys_beauty_validation_firewall.md`
- 历史证据矩阵：
  `artifacts/phase4/phase4_evidence_matrix.md`

本文只记录从 2026-07-28 起的新方向和决策。此前 GCDH、GACR、CHPR 等方向的详细
配置、结果和停止原因不再复制，以历史实验账本和证据矩阵为准。

## 1. 研究目标

继续在 Toys 和 Beauty 上研究 GRAM 的方法创新，但将“产生方法”与“读取 validation
结果”分离：

1. 新假设来自文献/理论、training-prefix-only 诊断、correctness 要求或外部证据；
2. Toys/Beauty validation 只用于对冻结方案做一次性证伪和验收；
3. validation 结果不得驱动同一方向的 feature、loss、weight、quota、seed、cohort、
   threshold 或结构补丁；
4. test 和 Sports 不参与方向选择。

## 2. 已继承的结论

以下内容视为边界条件，不在本计划中重复试验：

- 已完成方向均未通过预注册的 Toys/Beauty 双域 overall/tail effect chain；
- `CHPR-A0` 的固定结论为 **`STOP_CHPR_NO_PREFIX_RANKING_DEFICIT`**；
- 不继续通过 reranker、margin、negative quota 或相似 post-hoc 变体修补旧方向；
- Toys/Beauty 可以继续使用，但必须遵守非自适应 Validation Firewall；
- Sports 仅在最终方法和配置完全冻结后使用。

## 3. 数据与信息边界

| 数据/信息 | 当前用途 | 禁止用途 |
|---|---|---|
| Toys/Beauty training prefixes | 机制审计、训练、共享超参数校准、资源估计 | 访问 `sequence[-2]`/`sequence[-1]` |
| Toys/Beauty validation | 冻结方案的一次性 effect gate | 反向设计下一变体或事后调参 |
| Toys/Beauty test | 最终冻结评估 | 选方向、调试、early stopping、rescue |
| Sports | 最终独立确认 | 当前方法生成或筛选 |
| 历史 Phase-4 validation 结果 | negative-results ledger、排除已失败机制 | 针对最新失败逐项打补丁 |

## 4. 单个新方向的强制流程

### N0：独立机制与新颖性审查

在读取任何新的 Toys/Beauty validation 结果之前，形成一页 mechanism brief：

- 核心研究问题与可证伪假设；
- 方法为什么可能改变生成机制，而不只是增加容量；
- 假设的独立来源和可追溯证据；
- 与 GCDH/GACR/CHPR 及常见基线的实质区别；
- 最小关键消融；
- 预期成功模式、失败模式和停止条件。

只有同时满足“有独立机制依据、不是旧方向改名、可以最小化验证”才进入 N1。

### N1：Training-prefix-only premise audit

只使用训练前缀检查：

- 所需信号是否真实存在且样本量足够；
- 信号是否覆盖 Toys 和 Beauty，而非单域偶然；
- 输入、候选、Trie、mapping 和排除规则是否正确；
- 是否能用更简单的非学习基线解释；
- 资源和统计功效是否足以进入 pilot。

N1 不得读取新的 validation 指标。若 premise 不成立，方向直接关闭并记录原因。

### N2：冻结预注册

进入 validation 前必须一次性冻结：

- 方法结构、输入、loss、optimizer、训练步数/epoch、seed；
- 两域共享超参数及其仅基于 training prefix 的选择规则；
- baseline、ablation、overall/tail 指标与 bootstrap 单位；
- effect threshold、退化红线、双域合取规则；
- split salt、配置文件、代码 SHA、产物路径；
- 所有允许的结论及其对应下一动作。

预注册完成后才允许启动 N3。

### N3：一次性 Toys/Beauty validation effect gate

- 每个方向只进行一次锁定的 validation 读取；
- Toys 与 Beauty 默认按合取门判断；
- 通过：立即冻结方法，不再依据 validation 调参；
- 未通过：关闭该方向，不做同一证据驱动的 rescue；
- 工程错误只有在能够证明结果无效时才允许修复重跑，并必须保留审计记录。

### N4：确认与报告

仅当 N3 通过：

1. 运行冻结消融与必要重复；
2. 运行 Toys/Beauty test；
3. 最后解封 Sports 做独立确认；
4. 无论正负结果都写入 evidence matrix 和本计划。

## 5. 决策表

| 结果 | 固定动作 |
|---|---|
| N0 缺乏独立机制或新颖性 | `STOP_NO_INDEPENDENT_MECHANISM` |
| N1 双域 premise 不成立 | `STOP_PREMISE_FAILED` |
| N1 仅单域成立 | `STOP_NO_CROSS_DOMAIN_PREMISE`，除非 N2 前已有理论化单域声明 |
| N3 双域 effect gate 通过 | `FREEZE_FOR_CONFIRMATION` |
| N3 任一域 effect gate 失败 | `STOP_EFFECT_GATE_FAILED` |
| 发现可证明的工程无效性 | `INVALID_RUN_FIX_AND_EXACT_RERUN` |

禁止使用“接近阈值”“某个 seed 较好”“某个 post-hoc subgroup 显著”改变以上动作。

## 6. 记录规范

每个方向使用新的唯一代号，并至少产生：

```text
artifacts/phase4/<direction>/
├── mechanism_brief.md
├── preregistration.md
├── config.json
├── status.json
├── summary.json
└── decision.md
```

`status.json` 只记录运行状态；`summary.json` 记录冻结指标；`decision.md` 显式写出门槛、
结果、固定结论和是否触发下一阶段。

## 7. 当前执行队列

N0 已完成。选定新方向：

> **IALC：Inference-Aligned Legal-Child Learning。训练时只在当前 Trie prefix 的
> 合法 children 上计算条件似然，使训练支持集与 constrained inference 对齐。**

机制说明与检索边界见
`artifacts/phase4/ialc_n0/mechanism_brief.md`。N0 固定决定为
**`IALC_N0_PASS_TO_PREMISE_AUDIT`**。

当前开放动作是 IALC-N1：冻结 N1 配置并实现 training-prefix-only premise audit。
N1 不训练、不读取 validation/test、不解封 Sports；只有 N1 科学门通过才允许设计
IALC-S0 correctness smoke。

### 7.1 IALC-N1 实际结果

IALC-N1 已完成，完整性全部通过，未读取 validation/test/Sports。

| 数据集 | mean illegal mass | mean loss gap | large-mass sample rate | 固定门结果 |
|---|---:|---:|---:|---|
| Toys | 2.26% | 0.0238 | 17.19% | FAIL |
| Beauty | 4.53% | 0.0513 | 48.44% | FAIL |

两域平均 loss gap 均低于冻结门槛 0.10，large-mass sample rate 均低于 50%。固定决定
为 **`STOP_IALC_NO_SUPPORT_MISMATCH`**；不实现 IALC-S0，不降低门槛，不增加
post-hoc prior correction。

下一动作回到 N0，但不得从 IALC 数值构造 rescue。候选研究问题改为独立的
user-conditioned subtree semantic alignment：检查 decoder state 是否缺乏对目标
Trie child 所覆盖 item 子树的语义辨识，而不是继续研究非法 vocabulary mass。

### 7.2 新方向 LNDR

N0 已选择 **Lexical–Node Dual Readout (LNDR)**。它针对 GRAM native lexical token
在不同 Trie nodes 中大量复用、但 LM head 共享同一 token embedding 的结构问题，
拟将输出分解为 shared lexical score 与 prefix-specific low-rank node residual。

仅用 catalog 的初步结构审计显示：

- Toys：99.93% item path 含 reused token，87.51% path steps 使用复用 token；
- Beauty：100% item path 含 reused token，96.07% path steps 使用复用 token。

该方向与 IALC 的非法 vocabulary mass 无关，也不使用 validation。完整机制和
search-bounded novelty 见 `artifacts/phase4/lndr_n0/mechanism_brief.md`。

## 8. 当前停止点

- 当前状态：**`NEXT_N0_MECHANISM_REVIEW`**
- LNDR-N1 已按冻结配置完成，完整性通过，未读取 validation/test/Sports
- 下一里程碑：从独立文献/理论来源形成下一方向 N0 mechanism brief
- IALC 已在 N1 终止，未启动训练
- 尚未读取任何新的 validation/test/Sports 结果

### 8.1 LNDR-N1 正式结果

LNDR-N1 使用两域各 1,024 个 unique training-prefix users 和冻结的 GCDH-P0 C0
checkpoint，仅做前向审计；optimizer steps 为 0，参数 SHA 前后不变，mapping、
Trie、finite 与数据防火墙检查全部通过。

| 数据集 | competitive steps | eligible semantic nodes | centroid distance median | high-polysemy steps |
|---|---:|---:|---:|---:|
| Toys | 3,614 | 310 | 0.00864 | 0 |
| Beauty | 2,990 | 552 | 0.00978 | 0 |

N0 的 catalog token reuse 结构事实成立，但更严格的 N1 前提不成立：同一个 lexical
token 在同深度不同 parent 下复用时，其 descendant-item metadata centroid 并未达到
冻结的 0.10 语义距离。由于两域在第一条科学链已经失败，高多义 cohort 为空，后续
state-separation 与 shared-readout-deficit 链按冻结定义不可评估，也不得降低
polysemy threshold 重新构造 cohort。

固定决定为 **`STOP_LNDR_NO_NODE_POLYSEMY_DEFICIT`**。不设计 LNDR-S0，不把
same-depth 限制、metadata representation 或 descendant minimum 改成事后 rescue。
下一方向必须来自与 LNDR 数值无关的独立机制来源。

### 8.2 SCDL：独立 N0 与 N1

下一独立方向选择 **Sibling-Contrastive Discriminative Lexicalization (SCDL)**。
GRAM 原方法对每个 cluster 独立取最高 TF-IDF native token；SCDL 的研究问题是，
是否应在固定 hierarchy 的每个 sibling set 内联合分配 distinct native tokens，
同时优化 subtree representativeness 与 sibling contrast。该机制来自 GRAM 已发表
的 identifier construction objective，而不是 LNDR 的 N1 数值。search-bounded
novelty 与方法边界见 `artifacts/phase4/scdl_n0/mechanism_brief.md`。

SCDL-N1 只读取 catalog text 与已冻结 lexical IDs，未加载 checkpoint、interaction
targets、validation/test 或 Sports。首次 cache-path 启动与随后一个汇总字段错误均
未产生有效 summary；两次工程无效尝试及唯一字段修复已保存在
`artifacts/phase4/scdl_n1/invalid_run_audit.md`，最终精确重跑完整性通过。

| 数据集 | sibling sets | current nonpositive margin | improved set rate | mean margin gain | positive coverage gain |
|---|---:|---:|---:|---:|---:|
| Toys | 2,686 | 10.17% | 70.89% | +0.0222 | +6.09pp |
| Beauty | 3,173 | 8.67% | 64.10% | +0.0185 | +4.05pp |

联合 assignment 在两域均 100% feasible，且平均 representativeness 未下降；但现有
lexicalization 已使约 90% child 具有正 sibling margin。两域的 current deficit
均低于冻结的 25%，positive-margin coverage gain 也低于冻结的 20pp。故固定决定为
**`STOP_SCDL_NO_SIBLING_LEXICALIZATION_DEFICIT`**。

不进入 SCDL-S0，不根据本结果降低 deficit/coverage 门，不扩大 candidate top-K，
也不把 margin 的小幅提升改写成训练理由。当前动作再次回到独立 N0 mechanism review。

### 8.3 新方向 FPUG

N0 已选择 **Fine-grained Passage Utility Gating (FPUG)**。GRAM 的 coarse user
passage 已包含完整 history lexical IDs，但所有历史 item 的 fine-grained metadata
passages 仍直接进入 decoder cross-attention。FPUG 拟用 training-prefix
leave-one-detail-passage utility 学习 user-conditioned gate；coarse passage 永不
门控，gate 零偏置 identity 初始化，Trie、beam 与 lexical output 不变。

该方向来自 FiD context quality/causal evidence 文献与 sequential history denoising，
不是对 LNDR/SCDL 数值的修补。与 CF-SAT 不同，FPUG 不替换 passage 内的 CF
neighbor 内容，而是审计整个 item-detail passage 的条件效用。完整 N0 边界见
`artifacts/phase4/fpug_n0/mechanism_brief.md`。

当前已冻结 FPUG-N1：两域各 512 个 unique-user training-prefix samples，head/tail
各半，history 至少 5；只屏蔽一个 detailed passage 的 decoder cross-attention，
coarse lexical history 保持不变。只有双域 dynamic harmful-passage signal、tail、
recency coverage 及相对 fixed-oldest baseline advantage 全部过门，才允许设计 S0。

当前状态：**`FPUG_N1_PREREGISTERED`**。

### 8.4 FPUG-N1 正式结果

FPUG-N1 已按冻结配置完成。两域各 512 个 unique-user training-prefix samples，
coarse lexical history 在所有 counterfactual 中保持不变，每次只 mask 一个
fine-grained item passage 的 decoder cross-attention。完整性门全部通过，optimizer
steps=0，checkpoint SHA 不变，未读取 validation/test/Sports。

| 数据集 | harmful sample rate | tail harmful rate | best removal CE gain | oldest removal CE gain | oracle advantage | recency entropy |
|---|---:|---:|---:|---:|---:|---:|
| Toys | 69.14% | 70.70% | +0.1043 | −0.0418 | +0.1461 | 0.9766 |
| Beauty | 67.77% | 68.75% | +0.1218 | −0.0198 | +0.1415 | 0.9699 |

所有冻结科学门在两域均通过。harmful passages 覆盖全部四个 recency quartiles；固定
删除最旧 detailed passage 平均会恶化 gold legal-child CE，而 per-user oracle
removal 有稳定改善。因此该 signal 不能由简单 history truncation 解释。

固定决定为 **`FPUG_S0_DESIGN_ALLOWED`**。这只授权实现 correctness smoke，不授权
读取 validation 或修改 N1 门槛。当前已冻结 S0：验证 zero-gate identity、只门控
detailed passages、bounded gates、非零有限梯度、短程 loss 可优化、checkpoint
冻结与 reload identity。

当前状态：**`FPUG_S0_PREREGISTERED`**。

### 8.5 FPUG-S0 正式结果

S0 已使用两域各 8 个 training-prefix users 完成。只优化新增 gate 20 steps；
GCDH-P0 C0 backbone 全冻结，validation/test/Sports 未读。

| 数据集 | zero-logit diff | coarse-state diff | initial → final CE | relative decrease | trained gate range | reload diff |
|---|---:|---:|---:|---:|---:|---:|
| Toys | 0 | 0 | 0.5175 → 0.4212 | 18.60% | [0.5838, 1.4927] | 0 |
| Beauty | 0 | 0 | 0.4434 → 0.3801 | 14.28% | [0.5890, 1.3089] | 0 |

两域 zero identity、coarse identity、bounded gate、finite nonzero gradient、loss
decrease、trained nonidentity、reload、head/tail、Trie 与 backbone-freeze gates 全部
通过。固定决定为 **`FPUG_S0_CORRECTNESS_PASS`**。

S0 只证明实现正确和 gate 可优化，不构成 recommendation effect 证据。下一步进入
N2：冻结 fit/calibration split、gate architecture、训练预算、shared
training-prefix checkpoint-selection rule、validation cohort、overall/tail effect
gate、代码/config SHA 与固定结论；完成前不得读取 validation。

当前状态：**`FPUG_N2_PREREGISTRATION_DESIGN`**。

### 8.6 FPUG-P0 冻结设计

N2 已完成，正式配置见
`artifacts/phase4/configs/fpug_p0_preregistered.json`。每域使用 1,024 fit 与
256 calibration unique-user training-prefix samples；fit/calibration user-disjoint，
head/tail 各半且 history 至少 5。只训练 domain-specific gate 五个 epoch，两个域按
training-prefix calibration lexical CE 的平均相对下降共同选择同一个 epoch。

validation cohort 在读取 target 前按
`fpug-p0-validation-v1|dataset|user_id` 固定 hash 选 512 users/domain。baseline 与
FPUG 均使用 beam 50、相同 Trie/mapping/length penalty。双域均要求 overall 与 tail
NDCG@10 相对增益至少 1% 且 bootstrap 下界不为负，Recall@10 不降，baseline-hit
转 gated-miss 不超过 0.5%。

当前状态：**`FPUG_P0_PREREGISTERED`**。下一动作是先完成 training-prefix
fit/calibration 和 shared epoch 锁定，再执行唯一一次 validation read。

### 8.7 FPUG-P0 正式结果

FPUG-P0 已按冻结代码
`4c8c141a3794d1b84c250e36c12a982d4aca6996fc74039c663c14d01d4394ad`
完成。两域 fit/calibration user overlap 为 0，zero-gate logit diff 为 0，backbone
参数 SHA 不变且无梯度；训练前缀生成烟测、validation candidate mapping 与 finite
rate 均为 100%。未读取 test/Sports。两域 training-prefix calibration 平均相对
CE 改善按冻结规则选择共享 epoch 2。

| 数据集 | overall NDCG@10 相对变化（95% CI） | Recall@10 绝对变化 | tail NDCG@10 相对变化（95% CI） | baseline-hit→miss | 门结果 |
|---|---:|---:|---:|---:|---|
| Toys | +2.37% [−1.33%, +6.32%] | +0.39pp | +5.69% [+1.38%, +12.64%] | 0% | FAIL（overall CI） |
| Beauty | −5.87% [−11.36%, −1.25%] | −0.78pp | +0.29% [0%, +1.20%] | 0.78% | FAIL（4/6） |

Toys 的 point estimate 与 tail effect 为正，但 overall bootstrap 下界低于 0；
Beauty 的 overall NDCG/Recall 明确退化，tail point estimate 未达 1%，且
baseline-hit/gated-miss 超过 0.5%。因此双域合取门失败，固定决定为
**`STOP_FPUG_EFFECT_GATE_FAILED`**。不得根据 Toys 正向 subgroup、改用 epoch 1、
放宽 Beauty harm 门或改变 gate bound 做 validation-driven rescue；FPUG 不进入
test/Sports/确认阶段。

审计产物：

- `artifacts/phase4/fpug_p0/summary.json`
  SHA-256 `8b0af720745f64e85d79c649d9d6d2483c4dab2772c728a3aad431a5ff13f47f`
- Toys validation summary SHA-256
  `2972dae8654efc65e8374e4b67ccdd91ac2417888bea14935fd5ebd6a5e6ed3f`
- Beauty validation summary SHA-256
  `678540e4339808e5bf4895bc68c9e2e5c5aa0d812d0851f8e718183376ea45ff`

当前状态：**`NEXT_N0_MECHANISM_REVIEW`**。下一方向必须来自独立文献/理论机制，
不能把 FPUG 的 Toys/Beauty 差异用于构造门控 rescue。

### 8.8 新方向 TCDR

独立 N0 选择 **Tree-Coupling Decorrelation Regularization (TCDR)**。近期
generative recommendation 理论工作指出，单棵 autoregressive semantic-ID tree
会使 tree-near items 的跨用户概率响应结构性相关；其 Latte 修复通过新增 latent
route token 构造多棵树。直接复刻 latent token 缺少独立性，因此 TCDR 保留 GRAM
native lexical IDs、单 Trie 与推理过程，只考虑训练期对“tree-close 但
collaboratively dissimilar”item pair 的 excess cross-user score covariance 施加
校准正则。

TCDR 与 CHPR 的单用户 gold/negative first-divergence margin 不同：N1 测量固定 item
pair 在 128 个训练用户上的 exact path-score vectors；tree-far control 必须匹配两个
endpoint 的 train frequency bins 与低 collaborative cosine。完整机制与文献边界见
`artifacts/phase4/tcdr_n0/mechanism_brief.md`。

当前已冻结 TCDR-N1：两域各 128 个 training-prefix users、64 个 close/control
matched pairs；只有两域同时满足 close correlation、paired excess、positive-excess
rate 与 bootstrap 下界门，才允许设计 S0。N1 为 0-update audit，禁止读取
validation/test/Sports。

当前状态：**`TCDR_N1_PREREGISTERED`**。

### 8.9 TCDR-N1 正式结果

TCDR-N1 已按冻结代码完成。两域各使用 128 个 unique training-prefix users 和 64
组 close/frequency-matched-far pairs；checkpoint SHA 不变，optimizer steps=0，
mapping、Trie、finite、frequency-bin matching 与 unique-user rate 均为 100%，未读取
validation/test/Sports。

| 数据集 | close corr median | far corr median | paired excess median | positive excess | mean excess（95% bootstrap CI） |
|---|---:|---:|---:|---:|---:|
| Toys | 0.5366 | 0.3840 | +0.2188 | 65.63% | +0.1512 [+0.0449, +0.2477] |
| Beauty | 0.9355 | 0.3848 | +0.4951 | 95.31% | +0.5061 [+0.4166, +0.5897] |

五项冻结科学门在两域全部通过。即使 close pairs 的 train collaborative cosine 不超过
0.05，且 far controls 匹配 endpoint frequency bins，GRAM exact lexical path scores
仍表现出显著更强的跨用户相关性；Beauty 的结构绑定尤其强。

固定决定为 **`TCDR_S0_DESIGN_ALLOWED`**。该结果只证明 tree-coupling premise，
不证明 decorrelation loss 可微、可优化或能改善 recommendation effect。下一步仅
允许 training-prefix correctness smoke；在 S0 通过并完成 N2 前不得读取新的
validation。

审计 summary SHA-256：
`235bc0dd885568745eea8f07489b46dfefc4eeb3f3d028045dc3c9951af05f10`。

当前状态：**`TCDR_S0_PREREGISTRATION_DESIGN`**。

### 8.10 TCDR-S0 冻结设计

S0 已冻结为纯 correctness smoke：每域复用 TCDR-N1 的前 8 个训练用户与 pair
indices 0–7，不按 observed excess 挑 pair。encoder state detach，只有 decoder
最后一层可训练；以 differentiable legal-child exact path scores 计算 close
correlation 减 matched-far correlation 的正部均值，AdamW `1e-4` 优化 5 步。

两域均要求：`lambda=0` CE identity、初始 loss 正且有限、梯度非零有限、所有路径
分数/相关性有限、TCDR loss 至少下降 1%、lexical CE 相对上升不超过 10%、参数实际
变化且源 checkpoint SHA 不变。S0 不选择正式 loss weight，不新增 inference 参数，
不读取 validation/test/Sports。

当前状态：**`TCDR_S0_PREREGISTERED`**。

### 8.11 TCDR-S0 正式结果

TCDR-S0 已按冻结代码
`c14f0624dee03cb28327f97d083006a893c4bc9c76982faf951c13d84968d744`
完成。两域各使用 8 个 training-prefix users 和 N1 pair indices 0–7，仅更新 decoder
最后一层 5 步；所有路径分数、相关性与梯度有限，`lambda=0` identity diff=0，
参数总量保持 60,517,376，源 checkpoint SHA 不变，validation/test/Sports 未读。

| 数据集 | TCDR loss 初始→最终 | 相对下降 | lexical CE 初始→最终 | CE 相对变化 | 最大参数变化 |
|---|---:|---:|---:|---:|---:|
| Toys | 0.2716 → 0.0502 | 81.53% | 0.7538 → 0.7516 | −0.30% | 5.03e−4 |
| Beauty | 0.6128 → 0.2957 | 51.74% | 0.9820 → 0.9845 | +0.25% | 5.02e−4 |

两域 correctness conjunction 全部通过，固定决定为
**`TCDR_S0_CORRECTNESS_PASS`**。S0 证明去相关项能在不新增 inference 参数的前提下
通过原 decoder 反向传播并短程优化；它不构成 Recall/NDCG effect 证据。

summary SHA-256：
`4daefa1d30dfc7e644c698eb6c5b2046a4a726243840282881bd3831ca020a21`。

下一步进入 N2：冻结 matched CE continuation control、正式 TCDR loss weight/预算、
training-only mechanism calibration、锁定 validation cohort 与双域 effect gate。

当前状态：**`TCDR_N2_PREREGISTRATION_DESIGN`**。

### 8.12 TCDR-P0 冻结设计

N2 已完成。每域 C0/C1 从同一 GCDH-P0 C0 checkpoint 出发，使用相同 256 个
fit users、两 epoch、batch 16 与 decoder 最后一层更新；C0 只用 lexical CE，C1
固定使用 `CE + 0.1*TCDR`，每 batch 循环 4 个 N1 frozen pairs。另有 128 个
fit-disjoint training-prefix calibration users。

validation 前先执行双域机制门：相对 matched C0，C1 calibration mean paired
correlation excess 至少下降 10%，lexical CE 相对上升不超过 1%，完整性全通过。
任一域失败即 `STOP_TCDR_MECHANISM_GATE_FAILED`，不读取 validation。

机制门通过后，按 target-independent salt `tcdr-p0-validation-v1` 各锁定 512
validation users，C0/C1 使用相同 beam-50/Trie/mapping。双域均要求 overall/tail
NDCG@10 相对增益至少 1% 且 bootstrap 下界非负、Recall 不降、C0-hit/C1-miss
不超过 0.5%。

当前状态：**`TCDR_P0_PREREGISTERED`**。

### 8.13 TCDR-P0 正式结果

TCDR-P0 已按冻结代码
`9bcb578c6d52afbd4100e3ee70cef97f8835928d409997aca1802e96e1a0fa15`
完成 pre-validation 阶段。每个 dataset/control 均使用 256 fit users、128
fit-disjoint calibration users、两 epoch 和 32 matched steps；参数实际变化，
source checkpoint SHA 不变，mapping/Trie/finite 全通过。

| 数据集 | C0 excess | C1 excess | 相对下降 | C0 CE | C1 CE | CE 相对变化 | 机制门 |
|---|---:|---:|---:|---:|---:|---:|---|
| Toys | 0.08653 | 0.08359 | 3.40% | 0.844014 | 0.843975 | −0.0046% | FAIL |
| Beauty | 0.36127 | 0.35701 | 1.18% | 0.772005 | 0.772021 | +0.0021% | FAIL |

虽然 C1 在两域均保持 lexical CE，且 coupling excess 方向上略降，但降幅均低于冻结的
10% mechanism gate。故固定决定为
**`STOP_TCDR_MECHANISM_GATE_FAILED`**。程序按防火墙在 calibration 后终止：
`validation_read=false`，未构建 validation metrics，test/Sports 亦未读。

不得根据 S0 的微型 cohort 大降幅提高正式 `lambda`、增加 epoch、选择不同 pairs，
或把 10% 门降到 observed 3.40%/1.18%。TCDR 方向关闭，不进入 effect pilot。

审计产物：

- `artifacts/phase4/tcdr_p0/summary.json`
  SHA-256 `32c8e32e1887d4a0525501d39fcd177a74c1e742e20e2a926e3ad58b66658e84`
- `artifacts/phase4/tcdr_p0/prevalidation_summary.json`
  SHA-256 `f2972dcc748e6965b6a4fbba7381eab14db159d867a961f549fa39af3b7d7d59`

当前状态：**`NEXT_N0_MECHANISM_REVIEW`**。下一方向必须来自独立机制依据，不得将
TCDR 的正式降幅不足改写成更强正则或更大训练预算。

### 8.14 后续 N0 负筛选：Information-Gain Token Weighting

独立文献检索评估了“按 lexical hierarchy 的 candidate uncertainty reduction 对
token CE 做固定信息增益加权”。该想法推理期非自适应，也与 TCDR 数值无关，但
search-bounded novelty 不通过：

- 2026 年 *Token-Weighted Multi-Target Learning for Generative Recommenders*
  已直接提出 semantic-ID information-gain token weighting、front-greater 与
  frequency weighting；
- 2026 年 *Where Reasoning Matters* 已用 position information gain 分配
  semantic-ID token 的计算预算；
- LOHRec 已系统利用 generative recommendation 的 identifier order/hierarchy。

因此把信息增益 weighting 应用于 GRAM native lexical IDs 只是既有方法迁移，固定
决定为 **`STOP_N0_IG_WEIGHTING_PRIOR_ART_OVERLAP`**。不实现 N1、不读取数据或
validation。检索记录见
`artifacts/phase4/n0_screen_ig_weighting/rejection.md`。

当前状态保持 **`NEXT_N0_MECHANISM_REVIEW`**。

### 8.15 后续 N0 负筛选：词汇锚定与 Trie branching correction

首先检查 native lexical-ID token 的预训练语义锚定。训练-only structural probe
显示 ID tokens 相对 frequency-matched control 的 embedding cosine drift 比率较高：
Toys 3.75×、Beauty 4.44×；但绝对 drift 分别只有 `3.733e-5` 与 `4.817e-5`。
因此不能用相对比率放大一个数值上近乎不动的参数现象，固定决定为
**`STOP_N0_LEXICAL_ANCHOR_NO_ABSOLUTE_DRIFT`**。

随后检查按合法 Trie child 数做 deterministic path correction。2026 年 LBR 已明确用
valid Trie set 的 Hartley entropy `log2 |V_k|` 累积路径不确定性；Latte 与
information-budget allocation 亦直接覆盖 tree expressiveness / position
information gain。因此该方向固定为
**`STOP_N0_BRANCH_CORRECTION_PRIOR_ART_OVERLAP`**。

两项均未读取 validation/test/Sports。记录分别见
`artifacts/phase4/n0_screen_lexical_anchor/rejection.md` 与
`artifacts/phase4/n0_screen_branch_correction/rejection.md`。

### 8.16 新方向 CPIA

独立 N0 选择 **Cross-Passage Identifier Alignment (CPIA)**。GRAM 明确把同一 item
native lexical ID 在 coarse user prompt 与对应 fine item prompt 中的重复出现称为
information-linking bridge，但论文训练式只有 next-ID teacher-forced CE，并未显式
监督两处 contextual states 相互识别。

CPIA 候选 loss 在同一训练样本内，把 coarse 段中 item `i` 的 lexical-ID span 与其
对应 fine 段的同 ID span作为正对，其他 history item fine spans 作为负对，使用
contrastive alignment。正式推理不保留 loss，不改 prompt、ID、FiD layout、decoder
或 Trie，因而属于非自适应训练期机制。

新颖性边界是“GRAM 同一次 FiD 输入内部的 same-native-ID coarse→fine contextual
alignment”；不声称 contrastive alignment 首创，也不同于 RA-Rec 的 collaborative
ID embedding→LLM alignment、GENPLUGIN 的独立 language/ID views alignment 或
identifier-construction contrastive loss。完整边界见
`artifacts/phase4/cpia_n0/mechanism_brief.md`。

本地可观测性检查已确认实际 tokenized input 中，同一 5-token lexical ID 能在 coarse
段和其 fine 段精确定位，batch 保持 `[user, passage, 128]`，无需修改数据。

### 8.17 CPIA-N1 冻结设计

N1 已冻结为 0-update training-prefix audit：每域 128 个 deterministic unique users，
每用户 latest eligible sample，history 至少 5，只审计最近 5 个 item，共 640 个
coarse/fine span pairs。对 frozen encoder final states 做 span mean-pool 与 L2
normalize，构造每用户 5×5 coarse-to-fine cosine matrix。

双域各自必须同时满足：

1. top-1 matched-passage retrieval ≥30%，且 1,000 次 user bootstrap 95% CI 下界
   严格高于 chance 20%，证明 bridge 不是完全无信号；
2. top-1 ≤60%、median hard margin ≤0.05、mismatch rate ≥30%，证明 bridge 尚不
   可靠；
3. matched-minus-mean-mismatch 的 bootstrap 下界严格为正；
4. exact span mapping、attention validity、finite、checkpoint identity 均 100%，
   optimizer steps=0，validation/test/Sports 未读。

全部通过才是 **`CPIA_S0_DESIGN_ALLOWED`**；科学门失败固定为
**`STOP_CPIA_NO_ACTIONABLE_LINK_DEFICIT`**，完整性失败为 `EXECUTION_INVALID`。

代码在完整性计数错误审计与回归测试通过后锁定 SHA-256：
`05e99d3611d883d8c38ce3cdd5a038632eeb644bdb936b02304fdccd5883fda4`。
失效运行记录见 `artifacts/phase4/cpia_n1/invalid_run_audit.md`；该修复未改变
cohort、representation、metric、threshold、model input 或允许结论。

当前状态：**`CPIA_N1_PREREGISTERED`**。

### 8.18 CPIA-N1 正式结果

CPIA-N1 在修复仅影响完整性汇总的 paired-span 计数错误后按冻结配置精确重跑。
两域各使用 128 个 deterministic unique training users、640 个 coarse/fine span
pairs；mapping、attention、finite 与 checkpoint identity 全部通过，optimizer
steps=0，未读取 validation/test/Sports。失效尝试与唯一修复见
`artifacts/phase4/cpia_n1/invalid_run_audit.md`。

| 数据集 | top-1 accuracy（95% CI） | median hard margin | mismatch rate | matched−mean mismatch（95% CI） |
|---|---:|---:|---:|---:|
| Toys | 98.91% [98.13%, 99.69%] | 0.2576 | 1.09% | 0.3321 [0.3229, 0.3416] |
| Beauty | 79.53% [76.56%, 82.66%] | 0.0791 | 20.47% | 0.1386 [0.1342, 0.1430] |

两域 bridge signal 均显著高于 chance 20%，但也均超过冻结的 60% top-1 上限；
median hard margin 高于 0.05，mismatch rate 低于 30%。因此 GRAM coarse prompt
与对应 fine passage 的 repeated native lexical IDs 已具有很强 contextual
identifiability，不存在预注册所要求的 actionable weak-link deficit。

固定决定为 **`STOP_CPIA_NO_ACTIONABLE_LINK_DEFICIT`**。不进入 CPIA-S0，不根据
Beauty 相对较弱的数值提高 top-1 上限、放宽 mismatch 门或构造 post-hoc cohort。

审计产物：

- `artifacts/phase4/cpia_n1/summary.json`
  SHA-256 `6ae4b4b15256093d7d65d8e3f09fc60624d57c891f7f239fff3f56e402a906b5`
- `artifacts/phase4/cpia_n1/item_rows.csv`
  SHA-256 `f393cb3cc2bc5016d43d89b6cc5bd7d3d9fe4a37a5c05dd25862076599e4486c`

当前状态：**`NEXT_N0_MECHANISM_REVIEW`**。下一方向必须来自独立机制依据，不得将
CPIA 的 Toys/Beauty 强弱差异改写成 contrastive-loss rescue。
