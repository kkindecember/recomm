# GRAM 第三阶段：MARC L0 反事实效用与 critic 可学习性报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-07-24
- Verification Status: ANALYZED（单次修复后有效运行；未作独立复跑）
- Version Label: `marc_l0_v2_source_reference_repair`

## 固定结论

**`STOP_MARC_NO_UTILITY_HETEROGENEITY`**。

Toys 的 collaborative sample-level negative utility rate 为 **14.0625%**，低于
预注册的 15% 必要门槛，因此双数据集串行决策在 L0-A 停止。差距只有
0.9375 个百分点，但不得事后降低门槛。L0-B 也提供了独立停止证据：两个数据集的
semantic corruption sanity 都是错误方向，Beauty 的 semantic active coverage/utility
和 budget regret 也未达标。

因此 **L1、MARC-lite 训练、逐层融合、二次 reflection、RL/bandit、
validation/test 全部不解锁**。

## 执行范围与数据边界

- Toys/Beauty 各 512 个 training-prefix users，固定 hash split 为
  fit 307 / calibration 103 / audit 102。
- 只使用 `sequence[-3]` training target 与 `sequence[:-3]` history；
  `sequence[-2]`、`sequence[-1]`、validation/test 均未读取。
- GRAM checkpoint 冻结，optimizer steps = 0；未运行 beam、RL 或推荐指标评估。
- source utility 使用 matched baseline full：Toys K5、Beauty K10；
  动态 budget action 仍为 `K ∈ {0,5,10,20}`。
- tiny critic 固定为 32/16 两层 MLP、Adam、early stopping；fit/calibration/audit
  之间用户零重叠。

## 完整性审计

| Dataset | users | split exact | current replay | raw components | target feature | Trie membership | finite | optimizer steps | critic converged |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Toys | 512 | 1 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0 | 1 |
| Beauty | 512 | 1 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 0 | 1 |

两数据集所有执行完整性门槛通过，当前结果可用于预注册的 L0 决策。

## L0-A：效用异质性与 oracle headroom

| Dataset | Sem + / - | CF + / - | Oracle CE reduction | K20 dominance | depth χ²(df), Cramér V | L0-A |
|---|---:|---:|---:|---:|---:|---:|
| Toys | 74.22% / 25.78% | 85.94% / **14.06%** | 35.24% | 16.56% | 1120.10(15), V=0.378 | **FAIL** |
| Beauty | 51.76% / 48.24% | 84.57% / 15.43% | 14.52% | 13.62% | 1913.19(21), V=0.421 | PASS |

- Toys depth-action `p=2.3441e-229`，所有期望频数均 ≥5。
- Beauty 的数值 p 值发生浮点下溢（记录为 0），但 3 个期望频数低于 5，
  违反常规 Pearson χ² 近似条件；该项只能作谨慎支持，不能作强确认。
- 两数据集 oracle headroom 都很大且 K20 不支配，说明“可选动作之间存在空间”；
  但这不能覆盖 Toys source heterogeneity 的预注册失败。

## L0-B：target-free 可学习性

| Dataset / source | AUROC (95% CI) | active coverage | active utility mean (95% CI) | source gate |
|---|---:|---:|---:|---:|
| Toys semantic | 0.8690 [0.8382, 0.8978] | 50.58% | 0.2655 [0.1862, 0.3456] | PASS |
| Toys collaborative | 0.8703 [0.8388, 0.9001] | 54.23% | 0.8431 [0.6330, 1.0791] | PASS |
| Beauty semantic | 0.8604 [0.8296, 0.8894] | **15.48%** | 0.0079 **[-0.0462, 0.0632]** | **FAIL** |
| Beauty collaborative | 0.9507 [0.9339, 0.9660] | 33.05% | 1.3686 [1.0889, 1.6724] | PASS |

| Dataset | learned/fixed budget regret ratio (≤0.75) | semantic corruption drop (>0) | CF corruption drop (>0) | L0-B |
|---|---:|---:|---:|---:|
| Toys | 0.2936 | **-0.0353** | +0.1629 | **FAIL** |
| Beauty | **0.9406** | **-0.0026** | +0.2583 | **FAIL** |

这里的正面结果集中在 collaborative reliability：两个数据集的 CF AUROC、
active-utility CI 和 corruption direction 都通过。失败集中在 semantic reliability
以及 Beauty 的 budget transport。这意味着当前“一个统一 critic 同时控制语义可信度、
协同可信度和邻居预算”的核心假设没有成立；不能只挑 CF 结果晋级完整 MARC。

## 首次无效尝试与修复审计

首次 scoring 把 source utility 的 full reference 错设为 K20。在 GRAM 固定
128-token 输入下，K20 会机械挤掉 metadata，使 semantic utility 恒为 0；该尝试被标记
为 `EXECUTION_INVALID_SOURCE_REFERENCE`，未用于科学结论，产物原样保存在
`marc_l0_attempt1_invalid_full20_source_reference/`。

修复只做两项、且在重新读取新 scores 前锁定：

1. source utility 改为 matched baseline full（Toys K5、Beauty K10），K20 只保留为
   dynamic-budget action；
2. 原 MLP 的 L-BFGS 达到迭代上限，改为同一 32/16 结构的确定性 Adam +
   early stopping，并把 `critic_converged` 保留为执行完整性门槛。

修复后 config SHA256 为
`c8c031fca4b6d335bc1a5542bf39cc14ae81d6bba8f807192741f6d03d959e07`，
code SHA256 为
`fd87bbe6bd1537387897b02954ea4760179e90eeabc5b29fe28a05bc51a825c8`。

## 统计解释与谬误扫描

总体置信等级：**CAUTION**。停止结论由预注册阈值的直接失败支撑，较稳健；对
“depth action 显著异质”和“critic 可泛化”的正面解释需谨慎。

- 共检查 6 个显式 p-value（2 个 depth χ²、4 个 Spearman），未作多重比较校正。
  若按 Bonferroni `α=0.00833`，Toys 的 depth/两项 Spearman 和 Beauty CF Spearman
  仍通过；Beauty semantic Spearman `p=0.04644` 不通过。Spearman 不是晋级门槛，
  因而不改变 STOP。
- depth χ² 同时报 effect size：Toys Cramér V=0.378，Beauty V=0.421；
  Beauty 有稀疏期望格，近似 p 值不宜强解释。
- bootstrap CI 以 node rows 计算，而同一 user 有多个 generation depths；
  尽管用户 split 无重叠，node-level CI 仍可能低估 user-cluster 相关性，不能当成
  完全独立重复。

11/11 statistical fallacies checked：

| 类型 | 结论 |
|---|---|
| Simpson's paradox | 未发现方向反转；但只检查了 dataset/depth，未穷举潜在分层。 |
| Ecological fallacy | 未发现；结论限定为 node/sample utility，不外推到个体因果。 |
| Berkson's paradox | CAUTION：样本来自满足训练历史条件且经固定 hash/cap 的用户。 |
| Collider bias | 未控制由 utility 与 critic 共同导致的变量，未发现明确 collider。 |
| Base-rate neglect | 已同时报告 utility 正负基率、AUROC 和 active coverage。 |
| Regression to the mean | 非极端组 pre-post 设计，不适用。 |
| Survivorship bias | 无运行中 dropout；但 eligibility selection 限制总体代表性。 |
| Look-elsewhere effect | 预注册门槛和全部失败项均报告；首次无效尝试未被隐去。 |
| Garden of forking paths | 修复有显式 lineage；有效运行后未改 cohort/features/gates。 |
| Correlation ≠ causation | 未把 Spearman/AUROC 表述为 MARC 会提升推荐指标。 |
| Reverse causality | 反事实 utility 来自冻结模型干预；critic 关联不作现实因果外推。 |

## 资源与可复现性

- 有效运行 wall time 191.02 s；GPU3 峰值 used memory 4,410 MiB、峰值 utilization 42%。
- CodeLlama 资源预留在运行后已恢复。
- 资源进程日志中的 `is_workload=0` 是监控启动时的 PID 捕获竞态：
  日志中唯一 compute PID 603769 与实验时间窗、显存曲线一致；该列不能作为
  “foreign process”证据。
- CPU 单元测试：3/3 通过。
- 当前状态为 `ANALYZED` 而非 `VERIFIED`：未进行独立环境/独立种子的复跑。

## 解释边界

L0 只判断 frozen GRAM 上的 utility 是否足够异质、以及 target-free critic 是否能按
预注册规则恢复它；它不测试 Recall/NDCG，也不证明 MARC 的完整架构无效。当前证据
支持的最窄结论是：**这版 unified MARC controller 不值得进入实现/训练阶段**。
若未来另立周期，只能把“CF-only reliability controller”当作新且更窄的假设重新
预注册，不能用它救援本次 MARC。
