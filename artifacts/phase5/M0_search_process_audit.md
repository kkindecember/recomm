# GRAM Phase 5 M0：创新搜索过程失效审计

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate + plan
- Created: 2026-07-29
- Verification Status: ANALYZED
- Inputs:
  - `plan/GRAM_第三阶段_创新探索与渐进式实验计划.md`
  - `plan/GRAM_第四阶段_方法创新与渐进实验计划.md`
  - `plan/GRAM_第四阶段_续篇_Toys_Beauty非自适应方法创新计划.md`
  - `artifacts/phase3/**/summary.json`
  - `artifacts/phase4/**/summary.json`
  - `artifacts/phase4/fpug_p0/{Toys,Beauty}/validation_metrics.csv`

## 1. 审计结论

过去的高失败率主要来自实验搜索制度，而不是已经证明“GRAM 没有可做的创新”：

1. **把方法发现变成了缺陷普查。** 大量方向要求先证明 baseline 存在一个达到人为
   阈值的缺陷；baseline 某方面表现好，反而会使方法在训练前被淘汰。IALC、LNDR、
   SCDL、CPIA 都属于“结构事实存在，但预设 deficit 不够大”。
2. **把 proxy 当成方法效果的必要条件。** teacher-forced gold-path log-prob、
   frozen counterfactual utility、AUROC、correlation excess 与 token census 都是
   机制代理，不是 Recall/NDCG 的必要条件。此前却常用它们阻止实际训练。
3. **双域、overall、tail、CI、harm、coverage 多门合取。** 一个方向常需同时通过
   8–14 个条件；任何一个边界性失败都会关闭整条方法链。即使单项通过率为 90%，
   12 项合取的理论通过率也只有 `0.9^12 = 28.2%`；这只是说明合取惩罚的量级，
   不是假定各 gate 真正独立。
4. **pilot 的统计功效与晋级规则不匹配。** 多个 effect pilot 只用 512 或 1,024
   validation users，却要求约 1% 相对提升且 bootstrap 下界非负。稀疏 top-10
   rank 变化下，这种设计很容易把真实的小正效应判成失败。
5. **局部修补多，端到端方法少。** 多数方向止于 0-update premise audit；真正进入
   可学习 recommendation effect gate 的主要实例只有 HBTR、GCDH、GACR、FPUG，
   TCDR 甚至在读取 validation 前停止。因而“二十个完整方法都失败”并不符合实际。
6. **方向切换太快。** 许多方向在一天内完成提出、阈值冻结、审计和停止。流程的
   可追溯性很好，但没有给一个有合理先验的方法足够的训练预算、优化诊断和公平基线。
7. **把新颖性当成先验零重叠。** 论文创新通常允许借鉴已知组件，只要问题定义、
   机制组合、实证结论或适用边界有新增价值。此前多次因一个核心组件被覆盖而直接
   终止，导致搜索偏向越来越窄、越来越脆弱的“无人做过的小缝隙”。

## 2. M0 定量检查

使用 FPUG-P0 保存的逐用户 NDCG@10 配对差，估计“观察到当前点效应时，使正态近似
95% 区间下界高于 0 所需的用户数”。该计算只用于诊断原 pilot 的分辨率，不复活
FPUG，也不构成新的效果声明。

| 数据集/组 | n | mean absolute NDCG difference | SD | 非零差用户 | 估计所需 n |
|---|---:|---:|---:|---:|---:|
| Toys overall | 512 | +0.001739 | 0.030678 | 29 | 1,196 |
| Beauty overall | 512 | -0.002791 | 0.027913 | 8 | 384（效应方向为负，无晋级意义） |
| Toys tail | 303 | +0.003750 | 0.031114 | 12 | 265 |
| Beauty tail | 248 | +0.000072 | 0.001135 | 1 | 953 |

关键事实：

- Toys overall 的点估计是 `+2.37% relative`，但 512 人不足以让区间下界转正；
  按观测方差约需 1,196 人。原 gate 因而混合了“效果是否存在”和“pilot 是否有足够
  分辨率”两个问题。
- Toys 只有 29/512、Beauty 只有 8/512 用户的 NDCG@10 发生变化。对这种稀疏差值，
  小 cohort 的 bootstrap 区间高度离散。
- Beauty 的负向结果仍是有效反证；功效审计不能把负效应改写为正效应。M0 只说明
  后续不应再用同样大小的 cohort 要求小效应的严格正下界。

## 3. 哪些做法应保留

- target/test exclusion、lineage、SHA、invalid-run 隔离；
- matched baseline、相同训练步数、失败产物保留；
- validation firewall 和一次性读取；
- 效应量、置信区间、tail 与 broad-harm 报告；
- Sports 作为未参与方向生成的确认域。

问题不是“太严谨”，而是把完整性 gate、机制 proxy、效果标准和确认标准混成同一个
早期停止器。Phase 5 将四类门分开。

## 4. 固定决定

**`RESET_SEARCH_PROCESS_AND_START_ONE_METHOD_CYCLE`**

下一周期不再要求先证明 baseline 有“大缺陷”。只要方法有独立理论/文献依据、
实现可归因、训练成本可承受，就允许一次有统计分辨率的真实 effect pilot。

