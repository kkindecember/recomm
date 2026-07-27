# GRAM 第四阶段：CCRR R0 结果与验证报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-27
- Verification Status: ANALYZED
- Version Label: `ccrr_r0_validation_v1`

## 1. 执行结果

- Experiment ID：`GRAM_PHASE4_CCRR_R0`
- 类型：CPU-only candidate-level logistic fit + calibration effect gate
- 状态：completed
- 命令：`bash experiment/phase4/run_phase4_ccrr_r0.sh start`
- 正式运行用时：21.23 秒
- 决策：**`STOP_CCRR_NO_CANDIDATE_CONDITIONAL_EFFECT`**
- calibration qualified：false；按预注册未打开 audit

首次 preflight 因未知候选检查在内循环重复构造 catalog set 而发生工程性 CPU
空转；该次未拟合模型、未产生科学结果。终止残留进程后，唯一修复是将 catalog set
移到循环外，特征、标签、split、模型和门槛均未改变。修复后 4/4 单元测试与完整
preflight 通过。

## 2. Calibration 结果

| 数据集 | B0 NDCG@10 | R1 NDCG@10 | 相对增益 | Recall@10 绝对增益 | tail NDCG 相对增益 | R1 vs B1 NDCG |
|---|---:|---:|---:|---:|---:|---:|
| Toys | 0.076589 | 0.081418 | **+6.305%** | +0.594pp | **+9.193%** | +5.843% |
| Beauty | 0.061007 | 0.067808 | **+11.147%** | +0.949pp | **-3.795%** | +8.687% |

两个域的 overall NDCG 和 Recall 都有大幅正变化，且显著超过固定 RPCD rank fusion。
这证明约 3pp 的 union coverage 并非完全不可兑现，也证明候选级条件化比全局权重更
合适。但 Beauty tail 的预注册 nondecrease 门槛失败，所以整体必须 STOP；不得利用
同一 calibration 结果追加 tail weight、GBDT、MLP 或查看 audit 进行救援。

## 3. 模型与完整性

| 数据集 | fit users | candidate rows | positive rows | 迭代 | 收敛 |
|---|---:|---:|---:|---:|---|
| Toys | 3,870 | 362,293 | 954 | 52 | 是 |
| Beauty | 4,424 | 399,115 | 1,053 | 50 | 是 |

- 两域共享完全相同的 16 维 target-free feature schema、scaler 规则和 logistic
  超参数；
- calibration user hash 与预检记录精确一致；
- audit rows used for fit = 0；
- candidate set identity rate = 100%；
- target match rate = 100%；
- unknown/duplicate candidates = 0；
- test prediction 未读取，`sequence[-1]` 未索引；
- GRAM/SASRec optimizer steps = 0；
- config SHA-256：
  `442100b14b357e5bdec7eb6f3b8289da1b7d7332e026d5f0295009ae2ea85a29`；
- preflight SHA-256：
  `e7d5dc038fec7cd6847cfaaa556586235aa789b1c02aef6328bb9c1f326a73f1`；
- summary SHA-256：
  `37bd5997090a845049a6b78d886f6bb077f478bee7c39795ddda566ccbba121e`。

## 4. 解释

CCRR 与此前负结果共同缩小了问题：

1. 不是总体排序完全学不动：一个线性候选级模型已经产生 6%–11% calibration
   NDCG 增益；
2. 不是继续增加训练轮数就必然解决：模型在拟合数据上已收敛，但 Beauty tail 仍与
   overall 反向；
3. 当前瓶颈是表示与目标偏向：单一 lexical path、头部占主导的候选标签和不带显式
   tail 约束的轻量排序器，会把可兑现收益集中到头部。

因此，下一周期不应继续搜索 SASRec/GRAM 的手工融合或轻量 ranker。若继续研究，应
允许修改 GRAM 架构和训练目标，直接给尾部 item-space 监督，而不是把同一冻结候选再
换一种权重。

## 5. Statistical Interpretation

Overall Confidence：**CAUTION**。

这是 calibration fit-set 上的机制可学习性/effect gate，不是独立泛化结果；正增益
可能包含拟合乐观偏差。预注册门槛正确地阻止了在 Beauty tail 失败后读取 audit。
因此可以说“线性特征具有可学习信号”，不能说“CCRR 已提升未见用户推荐效果”。

## 6. Fallacy Scan

覆盖：**11/11 checked**

| 谬误 | 严重度 | 检查结果 |
|---|---|---|
| Simpson's paradox | CAUTION | Beauty overall +11.15% 与 tail -3.79% 方向相反 |
| Ecological fallacy | NOTE | 不由平均增益断言每个用户获益 |
| Berkson's paradox | CAUTION | calibration 是选择与拟合集，不能当独立泛化样本 |
| Collider bias | NOTE | audit 未参与拟合或选择；tail 由 training popularity 预定义 |
| Base-rate neglect | NOTE | 报告 positive rows 仅约 0.26%，未只报总体指标 |
| Regression to mean | NOTE | 未按 audit 极端结果选择样本 |
| Survivorship bias | NOTE | 所有 calibration 用户和候选均保留 |
| Look-elsewhere effect | CAUTION | 本轮无模型网格，但项目已探索多轮方向 |
| Garden of forking paths | CAUTION | 未在失败后追加 tail weight/GBDT/MLP；跨方向仍属探索 |
| Correlation ≠ causation | NOTE | calibration 排序增益不解释现实用户行为因果 |
| Reverse causality | NOTE | 特征只来自目标发生前的历史和训练统计 |

## 7. 产物

- `artifacts/phase4/configs/ccrr_r0_preregistered.json`
- `artifacts/phase4/ccrr_r0/preflight.json`
- `artifacts/phase4/ccrr_r0/summary.json`
- `experiment/phase4/ccrr_r0.py`
- `experiment/phase4/test_ccrr_r0.py`
- `experiment/phase4/run_phase4_ccrr_r0.sh`
- `artifacts/phase4/logs/ccrr_r0.log`
