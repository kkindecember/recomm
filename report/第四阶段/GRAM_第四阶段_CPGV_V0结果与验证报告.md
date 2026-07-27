## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-27
- Verification Status: ANALYZED
- Version Label: `cpgv_v0_validation_v1`

# CPGV V0 结果与验证报告

## 1. 执行结果

- Experiment ID：`GRAM_PHASE4_CPGV_V0`
- 状态：completed
- 用时：171.20 秒
- Toys/Beauty eligible users：493/544
- candidate paths：24,650/27,200
- 决策：`STOP_CPGV_GRAM_CANNOT_VERIFY_PROPOSALS`
- GPU3：已恢复

## 2. 主要指标

| 数据集 | SASRec R@10 | Exact R@10 | Exact 95% CI | 配对差 | 配对差 95% CI |
|---|---:|---:|---:|---:|---:|
| Toys | 25.963% | 22.110% | [18.458%,25.761%] | -3.854pp | [-8.925pp,+1.217pp] |
| Beauty | 20.772% | 9.926% | [7.537%,12.684%] | -10.846pp | [-14.890pp,-6.618pp] |

Toys/Beauty exact-rescore Recall@10 均未达到 25%，也没有相对 SASRec 提高 5pp。
Beauty 的恶化稳定；Toys 的区间虽跨 0，但点估计方向错误且两项预注册门槛均失败。

Pairwise concordance：

- Toys：0.580，95% CI [0.558,0.601]；
- Beauty：0.506，95% CI [0.487,0.526]。

Beauty 基本无法由 GRAM exact score区分 gold 与其余 SASRec proposals。

## 3. 完整性

- mapping rate：1.0；
- finite rate：1.0；
- Trie membership rate：1.0；
- target input inclusion rate：0.0；
- optimizer steps：0；
- test prediction 未读取，`sequence[-1]` 未索引；
- config SHA-256：
  `d426eca6290af256116c3c531a6c7dc05fd849e88c9c4862cab7ecd567808baf`；
- preflight SHA-256：
  `df25a1fcbdf2a8d2727bae0816cf26c02925ede02c3bc369e851dce55925929a`；
- summary SHA-256：
  `4ea408c2ae5e147bbcbd2abacedb919e4ac32d246949d296d5392763f19f3a4e`。

## 4. 解释边界

eligible cohort 是为了机制诊断而用 target 定义，不能把这里的 Recall 当完整 validation
最终指标。cohort 又极度偏 head：Toys/Beauty tail 仅 21/8。因此 tail 子组点估计
不具备稳定解释力，也不能支持 subgroup rescue。

## 5. Fallacy Scan

覆盖：**11/11 checked**

| 谬误 | 严重度 | 结果 |
|---|---|---|
| Simpson's paradox | CAUTION | overall 与极小 tail 子组方向不同；以 overall gate 为准 |
| Ecological fallacy | NOTE | 不从域均值推断每个 user 可验证 |
| Berkson's paradox | CAUTION | cohort 按“外部 teacher 补回 gold”筛选，是 target-selected diagnostic |
| Collider bias | CAUTION | eligibility 同时受 GRAM miss 与 SASRec hit 影响，只作机制诊断 |
| Base-rate neglect | NOTE | 报告 eligible 总数及 head/tail base rate |
| Regression to mean | NOTE | 未按 exact score 极值选择 users |
| Survivorship bias | NOTE | eligible cohort 全量评分，无评分后丢弃 |
| Look-elsewhere effect | NOTE | 没有 score weight/grid；单一预注册 exact score |
| Garden of forking paths | CAUTION | V0 单一路径固定，但属于多方向探索的一环 |
| Correlation ≠ causation | NOTE | 只判断离线 rank verification |
| Reverse causality | NOTE | 输入历史在 validation target 之前 |

## 6. 产物

- `artifacts/phase4/cpgv_v0/preflight.json`
- `artifacts/phase4/cpgv_v0/summary.json`
- `artifacts/phase4/cpgv_v0/{Toys,Beauty}/cohort.csv`
- `artifacts/phase4/cpgv_v0/{Toys,Beauty}/user_results.csv`
- `artifacts/phase4/cpgv_v0/{Toys,Beauty}/candidate_scores.csv`
- `artifacts/phase4/logs/cpgv_v0.log`
