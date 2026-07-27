# GRAM 第三阶段：SMBR I0-D training-only benefit learnability

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-07-24
- Verification Status: ANALYZED
- Version Label: `smbr_i0_d_v1`

固定决策：**`STOP_SMBR_NO_CALIBRATED_SUBSET`**。

本阶段仅使用 `sequence[-3]` training target 与 `sequence[:-3]` history；
未读取 validation/test，未更新 GRAM。

## 数据集结果

| Dataset | Status | Integrity | Learnability |
|---|---|---:|---:|
| Toys | ANALYZED | True | False |
| Beauty | ANALYZED | True | False |

## 固定停止原因

两个数据集的 calibration predicted probability 都没有达到预注册网格的最小阈值
0.50：

| Dataset | Calibration probability range | Positive prevalence |
|---|---:|---:|
| Toys | 0.0349–0.3466 | 0.1725 |
| Beauty | 0.0931–0.3434 | 0.1650 |

因此没有候选能满足 active rate 0.10–0.40，双数据集均固定为
`NO_CALIBRATED_SUBSET`。按单次规则，不允许改为 percentile threshold 后重判主
结果。

## 不改变决策的排序解剖

绝对阈值网格与低 base rate 不匹配，是本设计的一个缺点。为避免把“阈值选错”误写成
“完全不可学习”，在主决策冻结后补做了 secondary descriptive ranking audit：

| Dataset | Audit AUROC (95% bootstrap CI) | Preregistered AUROC gate | Always-recover mean benefit |
|---|---:|---:|---:|
| Toys | 0.5172 [0.4286, 0.6037] | point ≥ 0.60, lower > 0.50 | -0.2948 |
| Beauty | 0.5614 [0.4858, 0.6355] | point ≥ 0.60, lower > 0.50 | -0.7024 |

即使忽略 0.50 absolute threshold，按模型概率直接取 audit top 10%–40%，所有切片的
active mean benefit 仍为负：

| Dataset | Top 10% | Top 20% | Top 30% | Top 40% |
|---|---:|---:|---:|---:|
| Toys | -0.1492 | -0.1975 | -0.2031 | -0.1944 |
| Beauty | -0.2453 | -0.3640 | -0.5033 | -0.5456 |

所以停止结论不只由 absolute threshold 造成：排序能力同样未通过，且高分子集不能把
recovery 的负均值变为正值。完整描述值见
`artifacts/phase3/smbr_i0_d/posthoc_descriptive.json`。

## 解释边界

- 证据支持：在当前 18 个 target-free census features、L2 logistic policy 和
  `sequence[-3]` training-prefix 标签定义下，净收益不能形成跨用户、非退化且高精度
  的可部署子集。
- 证据不支持：所有可能的 selector、所有 recovery action 或所有新数据集都必然失败。
- oracle positive-only policy gain 仍为 Toys 0.0137、Beauty 0.0145，但该 oracle 使用
  outcome，只表示很小的事后上界，不能转化为方法收益。
- 本次未训练 GRAM、未读取 validation/test、未做 beam generation；CodeLlama GPU3
  资源已恢复。

## Statistical fallacy scan

11/11 已检查：

1. Simpson's paradox：分别报告 Toys/Beauty，方向一致，无聚合反转证据。
2. Ecological fallacy：推断单位与样本单位均为 user-prefix，无跨层推断。
3. Berkson's paradox：存在 index/census eligibility，但没有按 outcome 选择；记为
   轻度 selection-boundary caution。
4. Collider bias：特征不含 target/outcome，未对共同结果变量条件化。
5. Base-rate neglect：已显式报告 12.5%/16.0% audit positive prevalence；原 absolute
   threshold 网格不匹配 base rate，已标记为设计缺点。
6. Regression to the mean：未按极端 outcome 选样本。
7. Survivorship bias：报告全部固定 cohort；无运行中样本丢失。
8. Look-elsewhere effect：主模型、18 特征和 9 个阈值均预注册；post-hoc 只作描述。
9. Garden of forking paths：单次规则执行，未用 percentile threshold 重判。
10. Correlation ≠ causation：policy learnability 是预测性结论；不声称 census feature
    导致 benefit。
11. Reverse causality：feature 来自 target 之前的 history，但仍只作预测解释。

综合置信等级为 **SOLID for the scoped stop decision**，而不是对所有 SMBR 变体的全局
否定。
