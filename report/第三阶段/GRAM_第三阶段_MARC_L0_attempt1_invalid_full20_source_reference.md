# GRAM 第三阶段：MARC L0 反事实效用与 critic 可学习性报告

> **无效尝试存档：不得用于科学结论。** 本次错误地用 K20 full 作为 source utility
> reference；在 128-token 截断下 metadata 被机械移除，semantic utility 恒为 0。
> 状态为 `EXECUTION_INVALID_SOURCE_REFERENCE`。下方是当时的原始自动报告，
> 其中 `STOP_MARC_NO_UTILITY_HETEROGENEITY` 已作废；有效结论见修复后 L0 报告。

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-07-24
- Verification Status: ANALYZED
- Version Label: `marc_l0_v1`

固定决策：**`STOP_MARC_NO_UTILITY_HETEROGENEITY`**。

仅使用 `sequence[-3]` training target 与 `sequence[:-3]` history；
未读取 validation/test，未更新 GRAM，未运行 beam 或 RL。

## 结果

| Dataset | Integrity | L0-A | L0-B | Oracle CE reduction | K20 dominance | Sem AUROC | CF AUROC | Budget regret ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Toys | True | False | False | 0.352398 | 0.165644 | nan | 0.787710 | 0.575370 |
| Beauty | True | False | False | 0.145181 | 0.136238 | nan | 0.919898 | 0.946837 |

## 解释边界

L0 只判断 utility 是否异质且能否由 target-free state 预测；
它不证明 MARC 会改善 Recall/NDCG。若固定决策为 STOP，L1、RL、
二次 refinement 与 validation 均不解锁。
