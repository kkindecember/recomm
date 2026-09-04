# Stage17 FP3 Full SETRec 正式结果报告

## Material Passport

- Origin Skill：`academic-research-suite / experiment-agent`
- Origin Mode：`run → validate`
- Step：`S17-FP3`
- 状态：`COMPLETED / FP3_NOT_STRONG_PASS`
- Verification Status：`ANALYZED`
- 外部评估用户：12833
- 数据边界：复用哈希冻结的 D0 bundle；未重开 raw projection；D1/D2、official test、Sports 未读

## 1. 正式判定

FP3 Gate：`FP3_NOT_STRONG_PASS`。
下一步：stop FP3 SETRec promotion; keep D1/D2 and FP4 closed; do not tune on D0。

## 2. 主结果

| Arm | Hit@10 | NDCG@10 | MRR@10 | Hit@50 | NDCG@50 |
|---|---:|---:|---:|---:|---:|
| S0 Ordered Control | 0.027585 | 0.014518 | 0.010590 | 0.079171 | 0.025569 |
| S1R Repo-Parity | 0.024702 | 0.012478 | 0.008826 | 0.074573 | 0.023113 |
| S1P Paper-Faithful | 0.023533 | 0.012062 | 0.008610 | 0.075197 | 0.023151 |
| S2 GRAM-SETRec-Paper-Full | 0.030468 | 0.015663 | 0.011203 | 0.085872 | 0.027491 |
| G0 GRAM-B0-Fresh | 0.097561 | 0.061700 | 0.050685 | 0.180472 | 0.079821 |

## 3. Paired effects

| Comparison | ΔNDCG@10 [95% CI] | ΔHit@10 [95% CI] | Gain/Loss/Tie |
|---|---:|---:|---:|
| S1R_MINUS_S0 | -0.002040 [-0.003231, -0.000885] | -0.002883 [-0.005299, -0.000701] | 160/224/12449 |
| S1P_MINUS_S1R | -0.000416 [-0.001641, +0.000793] | -0.001169 [-0.003662, +0.001169] | 183/194/12456 |
| S1P_MINUS_S0 | -0.002456 [-0.003708, -0.001170] | -0.004052 [-0.006312, -0.001558] | 167/232/12434 |
| S2_MINUS_S0 | +0.001145 [-0.000154, +0.002510] | +0.002883 [+0.000390, +0.005377] | 248/212/12373 |
| S2_MINUS_G0 | -0.046037 [-0.049474, -0.042445] | -0.067093 [-0.072393, -0.061950] | 252/1164/11417 |

## 4. 机制与效率

| Arm | Full-set recovery | Valid item | Query norms | Forbidden visibility | sec/user |
|---|---:|---:|---|---:|---:|
| S0 Ordered Control | 0.000078 | 1.000000 | True | 0 | 0.000418 |
| S1R Repo-Parity | 0.000234 | 1.000000 | True | 0 | 0.000397 |
| S1P Paper-Faithful | 0.000078 | 1.000000 | True | 0 | 0.000393 |
| S2 GRAM-SETRec-Paper-Full | 0.000468 | 1.000000 | True | 0 | 0.008975 |

每个 query 的 target grounding rank/recovery、semantic reconstruction、完整 latency 与分组结果见 canonical `analysis.json`。

## 5. Gate 审计

| Check | Status |
|---|---|
| s1r_mechanism_active | PASS |
| s1p_vs_s0_ndcg_positive | FAIL |
| s1p_mechanism_active | PASS |
| s2_vs_s0_ndcg_ge_0.0015 | FAIL |
| s2_vs_s0_ndcg_ci95_low_positive | FAIL |
| s2_vs_g0_ndcg_ge_0.0015 | FAIL |
| s2_vs_s0_hit_nonnegative | PASS |
| s2_mechanism_active | PASS |
| no_catastrophic_large_subgroup | PASS |
| integrity_valid | PASS |

## 6. 完整性

- 用户严格对齐：`True`。
- 四臂及 G0 top-50 合法：`True`。
- Bundle SHA 匹配：`True`。
- Raw external projection reopened：`false`。
- D1/D2、official test、Sports read：`false`。

## 7. 统计谬误扫描

Coverage：`11/11 checked`。

| Type | Finding |
|---|---|
| simpsons_paradox | NOTE: overall and preregistered subgroups are retained for direction checks |
| ecological_fallacy | NOTE: inference is paired at the user level |
| berksons_paradox | NOTE: the full frozen D0 cohort is used without efficacy-based selection |
| collider_bias | NOTE: no post-treatment covariate conditioning is used |
| base_rate_neglect | NOTE: not a diagnostic-classification study |
| regression_to_mean | NOTE: checkpoint selection used train-prefix internal dev; D0 is external |
| survivorship_bias | NOTE: exact complete user alignment |
| look_elsewhere_effect | NOTE: primary contrasts and thresholds were preregistered |
| garden_of_forking_paths | CAUTION: engineering attempts exist, but beta and best checkpoints were frozen before D0 |
| correlation_not_causation | NOTE: matched arm interventions support only this implementation and fold |
| reverse_causality | NOTE: not applicable to assigned model-arm comparisons |
