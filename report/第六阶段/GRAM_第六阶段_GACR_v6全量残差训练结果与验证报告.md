# GRAM 第六阶段：GACR-v6 全量残差训练结果与验证报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-02
- Verification Status: ANALYZED
- Version Label: `phase6_gacr_v6_full_fit_validation_v1`

## 1. 执行与完整性

- 实验：`GRAM_PHASE6_GACR_V6_FULL_FIT` 的 validation-only recovery；复用原运行已完成的
  Toys/Beauty × seeds 2023/2024/2025 共 6 个 residual checkpoint，**没有重新训练**。
- 有效科学产物：`summary.json` 与 12 份每用户 CSV 均于 2026-08-02 12:40+08:00 写完；原
  runner 随后在资源恢复路径产生 Bash exit=`2`，但发生在结果写完后，不能否定科学计算。
- 两域均为 1024 位 fresh validation 用户；6 个 v6 checkpoint SHA256 均与恢复配置一致。
- fit/calibration 用户隔离、fresh cohort 与训练/历史 validation cohort 零重叠、parent
  checkpoint SHA 不变、backbone optimizer steps=`0`，且 Sports/test 均未读取。

## 2. Fresh validation：v6 相对 GRAM

| 数据域 | seed | overall NDCG@10 相对增益 | 95% CI | Recall@10 | Recall@50 | tail NDCG@10 | tail Recall@50 | changed | broad harm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Toys | 2023 | +2.641% | [+0.847%, +4.793%] | +0.293pp | +0.098pp | +3.000% | +0.170pp | 14.551% | 0.098% |
| Toys | 2024 | +2.535% | [+0.473%, +4.814%] | +0.195pp | +0.098pp | +3.509% | +0.170pp | 14.355% | 0.195% |
| Toys | 2025 | +2.229% | [+0.388%, +4.295%] | +0.000pp | +0.195pp | +2.676% | +0.340pp | 14.258% | 0.195% |
| Beauty | 2023 | +3.064% | [+1.192%, +5.581%] | +0.293pp | -0.098pp | +2.482% | +0.000pp | 11.035% | 0.000% |
| Beauty | 2024 | +2.754% | [+0.883%, +5.219%] | +0.488pp | -0.195pp | +1.967% | -0.204pp | 11.133% | 0.000% |
| Beauty | 2025 | +3.097% | [+0.758%, +6.068%] | +0.488pp | -0.098pp | +2.553% | +0.000pp | 11.230% | 0.098% |

域均值的 v6 overall NDCG@10 相对 GRAM 为 Toys **+2.469%**、Beauty **+2.972%**；六 cell
宏平均为 **+2.720%**，6/6 点估计为正。四个标准指标的域均值相对 GRAM 均不低于 0；每 cell
broad harm 为 0.000%–0.195%，均低于 1% 门。

## 3. v6 相对冻结 GACR-v3 与预注册判定

| 判据 | 结果 | 判定 |
|---|---|---|
| 两域四项标准指标不低于 GRAM | 两域均满足 | 通过 |
| 两域 NDCG@10 相对 GRAM至少 +1% | Toys +2.469%；Beauty +2.972% | 通过 |
| 六 cell 宏平均相对 GRAM至少 +2% | +2.720% | 通过 |
| v6-v3 六 cell宏平均至少 +0.5% | **+0.559%** | 通过 |
| v6 NDCG@10 高于 v3 的 cell | 5/6；Beauty-2025 为 -0.173% | 通过 |
| 两域 mean tail NDCG@10、overall/tail Recall@50 不低于 v3 | Toys 全部改善；Beauty tail NDCG@10 -0.0181pp、overall Recall@50 -0.0977pp、tail Recall@50 -0.0679pp | **失败** |
| 每 cell broad harm ≤1%、checkpoint 不变 | 全部满足 | 通过 |
| cohort 隔离与 Sports/test 封存 | 全部满足 | 通过 |

正式决定为 **`KEEP_GACR_V3_FULL_FIT_SCALE_NOT_BENEFICIAL`**：v6 不替换冻结 GACR-v3。
这是由预注册的 Beauty safety 门决定，而非整体 NDCG 缺少信号。

## 4. 机制解释与下一假设

全量 fit 相对 v3 的增量集中在 Toys（+0.277% 至 +1.497%），Beauty 只有两个很小的正 cell
和一个 -0.173% cell；同时 Beauty 的深截断指标略有退化。v6 的唯一改动是训练 records 数量，
原始 target-vs-highest-negative hinge 并不区分 NDCG@10 与 Recall@50 截断线附近的交换。

这支持、但不证明，下一项可检验假设：在保持 **全量 fit** 不变时，将该 hinge 替换为冻结 base
rank 导出的 NDCG@10/Recall@50 截断敏感加权 pairwise logistic loss，可以保留 v6 的 NDCG
信号并恢复 Beauty tail/Recall@50。该假设以独立 v7 预注册检验，不能把本次结果用于调权重。

## 5. Statistical Interpretation

Overall Confidence：**CAUTION**。

6 个 v6-vs-GRAM NDCG@10 bootstrap 95% CI 的下界均为正，提供同一 fresh cohort 内的稳定正向
开发证据。v6-v3 的 primary retain/reject 采用事先规定的点估计门；本报告未将多次开发域观察
当作一次确认性 p 值检验，也不声称线上因果收益。项目跨多版本、多个指标与三批历史 cohort 的
探索意味着不能把 6/6 正向或本次 CI 当作最终泛化结论。

## 6. Fallacy Scan

覆盖：**11/11 checked**。

| 谬误 | 严重度 | 检查结果 |
|---|---|---|
| Simpson's paradox | NOTE | 报告了 Toys/Beauty 与 head/tail；Beauty safety 失败未被 overall 掩盖。 |
| Ecological fallacy | NOTE | 域均值不代替个体结果；逐用户 CSV、changed coverage 与 harm 均已保存。 |
| Berkson's paradox | CAUTION | 两域是重复开发域，非未见确认域。 |
| Collider bias | NOTE | fresh validation target 未用于训练、部署或配置选择。 |
| Base-rate neglect | NOTE | 同时报出 tail 指标、coverage 与 broad harm。 |
| Regression to mean | NOTE | fresh cohort 未依 v5/v6 极端表现筛选。 |
| Survivorship bias | NOTE | 每个 cell 的 1024 位用户均进入评估。 |
| Look-elsewhere effect | CAUTION | 长期多版本开发，CI 与 5/6 仅作开发证据。 |
| Garden of forking paths | NOTE | v6 的训练规模、门与 cohort 在运行前冻结；后继 loss 单独预注册。 |
| Correlation != causation | NOTE | 结论限于离线排序干预，不外推线上行为。 |
| Reverse causality | NOTE | 不进行观察性方向因果主张。 |

## 7. 结论

保留冻结 **GACR-v3** 为 incumbent；停止“仅扩大 fit records、其余保持 hinge loss 不变”的
v6 配置。下一主实验为独立的 **GACR-v7 全量 fit 指标对齐损失**，详见
`plan/GRAM_第六阶段_GACR-v7全量指标对齐残差训练实验计划.md`。
