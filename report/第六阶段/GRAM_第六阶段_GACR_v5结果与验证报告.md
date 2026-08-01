# GRAM 第六阶段：GACR-v5 结果与验证报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-01
- Verification Status: ANALYZED
- Version Label: `phase6_gacr_v5_validation_v1`

## 1. 执行结果

- Experiment ID：`GRAM_PHASE6_GACR_V5`
- 类型：冻结 GACR-v3 residual 与 GACR-v4 gate 的 target-free 连续 residual 加权
- 科学 workload：2026-08-01 16:32:57+08:00 开始，18:00:10 左右完成；exit=`0`
- 总 runner：16:31:57 启动，18:00:16 结束，status=`succeeded`
- 数据：Toys、Beauty；每域 3 seeds；每个域-seed 1024 个 fresh validation 用户
- 设备：物理 GPU0；实验后 CodeLlama 已恢复到 GPU0，`resource_reservation=restored_on_gpu0`
- Sports/test：均未读取

## 2. Alpha 选择结果

v5 唯一改动是把 residual 乘以
`m(p; alpha)=alpha+(1-alpha)*p`。`alpha=1` 应用完整 residual，是冻结 GACR-v3 的
精确 identity control。

| 数据域 | alpha | eligible | calibration mean NDCG@10 | mean tail NDCG@10 | mean Recall@50 | mean multiplier |
|---|---:|---|---:|---:|---:|---:|
| Toys | 0.00 | 是 | 0.245794 | 0.248125 | 0.562500 | 0.4861 |
| Toys | 0.25 | 是 | 0.246118 | 0.248546 | 0.562500 | 0.6146 |
| Toys | 0.50 | 是 | 0.246672 | 0.248577 | 0.562500 | 0.7431 |
| Toys | 0.75 | 是 | 0.246871 | 0.248699 | 0.562500 | 0.8715 |
| Toys | **1.00** | 是 | **0.246983** | 0.248614 | 0.562500 | 1.0000 |
| Beauty | 0.00 | 是 | 0.204377 | 0.209883 | 0.433594 | 0.5503 |
| Beauty | 0.25 | 是 | 0.204377 | 0.209883 | 0.433594 | 0.6627 |
| Beauty | 0.50 | 是 | 0.204547 | **0.210224** | 0.433594 | 0.7751 |
| Beauty | 0.75 | 是 | 0.204604 | 0.210177 | 0.433594 | 0.8876 |
| Beauty | **1.00** | 是 | **0.204620** | 0.210131 | 0.433594 | 1.0000 |

两个域的所有候选都通过安全门，但最大 overall NDCG 均出现在 `alpha=1`。Toys 的 overall
NDCG 随 residual 强度增加而近似单调上升；Beauty 也具有同一主趋势。Beauty 的
`alpha=0.5` 只在 tail NDCG 上比 identity 高约 `0.000093` 的绝对值，却损失 overall
NDCG，不能按预注册目标选中。最终 Toys、Beauty 均选择 **`alpha=1`**。

## 3. Fresh development validation

下表为最终 v5 相对 GRAM 的结果。因为选中 `alpha=1`，这些数值同时也是冻结 GACR-v3
在本次新 cohort 上的结果。

| 数据域 | seed | overall NDCG@10 相对增益 | 95% CI | Recall@10 | Recall@50 | head NDCG | tail NDCG | tail Recall@50 | changed | broad harm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Toys | 2023 | +2.389% | [-0.107%, +4.889%] | +0.195pp | +0.000pp | +4.426% | +1.077% | +0.000pp | 13.867% | 0.195% |
| Toys | 2024 | +3.850% | [+1.156%, +6.920%] | +0.488pp | +0.000pp | +6.646% | +2.049% | +0.000pp | 14.453% | 0.195% |
| Toys | 2025 | +3.037% | [+0.449%, +5.883%] | +0.391pp | +0.000pp | +6.417% | +0.859% | +0.000pp | 13.965% | 0.195% |
| Beauty | 2023 | +2.299% | [+1.024%, +3.945%] | +0.195pp | +0.293pp | +1.846% | +3.296% | +0.189pp | 12.598% | 0.000% |
| Beauty | 2024 | +1.554% | [+0.594%, +2.847%] | +0.098pp | +0.293pp | +1.501% | +1.671% | +0.189pp | 12.598% | 0.000% |
| Beauty | 2025 | +2.514% | [+1.125%, +4.324%] | +0.195pp | +0.293pp | +1.798% | +4.089% | +0.189pp | 13.086% | 0.000% |

域均值：

| 数据域 | overall NDCG | Recall@10 | Recall@50 | head NDCG | tail NDCG | tail Recall@50 | changed | broad harm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Toys | **+3.092%** | +0.358pp | +0.000pp | +5.829% | +1.328% | +0.000pp | 14.095% | 0.195% |
| Beauty | **+2.122%** | +0.163pp | +0.293pp | +1.715% | +3.019% | +0.189pp | 12.760% | 0.000% |

- 六个域-seed cell 宏平均 overall NDCG@10：**+2.607%**
- 6/6 点估计为正；最小 `+1.554%`，最大 `+3.850%`
- Beauty 3/3 CI 下界为正；Toys 2/3 CI 下界为正，seed 2023 的下界略跨 0
- Toys 本 cohort 没有 Recall@50 损失，但只是与 GRAM 持平，不能解释为 Recall@50 提升

## 4. v5 相对 v3 的预注册判定

| 保留判据 | 结果 |
|---|---|
| Toys mean NDCG 严格超过 v3 | **失败：精确增量 0** |
| 六 cell 宏平均严格超过 v3 | **失败：精确增量 0** |
| Beauty mean NDCG 不低于 v3 | 通过：精确相等 |
| Toys tail NDCG、overall/tail Recall@50 不低于 v3 | 通过：精确相等 |
| Toys mean Recall@50 不低于 GRAM | 通过：相等，增量 0 |
| 每个 cell broad harm ≤1%、Recall@10 不低于 GRAM | 通过 |
| 完整性门 | 通过 |

正式决定为 **`RETURN_TO_GACR_V3_STOP_GATE_WEIGHTING_FAMILY`**。v5 没有提供 v3 之外
的增量；停止的是 residual 之后的 gate/attenuation/soft-weighting 修改族，不是核心 GACR。

## 5. 跨 fresh-cohort 稳定性

| 验证批次 | Toys mean NDCG | Beauty mean NDCG | 六 cell 宏平均 | 正向 cell |
|---|---:|---:|---:|---:|
| GACR-v3 cohort | +1.238% | +4.470% | +2.854% | 6/6 |
| GACR-v4 cohort | +1.169% | +3.925% | +2.547% | 6/6 |
| GACR-v5 cohort | +3.092% | +2.122% | +2.607% | 6/6 |

冻结核心 GACR 在三批互斥 fresh development cohort 上累计 **18/18 overall NDCG 正向
点估计**。这比单次实验更有说服力，并且 v5 中 Toys 的证据增强；但 Toys/Beauty 都是
反复开发域，且整个项目探索过多个配置，因此这仍是开发证据，不能替代未见域或最终 test
上的一次性确认。

## 6. 机制结论与下一研究问题

连续三种部署侧“安全化”都选择完整 residual：

1. v3 的 residual-spread 衰减选择 identity；
2. v4 的 learned hard gate 选择 threshold 0，即全量应用；
3. v5 的 soft multiplier 选择 alpha 1，即全量强度。

三次结果共同说明，当前主要限制不是“哪些用户该少用 residual”。结果后进一步确认当前
residual 只在每域 1024 个抽样 fit records 上训练，因此研究者决定先测试更基础的数据规模
问题：保持模型、候选、loss 和部署方式不变，只使用既有 fit split 的全部 records。指标对齐
loss 保留为规模实验无增量时的独立备选，不与本次规模变化混合。

## 7. 完整性与复现状态

- fit/calibration 用户 overlap=`0`；fresh validation 与五批历史 validation cohort overlap=`0`
- Toys/Beauty fresh 用户 SHA256 分别为
  `2ed71748c09f59cbb473b960d5ff1c2f13180faf9e5d85abf7589d8d458de278`、
  `9fd9ca55161d1729614136558f82772904773708bc1236529d37a4dc498f81b6`
- parent checkpoint SHA 前后不变；v3 residual 与 v4 gate 均冻结
- `gate_optimizer_steps=0`；`backbone_optimizer_steps=0`
- `alpha1_exact_v3_identity=true`；六个 cell 均记录 `matches_v3=true`
- 12 份逐用户 CSV 均为 1024 数据行
- implementation SHA256：`47b28ca3acf7a36ddbaad67c7c7113dd537379adbb74e8c984409935e6d0b76d`
- test SHA256：`ea8ce9519d22a54ee1c887ac5b91e9bfee6a439854a2d2d24028cc0e765b906b`
- runner SHA256：`70108ac3251f281cd645d1cf01c2dea228ca56147f7e22f045e7df28b4b698be`
- config SHA256：`29eae16f514f14287a7e7dbcd7b9dc493262b119f9fb671ccf2683451606d86a`
- summary SHA256：`113da0d96dbccff48b7552e3dcdd8823d46abf2ea62e62803e5c8a29c4ad8df8`
- Reproducibility verdict：**CANNOT_VERIFY（未独立重跑）**；内部 lineage、隔离和 identity
  检查通过

## 8. Statistical Interpretation

Overall Confidence：**CAUTION**。

“v5 未超过 v3”是由预注册选择产生的精确 identity 结论，不依赖显著性检验。v5 cohort
相对 GRAM 的 Beauty 三个 CI、Toys 两个 CI 下界为正；但跨 cohort/配置探索没有做全项目
多重比较校正，且同一开发域被重复观察。报告因此只称为稳定正向开发信号，不称为最终确证。

## 9. Fallacy Scan

覆盖：**11/11 checked**

| 谬误 | 严重度 | 检查结果 |
|---|---|---|
| Simpson's paradox | NOTE | 双域 overall 同向，但 head/tail 强度不同，保留分域分组结果 |
| Ecological fallacy | NOTE | 域均值不代表每用户改善；报告 changed 与 harm |
| Berkson's paradox | CAUTION | Toys/Beauty 是重复开发域，不是未见确认域 |
| Collider bias | NOTE | target label 未进入部署 soft weight 或 fresh-cohort 选择 |
| Base-rate neglect | NOTE | 明确报告低 harm 与约 13%–14% changed coverage |
| Regression to mean | NOTE | v5 cohort 未按 v4 极端结果筛选，并展示三 cohort 波动 |
| Survivorship bias | NOTE | 六个 cell、每 cell 1024 用户均进入汇总 |
| Look-elsewhere effect | CAUTION | 项目长期多配置探索，18/18 不能按一次预注册检验解释 |
| Garden of forking paths | NOTE | alpha 和保留门运行前冻结，结果后机制分析单独标注 |
| Correlation != causation | NOTE | 仅解释离线排序干预，不外推线上行为因果 |
| Reverse causality | NOTE | 部署 multiplier 仅使用交互前 target-free 特征 |

## 10. 结论

保留 **冻结 GACR-v3 作为 incumbent**；停止 gate/attenuation/soft-weighting 修改族。经
研究者确认训练规模后，下一主实验改为 `GACR-v6 full-fit`：只扩大 residual fit records，
GRAM 仍冻结。详细方案见 `plan/GRAM_第六阶段_GACR-v6全量残差训练实验计划.md`。
