# GRAM 第六阶段：GACR-v4 结果与验证报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-01
- Verification Status: ANALYZED
- Version Label: `phase6_gacr_v4_validation_v1`

## 1. 执行结果

- Experiment ID：`GRAM_PHASE6_GACR_V4`
- 类型：冻结 GACR-v3 residual 的 target-free 用户级收益门控
- 科学计算：**completed**，有效 workload 于 2026-08-01 12:39:23+08:00 开始，
  `summary.json` 于 15:25:43+08:00 生成，实验 exit=`0`
- 数据：Toys、Beauty；每域 3 seeds；每个域-seed 1024 个 fresh validation 用户
- Sports/test：均未读取
- 首次 GPU0 启动因离线 T5 cache 环境缺失而在候选生成前退出，没有科学结果；研究者明确
  授权后以不变科学配置重启
- 实验后 CodeLlama 自动恢复曾因继承了 T5 的 `HF_HOME` 而失败；该工程问题已修复，
  CodeLlama 已于 GPU0 恢复为 `running`，不影响科学结果

## 2. 预注册问题与最终门控选择

v4 冻结 GRAM checkpoint、GACR-v3 residual、候选构造和原 6 维 residual 特征，只训练
一个使用 8 个 target-free 聚合特征的 logistic gate。它预测用户是否适合应用完整 residual；
threshold=`0` 对所有用户应用 residual，是 v3 的精确 identity control。

Toys 和 Beauty 最终都选择 **threshold=`0`**：

| 数据域 | 最优 threshold | gate application | calibration mean NDCG@10 | 最接近的非零门槛 | 非零门槛 mean NDCG@10 |
|---|---:|---:|---:|---:|---:|
| Toys | 0.00 | 100% | 0.246983 | 0.35 | 0.246534 |
| Beauty | 0.00 | 100% | 0.204620 | 0.35 | 0.204501 |

所有预注册非零门槛在 Toys 都降低 overall 与 tail NDCG；Beauty 的 0.45–0.65 虽使 tail
NDCG 极小幅增加，但 overall NDCG 仍低于 threshold 0。因而校准证据不支持丢弃低概率
用户，v4 在部署端精确退化为 v3。

## 3. Fresh development validation

下表的相对增益均以 GRAM 为基线。由于 v4 选择 threshold 0，表中数值同时也是冻结 v3
在本次新 cohort 上的结果。

| 数据域 | seed | overall NDCG@10 相对增益 | 95% CI | Recall@10 绝对增益 | Recall@50 绝对增益 | tail NDCG 相对增益 | broad harm |
|---|---:|---:|---:|---:|---:|---:|---:|
| Toys | 2023 | +0.561% | [-1.812%, +2.973%] | +0.098pp | -0.195pp | -0.510% | 0.293% |
| Toys | 2024 | +1.737% | [-0.552%, +4.222%] | +0.195pp | -0.195pp | +0.865% | 0.293% |
| Toys | 2025 | +1.208% | [-1.345%, +3.892%] | +0.195pp | +0.000pp | +0.943% | 0.391% |
| Beauty | 2023 | +3.869% | [+1.870%, +6.404%] | +0.391pp | +0.391pp | +6.632% | 0.000% |
| Beauty | 2024 | +3.143% | [+1.484%, +5.371%] | +0.293pp | +0.391pp | +5.421% | 0.000% |
| Beauty | 2025 | +4.764% | [+2.474%, +7.700%] | +0.586pp | +0.391pp | +10.708% | 0.000% |

- Toys 三 seed mean overall NDCG@10：**+1.169%**；3/3 点估计为正，但 CI 均跨 0
- Beauty 三 seed mean overall NDCG@10：**+3.925%**；3/3 CI 下界为正
- 六个域-seed cell 宏平均：**+2.547%**；6/6 点估计为正
- Toys mean Recall@10 为 **+0.163pp**，但 mean Recall@50 为 **-0.130pp**；tail
  Recall@50 mean 为 **-0.231pp**
- Beauty mean Recall@10/50 分别为 **+0.423pp/+0.391pp**，tail Recall@50 mean 为
  **+0.585pp**
- v3 与 v4 的 6 组逐用户 `baseline_rank/candidate_rank`、三项指标、changed 和 broad-harm
  字段均逐行完全一致

## 4. v4 相对 v3 的增量判定

预注册要求 v4 在 Toys mean overall NDCG 和双域宏平均上严格超过 v3，并保留 Beauty 与
Toys tail/Recall@50。实际结果为：

| 判据 | 结果 |
|---|---|
| Toys v4 mean NDCG 严格超过 v3 | **失败：增量 0** |
| 双域宏平均严格超过 v3 | **失败：增量 0** |
| Beauty 不低于 v3 | 通过：精确相等 |
| Toys tail NDCG/Recall@50 不低于 v3 | 通过：精确相等，但未修复其相对 GRAM 的 Recall@50 损失 |
| 相对 v3 的增量安全与完整性门 | 通过：精确相等；但 Toys 相对 GRAM 的 Recall@50 仍为负 |

正式决定为 **`RETURN_TO_GACR_V3_STOP_V4_HARD_GATE`**。这只否定 v4 的二元门控配置，
不否定冻结 GACR：后者已连续在新的 cohort 上保持 6/6 overall NDCG 正向点估计，且
Beauty 的证据稳定。

## 5. 门控机制诊断（结果后分析）

以下 AUC 使用 fresh validation 中“确实改变排序且 union-covered”的用户，在结果完成后
计算；它不是预注册决策门，只用于设计下一实验。

| 数据域 | seed | 改善/伤害用户数 | gate probability AUC | 改善用户平均概率 | 伤害用户平均概率 |
|---|---:|---:|---:|---:|---:|
| Toys | 2023 | 110 / 40 | 0.622 | 0.557 | 0.488 |
| Toys | 2024 | 105 / 59 | 0.596 | 0.577 | 0.541 |
| Toys | 2025 | 106 / 54 | 0.588 | 0.559 | 0.522 |
| Beauty | 2023 | 120 / 13 | 0.751 | 0.702 | 0.452 |
| Beauty | 2024 | 120 / 10 | 0.691 | 0.671 | 0.488 |
| Beauty | 2025 | 124 / 13 | 0.735 | 0.666 | 0.432 |

门控概率并非完全无信息，尤其 Beauty 的区分度较高；失败来自**硬开关的收益—覆盖权衡**：
伤害用户本来就是少数，提升 threshold 在过滤伤害的同时移除了更多改善用户，导致 overall
NDCG 下降。该结果支持测试连续 residual 强度，不支持继续搜索另一个 hard threshold。

## 6. 完整性与复现状态

- fit/calibration user overlap = 0；fresh validation 与所有已登记历史 cohort overlap = 0
- parent checkpoint SHA 前后不变；v3 residual 全部冻结；backbone optimizer steps = 0
- gate 部署特征 target-free；`test_data_read=false`；`sports_data_read=false`
- 12 份逐用户 CSV 均为 1024 数据行；6 个 gate checkpoint 均存在
- implementation SHA256：`eacef4f0780990a911535c11ef5a40dc1b9bff0954d314f066bad529ceda6c96`
- config SHA256：`865e39804aba493121e640190f9a1d58869d04e80f4af926d639e14c46a4846d`
- summary SHA256：`5c324c00dd0442c0a80b0f376cdffc2d227f3d6d3ee723f1245984812a323f1b`
- Reproducibility verdict：**CANNOT_VERIFY（未独立重跑）**；内部 identity 与完整性检查通过

## 7. Statistical Interpretation

Overall Confidence：**CAUTION**。

“v4 没有超过 v3”是精确的 identity 结论，不依赖统计显著性。Beauty 相对 GRAM 的三个
CI 均为正；Toys 的三个 CI 仍跨 0，且 Recall@50 有小幅负向，不应把 Toys 描述为已经
确证稳定。长期多配置探索未作全项目多重比较校正；Sports/test 仍然封存。

## 8. Fallacy Scan

覆盖：**11/11 checked**

| 谬误 | 严重度 | 检查结果 |
|---|---|---|
| Simpson's paradox | NOTE | 双域 overall 同向，但 Toys/Beauty 强度和 Recall@50 方向不同，继续分域报告 |
| Ecological fallacy | NOTE | 报告逐用户 coverage、harm 与 AUC，不由域均值断言全部用户改善 |
| Berkson's paradox | CAUTION | Toys/Beauty 均为重复开发域，不是未见确认域 |
| Collider bias | NOTE | target/validation label 未进入部署 gate 或 cohort 选择 |
| Base-rate neglect | NOTE | 明确报告改善/伤害用户基率；Beauty 伤害用户很少 |
| Regression to mean | NOTE | fresh cohort 未按历史极端结果筛选 |
| Survivorship bias | NOTE | 六个 cell 的 1024 用户全部进入结果 |
| Look-elsewhere effect | CAUTION | 长期多方向、多配置探索增加偶然正结果风险 |
| Garden of forking paths | CAUTION | 正式阈值按预注册选择；AUC 诊断明确标为结果后分析 |
| Correlation != causation | NOTE | 只解释离线排序干预，不外推线上行为因果 |
| Reverse causality | NOTE | gate 特征只来自 target 交互前可用信息 |

## 9. 下一步

下一主实验为 **GACR-v5 target-free soft benefit weighting**：冻结 v3 residual 与 v4
gate，只把“应用/不应用”的二元动作改为由 gate probability 连续缩放 residual。这样既
保留所有用户的候选收益，又尝试降低低置信用户的 residual 强度。若校准仍选择 v3 identity，
则停止这一整条 gate/weighting 因素，下一轮转向 residual 训练目标本身。

详细预注册方案见
`plan/GRAM_第六阶段_GACR-v5目标无关软收益加权实验计划.md`。
