# GRAM 第六阶段：GACR-v3 结果与验证报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-31
- Verification Status: ANALYZED
- Version Label: `phase6_gacr_v3_validation_v1`

## 1. 执行结果

- Experiment ID：`GRAM_PHASE6_GACR_V3`
- 类型：target-free per-user residual spread budget 校准与 fresh development validation
- 科学计算：**completed**，`summary.json` 于 2026-07-31 23:28:43+08:00 生成
- 数据：Toys、Beauty；每域 3 seeds；每个域-seed 1024 个 validation 用户
- Sports/test：均未读取
- 资源状态：科学产物已完整生成；截至报告时 CodeLlama 已重新加载到 GPU6，
  runner 仍在等待资源恢复探针返回终态。该状态不影响科学结果。

## 2. 单一改动因素与校准结果

GACR-v3 只增加 target-free 的 per-user residual spread budget，其余 parent checkpoint、
候选构造、特征、训练损失、seed 和 cohort 约束保持不变。Toys 与 Beauty 都选中
最大预注册 budget `0.4`。该 budget 对 validation 中的所有用户得到
`attenuation_rate = 0` 和 `mean_safety_multiplier = 1`，因此实际上退化为不衰减的
GACR-v2。

`incremental_vs_v2` 在 Toys/Beauty 的 overall NDCG@10、Recall@10 和 tail NDCG@10
上全部严格为 `0`。**GACR-v3 作为当前最佳安全版本保留并冻结；只停止继续重复或
调整这个未产生 v2 以上增量的衰减因素。**

## 3. Fresh development validation

| 数据域 | seed | overall NDCG@10 相对增益 | 95% CI | Recall@10 绝对增益 | tail NDCG 相对增益 | changed users | broad harm |
|---|---:|---:|---:|---:|---:|---:|---:|
| Toys | 2023 | +0.859% | [-1.476%, +3.310%] | +0.098pp | +0.166% | 11.82% | 0.39% |
| Toys | 2024 | +2.485% | [-0.001%, +5.287%] | +0.488pp | +1.669% | 12.50% | 0.29% |
| Toys | 2025 | +0.369% | [-2.071%, +2.853%] | +0.098pp | -0.583% | 12.11% | 0.39% |
| Beauty | 2023 | +4.827% | [+2.435%, +8.064%] | +0.488pp | +3.225% | 12.60% | 0.00% |
| Beauty | 2024 | +4.025% | [+1.946%, +6.795%] | +0.391pp | +2.433% | 12.50% | 0.00% |
| Beauty | 2025 | +4.557% | [+2.232%, +7.643%] | +0.391pp | +4.017% | 12.99% | 0.00% |

- Toys 三 seed mean overall NDCG@10：**+1.238%**，3/3 点估计为正，但三个 CI 均跨 0；
- Beauty 三 seed mean overall NDCG@10：**+4.470%**，3/3 CI 下界为正；
- 六个域-seed cell 宏平均：**+2.854%**，6/6 点估计为正；
- Toys 之前的负向点估计没有在这一 fresh cohort 上重现，但 Toys 的不确定性仍高于 Beauty；
- 这些结果支持 **GACR 方向的跨 cohort 可增长性**，不支持“v3 衰减机制优于 v2”。

## 4. 完整性与复现

- fit/calibration user overlap = 0；
- fresh validation 与训练、GCDH 及旧 GACR-P0 validation overlap = 0；
- parent checkpoint SHA 前后不变；残差状态与 v2 一致；backbone optimizer steps = 0；
- safety gate 为 target-free；`test_data_read=false`；`sports_data_read=false`；
- 18 份逐用户 CSV 均为 1024 数据行加 1 行表头；
- config SHA256：`45b8f8e7a1df6096aec63a6d795a623e6ae2f4d842eb45891371b2fc44b84a47`；
- implementation SHA256：`bb97d45f562b0f8be1d54b7abe92af13b0e34da9c4d655e758af918a07fe3050`；
- summary SHA256：`bdf586e3b2aeca4848f071c53ad4599a8d2c3f15636e232cf12f4ddc54c4dd99`。

## 5. Statistical Interpretation

Overall Confidence：**CAUTION**。

Beauty 在 fresh cohort 上的三个 CI 均为正，是最稳定的证据。Toys 的 3/3 正向点估计
改善了跨 cohort 信心，但 CI 仍跨 0，且 seed 2025 的 tail 为负，不应宣称 Toys
已获得确证性稳定改善。项目历史上已进行多方向和多配置探索，当前 CI 未作全项目多重比较校正。

## 6. Fallacy Scan

覆盖：**11/11 checked**

| 谬误 | 严重度 | 检查结果 |
|---|---|---|
| Simpson's paradox | NOTE | 宏平均与两域均同向，但 Beauty 明显强于 Toys，因此继续分域报告 |
| Ecological fallacy | NOTE | 同时报告逐用户 changed coverage 和 broad harm，不由域均值断言全部用户改善 |
| Berkson's paradox | CAUTION | Toys/Beauty 是反复开发域，不是未见的确认域 |
| Collider bias | NOTE | target/validation label 未参与 safety gate 或 cohort 选择 |
| Base-rate neglect | NOTE | 已报告 changed-user 和 broad-harm 基率 |
| Regression to mean | NOTE | fresh validation 用户未按历史极端结果筛选 |
| Survivorship bias | NOTE | 每个域-seed 的 1024 个用户全部进入结果 |
| Look-elsewhere effect | CAUTION | 多 budget、多 seed 与长期多方向探索增加偶然正结果风险 |
| Garden of forking paths | NOTE | budget 候选和安全门预注册；结果后未改阈值 |
| Correlation != causation | NOTE | 仅解释离线排序干预，不外推线上用户行为因果 |
| Reverse causality | NOTE | 机制只使用 target 交互前信息，没有未来 target 反向注入 |

## 7. 决策与下一门

- **保留并冻结 GACR-v3**：新 cohort 上 6/6 点估计为正，Beauty 三 CI 为正，Toys 旧负向没有重现；
- **停止继续优化 v3 衰减因素**：两域均选成 identity，不产生 v2 以上增量；这不等于停止使用 v3；
- **下一主实验进入 CET-v1 × 冻结 GACR-v3 组合**：必须并列 `GRAM`、`CET-v1`、
  `GACR-v3`、`CET-v1+GACR-v3` 四组对照；只有组合超过两个单组件才保留组合；
- Sports/test 继续封存。

## 8. 产物

- `artifacts/phase6/configs/gacr_v3_preregistered.json`
- `artifacts/phase6/gacr_v3/summary.json`
- `artifacts/phase6/gacr_v3/{Toys,Beauty}/*_per_user.csv`
- `artifacts/phase6/gacr_v3/gpu_telemetry.csv`
- `experiment/phase6/gacr_v3.py`
- `experiment/phase6/run_phase6_gacr_v3.sh`
