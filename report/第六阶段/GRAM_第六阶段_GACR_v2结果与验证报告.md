# GRAM 第六阶段：GACR-v2 结果与验证报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-31
- Verification Status: ANALYZED
- Version Label: `phase6_gacr_v2_validation_v1`

## 1. 执行结果

- Experiment ID：`GRAM_PHASE6_GACR_V2`
- 类型：冻结 GCDH-P0 checkpoint 的残差排序训练与 fresh development validation
- 实验状态：**completed，exit code 0**
- 资源状态：自动恢复 CodeLlama 超时，后于 2026-07-31 19:52 人工确认恢复
- 开始时间：2026-07-31 09:52:58+08:00
- 结果生成时间：2026-07-31 19:10:05+08:00
- 数据：Toys、Beauty；每域 3 seeds；每个域-seed 1024 个 validation 用户
- Sports/test：均未读取

2026-07-30 的首次启动因两个被忽略的 GCDH-P0 C1 checkpoint 文件缺失而在候选生成前
退出，没有产生科学结果。经研究者授权后，只使用锁定配置重建 checkpoint；重建 SHA256
与历史值完全一致，随后正式实验完成。最终 `failed_to_restore_resource` 只描述实验结束后
CodeLlama 自动恢复检查失败，不表示 GACR-v2 计算失败。

## 2. 核心结果

校准从 `0.75/1.0/1.25/1.5` 中选择共享部署强度。`1.0` 是满足所有域-seed Recall、
tail 和 broad-harm 门槛的最优配置：

| scale | eligible | calibration mean NDCG@10 相对增益 | changed-user coverage |
|---:|:---:|---:|---:|
| 0.75 | 是 | +1.619% | 17.253% |
| 1.00 | 是 | **+2.081%** | 19.336% |
| 1.25 | 是 | +2.066% | 20.833% |
| 1.50 | 否 | +2.371% | 21.615% |

由于 matched GACR-P0 的 scale 同样为 `1.0`，本轮 `gacr_v2` 与 `matched_p0` 在
全部 6 个域-seed cell 上逐用户完全相同。**统一 scale 校准没有产生 P0 以上的增量，
当前 GACR-v2 配置停止原样重复。**

| 数据域 | seed | overall NDCG@10 相对增益 | 95% CI | Recall@10 绝对增益 | tail NDCG 相对增益 | changed users | broad harm |
|---|---:|---:|---:|---:|---:|---:|---:|
| Toys | 2023 | -0.898% | [-3.714%, +1.666%] | -0.391pp | -1.721% | 12.60% | 0.59% |
| Toys | 2024 | +0.481% | [-2.059%, +3.045%] | -0.098pp | +0.618% | 13.38% | 0.49% |
| Toys | 2025 | -0.890% | [-3.352%, +1.354%] | -0.293pp | -1.620% | 13.18% | 0.49% |
| Beauty | 2023 | +2.510% | [+0.544%, +4.801%] | +0.195pp | +3.603% | 10.74% | 0.10% |
| Beauty | 2024 | +2.396% | [+0.445%, +4.611%] | +0.195pp | +3.603% | 11.23% | 0.10% |
| Beauty | 2025 | +3.774% | [+1.775%, +6.355%] | +0.488pp | +5.637% | 11.33% | 0.00% |

- Toys 三 seed mean overall NDCG@10：**-0.436%**；1/3 正向，全部 CI 跨 0；
- Beauty 三 seed mean overall NDCG@10：**+2.893%**；3/3 正向，全部 CI 下界为正；
- 六个域-seed cell 宏平均：**+1.229%**；4/6 正向；
- validation union coverage：Toys 22.754%，Beauty 21.777%；当前方法最多只能影响约五分之一用户；
- overall changed-user coverage：约 10.7%–13.4%。

## 3. Toys 受损来源诊断

三个 seed 的 Toys changed-user 集合高度一致：

- changed users：129 / 137 / 135；
- 三 seed union：143；intersection：121；intersection-over-union：84.6%；
- pairwise Jaccard：0.860 / 0.913 / 0.915；
- 三个 seed 都发生排序变化的 121 个用户，其跨 seed 平均 NDCG@10 delta 为 `-0.004480`。

因此 Toys 问题不是主要由随机 seed 改动不同用户造成。相同用户被稳定改动，但负向变化
幅度超过正向变化，说明当前统一残差部署对 Toys 的稳定用户群存在系统性排序伤害。
单纯做 seed ensemble 或继续扩大统一 scale 不能直接解决该问题。

Beauty 的 changed-user intersection-over-union 为 82.1%，三 seed 都改变的用户平均
NDCG@10 delta 为 `+0.017119`，说明同一残差机制在 Beauty 上方向稳定且有效。

## 4. 完整性与复现

- Toys/Beauty 重建 C1 checkpoint SHA256 分别为
  `1307ab9d3aa5e56af97fad7276d63cb276260efd3d314b199e350a611c798af6`、
  `5842f45998325cfee47427fd5d323ffdde23fda373016ca86e5164d3d908d2f2`；
- fit/calibration 用户 overlap = 0；fresh validation 与 GCDH、训练及旧 GACR-P0 validation overlap = 0；
- parent checkpoint 前后 SHA 不变；backbone optimizer steps = 0；
- zero-residual identity = true；
- 12 份逐用户 CSV 均为 1024 数据行；
- config SHA256：`0feb0c3970bdf488038fe03372ce29ffaf333b7bfa62a5076df5a6cb1046bbfd`；
- implementation SHA256：`4c303929b8dd66f6813a1e847f4fa92a75a18e720d29f5a6e8ff9817114230d7`；
- summary SHA256：`c09f2293c4615f3157c6e28b73b79f510e069658a1f98a10666ba17b77ca373d`。

## 5. Statistical Interpretation

Overall Confidence：**CAUTION**。

Beauty 的三 seed overall NDCG 区间均为正，构成当前最强的可增长信号；Toys 点估计
混合且区间跨 0，不能声称双域稳定增长。六个 cell 宏平均为正，但域间异质性明显，
不能用宏平均掩盖 Toys 的 Recall 与 tail 损失。本阶段属于开发域探索，多方向和多个
scale 已被比较；置信区间未经全项目多重比较校正，只用于衡量下一轮证据强弱。

## 6. Fallacy Scan

覆盖：**11/11 checked**

| 谬误 | 严重度 | 检查结果 |
|---|---|---|
| Simpson's paradox | CAUTION | 宏平均 +1.229%，但 Toys -0.436%、Beauty +2.893%，必须分域报告 |
| Ecological fallacy | NOTE | 报告逐用户 changed/harm，不由域均值断言所有用户获益 |
| Berkson's paradox | CAUTION | Toys/Beauty 已多轮用于方向开发，不是独立确认域 |
| Collider bias | NOTE | validation outcome 未参与 fit；用户排除规则在结果前锁定 |
| Base-rate neglect | NOTE | 同时报 union coverage、changed coverage 与 broad-harm 基率 |
| Regression to mean | NOTE | validation 用户未按既往极端结果筛选 |
| Survivorship bias | NOTE | 1024 用户全部保留，未按结果剔除 |
| Look-elsewhere effect | CAUTION | 四个 scale 与长期多方向探索增加偶然正结果风险 |
| Garden of forking paths | CAUTION | 本轮配置预注册且未事后换 scale；下一轮仍须重新预注册 |
| Correlation != causation | NOTE | 仅陈述离线排序干预，不外推线上用户行为因果 |
| Reverse causality | NOTE | 特征只使用目标交互前的信息，不使用未来 target 构造候选 |

## 7. 决策与下一门

- **停止当前配置**：不重复“共享 scale 搜索”的 GACR-v2；
- **保留 GACR 方向**：Beauty 三 seed 稳定增长，宏平均正向且 Toys 未灾难性下降；
- **暂不进入 CET x GACR**：先消除 Toys 的系统性稳定伤害，避免组合归因混乱；
- 下一轮只改变一个因素：在保持候选、特征、训练 seed、cohort 和 checkpoint 不变时，
  引入 target-free 的残差安全衰减，并用独立 calibration 选择安全阈值；
- Sports/test 继续保持未读取状态。

## 8. 产物

- `artifacts/phase6/configs/gacr_v2_preregistered.json`
- `artifacts/phase6/gacr_v2/summary.json`
- `artifacts/phase6/gacr_v2/{Toys,Beauty}/*_per_user.csv`
- `artifacts/phase6/gacr_v2/{Toys,Beauty}/residual_seed*.pt`
- `artifacts/phase6/gacr_v2/gpu_telemetry.csv`
- `experiment/phase6/gacr_v2.py`
- `experiment/phase6/run_phase6_gacr_v2.sh`

