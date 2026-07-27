# GRAM 第三阶段 S0 离线诊断报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-07-22
- Verification Status: ANALYZED
- Version Label: s0_offline_v1
- Upstream: `plan/GRAM_第三阶段_创新探索与渐进式实验计划.md`

## 1. 执行状态

S0 仅使用 CPU 和既有 best-checkpoint 预测，不训练模型、不占用 GPU。重排公式和小型网格只允许在 validation 上选择；本报告不会用 test 结果调参。

| 数据集 | validation 预测 | 状态 |
|---|---|---|
| Toys | 有 | ANALYZED |
| Beauty | 有 | ANALYZED |

## 2. Toys validation 结果

### 2.1 Lineage 与完整性

- 用户数：19,412
- 商品数：11,924
- 目标错配：0
- 未映射 gold：0
- 未映射 beam prediction：0
- CPU wall time：110.1 秒

### 2.2 Relation coverage

| k | 最近商品覆盖率 | 最近 20 条历史并集覆盖率 | 平均并集候选数 |
|---:|---:|---:|---:|
| 1 | 2.823% | 4.425% | 5.68 |
| 3 | 5.167% | 8.459% | 16.29 |
| 5 | 6.388% | 10.957% | 26.49 |
| 10 | 7.938% | 14.409% | 50.93 |
| 15 | 9.036% | 16.799% | 74.35 |
| 20 | 9.870% | 18.550% | 97.27 |

### 2.3 Beam 上限与离线重排

| 指标 | Baseline | 选中重排 | 变化 |
|---|---:|---:|---:|
| Recall@5 | 0.090923 | 0.091181 | +0.000258 |
| Recall@10 | 0.119411 | 0.122656 | +0.003245 |
| NDCG@10 | 0.076275 | 0.077585 | 1.718% relative |
| Beam-50 oracle Recall@5/10 | — | 0.211931 | 目标在 beam 内即可达到 |

选中配置：`k20_c0.25_w0.2`，即 k=20、consensus weight=0.25、fusion weight=0.2、recency decay=0.9。

预注册晋级判定：**GO**（primary gate=true，subgroup gate=true）。

## 3. Beauty validation 结果

### 3.1 Lineage 与完整性

- 用户数：22,363
- 商品数：12,101
- 目标错配：0
- 未映射 gold：0
- 未映射 beam prediction：0
- CPU wall time：121.9 秒

### 3.2 Relation coverage

| k | 最近商品覆盖率 | 最近 20 条历史并集覆盖率 | 平均并集候选数 |
|---:|---:|---:|---:|
| 1 | 2.102% | 3.220% | 5.83 |
| 3 | 4.185% | 6.439% | 16.63 |
| 5 | 5.299% | 8.313% | 26.97 |
| 10 | 7.374% | 11.930% | 51.71 |
| 15 | 8.814% | 14.385% | 75.30 |
| 20 | 9.909% | 16.366% | 98.02 |

### 3.3 Beam 上限与离线重排

| 指标 | Baseline | 选中重排 | 变化 |
|---|---:|---:|---:|
| Recall@5 | 0.077628 | 0.077405 | -0.000224 |
| Recall@10 | 0.108751 | 0.109198 | +0.000447 |
| NDCG@10 | 0.064974 | 0.065280 | 0.472% relative |
| Beam-50 oracle Recall@5/10 | — | 0.208112 | 目标在 beam 内即可达到 |

选中配置：`k20_c0_w0.05`，即 k=20、consensus weight=0.0、fusion weight=0.05、recency decay=0.9。

预注册晋级判定：**STOP_OR_MODIFY**（primary gate=false，subgroup gate=false）。

## 4. 整体晋级决策

双数据集整体判定：**MODIFY**。

Beauty/Toys 已按同一协议完成，但至少一个数据集未达到门槛。按照预注册规则，当前不得直接进入 S1；先进行一次有边界的 S0b 可靠性拒绝探针，判断能否在不使用目标信息的条件下避免 no-CF-covered 用户退化。S0b 属于结果后提出的探索性修正，必须与原 S0 分开标记，不能冒充预注册验证。

- 结果是离线相关性证据，不等于训练后的因果增益。
- 用户是配对评测单位，不是独立训练重复；S0 不做跨 seed 显著性主张。
- test 尚未用于公式或超参数选择。只有双数据集 validation 配置锁定后，才允许一次性 test 诊断。
- `ANALYZED` 不等于 `VERIFIED`；独立复跑前不得升级验证状态。

## 5. 产物

- 脚本：`experiment/phase3/s0_offline_diagnostics.py`
- 机器可读结果：`artifacts/phase3/s0/<dataset>/validation/`
- 后台状态：`experiment/phase3/phase3_s0_status.json`
- 后台日志：`artifacts/phase3/logs/s0_toys_validation.log`；`artifacts/phase3/logs/s0_beauty_validation.log`
