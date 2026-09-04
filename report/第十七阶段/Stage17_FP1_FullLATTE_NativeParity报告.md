# Stage17 FP1 Full LATTE Native Parity 正式结果报告

## Material Passport

- Origin Skill：`academic-research-suite / experiment-agent`
- Origin Mode：`validate`
- Step：`S17-FP1`
- 日期：2026-09-02
- 状态：`COMPLETED / FP1_NOT_STRONG_PASS / NO_PROMOTION`
- Verification Status：`ANALYZED`
- Overall Confidence：效果方向为 `CAUTION`；Gate 判定为 `SOLID`
- Canonical training：N0 `attempt_001`；N1 `attempt_001`
- Canonical analysis：`artifacts/phase17/fullport/external_d0/attempt_001/analysis.json`
- Analysis SHA256：`30933210b311253f525b6eb1610c8e9289145b3870a8bed3f4e708645948f620`
- 外部评估用户：12,833
- 数据边界：D0 只物化一次；D1/D2、official test、Sports 均未读取

## 1. 结论

N0 `Native-PSID` 与 N1 `Native-LATTE` 的 full-data fresh training、best-checkpoint 冻结及一次性 external D0 评估均已完成。N1 的 primary NDCG@10 比 N0 高 `+0.001394`，但 paired 95% CI 为 `[-0.000874, +0.003753]`，跨 0；Hit@10 同时下降 `-0.001091`。因此：

- 不能拒绝“真实收益为 0 或轻微负向”的可能；
- 不满足 CI 下界大于 0 与 Hit@10 不下降两个预注册条件；
- 正式判定为 `FP1_NOT_STRONG_PASS`，不登记 Native parity 复现成功，不进入 D1；
- 该结果保留为“机制激活但仅有不确定弱正 NDCG 点估计”的本地证据，不能外推为 LATTE 方法本身无效。

## 2. 实验身份与主结果

| 项目 | N0 `Native-PSID` | N1 `Native-LATTE` |
|---|---:|---:|
| canonical training | `formal/n0_native_psid/attempt_001` | `formal/n1_native_latte/attempt_001` |
| backend | pinned official PSID | pinned official LATTE |
| seed / train examples | 2023 / 56,421 | 2023 / 56,421 |
| internal-dev users | 1,283 | 1,283 |
| 完成 epochs / best epoch | 116 / 66 | 119 / 69 |
| best internal-dev NDCG@10 | 0.037027 | 0.034970 |
| checkpoint SHA256 | `96117bb9...d1643` | `897af5f4...25fe` |
| primary inference | beam 500 / identity | beam 500 / `agg_max` |
| external Hit@10 | 0.059378 | 0.058287 |
| external NDCG@10 | 0.030134 | 0.031528 |
| external MRR@10 | 0.021341 | 0.023426 |
| external Hit@50 | 0.152264 | 0.151874 |
| external NDCG@50 | 0.050256 | 0.051742 |

N1 只比 N0 多 1,024 个参数，即 8 个 128-d latent-token embeddings。两臂共享 semantic IDs、训练样本、seed、数据顺序、训练预算与 item evaluator，因而 N1−N0 是本地 Native LATTE 机制的 matched contrast。

## 3. 配对统计与 Gate

| 指标 | N1−N0 点差 | paired 95% CI | 判断 |
|---|---:|---|---|
| NDCG@10 | +0.001394 | [-0.000874, +0.003753] | 弱正点估计，区间跨 0 |
| Hit@10 | -0.001091 | [-0.005221, +0.003195] | 点估计下降 |

- Paired users：12,833；bootstrap replicates：2,000。
- NDCG@10 gain/loss/tie：527 / 512 / 11,794。
- target-rank changed rate：19.64%；完整 ranking changed rate：100%。

| 预注册条件 | 结果 | 状态 |
|---|---|---|
| `ΔNDCG@10 > 0` | +0.001394 | PASS |
| paired CI lower `>0` | -0.000874 | **FAIL** |
| `ΔHit@10 >= 0` | -0.001091 | **FAIL** |
| multi-path rate `>0` | 0.922648 | PASS |
| latent 不塌缩 | normalized entropy 0.997542；collapse rate 0 | PASS |
| 聚合 item ranking 合法 | 全部 primary rankings 非空；聚合有效 | PASS |
| 无 leakage / alias / evaluator drift | collision alias 0；受保护数据未读；用户严格对齐 | PASS |

总判定：`FP1_NOT_STRONG_PASS`。这不是 `WEAK_POSITIVE_FULLPORT` 标签；后者只用于 FP2 的 GRAM full-port 强门槛分支。

## 4. 机制与异质性

### 4.1 N1 机制实际激活

| 机制指标 | 结果 |
|---|---:|
| latent normalized entropy | 0.997542 |
| latent user collapse rate | 0 |
| multi-path item rate | 0.922648 |
| mean duplicate-path rate | 0.850852 |
| beam-500 mean unique items | 73.96 |
| mean aggregation gain NDCG@10 | +0.015860 |
| target-path survival rate | 0.179693 |
| valid generated-path rate | 0.991306 |

这些指标证明 latent forest 与 item aggregation 不是退化空路径。与此同时，机制激活没有转化为满足 Gate 的总体收益，这是本实验最重要的负向机制结论。

`agg_sum` 的 NDCG@10 为 0.031447，相对 N0 为 `+0.001313`，CI `[-0.000899, +0.003621]`，仍不通过；因此 frozen ablation 不能改变主判定。

### 4.2 预注册子组（描述性）

| 子组 | users | ΔNDCG@10 | ΔHit@10 |
|---|---:|---:|---:|
| short history ≤3 | 7,610 | -0.000301 | -0.003154 |
| medium history 4–9 | 3,435 | +0.004031 | +0.001456 |
| long history ≥10 | 1,788 | +0.003541 | +0.002796 |
| head | 4,166 | +0.001363 | -0.002160 |
| mid | 3,921 | +0.000819 | -0.003826 |
| tail | 4,746 | +0.001896 | +0.002107 |

子组提示收益可能集中在更长历史与 tail，但未为子组计算 confirmatory CI，且短历史占多数；这些结果只作异质性线索，不用于翻转 primary Gate。

## 5. 完整性、恢复与可复现性

- N0/N1 训练和推理均来自各自 canonical `attempt_001`；没有把旧 profile、失败 attempt 或 GPU 守护产物混入结果。
- 五臂共享 external D0 sealed bundle，`single_materialization_count=1`；N0/N1 predictions 在原 attempt 内完成。
- external users 严格对齐，五臂 primary ranking 均非空；PSID 重分配 1,337 个 item 后 collision alias 为 0。
- `automatic_retry=false`、`raw_external_projection_reopened=false`；D1/D2/test/Sports read flags 均为 false。
- 统计分析可由冻结 predictions 与 analysis 重算，但本轮没有重新训练或独立 rerun，故 Verification Status 为 `ANALYZED`，不是 `VERIFIED`。
- 共同 external closeout 报告记录恢复授权、prediction SHA 与运行 wall time：`Stage17_FP12_ExternalD0评测准备报告.md`。

## 6. 统计限制与 11 类谬误扫描

- Coverage：`11/11 checked`

| 类型 | 严重度 | 判断 |
|---|---|---|
| Simpson's paradox | CAUTION | 总体弱正，但短历史为负、较长历史为正；不能用单一总体或单一子组概括所有用户。 |
| Ecological fallacy | NOTE | 报告 gain/loss/tie，不把总体均值解释为每个用户都获益。 |
| Berkson's paradox | NOTE | external cohort 是冻结的完整 D0 用户，不是按效果筛选出的样本。 |
| Collider bias | NOTE | 未按 treatment 后变量筛选或调整。 |
| Base-rate neglect | NOTE | 已报告 head/mid/tail 与用户数；本任务不属于诊断测试。 |
| Regression to the mean | CAUTION | checkpoint 由 internal dev 多次评估选出；一次性 external D0 用于隔离该选择效应。 |
| Survivorship bias | NOTE | 12,833 用户全部对齐，未只保留成功预测用户。 |
| Look-elsewhere effect | NOTE | 主比较、beam 与 `agg_max` 事先冻结；`agg_sum` 只作 ablation。 |
| Garden of forking paths | NOTE | 不允许根据当前 D0 改 latent、beam、aggregation、seed 或阈值。 |
| Correlation != causation | NOTE | matched arm 支持本地机制干预比较，但不支持跨数据集或方法普遍性因果结论。 |
| Reverse causality | NOTE | treatment 在训练前指定，不存在结果反向决定 arm 的路径。 |

未报告 p-value；正式判断直接使用预注册 point-delta、paired bootstrap CI 与 Hit 门槛。secondary metrics 与子组不承担晋级判断，因此不以未校正的探索性差异替代 primary Gate。

## 7. 冻结决策

1. 关闭 Native LATTE 当前配置的推进资格；不读取 D1，不在 D0 上追调。
2. 不以 `agg_sum`、更大 beam、换 seed 或子组正向结果追认 parity。
3. 保留 N1 弱正 NDCG 点估计与已激活机制，作为论文中的本地不确定/负向复现证据。
4. Stage17 下一科学主线转向计划已注册的 FP3 Full SETRec；具体启动条件由 FP2 终报的下一步章节约束。
