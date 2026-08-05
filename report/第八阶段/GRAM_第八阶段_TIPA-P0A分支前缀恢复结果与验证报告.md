# GRAM 第八阶段：TIPA-P0A 分支前缀恢复结果与验证报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-03
- Verification Status: `ANALYZED_WITH_PROTOCOL_DEVIATIONS`
- Version Label: `phase8_tipa_p0a_branching_recovery_result_v1`
- Source plan: `plan/第八阶段/GRAM_第八阶段_TIPA-P0商品到词法路径对齐审计实验计划.md`
- Canonical result: `artifacts/phase8/tipa_p0_branching_recovery/summary.json`
- Execution status: `succeeded`
- Scientific decision: `STOP_TIPA_NO_PATH_REALIZATION`

## 1. 结论

TIPA-P0A recovery 已完整执行。恢复后的 `branching_teacher_path` 规则使 Toys、Beauty
都获得 256/256 个可训练分支前缀，故首次运行的 `insufficient non-null prefix records`
问题已经排除。模型训练、解码、审计和 CodeLlama 恢复均正常完成。

但核心机制未成立。C 相对固定 scalar 对照 B 的 teacher→path Kendall agreement 在两域
都下降；teacher 独占候选几乎没有被 C 兑现到 beam@50。Toys 同时出现 top-10 下降和
3.125% broad harm；Beauty 虽有 top-10 正增量，却只有 1 个 teacher-exclusive user 被
兑现，且少于 B 的 2 个。预注册要求两域全部通过，因此固定决定为：

**`STOP_TIPA_NO_PATH_REALIZATION; TIPA_P1_NOT_UNLOCKED`**。

这个结论否定的是当前“冻结 item teacher + 共享有界 prefix adapter”的转移接口，不否定
Beauty 上观察到的局部排序变化，也不外推为所有 item-to-token 对齐方法无效。

## 2. 预注册门控结果

| 门控 | Toys | Beauty | 要求 | 判定 |
|---|---:|---:|---:|---|
| branching-prefix 可用记录 | 256/256 | 256/256 | >=240/256 | 双域 PASS |
| Kendall(C-B) | -0.008157 | -0.018769 | >=+0.10 | 双域 FAIL |
| teacher-exclusive users | 6 | 14 | 描述量 | — |
| B 兑现数 | 0 | 2 | 对照 | — |
| C 兑现数 | 0 | 1 | >=5 且 >=B | 双域 FAIL |
| Recall@10 Δ | -0.78125pp | +1.5625pp | 与 NDCG@10 不同降且至少一项为正 | Toys FAIL |
| NDCG@10 Δ | -0.007506 | +0.007203 | 同上 | Toys FAIL |
| broad harm | 3.125%（8/256） | 0%（0/256） | <=1% | Toys FAIL |
| tail Recall@50 Δ | 0pp | 0pp | >=-0.5pp | 双域 PASS |

门控失败不是边界舍入造成的：Kendall 距离 +0.10 下限很远，C 的独占兑现数分别比下限
少 5 和 4 人，Toys broad harm 比 1% 上限高 2.125 个百分点。

## 3. 排序指标与配对不确定性

| 域 | C vs A Recall@10 Δ | NDCG@10 Δ | Recall@50 Δ | MRR Δ |
|---|---:|---:|---:|---:|
| Toys | -0.78125pp | -0.007506 | +0.390625pp | -0.006705 |
| Beauty | +1.5625pp | +0.007203 | +1.171875pp | +0.003393 |

为描述离散样本不确定性，本报告对 256 个用户的配对差执行 seed 2023、5,000 次用户级
bootstrap；这是结果后验证，不改变预注册门控：

| 域 / 指标 | 均值 | 95% bootstrap interval | 改善/伤害用户 |
|---|---:|---:|---:|
| Toys Recall@10 | -0.007812 | [-0.035156, 0.019531] | 6 / 8 |
| Toys NDCG@10 | -0.007506 | [-0.020637, 0.005882] | 19 / 26 |
| Toys Kendall(C-B) | -0.008157 | [-0.017972, 0.001818] | 113 / 142 |
| Beauty Recall@10 | +0.015625 | [0.003906, 0.031250] | 4 / 0 |
| Beauty NDCG@10 | +0.007203 | [-0.003361, 0.018852] | 27 / 15 |
| Beauty Kendall(C-B) | -0.018769 | [-0.027685, -0.009790] | 98 / 154 |

Beauty 的 Recall@10 正变化具有配对样本支持，但它没有伴随 teacher-exclusive path
realization 或 Kendall 对齐改善，因此不能作为当前机制通过的替代终点。两域方向相反也说明
不能用 Beauty 挽救 Toys 或宣称跨域稳定收益。

## 4. 分层结果

| 域 / 组 | Recall@10 Δ | NDCG@10 Δ | Recall@50 Δ | broad harm |
|---|---:|---:|---:|---:|
| Toys head | -0.78125pp | -0.000625 | +0.78125pp | 4.6875% |
| Toys tail | -0.78125pp | -0.014388 | 0pp | 1.5625% |
| Beauty head | +3.125pp | +0.009806 | +2.34375pp | 0% |
| Beauty tail | 0pp | +0.004601 | 0pp | 0% |

Toys tail 的 NDCG@10 和 MRR 分别下降 0.014388 与 0.015519；Beauty 的主要 Recall@10
增量集中在 head。该异质性属于机制诊断，不构成事后选择子组继续 TIPA 的许可。

## 5. 训练、完整性与资源

- 每域 256 个 fit prefix records；恢复后 null/single-child 记录为 0。
- zero-adapter identity max-abs delta=0；teacher subtree mass 最大误差
  `1.1920929e-07`，低于 `1e-6`。
- fit/calibration user overlap=0；GRAM optimizer steps=0；teacher optimizer steps=0。
- `test_read=false`、`sports_read=false`、`external_development_read=false`。
- 逐用户 arm 行数为 768/域，逐用户与逐前缀主表均为 256/域；所有值通过程序 finite/audit。
- Toys/Beauty decode 分别耗时 287.68/457.87 秒；完整 runner 约 15 分 36 秒。
- 物理 GPU6；PyTorch peak allocated 为 6963/6976 MiB，peak reserved 为
  15638/9184 MiB；GPU telemetry 观察到全卡最高 used 24038 MiB，未超过 30720 MiB
  总租约。
- 结束后 CodeLlama 已在物理 GPU6 恢复为 `state=running`，约持有 30.3 GiB。

训练轨迹 finite 且梯度非零，但固定 100 steps 并没有产生正的跨域 teacher→path transfer。
Toys step-1/100 loss 为 1.2784/1.5827；Beauty 为 3.0628/1.6532。由于这是随机 batch
训练轨迹且未报告同一固定 evaluation loss，不能把首末 batch loss 直接解释为收敛与否。

## 6. 协议偏离与可宣称边界

首次启动按原随机深度规则失败；研究者授权后仅将 prefix depth 改为 target-free 的
`branching_teacher_path`，并以独立 config、runner 和输出目录运行。该 recovery 已写入原计划
第 11 节，原失败日志和哈希均保留。

计划要求独立的 strata、integrity、timing 与 manifest 文件，并要求 runner 在 succeeded 前
逐项验证。实际 strata/integrity/timing 嵌入各域 `summary.json`，没有独立 manifest；主要
逐用户、逐前缀、telemetry、status 和 lineage 信息存在且可审计。因此本报告状态为
`ANALYZED_WITH_PROTOCOL_DEVIATIONS`，不是完整 `VERIFIED`。

本轮没有确定性重跑。可复现性判定为 `CANNOT_VERIFY`；config、代码与输入 lineage 已有
SHA 锁，但这不等价于结果已重复复现。

## 7. Statistical Interpretation 与 Fallacy Scan

Overall confidence: **CAUTION**。停止决定由预注册的双域机制门直接支持；但这是单 seed、
每域 256 用户的 train-only pilot，配对 bootstrap 为结果后描述，不能将局部增益外推为正式
泛化结论。

覆盖：**11/11 checked**。

| Fallacy | 检查结果 |
|---|---|
| Simpson's paradox | overall 与 head/tail 同报；Toys/Beauty 方向相反，没有用 macro 聚合掩盖。 |
| Ecological fallacy | 机制门基于逐用户记录与实际人数，不由域均值推断每个用户。 |
| Berkson's paradox | cohort 为固定 hash 抽样；结论限定于该 calibration 样本。 |
| Collider bias | 未按 C 的成功、teacher margin 或 target hit 事后筛选主 cohort。 |
| Base-rate neglect | teacher-exclusive 同时报比例含义与 6/14、兑现 0/1 等人数。 |
| Regression to mean | 未按极端 outcome 选样本或更换 seed。 |
| Survivorship bias | 两域 256 个用户全部进入审计，无结果后剔除。 |
| Look-elsewhere effect | 单 seed、单 adapter、固定 steps/bound；未搜索超参数。 |
| Garden of forking paths | recovery 只修复可用前缀构造；不因 Beauty 正增量改门或追加 TIPA 变体。 |
| Correlation != causation | 只陈述离线机制与配对排序变化，不声称线上因果收益。 |
| Reverse causality | adapter 特征不读 calibration target；target 只用于隔离评估。 |

## 8. 产物 SHA-256

- canonical summary: `a0b6b06717d234c224daa7cf2fc3f3bc08599b364ffdc360ae66ced9d9d09c2b`
- Toys summary: `1ac1e6755d02aaf92753b487be62a5d761f2926aa292cfbd48aadb980e802e96`
- Beauty summary: `5ce602cf212115a6d1f407127348c49b0772e97165d2ffa791dd9049c5667500`
- Toys per-prefix: `9bdb4f27a07f1c2d9b5e5d93873844a02f49ba397760dfc178055a25a835c63d`
- Beauty per-prefix: `b4d80fd10ef40ed02b324800c9f3f9a12b1f1bc7e67d34b50cd9c8ac5376e2fd`
- Toys per-user-arms: `7b78c3d3b1e1217884228e28bd63923031a3ac9f81edf8d370c26965cb3b060e`
- Beauty per-user-arms: `6f3142b5d33fc8fb9c84bdfe5256c816ddd36faa62baef8b0e03606cc64859d3`
- run log: `491410ce7b5975958728f507819f2a32af428fdc8fe48384f1231660efc90ae3`
- final status: `1e15ca4f296bb527762b69741bf6904fd869363b9544ade974c30d75170e0b7d`
- preregistered recovery config: `6c0785772cbbdaa70e19d422e1d8fe9df55a4964c9103c0b04f5173943d8a695`

## 9. 下一步

不设计或启动 TIPA-P1，不读 fresh development、Sports 或 test，不搜索同机制变体。下一项
仅允许执行 `TIPA-D0` analysis-only 失败归因审计：复用本轮现有 CSV/JSON，定位负迁移发生
在 teacher→path 对齐、prefix perturbation 还是 beam realization 环节。D0 无论得到什么结果，
都不能推翻当前 STOP 或解锁训练；它只用于决定后续应转向独立结构假设，还是结束方法迭代并
整理 falsification-first 负结果研究。
