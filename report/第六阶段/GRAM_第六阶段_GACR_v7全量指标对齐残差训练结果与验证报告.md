# GRAM 第六阶段：GACR-v7 全量指标对齐残差训练结果与验证报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-02
- Verification Status: ANALYZED — `CALIBRATION_GATE_STOPPED_BEFORE_FRESH_VALIDATION`
- Version Label: `phase6_gacr_v7_metric_aligned_full_fit_v1`
- Source plan: `plan/第六阶段/GRAM_第六阶段_GACR-v7全量指标对齐残差训练实验计划.md`

## 1. 执行结论

GACR-v7 已完成预注册的全量 fit 训练与 calibration；**未进入 fresh validation**。唯一改动是以 NDCG@10/Recall@50 截断敏感、head/tail 等权的 pairwise softplus 替换 v6 的 highest-negative hinge loss。6 个 domain-seed calibration cell 全部未满足预先冻结的非劣安全门，runner 因而按协议写出 `STOPPED_BEFORE_FRESH_VALIDATION_CALIBRATION_GATE_FAILED` 并恢复 GPU0 上的 CodeLlama。

这是一项有效的校准否决，而不是 fresh-validation 的负结果；不应把它同 v3/v6 在 fresh cohort 上的 NDCG 结果并列比较，也不能据此选择 loss 权重、训练步数或 deployment scale。

## 2. 完整性与执行审计

- Toys 与 Beauty 各训练 seeds 2023/2024/2025，共写出 6 个 finite residual checkpoint。
- 全量 fit records 已使用；fit/calibration user overlap=`0`；GRAM C1 parent checkpoint SHA 在训练前后不变；backbone optimizer steps=`0`。
- 每个 seed 均完成 30 个 full-batch steps；所有 zero-weight head/tail record 数均为 `0`；identity initialization 的排序一致率均为 `1.0`。
- `status.json` 记录为 `completed_without_validation`；fresh validation、Sports 与 test 均未读取。
- 科学产物位于 `artifacts/phase6/gacr_v7/`；训练日志和 GPU telemetry 已保留。

## 3. Calibration 非劣门

冻结门为每个 domain-seed 相对 frozen GRAM 同时满足：broad harm `<=1%`、overall Recall@10/50 delta `>=-0.2pp`、tail Recall@50 delta `>=-0.4pp`、tail NDCG@10 absolute delta `>=-0.0005`。

| 域 | seed | broad harm | Recall@10 delta | Recall@50 delta | tail Recall@50 delta | tail NDCG@10 delta | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| Toys | 2023 | 0.391% | +0.000pp | +0.000pp | +0.000pp | -0.003790 | tail NDCG 门失败 |
| Toys | 2024 | 0.781% | -0.391pp | +0.000pp | +0.000pp | -0.004813 | Recall@10、tail NDCG 门失败 |
| Toys | 2025 | 0.781% | -0.391pp | +0.000pp | +0.000pp | -0.003790 | Recall@10、tail NDCG 门失败 |
| Beauty | 2023 | 0.391% | -0.391pp | +0.000pp | +0.000pp | -0.002604 | Recall@10、tail NDCG 门失败 |
| Beauty | 2024 | 0.781% | -0.781pp | +0.000pp | +0.000pp | -0.002604 | Recall@10、tail NDCG 门失败 |
| Beauty | 2025 | 0.781% | -0.781pp | +0.000pp | +0.000pp | -0.002604 | Recall@10、tail NDCG 门失败 |

所有 cell 的 broad-harm、Recall@50 与 tail Recall@50 检查通过；tail NDCG@10 在全部 6 个 cell 失败，overall Recall@10 在 5/6 个 cell 失败。tail NDCG@10 的损失为阈值允许值的约 5.2–9.6 倍，所以不能将其解释为单次离散交换或边缘不确定性。

## 4. 优化健康度与机制解释

6 个 seed 的训练损失均下降，分别从 Toys 的 `0.5753` 降至 `0.4905–0.4940`、Beauty 的 `0.5474` 降至 `0.4691–0.4722`；末步 gradient norm 为 `0.0125–0.0252`，checkpoint 均 finite。因此停止不是数值发散、空样本、loss 权重为零、checkpoint 篡改或资源中断所致。

更合理的机制解释是：仅由冻结 base rank 构造的截断敏感 pairwise objective，会驱动足以改变约 10%–20% covered-user 排名的 residual，但它没有获得能区分“同一 rank 交换是否对应真实生成概率差”的表示。它在 calibration 中系统性降低 tail top-10 排名，且常伴随 Recall@10 损失；因此关闭“保持 rank-only 6 维特征、只替换相邻 loss”的路线。

## 5. 统计解释与局限

本次没有 fresh validation 逐用户输出，因而不能报告 bootstrap CI、v7-v3 增量或显著性，也不应宣称 v7 比 v3/v6 更差。可以作出的结论仅限于：在预注册 calibration 样本和冻结安全界下，这个 rank-only 指标对齐 loss **不具备进入新 fresh cohort 的资格**。

校准样本为每域 128 head + 128 tail，适合识别预注册的明显伤害，不能用于证明等效或泛化。跨版本的开发观察也不能替代 Sports/test confirmation；两者在本轮均保持封存。

## 6. Fallacy Scan

覆盖：**11/11 checked**。

| 谬误 | 结论 |
|---|---|
| Simpson's paradox / ecological fallacy | 已同时检查 overall 与 head/tail；未让总体 Recall@50 的持平掩盖 tail NDCG@10 下降。 |
| Berkson's / collider bias | calibration 与 fit user 隔离；未读取 fresh labels 来调整配置。 |
| Base-rate neglect | 同时报出 recall、NDCG、coverage 变化与 broad harm。 |
| Regression to mean / survivorship bias | 每个冻结 calibration cell 都纳入门，不筛选“好”的 seed。 |
| Look-elsewhere / garden of forking paths | 未调整 `0.25`、步数、模型容量或 scale；失败后不回跑 v7。 |
| Correlation != causation / reverse causality | 结论限于离线校准排序干预，不作线上行为或因果主张。 |

## 7. 正式决定与下一步

正式决定：**`KEEP_FROZEN_GACR_V3; CLOSE_RANK_ONLY_METRIC_LOSS_VARIANT; ADVANCE_TO_GACR_V8_PLANNING`**。

- v3 继续作为 deployment incumbent；v6 仍只保留为已分析的全量-fit 对照。
- 不重跑 v7，不调整 Recall@50 权重，不扫描 LambdaRank/ListMLE/SoftNDCG，也不重新打开 v4/v5 的 gate/soft-weighting 家族。
- 下一实验为 GACR-v8：在相同固定候选 union 上恢复真实 GRAM item-path likelihood 与 prefix uncertainty，并以小型有界 listwise residual 检验 path-score 与候选交互是否提供独立增益。详细预注册见 `plan/第六阶段/GRAM_第六阶段_GACR-v8路径感知列表残差校准实验计划.md`。
