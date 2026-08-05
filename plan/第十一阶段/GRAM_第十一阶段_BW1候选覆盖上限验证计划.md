# GRAM 第十一阶段 BW1：Beam Width 候选覆盖上限验证计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan + run + validate
- Origin Date: 2026-08-04
- Verification Status: PREREGISTERED
- Experiment ID: `GRAM_PHASE11_BW1_CANDIDATE_CEILING_VALIDATION_V1`

## 目标

在不训练 GRAM、不修改已确认 PCRF 的条件下，诊断 beam50 之后的主要瓶颈是候选覆盖不足，还是候选内
排序不足。实验仅使用 validation；Toys、Beauty 各固定 512 个用户，独立执行 beam width
`50 / 100 / 200` 的约束解码。

## 冻结项

- GRAM checkpoint：各数据集第一阶段已选定的最佳 checkpoint；
- item-head：P9-S seed 2023 冻结 checkpoint；
- PCRF：`lambda=1.0, beta=0.5, gamma=1.0`；
- cohort：对 validation 用户 ID 以 `sha256("2023:<user>")` 排序后取前 512；
- Trie、lexical ID、history 截断、length penalty、stable tie-break 与 P9-R 一致；
- 每个 width 必须独立调用 constrained beam search，不从 beam200 截断伪造 beam50/100；
- 禁止读取 test / Sports，禁止依据结果调整 width、cohort 或 PCRF。

## Primary measurements

每个 dataset-width 报告：

1. candidate recall（目标进入该 width beam 的比例）；
2. 原始 GRAM Hit@10 / NDCG@10；
3. 冻结 PCRF Hit@10 / NDCG@10；
4. 相对 width50 的 paired Hit@10 delta；
5. 合法、唯一、有限候选比例、峰值显存和运行时间。

## 完整性门控

1. 六个单元均为 512/512 合法且候选无重复、分数 finite；
2. 每个数据集 candidate recall 随 width 单调不降；
3. fresh beam50 相对既有 cache 的 baseline Hit@10 绝对差不超过 `0.002`；
4. GRAM 与 item-head checkpoint 运行前后 SHA256 不变；
5. `test_read=false`、`sports_read=false`。

完整性失败时停止，不解释候选上限，不自动重跑。

## 预注册决策规则

对每个数据集定义：

- `coverage_headroom = candidate_recall@200 - candidate_recall@50`；
- `pcrf_headroom = PCRF_Hit@10(width200) - PCRF_Hit@10(width50)`。

跨两个数据集按以下优先级决策：

1. 若两域 `coverage_headroom < 0.005`：候选覆盖基本饱和，停止扩大 beam，转向生成模型/表征；
2. 若至少一域 `coverage_headroom >= 0.005`，且两域 `pcrf_headroom >= 0`、至少一域
   `pcrf_headroom >= 0.002`：进入候选扩展正式实验，优先采用达到 width200 收益 90% 的最小 width；
3. 若 coverage 有余量但 PCRF headroom 未兑现：候选召回不是充分条件，下一步研究大候选集的分层筛选或
   width-aware calibration，不把增大 beam 作为当前结论；
4. 若两域方向冲突：标记 domain-dependent，仅做机制诊断，不读取 test。

本轮为 512 用户 validation pilot，效应量用于方向与成本判断，不宣称完整数据集统计显著性。
