# GRAM 第六阶段：F0-T 多源候选覆盖与 Oracle 审计报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-03
- Verification Status: `ANALYZED_WITH_PROTOCOL_DEVIATIONS`
- Version Label: `phase6_f0t_trained_drafter_coverage_oracle_result_v1`
- Source plan: `plan/第六阶段/GRAM_第六阶段_F0多源候选覆盖与Oracle审计实验计划.md`
- Canonical result: `artifacts/phase6/f0_multisource_coverage_oracle/summary.json`
- Scientific decision: `STOP_CANDIDATE_DRAFTING; DO_NOT_TRAIN_VERIFIER`

## 1. 执行结论

F0-T 已在物理 GPU6 完成，从 10 个固定 epoch 中按 Toys/Beauty 内部 holdout
macro NDCG@10 选定共享 epoch 9，随后在每域 256 个隔离 calibration 用户上审计独立
SASRec top-50 候选。未读取 test 或 Sports，未训练 verifier。

SASRec 在 Toys 和 Beauty 分别命中 6 个和 7 个原 GRAM union 未覆盖的 target
users，但两域均低于预注册的每域 10 人下限；Beauty tail 的独占命中为 0。因此即使
不使用存在口径问题的 union Recall@50 字段，P0-T 仍明确失败。固定决定为：

**`STOP_CANDIDATE_DRAFTING; DO_NOT_TRAIN_VERIFIER; F1_NOT_UNLOCKED`**。

这一结论否定的是“继续增加独立 candidate source 再做 verifier”的当前路线，不否定序列信号
本身。后续应转向 item-level 信号与 GRAM lexical path 的接口对齐，而不是扩充 source、
top-k 或融合权重搜索。

## 2. 主要结果

| 域 | n | 原 GRAM union 命中率 | SASRec 单源命中率 | SASRec 独占覆盖 | 独占用户 | 独占门槛 |
|---|---:|---:|---:|---:|---:|---:|
| Toys | 256 | 53.516% | 12.109% | 2.344% | 6 | FAIL（<10） |
| Beauty | 256 | 45.703% | 17.188% | 2.734% | 7 | FAIL（<10） |

分层结果显示，SASRec 信号明显更偏 head：

| 域 / 组 | n | 原 union 命中率 | SASRec 单源命中率 | SASRec 独占覆盖 |
|---|---:|---:|---:|---:|
| Toys head | 128 | 60.938% | 17.969% | 3.125% |
| Toys tail | 128 | 46.094% | 6.250% | 1.562% |
| Beauty head | 128 | 56.250% | 32.031% | 5.469% |
| Beauty tail | 128 | 35.156% | 2.344% | 0.000% |

去重后原 GRAM beam+catalog union 平均大小为 Toys 98.06、Beauty 98.86；加入 SASRec
后分别为 139.21 和 137.27。候选数量明显增加，但独占 target 覆盖很少，且未改善
Beauty tail。

## 3. 训练与 epoch 选择

- Toys 内部 holdout n=1,943；epoch 9 NDCG@10=`0.013797`，高于 epoch 10 的
  `0.012537`。
- Beauty 内部 holdout n=2,241；epoch 9 NDCG@10=`0.022276`，epoch 10 为
  `0.020837`。
- 两域 epoch 9 的 NDCG@10 和最大，因此按冻结规则选为共享 epoch。
- 训练 loss 在 10 epochs 内持续下降；选择依据是内部 holdout NDCG@10，不是外部
  calibration 结果。

## 4. union@50 口径偏差

实现中的 `base` 是 beam top-50 与 catalog top-50 稳定去重后的列表，平均长度约 98；
`ext=dedup(base+sasrec)` 将 SASRec 放在整个 base 之后。然而 `extended_hit50` 又只检查
`ext[:50]`。因此 SASRec 的新候选机械地位于前 50 之外，令两域
`extended_recall50 == base_recall50`、`tail_delta_recall50 == 0`。

这是**评估预算/重排序定义不完整**，不是“SASRec 新候选在集合中不存在”的证据。
故报告不把 summary 中的零 `delta_recall50` 或零 tail delta 当作独立科学结论，也不用它们
声称融合排序无效。

该偏差不改变当前停止决定，因为可直接复算的独占用户数在两域均低于 10，而且
Beauty tail 无独占命中。不允许为修正该口径而临时搜索融合策略；如果未来需要比较固定
总预算，必须在新计划中预先定义 source quota 或 target-free ranker。

## 5. 协议偏离与不可宣称项

计划要求 source-overlap CSV、beam/catalog 分开覆盖、Recall@10、完整 strata、filter 计数、
latency、checkpoint/candidate-cache SHA。实际产物只包含 `summary.json`、`per_user.csv`、
`run.log`、`status.json` 和 GPU lease；`per_user.csv` 只保留 base/SASRec/extended 的部分
hit 字段。因此：

- 本次可以审计独占覆盖、head/tail 独占覆盖和 Sports/test/verifier 禁止项；
- 不能完整复算 beam-vs-catalog 交集、短/长 history、confidence strata、过滤数和延迟；
- `base_oracle_ndcg10`/`extended_oracle_ndcg10` 实际是“target 在集合中则置于第一”的
  set-membership oracle，不是候选排序器指标；
- 本报告的验证状态因而是 `ANALYZED_WITH_PROTOCOL_DEVIATIONS`，不是完整的
  `VERIFIED`。

不对本已停止方向进行结果后补表、增加 source 或重训 SASRec。缺失字段作为完整性
经验进入下一计划的强制产物门。

## 6. 完整性与运行异常

- 最终 runner 状态：`succeeded`，完成时间 `2026-08-03T21:01:31+08:00`。
- `test_data_read=false`、`sports_data_read=false`、`verifier_trained=false`。
- 首次启动因 repo root 未进入 Python path 失败；第二次因离线 HuggingFace cache 路径错误
  失败。修复后才开始有效训练，失败日志保留在同一 `run.log`。
- 最终运行使用物理 GPU6。runner 请求在退出时将 CodeLlama 恢复到 GPU6；但本报告
  完成时状态为 `waiting_for_model`，且状态输出中出现 NVIDIA driver 通信失败。因此只能
  声称“已发出 GPU6 恢复请求”，不能声称 CodeLlama 已重新持有显存。

## 7. Statistical Interpretation 与 Fallacy Scan

Overall confidence: **CAUTION**。独占用户门的失败是可复算的开发证据，但本轮是单 seed、
每域 256 人的机制 pilot，且存在 union@50 口径和产物缺失，不能外推为序列 drafter 的
普遍无效性。

覆盖：**11/11 checked**。

| 风险 | 检查结果 |
|---|---|
| Simpson's paradox | 分域且分 head/tail 报告；Beauty tail 的零独占覆盖未被 head 聚合掩盖。 |
| Ecological fallacy | 保留逐用户 hit 记录，决策不只依赖跨域 macro 值。 |
| Berkson's paradox | cohort 为固定 calibration 抽样；结论限于该开发人群。 |
| Collider bias | 未按 SASRec 输出或 target hit 事后筛选用户。 |
| Base-rate neglect | 同时报告比例与实际独占用户数。 |
| Regression to mean | 未基于极端表现选择 cohort，epoch 选择与外部 calibration 隔离。 |
| Survivorship bias | 每域 256 人全部保留；无结果后剔除。 |
| Look-elsewhere effect | 只使用预注册 seed 和 10 epochs；未做容量、lr 或 top-k 网格。 |
| Garden of forking paths | 不用口径修正为方向增加新机会；决策由可复算独占用户门完成。 |
| Correlation != causation | 只作离线 coverage 机制审计，不声称在线效果。 |
| Reverse causality | drafter 对外部 target 不可见；不使用 target 反向构造候选。 |

## 8. 产物 SHA-256

- summary: `680211f534451f7033e6267f8f3c719448a5a63337a7451e6f844ffee4f58133`
- per-user CSV: `ef4e8ee74e2264eace5921ee22daa495124d69704ec40d14021f1d945946c5d7`
- run log: `a1bb8df943e6d3a6214f2603679a3ccc56d1da9c24a9c9a0acc483ea17a41a0f`
- preregistered config: `408a199098ded4c1ea2ff6936c3280e3b490c395a17d1498bde1d2b9360cfaf1`

## 9. 下一步

F1 和 verifier 均不解锁。下一项只能是独立的 item-to-path 对齐机制 P0：先审计
item-level ranking 信号是否能在固定 lexical Trie 上转化为有界、prefix-consistent 的 token
偏移，再决定是否有依据训练新 backbone/adapter。详见第八阶段 TIPA-P0 计划。
