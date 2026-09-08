# Stage18 S18-1 可作用性与 first-drop 诊断报告

## Material Passport

- Experiment：`s18_s1_actionability_beauty_reservation_recovery / run-0007`
- Verification Status：`ANALYZED`
- Decision：`NO_ACTIONABLE_PREFIX_BOTTLENECK`
- Protected data：D1/D2、official validation/test、Sports 均未读取
- Treatment training：未发生

## Domain Gate

| Domain | headroom | beam200-only | nonempty pruner | K8 recall | CF z drift | Decision |
|---|---:|---:|---:|---:|---:|---|
| Toys | 0.108398 | 222 | 0.554054 | 0.693548 | 0.031767 | `ACTIONABILITY_PASS` |
| Beauty | 0.108398 | 222 | 0.720721 | 0.363636 | 0.004315 | `NO_ACTIONABLE_PREFIX_BOTTLENECK` |

## Execution Contract

- 每个 domain×fold parent 与 item-head 均只见 fold-visible transition。
- parent 从冻结通用 t5-small 独立初始化；未复用未来泄漏 checkpoint。
- beam200 只用于诊断；未训练 PCPS treatment。
- 未自动重试，未自动启动 S18-2。

## Infrastructure Correction

- Recovery of：`run-0002 Toys:I0 + run-0003 Toys:I-1 + run-0007 Beauty:I0 and Beauty:I-1`
- 仅加载原 attempt 的冻结 epoch-10 checkpoints 并重跑诊断；未重新训练。
- 修复范围仅为 NumPy 标量的 JSON 序列化，不改变 cohort、模型、teacher、beam 或 Gate。

## Infrastructure Recovery

- Toys:I0 was carried from run-0002 and Toys:I-1 from run-0003.
- Beauty:I0 and Beauty:I-1 were regenerated from the frozen run-0001 epoch-10 checkpoints.
- No parent or item-head retraining occurred; cohort, beam widths, scores, and Gates were unchanged.
- D1, D2, official test, and Sports remained unread.
- S18-2 was not started automatically.
