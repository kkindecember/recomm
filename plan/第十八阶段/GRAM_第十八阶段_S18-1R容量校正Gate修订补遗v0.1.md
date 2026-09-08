# GRAM 第十八阶段：S18-1R 容量校正 Gate 修订补遗 v0.1

## Material Passport

- Parent Plan：`GRAM_第十八阶段_PCPS-GRAM词法锚定协同前缀生存与低风险验证计划v0.2.md`
- Parent Result：`artifacts/phase18/s1_actionability/summary.json`
- Created：2026-09-07
- Status：`RETROSPECTIVE_AUDIT_AUTHORIZED / DISJOINT_CONFIRMATION_NOT_STARTED`
- Scope：修复 S18-1 Gate 4 的 top-K 容量不一致；不覆盖 v0.2 原始结果，不授权自动进入 S18-2
- Protected Data：I1/I2、D1/D2、official validation/test、Sports 继续封存

## 1. 修订原因

v0.2 将 `K=8` hard-negative set 的 item-level micro recall 门槛冻结为 `>=0.50`。S18-1 Beauty
结果为 `524/1441=0.363636`，因而按原合同得到 `NO_ACTIONABLE_PREFIX_BOTTLENECK`。逐事件容量审计随后发现：

- 160 个非空事件全部至少命中一个 actual pruner；
- 20 个 depth-1 事件各有 50 个 actual-pruner items，共占 1,441 个 item 分母中的 1,000 个；
- K=8 对这些事件的 raw recall 上限只能是 `8/50=0.16`；
- 全部 Beauty 事件的可覆盖 item 总数为 537，因此 raw micro 的机械上限只有
  `537/1441=0.372658`，严格低于原门槛 0.50；
- 实际命中 524，即达到可覆盖容量的 `524/537=0.975791`。

这说明原 Gate 4 混合了“选择器质量”和“每事件 pruner cardinality”，且在 Beauty 的实际分母下
数学不可达。该发现不把 v0.2 的失败事后改写为通过，而是触发一个具名的新 Gate 修订与独立确认。

## 2. 回顾性容量审计 S18-1R-A

只读取四份已冻结的 S18-1 `per_user_diagnostics.jsonl`，逐行从 actual/selected item set 重新计算：

```text
raw_micro_recall@8 = sum(intersection) / sum(|actual_pruners|)

raw_micro_mechanical_ceiling@8 =
    sum(min(8, |actual_pruners|)) / sum(|actual_pruners|)

capacity_normalized_recall@8 =
    sum(intersection) / sum(min(8, |actual_pruners|))

event_any_hit@8 = mean(intersection > 0)
```

原 raw micro recall 保留为描述指标，不再单独决定可作用性。审计必须按 first-drop depth 分层报告，
不得删除或降权 depth-1 事件。

回顾性修订 Gate：两域分别满足：

1. `capacity_normalized_recall@8 >= 0.80`；
2. `event_any_hit@8 >= 0.80`；
3. 原 S18-1 Gate 1、2、3、5、6 继续通过。

即使本 Gate 通过，也只得到
`RETROSPECTIVE_CAPACITY_PASS_REQUIRES_DISJOINT_CONFIRMATION`，不得直接启动 S18-2。

## 3. 未见 cohort 确认 S18-1R-B

为避免依据已经观察到的 Beauty 结果事后放宽标准，修订 Gate 必须在未见用户上确认：

- eligibility、fold、parent、item-head、frequency、Trie、beam50/200、PCRF 与 K=8 全部不变；
- 每域继续使用 I-1/I0 共同 eligible intersection；
- 沿原冻结排序键 `sha256("S18-1|2023|<domain>|<user>")` 排序；
- 原 cohort 为排名 `[0,1024)`；新 confirmation cohort 固定为 `[1024,2048)`；
- 两个 fold 使用同一新 cohort；Toys 与 Beauty 都必须运行，不允许 Beauty-only confirmation；
- 只复用 run-0001 已冻结的四个 epoch-10 fold-local parent/item-head checkpoints，不重新训练；
- cohort SHA 必须在任何新用户生成前写入 config；不得根据 target、first-drop 或效果过滤用户。

确认 Gate 与第 2 节完全一致。任一域失败则 `S18_1R_DISJOINT_CONFIRMATION_FAILED`，PCPS 主线关闭。
两域均通过才得到 `S18_1R_ACTIONABILITY_REPAIRED_PASS`，随后才可单独授权 S18-2。

## 4. 后续边界

- 不通过增大 K、删除 depth-1、只报 macro、降低 0.80 门槛或 Beauty-only 复核来挽救结果；
- S18-2 仍需保留 `C0/A0/M0/S0` 四臂，并由 `A0/S0` 判断 CF guidance 是否真的有增量；
- alpha 仍只按 gradient ratio 从 `{0.1,0.3}` 冻结，不按 accuracy/NDCG 选择；
- S18-1R-B 与 S18-2 均不得读取 I1/I2；S18-3 通过前不得读取 D1；
- 回顾性审计为 CPU only；未见 cohort bounded generation 预计超过 10 分钟，必须使用具名 tmux 和独立 status；
- 任一步失败不自动重试、不自动进入下一步。
