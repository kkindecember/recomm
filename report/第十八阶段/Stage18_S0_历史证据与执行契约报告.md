# Stage18 S0 历史证据与执行契约报告

## Material Passport

- Origin Skill：`academic-research-suite / experiment-agent (run mode)`
- Origin Date：2026-09-03T05:44:35.331817+00:00
- Verification Status：`VERIFIED`
- Version Label：`stage18_s0_audit_v1`
- Experiment ID：`s18_s0_audit`
- Attempt ID：`s18_s0_audit_attempt_001`
- Canonical Plan：`plan/第十八阶段/GRAM_第十八阶段_PCPS-GRAM词法锚定协同前缀生存与低风险验证计划v0.2.md`

## 结论

S18-0 总 Gate 为 **ENGINEERING_PASS**。本步骤没有使用 GPU、没有训练模型、没有运行 bounded generation，
也没有读取 D1、D2、Sports、原始 official validation/test 或 Stage17 external-D0 原始目标/预测。

## 机器审计结果

| Contract | Result | Passed / Total |
|---|---:|---:|
| 历史证据 SHA + 数值回溯 | passed | 79 / 79 |
| 数据权限与封存边界 | passed | 23 / 23 |
| Phase9 frozen PCRF 重建 | passed | 40 / 40 |

Phase9 全量 derived rank cache 共 19412 行，重新聚合后的最大指标绝对误差为
`6.661e-16`，门限为 `1.0e-12`。另在 512-user fresh-beam
审计 cache 上确认 candidate set、sequence top10、PCRF top10 与 target rank 均逐行一致。

这里读取的是 Phase9 已冻结且已做 SHA 绑定的派生 rank cache，只用于复核历史口径；没有重新打开
原始 official test 数据、候选分数或 target item，也不得将该 cache 用于方法、alpha、epoch 或 checkpoint 选择。

## Baseline 与禁止路径冻结

- 训练对照：`C0_CONT`，相同 fold、相同预算的 matched continuation；
- 主基线视图：`C1_CONT_PCRF`，即 native lexical GRAM + frozen Phase9 PCRF；
- `alpha=0` 必须退化等价于 C0；beam 固定为 50，identifier 固定为 native lexical，Trie 固定；
- 共冻结 12 条 hard exclusions，包括 identifier replacement、推理后候选准入、
  重跑 C1/C2/A0、PAWA 换权重、外部 fold/seed 挽救和自动 scientific retry。

## 数据边界

- 后续 internal runner 仅可读取两域 D0 shadow 的两个精确路径，并必须只返回 `shadow_items[:-2]`；
- `shadow_items[-2]` 的已消耗 external D0 target 与 `shadow_items[-1]` guard 不得进入 internal runner；
- D1/D2、原始 monolithic sequence、Sports、external-D0 raw/materialized examples 与 predictions 均 fail closed；
- S18-0 不自动解锁 S18-1。

失败检查：无。

## Gate 与下一步

当前裁决：`S18_0_COMPLETE_AWAIT_S18_1_AUTHORIZATION`。只有研究者另行明确同意，才可启动 S18-1 的 CPU + bounded generation
可作用性诊断；本报告本身不构成该授权。
