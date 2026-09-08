# Stage18 S18-1R 容量校正 Gate 审计报告

## Material Passport

- Origin Skill：`academic-research-suite / experiment-agent`
- Origin Mode：`validate + plan`
- Origin Date：2026-09-06T16:30:10.626393+00:00
- Verification Status：`ANALYZED`
- Version Label：`s18_s1r_capacity_audit_v1`
- Protected Data：未读取 I1/I2、D1/D2、official validation/test 或 Sports

## 结论

回顾性容量校正裁决：`S18_1R_RETROSPECTIVE_CAPACITY_PASS_REQUIRES_DISJOINT_CONFIRMATION`。该裁决不覆盖 v0.2 原始 Gate，
也不直接解锁 S18-2；必须先在预先冻结的未见 cohort 上确认。

## Domain 结果

| Domain | raw micro | mechanical ceiling | fraction of capacity | event any-hit | macro event recall | Decision |
|---|---:|---:|---:|---:|---:|---|
| Toys | 0.693548 | 0.745968 | 0.929730 | 1.000000 | 0.917771 | `CAPACITY_ADJUSTED_ACTIONABILITY_PASS` |
| Beauty | 0.363636 | 0.372658 | 0.975791 | 1.000000 | 0.863553 | `CAPACITY_ADJUSTED_ACTIONABILITY_PASS` |

## Beauty denominator 诊断

- Beauty 共 `160` 个非空事件、`1441` 个 actual-pruner items。
- K=8 最多覆盖 `537` 个，raw micro 的机械上限只有 `0.372658`。
- 实际覆盖 `524` 个，即达到可覆盖容量的 `0.975791`。
- depth-1 有 `20` 个事件、`1000` 个 items；其 raw recall `0.157000`，机械上限 `0.160000`。

## 下一强制 Gate

按冻结 hash 顺序选择每域排名 `[1024,2048)` 的 1,024 名未见用户，在 I-1/I0 上复核同一容量校正 Gate。
该 confirmation 通过前，不训练 treatment、不读取 I1/I2，也不启动 S18-2。
