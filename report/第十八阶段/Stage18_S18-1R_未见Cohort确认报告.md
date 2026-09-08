# Stage18 S18-1R 未见 Cohort 确认报告

## Material Passport

- Origin Skill：`academic-research-suite / experiment-agent`
- Origin Mode：`validate + run`
- Verification Status：`COMPLETED`
- Decision：`S18_1R_ACTIONABILITY_REPAIRED_PASS`
- Cohort：每域冻结哈希排序 `[1024,2048)`，与原 cohort 零重叠
- Execution：仅复用四组 epoch-10 checkpoint；未重训，未读保护数据

## Domain Gate

| Domain | headroom | beam200-only | nonempty pruner | raw micro@8 | capacity-normalized@8 | any-hit@8 | CF z drift | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Toys | 0.113281 | 232 | 0.650862 | 0.757366 | 0.904762 | 0.993377 | 0.094199 | `PASS` |
| Beauty | 0.130371 | 267 | 0.689139 | 0.306290 | 0.973913 | 1.000000 | 0.099620 | `PASS` |

## Boundary

本结果不自动启动 S18-2。I1/I2、D1/D2、official validation/test 与 Sports 均未读取。
