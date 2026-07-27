# GRAM 第三阶段：NLPL D0-D 诊断报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-07-24
- Verification Status: ANALYZED
- Version Label: `nlpl_d0_diagnostic_v1`
- Runtime: CPU only; no GRAM checkpoint; no test data

## 决策

固定决策为 **`STOP_NLPL_NO_EXPOSURE`**。

## 预注册 gate

| 数据集 | non-tie pairs | concordance | bootstrap 95% CI | permutation p | tail miss OR | 全部通过 |
|---|---:|---:|---:|---:|---:|---|
| Toys | 1129 | 0.443756 | [0.399544, 0.483810] | 0.9959 | 0.549751 | False |
| Beauty | 266 | 0.537594 | [0.460673, 0.616862] | 0.188981 | 0.542580 | False |

全部门槛是双数据集必要条件，任一失败不能由其他门槛抵消。完整逐项结果见
`artifacts/phase3/nlpl_d0/summary.json`。

## 完整性

- 两数据集 Recall@10/50 均按冻结 prediction 精确复算；
- 每行 50 个候选均可映射且无重复；
- 修改 `sequence[-2:]` 不改变 training-only frequency；
- native prior 全部来自本地冻结原始 T5-small；
- 未加载 GRAM checkpoint、未读取 test、未训练、未使用 GPU。
