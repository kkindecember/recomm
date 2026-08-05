# GRAM 第八阶段：TIPA-D0 路径对齐失败归因审计报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-03T22:31:28.318922+08:00
- Verification Status: ANALYZED
- Version Label: phase8_tipa_d0_failure_attribution_v1

## 结论

本次 CPU-only、analysis-only 审计成功完成。诊断标签为：`TEACHER_TO_PATH_TRANSFER_NEGATIVE`、`PATH_PERTURBATION_UNSAFE`、`DOMAIN_SPECIFIC_RANK_SHIFT`、`REALIZATION_BOTTLENECK`、`INCONCLUSIVE_ATTRIBUTION`。这些标签描述 P0A 的失败链，不构成新方法选择器；`TIPA_P1` 仍永久锁定。

Toys 与 Beauty 的 C−B Kendall 均为负；Toys 的 C broad harm 为 3.125%，超过原 1% 上限，而 Beauty 为 0%。两域的 C Recall@10/NDCG@10 方向相反。teacher-exclusive 用户存在，但 C 在 Toys 为 0/6、Beauty 为 1/14 进入 beam@50。综合证据指向 teacher→path transfer 负向、扰动安全性失败、跨域 rank shift 与 realization bottleneck 并存；现有字段仍不能证明 teacher 本身正确，也不能作因果分解。

## 固定边界与完整性

- Parent decision: `STOP_TIPA_NO_PATH_REALIZATION`
- 仅读取 P0A recovery 的锁定 JSON/CSV；没有 forward、训练或解码。
- optimizer steps: `0`；GPU: `0`。
- Sports/test/external development read: `False/False/False`。
- 空 beam rank 固定右删失为 51，仅用于配对 rank-change；原始空值未被改写。
- 9 个父输入 SHA-256 均与预注册值一致；A/B/C key、行数、schema、finite 值及 summary 聚合均复算通过。

## 四段失败链

### Toys

- Teacher availability：teacher-exclusive `6/256`；该数量不代表 teacher 正确性。
- Item→path：C−B Kendall mean `-0.008157`，median `-0.014694`，95% paired bootstrap CI `[-0.017717, 0.001786]`。
- Perturbation：C null-rate mean `0.480011`；max-abs-delta 触及 0.3 bound 的比例 `100.000%`；C broad harm `3.125%`。
- Beam realization：B `0/6`（Wilson 95% `[0.000, 0.390]`）；C `0/6`（`[0.000, 0.390]`）。

### Beauty

- Teacher availability：teacher-exclusive `14/256`；该数量不代表 teacher 正确性。
- Item→path：C−B Kendall mean `-0.018769`，median `-0.020408`，95% paired bootstrap CI `[-0.027698, -0.009732]`。
- Perturbation：C null-rate mean `0.647300`；max-abs-delta 触及 0.3 bound 的比例 `100.000%`；C broad harm `0.000%`。
- Beam realization：B `2/14`（Wilson 95% `[0.040, 0.399]`）；C `1/14`（`[0.013, 0.315]`）。

## 分层、多重比较与可解释性限制

预冻结用户分层共 `96` 行，fit-prefix census 共 `20` 行。探索性相关与分层检验统一使用 Benjamini–Hochberg FDR 0.05；完整 p/q 值保存在 `summary.json`，主结论不依赖未经校正的 p 值。fit-prefix 与 calibration-user cohort 未连接。

D0 是结果知情的 post-hoc 归因审计。它能定位共现的失败环节，但不能从这些既有字段识别反事实因果，也不能判定 teacher 的 item preference 是否正确。因此 `INCONCLUSIVE_ATTRIBUTION` 与其他机制标签并存。

## 封存决定

- `tipa_p1_unlocked=false`
- 不补数据、不改 cohort、不重跑 P0A。
- 不搜索 bound、层数、loss、teacher、seed 或 beam。
- 本报告完成后不自动实现下一结构或读取新数据。
