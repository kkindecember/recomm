# GRAM 第八阶段：TIPA-D0 路径对齐失败归因审计实验计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-03
- Verification Status: `PREREGISTERED_ANALYSIS_ONLY_PENDING_IMPLEMENTATION`
- Version Label: `phase8_tipa_d0_failure_attribution_audit_v1`
- Parent decision: `STOP_TIPA_NO_PATH_REALIZATION`
- Parent result: `artifacts/phase8/tipa_p0_branching_recovery/summary.json`
- Compute boundary: CPU only, 0 GPU, 0 optimizer steps
- Sports/Test/Fresh validation: 封存

## 1. 目的与不可改变的边界

TIPA-P0A 已经否决当前 adapter，D0 不再检验“TIPA 是否有效”，而只回答：负迁移主要发生在
teacher item signal、item→prefix projection、adapter perturbation，还是 beam realization 的哪一环。

D0 是结果知情、post-hoc、analysis-only 审计。它必须满足：

- 只读 P0A recovery 已有 JSON/CSV，不重新 forward、不训练、不解码；
- 不改 cohort、seed、teacher、bound、层数、loss、beam 或指标定义；
- 不读取原始 sequence 的 `[-2:]`、Sports、test、fresh development 或其他实验 outcome；
- 不用分层结果选择用户、阈值或新 TIPA 变体；
- 任意结果都保持 `TIPA_P1_LOCKED`，不能把 D0 当成晋级门。

## 2. 固定输入与哈希

| 输入 | SHA-256 |
|---|---|
| canonical summary | `a0b6b06717d234c224daa7cf2fc3f3bc08599b364ffdc360ae66ced9d9d09c2b` |
| Toys summary | `1ac1e6755d02aaf92753b487be62a5d761f2926aa292cfbd48aadb980e802e96` |
| Beauty summary | `5ce602cf212115a6d1f407127348c49b0772e97165d2ffa791dd9049c5667500` |
| Toys per-prefix | `9bdb4f27a07f1c2d9b5e5d93873844a02f49ba397760dfc178055a25a835c63d` |
| Beauty per-prefix | `b4d80fd10ef40ed02b324800c9f3f9a12b1f1bc7e67d34b50cd9c8ac5376e2fd` |
| Toys per-user | `51a9c734e37f5df749d5549188eab3aaeea45b471e7f2a6adf15a58a058fea08` |
| Beauty per-user | `4fd27a30b2dcc0eeb636f816cf1de177e108539ad1f9ce9addf19d0d0533939f` |
| Toys per-user-arms | `7b78c3d3b1e1217884228e28bd63923031a3ac9f81edf8d370c26965cb3b060e` |
| Beauty per-user-arms | `6f3142b5d33fc8fb9c84bdfe5256c816ddd36faa62baef8b0e03606cc64859d3` |

启动前必须补齐 analysis script 自身哈希；任何父输入变化均
fail-closed 为 `BLOCKED_PARENT_ARTIFACT_DRIFT`。

## 3. 固定分析链

### 3.1 完整性重算

逐域验证：256 个唯一 sample keys；A/B/C 各 256 行且 key 完全对齐；所有 rank、Kendall、
margin、null-rate、delta 与指标 finite；summary 聚合可从 CSV 精确复算；prefix 表 256 行且
`legal_children>1`；Sports/test/external development 标志均为 false。

### 3.2 四段失败链

1. **Teacher availability**：teacher-exclusive users、teacher target rank 与 margin 分布；
   不把 teacher-exclusive 当作 teacher 正确性的充分条件。
2. **Item→path alignment**：逐用户 `C_kendall-B_kendall`，报告均值、中位数、正/负/零人数、
   95% paired bootstrap interval。
3. **Perturbation behavior**：`C_null_rate`、`C_max_abs_delta` 与 Kendall delta、rank delta 的
   Spearman；检查 adapter 是否始终碰到 bound 或在高 null-rate 下失去可控性。
4. **Beam realization**：在 A beam@50 外且 teacher top-50 内的用户中，比较 B/C 进入
   beam@50、进入 top-10、broad harm 与 rank change；实际人数和 Wilson interval 同报。

### 3.3 预冻结分层

- calibration user：target group（head/tail）、history group（short/long）、teacher margin
  group（low/high）、transition covered/uncovered；
- fit prefix：depth=`prefix token count`；legal children bins=`2–4 / 5–16 / 17–64 / >64`；
- 不将 fit-prefix 行与 calibration-user 行连接，二者 cohort 隔离；只分别描述。

所有分层同时报告 n、effect 和 interval。探索性相关/分层检验统一使用 Benjamini–Hochberg
FDR 0.05；未经校正的 p 值不得作为结论。主报告仍以效应量和区间为先。

## 4. 固定诊断标签

D0 不设置 GO gate，只按既有证据生成可并列的标签：

- `TEACHER_TO_PATH_TRANSFER_NEGATIVE`：C-B Kendall 在双域均不为正，且 C 兑现不多于 B；
- `PATH_PERTURBATION_UNSAFE`：任一域 broad harm 超过原 1% 上限，或 harm 与 perturbation
  强度呈稳定正关联；
- `DOMAIN_SPECIFIC_RANK_SHIFT`：Recall/NDCG 或配对 rank change 在 Toys/Beauty 方向相反；
- `REALIZATION_BOTTLENECK`：teacher-exclusive 存在，但 C 的 beam@50 兑现比例及区间仍低；
- `INCONCLUSIVE_ATTRIBUTION`：现有字段不足以区分上述环节。

这些标签允许并存，属于失败描述，不是新方法选择器。不得根据标签回到相同 cohort 训练
修补版 adapter。

## 5. 预期产物

输出根目录固定为 `artifacts/phase8/tipa_d0_failure_attribution/`：

- `summary.json`：输入哈希、完整性、四段失败链、标签与封存状态；
- `paired_effects.csv`：逐用户 B/C-A 与 C-B 配对量；
- `strata.csv`：所有预冻结用户分层结果；
- `prefix_census.csv`：depth/legal-children 分层；
- `bootstrap_intervals.csv`：固定 seed 2023、10,000 次 resamples；
- `integrity.json`、`manifest.json`、`run.log`、`status.json`；
- `report/第八阶段/GRAM_第八阶段_TIPA-D0路径对齐失败归因审计报告.md`。

runner 必须在 `succeeded` 前校验每个文件存在、非空、行数/schema 正确，且 summary 中
明确写入 `tipa_p1_unlocked=false`、`sports_read=false`、`test_read=false`、
`optimizer_steps=0`。

## 6. 资源、监控与停止规则

- Python/CPU only；预计 1–3 分钟；不得停止或迁移 CodeLlama，不申请 GPU lease。
- hard timeout=10 分钟；具名 runner/tmux 和独立 status；禁止自动 retry。
- 输入哈希漂移、key/arm 不对齐、summary 无法精确复算、非 finite 或禁读标志异常时，
  立即 `EXECUTION_INVALID`，保留日志，由研究者决定是否修复。
- 分析成功后的唯一动作是写 D0 报告；不得自动实现下一结构、训练模型或读取新数据。

## 7. 强制决策记录

```text
阶段：TIPA-D0，post-hoc analysis-only failure attribution
唯一问题：TIPA-P0A 的负迁移位于 teacher→path→perturbation→beam realization 的哪一环。
固定输入：P0A recovery 已有 JSON/CSV 与 SHA256；不重新 forward。
直接机制指标：C-B Kendall、perturbation/null、teacher-exclusive realization、paired rank harm。
晋级门：无；TIPA-P1 永久锁定。
失败后停止项：不补数据、不改 cohort、不重跑 P0A。
禁止的邻近补丁：bound/层数/loss/teacher/seed/beam 搜索，target-conditioned selector。
资源：CPU only，0 GPU，0 optimizer steps，10 分钟 hard timeout。
Sports/test read：false / false
```

## 8. 当前状态

计划已写入，但 analysis script、runner、完整哈希锁尚未实现。只有研究者明确授权后，才进入
实现、测试、冻结和启动流程；本文件本身不是运行授权。
