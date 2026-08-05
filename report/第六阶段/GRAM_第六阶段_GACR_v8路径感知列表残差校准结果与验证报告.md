# GRAM 第六阶段：GACR-v8 路径感知列表残差校准结果与验证报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-03
- Verification Status: ANALYZED — `FIXED_CANDIDATE_GACR_CLOSED`
- Version Label: `phase6_gacr_v8_path_aware_listwise_v1`
- Source plan: `plan/第六阶段/GRAM_第六阶段_GACR-v8路径感知列表残差校准实验计划.md`
- Canonical result: `artifacts/phase6/gacr_v8_validation_recovery/summary.json`

## 1. 执行结论

GACR-v8 已完成两次仅修复工程接口的恢复，并在隔离的 Toys、Beauty fresh development cohort 上完成 E（path-aware listwise residual）的 3-seed 验证。固定候选 union 上，E 对 frozen GRAM 的 NDCG@10、Recall@10/50 均为零增益；补充的逐用户 E-vs-B（GACR-v3 incumbent）汇总在两个域亦为负，且出现 broad harm。因此正式决定为：

**`KEEP_GACR_V3_INCUMBENT; CLOSE_FIXED_CANDIDATE_GACR_GROWTH; ADVANCE_TO_F0_COVERAGE_ORACLE_AUDIT`**。

这不是“再调一个 residual”的信号。v7 已否决 rank-only metric-aligned loss；v8 则显示加入真实 lexical path 特征和一层候选列表交互也未将固定候选覆盖兑现为增益。后续不增加 attention 层数、hidden size、训练步数、loss 权重、scale 或用户 gate。

## 2. 执行与恢复审计

- 初始运行在 Toys 全部 22,095 个候选构造后，因旧 `to_cpu_record` 访问不存在的 `features` 键而退出；发生于任何 residual 训练、calibration 或 fresh validation 前。
- 第一恢复仅将 CPU record 接口锁定为 `base/features6/features10`，保留原科学配置；训练、calibration 与 E checkpoint 随后正常完成。
- 第二恢复修复资格聚合错误：`test_data_read=false`、`sports_data_read=false` 是合规条件，不能被 `all()` 当作失败。恢复只校验已冻结的 parent summary 与 6 个 E checkpoint SHA，并执行 E-only fresh validation；不重训、不运行 D。
- fit/calibration overlap=`0`；parent C1 SHA 在训练前后不变；backbone optimizer steps=`0`；Sports/test 均未读取。最终 runner 状态为 `succeeded`，GPU0 的 CodeLlama 已恢复。

## 3. Calibration 门

门要求每个 domain-seed 相对 frozen GRAM 同时满足：broad harm `<=1%`、overall Recall@10/50 delta `>=-0.2pp`、tail Recall@50 delta `>=-0.4pp`、tail NDCG@10 delta `>=-0.0005`。

| 臂 | Toys（3 seeds） | Beauty（3 seeds） | 验证资格 |
|---|---|---|---|
| D：path-aware pointwise | 全部通过 | 全部失败：Recall@10 `-0.391pp`、tail NDCG@10 `-0.002604` | 否 |
| E：path-aware listwise | 全部通过 | 全部通过 | 是 |

E 因而是唯一进入 fresh validation 的预定义候选；D 没有被投向 fresh cohort。

## 4. Fresh-validation 结果

每域为 1,024 个新 development users，和训练及所有规定历史 cohort 零重叠。E 对 frozen GRAM 的 canonical summary 在三 seeds 完全一致：

| 域 | NDCG@10 delta | Recall@10 delta | Recall@50 delta | broad harm | 说明 |
|---|---:|---:|---:|---:|---|
| Toys | `0.000000` | `0.000000` | `0.000000` | `0.000%` | E 未改变最终 top-10 指标 |
| Beauty | `0.000000` | `0.000000` | `0.000000` | `0.000%` | E 未改变最终 top-10 指标 |

验证目录还保留了同一 cohort 的 E-vs-B 三 seed、逐用户平均 CSV。直接对这些已写入记录取用户均值的结果如下；这只是已冻结 B 对照的补充核查，不用于再次选择模型：

| 域 | E-vs-B Recall@10 | E-vs-B NDCG@10 | E-vs-B Recall@50 | changed users | broad harm |
|---|---:|---:|---:|---:|---:|
| Toys | `-0.002279` | `-0.000927` | `-0.001628` | `14.583%` | `0.684%` |
| Beauty | `-0.005208` | `-0.003117` | `-0.000977` | `11.654%` | `0.553%` |

两域都没有主指标正增益，且 E-vs-B 的 Recall@10 已低于 v8 所冻结的 `-0.2pp` 安全界。故不满足“macro NDCG@10 > 0、至少 4/6 cells 为正、两域不越界”的固定候选继续条件，也不具备替换 v3 的资格。

## 5. 机制解释与边界

训练本身是健康的：所有 checkpoint finite、零初始化排序一致率为 `1.0`、每 seed 完成 30 个 full-batch steps。负结论不是数值崩溃或 cohort 泄漏。

证据支持的解释是：在现有 GRAM-beam + catalog top-50 union 的覆盖边界下，teacher-forced path likelihood 与候选内 attention 没有产生可部署的、稳定的重排空间。尤其 E 在 canonical fresh summary 中保持 identity，说明其所学扰动未落实到 top-10；即使观察补充 B 对照，它也只带来伤害而无收益。该结论仅针对固定候选 GACR residual 主线，不等价于“生成路径信息无价值”或“多源候选后 verifier 无价值”。

## 6. 统计与完整性限制

- 不把 3,072 domain-seed 行当作独立用户样本；主汇总以同一用户三 seed 均值为单位。
- v8 没有通过 incumbent 的开发替换门，因而不读取 Sports confirmation 或 test；不能做确认性或泛化主张。
- 两次恢复均为实现完整性修复，未改变候选、模型容量、loss、seed、cohort、checkpoint 或阈值；原失败和恢复产物均保留审计。

## 7. Fallacy Scan

覆盖：**11/11 checked**。

| 风险 | 处理 |
|---|---|
| 选择性报告 / survivorship bias | D 的 Beauty 失败与 E 的零增益均保留；未只报告通过 calibration 的 E。 |
| 多重比较 / garden of forking paths | 未针对 v8 结果搜索容量、步数、scale、gate 或 loss。 |
| 聚合掩盖尾部伤害 | calibration 与 validation 均保留 head/tail、Recall 与 NDCG。 |
| 泄漏 / collider bias | fit、calibration、fresh cohort 分离；Sports/test 未读。 |
| 因果过度解释 | 结论限于离线候选排序，不外推到在线行为。 |
| 伪重复 | 三 seed 先按用户平均；不将 seed 视为独立用户。 |

## 8. 下一步

固定候选 GACR 主线到此关闭。下一项仅为 F0：无训练的多源 candidate coverage/oracle 审计，判断独立 sequence/full-catalog drafter 是否能在两个 development 域提供可重复的独占 coverage；详见 `plan/第六阶段/GRAM_第六阶段_F0多源候选覆盖与Oracle审计实验计划.md`。F0 尚未获运行授权，Sports/test 继续封存。
