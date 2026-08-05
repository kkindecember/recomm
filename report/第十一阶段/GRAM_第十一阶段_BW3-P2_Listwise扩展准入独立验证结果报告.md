# GRAM 第十一阶段 BW3-P2：Listwise 扩展准入独立验证结果报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-08-05
- Verification Status: `ANALYZED`
- Version Label: `phase11_bw3_p2_one_shot_validation_result_v1`
- Experiment ID: `GRAM_PHASE11_BW3_P2_LISTWISE_ADMISSION_ONE_SHOT_VALIDATION_V1`
- Parent Experiment: `GRAM_PHASE11_BW3_P1C_LISTWISE_ADMISSION_CORRECTION_V1`
- Overall Confidence: `CAUTION`（对本次固定验证结果为高置信；对更广泛泛化结论保持谨慎）

## 1. 执行结论

P2-1 于 2026-08-05 12:29:54（Asia/Shanghai）在后台启动，约 12 秒后完成。
两域各固定 512 用户，全部输入和代码 SHA 锁通过，CodeLlama 在物理 GPU6 保持 30 GiB
现存占位，实验为 CPU-only，资源审计通过，test/Sports 未读取。

预注册 P2 gate 结论为 **FAIL**。失败不是因为性能下降，而是因为没有产生预注册要求的
最小正增益：

- Toys 与 Beauty 的 Hit@10、NDCG@10、tail Hit@10 delta 均为 `0`；
- 两域均无 promotion，也无 regression；
- Toys 准入 21 个候选，Beauty 准入 15 个候选，但两域准入候选中的真实 target 均为 0；
- 因此“至少一域 Hit@10 `>= +0.002`”和“两域平均 Hit@10 `>= +0.001`”两条未通过。

本轮最准确的判定是：

> P1C gate 在独立 `t=-2` split 上保持了安全性，但 calibration 中的正收益没有泛化；
> 当前扩展准入机制未证明具有独立验证收益。

Validation 已消耗，结果已原子揭示。按照预注册协议，不允许把调 margin、改特征或重跑
解释为同一个确认性 P2。

## 2. 主指标

| Dataset | Users | Base Hit@10 | P2 Hit@10 | Delta | Base NDCG@10 | P2 NDCG@10 | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| Toys | 512 | 0.123047 | 0.123047 | 0.000000 | 0.084175 | 0.084175 | 0.000000 |
| Beauty | 512 | 0.103516 | 0.103516 | 0.000000 | 0.064186 | 0.064186 | 0.000000 |
| 两域简单平均 | 1,024 | — | — | 0.000000 | — | — | 0.000000 |

paired user bootstrap 使用冻结 seed 2023、2,000 replicates。由于每位用户的 base 与 final
Hit@10 状态完全一致，两域 Hit@10 delta 的 95% bootstrap interval 均精确为 `[0, 0]`。
这不是“样本量不足导致区间跨 0”，而是观察到的 paired effect 本身处处为 0。

## 3. Tail 与安全性

| Dataset | Tail users | Base tail Hit@10 | P2 tail Hit@10 | Delta | Promotions | Regressions |
|---|---:|---:|---:|---:|---:|---:|
| Toys | 128 | 0.101563 | 0.101563 | 0.000000 | 0 | 0 |
| Beauty | 140 | 0.042857 | 0.042857 | 0.000000 | 0 | 0 |

安全性相关条款全部通过：两域 Hit/NDCG/tail 均不退化，无 admission 用户的 fallback 与
冻结 base 完全一致，所有数值 finite，每用户最多准入 3 个。Toys/Beauty 的 base Hit@10、
NDCG@10、q1 和 item-head SHA 也都与 BW1 冻结口径一致。

## 4. 准入行为与失败机制

| Dataset | Admissions | Admission users | Fallback users | Target admitted | Target expansion-only | Expansion-only users with any admission |
|---|---:|---:|---:|---:|---:|---:|
| Toys | 21 | 18 | 494 | 0 | 60 | 4 |
| Beauty | 15 | 11 | 501 | 0 | 65 | 3 |

准入数量分布：

- Toys：494 个用户准入 0 个，16 个准入 1 个，1 个准入 2 个，1 个准入 3 个；
- Beauty：501 个用户准入 0 个，8 个准入 1 个，2 个准入 2 个，1 个准入 3 个。

Toys 被准入候选的 logit 范围为 `0.0548–3.5275`，中位数 `0.6521`；Beauty 为
`0.0598–2.7170`，中位数 `0.8590`。因此 margin `0.0` 确实激活了 gate，不是实现上把所有
候选拒绝。问题在于通过 gate 的 expansion candidates 没有包含真实 target。

候选覆盖本身仍有空间：Toys 有 60 个 target 只在 expansion pool，Beauty 有 65 个；但其中
分别只有 4 和 3 个用户发生了任意 admission，而且被准入的都不是 target。这表明 P1C
学到的正候选识别边界在 `t=-2` 上发生了选择性失配。现有证据支持“gate selection
generalization failure”，不支持把失败归因于 beam200 没有 target，也不支持归因于 base
被破坏。

## 5. 预注册 gate 逐条审计

| Gate | Result | Evidence |
|---|---|---|
| 两域 Hit@10 不退化 | PASS | 两域 delta 均 0 |
| 至少一域 Hit@10 `>= +0.002` | **FAIL** | 两域 delta 均 0 |
| 两域平均 Hit@10 `>= +0.001` | **FAIL** | mean delta = 0 |
| 两域 NDCG@10 delta `>= -0.001` | PASS | 两域 delta 均 0 |
| 两域 tail Hit@10 不退化 | PASS | 两域 delta 均 0 |
| 两域 admissions 非零 | PASS | 21 / 15 |
| 两域 promotions `>=` regressions | PASS | 0 = 0 / 0 = 0 |
| 两域完整性 | PASS | 512 + 512、base identity、finite、fallback 全通过 |
| 资源与访问协议 | PASS | CPU-only、CodeLlama 保持、telemetry 通过、test/Sports 未读 |

最终状态：`failed`。其中 promotions `>=` regressions 虽形式上通过，但 `0 = 0` 只说明没有
伤害，不能作为有效性的正证据。

## 6. 执行与资源审计

- 后台会话：`gram_phase11_bw3_p2_one_shot_validation`，正常退出；
- 合成测试：11/11 通过；
- 计算设备：CPU，实验 GPU PID 全程为 0；
- CodeLlama：物理 GPU6，reported reserved `30,886 MiB`，live used `31,206 MiB`；
- GPU telemetry：运行期样本均显示 CodeLlama running，30 GiB 门通过；
- CPU RSS：观测峰值约 `696,696 KiB`；
- resource audit：`passed`，CodeLlama 终态 `preserved_running`；
- `validation_access_started=true`、`validation_consumed=true`、`results_revealed=true`；
- `test_read=false`、`sports_read=false`、`automatic_retry=false`。

首次沙箱内启动尝试因无法连接 tmux 而在 runner 外退出，未建立会话、未读取 validation；
获得持久 tmux 权限后的正式启动才构成本次 P2-1。它不构成科学重试。

## 7. 统计解释与 11 类谬误扫描

本轮没有 p-value 或传统参数检验。主要证据是完整固定样本上的 paired metric delta、
逐用户状态和描述性 bootstrap interval。结果应解释为“在本次固定两域 `t=-2` 样本中观察到
零 top10 效果”，而不是证明任何未来数据上的真实效应严格等于零。

- Simpson's paradox：两域总体与 tail 分层方向均为 0，未见方向反转；
- Ecological fallacy：主分析以 per-user paired rank 为单位，未由域均值推断单用户收益；
- Berkson's paradox：固定 512 用户来自既有 BW1 样本，存在样本边界但未见由二次筛选制造的相关性；
- Collider bias：没有后验控制变量或按 P2 结果条件化；
- Base-rate neglect：beam50/beam200/expansion-only/outside-union coverage 已完整报告；
- Regression to the mean：没有按极端 P2 表现选样或 pre-post 归因；
- Survivorship bias：两域均完成全部 512 用户，无掉队用户；
- Look-elsewhere effect：指标、双域和 gate 预先冻结，没有择优报告；
- Garden of forking paths：gate、margin、特征、base 和阈值均在 P2 前冻结；
- Correlation ≠ causation：本报告只陈述算法在固定离线 split 上的表现，不作人群因果声明；
- Reverse causality：不适用于本次离线算法对照，未作方向性因果推断。

Fallacy scan coverage：`11/11`。未发现需要推翻本轮数值结论的 RED_FLAG；总体标为
`CAUTION`，原因是独立证据只来自当前两个域和一个固定 validation split，不能外推成普遍无效。

## 8. 可复现性与一次性边界

- 输入、gate、代码和正式 config 均有 SHA256 锁；
- base identity、用户数、target offset、q1、item-head 和禁读标记全部通过；
- 本轮按计划只能运行一次，因此不执行 reproducibility re-run；
- Reproducibility verdict：`CANNOT_VERIFY_BY_RERUN_WITHOUT_VIOLATING_PREREGISTRATION`；
- 审计完整性足以确认本次结果是按冻结协议产生，但不授权第二次确认性运行。

## 9. 结论与讨论边界

P2 否定了“当前 P1C gate 已经获得独立正收益证据”这一判断。它同时保留了一个较窄结论：
gate 在本次验证中很保守，没有损害 base，但这种保守性没有转化为 target promotion。

下一步不能在已消耗 P2 上继续调 margin 后声称确认性成功。可讨论的方向包括：

1. 使用已揭示 P2 per-user 数据做明确标注为探索性的 distribution-shift/target-selection 诊断；
2. 停止 admission-gate 路线，回到候选生成或直接 item-aware ranking；
3. 若未来仍要做确认性验证，必须重新定义方法并准备真正未使用的新 holdout，而不是重跑 P2。

本报告不替研究者选择下一路线，也不写入新的实验计划；下一步先讨论，再另行冻结计划。

### 2026-08-05 探索诊断更新

后续只读诊断确认：validation 的 125 个 expansion target 全部低于 margin 0，没有任何
“过阈值后被 top3 竞争淘汰”的案例。相较 calibration，Toys/Beauty target 平均 logit 分别
下降 4.781/4.985；非 target 平均 logit 基本不变。约 89%–90% 的下降来自结构上共享
item-head anchor 的 `item_anchor_z` 与 `cf_pop_adjusted`。这将失败机制收窄为 target-specific
item-head selection shift，而不是 coverage、top3 容量或整体 score 平移。完整探索报告：
`report/第十一阶段/GRAM_第十一阶段_BW3-P2_扩展Target选择漂移探索诊断报告.md`。

## 10. 核心产物与 SHA256

| Artifact | SHA256 |
|---|---|
| `scientific/summary.json` | `0b644b5d8318ee9ce57b4296bf4888f22b5c0a8515331c466c6d81fc18a3f0aa` |
| `scientific/Toys/per_user.tsv` | `6fb4474c42df850f07a249864414b0eb4b85063620c547f621f7a855a86a3ae9` |
| `scientific/Beauty/per_user.tsv` | `ae807c61b082be22143ef83239171083aab6773facf1887a202ac26cceec0916` |
| `gpu_telemetry.csv` | `9666e87109a2a3cc52dc4abaa8bb6f678ef948c8f56f380dd2ab9ead019cfd0a` |
| `cpu_telemetry.csv` | `7a88bc50e1ad5fa0bf3347a2d497577d498bbc6b6c5ad14ec3418bba43faffe2` |
