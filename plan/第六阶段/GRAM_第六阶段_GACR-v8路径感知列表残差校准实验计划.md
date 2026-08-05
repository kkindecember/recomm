# GRAM 第六阶段：GACR-v8 路径感知列表残差校准实验计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Created: 2026-08-02
- Verification Status: PLANNED_PENDING_IMPLEMENTATION_AND_PREREGISTRATION_LOCK
- Version Label: `phase6_gacr_v8_path_aware_listwise_v1`
- Parent evidence: GACR-v7 calibration gate stop; GACR-v3 remains incumbent
- Development domains: Toys、Beauty；confirmation domain: Sports（封存）；test（封存）

## 1. 研究问题与边界

v7 表明“仅以 base rank 和原 6 维特征替换损失”会在 calibration 中伤害 tail top-10。它没有检验 GRAM 对每个 item lexical-ID 的实际条件生成概率，也没有让 residual 感知同一用户候选之间的相对结构。

本轮唯一问题是：**在冻结 GRAM、固定 GRAM-beam + catalog union 的条件下，真实 item-path score 与轻量候选列表交互能否安全地提高 NDCG@10？**

这不是 v7 loss 的重跑或权重搜索。v8 保持 v7 的截断敏感 objective、候选预算、全量 fit、80/20 fit/calibration 隔离和 seed 设置；改变的是 score interface 与 listwise interaction。不会扩大候选源、解冻 backbone、读取 Sports/test，或根据 calibration 调参。

## 2. 冻结的方法

### 2.1 候选与基础分数

- 候选固定为每用户 GRAM constrained beam top-50 与 catalog projection top-50 的 target-free 去重 union；排除 history item，stable tie-break 为 candidate insertion order。
- A/B/C/D/E 使用完全相同的候选、用户、GRAM C1 checkpoint 和 rank-only base score `1 / beam_rank`（catalog-only 为 `0`）。候选覆盖不属于本轮可优化对象。
- 对 GRAM beam item 记录 generate 返回的 sequence score；对 catalog-only item 以冻结 GRAM encoder hidden states teacher-force 其完整 lexical ID。二者均为完整有效 token 的**平均** log probability（BOS/pad 排除，EOS 按现有 lexical-ID 有效标签规则）。

### 2.2 路径特征

每个 candidate 由同一冻结 encoder 计算以下 target-free 特征；连续 path 特征仅在该用户 union 内 z-score（std 下限 `1e-6`，clip 至 `[-10,10]`）：

1. `path_mean_logp`：完整 lexical-ID 的 mean log probability；
2. `min_prefix_margin`：每层正确 token logp 减该层最强其它合法 trie child logp 的最小值；
3. `mean_prefix_entropy`：各层合法 trie-child softmax entropy 的均值；
4. `beam_sequence_gap`：仅 beam item 的 normalized sequence score 减同一 beam 第一名分数；catalog-only 置 `0`，并以原有 beam-membership 特征区分。

若 token 没有其它合法 child，margin 记为 `0`；任何 path feature 非 finite、candidate mapping 不一一对应、或 teacher-forcing 与 beam lexical ID 不一致均为完整性错误，不以填补值继续。

### 2.3 五臂嵌套消融

| 臂 | 输入与模型 | 目的 |
|---|---|---|
| A | frozen GRAM 原始排序，无 residual | 主基线 |
| B | frozen GACR-v3 rank-only residual | deployment incumbent |
| C | 原 6 维 `BoundedResidualRanker(6,16,0.2)`，v7 metric-aligned loss，按 v8 fit split 重训 | 诊断 v7 loss 单独作用 |
| D | 10 维（原 6 维 + 4 path features）`BoundedResidualRanker(10,16,0.2)`，同 C 的 loss | 检验 path score |
| E | 与 D 相同 10 维输入，线性投影至 16 维后经 1 层 pre-norm self-attention set encoder（2 heads，FFN=32，dropout=0），逐 item 线性输出并以 `0.2*tanh` 有界 | 检验列表交互 |

所有 residual 末层零初始化，训练前严格等同 base ranking；optimizer 仅更新 C/D/E residual 参数。D/E 的 path feature、loss、fit records、30 个 full-batch steps、AdamW (`lr=0.01`, `weight_decay=0.01`) 与 gradient clip=`10` 完全相同。E 的 attention mask 只屏蔽 padding，不允许跨用户 attention。

### 2.4 损失与训练

沿用 v7 的冻结 NDCG@10/Recall@50 cutoff-sensitive weighted pairwise softplus：

`w_j = |D10(r_t)-D10(r_j)| + 0.25 * |I(r_t<=50)-I(r_j<=50)|`，`L = sum_j w_j softplus(s_j-s_t) / sum_j w_j`。

base ranks 仅用于这个权重与 A 基线，不因 path feature 重排后重算；zero-weight records 从所属 group 均值排除。covered head/tail fit records 的 group mean 等权。训练 seeds=`2023,2024,2025`，每臂 30 steps，deployment scale=`1.0`，不做 scale、hidden size、层数、loss 或步数搜索。

## 3. 数据隔离与冻结

- Toys、Beauty 使用既有训练 split 和全量 fit records；calibration 从原 80/20 user split 确定性抽取每域 128 head + 128 tail，且与 fit user 零重叠。
- fresh validation 为每域 1024 个新 development users，salt=`phase6-gacr-v8-path-listwise-development-v1`；排除 GCDH、GACR-P0 至 v6 的所有 historical validation cohort。v7 没有 fresh-validation cohort，故无额外 user 排除集。
- GRAM C1 checkpoint、v3 residual SHA、数据 split、候选构造、lexical-ID mapping 与 config SHA256 必须在运行前锁定；backbone optimizer steps 必须为 `0`。
- Sports 与 test 禁读。fresh label 不得用于 feature、loss、calibration、超参数选择或 code 分支。

## 4. Calibration 门与 fresh-validation 规则

D 与 E 是 fresh-validation 候选方法；各 domain-seed 相对 A 必须逐一满足：broad harm `<=1%`、overall Recall@10/50 delta `>=-0.2pp`、tail Recall@50 delta `>=-0.4pp`、tail NDCG@10 delta `>=-0.0005`，以及 finite loss/gradient/checkpoint、parent SHA 不变、fit/calibration 隔离。

C 是机制诊断臂：若重现 v7 的 calibration 失败，则不得读取其 fresh labels，也不阻止 D/E 在各自安全门通过后进入 fresh validation；报告须把 `D-vs-C` 标为 calibration-only mechanism contrast。若 C 通过才在同一 fresh cohort 报告 D-vs-C。这样不会把已知不安全的 rank-only loss 投向新 cohort，同时保留对 path interface 的预定义检验。

若 D 或 E 的任一 domain-seed 未过其安全门，该臂不得进入 fresh validation，且不调整任何参数；若 D/E 均不合格，v8 在 calibration 止步并关闭固定候选 GACR 增长主线，转入 F0 coverage/oracle 审计。

## 5. Fresh-validation 统计与决定

对每个合格臂在同一 fresh cohort 并列 A、B（及合格的 C、D、E）。主要比较为 E-vs-B；辅助比较为 D-vs-B、E-vs-D，及 C 合格时的 D-vs-C。主指标 NDCG@10；次要指标 Recall@10/50、tail NDCG@10、tail Recall@50、changed coverage、broad harm、union coverage、延迟、峰值显存、参数量。

对同一 user 的三 seed 先取指标均值，再进行 10,000 次 user-level paired bootstrap；不得把 3,072 rows 当独立样本。报告绝对差、相对增益和 95% CI。

**固定候选仍可增长**要求 E-vs-B 同时满足：双域 macro NDCG@10 `>0`、至少 4/6 cell 正、两域均不越安全界且完整性门通过。E 替换 B 还要求跨 seed user-level bootstrap macro NDCG@10 的 95% CI 下界 `>0`，或两域各至少 2/3 seed 为正且两域均值为正，并且各 guardrail CI 下界不低于相应非劣界。

若仅 D 成功，冻结 path-aware pointwise residual、停止扩大 listwise 容量；若 E 相对 D 有正信号但不超过 B，记录机制信号但不替换 incumbent；若 D/E 均无增量或越界，正式关闭固定候选 GACR 主线并进入 candidate drafting + GRAM verification 的 F0 审计。

## 6. 实现、测试与资源

实现前先完成不读取 fresh validation 的工程 smoke：验证 beam sequence score 与 teacher-forced score 的长度归一化、path feature finite/mapping、attention padding isolation、zero-init identity、D/E 参数边界、five-arm sample-key alignment、C 的诊断臂隔离、cohort exclusion 和 Sports/test 禁读。smoke 只测实际 workload peak，不用于选择科学超参数。

随后创建并冻结以下产物后才可启动完整 v8：

- `experiment/phase6/gacr_v8.py`、`experiment/phase6/test_gacr_v8.py`、`experiment/phase6/run_phase6_gacr_v8.sh`；
- `artifacts/phase6/configs/gacr_v8_preregistered.json`；
- `artifacts/phase6/gacr_v8/`（summary、per-user CSV、checkpoint、telemetry、lease/status/log）；
- `report/第六阶段/GRAM_第六阶段_GACR_v8路径感知列表残差校准结果与验证报告.md`。

完整运行前以实测 peak 声明 GPU0 workload/sidecar 合计 `30,720 MiB` 租约和 hard timeout；CodeLlama 只在显存门通过后停止，并在所有退出路径恢复。非零 scientific exit、timeout、完整性失败或 calibration gate failure 均不自动重试。运行结束仅写结果，不自动读取 Sports/test 或启动 fallback。

## 7. 实现冻结与启动授权（2026-08-03）

- implementation：`experiment/phase6/gacr_v8.py`；tests：`experiment/phase6/test_gacr_v8.py`；runner：`experiment/phase6/run_phase6_gacr_v8.sh`；config：`artifacts/phase6/configs/gacr_v8_preregistered.json`。
- CPU 单元测试（v7 + v8）`12 passed`，Python compile、Bash syntax 与 whitespace 检查均通过。smoke 覆盖 path z-score finite、zero-init identity、listwise list isolation、frozen noninferiority boundary 与 Sports/test source guard；它不读取 fresh validation。
- 资源声明冻结为：GPU0 总租约 `30,720 MiB`，预计 workload peak `24,576 MiB`，sidecar `6,144 MiB`，hard timeout `48h`；完整性或校准失败不得自动重试。
- 研究者已于 2026-08-03 明确授权启动。完整运行仅在 SHA 和 config 写入后开始；它不授权改动 GACR-v3 incumbent、读取 Sports/test 或启动 candidate drafting fallback。

## 8. 实现错误恢复授权（2026-08-03）

- 原运行在 Toys 的候选构造完成后、任何 C/D/E residual 训练、calibration 或 fresh validation 前，因旧版 `to_cpu_record` 要求不存在的 `features` 键而以 `KeyError` 退出；Sports/test 与 fresh validation 均未读取。
- 研究者随后明确授权恢复。恢复仅将 v8 record 的 CPU 拷贝接口固定为 `base`、`features6`、`features10`，并新增该接口回归测试；不改变科学配置、候选、checkpoint、split、seed、训练预算或门槛。
- 恢复运行输出独立写入 `artifacts/phase6/gacr_v8_recovery/`；原失败运行的 `artifacts/phase6/gacr_v8/` 保留作审计记录。恢复前重新锁定实现、测试、runner 与本计划 SHA256，仍禁止自动重试、Sports/test 读取和 fallback。

## 9. E-only fresh-validation 恢复授权（2026-08-03）

- `gacr_v8_recovery` 的训练与 calibration 已完成且完整性门通过；E 在 Toys、Beauty 的全部 6 个 domain-seed 均通过预冻结 calibration 非劣门。D 在 Beauty 的全部 3 个 seed 因 overall Recall@10 与 tail NDCG@10 越界，不进入 validation。
- `completed_without_validation` 来自资格判定实现错误：禁止读取标志 `test_data_read=false`、`sports_data_read=false` 被错误地以 `all()` 视为失败，而非合规条件。它不是科学性 calibration gate failure。
- 研究者明确授权仅恢复 E 的既定 fresh validation：修复资格逻辑、锁定原 recovery summary 与 6 个 E checkpoint SHA256，并在 `artifacts/phase6/gacr_v8_validation_recovery/` 输出 summary、per-user CSV、telemetry/status/log。不得重训、重算 calibration、运行 D、改变 seed/队列/超参数，或读取 Sports/test。

## 9. Fresh-validation 门判定错误与 validation-only 恢复授权（2026-08-03）

- 已完成的 `gacr_v8_recovery` 实际写出了 Toys/Beauty 上 C/D/E 的全部 18 个训练与 calibration cell；其中 E 的 6/6 domain-seed cell 均通过预注册非劣门，D 因 Beauty 3/3 cell 失败而不合格。
- 原实现将 `backbone_optimizer_steps=0`、`test_data_read=false`和 `sports_data_read=false` 与必须为 true 的完整性字段一起传入 `all(integrity.values())`，把正确的零步骤/禁读证据错当为失败，因而在 fresh validation 前错误止步。这是 typed integrity predicate 实现错误，不是 E 的 calibration gate failure。
- 研究者已明确授权最小 validation-only 恢复：保留原实现、summary 与 checkpoint 不变，以独立程序显式校验布尔完整性字段、`backbone_optimizer_steps == 0` 以及 Sports/test 禁读字段为 false，并且只放行 E；D/C 不读取 fresh labels。
- 恢复同时补齐第 5 节预注册的主比较：在同一 fresh cohort 上评估 A、冻结 incumbent B（GACR-v3）和 E，以 E-vs-B 为主比较；同一 user 先对 3 seed 指标取均值，再做 10,000 次 user-level paired bootstrap。
- 恢复仅复用 `artifacts/phase6/gacr_v8_recovery/` 的 6 个 E checkpoint 和冻结 GACR-v3 residual/budget；不训练 residual、不改参数、不更换 cohort/salt、不读取 Sports/test、不自动重试。输出独立写入 `artifacts/phase6/gacr_v8_validation_recovery/`。
- 恢复实现为 `experiment/phase6/gacr_v8_recover_validation.py`，回归测试为 `experiment/phase6/test_gacr_v8_recover_validation.py`，runner 为 `experiment/phase6/run_phase6_gacr_v8_validation_recovery.sh`，冻结配置为 `artifacts/phase6/configs/gacr_v8_validation_recovery.json`。
