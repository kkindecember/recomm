# GRAM 第十三阶段：R²-v2 Stage S catalog-cold 等价恢复正式结果

> **正式结论（2026-08-19）**：唯一一次等价 recovery 完整完成，预注册 verdict 为 **`FAIL_STOP_R2_V2_SOURCE`**。CBSA 相对冻结 B1 `portfolio@2` 的 overall NDCG@10 与聚合 cold H@50 通过，但 warm NDCG@10 的 95% CI 跨 0，且 Beauty cold H@50 retention 仅 `91.86% < 95%`，方向一致性硬门失败。R²-v2 到此停止；不启动 Sports，不做 source 参数 rescue。

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-19
- Verification Status: `ANALYZED_WITH_EXACT_ARTIFACT_GATE_RECOMPUTATION`
- Version Label: `r2_v2_stage_s_recovery_validation_v1`
- Experiment ID: `GRAM_PHASE13_R2_V2_CBSA_SOURCE_OOF_COLD_CANDIDATE_RECOVERY`
- Dataset / split: Toys_cold50 + Beauty_cold50 / validation 5-fold OOF
- Primary budget: `rho=0.97`
- Comparator: frozen B1 catalog-cold `unconditional portfolio@2`
- Formal verdict: **`FAIL_STOP_R2_V2_SOURCE`**
- Overall Confidence: **SOLID**（对停止结论）

## 1. Recovery 边界与运行完整性

本轮是用户授权的唯一一次纯工程等价恢复。唯一改动是把 `a2/a3` 的候选池纠正为 B1 定义：resolver 中 catalog-cold、排除 v0 top-7 的前三个候选；`a2` 使用前两个，`a3` 使用三个。网络、36 维 feature schema、loss、optimizer、50 epochs、5-fold salt、seed、budget grid、主 `rho=.97`、bootstrap 与全部 Gate 均冻结不变。

启动前的全量对齐在 Toys 8,789 + Beauty 10,655 个用户上得到 `a2/B1 exact-ranking mismatch=0`；Phase-13 全套 `101 tests passed`。原 comparator-invalid artifact 未覆盖。

| 完整性项 | 结果 |
|---|---|
| 状态 / runtime | completed；CPU；`156.88s` |
| OOF 覆盖 | 19,444 / 19,444；`(domain,user_id)` 唯一 |
| Fold | 5 折；train/held overlap 全为 0 |
| Ranking | `catalog_unique=true` 19,444 / 19,444 |
| Action degradation | selected 与 effective 完全一致 |
| Checkpoint | 5/5 存在且 SHA256 与 summary 完全匹配 |
| 数值 | OOF 四个主 metric 字段全部 finite |
| Frozen hashes | config / code / feature schema / inputs 全部通过 |
| 防火墙 | `sports_read=false`；`test_read=false` |
| 自动 retry | false；首次 tmux launcher failure 在 workload 前独立归档 |

## 2. 预注册 Gate

跨域汇总先求域内均值，再对 Toys/Beauty 等权；95% CI 为 10,000 次 paired bootstrap。

| Gate | CBSA − portfolio@2 | 相对变化 | 95% CI / 门槛 | 判定 |
|---|---:|---:|---:|---|
| overall NDCG@10 | `+0.00056212` | `+1.49%` | `[+0.00018388,+0.00094513]` | PASS |
| warm NDCG@10 | `+0.00039824` | `+0.60%` | `[-0.00011795,+0.00092591]` | **INCONCLUSIVE** |
| cold H@50 | `+0.00199635` | `+6.41%` | `[+0.00008954,+0.00389788]`；NI 界 `-0.00155753` | PASS |
| 每域方向一致性 | Beauty cold retention=`91.86%` | 要求每域 `>=95%` | shortfall=`-0.00102137` H@50 | **FAIL** |
| intervention coverage | `60.68%` | 允许 `[5%,95%]` | 范围内 | PASS |
| event density | Toys=`130`；Beauty=`172` | 每域要求 `>=30` | 充足 | PASS |
| integrity | 全通过 | — | — | PASS |

正式 verdict 为 **`FAIL_STOP_R2_V2_SOURCE`**。warm CI 跨 0 单独会产生 INCONCLUSIVE，但 Beauty cold 方向一致性是明确硬失败，因此总 verdict 是 FAIL，而不是 INCONCLUSIVE。

## 3. 分域结果与机制解释

| 域 | overall NDCG@10 Δ | warm NDCG@10 Δ | cold H@50：CBSA / incumbent | cold retention |
|---|---:|---:|---:|---:|
| Toys | `+0.00082855`（`+2.37%`） | `-0.00045957`（`-0.75%`） | `0.036409 / 0.029769` | `122.31%` |
| Beauty | `+0.00029570`（`+0.73%`） | `+0.00125605`（`+1.76%`） | `0.029885 / 0.032533` | **`91.86%`** |

两个域的 overall 点估计都为正，但取舍方向相反：Toys 以 warm 损失换 cold 增益，Beauty 以 cold 损失换 warm 增益。CBSA 因而没有形成预注册要求的跨域统一 Pareto 改进。

主预算动作分布为：`a0=7,645 (39.32%)`、`a2=1,703 (8.76%)`、`a3=10,096 (51.92%)`。Toys intervention coverage 约 `71.68%`，Beauty 约 `51.61%`，域间策略差异明显。

`rho=.93/.95/.97/.99` 的 coverage 仅从 `60.54%` 变到 `60.74%`，四个预算下汇总指标完全相同；五折 dual 最终为 0 或接近 0。这表明 warm constraint 基本 slack，预算条件没有形成有意义的可部署前沿。该诊断解释机制，但不改变冻结 Gate。

## 4. 复算与可复现性

- 从 `predictions_oof.jsonl` 重建运行时 fold/domain/source 顺序后，verdict、三组 bootstrap CI、per-domain 数值、coverage 与全部 Gate **逐字段精确复现** summary。
- 若直接使用为了落盘而按 `(domain,user_id)` 排序的 JSONL 顺序，固定 seed 的有限次 bootstrap CI 会有约 `1e-5` 量级差异，但三项 Gate 状态和最终 verdict 不变。原因是 bootstrap RNG 索引与数组顺序绑定；不是数据或统计方向改变。
- 未做第二次训练复跑：预注册只允许这一次 recovery，正式 FAIL 后禁止自动重跑。因此严格的训练级 reproducibility verdict 为 `CANNOT_VERIFY_WITHOUT_PROHIBITED_RERUN`；artifact-level Gate arithmetic 已精确验证。

## 5. Statistical fallacy scan

Coverage：**11/11 checked**。

| Fallacy | 结论 | 说明 |
|---|---|---|
| Simpson's paradox | CAUTION | 非严格 Simpson reversal，但聚合正值掩盖了 Toys/Beauty warm-cold 相反取舍；已分域报告。 |
| Ecological fallacy | 无发现 | Gate 基于 user-level paired outcomes；未从域均值推断个体因果。 |
| Berkson's paradox | NOTE | 仅 source validation 与 cold50 构造，限制外推；不影响同用户 comparator 配对。 |
| Collider bias | 无发现 | 未加入由 action 与 outcome 共同决定的控制变量。 |
| Base-rate neglect | 无发现 | 两域 cold event 数与 H@50 分母均报告，且 event-density 门通过。 |
| Regression to the mean | 不适用 | 不是按极端 outcome 选人后的 pre/post 设计。 |
| Survivorship bias | 无发现 | 19,444 个 source user 全覆盖，无训练后样本剔除。 |
| Look-elsewhere effect | 无发现 | 三个主指标、rho、CI 和方向门均预注册；未挑预算或指标。 |
| Garden of forking paths | 无发现 | recovery 只纠正已审计的 comparator 错误；原 artifact 保留，参数未救火。 |
| Correlation != causation | NOTE | 结论限定为 OOF comparative performance，不外推为普遍因果机制。 |
| Reverse causality | 不适用 | 没有横截面方向性因果主张。 |

## 6. 决策与后续边界

1. R²-v2 Stage S 正式记 **`FAIL_STOP_R2_V2_SOURCE`**；
2. Sports validation/test 继续封存，Stage C 不解锁；
3. 不修改 hidden size、loss、dual lr、rho、feature、action、threshold 或 Gate；
4. 不创建 R²-v2.1，不在 Toys/Beauty 上继续 rescue；
5. 方法证据可以保留为：CBSA 带来小幅 overall 增益，但无法跨域同时保证 warm 改善与 cold 非劣，固定 `portfolio@2` 仍是当前接口下更可靠的简单前沿。

## 7. 主要产物

- `artifacts/phase13/explore/v1_r2_v2_source_screen_recovery_cold_candidates/status.json`
- `artifacts/phase13/explore/v1_r2_v2_source_screen_recovery_cold_candidates/summary.json`
- `artifacts/phase13/explore/v1_r2_v2_source_screen_recovery_cold_candidates/predictions_oof.jsonl`
- `artifacts/phase13/explore/v1_r2_v2_source_screen_recovery_cold_candidates/frozen_config.json`
- `artifacts/phase13/explore/v1_r2_v2_source_screen_recovery_cold_candidates/recovery_protocol.json`
- `artifacts/phase13/explore/v1_r2_v2_source_screen_recovery_cold_candidates/allocator_fold{0..4}.pt`
