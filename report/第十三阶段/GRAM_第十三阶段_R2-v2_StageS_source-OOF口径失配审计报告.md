# GRAM 第十三阶段：R²-v2 Stage S Source OOF 结果与 Comparator 口径失配审计报告

> **最终结论（2026-08-19）**：运行本身完整完成，生成 summary 的机械 verdict 为 **`INCONCLUSIVE_STOP_R2_V2_SOURCE`**；但 post-run 完整性审计发现，代码中的 `portfolio@2` 并非 Stage S 声明要挑战的 Beauty B1 冻结 incumbent。因此本次运行的科学完整性 verdict 为 **`INVALID_FOR_PREREGISTERED_INCUMBENT_COMPARISON`**。不得开启 Sports，不得把机械 INCONCLUSIVE 当作正式 R²-v2 Stage S 结论，也不得自动重跑。

> **后续状态（2026-08-19 11:27）**：用户随后授权了唯一一次 catalog-cold 等价 recovery。修正版正式 verdict 为 **`FAIL_STOP_R2_V2_SOURCE`**；详见 `GRAM_第十三阶段_R2-v2_StageS_catalog-cold等价恢复正式结果.md`。本报告仍作为原无效运行的历史审计保留。

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate + experiment closeout
- Origin Date: 2026-08-19
- Verification Status: `ANALYZED_FROM_COMPLETED_RUN_WITH_COMPARATOR_RED_FLAG`
- Experiment ID: `GRAM_PHASE13_R2_V2_CBSA_SOURCE_OOF`
- Dataset / split: Toys_cold50 + Beauty_cold50 / validation 5-fold OOF
- Primary budget: `rho=0.97`
- Declared comparator: frozen B1 `unconditional portfolio@2`
- Implemented comparator: raw resolver-top2 portfolio
- Mechanical status: `INCONCLUSIVE_STOP_R2_V2_SOURCE`
- Integrity status: **`INVALID_FOR_PREREGISTERED_INCUMBENT_COMPARISON`**

---

## 0. 摘要

Stage S 在 GPU2 上正常完成，共覆盖 Toys 8,789、Beauty 10,655，合计 19,444 个 source validation user。五折 train/held overlap 均为 0，输出完整，Sports/test 未读取，无 crash/OOM/NaN。

针对代码内 comparator，CBSA 的 overall NDCG@10 差值 CI 全为正，cold H@50 也通过非劣；warm NDCG@10 点估计为正但 CI 跨 0，因此机械 verdict 为 INCONCLUSIVE。

然而 comparator 审计发现：B1 通过的 incumbent 从 resolver 中选择 **catalog-cold candidates**，Stage S 代码却选择任意 catalog item 的 raw resolver top2/top3。数值上，Stage S 的 comparator 在 Toys/Beauty cold H@50 仅为 `0.016716/0.022319`，而真实 B1 incumbent 为 `0.029769/0.032533`。这是不同方法，不是同一方法的抽样波动。

只读诊断表明，当前 CBSA OOF 输出相对真实 B1 incumbent 的 cold H@50 仅保留 `68.06%`，远低于 95% 非劣界。但 CBSA 训练时也使用了错误动作，因此该诊断不能替代修正版 Stage S，只用于证明本轮不能验证预注册研究问题。

---

## 1. 实验目的

预注册问题是：单一跨域共享、预算条件化 CBSA 能否相对已通过 Beauty B1 的冻结 `portfolio@2`：

1. 提高 domain-balanced overall NDCG@10；
2. 提高 warm NDCG@10；
3. 保持 cold H@50 相对非劣（最多下降 5%）。

Toys/Beauty 只作 source OOF；只有全部 Gate 通过才允许讨论 Sports。

---

## 2. 配置

| 项 | 冻结值 |
|---|---|
| Source domains | Toys + Beauty，等权汇总 |
| OOF | 5 folds，`user_id + domain + fixed salt` |
| Users | Toys 8,789；Beauty 10,655；总计 19,444 |
| Features | 36 continuous features + missing indicators + `rho` |
| Allocator | `input → 64 → GELU → Dropout(.1) → 32 → GELU → 3` |
| Parameters | 6,915 |
| Actions | `a0 / a2 / a3` |
| Budget grid | `.93/.95/.97/.99`；主 Gate `.97` |
| Optimizer | AdamW，lr=1e-3，wd=1e-4，batch=512 |
| Training | 50 epochs/fold，1,700 steps/fold，8,500 total |
| Seed | 20260819 |
| Bootstrap | paired 10,000 resamples，seed=20260819 |

---

## 3. 命令与产物路径

正式启动命令：

```bash
bash experiment/phase13/run_v1_r2_v2_source_screen.sh start 2
```

主要产物：

- `artifacts/phase13/explore/v1_r2_v2_source_screen/status.json`
- `artifacts/phase13/explore/v1_r2_v2_source_screen/summary.json`
- `artifacts/phase13/explore/v1_r2_v2_source_screen/predictions_oof.jsonl`
- `artifacts/phase13/explore/v1_r2_v2_source_screen/allocator_fold{0..4}.pt`
- `artifacts/phase13/explore/v1_r2_v2_source_screen/comparator_integrity_audit.json`
- `artifacts/phase13/explore/v1_r2_v2_source_screen/gpu_telemetry.csv`
- `artifacts/phase13/explore/v1_r2_v2_source_screen/run.log`

首次 sandbox tmux 启动失败发生在 workload 前，保留为 `status.launch_failed_20260819T104203.json`；它不属于科学运行。

---

## 4. 机械结果：仅针对代码内 comparator

### 4.1 Aggregate Gate

| Gate | 差值（CBSA − implemented comparator） | 95% CI | 机械判定 |
|---|---:|---:|---|
| overall NDCG@10 | +0.00046688（+1.26%） | `[+0.00007234,+0.00085864]` | PASS |
| warm NDCG@10 | +0.00038134（+0.56%） | `[-0.00023223,+0.00100549]` | **INCONCLUSIVE** |
| cold H@50 | +0.00168263（+8.62%） | `[+0.00003493,+0.00331531]` | PASS（且高于非劣要求） |

其他机械 Gate：方向一致性=true、intervention coverage=`55.19%`、两域 incumbent cold H@50 events=`73/118`、完整性=true。

因此 summary 按代码内 comparator 给出：**`INCONCLUSIVE_STOP_R2_V2_SOURCE`**。

### 4.2 分域结果

| 域 | overall Δ | warm NDCG@10 Δ | cold H@50：CBSA / implemented comparator |
|---|---:|---:|---:|
| Toys | +0.00071438 | +0.00017270 | 0.020838 / 0.016716 |
| Beauty | +0.00021937 | +0.00058997 | 0.021562 / 0.022319 |

动作分布：`a0=44.81%`、`a2=4.89%`、`a3=50.30%`；requested/effective action 完全一致。

预算曲线几乎不变：coverage 从 rho=.93 的 55.07% 到 rho=.99 的 55.26%，主指标也基本相同。五折 dual 最终为 0 或接近 0，说明训练 warm constraint 大部分时间处于 slack；预算条件没有形成明显可区分的部署前沿。这是机制诊断，不改变机械 verdict。

---

## 5. Comparator 完整性 RED_FLAG

### 5.1 代码证据

- B1 incumbent：`b1_portfolio_confirmation.py:103-108` 明确只从 `item in cold_items` 的 resolver candidates 构造 portfolio；
- Stage S：`r2_v2_budgeted_slate_allocator.py:199-207` 只检查 `item in catalog`，没有 cold-state filter；
- 计划 Stage S：明确声明主对照为“当前冻结 incumbent unconditional portfolio@2”。

因此 implemented comparator 与 frozen B1 incumbent 不同。

### 5.2 数值指纹

| 域 | 指标 | 本次 implemented comparator | 冻结 B1 incumbent |
|---|---|---:|---:|
| Toys | overall NDCG@10 | 0.033723 | 0.035016 |
| Toys | cold H@50 | 0.016716 | 0.029769 |
| Beauty | overall NDCG@10 | 0.040220 | 0.040550 |
| Beauty | cold H@50 | 0.022319 | 0.032533 |

差异远大于浮点误差，确认是方法口径不同。

### 5.3 只读对齐诊断（不是修正版 Gate）

将已生成的 CBSA OOF metrics 与真正 B1 incumbent 在同批用户上配对，仅作完整性诊断：

| 指标 | 差值 | 95% CI | 相对变化 |
|---|---:|---:|---:|
| overall NDCG@10 | −0.00034483 | `[-0.00080620,+0.00012151]` | −0.91% |
| warm NDCG@10 | +0.00221657 | `[+0.00155264,+0.00290202]` | +3.35% |
| cold H@50 | **−0.00995046** | `[-0.01209093,−0.00793444]` | **−31.94%；retention 68.06%** |

预注册 cold 非劣边界是 `−0.00155753`；诊断 CI 上界仍远低于该边界。Toys/Beauty 的 cold H@50 分别从 `0.029769→0.020838`、`0.032533→0.021562`。

此诊断不能成为正式 FAIL：CBSA 的动作 reward 与训练本身也建立在 raw-resolver action 上。它只证明本次模型/结果不能冒充“相对 B1 incumbent”的有效 Stage S。

---

## 6. Gate 与科学结论

| 层次 | 结论 |
|---|---|
| 运行完整性 | COMPLETED；19,444/19,444 OOF，五折无交叉 |
| 代码内机械 Gate | `INCONCLUSIVE_STOP_R2_V2_SOURCE` |
| 预注册研究问题完整性 | **`INVALID_FOR_PREREGISTERED_INCUMBENT_COMPARISON`** |
| Sports | **LOCKED；不得启动** |
| 自动 recovery | 禁止 |

证据、推断与建议分开如下：

- 证据：implemented comparator 与 B1 incumbent 的候选 filter 和指标均不同；
- 推断：本次运行没有回答“CBSA 是否优于已通过的 portfolio@2”；
- 建议：停止并由用户决定是否将其认定为纯实现错误、授权一次严格等价 recovery。若不授权，本 R²-v2 以 invalid/inconclusive 收尾。

---

## 7. 统计解释与 11/11 谬误扫描

| 检查 | 状态 | 结论 |
|---|---|---|
| Simpson's paradox | checked | 双域分别报告；implemented comparator 下 overall 同向，但 true incumbent 下两域均为负点估计 |
| Ecological fallacy | checked | bootstrap 以 user 配对，未用域均值推断个体机制 |
| Berkson's paradox | checked | 19,444 source users 全覆盖，无按候选成功筛除 |
| Collider bias | checked | 无事后协变量控制 |
| Base-rate neglect | checked | 报告 cold events 73/118，并核对真实 incumbent 事件密度 |
| Regression to the mean | checked | 用户未按极端表现选择 |
| Survivorship bias | checked | 五折 OOF 无缺失；0 skipped |
| Look-elsewhere effect | checked | 主 budget 固定 .97；其他预算未替代主结论 |
| Garden of forking paths | RED_FLAG | 不是事后挑参数，而是预注册内部“动作字面定义 vs incumbent 身份”冲突；已明确降级为 invalid |
| Correlation ≠ causation | checked | 只解释离线排序差异，不外推线上因果收益 |
| Reverse causality | checked | 不适用于同 target 的确定性 slate 对比 |

Fallacy scan coverage: **11/11**。Comparator mismatch 是终止性完整性 RED_FLAG。

---

## 8. 异常与局限

1. **Comparator mismatch（科学终止项）**：本报告的主要 RED_FLAG。
2. **首次 tmux sandbox 失败**：workload 前失败，已归档，不影响正式运行。
3. **GPU 峰值增量**：GPU2 telemetry used memory 42,376→44,425 MiB，观测增量 2,049 MiB，比预估 2,048 MiB 高 1 MiB；共享卡并发和遥测粒度可能贡献该 1 MiB。最低空闲 4,145 MiB，高于 3,072 MiB 准入线，无 OOM。
4. **没有重跑**：本报告为完成运行的独立解析与只读对齐审计，未执行 recovery，Verification Status 不写为 VERIFIED。

---

## 9. 下一步动作

本报告完成当时的冻结动作（后续 recovery 结果见页首更新）：

1. 不启动 Sports；
2. 不修改现有 summary、OOF、checkpoint 或原 decision；
3. 不把 post-hoc true-incumbent 诊断当正式 Gate；
4. 当时等待用户决定是否授权一次“纠正 cold-candidate action/comparator、其余 SHA/seed/Gate 不变”的等价 recovery；
5. 用户后来已授权且 recovery 正式 `FAIL_STOP_R2_V2_SOURCE`，故 R²-v2 最终按修正版 FAIL 收尾。

---

## 10. 资源使用与闭环

| 项 | 结果 |
|---|---|
| GPU | physical GPU2 |
| 开始 / 完成 | 10:43:07 / 10:46:17 +08:00 |
| Scientific runtime | 178.56 s |
| 遥测样本 | 19 |
| GPU used min / max | 42,376 / 44,425 MiB |
| 观测增量 | 2,049 MiB |
| minimum free | 4,145 MiB |
| OOM / timeout / crash | 无 |
| Workload PID after completion | 已退出 |
| GPU0 / GPU5 | 未调整 |
| API 成本 | 0 |

本次不使用 holder/lease，不停止或迁移任何已有 GPU 进程。资源闭环记为 **completed / workload exited / shared GPU untouched**。
