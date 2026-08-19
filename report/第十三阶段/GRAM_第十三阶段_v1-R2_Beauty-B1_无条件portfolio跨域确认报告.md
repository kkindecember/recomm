# GRAM 第十三阶段：v1-R² Beauty B1 无条件 Portfolio 跨域确认报告

> **最终结论（2026-08-18）**：预注册主候选 `unconditional_portfolio2` 在 Beauty validation 上同时通过 overall NDCG@10 和 cold H@50 的 paired-bootstrap Gate，冻结 verdict 为 **PASS**。该结论支持“无条件 portfolio 相对 v0 的收益可跨域复现”；由于本轮未生成 Beauty P6 对照，不足以单独支持“在 Beauty 上也优于复杂 learned gating”的更强表述。

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate + experiment closeout
- Origin Date: 2026-08-18
- Verification Status: `ANALYZED_FROM_COMPLETED_RUN`
- Version Label: `phase13_v1_r2_beauty_b1_portfolio_confirmation_v1`
- Experiment ID: `GRAM_PHASE13_V1_R2_BEAUTY_B1_PORTFOLIO_CONFIRMATION`
- Dataset / split: `Beauty_cold50` / validation
- Primary candidate: `unconditional_portfolio2`
- Bootstrap: 10,000 paired user resamples, seed=20260818
- Gate status: **PASS**

---

## 0. 30 秒摘要

Toys validation 上的 P1–P7 表明，递增复杂度的 learned gating 始终没有超过无条件 candidate portfolio。B1 将 Toys 上冻结的 `portfolio@2` 参数原样迁移到之前未被 P1–P7 调参的 Beauty validation，不重训 GRAM、不修改 hierarchical ID、不读取 Beauty/Toys test。

Beauty 上，`portfolio@2` 使 overall NDCG@10 从 `0.038936` 提高到 `0.040550`（**+4.15%**），cold H@50 从 `0.013051` 提高到 `0.032533`（**2.49×**，69→172 个事件）。两个差值的 95% CI 下界均大于 0，且 v0 cold H@50 有 69 个事件，没有触发 `<30` 的事件密度保护条款。

代价也必须同时报告：warm NDCG@10 从 `0.075361` 降到 `0.071397`（**−5.26%**，95% CI 全为负）。因此 B1 验证的是显著的 cold–warm Pareto tradeoff，而不是“无 warm 损失”。

---

## 1. 实验目的

本轮只回答一个预注册问题：

> Toys 上观察到的“简单无条件 portfolio 有效”是否可以在未用于 P1–P7 调参的 Beauty validation 上重现？

本轮不尝试证明原始 collision-safe v1 Semantic Bridge 通过。原始 v1 仍为双域 FAIL；B1 属于后续重立基的 v1-R² route-and-resolve / portfolio 路线。

---

## 2. 配置

| 项 | 冻结值 |
|---|---|
| Dataset | `Beauty_cold50` |
| Evaluation split | validation，10,655 users |
| Cold / warm users | 5,287 / 5,368 |
| Cold-split seed | 12345 |
| Item encoder | `BAAI/bge-large-en-v1.5` |
| Pooling / normalization | CLS / L2 |
| Item embeddings | 12,101 × 1,024, fp32 |
| Resolver supervision | warm-only next-item transitions |
| Resolver train examples | 49,450，cold target count=0 |
| Resolver | residual user projector + in-batch contrastive retrieval |
| Resolver epochs / seed | 12 / 12345 |
| GRAM | 使用冻结 v0 epoch-30 validation prediction，不重训 |
| Primary policy | protect GRAM head，`portfolio@2` 放入 ranks 9–10 |
| Context policy | `portfolio@3` 放入 ranks 8–10 |
| Bootstrap | paired by user, 10,000 resamples, seed=20260818 |

说明：本轮**存在模型训练**——Beauty domain-local resolver 完成了 12 epochs 的 warm-only 训练。但最终 PASS 的新部件是不训练的 portfolio 组合规则，不是新的深度模型架构。

---

## 3. 命令与产物路径

实际启动命令：

```bash
bash experiment/phase13/run_v1_r2_beauty_b1_portfolio.sh start 7
```

主要产物：

- B1 status: `artifacts/phase13/explore/v1_r2_beauty_b1_portfolio_confirmation/status.json`
- B1 summary: `artifacts/phase13/explore/v1_r2_beauty_b1_portfolio_confirmation/summary.json`
- B1 log: `artifacts/phase13/explore/v1_r2_beauty_b1_portfolio_confirmation/run.log`
- GPU telemetry: `artifacts/phase13/explore/v1_r2_beauty_b1_portfolio_confirmation/gpu_telemetry.csv`
- Beauty P0 resolver: `artifacts/phase13/explore/v1_r2_beauty_p0/resolver.pt`
- Beauty P0 config / summary: `artifacts/phase13/explore/v1_r2_beauty_p0/{config.json,summary.json}`
- Validation prediction: `artifacts/phase13/explore/v1_r2_beauty_p0/predictions_validation.jsonl`

---

## 4. 核心数字

### 4.1 Beauty Pareto 点

| 方案 | overall NDCG@10 | cold H@50 | cold H@50 事件 | cold NDCG@10 | warm NDCG@10 | warm 保留 |
|---|---:|---:|---:|---:|---:|---:|
| v0 GRAM | 0.038936 | 0.013051 | 69 | 0.001953 | 0.075361 | 100% |
| resolver-only | 0.022670 | 0.110270 | 583 | 0.018272 | 0.027002 | 35.83% |
| **portfolio@2** | **0.040550** | **0.032533** | **172** | **0.009231** | **0.071397** | **94.74%** |
| portfolio@3 | 0.040391 | 0.039153 | 207 | 0.011297 | 0.069046 | 91.62% |

### 4.2 相对 v0

| 方案 | overall NDCG@10 | cold H@50 | cold NDCG@10 | warm NDCG@10 |
|---|---:|---:|---:|---:|
| **portfolio@2** | **+4.15%** | **2.49×（+149.28%）** | **4.73×（+372.58%）** | **−5.26%** |
| portfolio@3 | +3.74% | 3.00×（+200.00%） | 5.78×（+478.38%） | −8.38% |
| resolver-only | −41.78% | 8.45× | 9.35× | −64.17% |

Beauty 上 `portfolio@2` 的 overall NDCG@10 略高于 `portfolio@3`，而后者以更大 warm 代价换取更高 cold reachability。这与将 `portfolio@2` 作为保守主候选、`portfolio@3` 作为激进 Pareto 端点的预注册定位一致。

---

## 5. 对比

### 5.1 与 Beauty v0 对比

`portfolio@2` 对 overall NDCG@10 的绝对增益为 `+0.00161394`，相对增益为 **+4.15%**。cold H@50 增加 103 个事件（69→172），cold H@10 增加 130 个事件（20→150）。

### 5.2 与 Toys 同配置对比

Toys 并非纯规则后处理：它也独立训练了 domain-local residual user projector（12 epochs，40,344 个 warm-only transition，5,810 个唯一 warm target，cold target count=0）。Beauty 使用另一个从头训练的 resolver，没有迁移 Toys resolver 权重。

Toys validation 上，`portfolio@2` 相对 v0 的 overall NDCG@10 为 **+4.96%**、cold H@50 为 **2.89×**、warm NDCG@10 为 **−4.09%**。Beauty 的对应数字为 **+4.15% / 2.49× / −5.26%**，方向一致，幅度接近，支持“各域独立训练 resolver + 冻结 portfolio规则”相对 v0 的跨域复现。

### 5.3 与上一版对比的口径限制

Beauty B1 的 `p6_comparison_included=false`，未生成 domain-local P6 对照。因此：

- 可支持：无条件 portfolio 相对 v0 的收益在 Toys/Beauty 双域方向一致；
- 仅有 Toys 直接支持：无条件 portfolio 高于 P1–P7 复杂 gating；
- 不可表述：Beauty 已直接证明 portfolio 高于 domain-local P6。

---

## 6. Gate 结论

预注册只使用 `portfolio@2` 判定 Gate：

| Gate | observed delta | paired-bootstrap 95% CI | 结论 |
|---|---:|---:|---|
| overall NDCG@10 > v0 | +0.00161394 | `[+0.00085109, +0.00239939]` | **PASS** |
| cold H@50 > v0 | +0.01948175 | `[+0.01569888, +0.02326934]` | **PASS** |
| v0 cold H@50 events ≥30 | 69 | threshold=30 | 保护条款未触发 |
| `test_predictions_opened` | false | — | PASS |
| skipped users | 0 / 10,655 | — | PASS |

冻结 verdict：**`PASS`**。

这个 PASS 应记为 **`PASS_TO_PUBLICATION_PREPARATION`**，不等于已完成 publication-level 多 seed、多 cold ratio、多数据集验证。

---

## 7. 统计解读与偏差扫描

### 7.1 统计解读

- 两个共同主指标都必须通过，属于 intersection Gate；不是从多个显著结果中事后挑一个。
- overall NDCG@10 绝对增益较小（+0.00161）但 CI 全为正；cold H@50 绝对增益和事件数增量更大。
- warm NDCG@10 差值为 `−0.00396410 [−0.00490189, −0.00308468]`，说明 warm 损失是可测的真实 tradeoff，不能归因于抽样噪声。
- 本轮是单 split、单 resolver seed 的 exploratory confirmation；对更广泛数据分布的结论仍需 publication-level replication。

### 7.2 11/11 统计谬误覆盖

| 检查 | 状态 | 结论 |
|---|---|---|
| Simpson's paradox | checked | overall 正、cold 正、warm 负是显式报告的异质性，未隐藏子组反转 |
| Ecological fallacy | checked | 配对单位是 user，没有由组均值推断个体机制 |
| Berkson's paradox | checked | 使用完整 validation users，不因候选成功筛选；skipped=0 |
| Collider bias | checked | 无事后协变量控制 |
| Base-rate neglect | checked | 报告 cold user 基数、绝对事件数和事件密度保护 |
| Regression to the mean | checked | 未根据极端用户表现选样 |
| Survivorship bias | checked | 10,655 个用户全部进入评估，0 skipped |
| Look-elsewhere effect | checked | `portfolio@2` 为唯一预注册主候选；其他点只作上下文 |
| Garden of forking paths | checked | Beauty 参数从 Toys 冻结且禁止重调；P6 点缺失作为 protocol deviation 明示 |
| Correlation ≠ causation | checked | 只宣称离线排序指标改变，不外推为线上用户因果收益 |
| Reverse causality | checked | 不适用于同一 target 上的确定性排序方法对比 |

Fallacy scan coverage: **11/11 checked**。未发现使预注册主 Gate 失效的 RED_FLAG；单 split/seed 和 P6 对照缺失记为 CAUTION。

---

## 8. 异常与局限

1. **P6 对照缺失**：预注册文本要求同批给出 `v0 / P6 / portfolio@2 / portfolio@3 / resolver-only` 五点，实际 summary 为 `p6_comparison_included=false`。主 Gate 只依赖 `portfolio@2 vs v0`，因此 PASS 不受影响；但“复杂 gating 在 Beauty 上也更差”未获直接验证。
2. **warm 95% 不是跨域保证**：Beauty `portfolio@2` 的 warm retention 为 94.74%，比 95% 低 0.26 percentage point。既定主 Gate 未使用 warm 保护线，故不改写 PASS；但论文不得声称“跨域 warm≥95%”。
3. **P0 元数据标签错误**：`v1_r2_beauty_p0/summary.json` 内的 `experiment_id` 仍硬编码为 `...TOYS_P0`，但 config 、数据路径、样本数和 B1 experiment ID 均明确指向 Beauty。这是元数据 bug，不是数据域混用；历史 artifact 保留原样，publication runner 必须修复。
4. **没有独立重跑**：本报告解析已完成运行的 summary/log，未再次执行整个 stochastic resolver 训练，因此 Verification Status 为 `ANALYZED_FROM_COMPLETED_RUN`，不写为多 seed `VERIFIED`。

---

## 9. 下一步动作

决策：**停止 v1-R² 内的继续机制调参，进入 publication preparation，但不直接启动全矩阵。**

按顺序执行：

1. 冻结 `domain-local BGE resolver + unconditional portfolio@2`为主配置，`portfolio@3` 为 Pareto 激进端点；
2. 不启动 P8，不在 Beauty validation 或 Toys validation/test 上继续调 prefix、quota、threshold 或 gate；
3. 重写已过期的 CANARD publication plan，从“LLM + alignment + uncertainty model”改为“cold-collapse diagnosis + exact resolver ceiling + Pareto portfolio”；
4. 在新计划中先评估创新性是否足够，再决定是否需要一个真正的 learned allocation 模块；新模块不得回到已污染的 Beauty/Toys validation 上开发；
5. 若冻结现有简化方法不再加模型，则下一轮为 publication-level 多 seed / 新域 replication，不再是 exploratory 微调。

---

## 10. 资源使用与 GPU 闭环

| 项 | 结果 |
|---|---|
| GPU | physical GPU7 |
| 开始 / 完成 | 2026-08-18 18:35:21 / 18:59:01 +08:00 |
| 总 wall time | 约 23 分 40 秒 |
| P0 resolver runtime | 709.54 s |
| 遥测样本 | 139 |
| GPU used memory min / max | 39,151 / 42,667 MiB |
| 观测到的峰值增量 | 3,516 MiB |
| 预注册增量上限 | 5,120 MiB |
| minimum free observed | 5,904 MiB |
| OOM / timeout / crash | 无 |
| API 成本 | 0 |

本实验不运行 GRAM training/beam search，不要求 30G lease，且未停止或修改其他 GPU 进程。完成后 GPU7 显存回到运行起点的 39,151 MiB，B1 runner/workload 进程已退出。资源闭环记为 **completed / no dedicated holder transition required**。
