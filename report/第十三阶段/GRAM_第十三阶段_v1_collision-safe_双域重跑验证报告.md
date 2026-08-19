# GRAM 第十三阶段：v1 Collision-Safe 双域重跑验证报告

> **最终状态（2026-08-18）**：Toys 与 Beauty 正式实验均已完成，科学结果通过完整性核对，两个单域 Gate 均为 **FAIL**，因此双域汇总 Gate 为 **FAIL**。两域训练均成功，但实验后资源保护恢复均为 degraded；科学结果与资源闭环状态在本文中分别记录。

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent + academic-paper
- Origin Mode: validate + staged report drafting
- Origin Date: 2026-08-17
- Completion Date: 2026-08-18
- Verification Status: `DUAL_DOMAIN_SCIENTIFIC_VERIFIED_RESOURCE_DEGRADED`
- Version Label: `phase13_v1_collision_safe_dual_domain_rerun_v1`
- Experiment IDs:
  - `GRAM_PHASE13_V1_COLLISION_SAFE_TOYS_V1`
  - `GRAM_PHASE13_V1_COLLISION_SAFE_BEAUTY_V1`
- Datasets: `Toys_cold50` / `Beauty_cold50`
- Training seed: 2023
- Cold-split seed: 12345
- Gate status: **TOYS FAIL；BEAUTY FAIL；DUAL-DOMAIN FAIL**

---

## 0. 30 秒摘要

原 v1 报告曾显示 Semantic Bridge 在两个数据集上带来显著 cold 推荐增益，但后续 lexical-ID 碰撞审计发现：大量 cold item 与 warm item 或其他 cold item 共用同一完整 lexical ID。GRAM 的原始评测只比较解码后的 lexical-ID 字符串，因此无法确认命中的是哪一个底层 item，原 v1 的强增益不能直接解释为 item-level cold 推荐能力提升。

本轮实验保持原 v1 的数据、Semantic Bridge MLP、GRAM 训练配置和评测协议不变，只将合并后的 v1 ID 表转换为 **collision-safe、全局唯一**的 item lexical ID，再从头训练 Toys 与 Beauty 两个完整 30-epoch GRAM 模型。本轮结果将回答：

> **在彻底消除 lexical-ID 别名后，Sentence-BERT + MLP Semantic Bridge 是否仍能相对 vanilla GRAM 改善 cold NDCG@10？**

截至 2026-08-18 06:06 +08:00，Toys 与 Beauty 均已完成 30/30 epoch、6/6 validation 和 1/1 test。Toys / Beauty 的 collision-safe cold NDCG@10 分别为 **0.154617% / 0.161338%**，相对同域 v0 分别下降 **49.31% / 9.87%**。两域均未达到预注册的 +5% 门槛，因此：

- Toys 单域 Gate：**FAIL**；
- Beauty 单域 Gate：**FAIL**；
- 双域汇总 Gate：**FAIL**；
- 旧 v1 raw 指标仍不能作为 item-level 结论，最终判定只依据本轮 collision-safe 全量 test。

---

## 1. 为什么必须重跑 v1

### 1.1 原 v1 的设计目标

v1 Minimum Semantic Bridge 使用冻结的 item 文本 embedding 学习从文本语义到 GRAM hierarchical ID 的映射：

1. 用 `all-MiniLM-L6-v2` 编码 item 文本；
2. 只用 warm item 监督训练逐层 MLP 分类头；
3. 对 cold item 预测 hierarchical ID；
4. warm item 保持原始 GRAM ID，cold item 使用 MLP 预测 ID；
5. 将所有 ID 加入 GRAM 的约束解码 Trie，重新训练并评测。

按照原始结果，v1 的 cold 指标看似在双域均显著超过 v0：

| Dataset | v0 cold H@10 | 原 v1 raw cold H@10 | v0 cold NDCG@10 | 原 v1 raw cold NDCG@10 |
|---|---:|---:|---:|---:|
| Toys | 0.608% | 1.351% | 0.305% | 0.872% |
| Beauty | 0.306% | 0.802% | 0.179% | 0.418% |

这些 raw 指标曾被解释为 v1 Gate PASS，但该解释后来被碰撞审计推翻。

### 1.2 碰撞审计推翻了什么

原 v1 对每个 cold item 独立取 MLP 各层 argmax，但没有保证合并后的完整 ID 全局唯一。因此，一个被模型生成的 lexical ID 可能同时对应多个 item。原评测只判断预测字符串是否等于 gold 字符串，无法辨别具体 item。

将所有 ambiguous gold ID 保守置零后，原 v1 的 strict 指标如下：

| Dataset | 原 v1 raw H@10 | 原 v1 strict H@10 | 原 v1 raw NDCG@10 | 原 v1 strict NDCG@10 | v0 NDCG@10 |
|---|---:|---:|---:|---:|---:|
| Toys | 1.351% | 0.270% | 0.872% | 0.172% | 0.305% |
| Beauty | 0.802% | 0.210% | 0.418% | 0.135% | 0.179% |

双域的 strict H@10 和 NDCG@10 均未超过 v0。这说明原 v1 的强提升主要来自 lexical-ID 别名，不能作为 item-level cold 推荐收益的可靠证据。

需要强调：该审计否定的是**原 v1 评测结论的有效性**，并未直接证明 Semantic Bridge 本身无效。本轮 collision-safe 重跑正是区分这两种解释的必要实验。

---

## 2. 本轮研究问题与 Gate

### 2.1 核心研究问题

在数据、模型、训练 seed 和 GRAM 超参数不变的条件下，消除 v1 lexical-ID 碰撞后：

1. collision-safe v1 的 cold NDCG@10 是否仍高于 v0？
2. 两个数据集是否给出一致方向？
3. warm 侧是否出现明显退化？
4. 原 v1 的强提升中，有多少能够在 item-level 唯一 ID 口径下保留？

### 2.2 预注册 Gate

沿用 CANARD 探索计划的 v1 Gate：

- 单域 PASS：collision-safe v1 的 cold NDCG@10 相对同域 v0 提升 **≥5%**；
- 单域 FAIL：相对提升 **<5%**；
- 双域汇总：
  - `PASS`：Toys 与 Beauty 均达到 +5%；
  - `MIXED`：仅一个数据集达到 +5%；
  - `FAIL`：两个数据集均未达到 +5%。

相对变化计算式：

```text
Δcold_NDCG@10 = (NDCG@10_collision_safe_v1 / NDCG@10_v0 - 1) × 100%
```

冻结的 v0 门槛为：

| Dataset | v0 cold NDCG@10 | v1 PASS 所需最低值（+5%） |
|---|---:|---:|
| Toys | 0.305% | 0.320% |
| Beauty | 0.179% | 0.188% |

由于两域均为单 seed，最终报告除相对增益外还必须同时报告绝对差值（percentage point），避免在极小基线上仅用相对百分比夸大效果。

---

## 3. Collision-Safe v1 的构造

### 3.1 保持不变的部分

本轮没有重新定义 Semantic Bridge，也没有引入 v2/v3 组件：

- 使用原 v1 的 Sentence-BERT embeddings；
- 使用原 v1 已训练的 MLP checkpoint；
- cold item 的基础 L-token MLP 预测保持不变；
- warm item 的完整原始 GRAM ID 逐字节保持不变；
- GRAM backbone、训练超参数、beam size、SASRec similar-item 输入及评测脚本保持不变。

因此，本轮相对原 v1 的唯一实验变量是：**完整 lexical ID 是否具有 item-level 全局唯一性**。

### 3.2 碰撞消解规则

对每组存在冲突的 cold ID：

1. warm ID 被视为不可修改的保留路径；
2. 无碰撞 cold ID 保持原样；
3. 与 warm 或其他 cold item 冲突的 cold ID，在原路径末尾追加最小可用数字 token；
4. 若候选后缀已被其他完整路径占用，则继续递增；
5. 保持 item 覆盖、文件行序和 cold ID 的原始语义前缀不变。

示意：

```text
warm_A  |a|b             → |a|b
cold_X  |a|b             → |a|b|0
cold_Y  |a|b             → |a|b|1
```

该后缀只承担 item-level 消歧作用，不应被解释为新增语义层。

### 3.3 完整 ID 审计

| Dataset | Items | Warm | Cold | 修改的 cold ID | 修改率 | 输入 duplicate excess | 输出 duplicate excess | 最大后缀 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Toys | 11,924 | 5,961 | 5,963 | 1,686 | 28.27% | 1,352 | **0** | 18 |
| Beauty | 12,101 | 6,049 | 6,052 | 1,167 | 19.28% | 922 | **0** | 10 |

硬不变量均已通过：

- warm IDs unchanged；
- row order unchanged；
- modified cold IDs 只追加一个数字 token；
- raw token-path 全局唯一；
- 输出 item 数与输入完全一致。

---

## 4. GRAM 真实链路端到端审计

仅证明 ID 文件中的 token 元组唯一还不充分。GRAM 实际执行过程为：读取 ID 字符串 → T5 SentencePiece 编码 → 删除 `|` 分隔 token → 构造约束解码 Trie → beam generation → T5 decode → 字符串精确匹配。因此，本轮在正式重跑前额外验证了编码后的唯一性。

| Dataset | raw paths unique | T5 encoded paths unique | decoded strings unique | encoded duplicate excess | 最大编码长度 | `target_max_len=32` 截断风险 |
|---|---:|---:|---:|---:|---:|---:|
| Toys | 11,924 / 11,924 | 11,924 / 11,924 | 11,924 / 11,924 | **0** | 7 | 0 |
| Beauty | 12,101 / 12,101 | 12,101 / 12,101 | 12,101 / 12,101 | **0** | 9 | 0 |

此外：

- 两域 embedding 均覆盖全部 item；
- cold/warm 集合完整、互斥并覆盖全部 ID；
- `n_cold_fallback=0`，不存在缺 embedding 后退回 source ID 的情况；
- collision-safe 输出可由当前 v1 文件确定性重新生成并逐行匹配；
- smoke test 两域各完成 100 条 test 推理，gold 编码链路 100/100 一致；
- smoke test 实际覆盖 12 个 Toys、10 个 Beauty 的 modified cold target，确实经过新增后缀路径。

### 4.1 可复现性注意事项

Beauty 中有 7/6,052 个 cold item 的 MLP top-2 logit 间隔仅约 `4.5e-5～1.8e-4`，不同设备或矩阵 batch 形状可能使 argmax 在两个近似并列 token 间翻转。本轮正式训练不会在线重新计算 MLP，而是读取已经冻结并通过唯一性审计的 ID 文件，因此该数值边界不影响当前运行的内部一致性。

若未来重新生成 Beauty v1 ID，应固定设备、推理 batch、checkpoint 与输入文件 hash，并把最终 ID 文件 hash 一同归档。

---

## 5. 正式实验配置

### 5.1 共同配置

| 项 | Toys | Beauty |
|---|---:|---:|
| Dataset | `Toys_cold50` | `Beauty_cold50` |
| Items / cold / warm | 11,924 / 5,963 / 5,961 | 12,101 / 6,052 / 6,049 |
| Hierarchical base | c32 / L5 | c128 / L7 |
| ID type | `split_v1_mlpcold_collision_safe` | `split_v1_mlpcold_collision_safe` |
| Backbone | t5-small | t5-small |
| rec epochs | 30 | 30 |
| rec batch / grad accumulation | 16 / 8 | 16 / 8 |
| rec learning rate | 1e-3 | 1e-3 |
| validation / checkpoint interval | 5 epochs | 5 epochs |
| beam size | 50 | 50 |
| top-k similar item | 5 | 10 |
| training seed | 2023 | 2023 |
| cold split | η=0.5, seed=12345 | η=0.5, seed=12345 |

### 5.2 正式运行记录

| Dataset | Sub-experiment | Started at | GPU | Hard timeout | 最终状态 |
|---|---|---|---:|---:|---|
| Toys | `v1_collision_safe_toys` | 2026-08-16 23:08 +08:00 | 0 | 259,200s（72h） | `succeeded_resource_degraded`（科学运行完成于 2026-08-17 14:22） |
| Beauty | `v1_collision_safe_beauty` | 2026-08-16 23:08 +08:00 | 5 | 259,200s（72h） | `succeeded_resource_degraded`（科学运行完成于 2026-08-18 06:05） |

两个实验均从干净输出目录启动，`rec_model_path=None`，没有加载旧 v1 checkpoint。此前因资源与控制链审计主动中止的尝试已单独归档，不纳入本轮结果。

---

## 6. 最终结果

> 取数口径：只使用当前正式输出目录中最后一次完整 test prediction；必须同时满足 30/30 epoch、6/6 validation、1/1 test、预测行数完整、`n_pred_rows_missing_user_map=0`。

### 6.1 Toys 全量 test

| Subset | n | hit@1 | hit@3 | hit@5 | hit@10 | hit@20 | hit@50 | ndcg@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 8,789 | 1.331% | 2.742% | 3.584% | 4.733% | 6.349% | 9.250% | 2.847% |
| warm | 4,347 | 2.599% | 5.406% | 7.085% | 9.294% | 12.399% | 18.058% | 5.599% |
| **cold** | **4,442** | **0.090%** | **0.135%** | **0.158%** | **0.270%** | **0.428%** | **0.630%** | **0.155%** |

精确的 cold H@10 / NDCG@10 分别为 0.270149% / 0.154617%。表中百分比仅为显示取整，Gate 计算使用未取整值。

### 6.2 Beauty 全量 test

| Subset | n | hit@1 | hit@3 | hit@5 | hit@10 | hit@20 | hit@50 | ndcg@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | 10,655 | 1.802% | 3.182% | 4.327% | 5.969% | 8.062% | 11.441% | 3.590% |
| warm | 5,421 | 3.486% | 6.124% | 8.283% | 11.437% | 15.458% | 21.933% | 6.900% |
| **cold** | **5,234** | **0.057%** | **0.134%** | **0.229%** | **0.306%** | **0.401%** | **0.573%** | **0.161%** |

精确的 cold H@10 / NDCG@10 分别为 0.305694% / 0.161338%。表中百分比仅为显示取整，Gate 计算使用未取整值。

### 6.3 与 v0 的主要比较

| Dataset | Metric | v0 | Collision-safe v1 | 绝对差值 | 相对变化 | Gate |
|---|---|---:|---:|---:|---:|---|
| Toys | cold H@10 | 0.608% | 0.270149% | −0.337851 pp | −55.57% | 辅助指标 |
| Toys | **cold NDCG@10** | **0.305%** | **0.154617%** | **−0.150383 pp** | **−49.31%** | **FAIL** |
| Toys | warm H@10 | 8.948% | 9.293766% | +0.345766 pp | +3.86% | 退化监测：未退化 |
| Toys | warm NDCG@10 | 5.404% | 5.598919% | +0.194919 pp | +3.61% | 退化监测：未退化 |
| Beauty | cold H@10 | 0.306% | 0.305694% | −0.000306 pp | −0.10% | 辅助指标 |
| Beauty | **cold NDCG@10** | **0.179%** | **0.161338%** | **−0.017662 pp** | **−9.87%** | **FAIL** |
| Beauty | warm H@10 | 11.621% | 11.437004% | −0.183996 pp | −1.58% | 退化监测：轻微下降 |
| Beauty | warm NDCG@10 | 6.919% | 6.900067% | −0.018933 pp | −0.27% | 退化监测：轻微下降 |

### 6.4 原 v1 raw、原 v1 strict 与 collision-safe v1

| Dataset | Version / 口径 | cold H@10 | cold NDCG@10 | 是否能支持 item-level 结论 |
|---|---|---:|---:|---|
| Toys | 原 v1 raw | 1.351% | 0.872% | 否：含 lexical-ID alias |
| Toys | 原 v1 collision-aware strict 下界 | 0.270% | 0.172% | 保守分析值，不是新模型结果 |
| Toys | **collision-safe v1 重跑** | **0.270149%** | **0.154617%** | 是：全局唯一 ID 的完整 test 结果 |
| Beauty | 原 v1 raw | 0.802% | 0.418% | 否：含 lexical-ID alias |
| Beauty | 原 v1 collision-aware strict 下界 | 0.210% | 0.135% | 保守分析值，不是新模型结果 |
| Beauty | **collision-safe v1 重跑** | **0.305694%** | **0.161338%** | 是：全局唯一 ID 的完整 test 结果 |

### 6.5 Validation 曲线

#### Toys

| Epoch | val H@10 | val NDCG@10 |
|---:|---:|---:|
| 5 | 4.574% | 2.756% |
| 10 | 5.257% | 3.043% |
| 15 | 5.427% | 3.221% |
| 20 | 5.518% | 3.291% |
| 25 | 5.382% | 3.304% |
| 30 | 5.177% | 3.133% |

#### Beauty

| Epoch | val H@10 | val NDCG@10 |
|---:|---:|---:|
| 5 | 5.584% | 3.349% |
| 10 | 6.091% | 3.570% |
| 15 | 6.335% | 3.694% |
| 20 | 6.335% | 3.746% |
| 25 | 6.363% | 3.785% |
| 30 | 6.363% | 3.770% |

---

## 7. 完整性与资源闭环

| 检查项 | Toys | Beauty |
|---|---|---|
| 训练 epoch | 30/30 | 30/30 |
| validation | 6/6 | 6/6 |
| test inference | 1/1 | 1/1 |
| test prediction rows | 8,789/8,789 | 10,655/10,655 |
| missing user map | 0 | 0 |
| 最终 status | `succeeded_resource_degraded` | `succeeded_resource_degraded` |
| GPU peak allocated / reserved | 15,753.175 / 24,292 MiB | 15,812.584 / 19,404 MiB |
| exact holder restored | 否 | 否 |
| watchdog final state | `protected_degraded`（40,239 MiB；requested 30,500 MiB） | `protected_degraded`（18,263 MiB；requested 15,500 MiB） |

若最终状态为 `succeeded_resource_degraded`，科学结果仍可单独判断，但资源闭环必须如实写为 degraded，不得表述为 exact holder restore。

Toys 与 Beauty 均满足全部科学结果完整性条件。Beauty 的 10,655 个 prediction 用户无重复，与 test 用户集合完全一致，warm/cold 分区样本数之和与 overall 一致，各项聚合指标也通过加权一致性检查。两域的 degraded 终态都来自训练完成后的资源保护精确恢复失败（`restore_rc=1`），不改变已落盘且用户行数完整的 test prediction 与 cold/warm 评测结果；但资源闭环明确记为 degraded，不得与科学运行成功合并表述。

---

## 8. Gate 判决

### 8.1 Toys

- collision-safe cold NDCG@10：`0.154617%`
- 相对 v0（0.305%）：`−0.150383 pp / −49.31%`
- 单域 Gate：**FAIL**

Toys 的 collision-safe cold NDCG@10 不仅未达到 +5% 门槛（0.320%），还低于 v0 和原 v1 collision-aware strict 下界（0.172%）。warm H@10 与 NDCG@10 分别提高 3.86% 和 3.61%，因此主要失败信号集中在 cold item 推荐，而非 warm 侧整体退化。

解释边界：这是单 seed、预注册阈值下的描述性 Gate 判定，不是统计显著性检验；当前没有重复 seed、置信区间或 p-value，不能把差异表述为“统计显著”。统计/方法谬误扫描已覆盖 11/11 项：本结果按 warm/cold 分层报告并给出样本数，主 Gate 在实验前冻结，未发现 Simpson、生态层级、选择/存活、multiple-testing 后挑选或因果措辞问题；剩余不确定性主要来自单 seed，故总体置信度记为 **CAUTION**，但不改变预注册 Gate 的 FAIL 判定。

### 8.2 Beauty

- collision-safe cold NDCG@10：`0.161338%`
- 相对 v0（0.179%）：`−0.017662 pp / −9.87%`
- 单域 Gate：**FAIL**

Beauty 的 collision-safe cold NDCG@10 低于 +5% 门槛（0.188%），也低于 v0；cold H@10 与 v0 基本持平（−0.10%），warm H@10 / NDCG@10 分别轻微下降 1.58% / 0.27%。这些 warm 变化不构成大幅整体退化，但也没有提供可以抵消 cold Gate 失败的证据。

### 8.3 双域结论

**FAIL**

Toys / Beauty 的 collision-safe cold NDCG@10 均低于同域 v0，且均未达到预注册 +5% 门槛，因此双域 Gate 按冻结规则判为 **FAIL**。原 v1 raw cold NDCG@10 为 0.872% / 0.418%，而本轮独立 collision-safe 重跑得到 0.154617% / 0.161338%；Beauty 虽高于原 strict 下界 0.135%，仍低于 v0 的 0.179%。这支持“原 v1 强信号主要由 lexical-ID alias 造成”的解释，而不支持现有 Minimum Semantic Bridge 已实现稳定 item-level cold 改善的解释。

统计/方法谬误扫描已覆盖 11/11 项。两域均分层报告 warm/cold 且样本完整，主 Gate 事先冻结，未见 Simpson 方向反转、生态层级错配、基准率忽略、存活者偏差、多重比较后挑选或因果/反向因果措辞问题。但两域均只有单 seed，没有重复 seed、置信区间或预注册统计检验；Toys 与 Beauty 又是反复用于方向开发的数据域，因此总体证据置信度为 **CAUTION**。这一不确定性限制对效应普遍性的解释，但不改变本次预注册 Gate 的 FAIL 判定。

---

## 9. 最终结论

在消除 lexical-ID 碰撞后，collision-safe v1 在 Toys / Beauty 上的 cold NDCG@10 分别为 **0.154617% / 0.161338%**，相对 v0 分别变化 **−49.31% / −9.87%**。两域均低于 v0 且未达到 +5% 门槛，所以 Toys、Beauty 单域 Gate 和双域汇总 Gate 均判为 **FAIL**。这些是单 seed 的描述性 Gate 结果，不应表述为统计显著差异。

结果表明旧 v1 的强收益主要由 lexical-ID alias 造成，现有 Minimum Semantic Bridge 不足以在两域稳定改善 item-level cold 推荐。因此，不应再将旧 v1 raw 结果当作 v2/v3 的已验证前提；后续语义方案应以 collision-safe 口径为硬约束，转向 Plan Z、显式 retrieval/reranking 或重新设计具有容量约束的 ID assignment，而不是在旧 v1 结论上继续堆叠组件。

---

## 10. 产物路径

### 10.1 代码与协议

- Collision-safe ID 构造：`experiment/phase13/protocol/make_collision_safe_ids.py`
- 单元测试：`experiment/phase13/tests/test_collision_safe_ids.py`
- 正式 runner：`experiment/phase13/run_phase13_explore.sh`
- cold/warm 评测：`experiment/phase13/protocol/eval_cold_warm.py`
- 原碰撞审计：`report/第十三阶段/GRAM_第十三阶段_v1_success-mechanism_碰撞审计报告.md`

### 10.2 ID 与审计报告

- Toys ID：`GRAM/rec_datasets/Toys_cold50/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split_v1_mlpcold_collision_safe.txt`
- Beauty ID：`GRAM/rec_datasets/Beauty_cold50/item_generative_indexing_hierarchy_v1_c128_l7_len32768_split_v1_mlpcold_collision_safe.txt`
- Toys ID audit：`artifacts/phase13/explore/v1_collision_safe/toys_id_report.json`
- Beauty ID audit：`artifacts/phase13/explore/v1_collision_safe/beauty_id_report.json`

### 10.3 正式运行

- Toys：`artifacts/phase13/explore/v1_collision_safe_toys/`
- Beauty：`artifacts/phase13/explore/v1_collision_safe_beauty/`
- 关键文件：`status.json`、`run.log`、`gpu_telemetry.csv`、`predictions/`、`metrics_cold_warm.json`

---

## 11. 结果完成核对清单

- [x] 两域 `status.json` 均进入终态；
- [x] Toys 确认 30/30 epoch、6/6 validation、1/1 test；
- [x] Toys 确认 test prediction 为 8,789/8,789 行；
- [x] Toys 确认 `n_pred_rows_missing_user_map=0`；
- [x] Beauty 确认 30/30 epoch、6/6 validation、1/1 test；
- [x] Beauty 确认 test prediction 行数完整且 `n_pred_rows_missing_user_map=0`；
- [x] 将 Toys `metrics_cold_warm.json` 填入 §6.1–6.3；
- [x] 将 Beauty `metrics_cold_warm.json` 填入 §6.2–6.3；
- [x] 从日志填写 Toys validation 曲线；
- [x] 从日志填写 Beauty validation 曲线；
- [x] 计算 Toys 绝对差值与相对变化，填写 §6.3；
- [x] 计算 Beauty 绝对差值与相对变化，填写 §6.3；
- [x] 按冻结门槛完成 Toys §8.1，单域 Gate 判为 FAIL；
- [x] 按冻结门槛完成 Beauty §8.2 和双域 §8.3；
- [x] 检查并记录 Toys degraded holder 恢复状态；
- [x] 检查并记录 Beauty exact/degraded holder 恢复状态；
- [x] 将 Material Passport 的 Verification Status 更新为最终状态；
- [x] 确认报告中已无结果占位符，并将报告标记为完成。
