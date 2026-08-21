# GRAM 第十三阶段：Tier-0 诊断三连 —— 核心论断更正与瓶颈重定位

> **结论（2026-08-19）**：三个纯 CPU、evaluation-only 的诊断实验一致表明：
> (1) 原"简单无条件 portfolio 优于学习型 gating"的论断**必须撤回**，它源于未匹配 warm 代价的比较；
> (2) 在**匹配代价**后，学习型分配在 Toys/Beauty 双域、两种机制上**均显著优于随机**；
> (3) 真正的瓶颈是 **resolver 召回**（cold target 有 ~89% 根本不在候选池内），而非 slate 分配。
>
> 本报告不产生新的 efficacy Gate，不读取任何 test，不启动 GPU 训练。

## Material Passport

- Origin Date: 2026-08-19
- Verification Status: `EVALUATION_ONLY_DIAGNOSTIC`
- Experiment IDs: `TIER0_A_WARM_COST_MATCHED_RANDOM_BASELINE`,
  `TIER0_A2_LEARNED_VS_RANDOM_PRIORITISATION_SWEEP`,
  `TIER0_A3_CBSA_MATCHED_ACTION_MULTISET_PERMUTATION`,
  `TIER0_B_CANDIDATE_POOL_CEILING_DECOMPOSITION`
- Datasets: `Toys_cold50` / `Beauty_cold50`，validation only
- test_read: **false**（全部四个实验）
- GPU 使用: 无（全部纯 CPU，秒级）

---

## 0. 30 秒摘要

Section 3.5 曾据 Toys validation 的点估计得出结论：P1–P7 七轮学习型 gating 无一超过无条件 `portfolio@2/@3`，并据此把"简单打败复杂"列为论文主张 3（plan Section 3.5.7 第 3 条）。

本轮发现该比较存在**工作点未对齐**的混淆：`portfolio@2` 的 warm 保留为 95.91%，而 P6 为 99.56%。二者不在同一 warm 预算上，因此原比较无法区分"机制更好"与"花了更多 warm"。

匹配代价后结论**反转**：

| 检验 | 域 | 结果 |
|---|---|---|
| A：随机子集 @ P6 的 warm 代价 | Toys | P6 胜，**0/20 随机种子**能追平 |
| A2：学习效用排序 vs 随机，全覆盖率扫描 | Toys | 低覆盖区间（0.05–0.45）学习型**全部**超出随机 2sd |
| A3：CBSA 动作多重集固定，随机置换用户映射 | Toys | 学习型胜，**0/200 置换** ≥ 学习型，p=0.005 |
| A3：同上 | Beauty | 学习型胜，**0/200 置换** ≥ 学习型，p=0.005 |

同时 B 实验定位了真实瓶颈：cold target 在 resolver top-50 内的比例仅 **11.40%（Toys）/ 11.03%（Beauty）**，即约 **89% 的 cold 用户，正确答案根本不在候选池里**。

---

## 1. 实验 A：warm 代价匹配的随机基线

**问题**：把随机选中的一部分用户施加 `portfolio@2`，令其 warm 代价等于 P6 的 warm 代价，cold 表现能否追平 P6？

**设计**：随机子集的选择只依赖种子，与 target/is_cold/任何 outcome 无关；覆盖率在 warm 代价（约束侧）上校准，cold（结果侧）事后读出；20 个种子取均值并报告离散度。**该设计可证伪原论断，也可证伪本假设。**

**Toys 结果**（n=8,789；cold=4,367）：

| 方案 | cold H@50 | 事件 | cold H@10 | warm 保留 | overall N@10 |
|---|---:|---:|---:|---:|---:|
| v0 GRAM | 0.010305 | 45 | 0.005267 | 100% | 0.033361 |
| P6（学习型） | **0.018090** | 79 | 0.013968 | 99.56% | 0.034538 |
| 随机 @2，cov=0.10 | 0.012411 | 54 | 0.007419 | 99.59% | 0.033546 |
| 随机 @3，cov=0.05 | 0.011770 | 51 | 0.006767 | 99.72% | 0.033500 |
| portfolio@2 无条件 | 0.029769 | 130 | 0.025418 | **95.91%** | 0.035016 |
| portfolio@3 无条件 | 0.039615 | 173 | 0.035493 | **93.45%** | 0.035814 |

**判定**：在 ~99.6% 的同一 warm 保留上，P6 的 0.018090 显著高于随机的 0.012411（sd 0.000617），**20/20 随机种子全部落后**。

**含义**：原论断的比较对象 `portfolio@2/@3` 之所以 cold 更高，是因为它们分别多花了 3.65 / 6.11 个百分点的 warm。**主张 3 的原表述不成立。**

产物：`artifacts/phase13/explore/tier0_a_matched_random_toys/summary.json`

---

## 2. 实验 A2：学习效用排序 vs 随机，全覆盖率扫描

**问题**：A 只检验了 P6 自身的一个工作点。在**所有** warm 预算上，按学习效用给用户排序是否都优于随机选人？

**设计**：取 P6 每用户的 `predicted_utilities`（out-of-fold 产出，该用户未参与其模型拟合），**仅用作用户排序**，覆盖率从 0 扫到 100%。两臂使用完全相同的冻结插入规则，唯一差异是"谁被干预"。另报告一个按已实现增益排序的 oracle 臂，仅作不可部署上界。

**Toys / portfolio@2 结果**（节选）：

| 覆盖率 | warm 保留 | 学习型 cold H@50 | 随机 cold H@50 | 差值 | >2sd |
|---:|---:|---:|---:|---:|:--:|
| 0.05 | 100.00% | 0.012136 | 0.011198 | +0.000939 | YES |
| 0.10 | 99.90% | 0.014884 | 0.012159 | +0.002725 | YES |
| 0.20 | 99.48% | 0.018090 | 0.014461 | +0.003629 | YES |
| 0.25 | 99.16% | 0.019235 | 0.014827 | +0.004408 | YES |
| 0.30 | 99.06% | 0.020380 | 0.016029 | +0.004351 | YES |
| 0.45 | 98.42% | 0.021067 | 0.019292 | +0.001775 | YES |
| 0.75 | 97.06% | 0.024731 | 0.025029 | −0.000298 | – |
| 1.00 | 95.91% | 0.029769 | 0.029769 | 0.000000 | – |

学习型在 **9/20** 个覆盖点超出随机 2sd，且**全部集中在低覆盖（高 warm 保真）区间**——即实际有部署意义的区间。覆盖率趋近 100% 时两臂按定义收敛。

`portfolio@3` 上信号更强：**12/20** 个覆盖点超出 2sd，且优势延伸到 cov=0.55。

**oracle 观察（关键）**：Toys `portfolio@2` 的 oracle 上界为 **0.029998**，而无条件 `portfolio@2`（cov=1.0）为 **0.029769**。二者几乎相同 —— 说明**在当前候选池下，无论如何优化"选哪些用户"，cold H@50 的上限就在 0.03 附近**。这直接指向实验 B。

产物：`artifacts/phase13/explore/tier0_a2_prioritisation_toys{,_p3}/summary.json`

---

## 3. 实验 A3：CBSA 动作多重集置换检验（跨域 + 跨机制）

**问题**：A2 依赖 P6 的效用分数与 Toys。换一个学习机制、换一个域，结论是否复现？

**设计**：R²-v2 CBSA recovery 为 Toys/Beauty 每个用户输出 `effective_action ∈ {a0,a2,a3}`。**固定动作多重集**，仅随机置换"哪个用户拿哪个动作"，做 200 次置换检验。每种动作的数量逐一相同、插入规则相同，因此 **warm 代价按构造精确匹配**，无需校准。完整性校验：重算的 a2 结果与 CBSA 存档的 `portfolio2_hit50` 逐用户一致，不一致即中止。

| 域 | 动作分布 | 学习型 cold H@50 | 随机置换均值 (sd) | 置换 ≥ 学习型 | p |
|---|---|---:|---:|---:|---:|
| Toys | a0=2489 / a2=600 / a3=5700 | **0.036409**（159 事件） | 0.030653 (0.001083) | 0/200 | 0.005 |
| Beauty | a0=5156 / a2=1103 / a3=4396 | **0.029885**（158 事件） | 0.025903 (0.001062) | 0/200 | 0.005 |

warm 侧几乎不动（Toys 0.060522 vs 0.060723；Beauty 0.072653 vs 0.072358），确认代价确实匹配。overall N@10 学习型亦更高（Toys 0.035845 vs 0.035078；Beauty 0.040846 vs 0.039721）。

**判定**：**双域、双机制（P6 风险模型 + CBSA allocator）一致显示学习型分配显著优于随机。** 这是本轮最强的单项证据，因为代价匹配是构造性的而非校准出来的。

产物：`artifacts/phase13/explore/tier0_a3_cbsa_perm_{toys,beauty}/summary.json`

---

## 4. 实验 B：候选池天花板分解

**问题**：在设计第 9 个 allocator 之前，量化分配层最多还有多少空间。

**cold target 在各排序中的位置**：

| 位置 | Toys resolver | Toys GRAM v0 | Beauty resolver | Beauty GRAM v0 |
|---|---:|---:|---:|---:|
| rank 1 | 0.14% | 0.07% | 0.34% | 0.04% |
| rank 2–3 | 0.76% | 0.18% | 1.25% | 0.15% |
| rank 4–10 | 2.98% | 0.27% | 2.33% | 0.19% |
| rank 11–50 | 7.53% | 0.50% | 7.11% | 0.93% |
| **top-50 之外** | **88.60%** | **98.97%** | **88.97%** | **98.69%** |

**cold 可达性（top-50 内）**：

| 路径 | Toys | Beauty |
|---|---:|---:|
| resolver | 498 / 11.40% | 583 / 11.03% |
| GRAM v0 | 45 / 1.03% | 69 / 1.31% |
| 二者并集 | 522 / 11.95% | 616 / 11.65% |

**插入 N 个候选的理论天花板**（该用户 target 必须落在 eligible slice 前 N 位）：

| | Toys | Beauty |
|---|---:|---:|
| portfolio@1 | 1.19% | 1.44% |
| portfolio@2 | 2.11% | 2.48% |
| portfolio@3 | 3.11% | 3.14% |
| portfolio@5 | 4.65% | 4.22% |
| portfolio@10 | **7.17%** | **6.64%** |

**含义**：即便把 slate 分配做到完美、把插入位扩到 10 个，也只能触及约 7% 的 cold 用户。而 **89% 的 cold 用户的正确答案根本不在 resolver top-50 里** —— 对这些用户，任何 allocator 都无能为力。

**这解释了 P1–P7 与 CBSA 为何全部失败**：它们的可学监督信号极度稀疏（P3 记录 proposed candidate 正确率 base rate 仅 0.5916%，正例总数 52），因为可学的部分本来就只有 ~3%。

产物：`artifacts/phase13/explore/tier0_b_pool_ceiling_{toys,beauty}/summary.json`

---

## 5. 对既有结论的更正

| plan 位置 | 原表述 | 更正后 |
|---|---|---|
| 3.5.7 主张 3 | "简单 portfolio 即可兑现该 ceiling，而复杂 gating 反而更差" | **撤回**。原比较未匹配 warm 代价。匹配后学习型在双域双机制上均显著优于随机分配 |
| 3.5.4 决定性观察 | "P1–P7 七轮全部 FAIL 的唯一原因，是那个未经论证的 warm≥0.97 门槛挡掉了一个 overall 显著为正的方案" | 部分成立：门槛确实未经论证；但"portfolio@3 更好"是因为它买了更多 cold 而非机制更优。**两条修正互相独立** |
| memory `project_current_run.md` | 同上主张 3 | 需同步更正 |

**保持不变的结论**：
- GRAM 生成路径对零交互 item 结构性不可达（本轮 B 实验进一步佐证：cold target 在 GRAM top-50 内仅 1.03% / 1.31%）；
- exact resolver 存在强 cold ceiling 但 warm 不足，不能单独作推荐器；
- 碰撞审计结论（原 v1 增益来自 lexical-ID 别名）不受影响。

---

## 6. 方向重定位

**结论：停止设计第 9 个 slate allocator；瓶颈在 resolver 召回。**

依据：
1. 分配层天花板 ~7%（B 实验），resolver 召回天花板 11.4%，而当前 portfolio@2 实得 2.1%；
2. oracle 用户选择（0.029998）与无条件全覆盖（0.029769）几乎相同，说明用户选择维度已近饱和；
3. resolver 目前明显欠训练：`residual user projector`、**12 epochs**、in-batch 随机负样本、无 hard negative mining、温度与容量均未调过。

**resolver 召回每提升 1 个百分点，下游 portfolio 收益按比例放大，且不需要任何新的 gating 机制。**

建议的下一步（Tier 1，均不重训 GRAM，复用冻结 v0 预测，小 GPU 分钟到小时级）：

| 优先级 | 改动 | 理由 |
|---|---|---|
| 1 | epochs 12 → 50/100 | 最可能白捡；当前几乎确定欠训练 |
| 2 | hard negative mining（BGE 空间近邻） | in-batch 随机负样本对 12k item 检索过弱 |
| 3 | 温度 / projector 容量调参 | 从未调过 |
| 4 | 用户表示：均值 → attention 加权 / 多向量 | 长历史信息被均值抹平 |
| 5 | item 侧轻量微调 | BGE 当前完全冻结 |

主指标：**cold recall@50**（当前 Toys 11.40% / Beauty 11.03%）。

**纪律要求**：Tier 1 属于方法开发，只能在 Toys/Beauty 的 **train/validation** 上进行；两域 test 与 Sports 全部保持封存。任何 Tier 1 候选若要声称 efficacy，需另行预注册并使用未被查看的数据。

---

## 7. 产物清单

- `experiment/phase13/protocol/tier0_matched_cost_baseline.py`
- `experiment/phase13/protocol/tier0_prioritisation_sweep.py`
- `experiment/phase13/protocol/tier0_cbsa_permutation.py`
- `experiment/phase13/protocol/tier0_pool_ceiling.py`
- `artifacts/phase13/explore/tier0_a_matched_random_toys/summary.json`
- `artifacts/phase13/explore/tier0_a2_prioritisation_toys/summary.json`
- `artifacts/phase13/explore/tier0_a2_prioritisation_toys_p3/summary.json`
- `artifacts/phase13/explore/tier0_a3_cbsa_perm_{toys,beauty}/summary.json`
- `artifacts/phase13/explore/tier0_b_pool_ceiling_{toys,beauty}/summary.json`

## 8. 局限

1. 全部为单 split、单 resolver seed 的 validation-only 诊断，不构成 publication 级证据；
2. A2 使用 P6 的 out-of-fold 效用作排序，虽然每个用户的分数来自未见过该用户的模型，但这些效用是在同一 Toys validation 上产生的，跨数据泛化性未验证；A3 的置换检验不依赖此点，因此 A3 是更可靠的主证据；
3. B 实验的 11.4% 召回上限是针对**当前冻结 resolver**；Tier 1 的目的正是改变该数字，因此它是可移动的上界而非固有极限；
4. 本轮未做文献核对（调研 agent 因 API 余额中断），"学习型分配 vs 随机"的对照设计是否已有前人工作尚未确认。
