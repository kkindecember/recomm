# GRAM 第七阶段：GCGD-v1 P1 结果与验证报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-02
- Verification Status: ANALYZED
- Version Label: `phase7_gcgd_v1_p1_analysis_v1`
- Scientific Result: `REJECT_GCGD_V1_KEEP_GACR_V3`
- Governance Result: `RESOURCE_LEASE_OVERSHOOT`

## 1. 执行结论

`GRAM_PHASE7_GCGD_P1_V1` 已于 2026-08-02 14:38:50–15:14:59 完成，墙钟时间约
**36 分 09 秒**。Toys、Beauty 各评估 512 位 fresh development 用户，Sports/test 均未读取，
parent checkpoint SHA 前后一致，`alpha=0` 与 GRAM 完全一致，CodeLlama 已在 GPU0 恢复占位。

科学结论明确：**GCGD-v1 不保留，incumbent 继续使用 GACR-v3。** 固定图融合 B 与可学习门控 C
在两域的 NDCG@10、MRR 均低于原始 GRAM；相对 GACR-v3 的差距更大。图分支虽然改变了每一位
用户的生成结果，却没有产生任何一个“原 GRAM beam-50 外的新 top-10 命中”。这不是信号偏弱，
而是当前静态 LightGCN 图信号与下一商品生成目标错配，且门控没有学会可靠回退。

同时，本轮存在独立的工程合规问题：声明的总显存租约为 30,720 MiB，但遥测峰值达到 Toys
46,582 MiB、Beauty 46,026 MiB。结果文件可用于开发分析，但本轮不能标记为完全合规的正式 P1。

## 2. 主指标：相对同 cohort 原始 GRAM

括号中为相对 GRAM 的百分比变化；`pp` 表示绝对百分点变化。A=原始 GRAM，B=固定图融合，
C=B+adapter/gate，V3=冻结 GACR-v3。

### 2.1 Toys（n=512）

| 方法 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | Recall@50 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| A / GRAM | 0.089844 | 0.068298 | 0.109375 | 0.074761 | 0.191406 | 0.067602 |
| B | 0.087891 (-2.174%) | 0.065011 (-4.813%) | 0.107422 (-1.786%) | 0.071464 (-4.410%) | 0.183594 (-4.082%) | 0.063814 (-5.604%) |
| C | 0.087891 (-2.174%) | 0.063934 (-6.390%) | 0.109375 (+0.000%) | 0.070952 (-5.095%) | 0.183594 (-4.082%) | 0.062441 (-7.635%) |
| GACR-v3 | 0.089844 (+0.000%) | 0.068433 (+0.198%) | 0.111328 (+1.786%) | 0.075435 (+0.902%) | 0.189453 (-1.020%) | 0.068165 (+0.832%) |

- B 的 NDCG@10 绝对下降 0.3297pp，预注册 paired bootstrap 95% CI 为
  **[-8.986%, -0.293%]**；C 绝对下降 0.3809pp，CI 为 **[-10.146%, -0.219%]**。
- B/C 的 Recall@50 都下降 0.7812pp；V3 的 NDCG@10 则提升 0.0674pp。

### 2.2 Beauty（n=512）

| 方法 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | Recall@50 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| A / GRAM | 0.060547 | 0.046375 | 0.083984 | 0.054079 | 0.173828 | 0.049690 |
| B | 0.058594 (-3.226%) | 0.042713 (-7.897%) | 0.080078 (-4.651%) | 0.049542 (-8.390%) | 0.171875 (-1.124%) | 0.045094 (-9.249%) |
| C | 0.054688 (-9.677%) | 0.039628 (-14.548%) | 0.078125 (-6.977%) | 0.047164 (-12.786%) | 0.173828 (+0.000%) | 0.042792 (-13.882%) |
| GACR-v3 | 0.062500 (+3.226%) | 0.047216 (+1.814%) | 0.089844 (+6.977%) | 0.056151 (+3.831%) | 0.175781 (+1.124%) | 0.051027 (+2.690%) |

- B 的 NDCG@10 绝对下降 0.4537pp，95% CI 为 **[-14.641%, -3.384%]**；C 绝对下降
  0.6915pp，CI 为 **[-19.665%, -6.884%]**。
- Beauty 的结论最强：B/C 的主要 top-10 指标全部下降，而 V3 的六项指标全部不低于 GRAM。

## 3. 相对 GACR-v3 的直接比较

下表的 CI 是在结果完成后基于同一逐用户 CSV 做的 10,000 次 paired bootstrap，属于
**post-hoc exploratory analysis**，不能包装成预注册确认性检验。

| 数据域 | 比较 | NDCG@10 相对差 | 95% CI | Recall@10 绝对差 | 95% CI |
|---|---|---:|---:|---:|---:|
| Toys | B - V3 | -5.264% | [-10.543%, -0.294%] | -0.3906pp | [-1.1719pp, +0.3906pp] |
| Toys | C - V3 | -5.943% | [-11.649%, -0.378%] | -0.1953pp | [-0.9766pp, +0.5859pp] |
| Beauty | B - V3 | -11.770% | [-18.849%, -5.868%] | -0.9766pp | [-1.9531pp, -0.1953pp] |
| Beauty | C - V3 | -16.004% | [-23.975%, -9.413%] | -1.1719pp | [-2.1484pp, -0.3906pp] |

因此“只要比 v3 有提升即可继续”的开发标准在本轮没有满足；B/C 不是接近 v3，而是在两域都
明显落后。C 相对 B 在 Toys 的差异不确定，在 Beauty 又额外下降 4.799% NDCG@10
（95% CI [-9.208%, -1.436%]）。

## 4. 分组与逐用户机制

| 数据域 | 方法 | head NDCG@10 vs A | tail NDCG@10 vs A | NDCG@10 改善/下降用户 | changed | 新 top-10 命中 |
|---|---|---:|---:|---:|---:|---:|
| Toys | B | -6.010% | -3.315% | 5 / 16 | 512/512 | 0 |
| Toys | C | -4.683% | -5.377% | 6 / 17 | 512/512 | 0 |
| Toys | V3 | +4.974% | -1.878% | 11 / 6 | 63/512 | 0 |
| Beauty | B | -11.352% | -4.675% | 0 / 16 | 512/512 | 0 |
| Beauty | C | -16.375% | -8.286% | 0 / 22 | 512/512 | 0 |
| Beauty | V3 | +2.984% | +4.892% | 8 / 1 | 53/512 | 0 |

关键诊断是“**覆盖过宽、收益为零**”：B/C 对全部用户强制注入图分数，但没有一例 target 从 A 的
beam-50 外进入 top-10；Toys 的真实 graph beam@50 目标命中 B 为 0 增 4 减、C 为 1 增 5 减，
Beauty 的 B 为 0 增 1 减、C 为 1 增 1 减。当前方法主要是在错误地扰动已有候选排序，并未扩大
有效候选覆盖。

原 summary 中 GACR-v3 的 `target_in_candidate_beam50` 实际表示 GRAM+catalog union 中存在
candidate rank，rank 可能大于 50，不能解释成真正 beam-50 coverage；报告仅把 B/C 的该字段用于
beam 机制判断。GACR-v3 的标准 Recall@50 指标不受此命名问题影响。

## 5. 训练诊断

- LightGCN raw BPR+L2：Toys 0.684303 → 0.677604，Beauty 0.685393 → 0.678206，仅下降约
  0.98%/1.05%，仍接近随机 pairwise 区分的 `ln(2)` 区域；图模型学到的目标区分力有限。
- C 的 calibration mean gate 为 Toys 0.5826、Beauty 0.6669；gate accuracy 为 0.5938、
  0.7578。即使图证据不可靠，门仍较开放，Beauty 尤其明显。
- adapter 的 step 1 与最后一步来自循环中的不同 mini-batch，不能仅凭 Toys loss
  0.605→1.032 或 Beauty 0.710→0.802 断言训练发散。
- `graph_covered=100%`，因此没有 uncovered 子组可解释整体下降。

## 6. 假设判定

| 假设 | 判定 | 证据 |
|---|---|---|
| H1：扩大有效 beam 覆盖 | 失败 | Recall@50 不增，且无新增 top-10 命中 |
| H2：两域 top-10 指标提高 | 失败 | B/C 两域 NDCG@10 均下降 |
| H3：出现有意义的正向开发信号 | 失败 | 相对 GRAM、GACR-v3 均无可保留信号 |
| H4：head/tail 安全 | 失败 | B/C 的 head、tail NDCG@10 全部下降 |
| H5：超越 residual 的候选生成能力 | 失败 | `new_hit_at10_outside_A_beam=0` |

## 7. 完整性与资源审计

- 每域 512 用户，无 attrition；每域 CSV 1,536 行，B/C/V3 各 512 行，sample key 唯一。
- Toys cohort SHA256：`971e85e01e65364b57b0e9f71b29dfc1a68d391fa3f1c79b6d4cacf27460bf85`；
  Beauty：`70642760ff95ed0294c24c7e8ef75b52123cd0680e648551c729b832fd63fad0`；历史 cohort overlap=0。
- parent checkpoint 前后 SHA 一致；`alpha=0` identity exact；未发现 NaN/Infinity；test/Sports=false。
- 日志中的两处 `mean of empty slice` 来自 identity processor 的空诊断集合，不影响指标，但下轮需修。
- Toys workload 预声明 5,120 MiB、sidecar 25,600 MiB，实测总峰值 46,582 MiB；Beauty
  预声明 2,048 MiB、sidecar 28,672 MiB，实测总峰值 46,026 MiB。根因是 smoke 未覆盖正式运行
  长生命周期的 CUDA reserved/cache 峰值，造成 workload budget 严重低估。

资源治理据此标记为 **`RESOURCE_LEASE_OVERSHOOT`**。指标计算本身没有因此失真，但下一个正式
实验必须先通过全路径峰值测量，并让“workload 实际峰值 + sidecar = 30,720 MiB”，不得复用本轮
5,120/2,048 MiB 的预算。

## 8. Statistical Interpretation 与 Fallacy Scan

Overall Confidence：**CAUTION**。对 B/C 的负向结论具有两域一致性，且多个 CI 排除 0；但这是
单 seed、开发 cohort、多方法多指标探索，不能外推为所有数据域的普遍结论，也不能声称线上因果效果。

覆盖：**11/11 checked**。

| 谬误 | 严重度 | 检查结果 |
|---|---|---|
| Simpson's paradox | NOTE | B/C 在两域方向一致；head/tail 分开报告，无域聚合反转。 |
| Ecological fallacy | NOTE | 使用逐用户 paired 数据，并报告改善/下降用户数。 |
| Berkson's paradox | CAUTION | fresh cohort 排除了历史开发用户，结论限于该筛选后的总体。 |
| Collider bias | NOTE | 未使用处理后的变量作控制或筛选。 |
| Base-rate neglect | NOTE | 同时报绝对值、pp 与相对百分比，避免小基数夸大。 |
| Regression to mean | NOTE | cohort 未按历史极端表现选择。 |
| Survivorship bias | NOTE | 两域均为 512/512，无评估用户丢失。 |
| Look-elsewhere effect | CAUTION | 多 arms、指标和分组未做 multiplicity correction，CI 仅作开发证据。 |
| Garden of forking paths | CAUTION | 原配置预冻结；直接 v3 CI 明确标为 post-hoc，旧 cohort 不再用于调参后确认。 |
| Correlation != causation | NOTE | 仅解释同 cohort 离线算法干预，不外推线上用户行为。 |
| Reverse causality | NOTE | 本实验不作观察性反向因果推断。 |

## 9. 决定与下一步

停止 GCGD-v1 的 B/C 配置，不进入 v1 的三 seed P2，也不在当前 P1 cohort 上扫描 alpha 或 gate。
下一版使用新的 **ST-GCGD-v2（时序多关系图条件解码）**：用 item-transition 与 recency-aware
user-item 关系替代静态二部 LightGCN，并把 gate 改成 train-only 的“相对 GRAM 是否改善”的
优势门控。先修复资源租约与统计字段，再在全新 development cohort 上比较 A、GACR-v3、
静态 v1-B、v2 固定融合和 v2 优势门控。

详细设计见 `plan/GRAM_第七阶段_ST-GCGD-v2时序多关系图条件解码实验计划.md`。

## 10. 关键产物 SHA-256

- combined summary：`aff283bba4e655ab6dd746f37ef78cc75fa21ba6610afa00cc189bd6b171dfb9`
- Toys summary：`d9ff4225bc803d9139f79608d8d337758d5e3fba7b50d9480016dbe72a5829c4`
- Beauty summary：`5333f526525a8da5e5289eb5341ff56e35388894fc3a89c1a1d7fc46b7140d09`
- Toys per-user CSV：`6998fb6d63f14018f9fcc88259fdcdea6afbec60a379e87097f4e2a6adcd725f`
- Beauty per-user CSV：`68a851f1bcd72eb65c684ae4243c8a9d906ff4cec7cafa1d786da1d9a1e758b9`
