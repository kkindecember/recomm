# GRAM 第十三阶段：Tier-1 A/B 后处理层饱和验证（融合与 RTP 混合均 FAIL）

> **结论（2026-08-19）**：item 级分数融合与 recall-then-place 混合方案均未能超越 `portfolio@2`。结合 Tier-0 的两项饱和结论，**在冻结 GRAM + 冻结 resolver 的前提下，后处理层已到达上界**。本报告为负结果记录，不产生新的 efficacy Gate，未读取任何 test。

## Material Passport

- Origin Date: 2026-08-19
- Verification Status: `EVALUATION_ONLY_DIAGNOSTIC`
- Experiment IDs: `TIER1_A_ITEM_LEVEL_SCORE_FUSION_SWEEP`、`TIER1_B_HYBRID_RECALL_THEN_PLACE`
- Datasets: `Toys_cold50`（双实验）/ `Beauty_cold50`（仅 A）
- test_read: **false**；GPU 使用：无（纯 CPU）

---

## 1. 动机

Tier-0 已确立两条结论：候选池天花板极低（插入 10 个候选也只覆盖 7.17% 的 cold 用户），且用户选择维度已近饱和（oracle ≈ 无条件全覆盖）。

但仍有一个维度未被检验：**候选的排序方式**。P0 曾尝试 route 融合并失败，其根因分析写的是"depth3 route 接口被否定"，而非"融合无效"。**干净的 item 级融合从未被测试过**。本轮补上这个实验。

---

## 2. Tier-1 A：item 级分数融合

### 2.1 方法

对每个用户取 `v0_top50 ∪ resolver_top50`，按 rank 融合：

- RRF：`s(i) = w/(K+rank_gram) + (1-w)/(K+rank_res)`，K=60
- Borda：`s(i) = w·(N+1-rank_gram) + (1-w)·(N+1-rank_res)`

缺席某列表的 item 取该列表最差 rank（确定性、target-free 约定）。`w` 从 0 扫到 1，全网格报告，**不做拟合**。另设 cold-gated 变体（仅当 resolver 有 cold 候选时才融合）。

### 2.2 结果

**Toys**（n=8,789；cold=4,367），在匹配 `portfolio@2` 的 warm 保留（0.9591）时：

| 方案 | cold H@50 | 事件 | cold H@10 | warm 保留 | overall N@10 |
|---|---:|---:|---:|---:|---:|
| v0 GRAM | 0.010305 | 45 | 0.005267 | 1.0000 | 0.033361 |
| portfolio@2 | 0.029769 | 130 | 0.025418 | 0.9591 | **0.035016** |
| RRF w=0.75 | **0.066865** | **292** | 0.005267 | 0.9611 | 0.032117 |
| Borda w=0.85 | 0.044195 | 193 | 0.005267 | 0.9703 | 0.032465 |
| resolver-only | 0.114037 | 498 | 0.038699 | 0.3335 | 0.018584 |

**Beauty**（n=10,655；cold=5,287），匹配 warm 保留 0.9474：

| 方案 | cold H@50 | 事件 | warm 保留 | overall N@10 |
|---|---:|---:|---:|---:|
| v0 GRAM | 0.013051 | 69 | 1.0000 | 0.038936 |
| portfolio@2 | 0.032533 | 172 | 0.9474 | **0.040550** |
| RRF w=0.60 | **0.073955** | **391** | 0.9545 | 0.038093 |
| Borda w=0.70 | 0.061850 | 327 | 0.9526 | 0.037975 |

### 2.3 关键观察：表面成功是假象

融合的 cold H@50 达到 portfolio@2 的 **2.25×(Toys) / 2.27×(Beauty)**，但：

1. **cold H@10 完全没变**：Toys 始终 `0.005267`，与 v0 逐位相同。融合把正确的 cold item 从"top-50 之外"搬进了 **rank 11–50**，而 NDCG@10 根本不看这个区间；
2. **overall NDCG@10 反而低于 v0**（0.032117 vs 0.033361）。全局重排打乱了 warm item 的精确顺序；
3. **cold-gated 变体与非 gated 逐值相同** → 几乎所有用户的 resolver 都有 cold 候选，门控从未触发；
4. RRF 显著优于 Borda，因为 `1/(K+rank)` 对头部更陡峭。

**结论：融合提高的是"可达性"，不是"可用性"。** 它无法把答案送进决策窗口。

---

## 3. Tier-1 B：recall-then-place 混合（RTP）

### 3.1 方法

A 与 portfolio 的失败是互补的：融合擅长召回、不擅长放置；portfolio 擅长放置、召回受限于 resolver 自身 top-3。因此拆分两个决策：

1. **RECALL**：用 RRF（权重 `w`）在 `gram ∪ resolver` 全并集上重排 **eligible cold 候选**；
2. **PLACE**：保护 v0 前 `anchor` 位，把融合后的前 `n` 个 cold 候选插入 top-10 剩余槽位。

`w=0` 精确复现 `portfolio@n`（纯 resolver 顺序），因此 **portfolio 是 RTP 的严格特例，比较是嵌套的**。网格：`w∈{0,0.25,0.5,0.75,0.9,1.0}`、`anchor∈{7,8}`、`n∈{2,3}`。

### 3.2 结果（Toys，匹配 warm 保留 0.9591）

| w | anchor | n | cold H@50 | 事件 | cold H@10 | 事件 | overall N@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.00** | 8 | 2 | **0.029769** | **130** | 0.025418 | 111 | 0.035016 |
| 0.25 | 8 | 2 | 0.029540 | 129 | 0.025418 | 111 | 0.035016 |
| 0.50 | 8 | 2 | 0.029082 | 127 | 0.025189 | 110 | 0.034986 |
| 0.75 | 8 | 2 | 0.027479 | 120 | 0.026105 | 114 | 0.035122 |
| 0.90 | 8 | 2 | 0.027479 | 120 | 0.026563 | 116 | 0.035192 |
| 1.00 | 8 | 2 | 0.027021 | 118 | 0.026563 | 116 | 0.035190 |

**实现正确性验证**：`w=0, anchor=8, n=2` 与 `portfolio@2` 的三项 paired 差值均为 `+0.000000 [0,0]`，确认嵌套关系成立。

**paired bootstrap（10,000 次，seed=20260819）**，最优 overall 配置 `w=0.9` vs `portfolio@2`：

| 指标 | 差值 | 95% CI | 判定 |
|---|---:|---:|---|
| cold H@50 | −0.002290 | `[−0.003893, −0.000916]` | **FAIL（显著更差）** |
| cold H@10 | +0.001145 | `[−0.001145, +0.003435]` | INCONCLUSIVE |
| overall N@10 | +0.000175 | `[−0.000143, +0.000514]` | INCONCLUSIVE |

### 3.3 结论

**加入 GRAM 分数后 cold 单调下降，`w=0` 在整个网格中最优。** 即：在"挑选前 2–3 个 cold 候选"这件事上，**朴素的 resolver 顺序已是最优**，GRAM 分数对 cold item 不仅无帮助，还是噪声。

这与 Tier-0 B 完全一致：cold target 落在 GRAM top-50 内仅 1.03%(Toys)/1.31%(Beauty)。

---

## 4. 三层饱和：综合结论

| 维度 | 证据 | 结论 |
|---|---|---|
| 候选池 | Tier-0 B：cold 可达 11.40%/11.03%；插入天花板 @2=2.11%、@10=7.17% | 89% 的 cold 答案不在池内 |
| 用户选择 | Tier-0 A2：oracle `0.029998` ≈ 无条件全覆盖 `0.029769` | 已饱和 |
| 候选排序 | **Tier-1 A/B（本报告）**：融合、学习型 selector(P5) 均不敌朴素 resolver 顺序 | 已饱和 |

**在冻结 GRAM + 冻结 resolver 的前提下，`portfolio@2/@3` 就是最优解。**

这解释了为何 P1–P7 与 CBSA 共 8 轮机制迭代全部失败：不是机制不够聪明，而是**操作空间内已无可捞之物**（P3 记录可学正例仅 52 个，base rate 0.5916%）。

---

## 5. 对后续方向的约束

**禁止**：新增第 9 个 allocator、继续调融合权重/prefix/quota、恢复 P1–P7/CBSA/旧 v2–v5。

**唯一可行的两个方向**（详见 `plan/第十三阶段/GRAM_第十三阶段_Tier1_Resolver召回与路线B执行计划.md`）：

1. **提升 resolver 本身**：所有饱和结论都以当前**欠训练**的 resolver（12 epochs、in-batch 随机负样本、无 hard negative、温度/容量未调）为前提。召回上限 11.4% 是可移动的；
2. **修复 GRAM 生成路径**（路线 B）：唯一能突破 11% 上限的方向，但需 10–30h/域重训。

---

## 6. 局限

1. 单 split、单 resolver seed 的 validation-only 诊断，不构成 publication 级证据；
2. Tier-1 B 的网格为节省内存做过收缩（`w` 6 点、`anchor∈{7,8}`、`n∈{2,3}`）。原始更细网格（`w` 11 点、`anchor∈{6,7,8}`、`n∈{1,2,3,4}`）因 bootstrap 内存超时被终止。由于 `w` 方向上 cold 单调下降的趋势在 6 个点上已非常清晰，收缩不影响结论方向，但更细网格未被穷举；
3. 融合仅测试了 rank-based（RRF/Borda）方案，未测试 score-based 融合（需要原始 beam 分数与 cosine 分数的标定，P0 的 `r2_top50` 曾走这条路并失败）；
4. 未做文献核对（调研 agent 因 API 余额中断），"匹配代价的融合对照"设计是否已有前人工作尚未确认。

---

## 7. 产物

- `experiment/phase13/protocol/tier1_fusion_sweep.py`
- `experiment/phase13/protocol/tier1_recall_then_place.py`
- `artifacts/phase13/explore/tier1_a_fusion_{toys,beauty}/summary.json`
- `artifacts/phase13/explore/tier1_b_rtp_toys/summary.json`
