# v3 hypothesis — Hierarchical Contrastive Alignment Loss(层深加权版)

**创建日期**: 2026-08-14
**前置**: v1 双域强 PASS;v2(LLM prior as KL regularizer)两次 iteration 均 FAIL,组件已 abandoned
**基线**: **v1**(不是 v2)。v2 已 abandoned,gate 应为"cold NDCG@10 相对 v1 提升 ≥3%,warm 退化 ≤3%"

---

## 1. 这次要改什么

按探索计划 §2 v3,在 **v1** 基础上加 hierarchical contrastive alignment loss:

```
v1 : L = L_CE(MLP, GRAM_ground_truth)
v3 : L = L_CE + Σ_l λ_l · L_align_l
```

每层 l 采样 triplet(anchor / positive=同层同 cluster 的 warm item / negative=同 l-1 层但不同 l 层的 hard negative),InfoNCE,每层独立 temperature τ_l。

**与计划原文的唯一偏离**:计划写的是"在 v2 基础上加",但 v2 已 abandoned,所以**去掉 L_llm_prior 项,直接在 v1 上加 alignment**。

---

## 2. 核心假设(v2 失败换来的关键洞察)

**LLM/语义信号的可靠性随层深单调衰减 —— 因此按层均匀加权是错的。**

v2_iter2 实测:warm item 上 LLM 预测与 GRAM 真值的一致率(仅统计词表内的真实回答)

| 层 | Beauty 一致率 | Toys 一致率 | vs random |
|---|---|---|---|
| L1 | 44.5% | 60.4% | 18-48x |
| L2 | 22.7% | 27.5% | 184-867x |
| L3 | 10.1% | 16.5% | 496-753x |
| L4 | 5.8% | 8.2% | 292-455x |
| L5 | 5.4% | 6.4% | 273-370x |
| L6 | 3.5% | — | 175x |
| L7 | 4.2% | — | 200x |

**解读**:GRAM 的 hierarchical id 来自 SASRec **协同过滤空间**的聚类;语义信号来自**文本/类目空间**。两者在**浅层(粗类目)显著重合**,在**深层(细粒度 cluster)几乎无关** —— 深层的簇划分由共现模式决定,不是语义决定的。

v2 的失败正是因为把语义信号**均匀地**施加到所有层:在 L3+ 的 85-96% 样本上,KL 把 MLP 往错误 cluster 拉。佐证:MLP val_acc 随 KL 项单调下降(Toys v1=0.4060 → λ=0.5 时 0.3930 → λ=0.2 时 0.3846),且降幅集中在 L1/L2;**λ 从 0.5 降到 0.2 时 val_acc 继续下降而非回升**,说明这不是权重大小问题,是**施加位置**问题。

---

## 3. 预期效果与具体设计

**H1(主假设)**:alignment loss 按层加权(浅层高、深层低或为 0)后,cold NDCG@10 相对 v1 提升 ≥3%,warm 退化 ≤3%。

**H2(机制假设)**:如果 H1 成立,那么**均匀加权版本应当显著劣于加权版本** —— 这条必须验证,否则无法区分"alignment 有用"和"按层加权有用"。

### 建议的 λ_l 设置

以一致率为先验,起点建议(Toys 5 层 / Beauty 7 层):

```
λ_l ∝ agreement_l  →  归一化后大致:
  Toys  : λ = [1.0, 0.45, 0.27, 0.14, 0.11]
  Beauty: λ = [1.0, 0.51, 0.23, 0.13, 0.12, 0.08, 0.09]
```

**或者更激进的降级版**:只在 L1/L2 加 alignment,L3+ 全部 λ_l=0。这个版本实现最简单,且直接检验"深层语义无用"这一论断 —— **建议作为 iter1 先跑**,因为它同时是最省算力和信息量最大的配置。

### 必做的对照

| 配置 | 目的 |
|---|---|
| v1(已有) | 基线 |
| v3-a: 只在 L1/L2 加 alignment | 主候选,验证 H1 |
| v3-b: 均匀 λ_l 全层加 | 验证 H2,若 v3-b ≈ v1 或更差,则"按层加权"是真正的贡献点 |

v3-b 不需要单独跑 GRAM —— **先比 MLP val_acc**(见 §4),差异明显再决定是否上 GRAM。

---

## 4. 如何验证(先便宜后昂贵)

**Stage 1 — MLP val_acc 快筛(几分钟,不占 GPU 训练档期)**
MLP 训练只需 200 epoch × 0.2-0.4 s/epoch ≈ 1 分钟。先看 v3-a / v3-b 的 `val_avg_acc` 与 **per-level acc** 相对 v1 的变化:

- 参照基线:v1_beauty best=**0.2630**、v1_toys best=**0.4060**
- **门槛:如果 val_avg_acc 低于 v1,不要上 GRAM。** v2 两次失败在这一步就已经能看出来(0.3930 / 0.3846 都低于 0.4060),但当时没设这道闸,白烧了两次 25 小时的 GRAM 训练。
- 特别看 **per-level acc**:期望 L1/L2 提升,L3+ 至少不退化

**Stage 2 — 只有 Stage 1 通过才跑 GRAM**,单域先跑(建议 Toys,训练 25h vs Beauty 15h 但 test 推理快得多:0.39 s/样本 vs 3.36 s/样本),通过再跑第二域确认双域一致。

---

## 5. 风险与预案

| 风险 | 处置 |
|---|---|
| triplet mining pipeline 复杂,容易引入 bug | 先写单测:验证 positive 同 cluster、hard negative 同 l-1 层不同 l 层 |
| λ_l / τ_l 需要联调,搜索空间大 | 先固定 τ_l=0.07 只调 λ_l;用 Stage 1 快筛,一次几分钟 |
| Stage 1 通过但 GRAM 上不涨 | 说明 MLP val_acc 与 end-to-end 指标脱钩 —— 这本身是重要发现,写进 report |
| alignment 也在深层失效 | 降级为 flat alignment(计划 §2 iteration 选项 3),或直接进 v5 |

---

## 6. 启动前必须确认

- [x] v2 组件已 abandoned,report 与进度表已更新(2026-08-14)
- [x] `generate_llm_priors_v2iter2.py` API 失败不再伪装成 `<unk>`+confidence 1.0
- [x] protector 恢复改为按实时 free 自适应;`holding_post_training` 加 6h 超时
- [x] **v2_verify 结论**(2026-08-14):完整 KL 覆盖下重训 MLP,Beauty 0.2505 / Toys 0.3889,**均仍低于 v1 的 0.2630 / 0.4060**,且 Beauty 补齐后反而更差 —— 误判假设已排除,v2 维持 abandoned。见 `artifacts/phase13/explore/v2_verify/CONCLUSION.md`
- [ ] triplet mining 单测通过
- [ ] Stage 1 快筛闸门写进 runner 或 checklist

---

## 7. 关联

- Report: `report/第十三阶段/GRAM_第十三阶段_v2_iter2_vocab-constrained-LLM-prior_双域gate-FAIL报告.md`
- 计划: `plan/第十三阶段/GRAM_第十三阶段_CANARD探索计划v0.1.md` §2 v3
- 一致率与 val_acc 的原始计算见上述 report §5.3
