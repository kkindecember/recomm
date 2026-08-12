# GRAM 第十三阶段 v2_toys 实验报告
## LLM-Prior Regularization — Gate FAIL (关键负结果)

**实验版本**: v2_toys  
**数据集**: Toys_cold50  
**实验日期**: 2026-08-11 ~ 2026-08-12  
**Gate 判定**: ❌ **FAIL** (cold ndcg@10 相对 v1 **-48.0%**，远低于 +3% 阈值)  
**核心结论**: LLM prior 显著**恶化** cold-start 性能，是**重要 negative result**  
**⚠️ 更正**: 本报告 Section 4 的"token 空间不对齐"分析**不准确**，真实根因见诊断报告：`GRAM_第十三阶段_v2_toys_失败根因诊断报告.md`。真正原因是 **LLM 输出 OOV 时代码退化为 uniform 分布**（61% 样本受影响）。

---

## 一、实验目的与配置

### 1.1 实验目的
按 CANARD 探索计划 Section 2.v2，验证在 v1（MLP semantic bridge）基础上引入 LLM prior regularization 能否带来 cold-start 增量提升。

### 1.2 关键配置

| 配置项 | 值 |
|--------|---|
| 数据集 | Toys_cold50 (η=50%) |
| Seed | 12345 |
| GRAM epochs | 30 |
| MLP 训练 epochs | 200 |
| λ_llm | 0.5 (计划推荐起点) |
| LLM 模型 | DeepSeek-chat, 5-shot prompting |
| Prior 覆盖 | Warm+Cold 全部 items (11924 个) |
| Hierarchical ID | `hierarchy_v1_c32_l5_len32768_split_v2_mlpcold_llmprior` |

### 1.3 v1 → v2 关键改动
```
v1 loss: L_CE(MLP, ground_truth)
v2 loss: L_CE + 0.5 · KL(MLP || LLM_prior)
```

---

## 二、核心结果对比 (v0 / v1 / v2)

### 2.1 Cold-Start 用户 (n=4442) — **严重回退**

| 指标 | v0 | v1 | **v2** | v2 vs v1 | v2 vs v0 |
|------|-----|-----|--------|----------|----------|
| hit@1 | 0.00090 | 0.00473 | **0.00203** | **-57.1%** | +125.0% |
| hit@5 | 0.00360 | 0.00946 | **0.00540** | **-42.9%** | +50.0% |
| **hit@10** | 0.00608 | 0.01351 | **0.00765** | **-43.3%** | +25.9% |
| hit@20 | 0.00855 | 0.01869 | **0.01013** | **-45.8%** | +18.4% |
| **ndcg@10** | 0.00305 | 0.00872 | **0.00453** | **-48.0%** | +48.7% |
| ndcg@20 | 0.00370 | 0.01004 | **0.00516** | **-48.7%** | +39.5% |

**观察**: v2 保留了 v0→v1 收益的约 **50%**（相对 v0 仍是提升），但相对 v1 **系统性回退 40-50%**

### 2.2 Warm 用户 (n=4347) — 微弱提升

| 指标 | v0 | v1 | v2 | v2 vs v1 |
|------|-----|-----|-----|----------|
| hit@10 | 0.08949 | 0.08765 | 0.09271 | **+5.8%** |
| hit@20 | 0.12215 | 0.12169 | 0.12399 | +1.9% |
| ndcg@10 | 0.05404 | 0.05360 | 0.05504 | +2.7% |

**观察**: warm 用户轻微改善（+2-6%），但 cold 恶化远超此收益

### 2.3 Overall — 净退化

| 指标 | v1 | v2 | Δ |
|------|-----|-----|---|
| hit@10 | 0.05018 | 0.04972 | **-0.9%** |
| ndcg@10 | 0.03092 | 0.02951 | **-4.5%** |
| hit@50 | 0.10024 | 0.09330 | **-6.9%** |

---

## 三、Gate 判定

### CANARD 计划 v2 Gate 标准
- ✅ 通过: cold NDCG@10 相对 v1 提升 ≥ 3%
- ❌ 失败: 提升 <3% 或退化 → LLM prior 无用

### 实测结果
- v1 cold ndcg@10 = **0.00872**
- v2 cold ndcg@10 = **0.00453**
- **Δ = -48.0%** ❌ **FAIL (严重级别)**

---

## 四、失败原因分析

### 4.1 MLP 训练看似正常

从 `training_history.json` 观察：
- Epoch 1: train_loss_ce=7.08, train_loss_kl=3.17, val_avg_acc=6.5%
- Epoch 200: train_loss_ce=0.62, train_loss_kl=1.68, val_avg_acc=39.2%
- **MLP 收敛良好**（val_avg_acc 从 6.5% 提升到 39.2%）

**MLP 训练准确率对比**：
| 指标 | v1_toys | v2_toys | Δ |
|------|---------|---------|---|
| val_avg_acc (final) | 40.5% | 39.2% | **-1.3pp** |
| L1 acc | 86.9% | 86.4% | -0.5pp |
| L2 acc | 68.1% | 61.7% | **-6.4pp** |
| L3 acc | 25.5% | 25.0% | -0.5pp |
| L4 acc | 13.1% | 13.8% | +0.7pp |
| L5 acc | 9.1% | 9.1% | 0 |

**关键观察**: v2 的 MLP val_acc 甚至**略低于 v1**（L2 层明显退化 -6.4pp）。KL loss 引入的 LLM prior **反而干扰了 MLP 学习 ground truth**。

### 4.2 LLM Prior 质量问题

LLM 预测的 hierarchical tokens **与 GRAM 内部 clustering 语义不对齐**：

1. **GRAM hierarchical id 是 K-means 聚类结果**：token 是聚类中心索引，**没有语义含义**（例如 "▁cluster_47"）
2. **LLM 预测的是自然语言 token**：例如 "▁dolls", "▁wooden", "▁monster"
3. **两者根本不是同一个空间**！LLM prior 把 MLP 朝错误方向拉

从 `llm_priors_all.jsonl` 观察：
```json
{"item_id": "B001P9OGRS", "predicted_tokens": ["▁dolls", "▁dollhouse", "mini", "▁fold", "▁wooden"]}
```

这些是**语义 token**，而 GRAM 训练时的 hierarchical id 层次 token 是**聚类索引**：
```
c32_l5 意为: 32-way clustering × 5 levels = 完全数值化的 token
```

### 4.3 训练目标失配（根本原因）

```
v1 目标: text_embedding → cluster_id (通过 K-means 得到的标签)
v2 目标: text_embedding → cluster_id + KL(pred_dist || LLM_semantic_tokens)
                                       ↑
                                    错误方向
```

**根本问题**: KL divergence 在两个**不同 token 空间**之间没有意义。LLM 输出的语义 token 和 GRAM 的 cluster id token 在数值上可能重叠但**语义完全不同**。

### 4.4 warm 微弱提升的解释
v2 warm 稍好可能是因为 KL loss 起到了**正则化**作用（防止过拟合），但这种"正则化"对 cold 是灾难性的——cold items 没有历史训练信号，完全依赖 MLP 预测，MLP 被错误拉走，cold 就崩了。

---

## 五、v0 / v1 / v2 完整对比表

### Cold NDCG@10 演化路径
```
v0 (vanilla)     0.00305  ─── 基线
       ↓ +186%
v1 (MLP bridge)  0.00872  ─── 强信号确认
       ↓ -48%  ← 严重回退!
v2 (+ LLM prior) 0.00453  ─── 保留 v0→v1 收益的约 50%
```

### 累积增益 vs 计划预期

| 版本 | 计划累积 cold ndcg 增益 | 实测累积 cold ndcg 增益 | 达标? |
|------|------------------------|------------------------|-------|
| v0 → v1 | ≥ 5% | +186% | ✅ 超预期 |
| v1 → v2 | ≥ 3% | **-48%** | ❌ **严重不达标** |
| v0 → v2 (累积) | ≥ 8% | +49% | ✅ (但依赖 v1) |

---

## 六、Gate 决策与后续动作

### 6.1 按 CANARD 计划规则
按计划 Section 2.v2 iteration 选项：
1. ✅ 调 λ_llm（0.3, 0.5, 1.0）
2. ✅ 换 LLM（DeepSeek → GPT-4o mini）
3. ✅ 换 prompt 措辞

按计划 Section 1.4 iteration 上限：**每一步 vN 最多 3 次调整**

### 6.2 但更深层问题（诊断优先于 iteration）

上述 iteration 选项**都在治标**，根本问题是 **LLM 语义 token 空间 ≠ GRAM cluster id 空间**。即使换 LLM 或调 λ_llm，token space mismatch 依然存在。

**真正需要修的**：
- **方案 A**: 让 LLM prior 输出 cluster id 而非语义 token（需要给 LLM few-shot examples of cluster ids，让它学 cluster 分布）
- **方案 B**: 弃用 KL loss，改用**间接监督**（例如让 LLM 输出 semantic embedding，再和 warm items 的 embedding 做 InfoNCE）
- **方案 C**: 直接把 LLM 当作 **retrieval helper**（不做 loss，只用 LLM 的语义预测帮助选择 few-shot warm items）

### 6.3 推荐动作

**优先建议**: **不做 iter_2/iter_3 常规调优**，而是先**诊断 root cause 后重新设计 v2**。

理由：
1. 计划中 iteration 是给"调优"用的，不是给"设计错误"用的
2. -48% 不是"边缘失败"，是"方向错误"
3. v1_beauty 已跑通（+162%），v2_beauty 应用同样错误设计会浪费 20h

---

## 七、对 v2_beauty 的紧急影响

⚠️ **重要警告**: v2_beauty 已在后台运行（当前在生成 cold LLM priors，进度约 1%）。

### 决策选项

**选项 A: 立即停止 v2_beauty**（推荐）
- 理由: 相同设计缺陷，跑完只会得到相同 negative result
- 节约: 18-22 小时 GPU 时间 + ~$3-5 API 成本
- 命令:
  ```bash
  # 停止 prep 进程
  pkill -f "prep_v2_beauty"
  pkill -f "generate_llm_priors.*v2_beauty"
  # 停止自动启动守护进程
  pkill -f "auto_start_gram.sh"
  ```

**选项 B: 继续跑 v2_beauty**（不推荐）
- 理由: 得到 Beauty 域的 negative result 作为跨域证据
- 代价: 20+ 小时 GPU + 中断其他计划

**选项 C: 停止后立即启动重新设计的 v2**
- 需要先设计新的 v2 loss（Section 6.2 方案 A/B/C）
- 时间: 1-2 天设计 + 编码，然后重跑

---

## 八、下一步建议

### 短期行动（今天决定）

1. **决定 v2_beauty 命运**（选项 A/B/C）
2. **写 v2_toys report**（本文档）✅
3. **更新 project_current_run.md memory**
4. **更新 CANARD 计划的进度表** (Section 9)

### 中期决策（本周）

按 CANARD 计划 Section 1.4，v2 首次 iteration 失败可有 2 次调整。但鉴于本次是**方向性失败**，建议：

**方案 1: 跳过 v2，直接进入 v3**（Hierarchical Contrastive Alignment）
- 理由: v3 不依赖 LLM prior，是独立的 alignment loss
- v3 gate 参考基线换成 v1（不是 v2）
- 论文: v2 作为 negative result 写入 ablation

**方案 2: 重新设计 v2（v2_iter2 with 方案 C）**
- 用 LLM 做 few-shot example selector（不做 loss）
- 保持 MLP loss 不变，只是让 warm few-shot 更好
- 相对温和的改动

**方案 3: 加深理解，做小 pilot**
- 分析 LLM 输出的 top-k tokens 分布 vs GRAM cluster id 分布
- 用 32-item pilot 验证 token space alignment 假设
- 1-2 天可得结论

**推荐**: **方案 1 (跳到 v3) + 方案 3 (小 pilot 分析根因)** 并行

---

## 九、资源使用总结

| 项目 | 值 |
|------|-----|
| GPU | GPU 0 (A6000) |
| GPU lease | 20 GB |
| 训练总耗时 | ~16h (2026-08-11 20:34 → 2026-08-12 12:46) |
| LLM API 调用 | ~11924 次 (cold+warm, cached) |
| API 成本 | ~$3-4 |
| MLP 训练 | 200 epochs, ~2-3 min |

---

## 十、Negative Result 论文价值

本次 negative result 对最终论文有价值：

1. **Ablation 章节素材**: "We show that naive LLM prior integration harms cold-start performance (-48% cold ndcg@10 vs MLP-only baseline)"
2. **Discussion 素材**: 揭示 **semantic token space 与 clustering-based ID space 的根本不匹配**，是 LLM-augmented GenRec 的关键设计陷阱
3. **Method 章节的对比 baseline**: 展示我们最终方案（v3/v4/v5）如何避免这个 pitfall

**引用价值**: "Prior work has proposed LLM-based cold-start solutions for GenRec, but simple KL alignment fails when the target ID space is derived from unsupervised clustering rather than semantic tokens."

---

## 附录: 完整指标表

### A.1 Cold (n=4442)

| 指标 | v0 | v1 | v2 | v2 vs v1 (%) |
|------|-----|-----|-----|--------------|
| hit@1 | 0.00090 | 0.00473 | 0.00203 | -57.1 |
| hit@3 | 0.00248 | 0.00855 | 0.00473 | -44.7 |
| hit@5 | 0.00360 | 0.00946 | 0.00540 | -42.9 |
| hit@10 | 0.00608 | 0.01351 | 0.00765 | -43.3 |
| hit@20 | 0.00855 | 0.01869 | 0.01013 | -45.8 |
| hit@50 | 0.01171 | 0.02611 | 0.01306 | -50.0 |
| ndcg@10 | 0.00305 | 0.00872 | 0.00453 | -48.0 |
| ndcg@20 | 0.00370 | 0.01004 | 0.00516 | -48.7 |

### A.2 Warm (n=4347)

| 指标 | v0 | v1 | v2 | v2 vs v1 (%) |
|------|-----|-----|-----|--------------|
| hit@10 | 0.08949 | 0.08765 | 0.09271 | +5.8 |
| hit@20 | 0.12215 | 0.12169 | 0.12399 | +1.9 |
| hit@50 | 0.17805 | 0.17598 | 0.17529 | -0.4 |
| ndcg@10 | 0.05404 | 0.05360 | 0.05504 | +2.7 |

---

**报告生成时间**: 2026-08-12  
**Gate 判定**: ❌ FAIL (严重级别，方向性错误)  
**下一步**: 停止 v2_beauty + 决定跳到 v3 或 重新设计 v2