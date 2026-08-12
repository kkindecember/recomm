# v2_toys 失败根因诊断报告

**诊断日期**: 2026-08-12  
**目的**: 定位 v2 cold ndcg@10 相对 v1 回退 -48% 的根本原因

---

## 一、错误假设修正

### 我最初的推测（错误）
> "LLM 输出语义 token, GRAM 使用 K-means cluster id (数值 token), 两者空间根本不对齐"

### 实际情况
GRAM 的 hierarchical ID **也是自然语言 token**（不是数值索引）。示例：
```
B0000A1Z5K |▁animals|stuffed|▁se|▁cat|hat
B001RNHE6W |▁game|monopol|▁dealing|▁developer|▁city
```

LLM 预测也是同类 token：
```
B001P9OGRS  →  ▁dolls, ▁dollhouse, mini, ▁fold, ▁wooden
```

**Token 空间是"部分对齐"的**（L1 有 96.7% 的 GRAM tokens 都在 LLM 输出集合里），最初假设错误。

---

## 二、真正的失败原因

### 2.1 GRAM vocab 是"精选闭集"，LLM 是"开放语言"

| 层 | GRAM vocab 大小 | LLM 用了 token 数 | LLM Overlap 比例 |
|----|-----------------|-------------------|-------------------|
| L1 | 30 | 1201 | 96.7% GRAM ⊂ LLM |
| L2 | 670 | 2333 | 53.4% |
| L3 | 4568 | 3545 | 28.5% |
| L4 | 5533 | 3780 | 25.8% |
| L5 | 5777 | 3859 | 24.6% |

- GRAM 的每层 vocab 是**从数据聚类出来的精选 token**（例如 L1 只有 30 个代表性词）
- LLM 生成时**不知道这些约束**，会输出大量 GRAM 词表**外**的合法英文词

**举例**（都是 L1 的合法 LLM 输出，但 GRAM L1 词表里没有）：
- `▁princess`, `▁batcave`, `▁novelty`, `▁captain`, `▁view`

### 2.2 代码在遇到 OOV 时的处理是灾难性的

`semantic_bridge_v2.py` 的 `load_llm_priors` 函数：
```python
if token in vocab.per_level_token_to_idx[level_idx]:
    dist[token_idx] = 1.0        # 有效: 该 token 概率=1
else:
    dist = [1.0 / vocab_size] * vocab_size   # OOV: 全均匀分布!
```

### 2.3 OOV 比例极高

| 层 | LLM 预测 OOV 比例 |
|----|-------------------|
| L1 | **61.0%** |
| L2 | 61.7% |
| L3 | 53.9% |
| L4 | 50.2% |
| L5 | 51.3% |

**超过一半的样本，LLM prior 变成了 uniform 分布（熵最大的目标）**

### 2.4 灾难性后果

KL(MLP || LLM_prior) 中，如果 LLM_prior = uniform：
- 该 loss 项在**推 MLP 输出朝均匀分布走**
- 相当于**破坏 MLP 的判别能力**（把决定性预测拉平）
- 影响所有 items（不只 cold）

**为什么 cold 崩得比 warm 惨？**
- Warm items 有强 supervised signal (L_CE)，能抵抗 KL 的破坏
- Cold items 完全依赖 MLP 学到的表示能力，MLP 一被弄糊涂，cold 就崩

---

## 三、数据验证（LLM prior 本身有能力吗？）

在 warm items 上，LLM 预测每层完全匹配 GRAM 真值的准确率：

| 层 | 匹配率 |
|----|--------|
| L1 | **26.9%** |
| L2 | 12.1% |
| L3 | 9.1% |
| L4 | 5.7% |
| L5 | 2.9% |

对比：
- 随机猜测 L1 命中率 ≈ 1/30 = 3.3%
- LLM 达到 26.9%，**明显有语义能力**（8× 随机基线）
- 但因为剩下 73% 是 OOV 或其他 vocab token，都会触发 uniform → 破坏 MLP

**结论**: LLM 本身有能力，但**用 KL loss 强对齐**这个设计有致命缺陷。

---

## 四、根本设计缺陷分析

### 缺陷 1: **约束不匹配**
- GRAM vocab = 从数据聚类出的**精选闭集**
- LLM = 生成**开放语言**
- 强制 KL 对齐 = 用开放输出监督闭集分类器 → 语义空间冲突

### 缺陷 2: **OOV 退化策略最差**
- 遇到 OOV 时用 uniform 分布是数学正确但**语义错误**
- 应该 **跳过（不计 loss）** 或 **只监督 top-k in-vocab tokens**

### 缺陷 3: **一次性 one-hot 硬对齐**
- 代码把 LLM 预测转成 one-hot (dist=1.0)
- 完全丢失 LLM 的 top-k 语义信息
- 只用 top-1 就损失了 LLM 的分布信息

---

## 五、可行的修正方向

### 方向 A: **修 OOV 处理**（最小改动）
- OOV 时**不计算 KL loss**（skip 该样本 or 该层）
- 影响面：从 61% OOV → 只在 39% in-vocab 上做 KL

**预期**: 可能挽回，但只解决了 OOV 问题，没解决"闭集 vs 开放"根本冲突

### 方向 B: **软化 KL target**（中等改动）
- LLM 预测 top-5 tokens，都在 vocab 内的按 confidence 加权
- 不再是 one-hot，而是 top-k 软分布
- OOV 也跳过

**预期**: 更好，因为保留了 LLM 分布信息

### 方向 C: **LLM 只做 few-shot selector**（重设计）
- LLM 不做 loss
- LLM 用来选**语义最相似的 warm items** 作为 cold item 的"参考样例"
- MLP 训练时，cold item 的 loss 用其参考样例的 GRAM ID 做辅助目标

**预期**: 最稳，但需重写代码

### 方向 D: **让 LLM 输出被限制在 GRAM vocab 内**（Prompt 改造）
- 修改 prompt，把 GRAM 每层的合法 token 列表**给 LLM 看**
- LLM 只能从合法 token 里选
- OOV 从 61% 降到 0%

**预期**: 保留 KL loss 设计，只修 LLM 输出

---

## 六、推荐方案

**选 D + A 组合**（渐进式修复）：

1. **Prompt 改造**（方向 D）：
   - 给 LLM 提供每层的 vocab（L1: 30 个，L2: 670 个，... 太大的层可能只给 top-100 常见）
   - 要求 LLM **必须**从该 vocab 中选
   - 目标: OOV rate < 5%

2. **代码修复**（方向 A）：
   - 万一还是 OOV，**跳过该层的 KL loss**（mask 掉），不再用 uniform
   - `mask[level] = 0 if OOV else 1`
   - `L_kl = sum(mask * kl_per_level) / sum(mask)`

3. **保留 v1 的强 signal**：λ_llm 从 0.5 → 0.2（削弱 KL 影响，主要让 L_CE 起作用）

**预期效果**:
- OOV 问题被彻底解决（方向 D）
- 即使有 OOV 也不会破坏（方向 A）
- LLM prior 作为"锦上添花"而非"喧宾夺主"（λ_llm 降低）

---

## 七、代码修改点清单

### 需要改的文件
1. `experiment/phase13/protocol/generate_llm_priors.py`
   - Prompt 里加入 vocab constraint
2. `experiment/phase13/protocol/semantic_bridge_v2.py`
   - `load_llm_priors`: OOV 返回 `None` 而不是 uniform
   - `train_cmd`: KL loss 计算时 mask 掉 OOV 层

### 不用改的部分
- MLP 架构（v1 复用）
- GRAM 训练代码
- Assign cold IDs 流程

---

## 八、行动计划

### 立即（本次会话）
1. ✅ 诊断根因（本报告）
2. 🔄 更新 CANARD plan，标记 v2_iter1 FAIL，加入 v2_iter2 设计
3. 🔄 实施新 v2 代码（方向 D + A）
4. 🔄 启动 v2_iter2_toys 和 v2_iter2_beauty 实验

### 之后
- 等 v2_iter2 结果（预计 20h）
- 如果 pass → 继续 v3
- 如果还 fail → 用 v2_iter3 方向 C（LLM as retriever）

---

**报告完成时间**: 2026-08-12  
**核心结论**: v2 失败不是 token 空间问题，而是 **OOV 处理策略 + KL loss 硬对齐**的双重设计缺陷。修复方向明确：给 LLM 加 vocab 约束 + OOV 时 mask 而非 uniform。
