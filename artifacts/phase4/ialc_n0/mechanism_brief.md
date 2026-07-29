# IALC-N0 Mechanism Brief

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Created: 2026-07-28
- Verification Status: ANALYZED（本地代码、历史证据与检索到的论文原文）
- Data Scope: 本阶段未读取新的 Toys/Beauty validation、test 或 Sports 结果
- Direction ID: `IALC`
- Full Name: **Inference-Aligned Legal-Child Learning**
- Decision: **`IALC_N0_PASS_TO_PREMISE_AUDIT`**

## 0.1 N1 实际结果与终止决定

IALC-N1 已于 2026-07-28 按冻结配置完成，Toys/Beauty 各审计 512 个 unique
training-prefix users；完整性、mapping、Trie、finite、公式复算、0 update、checkpoint
SHA 与 validation/test/Sports exclusion 全部通过。

| 数据集 | competitive steps | mean illegal mass | mean loss gap | large-mass sample rate | tail/overall gap |
|---|---:|---:|---:|---:|---:|
| Toys | 1,811 | 2.26% | 0.0238 | 17.19% | 1.225 |
| Beauty | 1,494 | 4.53% | 0.0513 | 48.44% | 1.249 |

两域的 mean loss gap 均低于冻结的 0.10，large illegal-mass sample rate 也均低于
50%。这说明原 GRAM 即使在全词表训练，冻结模型已把绝大部分概率质量集中在合法
Trie children；train/inference support mismatch 太小，不足以支持新训练方向。

固定决定：**`STOP_IALC_NO_SUPPORT_MISMATCH`**。不进入 IALC-S0，不降低门槛，不把
Beauty 接近 50% 作为单域 rescue，也不追加 prior correction 或混合 loss。

## 1. 选择结论

下一主方向选择 IALC：把 GRAM 的训练目标从“全 T5 词表 token CE”改为“当前
Lexical-ID Trie 节点合法 children 上的条件似然”，使训练目标的支持集与 constrained
beam inference 完全一致。

这个方向优先于继续增加 reranker、catalog head 或 fusion weight，原因是：

1. 它直接修改原 generator 的学习问题，而非在冻结输出后修补；
2. 不需要外部 teacher、候选 quota 或额外推理分支；
3. 只依赖固定 catalog Trie 和 training target prefix，不读取 validation；
4. 若有效，推理结构、beam size 和在线开销均保持不变；
5. 实现改动集中在 loss，适合先做小成本 correctness smoke。

## 2. 研究问题与可证伪假设

### RQ

GRAM 训练时在整个 T5 vocabulary 上归一化、推理时却只在 Trie 合法 children 中
选择，这一 support mismatch 是否使有限训练预算被大量非法 token 消耗，并削弱合法
sibling 间的相对学习？

### H1：机制前提

在 Toys 和 Beauty 的 training-prefix 样本上，原 GRAM 在 gold prefix 处会把不可忽略
的概率质量分配给 Trie 非法 token；因此 full-vocabulary CE 与 inference-consistent
legal-child CE 存在稳定差距。

### H2：效果假设

在相同训练数据、初始化、optimizer steps 和 constrained beam 下，IALC 会强化合法
sibling 间的梯度竞争，从而提高 Recall@10/NDCG@10，且不需要增加推理开销。

H1 失败则不实现效果 pilot；H1 成立也不等于 H2 必然成立。

## 3. 方法定义

对 target lexical ID `y=(y1,...,yT)`，在第 `t` 步令：

- `p_t = (y1,...,y_{t-1})` 为 gold prefix；
- `C(p_t)` 为 catalog Trie 在该 prefix 下的全部合法 children；
- `z_t(c)` 为 decoder 对 token `c` 的 logit。

原始 GRAM 使用：

```text
L_full(t) = -log exp(z_t(y_t)) / sum_{v in V} exp(z_t(v))
```

IALC 使用：

```text
L_legal(t) = -log exp(z_t(y_t)) / sum_{c in C(p_t)} exp(z_t(c))
L_IALC = mean_t L_legal(t)
```

EOS 视为 Trie 的合法 child；pad 位置继续忽略。训练和推理使用同一个 catalog Trie
构造函数及同一 item-to-lexical-ID mapping。

### 冻结的最小实现

- 不改变 GRAM encoder、decoder、position embedding、prompt 或 lexical ID；
- 不增加 teacher、negative miner、reranker、catalog head 或新参数；
- 不改变 constrained beam search；
- 主方法不加入 popularity correction、tail weight、label smoothing 或 margin；
- 所有合法 child 在 local softmax 中同等对待。

先保持方法单一，是为了让效果能够归因于 train/inference support alignment。

## 4. 为什么可能提升

普通 full-vocabulary CE 对所有非法 token 都产生正梯度。可是在 constrained inference
中，这些 token 会被 Trie 永久屏蔽，模型真正需要解决的是同一 prefix 下合法 siblings
之间的排序。

令原模型在合法集合上的总概率质量为：

```text
S_legal(t) = sum_{c in C(p_t)} softmax_V(z_t)(c)
```

则两种单步损失满足：

```text
L_full(t) - L_legal(t) = -log S_legal(t)
```

因此 `S_legal` 越低，当前训练花在推理不可能事件上的目标差距越大。IALC 不保证
自动改善排序，但它会把非 gold 梯度集中到真实可竞争的 legal siblings，机制明确且
可直接审计。

## 5. 与既有方向和近邻工作的边界

### 与本项目失败方向

- 不同于 CHPR：不挑 hard negative，不依赖 collaborative proposal，不做 pairwise
  margin；每个 Trie 节点的所有合法 siblings 构成精确归一化集合。
- 不同于 GACR/CCRR：不是冻结模型后的 candidate reranking。
- 不同于 GCDH：没有 catalog head，正式排序仍来自 lexical generator。
- 不同于 RPCD/FCRD/PRPD：不融合 SASRec 分数或 popularity residual。
- 不同于 CAMI：不增加 identifier alias，也不做多路径 item aggregation。

### 与外部近邻工作

- GRAM 原论文明确使用 sequence-to-sequence full-vocabulary CE，推理才使用 Trie
  constrained beam；这构成当前 support mismatch。
- SEATER 用 balanced tree、InfoNCE 与 triplet loss学习 identifier hierarchy，并非
  在每个 gold prefix 的合法 children 上做精确条件似然。
- TrieRec 把 Trie 拓扑编码进 absolute/relative positional representations，并在论文
  中把 SEATER列为 trie-aware loss 近邻；其核心不是训练时 legal-child normalization。
- ReSID 在 tokenizer 阶段通过表示学习与量化降低 prefix-conditional uncertainty；
  IALC 固定 GRAM lexical IDs，只改变下游 generator 的 likelihood support。
- long-tail logit adjustment 是可借鉴的分类原理，但 IALC 主方法暂不做 prior
  adjustment，以免把结构对齐与 tail reweighting 混为一项贡献。

截至 2026-07-28 的定向检索，尚未发现把“训练时按当前 catalog Trie 的合法 children
精确归一化、推理保持同一 Trie”作为 lexical/Semantic-ID generative recommendation
主训练目标的工作。该结论是 **search-bounded novelty**，不是“全球首次”的证明。

## 6. 关键参考

1. Lee et al. GRAM（ACL 2025）：
   <https://aclanthology.org/2025.acl-long.1596/>
2. Si et al. SEATER（SIGIR-AP 2024）：
   <https://arxiv.org/abs/2309.13375>
3. Xu et al. TrieRec（arXiv 2026）：
   <https://arxiv.org/abs/2602.21677>
4. Liang et al. ReSID（arXiv 2026）：
   <https://arxiv.org/abs/2602.02338>
5. Menon et al. Long-tail learning via logit adjustment（ICLR 2021）：
   <https://arxiv.org/abs/2007.07314>

## 7. N1：Training-prefix-only premise audit

N1 使用冻结 GRAM checkpoint，不训练、不开 validation/test prediction，只在
training-prefix cohort 上记录每个 target step：

- Trie depth 与合法 child 数；
- `S_legal`、illegal probability mass `1-S_legal`；
- `L_full-L_legal=-log(S_legal)`；
- gold 在全词表和合法 children 中的 rank；
- head/tail 分组、用户与 item 覆盖；
- 各 depth 的样本数和分布。

### 完整性门

- Toys/Beauty 各 512 个 unique training users，head/tail 各 256；
- target、history、mapping 和 Trie membership 均为 100%；
- `sequence[-2:]` 不进入 cohort、统计或模型输入；
- finite rate、公式复算、同输入确定性均为 100%；
- optimizer steps=0，checkpoint SHA 前后相同；
- validation/test prediction 未打开，Sports 未读取。

### 科学门

两个数据集必须同时满足：

1. 至少 90% 的样本含有一个 `|C(p)| >= 2` 的可竞争 step；
2. 至少 2 个 depth 各有不少于 200 个可竞争 steps；
3. `mean[-log(S_legal)] >= 0.10`；
4. 至少 50% 的样本存在一个 step 的 illegal mass `>= 0.10`；
5. tail 样本的 mean loss gap 不低于 overall 的 80%。

固定决定：

- 全部通过：`IALC_S0_DESIGN_ALLOWED`；
- 科学门任一失败：`STOP_IALC_NO_SUPPORT_MISMATCH`；
- 完整性失败：`EXECUTION_INVALID`。

门槛冻结后不得因 Toys/Beauty 结果改成某个 depth、tail-only loss、prior correction
或混合 CE 来 rescue。

## 8. 最小后续实验链

```text
IALC-N1：training-prefix premise audit，0 update
  -> PASS
IALC-S0：loss correctness smoke（identity、finite、gradient、reload）
  -> PASS
IALC-P0：单一冻结配置的 Toys/Beauty validation pilot
  -> PASS
冻结确认与最终 test/Sports
```

P0 前另写完整预注册；当前 brief 不预支 validation 阈值，也不授权读取 validation。

## 9. 风险

- full-vocabulary CE 可能通过预训练词汇结构提供有益正则化，移除非法 token 后可能
  过拟合合法 siblings；
- 若 `S_legal` 已接近 1，实际 mismatch 太小，方法没有必要；
- legal-child loss 只改变训练梯度，冻结模型即时排序不会变化；
- 该方法的数学组件简单，论文贡献必须依赖清晰的 mismatch 证据、跨域稳定效果和
  与 TrieRec/SEATER 的充分差异实验，不能只靠命名包装。
