# LNDR-N0 Mechanism Brief

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Created: 2026-07-28
- Verification Status: ANALYZED（本地 catalog 结构与检索到的论文原文）
- Data Scope: catalog + training-prefix only；未读取新的 validation/test/Sports
- Direction ID: `LNDR`
- Full Name: **Lexical–Node Dual Readout**
- Decision: **`LNDR_N0_PASS_TO_PREMISE_AUDIT`**

## 1. 研究问题

GRAM 用 T5 原生词汇 token 表示层次化 item ID。相同 lexical token 会在大量不同 Trie
prefix 下重复出现，但标准 LM head 始终用同一个 token embedding 对它们打分。一个
词在不同子树节点上的局部推荐含义可能完全不同，shared lexical readout 因而必须仅靠
decoder hidden state 消解 node polysemy。

研究问题是：

> 在保留预训练 lexical semantics 的同时，为每个 Trie edge 增加轻量、
> prefix-specific 的 node residual readout，能否改善合法 sibling 排序？

## 2. 独立结构证据

以下统计只读取固定 item-to-lexical-ID catalog，不涉及 validation：

| 数据集 | Trie node occurrences | unique lexical tokens | reused tokens | reused node occurrences |
|---|---:|---:|---:|---:|
| Toys | 32,426 | 9,214 | 5,789 | 29,001 |
| Beauty | 63,789 | 9,864 | 7,595 | 61,520 |

按完整 item path 计算：

| 数据集 | 含 reused token 的 item | reused path-step rate |
|---|---:|---:|
| Toys | 99.93% | 87.51% |
| Beauty | 100.00% | 96.07% |

因此 lexical-token reuse 不是少数 corner case，尤其在 Beauty 几乎覆盖全部路径。
这项前提独立于 IALC 的非法词表概率质量，也不是对 IALC 失败门槛的修补。

## 3. 暂定方法

对 Trie prefix `p` 的合法 child token `c`，令该 edge/node 为 `n=(p,c)`。标准
lexical score 为：

```text
s_lex(c) = h_t^T E_c
```

LNDR 增加 node-specific 低秩残差：

```text
s_node(n) = (W_h h_t)^T r_n
s_LNDR(c | p) = s_lex(c) + beta * tanh(s_node(n))
```

- `E_c` 保留 T5 原生 lexical embedding；
- `r_n` 是 edge-specific 小维度 embedding；
- `W_h` 是共享低秩投影；
- node 分支零初始化，使 step 0 与原 GRAM 完全一致；
- 只对当前 Trie 的合法 children 计算 residual；
- 正式输出仍是原 lexical ID，item mapping、Trie 和 beam size 不变。

训练目标保留原 GRAM token CE，并增加合法 siblings 上的 node-local CE。推理时把
同一 node residual 加到 constrained beam 的合法 child score，不引入 SASRec、
catalog proposal、reranker 或第二 decoder。

## 4. 潜在创新边界

- GRAM 使用 native lexical tokens，但没有 node-specific output readout。
- SEATER 学 balanced tree identifier，并用 InfoNCE/triplet 学 token hierarchy；
  LNDR 固定 GRAM lexical tree，处理的是同一原生词在不同 prefix 节点的 readout
  多义性。
- TrieRec 用 absolute/relative positional encoding 将 Trie 拓扑注入 Transformer；
  LNDR 不重写 token input 或 attention，而是在 constrained output layer 将
  lexical identity 与 node identity 显式分解。
- STAR 对新增 Semantic-ID token 做 embedding alignment；GRAM 没有新增 SID
  vocabulary，LNDR 处理 native-token reuse。
- ReSID 改 tokenizer/quantizer；LNDR 保持 identifier 和 catalog 完全不变。

截至 2026-07-28 的定向检索，没有发现上述“native lexical score + prefix-specific
Trie-edge residual readout”组合用于 generative recommendation。该判断仅为
search-bounded novelty，正式论文仍需扩大检索。

## 5. 为什么比 IALC 更值得继续

IALC-N1 证明非法 vocabulary mass 已很小，因此简单缩小 softmax support 不足以改变
学习问题。LNDR 针对的是合法空间内部的表示共享：即使合法概率总和接近 1，同一
lexical classifier weight 仍在大量不同 Trie 节点间绑定。

LNDR 直接增加合法 sibling 的条件表达能力，同时具备：

- step-0 baseline identity；
- 参数量由低秩 node residual 控制；
- 无外部候选质量依赖；
- 不增加候选集合或 beam budget；
- 可用 training-prefix 做完整 premise/correctness 审计。

## 6. N1 必须回答的问题

N1 在训练前缀和冻结 checkpoint 上检查：

1. reused token 是否跨多个同深度、不同 parent 节点出现；
2. 同一 token 对应节点的 descendant-item metadata centroids 是否具有足够语义差异；
3. user-conditioned decoder states 在这些节点间是否可分；
4. shared lexical score 的 gold-sibling margin deficit 是否集中于 reused-token edge；
5. Toys/Beauty 是否同时存在足够样本和多个 depth 支持。

只有“node meanings 不同 + user states 可分 + shared readout 存在排序缺口”三者在
两域同时成立，才允许设计 S0。否则固定
`STOP_LNDR_NO_NODE_POLYSEMY_DEFICIT`。

N1 的 cohort、metric、threshold 和代码 SHA 必须在运行前另行冻结；本 brief 不授权
读取 validation。

## 7. 关键参考

1. GRAM：<https://aclanthology.org/2025.acl-long.1596/>
2. SEATER：<https://arxiv.org/abs/2309.13375>
3. TrieRec：<https://arxiv.org/abs/2602.21677>
4. ReSID：<https://arxiv.org/abs/2602.02338>
5. MERGE：<https://aclanthology.org/2025.acl-long.497/>
