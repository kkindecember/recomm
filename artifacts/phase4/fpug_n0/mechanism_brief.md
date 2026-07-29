# FPUG-N0 Mechanism Brief

## Material Passport

- Origin Skill: academic-research-suite / deep-research + experiment-agent
- Origin Mode: three-way scan → plan
- Created: 2026-07-28
- Verification Status: ANALYZED（论文原文/官方论文页 + 本地 GRAM 实现）
- Data Scope: literature + architecture/code；未读取 validation/test/Sports
- Direction ID: `FPUG`
- Full Name: **Fine-grained Passage Utility Gating**
- Decision: **`FPUG_N0_PASS_TO_PREMISE_AUDIT`**

## 1. 研究问题

GRAM 的 coarse user passage 已序列化完整 history lexical IDs；同时，每个历史 item
还作为独立 fine-grained metadata passage 编码，所有 passage states 在 decoder
cross-attention 中直接拼接。位置 embedding 告诉模型 passage 的相对位置，但没有
显式机制判断某个详细 item passage 对当前 user/target prefix 是证据还是干扰。

研究问题是：

> 在 coarse lexical history 完全保留时，是否有大量 fine-grained history passages
> 会降低冻结模型对 training-prefix gold legal children 的条件似然，而且这种有害
> passage 不能由“固定删除最旧 item”解释？

若前提成立，FPUG 在每个详细 passage encoder representation 上学习
user-conditioned scalar gate；gate 初始化为 1，使 step 0 与原 GRAM 完全一致。
coarse passage 永远不门控，正式输出、Trie、beam 与 lexical IDs 不变。

## 2. 独立机制与文献边界

- GRAM 使用 multi-granular late fusion，将独立编码的 coarse/fine-grained passages
  在 decoder 融合：<https://aclanthology.org/2025.acl-long.1596/>
- `Context Quality Matters in Training Fusion-in-Decoder` 直接研究 FiD context
  quality/quantity，说明 irrelevant passages 是 FiD 的独立结构风险：
  <https://aclanthology.org/2023.findings-emnlp.784/>
- MGFiD 为 FiD 增加 passage reranking、evidence guidance 和 pruning：
  <https://aclanthology.org/2024.findings-naacl.142/>
- RFiD 区分 FiD passages 中的 causal 与 spurious features：
  <https://aclanthology.org/2023.findings-acl.155/>
- Rec-Denoiser 针对 sequential recommendation 的 noisy historical interactions，
  但作用于 self-attentive sequential backbone，不是 GRAM 的 item-detail FiD
  passages：<https://arxiv.org/abs/2212.04120>

截至 2026-07-28 的定向检索，未发现“保留 coarse lexical history，以
training-prefix leave-one-detail-passage utility 监督 GRAM-style FiD passage
gate”的 generative recommendation 方法。该判断为 search-bounded novelty。

## 3. 与既有方向的区别

- 不改变候选或做 post-hoc reranking；
- 不修改 lexical IDs、LM head、合法 vocabulary support 或 node readout；
- 不使用 SASRec/catalog proposal、negative quota 或 popularity correction；
- 与 CF-SAT 不同：CF-SAT 替换 item passage 内的 collaborative neighbor text；
  FPUG 不改 passage 内容，只判断整个 fine-grained passage 是否应参与 decoder fusion；
- coarse history 始终完整，因此不是简单截短交互序列。

## 4. N1 与固定停止条件

N1 对冻结 checkpoint 和 unique-user training prefixes 做 leave-one-detail-passage-out
前向审计。主量是 competitive legal-child CE 的下降：

```text
removal_improvement = CE(full detail passages) - CE(with one detail passage masked)
```

正值表示删除该 passage 改善 gold legal-child likelihood。N1 同时检查 tail、recency
覆盖和相对 oldest-removal baseline 的 oracle advantage。

只有 Toys/Beauty 全部冻结门通过才允许设计 FPUG-S0 correctness smoke；否则固定
`STOP_FPUG_NO_DYNAMIC_PASSAGE_UTILITY_DEFICIT`。不得根据结果更改 harmful threshold、
history minimum、sample cohort、recency quartile 或改成固定 truncation 方法。

## 5. 最小关键消融

若未来进入训练，最小对照为：

1. original GRAM identity gate；
2. 固定 oldest-detail pruning；
3. learned user-conditioned FPUG gate。

三者必须共享 coarse history、backbone、training steps、Trie 与 beam。
