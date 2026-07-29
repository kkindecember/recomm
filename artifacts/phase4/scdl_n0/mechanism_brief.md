# SCDL-N0 Mechanism Brief

## Material Passport

- Origin Skill: academic-research-suite / deep-research + experiment-agent
- Origin Mode: three-way scan → plan
- Created: 2026-07-28
- Verification Status: ANALYZED（论文原文/官方论文页 + 本地 GRAM catalog）
- Data Scope: literature + catalog only；未读取 validation/test/Sports
- Direction ID: `SCDL`
- Full Name: **Sibling-Contrastive Discriminative Lexicalization**
- Decision: **`SCDL_N0_PASS_TO_PREMISE_AUDIT`**

## 1. 研究问题与可证伪假设

GRAM 对每个 hierarchy cluster 独立聚合 item vocabulary vector，再取最高分 native
token 作为代表。该局部 argmax 优化“这个 token 是否代表本 cluster”，但没有显式
优化推理时真正发生的 sibling competition：

> 同一 parent 的各 child token 是否既代表自己的 subtree，又能区别相邻 subtrees？

SCDL 固定 GRAM 的 item embedding、hierarchical clustering、tree depth 与 native
vocabulary，只在每个 sibling set 内做一对一联合 lexical assignment。对 child
subtree `c` 和 native token `v`：

```text
represent(c,v) = TFIDF subtree-centroid weight
contrast(c,v)  = represent(c,v) - max_{c' sibling of c} represent(c',v)
joint_score    = represent(c,v) + contrast(c,v)
```

在每个 child 的 top-K representativeness 候选并集上，以 Hungarian matching 最大化
joint score，强制 sibling tokens 不重复。假设是：现有 independent argmax 在两域均
存在足够的 sibling-contrast deficit，而 joint assignment 能在基本保留自身
representativeness 的前提下显著提升 contrast margin。

## 2. 独立机制来源

GRAM 原文说明 cluster-level vocabulary vector 是 cluster 内 item vector 的平均，
随后独立选择最高分 representative token；duplicate full IDs 再追加 digit 保证唯一。
这建立了局部代表性目标，但没有 sibling-relative assignment objective：
<https://aclanthology.org/2025.acl-long.1596/>

相邻研究支持“identifier 必须兼顾语义与可区分性”这一一般问题，但采用不同机制：

- GLEN 动态学习 lexical document identifiers，并用 retrieval relevance refinement；
  它不固定 GRAM hierarchy，也不做 sibling-set bipartite assignment：
  <https://aclanthology.org/2023.emnlp-main.477/>
- Multiview Identifiers 同时使用 synthetic/title/substrings 多种 passage IDs，而非对
  单一 fixed tree 的 native tokens 做 joint assignment：
  <https://aclanthology.org/2023.acl-long.366/>
- Structured Term Identifiers 用 context-aware LLM term generation 提升相似 item
  的 distinguishability；其对象是生成的 term sequence，不是 GRAM cluster-level
  TF-IDF token 的 sibling matching：
  <https://aclanthology.org/2026.findings-acl.984/>
- TS-Rec 对新增 SID tokens 做 semantic initialization/alignment；它不处理 native
  vocabulary token 的离线分配：
  <https://arxiv.org/abs/2602.22632>
- SEATER 为不同 tree nodes 分配 unique learned tokens 并做 hierarchy contrastive
  learning，不保留 native lexical vocabulary：
  <https://arxiv.org/abs/2309.13375>

截至 2026-07-28 的定向检索，未发现上述 fixed hierarchy + sibling-contrastive
native-token one-to-one assignment 组合用于 generative recommendation。该结论是
search-bounded novelty，不是“首次提出”的绝对声明。

## 3. 与既有 Phase-4 方向的边界

- 不修改 user representation、candidate set、beam score 或 post-hoc ranking；
- 不增加 node residual，和 LNDR 的 shared-readout 参数化无关；
- 不缩小 CE support，和 IALC 的 illegal vocabulary mass 无关；
- 不使用 collaborative hard negatives，和 CHPR 无关；
- 在任何 GRAM 训练前先离线改变 lexicalization；若 catalog premise 不成立即停止。

SCDL 不是根据 LNDR 的 0.0086/0.0098 数值构造。其依据是 GRAM 已发表的 independent
cluster argmax 算法与 sibling-constrained decoding 的目标错位。

## 4. 最小关键消融

若未来进入训练，最小对照必须共享原 hierarchy 与训练预算：

1. original GRAM independent lexicalization；
2. representativeness-only unique matching；
3. SCDL representativeness + sibling contrast matching。

不得同时更换 clustering、tree branching、tokenizer、decoder 或 reranker。

## 5. N1、成功模式与停止条件

SCDL-N1 仅使用 catalog text 与现有 lexical IDs，重建 frozen TF-IDF space：

- 检查两域 competitive sibling sets 与多个 depth 的支持；
- 计算 current child contrast margin 与 nonpositive-margin rate；
- 运行冻结 top-K joint oracle，计算 margin gain、set improvement rate 与
  representativeness retention；
- 不训练、不读取 checkpoint、training interaction targets、validation/test/Sports。

只有两域所有冻结门同时通过，才允许设计 SCDL-S0 identifier correctness smoke。
否则固定 **`STOP_SCDL_NO_SIBLING_LEXICALIZATION_DEFICIT`**，不改变 top-K、
contrast weight、文本字段、threshold 或 depth cohort rescue。

## 6. 搜索限制

检索集中于 2023–2026 年英文 generative retrieval/recommendation 论文及官方
ACL/EMNLP/arXiv 页面；未做完整系统综述，近期 preprint 可能遗漏。TS-Rec 目前按
arXiv 原文记录，未将其视为 peer-reviewed 证据。
