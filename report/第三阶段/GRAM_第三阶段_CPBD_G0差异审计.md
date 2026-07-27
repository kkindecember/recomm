# GRAM 第三阶段：CPBD G0-N 差异审计报告

## Material Passport

- Origin Skill: academic-research-suite / deep-research
- Origin Mode: run
- Origin Date: 2026-07-24
- Verification Status: ANALYZED
- Version Label: `cpbd_g0_novelty_audit_v1`
- Design Status: PREREGISTERED G0-N EXECUTED BEFORE G0-D1

## 固定结论

G0-N 的固定决策为
**`NOVELTY_SCOPE_PASS_WITH_TRANSFER_AND_GRAM_NARROWING`**。三项必要 gate 均通过，
因此只允许在读取新截断统计前另行预注册 G0-D1；不解锁模型效果诊断、训练、GPU
或 test。

保留下来的最窄问题不是“prompt 太长”，而是：GRAM 把 collaborative lexical IDs
固定放在 metadata 之前，并把每个 fine passage 从左到右限制为 128 tokens；这个
前置字段是否系统性地挤出后置 metadata，以及能否在固定总预算和固定 CF identity
下把“CF 自身噪声”与“metadata displacement”区分开。

## 检索范围与限制

审计截至 2026-07-24，覆盖 GRAM、推荐 length bias、推荐 token-efficient item
representation、prompt compression、rate-distortion、FiD/RAG evidence allocation、
behavior retrieval、动态 retrieval budget、输出 token weighting 与 variable-length
Semantic IDs。优先使用论文官方页面和原文；只有预印本的工作明确标为 preprint。

本审计只能给出“在固定近邻簇中未检出实质覆盖”，不能证明绝对 first。逐篇矩阵见
`artifacts/phase3/cpbd_g0/novelty_matrix.csv`，结构化 claim–evidence 见
`artifacts/phase3/cpbd_g0/claim_evidence.json`。

## 已被覆盖的宽泛贡献

- [GRAM](https://aclanthology.org/2025.acl-long.1596/) 已定义 collaborative
  verbalization、information linking 与 multi-granular late fusion，也报告过细粒度
  item information 的收益和 `top-k` sensitivity；所以不能把“similar items 太多会有
  噪声”或调 `top-k` 当作新贡献。
- [LBR](https://arxiv.org/abs/2607.04270) 已研究 LLM recommendation 的 input/output
  length bias、equal-length truncation/padding 和 length-aware calibration；不能声称
  首次发现推荐中的长度偏差。它没有把同一 item passage 内的 collaborative/metadata
  字段顺序作为固定预算下的因果机制。
- [LLMLingua](https://aclanthology.org/2023.emnlp-main.825/)、
  [LongLLMLingua](https://aclanthology.org/2024.acl-long.91/) 与
  [LLMLingua-2](https://aclanthology.org/2024.findings-acl.57/) 已覆盖 budget
  controller、token compression、document reordering、动态 compression ratio 和
  learned token selection；[NeurIPS rate-distortion work](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ac8fbba029dadca99d6b8c3f913d3ed6-Abstract-Conference.html)
  也已形式化 variable-rate prompt compression。
- [RECOMP](https://arxiv.org/abs/2310.04408)、
  [RFiD](https://aclanthology.org/2023.findings-acl.155/)、
  [MGFiD](https://aclanthology.org/2024.findings-naacl.142/) 与
  [FiDO](https://aclanthology.org/2023.findings-acl.732/) 已覆盖 RAG/FiD 的
  evidence compression、guidance、pruning 与结构效率；它们没有研究 GRAM item
  passage 内 collaborative prefix 对 metadata 的 displacement。
- [ReLLa](https://arxiv.org/abs/2308.11131) 与
  [ReLLaX](https://arxiv.org/abs/2501.13344) 已覆盖长行为 retrieval 和 collaborative
  fine-tuning；动态 retrieval-count allocation 也已有 2026 preprint。因此不能声称
  首次做 adaptive context/retrieval budget。
- [MSL](https://doi.org/10.1145/3726302.3730041) 已在推荐训练中选择性 mask 输出
  loss token；[I-LLMRec](https://openreview.net/forum?id=vizM7B7vuW) 已直接研究
  token-efficient item representation；VarLenRec 与 Token-Weighted 已覆盖输出
  Semantic-ID 的长度和重要性分配。CPBD 不能泛化为“首次发现 token 不等价”。

## 三项必要 gate

| Gate | 结果 | 原文级差异 |
|---|---|---|
| GRAM structural specificity | PASS | 未检出工作把 GRAM 类 fine passage 的前置 collaborative lexical field 导致后置 metadata 右截断定义为可检验推荐机制 |
| displacement identifiability | PASS | 未检出工作同时固定 128-token budget、passage 数、原始字段和 CF identity，再用 paired reserialization 区分直接 CF noise 与 metadata displacement |
| intervention room | PASS WITH STRONG NARROWING | 通用 compression、length calibration、reordering、动态 retrieval budget 与 top-k 均已有工作；只剩经机制证据驱动的 GRAM-specific、target-free collaborative/metadata field allocation |

## 最强反对意见与证伪要求

最强反对意见是：这可能只是本地 serialization/configuration defect。长 description
的末尾被截断在机械上并不意外，被截内容可能只是低价值 prose；把 metadata 放到前面
也会让 CF evidence 后移或消失，收益可能只是 positional advantage。

所以后续不能以“统计到 lost tokens”直接进入训练：

1. G0-D1 必须用当前生产 tokenizer/filter/truncation 精确证明双数据集都有广泛、
   实质的可恢复 metadata loss；
2. 后续 frozen diagnosis 必须固定 128-token budget、passage 数、原始字段与 CF
   identity，并加入 matched within-metadata position control；
3. 必须证明 gold-path 收益跨数据集成立，且不能由一般位置变化或删掉 collaborative
   evidence 解释；
4. 最终 allocator 必须使用 target-free、training-only 信号并胜过固定
   metadata-first；否则最多是工程修复，不够成为论文方法。

## 可写与不可写的论文边界

当前只可写：截至审计日期，在固定近邻簇中未检出对
“GRAM-native collaborative-prefix metadata displacement + fixed-budget/fixed-CF
paired diagnosis”的实质覆盖。

不可写：首次 prompt compression、length bias、dynamic budget、field selection、
reordering、top-k 或 token-efficient recommendation；也不可写 similar-item IDs
一般有害，或 metadata-first 本身已经证明有效。

## 下一步

在读取任何新的 raw/visible/lost token 统计前，冻结 G0-D1 static truncation census。
G0-N 通过只说明值得做这个低成本结构诊断，不说明 displacement 已达到建模价值。

## AI 辅助说明

本审计由 AI 辅助完成文献检索、差异矩阵整理与反对意见压力测试；关键来源均以论文
官方页面、DOI 或明确标注的 arXiv 记录核验。由于检索是有截止日期和查询簇边界的，
结果不构成绝对新颖性证明。
