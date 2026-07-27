# GRAM 第三阶段：LEI F0-N 差异审计报告

## Material Passport

- Origin Skill: academic-research-suite / deep-research
- Origin Mode: run
- Origin Date: 2026-07-24
- Verification Status: ANALYZED
- Version Label: `lei_f0_novelty_audit_v1`
- Design Status: PREREGISTERED F0-N EXECUTED BEFORE F0-D

## 固定结论

F0-N 的固定决策为
**`NOVELTY_SCOPE_PASS_WITH_TRANSFER_AND_GRAM_NARROWING`**。三项必要 gate 均通过，
因此只解锁在读取任何新分数前预注册的 frozen F0-D；不解锁训练、模型修改、GPU
或 test。

这不是“copy bias 是新问题”的结论。保留下来的最窄问题是：在 GRAM 的 multi-passage
late-fusion 架构中，同一原生 lexical vocabulary 被同时用于 coarse history、
fine `item:` link anchor、fine `similar items:` attribute 和 decoder target；
是否可以用保持布局的 span 反事实，将 identifier span 的负贡献与 metadata 的正贡献
分离。

## 检索范围与限制

审计截至 2026-07-24，覆盖六个预注册近邻簇：GRAM 原论文；lexical generative
retrieval；生成推荐 token representation；token/popularity optimization；
history/repeat；通用 copy 与 lexical-overlap bias。优先使用论文官方页面与原文；
只有预印本的工作明确标为 preprint。

本审计只能给出有日期和查询范围的“未检出实质覆盖”，不能证明全世界绝对不存在。
逐篇矩阵见
`artifacts/phase3/lei_f0/novelty_matrix.csv`，结构化 claim–evidence 见
`artifacts/phase3/lei_f0/claim_evidence.json`。

## 已被覆盖的宽泛贡献

- [GRAM](https://aclanthology.org/2025.acl-long.1596/) 已定义
  semantic-to-lexical translation、information linking 和 collaborative
  verbalization，并用总体 `w/o linking` ablation 证明 linking 净效用；所以不能声称
  linking 本身有害或首次分析 linking。
- [GLEN](https://aclanthology.org/2023.emnlp-main.477/) 与
  [PAG](https://arxiv.org/abs/2404.14600) 已覆盖 lexical identifier 学习、
  collision/ranking 与 lexical/sequential identifier 联合解码。
- [DECOR](https://arxiv.org/abs/2509.10468) 是最接近的干预邻居：它已经使用
  decomposed semantic/collaborative embeddings 和 contextual token composition。
  [ActionPiece](https://arxiv.org/abs/2502.13581) 也已指出相同行为的固定 tokenization
  缺少上下文。因此不能把“上下文化 token”或“分解 embedding”单独作为创新。
- [Token-Weighted Multi-Target Learning](https://arxiv.org/abs/2601.17787) 与
  [Ghost](https://arxiv.org/abs/2605.16825) 已研究 Semantic-ID token 的位置、
  频率、梯度和 popularity bias；不能声称首次发现 token imbalance 或 tail token
  优化问题。
- [MHL](https://aclanthology.org/2026.acl-long.475/) 已做 history masking/reconstruction；
  [OneRec-Think](https://aclanthology.org/2026.acl-long.123/) 已做生成推荐中的历史
  reasoning 与 context pruning。
- [RepeatNet](https://ojs.aaai.org/index.php/AAAI/article/view/4408) 已把 repeat
  与 explore 作为推荐模式建模；[CopyNet](https://aclanthology.org/P16-1154/)
  已建立通用输入到输出复制机制；[word-overlap bias](https://aclanthology.org/2022.emnlp-main.725/)
  在其他 NLP 任务也有系统分析。LEI 不能声称首次研究重复、复制或词面重叠。
- [Structured Term IDs](https://aclanthology.org/2026.findings-acl.984/) 已在生成推荐中
  使用原生词表的结构化文本 identifier 与 grounding，进一步压缩了“native token”
  本身的创新空间。

## 三项必要 gate

| Gate | 结果 | 原文级差异 |
|---|---|---|
| GRAM-role specificity | PASS | 未检出工作把 GRAM 原生 lexical token 的 link-anchor、CF-attribute、decoder-symbol 三重角色定义为一个可证伪机制 |
| span-factorized attribution | PASS | GRAM 只有总体 `w/o linking`；未检出保持 passage/position 并分别审计 `item:` ID、`similar items:` IDs、metadata 对 gold path 贡献的工作 |
| intervention room | PASS WITH STRONG NARROWING | 通用 copy suppression、role/segment embedding、contextual token composition、token weighting 与 identifier redesign 均已有先例；只剩“经机制证据驱动的 GRAM-specific target-free role disambiguation” |

## 最强反对意见与证伪要求

最强反对意见不是“别人已经做过”，而是：重复 lexical ID 很可能是 GRAM 有意设计的
对齐/复制通路，正是 linking 有效的原因；tail miss 上的任何变化也可能只是删 token
或破坏语义，而不是 pathological echo。

因此 F0-D 必须同时满足以下逻辑才可继续：

1. tail miss 中移除 identifier span 对 gold path 有**原始正收益**，不能只依赖相对
   control；
2. 该收益显著超过同 passage、同 token 数的确定性 metadata mask；
3. 保留 identifier 以外的 fine metadata 相对 coarse-only 仍有正贡献；
4. adjusted effect 在 tail miss 上显著大于 tail hit，并在 Toys 与 Beauty 同时成立。

任一条件失败都固定停止 LEI。不得改称一般 copy bias、改 cohort、扫描 span 定义或
用单数据集故事挽救。

## 可写与不可写的论文边界

只有未来数据 gate 通过，才可写“GRAM-native lexical linking role attribution”
以及随后通过验证的 recommendation-specific intervention。所有“first”表述都必须带
检索范围与日期限定。

明确不可写：首次 copy/repetition/overlap/popularity/token-weighting；首次 role
embedding/contextual token；fine passage 整体有害；或看到推荐与历史相似就称为
lexical echo。

## 下一步

在不读取新 checkpoint 分数的前提下预注册 F0-D。F0-N 通过本身不构成机制证据，
也不允许训练或使用 GPU。
