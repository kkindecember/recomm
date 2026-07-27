# GRAM 第三阶段 HBTR-B0 可行性与差异审计报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + search-bounded literature audit
- Origin Date: 2026-07-22
- Verification Status: ANALYZED
- Version Label: hbtr_b0_v1
- Design Status: RESULT-INFORMED NEW CYCLE AFTER LRC-F0 STOP

## 1. 结论

HBTR-B0 整体决策：**GO WITH NOVELTY NARROWING**。数据可行性四项门槛在 Beauty/Toys 均通过；但相关工作已覆盖“hierarchy”、“ranking loss”、“beam hard negatives”和“popularity-aware weighting”的单独贡献，因此不得把任一单点宣称为首创。

可继续验证的狭化假设为：

> 对 GRAM 的词汇化层级 ID，从当前学生模型的 miss@10/hit@50 beam 错误中挖掘负样本，并让 pairwise sequence-likelihood margin 同时依赖 semantic-prefix 混淆深度与仅由训练交互得到的目标流行度，能否以不增加独立 ranker/RL 的方式修复 top-10 排序，并优先改善 tail。

该表述只是**搜索边界内的候选差异**，不是绝对新颖性结论。

## 2. B0-Diag 数据结果

| 数据集 | Recall@10 | Recall@50 | 绝对 gap | miss@10/hit@50 | Tail 可用数 | 共享前缀率 | Baseline NDCG@10 | Oracle 相对空间 | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Toys | 11.941% | 21.193% | 9.252 pp | 1,796 | 683 | 81.180% | 0.076275 | +177.853% | True |
| Beauty | 10.875% | 20.811% | 9.936 pp | 2,222 | 338 | 58.911% | 0.064974 | +220.301% | True |

这些数字表明：

1. 两数据集都有大量“已进 beam-50 但未进 top-10”的可训练错误，不需要先扩大候选召回。
2. 层级混淆不是空机制：在该错误子集中，正确商品与至少一个错误 top-10 商品共享非空前缀的比例显著高于 25% 门槛。
3. Tail 子集也有足够的错误样本，但 Beauty tail 的 Recall@10→50 gap 仅 2.994 pp，因此后续不能只依赖 beam 内排序解决全部长尾问题。
4. Oracle 是把 beam-50 中的 target 人为提到第 1 位的理想上界，不是可实现收益预测，不得用它宣称模型将获得同等增益。

## 3. 数据与泄漏边界

- 仅读取 `GRAM/preds/20260722_020042_Toys_sequential_pred_validation.tsv` 和 `GRAM/preds/20260722_125916_Beauty_sequential_pred_validation.tsv`。
- validation target 为 `sequence[-2]`；test target `sequence[-1]` 未读取。
- Head/tail 流行度只使用 `sequence[:-2]`。
- Beauty/Toys 用户、gold、prediction 映射错配数均为 0。
- 输入 SHA-256 、Python 环境和 wall time 保存于 `artifacts/phase3/hbtr_b0/diagnostic_summary.json`。

## 4. B0-Lit 相关工作差异审计

| 工作 | 已覆盖的关键点 | 对 HBTR 的约束 |
|---|---|---|
| [GRAM](https://aclanthology.org/2025.acl-long.1596/) | lexical hierarchical ID、constrained beam、head/tail 分析 | 必须保持原 ID/Trie 和 matched baseline；不能把 lexical hierarchy 当新贡献 |
| [LOHRec](https://aclanthology.org/2025.findings-emnlp.977/) | quantized-ID hierarchy、ordered candidates、sequence-likelihood ranking loss | 不能宣称首个 hierarchy/ranking loss；必须对比无权 beam pairwise 与 LOHRec-style control |
| [OneRec](https://arxiv.org/abs/2502.18965) | 从自身 beam 生成 self-hard rejected sessions 并用 DPO 对齐 | 不能宣称首个 beam hard negatives；差异必须落在 lexical-prefix×tail 联合 margin 和无 RM/RL |
| [MERGE](https://arxiv.org/abs/2601.20199) | 动态平衡聚类与层级 indexing | HBTR 不重建 ID，仅修改训练目标 |
| [Token-Weighted Multi-Target Learning](https://arxiv.org/abs/2601.17787) | 前缀 token 信息增益加权、稀有 token 加权、head/tail 改善 | 不能宣称首个层级/长尾加权；必须使用 item-level beam confusion 而非仅 token CE |
| [Gryphon](https://arxiv.org/abs/2606.08604) / [RecoChain](https://arxiv.org/abs/2604.25787) | 为 generative candidates 增加 item-level 打分/排序 | HBTR 必须证明不增加独立 ranker 仍有价值，并报告效率差异 |
| [WPAUC/TAWin](https://arxiv.org/abs/2604.22504) | beam-search negatives + Top-K-aligned RL objective | 不能把 beam negative/Top-K alignment 当新贡献；狭化为轻量监督式、层级与长尾联合 margin |
| [AKT-Rec](https://arxiv.org/abs/2605.23310) | generative semantic IDs 下的 head-to-tail asymmetric contrastive transfer | Tail-aware semantic-ID 学习也不是首创；HBTR 聚焦 constrained-beam 内的生成排序错误 |

完整矩阵：`artifacts/phase3/hbtr_b0/literature_matrix.csv`。本轮检索截至 2026-07-22；多篇 2026 工作仍为 arXiv 预印本，所以不把检索未发现等同于绝对首创。

## 5. 允许进入 B1 的范围

B0 GO 只解锁 **B1 设计与正确性 smoke**，不解锁 10% pilot 或全量训练。B1 在启动 GPU 前必须先锁定：

1. 唯一个 pairwise sequence-likelihood loss 公式；
2. 唯一个 beam-negative 数量与缓存刷新策略；
3. prefix-depth 与 popularity 的封顶权重，不得在 smoke 后扫大网格；
4. `lambda=0` 精确回退 baseline、负样本无 target 泄漏、head/tail 权重可检查；
5. <15 分钟、100–500 样本的 GPU3 smoke，只验证 forward/backward/checkpoint/full-ranking 链路，不用 smoke 指标宣称效果。

## 6. 产物

- `experiment/phase3/hbtr_b0_diag.py`
- `experiment/phase3/test_hbtr_b0_diag.py`
- `artifacts/phase3/hbtr_b0/diagnostic_summary.json`
- `artifacts/phase3/hbtr_b0/diagnostic_metrics.csv`
- `artifacts/phase3/hbtr_b0/literature_matrix.csv`
- `artifacts/phase3/hbtr_b0/literature_audit.json`
- `report/第三阶段/GRAM_第三阶段_HBTR_B0可行性与差异审计报告.md`
