# GRAM 第三阶段 PENS H0 差异审计

## Material Passport

- Origin Skill: academic-research-suite / deep-research
- Origin Mode: bounded novelty audit
- Origin Date: 2026-07-24
- Verification Status: ANALYZED_BOUNDED_SEARCH
- Version Label: `pens_h0_n_v1`

## 结论

固定决策为
**`NOVELTY_SCOPE_PASS_WITH_STRONG_MECHANISTIC_NARROWING`**。

已有研究已经覆盖 learned position embedding 的几何、低维结构、absolute-position
脆弱性、additive position/semantics 解耦、sequential positional attention、FiD
passage guidance 和一般 embedding-norm bias。因此 PENS 不能把这些宽泛主题作为
创新点。

截至 2026-07-24，本轮固定近邻簇中未检出同时覆盖以下三点的工作：

1. 生成式序列推荐中、在 encoder 输出后广播到 passage 全部 token 的 learned
   passage-position embedding；
2. passage-position 的 training exposure 与 embedding norm 分层；
3. 保留每个位置向量方向、仅统一范数的 frozen-checkpoint 因果诊断。

允许保留的窄主张仅为上述交集，不使用绝对“首次”措辞。

## 最强反例

GRAM 原论文表明 position embedding 有用；当前长历史 validation 表现也未普遍下降。
因此范数随位置增长可能是有效的 recency/age code。只有预注册的 H0-D 在双数据集
证明 direction-preserving norm equalization 改善 tail-miss 且不伤 tail-hit，才能
继续；结构相关、zero-position ablation 或单数据集变化均不足以晋级。

## 近邻

详见 `artifacts/phase3/pens_h0/novelty_matrix.csv`。核心近邻包括 GRAM、PARec/FPARec、
What Do Position Embeddings Learn、The Curious Case of Absolute Position
Embeddings、DIET、RFiD 与 learned-PE low-dimensional structure。
