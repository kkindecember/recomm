# GRAM 第三阶段：NLPL D0-N 差异审计报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-07-24
- Verification Status: ANALYZED
- Version Label: `nlpl_d0_novelty_audit_v1`
- Design Status: PREREGISTERED D0-N EXECUTED BEFORE D0-D

## 结论

固定决策为 **`NOVELTY_SCOPE_PASS_WITH_NARROWING`**，因此只解锁计划中已经写定的
双数据集 D0-D CPU 诊断，不解锁模型干预、GPU 或 test。

保留下来的论文问题不是宽泛的“语言模型有 token bias”，而是：

> GRAM 类方法把结构化 item ID 映射到预训练模型原生词元后，frozen-base conditional
> prior 是否在控制 lexical parent 与训练流行度后仍泄漏进 constrained beam exposure，
> 并造成 tail candidate starvation？

这一机制组合在本次截至 2026-07-24 的原文审计中没有被实质覆盖，但贡献边界很窄。

## 最接近工作的约束

- [GRAM](https://aclanthology.org/2025.acl-long.1596/) 明确把 Semantic ID 翻译到
  LLM vocabulary，却没有审计原始 T5 对这些 lexical path 的不等先验。
- [GRLM / Structured Term IDs](https://aclanthology.org/2026.findings-acl.984/)
  进一步证明 native textual identifiers 是活跃方向，但其问题是 semantic gap、
  hallucination 与 grounding，而非 frozen-base prior 导致的曝光差异。
- [Decoding Matters](https://aclanthology.org/2024.emnlp-main.589/) 已经发现推荐解码
  amplification bias：长度归一化会放大含 near-one “ghost token”的 item，并用
  text-free assistant 鼓励低频词元。这是最接近的推荐工作，但其病因和控制对象分别是
  length normalization 与生成频率，不是 sibling/popularity matched 的原生词元先验。
- [Calibrate Before Use](https://proceedings.mlr.press/v139/zhao21c.html) 已证明
  预训练常见 answer 会形成偏好，并可用 content-free input 估计、校准。因此 NLPL
  不能声称首创 content-free calibration，也不能把简单 prior subtraction 单独作为创新。
- [APAO](https://arxiv.org/abs/2603.02730) 处理 teacher forcing 与 beam pruning
  的训练—推理不一致；[Latte](https://arxiv.org/abs/2605.06331) 处理 Semantic-ID
  tree 导致的概率耦合。NLPL 的 D0-D 分别通过 frozen base 与 matched sibling
  把这两类机制排除在主统计量之外。
- [Token-Weighted Multi-Target Learning](https://arxiv.org/abs/2601.17787) 和
  [Ghost](https://arxiv.org/abs/2605.16825) 已覆盖 token frequency、训练目标和
  popularity bias；NLPL 必须控制 training frequency，不能把 tail 增益本身当作机制证据。

完整逐工作矩阵见
`artifacts/phase3/nlpl_d0/novelty_matrix.csv`，结构化证据和 gate 见
`artifacts/phase3/nlpl_d0/claim_evidence.json`。

## 三项 gate

| Gate | 结果 | 原因 |
|---|---|---|
| mechanism isolation | PASS | 未发现工作把 frozen pretrained native lexical prior 从 popularity、tree coupling 和 length effect 中分离 |
| GRAM-specific audit | PASS | 未发现 semantic-to-lexical item path 的 parent/popularity matched beam-exposure 审计 |
| intervention space | PASS（收窄） | 未发现 Trie lexical-ID recommendation 的 frozen-base path neutralization；但通用 calibration 和推荐 decoding debiasing 已存在 |

## 强制主张边界

允许：把贡献写成“native lexical prior exposure mechanism + 结构化控制 + 推荐特异
干预（若后续实验支持）”。

禁止：首次发现 token/answer prior、首次概率校准、首次推荐 decoding bias、首次生成
推荐 popularity debiasing，以及把一个减去 base log-prob 的公式单独宣称为算法贡献。

## 执行边界

D0-N 在任何 D0-D 数据结果之前完成。审计过程未加载 GRAM checkpoint、未读取 test、
未训练、未使用 GPU。下一步只能原样运行 plan 第 17.4–17.5 节的 D0-D；任一双数据集
必要门槛失败即停止 NLPL，不在同一 validation 上改 prompt、matching 或阈值。
