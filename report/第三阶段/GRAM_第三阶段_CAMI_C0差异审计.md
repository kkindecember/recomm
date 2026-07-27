# GRAM 第三阶段：CAMI C0-N 差异审计报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-07-24
- Verification Status: ANALYZED
- Version Label: cami_c0_novelty_audit_v1
- Design Status: PREREGISTERED C0-N EXECUTED; C0-D NOT UNLOCKED

## 结论

固定决策为 **`STOP_CAMI_NOVELTY`**。

CAMI 原计划要求下面三项新颖性门槛同时成立：

1. 根据 candidate starvation / prefix congestion 稀疏分配 alias；
2. 单 decoder 使用同一 item 的多条合法训练路径，并在 item 层聚合；
3. 在固定总候选预算下主张 tail candidate recovery。

审计后，第 1、3 项仍可形成较窄差异，但第 2 项已经被 Pctx 实质覆盖。由于三项均为
必要门槛，不能用另外两项通过来抵消这一失败，也不能把 CAMI 改名后继续 C0-D。

## 决定性证据

### Pctx

[Pctx](https://arxiv.org/pdf/2510.21276) 已经完成了 CAMI 核心机制的大部分组合：

- 同一个 item 根据完整用户历史拥有多个 personalized Semantic ID；
- 训练时可把某条 SID 替换成同一 item 的其他合法 SID，作为数据增强；
- 单个自回归 encoder-decoder 生成这些不同 SID 路径；
- 推理时把映射到同一 item 的不同 SID 概率聚合成 next-item probability。

CAMI 剩余的“只为 candidate-starved item 分配 alias”和“固定 50 候选下检验 tail
recovery”更接近问题选择与分配规则，而不足以支撑原计划所声称的完整新机制。

### PIT

[PIT](https://arxiv.org/pdf/2602.08530) 进一步维护每个 item 的多个有效 SID，用
User-to-Token recommendation loss 在候选 SID 中选择，并建立 multiple-SID-to-one-item
的动态逆索引。它采用 minimum-loss selection 而非 CAMI 暂定的 path probability sum，
研究重点也是流式生产索引稳定性，但说明多 SID 与 item grounding 已非常拥挤。

### MTGRec

[MTGRec](https://arxiv.org/pdf/2504.04400) 已在 SIGIR 2025 提出 multi-identifier item
tokenization：用相邻 RQ-VAE checkpoint 为每个 item 产生多个 identifier，并用于
curriculum pre-training。其部署阶段回到单 tokenizer，因此没有覆盖多路径推理聚合，
但已经覆盖“一 item 多 identifier 用于 long-tail/数据稀疏”的宽泛贡献表述。

### 其他关键边界

- [MINDER](https://aclanthology.org/2023.acl-long.366/) 已在 ACL 2023 使用多视图
  identifier 并组合到 passage ranking；领域不同，但“multi-view identifier”不能声称
  为新。
- [ActionPiece](https://proceedings.mlr.press/v267/hou25f.html) 已在 ICML 2025
  研究上下文相关 action tokenization。
- [EAGER](https://arxiv.org/abs/2406.14017) 已在 KDD 2024 结合行为和语义生成流。
- [LETTER](https://doi.org/10.1145/3627673.3679569) 与
  [CoST](https://arxiv.org/abs/2404.14774) 已把 collaborative signal 注入 item
  tokenization。
- [RPG](https://arxiv.org/abs/2506.05781) 已在 KDD 2025 研究长 SID 与并行解码。

完整逐工作矩阵见
`artifacts/phase3/cami_c0/novelty_matrix.csv`，结构化 claim 与 gate 见
`artifacts/phase3/cami_c0/claim_evidence.json`。

## 三项 gate

| Gate | 结果 | 原因 |
|---|---|---|
| adaptive allocation | PASS（差异很窄） | 未发现专门按 candidate starvation/prefix congestion 只给部分 item 分配 alias 的工作 |
| single-decoder item marginalization | **FAIL** | Pctx 已做多合法 SID、单生成器和 SID 概率到 item 概率的聚合 |
| fixed-budget tail recovery | PASS | 近邻工作未以固定总 beam 的 tail candidate recovery 作为主要机制问题 |

## 执行边界

- 未实现或运行 C0-D；
- 未读取 validation 效果数据来选择新公式；
- 未读取 test prediction/test metric；
- 未训练、未加载 checkpoint、未使用 GPU；
- 后续如继续，必须先建立一个不同于“上下文多标识符 + item 概率聚合”的新研究问题。

本 STOP 只否定当前 CAMI 论文点的新颖性，不否定固定 Semantic ID 可能造成 candidate
starvation，也不否定多 identifier 在本项目数据上可能有效。
