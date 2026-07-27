# GRAM 第三阶段：FFNF J0-S 固定预算可行性

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-07-24
- Verification Status: VERIFIED
- Version Label: `ffnf_j0_s_v1`

固定决策：**`STOP_FFNF_BUDGET_INFEASIBLE`**。

本阶段仅运行 CPU tokenizer census；未加载 checkpoint、未训练、未读取
validation/test 效果。

## 双数据集结果

| Dataset | CF ratio | Metadata gain | Gain-positive items | Title ratio | Brand ratio | Categories ratio | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Toys | 1.0000 | -234,052 | 0.0000 | 0.9995 | 0.9930 | 0.9969 | False |
| Beauty | 0.6320 | 345,311 | 1.0000 | 1.0196 | 1.0970 | 1.8918 | False |

固定预算指 padded tensor width / decoder capacity 为 64+64=128；短文本
不要求恰有 128 个 active tokens。active-token delta 与第二个 EOS 仅作
J1 confound 记录，不能救援或否定本轮字段覆盖 gate。

## 失败机制

失败不是同一个字段在两个数据集都不够，而是统一 quota 遇到相反的数据结构：

- Toys 的 `top-k=5` collaborative field 较短，current passage 原本能给 metadata
  留出明显超过 64 tokens 的空间。强制 META64 后，CF 虽保留 100%，metadata
  aggregate 反而下降 234,052 tokens，平均每 item -19.63；title、brand、
  categories 也分别降到 current 的 99.95%、99.30%、99.69%。
- Beauty 的 `top-k=10` collaborative field 较长，META64 的确恢复 345,311 tokens，
  但 CF64 只能保留 current collaborative tokens 的 63.20%，未达到 95% gate。

因此 64+64 不是“消除竞争”，而是把 dataset-dependent 竞争固化成同一人工边界。
按照单次规则，不允许看完结果后改成 Toys 96/32、Beauty 64/64 或搜索其它 quota。

## 解释边界

本结果只否定统一 64+64 field partition，不否定所有 field-factorized architecture。
若继续做方法，必须使用不依赖数据集手工 quota、且能证明 capacity/compute matching
的新机制；不能把本轮改成 dataset-specific quota 作为 FFNF 延续。
