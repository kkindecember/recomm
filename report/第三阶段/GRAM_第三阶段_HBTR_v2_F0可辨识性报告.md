# GRAM 第三阶段 HBTR-v2 F0 可辨识性报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-07-24
- Verification Status: ANALYZED
- Version Label: hbtr_v2_f0_v1
- Design Status: RESULT-INFORMED NEW CYCLE

## 结论

- F0 决策：**STOP_HBTR**
- 本结果不改变 HBTR-v1 STOP，也不解锁 GPU 或效果实验。

## 核心结果

| 数据集 | tail非平凡/有效行 | joint非平凡/有效行 | C4-v2=C2 pair率 | 最大tail权重 | 最大margin | gate |
|---|---:|---:|---:|---:|---:|---|
| Toys | 32.61% | 18.25% | 67.39% | 2.000 | 0.380 | PASS |
| Beauty | 15.79% | 8.91% | 84.21% | 2.000 | 0.332 | FAIL |

## 边界

F0 只读取 training-only cache 与 sequence[:-2] popularity；未读取 validation
效果、test、checkpoint 或模型。PASS 只允许设计数值等价的 negative-decoder
micro-batching，不证明推荐效果。
