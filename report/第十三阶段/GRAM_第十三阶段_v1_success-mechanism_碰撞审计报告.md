# GRAM 第十三阶段：v1 成功机制与 lexical-ID 碰撞审计

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-14T11:17:37.112244+00:00
- Verification Status: ANALYZED
- Version Label: phase13_v1_success_mechanism_v1

## 口径

- `base cluster size`：与 cold item 共享前 L 个语义 token 的 warm item 数。
- `exact warm overlap`：cold 与 warm 的完整 lexical-ID 字符串完全相同。
- `ambiguous ID`：同一完整 lexical ID 映射到多个 item。GRAM 原评测只比较该字符串，无法确认具体 item。
- `collision-aware strict`：ambiguous gold ID 的整行指标置零；这是保守的 item-level 可确认下界，不是新模型结果。

## 核心结果

| Dataset | Version | Original H@10 | Strict H@10 | Ambiguous hits / all hits | Exact warm-overlap items | Cold collision rate |
|---|---:|---:|---:|---:|---:|---:|
| Toys | v0 | 0.608% | 0.608% | 0 / 27 | 0.000% | 0.000% |
| Toys | v1 | 1.351% | 0.270% | 48 / 60 | 13.315% | 12.645% |
| Toys | v2_iter2 | 0.765% | 0.135% | 28 / 34 | 5.953% | 14.003% |
| Beauty | v0 | 0.306% | 0.306% | 0 / 16 | 0.000% | 0.000% |
| Beauty | v1 | 0.802% | 0.210% | 31 / 42 | 9.220% | 7.403% |
| Beauty | v2_iter2 | 0.516% | 0.076% | 23 / 27 | 6.328% | 7.617% |

## 逐层 warm 簇频率

### Toys

| Version | Level | Mean | Median | P90 | Non-zero |
|---|---:|---:|---:|---:|---:|
| v0 | L1 | 327.333 | 248.0 | 629 | 100.000% |
| v0 | L2 | 33.828 | 14.0 | 93 | 98.759% |
| v0 | L3 | 1.109 | 0.0 | 3 | 32.249% |
| v0 | L4 | 0.190 | 0.0 | 0 | 8.938% |
| v0 | L5 | 0.087 | 0.0 | 0 | 4.897% |
| v1 | L1 | 336.944 | 323.0 | 629 | 100.000% |
| v1 | L2 | 40.272 | 15.0 | 100 | 86.232% |
| v1 | L3 | 2.153 | 1.0 | 4 | 54.469% |
| v1 | L4 | 0.489 | 0.0 | 1 | 28.358% |
| v1 | L5 | 0.267 | 0.0 | 1 | 19.621% |
| v2_iter2 | L1 | 336.653 | 323.0 | 629 | 100.000% |
| v2_iter2 | L2 | 40.017 | 15.0 | 94 | 83.045% |
| v2_iter2 | L3 | 1.689 | 0.0 | 4 | 38.152% |
| v2_iter2 | L4 | 0.320 | 0.0 | 1 | 16.703% |
| v2_iter2 | L5 | 0.159 | 0.0 | 1 | 10.565% |

### Beauty

| Version | Level | Mean | Median | P90 | Non-zero |
|---|---:|---:|---:|---:|---:|
| v0 | L1 | 92.688 | 80.0 | 162 | 99.884% |
| v0 | L2 | 1.763 | 0.0 | 5 | 49.934% |
| v0 | L3 | 0.364 | 0.0 | 1 | 17.498% |
| v0 | L4 | 0.111 | 0.0 | 0 | 7.683% |
| v0 | L5 | 0.055 | 0.0 | 0 | 4.065% |
| v0 | L6 | 0.039 | 0.0 | 0 | 2.875% |
| v0 | L7 | 0.027 | 0.0 | 0 | 2.016% |
| v1 | L1 | 97.848 | 83.0 | 162 | 100.000% |
| v1 | L2 | 3.009 | 2.0 | 8 | 75.281% |
| v1 | L3 | 0.751 | 0.0 | 2 | 38.946% |
| v1 | L4 | 0.327 | 0.0 | 1 | 24.835% |
| v1 | L5 | 0.213 | 0.0 | 1 | 17.449% |
| v1 | L6 | 0.169 | 0.0 | 1 | 14.111% |
| v1 | L7 | 0.138 | 0.0 | 1 | 11.963% |
| v2_iter2 | L1 | 100.207 | 85.0 | 162 | 100.000% |
| v2_iter2 | L2 | 2.615 | 1.0 | 7 | 63.714% |
| v2_iter2 | L3 | 0.539 | 0.0 | 2 | 26.818% |
| v2_iter2 | L4 | 0.230 | 0.0 | 1 | 16.474% |
| v2_iter2 | L5 | 0.153 | 0.0 | 1 | 11.946% |
| v2_iter2 | L6 | 0.123 | 0.0 | 0 | 9.848% |
| v2_iter2 | L7 | 0.101 | 0.0 | 0 | 8.625% |

## H@10 × 最深层 base cluster size

### Toys

| Version | Warm items in base cluster | Events | H@10 |
|---|---:|---:|---:|
| v0 | 0 | 4270 | 0.539% |
| v0 | 1 | 111 | 3.604% |
| v0 | 2-4 | 53 | 0.000% |
| v0 | 5+ | 8 | 0.000% |
| v1 | 0 | 3581 | 0.335% |
| v1 | 1 | 721 | 6.657% |
| v1 | 2-4 | 116 | 0.000% |
| v1 | 5+ | 24 | 0.000% |
| v2_iter2 | 0 | 3988 | 0.176% |
| v2_iter2 | 1 | 359 | 7.521% |
| v2_iter2 | 2-4 | 81 | 0.000% |
| v2_iter2 | 5+ | 14 | 0.000% |

### Beauty

| Version | Warm items in base cluster | Events | H@10 |
|---|---:|---:|---:|
| v0 | 0 | 5071 | 0.256% |
| v0 | 1 | 140 | 2.143% |
| v0 | 2-4 | 23 | 0.000% |
| v0 | 5+ | 0 | 0.000% |
| v1 | 0 | 4599 | 0.261% |
| v1 | 1 | 571 | 5.254% |
| v1 | 2-4 | 64 | 0.000% |
| v1 | 5+ | 0 | 0.000% |
| v2_iter2 | 0 | 4709 | 0.085% |
| v2_iter2 | 1 | 488 | 4.713% |
| v2_iter2 | 2-4 | 37 | 0.000% |
| v2_iter2 | 5+ | 0 | 0.000% |

## 诊断结论

**双域一致：collision-aware strict H@10 均不超过 v0。** 现有 v1 强提升主要来自 lexical-ID 别名，不能继续作为已验证的 item-level cold 推荐收益。

这项结果否定的是当前 v1 评测有效性，不是否定 semantic bridge 本身。下一道必要 Gate 应是 collision-safe ID 赋值与评测；在该 Gate 前，不应把 v1 当作可靠前提直接进入 v4-retriever 或 v5。

## 产物

- Machine-readable: `artifacts/phase13/explore/v1_success_mechanism/analysis.json`
- 采用的 test prediction 文件记录在 JSON 的每个 dataset/version 下。
