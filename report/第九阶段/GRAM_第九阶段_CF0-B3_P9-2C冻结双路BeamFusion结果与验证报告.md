# GRAM 第九阶段：CF0-B3 P9-2C 冻结双路 BeamFusion 结果与验证报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-04
- Verification Status: `ANALYZED`
- Experiment ID: `GRAM_PHASE9_CF0_B3_TOYS_BEAMFUSION_P2C_V1`
- Scope: Toys cached validation beams、seed 2023、frozen GRAM + frozen P9-2A item-head
- Excluded: test、Beauty、Sports、重新解码、checkpoint 训练或改写

## 1. Executive conclusion

P9-2C 工程完整成功，calibration gate 通过，但预注册总科学门为 `failed_holdout_gate`。
这不是 overall 无效：在 15,316 个独立 holdout 用户上，lambda=0.75 的冻结 BeamFusion 将
Hit@10 从 `0.118504` 提高到 `0.122682`，绝对增量 `+0.004179`（相对约 `+3.53%`），
2,000 次 paired bootstrap 95% CI 为 `[+0.001827, +0.006531]`；NDCG@10 增量
`+0.002436`，Hit@50 严格不变。

失败来自预注册的 tail safety gate：tail Hit@10 从 `0.091797` 降到 `0.081543`，绝对下降
`-0.010254`。相反，middle/head Hit@10 分别提高 `+0.005080/+0.017530`。因此最准确的结论是：

> **P9-2A 协同信号能在 GRAM 合法 beams 内提供可泛化的互补排序收益，P9-2B 的失败主要是
> hidden-space 融合接口问题；但固定全局权重存在明显 popularity-dependent harm，当前版本
> 不满足部署或后续全域准入条件。**

## 2. 方法与完整性

- 历史 cache：19,412 用户 × 50 legal beams；
- catalog：11,924 items；所有 gold/candidate 均严格映射；
- cache 的 12 个额外文本行被确认是全局历史指标，不是额外用户；
- 程序重算的全量 baseline 与历史 Hit/NDCG@5/10/20/50 在 `1e-12` 内一致；
- 按 `SHA256("2023:" + user_id)` 固定分成 4,096 calibration 和 15,316 evaluation；
- 两个 checkpoint 全程只读；实验 CPU-only，未停止或占用 GPU6 CodeLlama；
- 5/5 tests 通过，正式 evaluator 墙钟 8.30 秒。

融合为每用户 50 candidates 内的标准化分数相加：

```text
joint = z(sequence_score) + lambda * z(item_head_score)
```

lambda 只在 calibration 固定网格选择，holdout 未参与选择。

## 3. Calibration 结果

冻结选择规则得到 `lambda=0.75`。

| 指标 | lambda=0 | lambda=0.75 | delta |
|---|---:|---:|---:|
| Hit@10 | 0.122803 | 0.127441 | +0.004639 |
| NDCG@10 | 0.079700 | 0.081286 | +0.001586 |
| Hit@20 | 0.157227 | 0.162598 | +0.005371 |
| Hit@50 | 0.209473 | 0.209473 | 0 |

lambda > 0、Hit@10 增量至少 0.002、NDCG@10 非劣化三项均通过。网格轨迹在 0.5–0.75
附近达到 top-10 最优，继续增大到 1.5–2.0 后开始下降，说明两路信号互补但不能由 CF 分数主导。

## 4. 独立 holdout 主结果

| 指标 | GRAM baseline | BeamFusion | delta |
|---|---:|---:|---:|
| Hit@1 | 0.040938 | 0.042374 | +0.001436 |
| Hit@5 | 0.090298 | 0.091604 | +0.001306 |
| Hit@10 | 0.118504 | 0.122682 | +0.004179 |
| Hit@20 | 0.153695 | 0.162836 | +0.009141 |
| Hit@50 | 0.212588 | 0.212588 | 0 |
| NDCG@10 | 0.075358 | 0.077795 | +0.002436 |
| NDCG@20 | 0.084241 | 0.087948 | +0.003707 |
| NDCG@50 | 0.095943 | 0.097955 | +0.002012 |
| MRR@50 | 0.066411 | 0.068481 | +0.002070 |

Hit@10 paired bootstrap CI 下界大于 0，说明 calibration 的改善方向在未参与调参的用户上复现。
因为只在固定 beam 集内重排，Hit@50 按设计完全不变；当前候选召回上限为 `0.212588`。

纯 item-head 在相同 beams 内的 Hit@10 为 `0.110799`，低于 GRAM 的 `0.118504`；组合却达到
`0.122682`。这表明收益来自两路排序的互补，而不是简单用 item-head 替换生成分数。

## 5. 分层结果与失败原因

### 5.1 Target popularity

| group | n | baseline Hit@10 | fused Hit@10 | delta |
|---|---:|---:|---:|---:|
| tail | 4,096 | 0.091797 | 0.081543 | **-0.010254** |
| middle | 7,284 | 0.120264 | 0.125343 | +0.005080 |
| head | 3,936 | 0.143039 | 0.160569 | +0.017530 |

收益随目标流行度单调增大，tail 反向且幅度大。这与 P9-2A 自身 tail Recall@50
(`0.07636`) 显著低于 head (`0.27387`) 一致：全局 lambda 没有表达 item-head 在不同流行度区域的
可靠性差异，导致 head-oriented collaborative score 压过了 GRAM 对 tail semantic-ID 的排序。

### 5.2 History length

| history | n | baseline Hit@10 | fused Hit@10 | delta |
|---|---:|---:|---:|---:|
| 1–5 | 9,994 | 0.126176 | 0.129378 | +0.003202 |
| 6–10 | 3,423 | 0.106339 | 0.111014 | +0.004674 |
| 11–20 | 1,899 | 0.100053 | 0.108478 | +0.008425 |

三个 history 组均改善，且长历史改善最大。因此当前失败不支持“item-head 不会使用序列历史”；
证据更集中指向 target-popularity reliability，而不是 history length。

## 6. 预注册科学门

| holdout check | 结果 |
|---|---:|
| Hit@10 delta ≥ +0.002 | PASS，+0.004179 |
| Hit@10 bootstrap lower > 0 | PASS，+0.001827 |
| NDCG@10 non-degradation | PASS，+0.002436 |
| tail Hit@10 non-degradation | **FAIL，-0.010254** |
| Hit@50 identity | PASS，delta 0 |

按“全部满足才通过”的冻结规则，总门必须报告 `failed_holdout_gate`。不能因为 4/5 项通过而
事后删除 tail safety gate。

## 7. 与顶会方法的关系及下一候选

本实验借鉴 [COBRA（NeurIPS 2025）](https://proceedings.neurips.cc/paper_files/paper/2025/hash/86ba836d4c5dd859d795a172911745e2-Abstract-Conference.html)
在 coarse sparse candidates 上组合 dense nearest-neighbor score 的 BeamFusion 思路，但不是其
端到端复现。结果支持 coarse-to-fine late fusion 在本项目成立，同时也印证了
[Adaptive Gradient Masking（NeurIPS 2025）](https://proceedings.neurips.cc/paper_files/paper/2025/hash/15163b538e6be5446614de2ecbbfa026-Abstract-Conference.html)
所强调的异构表示贡献不应被静态、无差别处理这一更一般的问题。

下一候选应是另行预注册的 P9-2D reliability-aware fusion，而不是重新调全局 lambda：

1. 用 item-head 在 beam 内的 margin/entropy 与 candidate popularity composition 估计可靠性；
2. 对 tail-heavy 或低置信用户将 CF 权重连续收缩到 0；
3. gate 特征不得使用 target label 或 target popularity，避免线上不可用信息泄漏；
4. 重新划分 calibration/evaluation，或在新 seed/重新解码 cache 上验证，不能继续窥视本轮 holdout；
5. 保留本轮 overall 提升和 tail non-degradation 两个共同门槛。

这只是后续假设。本报告未启动 P9-2D，也不允许用本轮 holdout 调其阈值。

## 8. 统计解释与谬误扫描

- Overall Confidence: `CAUTION`；
- 主结果使用预注册 holdout 和 2,000 次按用户 paired bootstrap；
- calibration 与 holdout 改善方向一致，降低了单次权重选择偶然性的解释空间；
- 单 seed、单 checkpoint、历史 cache，尚未构成跨 seed 或重新解码复现；
- Simpson's paradox：overall 为正而 tail 为负，已显式报告并由预注册 safety gate 阻止错误通过；
- look-elsewhere / garden of forking paths：lambda 网格与选择规则预注册，holdout 未用于重选；
- survivorship：全部 19,412 用户进入固定分区，无 attrition；
- correlation != causation：这是受控 paired reranking 干预，可归因于当前融合规则，但不能外推
  到所有数据集或所有 collaborative-generative 设计；
- 其余 ecological、Berkson、collider、base-rate、regression-to-mean、reverse-causality 未发现
  改变主结论的适用证据。

## 9. 可复核产物

- plan：`plan/第九阶段/GRAM_第九阶段_CF0-B3_P9-2C冻结双路BeamFusion验证计划.md`；
- frozen config：`artifacts/phase9/configs/cf0_b3_toys_beamfusion_p2c_preregistered.json`；
- summary：`artifacts/phase9/cf0_b3_toys_beamfusion_p2c/summary.json`；
- per-user ranks：`artifacts/phase9/cf0_b3_toys_beamfusion_p2c/per_user.tsv`；
- partition：`artifacts/phase9/cf0_b3_toys_beamfusion_p2c/partition.tsv`；
- status/log：同一 artifact 目录下 `status.json`、`run.log`。

关键 SHA256：partition `576e13f8...42e7a`；per-user `3e0e91db...ac54d`。
