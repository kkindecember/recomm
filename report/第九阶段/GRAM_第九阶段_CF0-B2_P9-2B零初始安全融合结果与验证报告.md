# GRAM 第九阶段：CF0-B2 P9-2B 零初始安全融合结果与验证报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-04
- Verification Status: `ANALYZED`
- Experiment ID: `GRAM_PHASE9_CF0_B2_TOYS_SAFE_FUSION_P2B_V1`
- Parent: `GRAM_PHASE9_CF0_B2_TOYS_ITEM_P2A_V1`
- Scope: Toys validation, seed 2023, frozen GRAM + frozen item encoder + trainable safe adapter
- Excluded: test, Beauty, Sports, full beam after failed NLL gate

## 1. Executive conclusion

P9-2B 工程上完整成功，但预注册科学结果为 `failed_nll_gate`。zero-init 安全性得到严格
验证：128 条 validation 上启用 `alpha=0` adapter 前后的 max absolute logit delta 和
per-example NLL delta 都是 `0`。因此，v1 那种“尺度为 0 仍破坏原模型”的结构性错误已被修复。

但训练后的最佳 epoch 1 在固定 4,096 条 validation 上将 mean token NLL 从 `1.680560`
提高到 `1.682325`，配对差为 `+0.001765`，95% bootstrap CI
`[+0.001456, +0.002100]`。方向与预注册的改善目标相反，且 54.47% 样本变差。
按冻结规则，full beam 未运行。

最准确的解读是：**P9-2A 证明了 item 序列信号可学，P9-2B 否定了当前“final user state
广播到 coarse prompt token”的融合接口。** 它仍不支持“协同机制无效”。

## 2. 完整性与安全性

| 项目 | 结果 |
|---|---:|
| train / validation samples | 109,361 / 19,412 |
| identity audit samples | 128 |
| paired NLL samples | 4,096 |
| epochs | 2/2 |
| best epoch | 1 |
| adapter trainable parameters | 786,945 |
| max identity logit delta | 0 |
| max identity NLL delta | 0 |
| source checkpoint integrity | PASS |
| test / Beauty / Sports read | false / false / false |
| wall time | 3,117.06 s（51.95 min） |
| peak allocated / reserved | 3,527.67 / 4,852 MiB |
| CodeLlama restoration | PASS, GPU6 |

原 GRAM 和 P9-2A item checkpoint 在运行前后的 SHA256 完全一致。optimizer 仅包含
`encoder.adapter.*`；训练期间未检出冻结参数 gradient。

## 3. 预注册分级门

### 3.1 Gate 0: identity

| 检查 | 门槛 | 观测 | 结果 |
|---|---:|---:|---|
| max absolute logit delta | ≤ 1e-7 | 0 | PASS |
| max absolute per-example NLL delta | ≤ 1e-8 | 0 | PASS |
| alpha initialization | 0 | 0 | PASS |
| trainable scope | adapter only | adapter only | PASS |

Gate 0 证明 safe residual 的实现满足严格恒等初始，这是对 v1 post-hoc LayerNorm 问题的有效修正。

### 3.2 Gate 1: paired teacher-forced NLL

| 检查 | 门槛 | 观测 | 结果 |
|---|---:|---:|---|
| mean delta | ≤ -0.002 | +0.001765 | FAIL |
| bootstrap 95% upper | < 0 | +0.002100 | FAIL |
| fused-worse fraction | ≤ 0.50 | 0.54468 | FAIL |

baseline/fused mean NLL 为 `1.680560/1.682325`，相对变化约 `+0.105%`。配对差的标准差
为 `0.010413`，paired Cohen's dz 约 `0.169`，是小效应，但方向稳定为劣化，不能写成
“持平”或“有改善趋势”。

### 3.3 Gate 2: full beam

`not_run`。原因为 Gate 1 失败，这是预注册流程的正常终态，不是产物缺失。

## 4. 训练轨迹

| epoch | train NLL | validation delta | 95% CI | worse fraction | actual scale |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.888476 | +0.001765 | [+0.001456, +0.002100] | 0.54468 | 0.11394 |
| 2 | 0.886447 | +0.002428 | [+0.001994, +0.002923] | 0.53223 | 0.14782 |

epoch 2 的 train NLL 继续降低，但 validation 劣化加大；同时 residual scale 从 `0.11394`
增长到 `0.14782`。因此 best checkpoint 按事先约定选为 epoch 1。该轨迹更接近 adapter
对 train prefixes 的过拟合/尺度过强，而不是“多训几轮就会转好”。

## 5. 事后分层诊断

以 best epoch 1 的 4,096 个配对样本分层。本节是 post-hoc，不改变预注册门结论。

### 5.1 History length

| history | n | mean delta | worse fraction |
|---|---:|---:|---:|
| 1–5 | 2,706 | +0.001344 | 0.53363 |
| 6–10 | 925 | +0.002896 | 0.57622 |
| 11–20 | 465 | +0.001962 | 0.54624 |

所有历史长度组均劣化，6–10 组最明显。没有“只是短历史冷启动失败”的证据。

### 5.2 Target popularity

| target group | n | mean delta | worse fraction |
|---|---:|---:|---:|
| tail | 1,265 | +0.002491 | 0.57233 |
| middle | 1,806 | +0.001483 | 0.53322 |
| head | 1,025 | +0.001365 | 0.53073 |

tail 目标受损最大。这和 P9-2A 中 tail item ranking 虽非零但显著弱于 head 的结果一致，暗示
直接广播 final user state 可能放大了不确定协同信号。

### 5.3 Baseline difficulty

| baseline-NLL quartile | NLL boundary | n | mean delta | worse fraction |
|---|---:|---:|---:|---:|
| Q1, easiest | ≤ 1.1061 | 1,024 | -0.001510 | 0.40039 |
| Q2 | 1.1061–1.5351 | 1,024 | +0.001310 | 0.54395 |
| Q3 | 1.5351–1.9739 | 1,024 | +0.002977 | 0.60938 |
| Q4, hardest | > 1.9739 | 1,024 | +0.004282 | 0.62500 |

这是最有信息量的事后现象：融合只在原 GRAM 最容易的四分之一样本上平均改善，随着原任务
难度上升，损害单调扩大。这支持“当前 gate 没有学到可靠的样本级不确定性抑制”这一推断，
但因为分层未预注册，只能作为下一轮假设来源。

## 6. 证据、推断与后续候选分离

### 已被直接证据支持

1. zero-init adapter 在实现上严格恒等；
2. 冻结 GRAM/item 和 optimizer 隔离成功；
3. 当前 final-user-state → coarse-prompt 广播融合使 validation NLL 小幅但稳定劣化；
4. 继续到 epoch 2 没有修复劣化；
5. 当前 checkpoint 不具备 full beam 准入资格。

### 需要新实验验证的推断

1. item state 和 GRAM semantic hidden space 可能存在坐标/目标不对齐；
2. coarse prompt 的广播残差可能丢失了 per-item/per-position 结构；
3. 当前 gate 可能需要 item-head uncertainty 或一致性信号，而不只是 hidden/cf state；
4. 直接在已生成合法 beams 内使用冻结 item-head 分数，可能比 hidden injection 更贴合 P9-2A
   实际学到的 ranking objective。

后续候选中，“固定 item-head 的 legal-beam coverage + reranking”比盲目调小 learning rate 或增加
epoch 更有诊断价值。现已据此预注册 P9-2C；它不改变本报告的 P9-2B 失败结论。

## 7. 顶会方法复核与下一机制决策（2026-08-04 补充）

后续设计参考了四项 primary sources。TIGER 将 sequential recommendation 建模成 semantic-ID
generative retrieval；LMIndexer 明确讨论了离散 semantic ID 与连续 latent distribution 的错配；
NeurIPS 2025 AGM 说明异构表示联合训练会出现收敛不一致；最直接相关的是 NeurIPS 2025
COBRA：其 coarse-to-fine inference 先生成 sparse IDs，再以 dense representation refinement，
并提出组合 beam search 和 nearest-neighbor scores 的 BeamFusion。

- [COBRA, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/86ba836d4c5dd859d795a172911745e2-Abstract-Conference.html)
- [Adaptive Gradient Masking, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/15163b538e6be5446614de2ecbbfa026-Abstract-Conference.html)
- [LMIndexer, ICML 2024](https://proceedings.mlr.press/v235/jin24h.html)
- [TIGER, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf)

据此，P9-2C 将机制接口从“训练 adapter 把 CF state 广播进 GRAM hidden tokens”改为
“GRAM 生成合法 beam，冻结 P9-2A head 在 beam 内打分，标准化后 late fusion”。这不是 COBRA
复现，而是针对现有资产的最小可证伪改造；它把 hidden-space alignment、joint gradient competition
和候选召回固定为受控因素，只回答排序空间融合是否有效。

历史预测缓存已经包含全部 19,412 个 validation 用户的 50 beams 和 sequence scores。文件总行数
为 19,425：1 行 header、19,412 行样本、12 行全局历史指标；并不存在 12 个额外用户。P9-2C
仍将按 user id 与当前数据严格 join，并以 4,096 calibration / 15,316 evaluation holdout 防止
在同一批用户上选 lambda 又报告结果。详细冻结规则见
`plan/第九阶段/GRAM_第九阶段_CF0-B3_P9-2C冻结双路BeamFusion验证计划.md`。

## 8. 统计解释与谬误扫描

- Overall Confidence: `CAUTION`；
- 预注册主结论使用 4,096 个配对样本和 2,000 次 paired bootstrap；
- 绝对效应很小（+0.001765 NLL），但 CI 不跨 0，且三项冻结门同时失败；
- 未做独立 seed/checkpoint 复现，所以 Verification Status 是 `ANALYZED`，不是 `VERIFIED`；
- 11/11 statistical fallacy types checked；
- Simpson's paradox：主结论为整体劣化，难度 Q1 组方向相反，已显式报告；它不推翻预注册整体门；
- ecological fallacy：不适用，分析和推断都在用户样本级；
- Berkson's paradox / collider bias：未发现相关选择或控制结构；
- base-rate neglect：不适用，不是诊断准确率任务；
- regression to the mean：未按极端结果选样；
- survivorship bias：4,096 条固定前缀全部进入分析，无 attrition；
- look-elsewhere effect：主门为预注册；分层结果标记为 post-hoc，不作新的通过声明；
- garden of forking paths：主流程被冻结配置限定；后续候选未回写为原假设；
- correlation != causation / reverse causality：这是受控配对模型干预，但只支持对当前 adapter
  设计的归因，不外推到所有协同机制。

## 9. 可复核产物

- 预注册计划：`plan/第九阶段/GRAM_第九阶段_CF0-B2_P9-2B零初始安全融合验证计划.md`
- 冻结配置：`artifacts/phase9/configs/cf0_b2_toys_safe_fusion_p2b_preregistered.json`
- summary：`artifacts/phase9/cf0_b2_toys_safe_fusion_p2b/summary.json`
- paired NLL：`artifacts/phase9/cf0_b2_toys_safe_fusion_p2b/paired_nll.tsv`
- best adapter：`artifacts/phase9/cf0_b2_toys_safe_fusion_p2b/best_adapter.pt`
- status/log/telemetry：同一 artifact 目录下的 `status.json`、`run.log`、`gpu_telemetry.csv`。
