# GRAM 第九阶段：CF0-B Item Head、融合与梯度诊断报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-04
- Verification Status: `ANALYZED`
- Version Label: `phase9_cf0_b_toys_p1_diagnostics_v1`
- Source Experiment: `GRAM_PHASE9_CF0_B_TOYS_P1_FULL_V1`
- Dataset / Split: Toys / validation
- Samples: item head 19,412；融合消融 512；梯度探针 4 batches × 4
- Test/Sports: 未读取

## 1. 结论

CF0-B 的负结果不能据此否定“协同序列增强”机制，但当前实现尚未形成有效的协同模型，
并且融合路径明显破坏了已训练好的 GRAM 表示。证据同时指向三个设计问题：

1. item head 的 Recall@10 仅为 `0.009376`，低于纯训练频次 popularity baseline
   `0.012312`，尚未达到可用于增强生成的最低信号质量；
2. 当前 checkpoint 在移除整个 CF0 路径后，teacher-forced generation NLL 从
   `4.8134` 降至 `1.6013`；完整 CF0 路径在 95.31% 样本上比 bypass 更差；
3. CF Transformer 上 generation gradient L2 为 item gradient 的约 26 倍，训练主导信号
   仍来自生成损失，而不是 next-item 监督。

因此，本轮更准确的判断是：**CF0-B v1 的表示接口、训练隔离和 item objective 均未达到
机制验证条件；目前不能把失败归因于协同信息本身。**

## 2. Item-head 独立评测

### 2.1 全量结果

| 指标 | CF0 item head | Popularity baseline |
|---|---:|---:|
| Recall@1 | 0.000567 | 0.002576 |
| Recall@5 | 0.006491 | 0.007470 |
| Recall@10 | 0.009376 | 0.012312 |
| Recall@20 | 0.014063 | 0.017669 |
| Recall@50 | 0.032403 | 0.034927 |
| NDCG@10 | 0.004570 | 0.006427 |
| NDCG@50 | 0.009347 | 0.011084 |
| MRR | 0.004853 | — |
| Median rank | 4,269 | — |

item head 在所有报告 cutoff 上均未超过非个性化 popularity baseline。训练日志中 item CE
从 `10.0878` 降至 `9.2872`，但对 11,924 个商品的均匀预测 CE 约为 `9.3863`；较低 CE
没有转化为有效的 top-k 个性化排序。

### 2.2 分层结果

| 分层 | 样本数 | Recall@10 | Recall@50 | MRR |
|---|---:|---:|---:|---:|
| history 1–5 | 12,673 | 0.010337 | 0.035193 | 0.005043 |
| history 6–10 | 4,319 | 0.009261 | 0.032646 | 0.005605 |
| history 11–20 | 2,420 | 0.004545 | 0.017355 | 0.002511 |
| tail target | 5,160 | 0 | 0 | 0.000176 |
| middle target | 9,235 | 0 | 0 | 0.000486 |
| head target | 5,017 | 0.036277 | 0.125374 | 0.017700 |

item head 的 top-50 命中全部来自 head 商品；tail 和 middle 商品均为 0。长历史没有带来
更强的序列建模收益，反而明显更差。这与“有效利用协同序列”的预期不符，更接近弱学习下的
热门商品收缩。

## 3. 融合路径消融

在相同 checkpoint 和前 512 个 validation 样本上计算逐样本 teacher-forced token NLL：

| 条件 | Mean NLL | Median NLL |
|---|---:|---:|
| 完整 CF0-B | 4.8134 | 5.0966 |
| 注入尺度设为 0、保留新增 LayerNorm | 11.9978 | 11.9863 |
| 完全 bypass CF0 路径 | 1.6013 | 1.5015 |

配对差值：

- full − bypass：`+3.2121`，95.31% 样本为正，说明当前 CF0 路径整体损害生成；
- zero-injection − bypass：`+10.3965`，100% 样本为正；
- full − zero-injection：`−7.1844`，说明协同残差在当前 checkpoint 中补偿了部分额外
  归一化破坏，但不足以恢复原生成表示。

该消融是 checkpoint 后验反事实；由于模型是在完整 CF0 路径下训练的，不能把
`zero-injection − bypass` 全部解释成 LayerNorm 的独立因果效应。但实现中对原 encoder
输出再次执行 `cf0_token_norm`，即使注入尺度为 0 也不保持恒等映射，确实违反“安全残差初始化”
原则，是下一版应首先移除的结构性风险。

## 4. 梯度诊断

梯度探针复现 joint-top-layer 阶段的可训练范围，item gradient 已乘原实验权重 `0.1`。

| 参数组 | Generation grad L2 | Weighted item grad L2 | 比值 Gen / Item | 梯度 cosine |
|---|---:|---:|---:|---:|
| item embedding | 0.02053 | 0.06696 | 0.31 | −0.0003 |
| position embedding | 0.02045 | 0.00057 | 35.71 | −0.0449 |
| CF Transformer | 1.95395 | 0.07510 | 26.02 | +0.0210 |
| CF sequence norm | 1.40856 | 0.06871 | 20.50 | −0.0051 |
| CF token norm | 2.75263 | 0 | — | — |
| encoder top-2 | 11.03871 | 0 | — | — |
| decoder top-2 | 5.25141 | 0 | — | — |
| shared embedding / LM head | 0.41283 | 0 | — | — |

item loss 确实到达 item embedding、position embedding、CF Transformer 和 sequence norm，
不存在简单的“梯度断掉”。但 CF Transformer 和 sequence norm 主要由 generation loss 驱动，
next-item 信号弱一个数量级以上。两种梯度在共享 CF 参数上的 cosine 接近 0，未发现强烈
方向冲突；主要问题是信号强度失衡。

另一个重要实现细节是 T5 的 `lm_head.weight` 与 `shared.weight` 绑定。当前训练代码解冻
`lm_head` 时，实际也解冻了 16,449,536 参数的共享 token embedding，从而同时改变 encoder
输入词向量和输出词表投影。这扩大了生成漂移，且不符合“仅联合微调顶层”的直觉定义。

## 5. 归因与下一版最低修订条件

本轮支持“设计尚未成熟”而非“机制无效”。在进入 arm C 或 Beauty 前，建议先建立 CF0-B2：

1. **先验收 item head**：完全冻结 GRAM，单独训练协同分支；只有 validation item-head
   Recall@10/50 明显超过 popularity baseline 后才允许融合。
2. **恒等初始化融合**：删除对原 encoder token 的额外 post-hoc LayerNorm；使用 zero-init
   residual gate，例如 `h' = h + alpha * normalized_cf` 且 `alpha=0` 初始化。
3. **隔离共享 embedding**：不通过 tied `lm_head` 解冻 `shared.weight`；先只训练 CF branch、
   gate/adapter，必要时再解冻极少量顶部参数。
4. **修订 item objective**：评估 normalized dot product、temperature、item bias 或 sampled
   softmax，避免当前 head-only top-k 行为；同时固定报告 popularity baseline。
5. **分阶段门槛**：item head 合格后先做 teacher-forced identity/融合消融，再跑 beam
   validation；不直接叠加动态 gate 和 rerank。

以上是诊断结论，不构成下一轮训练授权；未启动 arm C、Beauty 或 test。

## 6. 验证与完整性

- 诊断脚本 Python compile 通过，CPU smoke 通过，CF0 单测 5/5 通过；
- source checkpoint SHA256：
  `9680b0af83a4814b16730c41fe8b99a7192370d236e1186ed94399d86c4cccbf`；
- 全量 item ranks、诊断 JSON、日志和 GPU telemetry 已持久化；
- runner 终态 `succeeded`，CodeLlama 已恢复到物理 GPU6；
- GPU telemetry 最大整卡 used 为 28,686 MiB，未超过 30,720 MiB 总租约。

## 7. 统计解释与谬误扫描

- Overall Confidence: `CAUTION`；
- 11/11 statistical fallacy types checked；
- 未发现 Simpson、生态谬误、Berkson、collider、base-rate neglect、regression-to-mean、
  survivorship、look-elsewhere、相关即因果或反向因果的适用证据；
- Garden of Forking Paths：`NOTE`。主 P9-1 已预注册，但本报告属于 post-hoc 诊断，所有
  阈值应作为下一轮预注册依据，不能回写为原实验的先验门槛；
- 未做重复 seed 或 checkpoint 重跑，故 Verification Status 为 `ANALYZED`，不是
  `VERIFIED`。

## 8. 证据路径

- `artifacts/phase9/cf0_b_toys_p1_diagnostics/diagnostics.json`
- `artifacts/phase9/cf0_b_toys_p1_diagnostics/item_head_ranks.tsv`
- `artifacts/phase9/cf0_b_toys_p1_diagnostics/run.log`
- `artifacts/phase9/cf0_b_toys_p1_diagnostics/status.json`
- `artifacts/phase9/cf0_b_toys_p1_diagnostics/gpu_telemetry.csv`
- `artifacts/phase9/configs/cf0_b_toys_p1_diagnostics_preregistered.json`

