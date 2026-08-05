# GRAM 第九阶段：CF0-B2 P9-2B 零初始安全融合验证计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → run
- Origin Date: 2026-08-04
- Verification Status: `P9_2B_COMPLETED_FAILED_NLL_GATE`
- Version Label: `phase9_cf0_b2_safe_fusion_p2b_v1`
- Parent Evidence: `GRAM_PHASE9_CF0_B2_TOYS_ITEM_P2A_V1` (`scientific_gate=passed`)
- Authorized Scope: Toys validation, seed 2023, identity/NLL/beam gated safe fusion
- Excluded: test, Beauty, Sports, arm C reranking, GRAM/shared/lm-head fine-tuning

## 1. Experiment Overview

- **Title**：冻结协同表示的 zero-init residual 安全融合。
- **Objective**：区分“item 序列可学”和“item state 可安全用于生成”，验证严格恒等初始的
  小型 adapter 能否改善 teacher-forced NLL，并在 full validation beam ranking 上保持非劣。
- **Primary Hypothesis H1**：zero-init 时融合路径与原 GRAM 逐 logit 严格一致。
- **Primary Hypothesis H2**：只训练 adapter 后，固定 validation 子集的配对 token NLL 显著降低。
- **Safety Hypothesis H3**：通过 H2 后，full validation beam Hit@10/NDCG@10/Hit@50 相对原 GRAM 非劣。
- **Type**：training + staged gated validation。

## 2. 不可变的对照与数据边界

### 2.1 原 GRAM 对照

- checkpoint：`GRAM/log/Toys/1_20260720_1830/id_0_rec_30/model_rec_phase_1_epoch_30.pt`；
- seed 2023，Toys，validation split，beam size 50；
- 不加载 CF0-B v1 checkpoint，避免继承其生成漂移；
- 历史 full-validation 对照值：Hit@10 `0.1194106738`，NDCG@10 `0.0762745143`，
  Hit@50 `0.2119307645`。

### 2.2 协同分支

- checkpoint：`artifacts/phase9/cf0_b2_toys_item_p2a/best_item_head.pt`；
- best epoch 10，item Recall@10/50 = `0.090150/0.174634`；
- 整个 item encoder 和 item embedding 冻结，不接收 generation gradient；
- GRAM 输入中的 recent-first item IDs 先按 valid prefix 翻转为 chronological，再输入冻结 item encoder。

### 2.3 数据

- train：Toys `sequence[:-2]` 构造的 109,361 个 augmented prefixes；
- validation：19,412 用户，target 为 `sequence[-2]`；
- teacher-forced model selection：validation 前 4,096 条，顺序固定；
- identity audit：上述 validation 前 128 条；
- full beam：只在 NLL gate 通过后评估全部 19,412 条 validation；
- 不读取 test，不读取 Beauty/Sports。

## 3. 融合结构

只修改 encoder 第一个 coarse user-prompt passage 的 valid token：

```text
c       = frozen_item_encoder(history)
r       = normalize(Wc(c))
g       = sigmoid(Wh(h_user) + Wg(c) + b)
scale   = 0.20 * tanh(alpha), alpha_init = 0
h_user' = h_user + scale * g * r
```

- `alpha=0` 使初始 residual 逐元素为 0，无额外 post-hoc LayerNorm；
- 只训练 `Wc/Wh/Wg/b/alpha`；
- 原 GRAM encoder/decoder/position embedding/shared embedding/lm-head 全冻结；
- item encoder 全冻结；
- optimizer 中不得出现上述冻结参数；
- checkpoint 只保存 adapter state，不复制或改写原 GRAM/item checkpoint。

## 4. 训练配置

| 项目 | 冻结值 |
|---|---:|
| seed | 2023 |
| epochs | 2，固定完成 |
| batch size | 16 |
| optimizer | AdamW |
| learning rate | 3e-4 |
| weight decay | 0.01 |
| warmup | 5% linear warmup + linear decay |
| gradient clip | 1.0 |
| objective | generation token cross entropy only |
| max residual scale | 0.20 |
| checkpoint selection | minimum mean validation NLL; tie-break lower NLL p95 |

固结 GRAM 保持 eval mode，避免 dropout 使 adapter 学习目标漂移。不使用 item CE，也不进行 beam
指标导向的 checkpoint selection。

## 5. 分级科学门

### Gate 0：Identity（训练前，强制）

128 条 validation 上同一 frozen GRAM 分别 bypass adapter 与启用 `alpha=0` adapter：

1. max absolute logit difference ≤ `1e-7`；
2. max absolute per-example NLL difference ≤ `1e-8`；
3. `alpha == 0`，所有值有限；
4. trainable parameter names 仅属于 adapter。

任一失败立即终止，不训练。

### Gate 1：Teacher-forced NLL（决定是否允许 full beam）

在固定 4,096 条 validation 上计算配对差 `delta = fused_nll - baseline_nll`，使用 seed 2023
的 2,000 次 paired bootstrap：

1. mean delta ≤ `-0.002` token NLL；
2. mean delta 的 95% bootstrap upper bound < `0`；
3. fused-worse fraction ≤ `0.50`；
4. 训练、NLL、gradient 和 residual scale 全部有限；
5. 冻结参数未出现 gradient，且原 checkpoint 文件 SHA256 不变。

Gate 1 失败时产出有效负结果并停止，不运行 full beam、不自动调参。

### Gate 2：Full-validation beam safety

只在 Gate 1 通过后运行 fused checkpoint 的 19,412 条 validation，beam size 50：

1. Hit@10 ≥ `0.1174106738`（baseline − 0.002 absolute）；
2. NDCG@10 ≥ `0.0752745143`（baseline − 0.001 absolute）；
3. Hit@50 ≥ `0.2089307645`（baseline − 0.003 absolute）；
4. 19,412 条 prediction/rank 记录完整，无非法 beam mapping。

Gate 2 是“安全融合 pilot 通过”门，不要求单 seed 显著优于 baseline。真正的增益结论需要
之后多 seed 复现，本轮不自动扩展。

## 6. Setup 与执行

- **Language/Framework**：Python 3.9, PyTorch, Transformers；
- **Working Directory**：`/mnt/18T/jiangtangyunzhi/projects/recomm`；
- **Entry Command**：

  ```bash
  bash experiment/phase9/run_phase9_cf0_b2_safe_fusion_p2b.sh start
  ```

- **Status**：

  ```bash
  bash experiment/phase9/run_phase9_cf0_b2_safe_fusion_p2b.sh status
  ```

- **Stop**：

  ```bash
  bash experiment/phase9/run_phase9_cf0_b2_safe_fusion_p2b.sh stop
  ```

## 7. Expected Outputs

| 产物 | 路径 | 完整性要求 |
|---|---|---|
| adapter checkpoint | `artifacts/phase9/cf0_b2_toys_safe_fusion_p2b/best_adapter.pt` | 非空、严格重载 |
| summary | 同目录 `summary.json` | identity/NLL/beam gate 状态明确 |
| paired NLL | 同目录 `paired_nll.tsv` | 4,096 条 |
| beam predictions | 同目录 `beam_validation.tsv` | Gate 1 通过时 19,412 条 |
| telemetry/log/status | 同目录 | 可监控、可复核 |

## 8. Monitoring 与资源协议

- 使用物理 GPU6；运行前停止同卡 CodeLlama，任意终态都恢复；
- GPU smoke 使用与正式实验相同 batch size，之后冻结 workload peak 和 hard timeout；
- 总显存租约 30,720 MiB，sidecar 持有剩余额度；
- 具名 tmux 持久运行，每 5 秒记录 GPU telemetry；
- 训练日志每个固定步数更新，90 秒无输出只作 stall advisory；
- 除 hard timeout 外不自动终止，非零退出不自动重试；
- Gate 1/2 failed 是科学结果，不写成工程失败。

## 9. Analysis Plan

- Primary：paired validation token-NLL delta 与 95% bootstrap CI；
- Safety：full-validation Hit@10/NDCG@10/Hit@50 non-inferiority；
- Diagnostics：alpha/actual scale、gate mean/std、gradient norm、逐 epoch NLL；
- Comparison：同一 GRAM epoch-30 checkpoint 的历史 validation 结果；
- Scope：单 seed 机制 pilot，不作统计稳健的最终增益声明。

## 10. 实际结果（2026-08-04）

- 工程状态：`completed`；
- 科学状态：`failed_nll_gate`；
- Gate 0 identity：`passed`，128 样本 max logit/NLL delta 均为 `0`；
- best epoch：1；
- baseline/fused mean NLL：`1.6805601 / 1.6823248`；
- paired mean delta：`+0.0017646`，95% bootstrap CI
  `[+0.0014558, +0.0020999]`；
- fused-worse fraction：`0.54468`；
- Gate 1 三项检查全部失败，因此 Gate 2 full beam 按计划未运行；
- best adapter actual residual scale：`0.11394`；
- 训练墙钟：`3117.06 s`，peak allocated/reserved：`3527.67 / 4852 MiB`；
- GRAM/item checkpoint SHA256 前后一致，CodeLlama 已恢复 GPU6。

结论：zero-init 设计确实解决了初始时的生成破坏，但“冻结 final item user state →
coarse user-prompt 广播残差”这一表示接口没有将 P9-2A 的排序能力转化为生成收益。
该负结果否定当前融合设计，不否定协同信号本身。详见
`report/第九阶段/GRAM_第九阶段_CF0-B2_P9-2B零初始安全融合结果与验证报告.md`。
