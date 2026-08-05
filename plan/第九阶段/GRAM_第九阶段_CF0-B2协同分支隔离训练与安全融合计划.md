# GRAM 第九阶段：CF0-B2 协同分支隔离训练与安全融合计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → run
- Origin Date: 2026-08-04
- Verification Status: `P2A_COMPLETED_GATE_PASSED`
- Version Label: `phase9_cf0_b2_isolated_item_then_safe_fusion_v1`
- Authorized Scope: Toys、seed 2023、P9-2A 隔离 item-head 训练
- Not Authorized by This Run: P9-2B 融合训练、arm C、Beauty、test、Sports

## 1. Experiment Overview

- **Title**：CF0-B2 协同分支隔离训练与安全融合
- **Objective**：先回答原始商品 ID 序列能否独立学习有效 next-item 排序，再回答该信号能否以
  恒等初始化的方式安全注入 GRAM。
- **Primary Hypothesis H1**：在完全冻结并不加载 GRAM 文本参数的条件下，两层 causal item
  Transformer 能在 Toys validation 上明显超过非个性化 popularity baseline。
- **Secondary Hypothesis H2**：只有 H1 成立后，zero-init residual fusion 才有资格进入生成实验。
- **Type**：training + gated validation。

本计划是 CF0-B v1 负结果后的具名修订，不把 v1 的失败解释为协同机制失败。v1 诊断见
`report/第九阶段/GRAM_第九阶段_CF0-B_ItemHead融合与梯度诊断报告.md`。

## 2. 诊断依据与设计修订

| v1 诊断 | B2 修订 |
|---|---|
| item-head Recall@10/50 低于 popularity | 先隔离训练并设置独立科学门 |
| tail/middle Recall@50 为 0 | 增加 non-head Recall@50 门槛 |
| CF Transformer generation gradient 约为 item gradient 26 倍 | P9-2A 完全移除 generation loss |
| post-hoc LayerNorm 非恒等且损伤生成 | P9-2B 拟删除该 LayerNorm，使用 zero-init residual |
| 解冻 lm_head 连带解冻 shared embedding | P9-2B 拟保持 shared embedding 与 GRAM 全部参数冻结 |
| 未归一化 dot-product head-only | B2 使用 L2-normalized user/item cosine logits 与温度缩放 |

## 3. P9-2A：隔离 Item-head 实验

### 3.1 数据与隔离边界

- 数据域：仅 Toys；
- train：每位用户去掉 validation/test 两个末项后，对剩余序列构造非空前缀 next-item 样本；
- validation：历史为 `items[:-2]`、目标为 `items[-2]`；
- 最大历史长度 20，保持时间正序；
- raw item ID 按与 GRAM 相同的字典序映射到连续 ID 1…11,924，0 仅作 padding；
- 不构造或读取 test，不读取 Beauty/Sports；
- 不加载 GRAM checkpoint，确保文本模型参数不可能发生漂移。

预期样本规模：109,361 左右 train augmented prefixes、19,412 validation；实际数量以
`summary.json` 为准。

### 3.2 模型与训练

| 项目 | 冻结值 |
|---|---:|
| seed | 2023 |
| item embedding dim | 512 |
| causal Transformer | 2 layers、4 heads、FFN 2048 |
| dropout | 0.1 |
| scoring | normalized cosine × learnable temperature |
| initial temperature | 0.07 |
| loss | full-catalog cross entropy |
| epochs | 10，固定完成，不自动 early-stop |
| train batch | 512 |
| validation batch | 1024 |
| optimizer | AdamW |
| learning rate | 3e-4 |
| weight decay | 0.01 |
| warmup | 5% linear warmup，随后 linear decay |
| gradient clip | 1.0 |

每个 epoch 完成全量 validation。best checkpoint 先按 Recall@10、再按 NDCG@10 选择；
不以 test 指标选择模型。

### 3.3 对照与指标

对照：

- CF0-B v1 item head；
- 按 train-prefix 商品频次排序的 popularity baseline。

主指标：Recall@10、Recall@50、NDCG@10；辅助指标：Recall/NDCG@1/5/20、MRR、median
rank、history-length 分层、target-popularity 分层。

冻结 popularity baseline：

| 指标 | 值 |
|---|---:|
| Recall@10 | 0.01231197197609726 |
| Recall@50 | 0.03492684937152277 |
| NDCG@10 | 0.006426731134807865 |

### 3.4 科学门

P9-2A 只有同时满足以下条件才标记 `passed`：

1. Recall@10 ≥ popularity Recall@10 × 1.20，即约 `0.014775`；
2. Recall@50 ≥ popularity Recall@50 × 1.20，即约 `0.041912`；
3. tail 与 middle 合并后的 non-head Recall@50 ≥ `0.005`；
4. 全程 loss、gradient、logit scale 有限，checkpoint 与逐用户 ranks 完整。

科学门失败时停止在 P9-2A，不自动调整超参数或改 objective。科学门通过也只表示有资格另行
授权 P9-2B，不自动启动融合训练。

## 4. P9-2B：安全融合预定义（本轮不执行）

若 P9-2A 通过，下一轮拟采用：

```text
h_fused = h_gram + alpha * gate(h_gram, cf_state) * normalize(cf_state)
```

- `alpha` 初始化为 0，保证初始化时逐 token 输出与原 GRAM 完全相同；
- 不对 `h_gram` 增加新的 post-hoc LayerNorm；
- GRAM encoder、decoder、shared embedding、lm_head 全部冻结；
- 第一轮只训练 projection、gate 和 alpha；
- 先通过 identity 单测与 teacher-forced NLL 非劣化门，再运行 beam validation；
- item head checkpoint 固定，不由 generation loss 反向改写。

P9-2B 的训练长度、学习率和生成准入门在 P9-2A 结果后另写冻结配置。

## 5. Setup 与执行命令

- **Language/Framework**：Python 3.9、PyTorch、单卡 CUDA；
- **Working Directory**：`/mnt/18T/jiangtangyunzhi/projects/recomm`；
- **Entry Command**：

  ```bash
  bash experiment/phase9/run_phase9_cf0_b2_item_p2a.sh start
  ```

- **Status**：

  ```bash
  bash experiment/phase9/run_phase9_cf0_b2_item_p2a.sh status
  ```

- **Stop**：

  ```bash
  bash experiment/phase9/run_phase9_cf0_b2_item_p2a.sh stop
  ```

## 6. Inputs 与 Expected Outputs

| 类型 | 路径 | 成功条件 |
|---|---|---|
| user sequences | `GRAM/rec_datasets/Toys/user_sequence.txt` | SHA256 锁定、只读 |
| item map | `GRAM/rec_datasets/Toys/item_generative_indexing_hierarchy_v1_c32_l5_len32768_split.txt` | 11,924 items、SHA256 锁定 |
| trainer | `experiment/phase9/train_cf0_b2_item_head.py` | compile 与单测通过 |
| best checkpoint | `artifacts/phase9/cf0_b2_toys_item_p2a/best_item_head.pt` | 非空且可严格重载 |
| summary | `artifacts/phase9/cf0_b2_toys_item_p2a/summary.json` | 10 epochs、全量 validation、gate 明确 |
| ranks | `artifacts/phase9/cf0_b2_toys_item_p2a/best_validation_ranks.tsv` | 19,412 用户记录 |
| telemetry | `artifacts/phase9/cf0_b2_toys_item_p2a/gpu_telemetry.csv` | 运行期持续记录 |
| status/log | 同一 artifact 目录 | 科学状态与资源恢复状态分离 |

## 7. Monitoring 与资源协议

- 物理 GPU6；实验开始前确认并停止同卡 CodeLlama，结束时无条件恢复；
- 30,720 MiB 总显存租约；workload peak 在 GPU smoke 后冻结，sidecar 持有其余部分；
- 每 5 秒记录 GPU used/free/utilization；
- 正式运行使用具名 tmux，不依赖 Codex 会话；
- hard timeout 在 GPU smoke 后按实测值冻结；只有 hard timeout 可自动终止；
- 非零退出、OOM、NaN/Inf、缺失输出或 scientific gate failed 均不自动重试；
- `scientific_gate=failed` 是有效科学结果，不应覆盖为工程失败；
- 未经新授权，不启动 P9-2B、Beauty、test 或 Sports。

## 8. Analysis Plan

- Primary metric：best validation Recall@10；
- Secondary：Recall@50、NDCG@10、non-head Recall@50；
- 比较：预注册 popularity baseline 与 CF0-B v1 item head；
- 统计范围：单 seed 机制 pilot，只作准入判断；通过后仍需多 seed；
- 多重比较：主门只有三个预注册条件，其余分层为诊断性；
- 结果报告必须明确区分工程完成、科学门结果和资源恢复结果。

## 9. P9-2A 实际结果（2026-08-04）

- 工程状态：`completed`，10/10 epochs，全部数值有限；
- 科学门：`passed`，三项预注册检查全部通过；
- best epoch：10；
- validation Recall@10：`0.0901504224`（门槛 `0.0147743664`）；
- validation Recall@50：`0.1746342469`（门槛 `0.0419122192`）；
- non-head Recall@50：`0.1400486280`（门槛 `0.005`）；
- NDCG@10：`0.0553614615`；
- train/validation 样本：109,361 / 19,412；
- 训练墙钟：76.22 s，peak allocated/reserved：1,109.84 / 1,564 MiB；
- 资源恢复：GPU6 CodeLlama 已恢复。

结论：隔离后的协同序列信号具有明确的可学性，因此 CF0-B v1 不能被解读为
“协同机制无效”；证据更支持 v1 的目标竞争、分享参数梯度污染和非安全融合设计是主要问题。
P9-2B 现已具备另行预注册与授权的条件，但本轮未启动。详细结果见
`report/第九阶段/GRAM_第九阶段_CF0-B2_P9-2A隔离ItemHead结果与验证报告.md`。
