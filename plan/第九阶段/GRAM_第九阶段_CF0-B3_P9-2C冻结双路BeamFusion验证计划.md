# GRAM 第九阶段：CF0-B3 P9-2C 冻结双路 BeamFusion 验证计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → run
- Origin Date: 2026-08-04
- Verification Status: `EXECUTED_FAILED_HOLDOUT_GATE`
- Experiment ID: `GRAM_PHASE9_CF0_B3_TOYS_BEAMFUSION_P2C_V1`
- Parent Evidence: P9-2A item-head passed；P9-2B safe hidden fusion failed NLL gate
- Authorized Scope: Toys validation cache、seed 2023、冻结模型离线 reranking
- Excluded: test、Beauty、Sports、重新生成 beams、训练或改写任一 checkpoint

## 1. 决策与研究问题

本轮不再把协同向量注入 GRAM hidden state，而改成两个已经独立完成目标学习的分支在解码期
组合：GRAM 负责产生 50 个合法 semantic-ID candidates，P9-2A item-head 只在这些 candidates
内部提供协同排序分数。这是一次机制接口的大改，直接检验：

> P9-2A 已学到的 next-item ranking 信号，能否在不承担跨表示空间对齐、不反向污染 GRAM、
> 不改变候选集合的条件下，提高 GRAM top-10 排序？

主假设 H1：固定权重的标准化双路分数能在独立 holdout 上提高 Hit@10，并保持 NDCG@10、
tail Hit@10 和 Hit@50 不退化。

## 2. 顶会依据与本项目取舍

- TIGER 将序列推荐表达为 semantic-ID 自回归检索，说明 GRAM 的合法 beam 可被视为粗检索候选；
- LMIndexer 指出离散 semantic ID 与连续 latent representation 存在分布错配风险；
- NeurIPS 2025 COBRA 用 sparse-ID generation 后接 dense refinement，并以 BeamFusion 组合 beam
  search 与 nearest-neighbor scores；
- NeurIPS 2025 AGM 进一步表明异构表示联合训练可能出现分布差异和收敛不一致。

因此本轮只借鉴 COBRA 的 coarse-to-fine/BeamFusion 原则，不声称复现其端到端架构。相对于
P9-2B，冻结式 late fusion 把 hidden-space alignment 和 joint optimization 两个混杂因素移除。

Primary sources:

- [COBRA, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/86ba836d4c5dd859d795a172911745e2-Abstract-Conference.html)
- [Adaptive Gradient Masking, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/15163b538e6be5446614de2ecbbfa026-Abstract-Conference.html)
- [LMIndexer, ICML 2024](https://proceedings.mlr.press/v235/jin24h.html)
- [TIGER, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/20dcab0f14046a5c6b02b61da9f13229-Paper-Conference.pdf)

## 3. 冻结输入与完整性 Gate 0

输入：

1. GRAM epoch-30 历史 validation 预测缓存，包含每用户 50 beams 和 sequence scores；
2. P9-2A epoch-10 `best_item_head.pt`；
3. Toys `user_sequence.txt` 和固定 semantic item map。

Gate 0 必须全部通过：

- 恰好 19,412 个数据行，文件末尾 12 行历史汇总不作为样本；
- cache user id 与 `user_sequence.txt` user id 集合严格相同，均无重复；
- 每用户恰好 50 个 candidates 和 50 个有限 sequence scores；
- gold 与当前 validation 目标 lexical ID 严格一致；
- 每个 candidate 均能唯一映射到 1…11,924 的 item id；
- checkpoint 的 catalog size、维度与数据映射一致；
- 所有输入 SHA256 与冻结配置一致。

任一失败即 `failed_integrity_gate`，不修补数据、不重新生成 beam。

## 4. 数据分区与防泄漏

对 19,412 个 user id 计算 `SHA256("2023:" + user_id)`，按 `(digest, user_id)` 升序：

- 前 4,096 用户：`calibration`，仅用于选择融合权重；
- 后 15,316 用户：`evaluation holdout`，权重冻结后只评估一次。

不读取 test。分区文件与其 SHA256 持久化。不得依据 holdout 结果改权重、改网格或换 gating；
若失败，任何自适应融合都必须另写 P9-2D 预注册。

## 5. 方法与冻结超参数

对每个用户的 50 个 candidates，分别在用户内标准化：

```text
z_seq = (sequence_score - mean) / max(population_std, 1e-6)
z_cf  = (item_head_score - mean) / max(population_std, 1e-6)
joint = z_seq + lambda * z_cf
```

- item-head history 与 P9-2A validation 完全一致：`items[:-2]`，最多保留最近 20 项；
- item-head 使用冻结 normalized cosine score；不计算 full-catalog softmax；
- 稳定排序 tie-break：原 beam rank 更高者优先；
- lambda 网格固定为 `[0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]`；
- calibration 选择规则：Hit@10 最大，其次 NDCG@10 最大，再次 lambda 最小；
- `lambda=0` 是程序内重算的 baseline identity control；
- 同时报告 pure-CF-within-beam 和 target-in-beam oracle，仅作诊断，不参与选择。

## 6. 指标与科学门

主指标：holdout Hit@10。次指标：NDCG@10、Hit/NDCG@1/5/20/50、MRR；诊断分层：
history 1–5/6–10/11–20 和 target tail/middle/head。

### Gate 1：calibration 准入

- 被选择的 lambda 必须大于 0；
- 相对 lambda=0，calibration `ΔHit@10 >= +0.002`；
- calibration `ΔNDCG@10 >= 0`。

失败则终止为 `failed_calibration_gate`，仍输出完整诊断，但不形成 holdout 改善声明。

### Gate 2：evaluation holdout

同时满足才标记 `passed`：

1. `ΔHit@10 >= +0.002`；
2. 2,000 次按用户 paired bootstrap 的 `ΔHit@10` 95% CI 下界大于 0；
3. `ΔNDCG@10 >= 0`；
4. `Δtail Hit@10 >= 0`；
5. Hit@50 与 baseline 精确相同（容差 `1e-12`）。

CI 仅用于预注册主指标；其余指标不作多重检验后的显著性声明。单 seed 通过只表示机制 pilot
准入，后续仍需多 seed 与重新解码复现。

## 7. 实验产物

- runner：`experiment/phase9/run_phase9_cf0_b3_beamfusion_p2c.sh`；
- evaluator：`experiment/phase9/eval_cf0_b3_beamfusion.py`；
- tests：`experiment/phase9/test_cf0_b3_beamfusion.py`；
- frozen config：`artifacts/phase9/configs/cf0_b3_toys_beamfusion_p2c_preregistered.json`；
- output：`artifacts/phase9/cf0_b3_toys_beamfusion_p2c/`；
- 必需文件：`summary.json`、`per_user.tsv`、`partition.tsv`、`status.json`、`run.log`。

该实验是 CPU-only 离线评测，不停止 GPU6 CodeLlama，hard timeout 固定为 1,800 秒；非零退出、
完整性失败或科学门失败均不自动调参或重试。

## 8. 预期解释边界

- 若通过：支持“协同信号有效，但应在候选排序空间而非 GRAM hidden 空间融合”；
- 若 target-in-beam coverage 低而 conditional rerank 强：瓶颈在生成候选召回；
- 若 pure CF 强而任意混合弱：两路 score calibration/排序偏好仍不兼容；
- 若 calibration 改善但 holdout 失败：视为融合权重过拟合；
- 若全面失败：否定当前 P9-2A head 对 GRAM beam 的 late-fusion 可迁移性，不外推否定所有
  collaborative-generative 机制。

## 9. 实际终态（2026-08-04）

- engineering：completed；integrity gate passed；5/5 tests；CPU wall 8.30 s；
- calibration：lambda=0.75，Hit@10 delta `+0.004639`，NDCG@10 delta `+0.001586`，passed；
- holdout：Hit@10 delta `+0.004179`，bootstrap 95% CI `[+0.001827,+0.006531]`；
- holdout NDCG@10 delta `+0.002436`；Hit@50 delta `0`；
- tail Hit@10 delta `-0.010254`，未通过 safety gate；
- scientific gate：`failed_holdout_gate`。

详细解释见 `report/第九阶段/GRAM_第九阶段_CF0-B3_P9-2C冻结双路BeamFusion结果与验证报告.md`。
