# GRAM 第三阶段晋级记录

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-22
- Verification Status: ANALYZED
- Version Label: phase3_promotion_v2

## S0 → S1

整体决定：**MODIFY，不直接进入 S1**。

| 数据集 | validation NDCG@10 相对变化 | Recall@10 绝对变化 | Tail NDCG@10 相对变化 | Uncovered NDCG@10 相对变化 | 数据集判定 |
|---|---:|---:|---:|---:|---|
| Toys | +1.718% | +0.003245 | +6.575% | -17.271% | GO |
| Beauty | +0.472% | +0.000447 | +1.878% | -8.158% | STOP_OR_MODIFY |

依据：S0 要求 validation NDCG@10 相对提升至少 1% 且 Recall@10 不下降超过 0.5 个百分点，或困难子组相对提升至少 3% 且 overall 不下降。Toys 通过，Beauty 未通过，因此没有形成双数据集一致的 S1 放行证据。

下一步限定为一次 **S0b post-hoc exploratory reliability-abstention probe**：只使用推理时可得的 relation score、support count、score margin 和候选池统计构造拒绝规则；不得使用目标商品或 test；候选网格必须在运行前锁定并同时应用于 Beauty/Toys。若仍不能使 Beauty 达到门槛并控制两个数据集的 uncovered 退化，则停止当前离线重排路径，再决定是否仅以“可学习 reliability gate”作为机制假设进入新的预注册周期。

## S0b → S1

整体决定：**STOP**。这是 post-hoc exploratory amendment；16 个锁定共同配置中有 0 个通过。

诊断最优配置：`b0_l0.2_t0.75_s2`；macro NDCG@10 relative delta=+0.504397%。

## LRC-F0 → LRC-S1

整体决定：**STOP**。这是 UCRF-v1 STOP 后建立的独立预注册周期。

- Toys: pass=False, model=C2_hist_gradient_boosting, AUROC=0.613350, AUPRC lift=1.724817, Brier improvement=0.017758.
- Beauty: pass=False, model=C2_hist_gradient_boosting, AUROC=0.706400, AUPRC lift=3.045203, Brier improvement=0.129915.

## HBTR-B0 → HBTR-B1

整体决定：**GO WITH NOVELTY NARROWING**。B0 只解锁 B1 设计、CPU 单元测试与 <15 分钟正确性 smoke；不解锁 10% pilot 或全量。

- Toys: Recall@10→50 gap=+0.092520，miss@10/hit@50=1,796，tail=683，shared-prefix rate=0.811804，数据门槛全部通过。
- Beauty: Recall@10→50 gap=+0.099361，miss@10/hit@50=2,222，tail=338，shared-prefix rate=0.589109，数据门槛全部通过。
- 文献边界：LOHRec 已覆盖 hierarchy + ranking loss，OneRec/WPAUC 已覆盖 beam hard negatives，Token-Weighted Multi-Target Learning 已覆盖 prefix/frequency weighting。候选差异收缩为“GRAM lexical hierarchy 下，针对 student miss@10/hit@50 errors 的 prefix-depth × training-popularity joint margin”。
- 该差异是 search-bounded candidate claim，不是绝对首创证明；B1/S2 必须通过组件对照证明不是简单叠加。

## HBTR-B1 → HBTR 10% Pilot Design

整体决定：**PASS FOR PILOT DESIGN**。只解锁 pilot 预注册、一次性分层 split 与 C0–C4 实现；
在 pilot 配置与门槛锁定前不启动 GPU。

- CPU: 第三阶段 16/16 单元测试通过；B1 新增 6 项。
- Toys: 100 样本产生 12 个有效 cache rows；2 个优化步通过；30.51 s；peak reserved 15,020 MiB。
- Beauty: 100 样本产生 21 个有效 cache rows；2 个优化步通过；34.87 s；peak reserved 17,982 MiB。
- 两数据集 checkpoint reload max absolute difference 均为 0.0；test 未读，pilot split 未生成。
- 首次 correctness sample selection 未让 Beauty GPU backward 覆盖非平凡权重；产物已保留。
  repair smoke 只修改覆盖选择，不改预注册配置与 cache，最终 Toys 覆盖 joint/prefix，Beauty 覆盖 prefix/tail。
- 本决定是实现正确性决定，不是推荐效果 GO。

## MARC-L0 → MARC-L1

整体决定：**STOP**，固定标签
**`STOP_MARC_NO_UTILITY_HETEROGENEITY`**。L1、MARC-lite 训练、reflection、RL 和
validation/test 均不解锁。

- 首次 scoring 使用 full20 作为 source reference，128-token 截断机械移除 metadata；
  该次为 `EXECUTION_INVALID_SOURCE_REFERENCE`，只保留 lineage，不进入结论。
- 修复后 Toys/Beauty 各 512 training-prefix users；所有 integrity gates 通过，
  GRAM optimizer steps=0。
- Toys collaborative negative utility rate=14.0625%，低于预注册 15%，故 L0-A
  失败；门槛虽只差 0.9375 个百分点，也不作事后修改。
- L0-B 的独立失败包括：两数据集 semantic corruption direction 错误；
  Beauty semantic active coverage=15.48% 且 active utility CI 跨 0；
  Beauty budget regret ratio=0.9406 > 0.75。
- 两数据集 collaborative critic 的 predictability/active utility/corruption
  均通过，只支持未来另立、重新预注册的 CF-only 窄假设，不能救援 unified MARC。
