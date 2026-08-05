# GRAM 第十阶段：CF1 双源候选扩展与生成精排融合升级计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-04
- Verification Status: `PREREGISTERED_ROADMAP`
- Version Label: `phase10_cf1_dual_source_cascade_fusion_v1`
- Parent Evidence: P9-2E frozen PCRF confirmed on one-shot Toys test
- Closed Dataset Boundary: Toys test 已消费，只允许报告历史 P9-2E，不再用于开发或选择
- Immediate Authorized Next Unit: CF1-A candidate-union coverage/oracle（仅 Toys validation）
- Not Authorized by This Plan Alone: CF1-B/C GPU full run、Beauty test、Sports、再次读取 Toys test

## 1. 结论：为什么现在应升级融合

是的，下一阶段应升级融合，但升级重点不应是继续微调 PCRF 的 lambda/beta/gamma。P9-2E 已确认
PCRF 在 GRAM beam50 内有效：Toys test Hit@10 `+0.007573`，tail Hit@10 `+0.006241`；同时
Hit@50 精确不变，固定在 `0.172883`。这说明当前主要上限已经从“如何重排已有 beams”转为：

> **如何把 item-head 找到、但 GRAM beam50 没生成的 relevant items 安全加入候选，再由生成分数
> 与协同分数共同精排。**

第十阶段将 CF0 的 beam-only late fusion 升级为 CF1 dual-source cascade：

```text
GRAM legal beam candidates ─┐
                            ├─ candidate union ─ constrained GRAM path scoring ─ calibrated fusion ─ top-k
CF item retrieval candidates ┘
```

设计原则与 [COBRA（NeurIPS 2025）](https://proceedings.neurips.cc/paper_files/paper/2025/hash/86ba836d4c5dd859d795a172911745e2-Abstract-Conference.html)
的 sparse/dense coarse-to-fine 和 BeamFusion 思路一致，但本计划不声称复现其端到端模型。

## 2. Research questions

- **RQ1 / Coverage**：P9-2A item-head top-k 能否补回 GRAM beam50 之外的 target，特别是 tail target？
- **RQ2 / Scoring**：对 CF-only candidates 计算冻结 GRAM constrained path score，能否与历史 beam
  generation score处于可校准、可比较的尺度？
- **RQ3 / Ranking**：双源 union 在固定预算 top-10/20/50 下，能否超过已确认的 PCRF，同时保持
  tail、top-1 和原 beam candidates 的安全性？
- **RQ4 / Generalization**：冻结的 CF1 融合规则能否在 Beauty 上外部确认？

## 3. Overall staged design

| Unit | Purpose | Compute | Gate consequence |
|---|---|---:|---|
| CF1-A | frozen dual-source coverage/oracle | CPU / item-head | 判断扩候选是否值得 |
| CF1-B | arbitrary legal candidate constrained scoring | GPU pilot → full | 建立可比较生成分数 |
| CF1-C | calibrated union reranking | CPU/GPU mixed | 验证实际 top-k 增益 |
| CF1-D | Beauty external confirmation | train + inference | 跨域确认 |

每个 unit 独立预注册、独立配置、独立终态。前一 gate 失败时停止，不自动跳过或扩大超参数。

## 4. CF1-A：Candidate-union coverage/oracle（立即下一实验）

### 4.1 Data boundary

- 只读 Toys validation：history=`items[:-2]`、target=`items[-2]`；
- GRAM source：历史 validation beam50 cache；
- CF source：P9-2A frozen item-head full-catalog top50；
- train popularity：只由 `items[:-2]` train-prefix interactions；
- 不读取 Toys test；不训练、修改或重新选择 checkpoint。

### 4.2 Candidate sets

对每位用户构造：

- `G50`：GRAM 原 beam top50；
- `C10/C20/C50`：item-head full-catalog top-k；
- `U(k)=unique(G50 ∪ Ck)`，保留 source membership，最大规模分别 60/70/100；
- 排除 history items 的结果只作诊断；主分析保持与历史评测相同的 catalog 定义。

### 4.3 Metrics

主指标：`Coverage(U50)`，即 target 是否出现在 100 以内的 union。

同时报告：

- Coverage(G50)、Coverage(C10/20/50)、Coverage(U10/20/50)；
- complementary coverage：`target ∈ Ck and target ∉ G50`；
- lost-to-intersection、candidate overlap/Jaccard、平均/分位 union size；
- target popularity 与 history length 分层；
- 理论 oracle top-10/top-20/top-50 上限；
- 需要新增 constrained scoring 的 CF-only candidate 数量与估算算力。

### 4.4 Frozen CF1-A gate

同时满足才进入 CF1-B：

1. 所有 users/candidates/item IDs 完整映射，input/checkpoint hash 不漂移；
2. `Coverage(U50) - Coverage(G50) >= +0.030`；
3. tail complementary coverage@50 `>= +0.020`；
4. 至少 80% 的 union users 候选数不超过 90，避免近乎完全不重叠导致预算失控；
5. CF top50 coverage 与 P9-2A validation Recall@50 `0.174634` 在 `1e-12` 内一致。

Gate 失败时保留 PCRF 为第九阶段终版，不开发 CF1-B。Gate 通过只说明有 candidate recall 空间，
不等于最终排序会提升。

### 4.5 CF1-A outputs

- plan：本文件；
- evaluator：`experiment/phase10/eval_cf1_a_candidate_union.py`；
- tests：`experiment/phase10/test_cf1_a_candidate_union.py`；
- config：`artifacts/phase10/configs/cf1_a_toys_candidate_union_preregistered.json`；
- outputs：`artifacts/phase10/cf1_a_toys_candidate_union/summary.json`、
  `per_user_coverage.tsv`、`status.json`、`run.log`；
- proposed entry：`bash experiment/phase10/run_phase10_cf1_a_candidate_union.sh start`；
- CPU-only，hard timeout 1,800 s，预期小于 2 分钟。

## 5. CF1-B：Arbitrary-candidate constrained GRAM scoring（Gate A 后另行冻结）

CF-only items 没有原 beam sequence score，不能用 0、最小值或 source indicator 粗暴代替。CF1-B
将实现合法 lexical-ID path 的逐 token constrained log-probability：每一步只在 trie 允许 token 中
归一化，保持与 constrained beam search 的评分定义一致。

### Gate B0：score identity

先只重算历史 G50：

- gold lexical path、EOS、length penalty 与生成配置一致；
- recomputed 与 cached score Pearson/Spearman 均 `>=0.995`；
- 每用户 top-10 set agreement `>=0.98`；
- 重算 baseline Hit@10 absolute delta `<=0.001`。

若 exact identity 不成立，先解释 generation score定义，不得对 CF-only candidate 启动 full scoring。

### Gate B1：resource pilot

- 512 users × U50 same-batch pilot；
- finite score 100%，无 illegal path；
- peak GPU memory 不超过预注册租约；
- 基于实测吞吐冻结全量 timeout；
- CodeLlama/GPU6 资源释放与恢复协议沿用第九阶段。

## 6. CF1-C：Calibrated union reranking（Gate B 后另行预注册）

### 6.1 Baselines

1. GRAM beam50；
2. 已确认 frozen PCRF `(1.0,0.5,1.0)`；
3. pure CF retrieval；
4. source-agnostic sum；
5. CF1 calibrated union fusion。

### 6.2 Proposed features

仅使用线上可观测特征：

- constrained GRAM path score；
- popularity-corrected item-head score；
- PCRF reliability / tail_mass；
- source membership（GRAM、CF、both）；
- two-source rank、score margin、agreement；
- item train frequency。

不使用 target frequency、gold rank、hit label 作为 inference feature。第一版优先使用受约束的线性/
单调 calibrator；只有线性版本无法表达 source calibration 时才另行预注册小型 MLP，不在同一结果后
自动升级模型容量。

### 6.3 Development gate

Toys validation 使用 user-level 5-fold cross-fitting；Toys test 永久关闭。相对 frozen PCRF：

- OOF Hit@10 delta `>= +0.003`；
- OOF Hit@50 delta `>= +0.020`；
- tail Hit@10 delta `>= 0`；
- Hit@1 delta `>= -0.001`；
- paired Hit@10 bootstrap lower > 0；
- 五折至少四折主方向为正，参数/权重无单折崩塌。

如果 coverage 上升但 ranking gate 失败，结论应是“候选有价值但 scorer/calibrator 不成熟”，不能
退回 Toys test 调参。

## 7. CF1-D：Beauty external confirmation

CF1-C 通过后：

1. 在 Beauty train-prefix 单独训练同规格 item-head；
2. 仅以 Beauty validation 验收 item-head 独立可学性，不修改 CF1 fusion 参数；
3. 生成或复用 Beauty legal beams，并做 score identity；
4. 冻结 Toys 学到的 CF1 fusion，在 Beauty test 一次性确认；
5. 主门保持 Hit@10/NDCG@10、tail safety、Hit@1 safety，同时要求 candidate coverage 有正增量。

Beauty item-head 若未超过 popularity baseline，则停止，不用 Toys item embedding 跨 catalog 硬迁移。

## 8. Reproducibility track

性能升级之外保留一个正交工程检查：用原 epoch-30 checkpoint fresh decode 固定 512-user subset，
验证 legal candidate set、score order 和 PCRF output 对历史 cache 的复现。该 track 不参与 CF1 参数
选择；若发现版本/数值漂移，应在 Beauty confirmation 前解决。

## 9. Statistical and integrity boundaries

- Toys test 在 P9-2E 后关闭；所有第十阶段 Toys 工作仅是 validation development；
- CF1-A 是 coverage/oracle，不把 oracle coverage 写成实际 recommender performance；
- CF1-C 的多 baseline 以 frozen primary comparison 为准，其余为诊断，避免多重比较挑最好；
- 所有分层事先固定为 history 1–5/6–10/11–20 和 target tail/middle/head；
- 每阶段报告工程状态、科学 gate、资源恢复三类状态；
- 不因 gate failure 自动 retry、扩网格、换 metric 或改 margin；
- checkpoint、cache、mapping、code 与 partition 均 SHA256 锁定。

## 10. Decision summary

当前正确路线不是“继续把 PCRF 从 +0.0076 调到更高”，而是：

1. 先用 CF1-A 定量确认 item-head 是否补回 beam 外 target；
2. 若补回，建立 exact constrained scorer；
3. 再学习受约束的双源 calibration；
4. 最后在 Beauty 做一次性外部确认。

这把融合从 **beam-only reranking** 提升为 **retrieve → union → generative score → calibrated rank**，
同时保留 P9-2E 已确认的 PCRF 作为不可回退的安全基线。

## 11. CF1-A execution addendum（2026-08-04）

CF1-A 已按冻结配置单次完成，工程状态 `completed`，联合科学 gate 状态
`failed_candidate_union_gate`。详细证据见
`report/第十阶段/GRAM_第十阶段_CF1-A双源候选覆盖率与Oracle结果报告.md`。

- coverage hypothesis：通过。G50 `0.211931` → U50 `0.266691`，delta `+0.054760`；
- tail complementarity：通过。C50-not-G50 `0.023450`；
- candidate budget：失败。union size `<=90` fraction 仅 `0.367917`；
- exact CF identity：失败。CPU top-k 相对冻结 CUDA rank 净差 2/19,412 hits；按原规则保留 FAIL；
- Toys test 未读取，CPU wall `7.73 s`，无自动 retry。

因此原 CF1-B 的 full U50 scoring 暂停。下一次实验先插入 **CF1-A2 budgeted adaptive union**：
固定总候选硬上限 90，在不读取 test 的前提下比较 fixed CF slots 与基于 history/reliability 的自适应
slots，目标是在保留大部分 U50 oracle gain 的同时满足 100% 用户预算约束，并显式保护 tail
complementary targets。A2 通过后才恢复 CF1-B constrained scorer pilot。

## 12. CF1-A2 execution addendum（2026-08-04）

CF1-A2 已按冻结 primary `fill_cf_only_40` 完成并通过全部 gate：

- 100% 用户候选数 `<=90`，mean `87.52`；
- coverage `0.264733`，相对 G50 增益 `+0.052802`；
- 保留 raw U50 coverage gain 的 `96.43%`；
- tail complementary `0.022093`，保留率 `94.21%`；
- CF-only scoring 总量从 791,057 降到 728,305；
- Toys test/Beauty/Sports 未读取，CPU wall `8.93 s`。

因此 CF1-A 阶段的 coverage 与 budget 两类前置条件均已完成。恢复第 5 节 CF1-B，但 scorer pilot
的候选输入由 raw U50 更新为冻结的 `fill_cf_only_40`，每用户硬上限 90；其余 score identity 和
resource gates 不变。

## 13. CF1-B0 execution addendum（2026-08-04）

静态代码审计修正了第 5.2 节的 identity 定义：Transformers 4.26 cache score 是全词表
log-softmax 下的 token path score，不是 allowed-token renormalized score。64-user cached-G50
teacher-forced identity 已通过：

- 3,200 paths 全部 finite；
- Pearson `0.9999996844`、Spearman `0.9999986674`；
- mean top-10 set overlap `1.0`；
- cached/recomputed Hit@10 均 `0.09375`；
- peak allocated `1,578.83 MiB`，wall `15.49 s`。

第 5 节 Gate B0 已完成。下一授权单元为 B1：512-user `fill_cf_only_40` arbitrary-candidate
resource pilot，并在同一运行保留 cached G50 identity sentinel。

## 14. CF1-B1 execution addendum（2026-08-04）

B1 已在 512 deterministic validation users 上完成并通过全部 gate：

- 44,730 candidates，其中 19,130 CF-only；全部 legal、finite；
- union mean `87.36`、max `90`；
- G50 Pearson `0.9999996714`、Spearman `0.9999992863`；
- top-10 set overlap `0.999805`，cached/recomputed Hit@10 完全一致；
- peak allocated `1,578.83 MiB`，wall `106.06 s`；
- throughput `421.73 candidates/s`，projected full validation `1.117 h`。

启动后 GPU5 出现外部资源竞争，但全部冻结资源门仍通过；吞吐只作为共享 GPU 下的保守测量。
下一授权单元为 CF1-B2 full Toys-validation scoring。B2 只生成冻结 candidate-level score artifact，
不训练/选择融合器；完成后才进入 CF1-C cross-fitted calibration。

## 15. CF1-B2 execution addendum（2026-08-04）

CF1-B2 已在完整 19,412-user Toys validation 上单次完成并通过全部冻结 gate：

- 1,698,905 candidates，其中 728,305 CF-only；union mean `87.518`、max `90`；
- legal path、finite score、有效预算均为 `100%`；
- G50 Pearson `0.9999996652`、Spearman `0.9999993113`、top-10 overlap `0.999542`；
- cached/recomputed Hit@10 绝对差 `0.000103`；
- peak allocated `1,578.83 MiB`，wall `2,797.03 s`，吞吐 `607.40 candidates/s`；
- candidate-score artifact 恰好 1,698,906 行（含表头），SHA256 与 summary 一致；
- CodeLlama 已恢复至物理 GPU6，Toys test/Beauty/Sports 未读取。

CF1-B 至此结束。下一授权单元为 CF1-C0 feature-table 与 baseline identity audit；C0 不拟合融合器，
通过后才执行 CF1-C1 五折 cross-fitted monotone listwise calibration。详细计划见
`plan/第十阶段/GRAM_第十阶段_CF1-C跨折校准融合实验计划.md`。

## 16. CF1-C0 execution addendum（2026-08-04）

CF1-C0 已完成并通过全部 11 项冻结 identity/completeness/leakage gate：

- 19,412 users、1,698,905 candidates、728,305 CF-only，B2 逐行与 SHA256 身份一致；
- 无重复 user-candidate，全部特征 finite，union budget 100% 合法；
- 五折计数 `[3883,3883,3882,3882,3882]`，target/gold/label 不在 inference schema；
- frozen PCRF Hit@10/50 为 `0.125335/0.211931`；
- naive source-agnostic sum Hit@10/50 为 `0.116423/0.221255`；
- union oracle coverage 为 `0.264733`，仍保留明显排序可开发空间；
- CPU wall `40.21 s`，Toys test/Beauty/Sports 未读取。

结论是候选互补成立但直接相加无法校准双源分数。下一授权单元为 CF1-C1 五折 cross-fitted
monotone listwise calibration，主比较固定为 frozen PCRF。

## 17. CF1-C1 execution addendum（2026-08-04）

CF1-C1 已单次完成，工程完整性通过但 development gate 失败：

- frozen PCRF → C1 OOF Hit@10：`0.125335 → 0.125026`，delta `-0.000309`；
- Hit@10 paired bootstrap 95% CI `[-0.002113,+0.001597]`，仅 1/5 折为正；
- Hit@50：`0.211931 → 0.226561`，delta `+0.014630`，五折全部为正但低于 `+0.020` gate；
- tail Hit@10 delta `-0.004845`；Hit@1 safety 通过；
- 五折全部收敛、OOF finite、fold isolation 与 train-only scaling 通过；
- source-level Hit@10 net：GRAM-only `-134`、both `+107`、CF-only `+21`；
- Toys test/Beauty/Sports 未读取。

CF1-D Beauty 未授权。下一步固定为 CF1-C2 PCRF-anchored source-asymmetric safe insertion；C2 不
降低任何 C1 gate，只允许一个 preregistered primary。若 C2 再次出现 top-50 gain 但 Hit@10/tail
failure，则停止 Toys CF1 calibration，保留 frozen PCRF，不再增加模型容量。

## 18. CF1-C2 implementation smoke addendum（2026-08-04）

PCRF-anchored source-asymmetric C2 已完成 smoke-only preregistration 与 512-user implementation audit：

- 12/12 implementation checks 通过，五折全部收敛；
- PCRF baseline identity、anchor order、CF-only floor、fold isolation、finite 与 residual bound 均通过；
- evaluator wall `9.60 s`，无异常、无 retry；
- Toys test/Beauty/Sports 未读取；
- smoke metric 只作管线诊断，未用于改模型、loss 或 gate。

当前正式 C2 仍被 config 禁用。下一独立授权单元为锁定 full-OOF timeout 与正式 runner/config SHA256，
随后单次执行 19,412-user 五折；CF1-D Beauty 继续未授权。

## 19. CF1-C2 formal execution and CF1 closure（2026-08-04）

CF1-C2 formal full OOF 已完成但未通过 development gate：

- frozen PCRF → C2 Hit@10：`0.125335 → 0.124768`，delta `-0.000567`；
- Hit@10 bootstrap 95% CI `[-0.001648,+0.000515]`，仅 2/5 折正向；
- Hit@50：`0.211931 → 0.220122`，delta `+0.008191`，低于 `+0.020`；
- tail Hit@10 `+0.005814`、Hit@1 `+0.000103`，安全门通过但不足以替代主 gate；
- CF-only 有 366 个净新增 Hit@50，但 0 个新增 Hit@10；
- 五折全部收敛，identity/finite/isolation 通过，Toys test/Beauty/Sports 未读取。

CF1 在 Toys validation 上正式关闭。第十阶段最终保留 frozen PCRF `(1.0,0.5,1.0)`；CF1-D Beauty
取消，不继续调 residual cap、模型容量、seed、loss 或 gate。任何后续 retrieval/ranking 假设必须作为
新的独立阶段重新预注册。
