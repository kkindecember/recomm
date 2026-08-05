# GRAM 第十阶段：CF1-C 跨折校准融合实验计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-04
- Parent Plan: `GRAM_第十阶段_CF1双源候选扩展与生成精排融合升级计划.md`
- Upstream Gate: CF1-B2 `PASSED`
- Evidence Class: validation cross-fitted development, not independent confirmation
- Toys Test/Beauty/Sports Read: false

## 1. Research question

在候选预算固定为 `fill_cf_only_40`、GRAM arbitrary-candidate score 已完成身份验证的条件下，能否
仅使用线上可观测特征，把 CF-only 补回的目标稳定提升到 top-50/top-10，同时不破坏 frozen PCRF
已经取得的 Hit@1、tail 和 Hit@10 安全性？

本阶段检验的是融合设计，不再修改候选集合、item-head checkpoint、GRAM checkpoint 或 path-score
定义。

## 2. Phase split

### CF1-C0：feature-table and baseline identity audit

C0 不训练、不选择参数，只生成并审计 candidate-level feature table。必须确认：

- 恰好 19,412 users、1,698,905 candidates、每用户 50--90 candidates；
- B2 `gram_score`、source、union rank 和 candidate SHA256 身份一致；
- 为所有候选补齐 item-head score、item train frequency、GRAM/CF membership 与 source ranks；
- history length、PCRF tail_mass/reliability 均只由 train prefix 与候选产生；
- target label 只用于训练目标和评测，不进入 inference feature；
- G50 GRAM baseline 与 validation cache 指标 exact/容差身份一致；
- frozen PCRF `(1.0,0.5,1.0)` 与 P9 validation implementation 身份一致；
- pure-CF、source-agnostic sum 与 union-coverage oracle 可重算；
- 所有特征 finite，无重复 `(user,candidate)`，fold assignment 固定且五折用户不交叉。

C0 任一身份或泄漏检查失败，则停止，不进入拟合。

### CF1-C1：5-fold cross-fitted monotone listwise calibration

仅在 C0 通过后运行。按 user ID 与固定 seed `2023` 建立五折；每一折只用其他四折拟合，在留出折
产生 OOF rank。五折 OOF 合并后只做一次冻结 gate 判定。

## 3. Frozen candidate features

第一版 calibrator 只使用：

1. user 内标准化 constrained `gram_score`；
2. user 内标准化 popularity-corrected item-head score；
3. `in_gram`、`in_cf`、`both` source indicators；
4. GRAM 与 CF reciprocal rank（缺失置 0）；
5. two-source rank/score agreement；
6. PCRF `tail_mass` 与 `reliability=(1-tail_mass)`；
7. `reliability × corrected_item_score`；
8. log item train frequency 与 history-length bucket。

禁止使用 target frequency、gold rank、hit label、test statistics 或 fold evaluation metric 作为 inference
feature。连续特征的中心/尺度只能在每折训练用户上拟合，再应用于该折留出用户。

## 4. Primary model

Primary 为带 L2 的线性 listwise softmax ranker：每个用户的 gold candidate 为正类，候选集合内做
grouped cross-entropy。GRAM 与 corrected item-score 主效应施加非负约束；其余 source/calibration
系数可自由但受 L2 约束。固定 optimizer、正则、最大迭代数和收敛容差，不根据 OOF 指标搜索网格。

这一模型相当于受约束的可解释双源校准器。小型 MLP、LambdaMART、MoE/gating 或更大特征交互不在
本轮自动启用；若 C1 显示明确候选价值但线性表达不足，再另行预注册 C2。

## 5. Frozen comparisons

1. GRAM beam50；
2. frozen PCRF `(1.0,0.5,1.0)`，primary baseline；
3. pure CF retrieval；
4. source-agnostic standardized sum；
5. CF1-C1 OOF calibrated union，primary candidate。

多 baseline 只作诊断；科学 gate 的主比较固定为 CF1-C1 vs frozen PCRF。

## 6. Metrics and subgroups

- overall：Hit@1/10/50、NDCG@10/50；
- history：1--5、6--10、11--20；
- target popularity：tail/middle/head，边界只由 train-prefix frequency 冻结；
- source recovery：gold 为 G50、CF-only、both 或 union-miss；
- paired user bootstrap：Hit@10 delta，2,000 replicates，seed `2023`；
- fold stability：逐折 Hit@10/50 delta 和模型系数。

## 7. C1 development gate

相对 frozen PCRF，全部满足才通过：

- OOF Hit@10 delta `>= +0.003`；
- OOF Hit@50 delta `>= +0.020`；
- tail Hit@10 delta `>= 0`；
- Hit@1 delta `>= -0.001`；
- paired Hit@10 bootstrap 95% CI lower `> 0`；
- 五折至少四折 Hit@10 delta 为正；
- 全部 OOF scores/ranks finite，五折训练均收敛；
- 任一标准化/模型参数只由对应 training folds 估计。

NDCG@10/50、pure CF 与 source-agnostic sum 作为解释性指标报告，不替代主 gate。

## 8. Stop/go logic

- C0 FAIL：修复 feature identity/数据泄漏问题，不运行 C1；
- C0 PASS、C1 PASS：冻结 Toys 学到的算法规格，进入 CF1-D Beauty external confirmation；
- C0 PASS、Hit@50 明显提升但 Hit@10 gate FAIL：候选有价值、线性校准不足，另行设计 CF1-C2；
- Hit@50 也未提升：检查训练目标/候选 source 分布，但不读取 Toys test、不回调 A2/B2；
- 不因失败自动 retry、扩特征、换随机种子、改 bootstrap 或降低门槛。

## 9. Expected artifacts

### C0

- `feature_table.tsv` 或等价分片格式；
- `fold_assignments.tsv`；
- `baseline_metrics.json`；
- `summary.json`（identity/leakage/completeness gates 与 SHA256）。

### C1

- `per_user_oof.tsv`；
- `fold_models.json`；
- `fold_metrics.json`；
- `summary.json`（主 gate、subgroups、bootstrap、资源）；
- 结果报告与下一步决策。

## 10. Immediate authorization

当前只授权 CF1-C0。C0 通过后才允许编写并执行 C1 正式 calibrator；C0 不读取 Toys test、Beauty
或 Sports，不训练/选择融合参数。

## 11. CF1-C1 execution addendum（2026-08-04）

CF1-C0 与 C1 均已按冻结规格完成。C0 通过全部 identity/completeness/leakage gate；C1 工程状态
`completed`，但 development gate 为 `failed_development_gate`：

- Hit@10 delta `-0.000309`，bootstrap 95% CI `[-0.002113,+0.001597]`；
- Hit@50 delta `+0.014630`，低于 `+0.020` 门槛，但五折方向全部为正；
- tail Hit@10 delta `-0.004845`，Hit@1 delta `+0.000361`；
- Hit@10 仅 1/5 折为正；五折全部收敛、OOF finite、train-only scaling 通过；
- Toys test、Beauty、Sports 未读取。

只读 source decomposition 显示 Hit@10 net transitions 为 GRAM-only `-134`、both `+107`、CF-only
`+21`。因此不能进入 Beauty，也不重跑 C1；下一授权单元为另行预注册的 PCRF-anchored、
source-asymmetric CF1-C2。C2 沿用原 gate；若再次只提升 top-50 而不能保护 Hit@10/tail，则停止
Toys CF1 calibration 并保留 frozen PCRF。
