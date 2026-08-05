# GRAM 第十阶段：CF1-C2 锚定 PCRF 的跨折安全插入实验计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-04
- Version Label: `code_plan_v1`
- Verification Status: `UNVERIFIED`
- Parent Plan: `GRAM_第十阶段_CF1-C跨折校准融合实验计划.md`
- Upstream Gate: CF1-C1 `FAILED_DEVELOPMENT_GATE`
- Evidence Class: post-C1 validation development, not independent confirmation
- Toys Test/Beauty/Sports Read: false

## 1. Research question

在候选集合、GRAM arbitrary-candidate score、item-head、PCRF 参数、五折和原 development gate 均
保持冻结的条件下，PCRF-anchored、source-asymmetric 的 residual insertion ranker 能否保留 C1 对
`both`/CF-only target 的补回能力，同时避免 GRAM-only 与 tail target 被错误挤出 top-10？

C2 检验的是安全插入/目标函数设计，不重新选择候选预算，不修改生成模型，不读取 Toys test，也不
把 C1 的 post-hoc 诊断当作独立证据。

## 2. Motivation fixed before implementation

C1 相对 frozen PCRF：

- Hit@10 `-0.000309`，162 gain / 168 loss；
- Hit@50 `+0.014630`，534 gain / 250 loss，五折方向全部为正；
- tail Hit@10 `-0.004845`；
- source decomposition：GRAM-only `-134`、both `+107`、CF-only `+21` net Hit@10；
- 五折 `item_log_frequency` 系数全为正，且收益偏向 middle/head；
- 只兑现 union-oracle Hit@50 gap 的 `27.71%`。

因此 C2 不采用“完全冻结 G50 顺序”，也不采用无锚定的通用 MLP。Primary 固定为一个带 PCRF
anchor、source-specific residual 和非对称安全代价的可解释 ranker。

## 3. Frozen inputs and isolation

| input | frozen path | role |
|---|---|---|
| feature table | `artifacts/phase10/cf1_c0_toys_feature_audit/feature_table.npz` | candidates/features/folds |
| C0 summary | `artifacts/phase10/cf1_c0_toys_feature_audit/summary.json` | identity gate |
| C1 summary | `artifacts/phase10/cf1_c1_toys_crossfit_calibrator/summary.json` | failed upstream result |
| C1 diagnostic | `artifacts/phase10/cf1_c1_error_decomposition/summary.json` | mechanism motivation only |
| cached validation beams | `GRAM/preds/20260722_020042_Toys_sequential_pred_validation.tsv` | frozen PCRF anchor |
| item-head checkpoint | frozen P9 checkpoint referenced by C0 | no retraining |

实施前 preregistration JSON 必须锁定以上文件、C2 evaluator、tests、runner 与本计划的 SHA256。五折
继续使用 C0 assignment 与 seed `2023`；任何 normalization、threshold 或模型参数只能由对应四个
training folds 估计。

## 4. Primary model: PCRF-anchored asymmetric residual insertion

### 4.1 Anchor

- 对原 G50 candidate，base score 使用 frozen PCRF `(1.0,0.5,1.0)` 的 user 内标准化 score；
- 对 CF-only candidate，base score 固定为该用户 PCRF rank-50 score，不把缺失 PCRF score伪装为 0；
- 最终 score 为 `anchor + bounded source-specific residual`；
- residual 使用 `tanh` 限幅，幅度固定为 `1.0` 个 user-standardized PCRF score unit，不通过
  OOF metric 搜索；
- `gram_score` 与 corrected item-score 的主方向保持非负；raw `item_log_frequency` 不作为无约束正向
  主效应，只允许进入 popularity-correction 或受限的非正 residual。

### 4.2 Source-asymmetric residual

使用共享主效应加三类固定 source branch：`gram_only`、`both`、`cf_only`。branch 只使用 C0 已冻结
的线上特征：GRAM/item score、source ranks、agreement、reliability、history bucket 与 item train
frequency。禁止使用 target frequency、gold source、gold rank、hit label 或 OOF metric 作为 inference
feature。

`both` 与 CF-only 可以在 residual 证据充分时晋升；GRAM-only 不被永久锁死，但其 PCRF anchor
不得被无界 residual 覆盖。

## 5. Frozen training objective

每折只使用其他四折中 target-in-union users，优化以下单一复合目标；各项系数必须在运行前写入
preregistration，不根据 OOF 结果修改：

1. `rank-aware supervised loss`：以 gold candidate 对同 user candidates 的 pairwise logistic loss 为主，
   pair weight 固定为 `|delta NDCG@10| + 0.25 * |delta NDCG@50|`；
2. `PCRF top-10 safety loss`：对 frozen PCRF top-10 中非 gold candidate 的相对次序做蒸馏/锚定，
   safety loss coefficient 固定为 `0.25`；对训练用户中“原 PCRF gold hit@10 被挤出”对应 pair
   施加固定 `2.0` 倍非对称 retention penalty；
3. `popularity-balanced supervision`：按训练折中冻结的 tail/middle/head 正样本数使用 inverse-frequency
   权重并归一为均值 1；target popularity 只决定 training-label weight，不进入 inference；
4. `L2`：固定 `1e-3`；
5. 不运行 loss-weight、capacity、seed、cutoff 或 feature-subset 网格。

优化器固定为 L-BFGS-B，`max_iter=200`、`ftol=1e-9`、`gtol=1e-6`、`maxls=30`。paired
bootstrap 固定 2,000 replicates、seed 2023。上述常数均在 implementation smoke 前冻结。

实现审计必须证明 loss 的每一项只依赖训练 folds；留出折只生成一次 OOF rank。

## 6. Frozen comparisons and metrics

Primary comparison 保持 C2 OOF vs frozen PCRF。C1 OOF 只作为解释性历史结果，不成为可挑选的第二
primary。报告：

- overall Hit@1/10/50、NDCG@10/50；
- history 1--5、6--10、11--20；
- target popularity tail/middle/head；
- gold source GRAM-only/both/CF-only/union-miss；
- Hit@10 gain/loss transition 与 entrant source；
- paired user bootstrap 2,000 replicates、seed 2023；
- 五折 Hit@10/50 delta、参数、收敛与 residual saturation rate。

## 7. Development gate

不降低 C1 原门槛，全部满足才通过：

- OOF Hit@10 delta `>= +0.003`；
- OOF Hit@50 delta `>= +0.020`；
- tail Hit@10 delta `>= 0`；
- Hit@1 delta `>= -0.001`；
- paired Hit@10 bootstrap 95% CI lower `> 0`；
- 五折至少四折 Hit@10 delta 为正；
- 全部 OOF scores/ranks finite，五折训练均收敛；
- 所有 normalization/model/loss-weight 参数只由对应 training folds 或 preregistration 决定；
- Toys test、Beauty、Sports 未读取。

Source-level transition、NDCG 与 saturation 只作诊断，不替代 gate。

## 8. Stop/go logic

- implementation/identity audit FAIL：修复代码或身份问题，不训练正式 C2；
- C2 全部 gate PASS：冻结算法规格，进入 CF1-D Beauty external confirmation；
- C2 Hit@50 上升但 Hit@10 或 tail gate FAIL：停止 Toys 上 CF1 calibration，保留 frozen PCRF；
- C2 Hit@50 也无提升：停止 CF1，不回调 A2/B2，不读取 Toys test；
- 不自动 retry、换 seed、扩模型、增加第二个 primary、降低门槛或挑选 subgroup 胜者。

## 9. Expected artifacts

### Implementation audit

- `artifacts/phase10/configs/cf1_c2_toys_pcrf_anchored_preregistered.json`；
- evaluator、tests、runner；
- identity test：PCRF anchor exact、fold isolation、loss train-only、target absent from inference；
- smoke summary；
- code/config/inputs SHA256。

### Formal C2 run

- `per_user_oof.tsv`；
- `fold_models.json`；
- `fold_metrics.json` 或 summary 内等价字段；
- `hit10_transitions.tsv`；
- `summary.json`；
- 结果报告与下一步决策。

## 10. Monitoring and resource boundary

- Type: CPU optimization / analysis；若实现需要 GPU，必须在 preregistration 前另做资源 smoke；
- timeout 在 smoke 后冻结，正式运行只执行一次；
- runner 必须记录 process-alive、log progress、wall time 与 exit status；
- crash/timeout 不自动 retry；
- 不启动或停止与 C2 无关的模型服务。

## 11. Immediate authorization

当前只授权编写 C2 evaluator、unit tests、runner 和 preregistration config，并执行不超过 512 users 的
implementation smoke/identity audit。只有这些检查通过、所有 SHA256 与固定 loss coefficients 写入 config
后，才允许启动正式五折 C2。当前不授权读取 Toys test、Beauty、Sports 或执行 CF1-D。

## 12. Implementation smoke addendum（2026-08-04）

CF1-C2 evaluator、4 项 unit tests、smoke-only runner 与 preregistration 已完成；512-user deterministic
smoke 单次运行在 9.60 秒内通过全部 12 项 implementation gate：

- five-fold counts `[93,122,106,93,98]`，五折均在 31--38 iterations 收敛；
- PCRF baseline rank 与 C1 frozen per-user artifact exact identity；
- anchor finite、PCRF order preserved、CF-only rank-50 floor exact；
- OOF score finite 100%，max absolute residual `0.999258 <= 1.0`；
- `|residual|>=0.95` candidate fraction `0.006595`，未见大面积 saturation；
- train-only scaling/popularity weight 与 inference schema audit 通过；
- Toys test、Beauty、Sports 未读取。

Smoke Hit@10/50 delta `-0.001953/+0.009766` 仅为 implementation diagnostic，不应用正式 development
gate，也不据此修改任何常数。当前 config 仍固定 `formal_execution_enabled=false`；正式 19,412-user
五折尚未启动。

## 13. Formal full-OOF authorization（2026-08-04）

用户在 smoke gate 通过后明确授权继续。正式 C2 保持第 4--7 节全部模型、loss、feature、seed 与 gate
不变，仅新增 residual saturation rate 与 source subgroup 的观测字段；这些字段不进入训练或 ranking。
更新后的 evaluator 必须先在同一 512-user subset 上复现已冻结 smoke 的 `per_user_oof.tsv`、fold models、
fold metrics 与 transitions exact identity。

正式运行冻结为：

- users：完整 19,412 Toys validation users；
- folds/seed：C0 frozen five-fold assignment / 2023；
- bootstrap：2,000 / 2023；
- hard timeout：3,600 秒；
- runner：persistent tmux，30 秒级 status/log 检查；
- formal execution：单次授权，不自动 retry；
- Toys test、Beauty、Sports：继续禁止读取。

更新后的 evaluator、tests、runner、plan 与全部输入必须写入独立 formal preregistration config 并锁定
SHA256；formal config 与 smoke config 分离，保留 smoke 执行记录。

## 14. Formal execution addendum（2026-08-04）

CF1-C2 已按 formal preregistration 单次完成，工程状态 `completed`、implementation gate `passed`，
development gate `failed_development_gate`：

- Hit@10 delta `-0.000567`，51 gain / 62 loss，bootstrap 95% CI
  `[-0.001648,+0.000515]`；
- Hit@50 delta `+0.008191`，净增 159，低于 `+0.020` gate；
- tail Hit@10 delta `+0.005814`、Hit@1 delta `+0.000103`，两项 safety 通过；
- Hit@10 仅 2/5 折为正；五折全部收敛、score finite、train-only isolation 通过；
- source Hit@10 net：GRAM-only `+8`、both `-19`、CF-only `0`；
- CF-only Hit@50 净增 366，但没有 CF-only target 进入 top-10；
- evaluator wall `75.33 s`，无 experiment retry，Toys test/Beauty/Sports 未读取。

按第 8 节 stop logic，CF1 calibration 在 Toys validation 上停止。保留 frozen PCRF，不进入 Beauty，
不增加 residual cap、模型容量、seed 或第二个 primary，也不降低原 development gate。
