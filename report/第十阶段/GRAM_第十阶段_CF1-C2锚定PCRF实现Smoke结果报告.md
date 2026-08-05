# GRAM 第十阶段：CF1-C2 锚定 PCRF 实现 Smoke 结果报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-08-04
- Verification Status: `UNVERIFIED`（implementation smoke，非科学验证）
- Version Label: `exp_result_v1`
- Experiment ID: `GRAM_PHASE10_CF1_C2_TOYS_PCRF_ANCHORED_SMOKE_V1`
- Evidence Class: implementation smoke, not scientific evidence
- Toys Test/Beauty/Sports Read: false

## 1. Experiment result

- Type: CPU optimization / analysis smoke
- Status: `completed`
- Command: `bash experiment/phase10/run_phase10_cf1_c2_pcrf_anchored.sh smoke`
- Working Directory: repository root
- Users: 512 deterministic validation users, seed 2023
- Duration: 9.60 seconds evaluator wall time
- Exit Code: 0
- Implementation Gate: `PASSED`（12/12）
- Hard Timeout: 900 seconds, not reached
- Anomalies Detected: none

## 2. Frozen implementation identity

Primary 固定为 `PCRF_anchored_source_asymmetric_bounded_residual`：

- PCRF anchor 先按 user 标准化；
- CF-only anchor 等于该 user PCRF rank-50 floor；
- residual 为 `1.0 * tanh(Xw)`；
- supervised pair weight 为 `|delta NDCG@10| + 0.25|delta NDCG@50|`；
- frozen PCRF top-10 safety coefficient `0.25`；
- original PCRF gold hit@10 retention multiplier `2.0`；
- popularity group 使用每折 training-positive users 的 inverse-frequency weight；
- L2 `1e-3`，L-BFGS-B `max_iter=200`；
- raw item frequency 只能通过 `negative_item_log_frequency_z` 的非负系数进入，等价于对 raw
  popularity 施加非正方向约束；
- target frequency 只用于 training-label weight，不进入 inference schema。

## 3. Implementation gate

| check | observed | status |
|---|---|---|
| smoke users <= 512 | 512 | PASS |
| five frozen folds present | `[93,122,106,93,98]` | PASS |
| baseline rank identity with C1 | exact | PASS |
| all anchor scores finite | true | PASS |
| PCRF order preserved after standardization | exact | PASS |
| all CF-only anchors equal rank-50 floor | true | PASS |
| all folds converged | 5/5 | PASS |
| all OOF scores finite | 100% | PASS |
| residual bound respected | max `0.999258 <= 1.0` | PASS |
| train-only scaling/popularity weights | true | PASS |
| target absent from inference feature schema | true | PASS |
| Toys test/Beauty/Sports unread | true | PASS |

五折分别在 31--38 iterations 收敛；training-positive users 为 `[114,99,104,113,114]`。所有折的
training data 均包含 tail/middle/head positive examples。

额外只读 saturation audit 显示，在 44,730 个 smoke candidate residual 中，`|residual|>=0.95`
占 `0.6595%`，`>=0.99` 占 `0.0402%`。虽然最大值接近 cap，但没有出现大面积饱和；该结果只用于
检查数值行为，不授权修改 cap。

## 4. Diagnostic-only smoke metrics

| metric | PCRF baseline | C2 smoke OOF | delta |
|---|---:|---:|---:|
| Hit@1 | 0.046875 | 0.050781 | +0.003906 |
| Hit@10 | 0.123047 | 0.121094 | -0.001953 |
| Hit@50 | 0.208984 | 0.218750 | +0.009766 |
| NDCG@10 | 0.084175 | 0.084336 | +0.000162 |
| NDCG@50 | 0.102963 | 0.105381 | +0.002418 |

这些数值来自为身份/收敛而固定的 512-user 子集，样本量和正例数不足以应用 development gate，
不得用于调参、改 loss、改变 residual cap 或决定 Beauty。正式科学判断只能来自完整 19,412-user
五折 OOF。

## 5. Output files

| artifact | path |
|---|---|
| smoke summary | `artifacts/phase10/cf1_c2_toys_pcrf_anchored_smoke/summary.json` |
| fold models | `artifacts/phase10/cf1_c2_toys_pcrf_anchored_smoke/fold_models.json` |
| fold metrics | `artifacts/phase10/cf1_c2_toys_pcrf_anchored_smoke/fold_metrics.json` |
| per-user OOF | `artifacts/phase10/cf1_c2_toys_pcrf_anchored_smoke/per_user_oof.tsv` |
| Hit@10 transitions | `artifacts/phase10/cf1_c2_toys_pcrf_anchored_smoke/hit10_transitions.tsv` |
| log/status | `artifacts/phase10/cf1_c2_toys_pcrf_anchored_smoke/run.log`, `status.json` |

## 6. Decision boundary

Implementation smoke 已通过，可以准备正式 C2 full-OOF authorization；但当前 preregistration 中
`formal_execution_enabled=false`，因此本次没有启动正式 19,412-user 五折实验。下一授权动作必须：

1. 保持所有模型、loss、seed、feature 和 development gate 不变；
2. 将正式 timeout 根据 smoke 资源观测冻结；
3. 为正式 runner 锁定新的 code/plan/config SHA256；
4. 只运行一次 full OOF，不因结果或异常自动 retry。

