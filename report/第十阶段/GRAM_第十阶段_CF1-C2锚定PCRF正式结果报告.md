# GRAM 第十阶段：CF1-C2 锚定 PCRF 正式结果报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-08-04
- Experiment ID: `GRAM_PHASE10_CF1_C2_TOYS_PCRF_ANCHORED_V1`
- Verification Status: `ANALYZED`
- Evidence Class: post-C1 cross-fitted validation development, not independent confirmation
- Engineering Status: `COMPLETED`
- Implementation Gate: `PASSED`
- Development Gate: `FAILED`
- Toys Test/Beauty/Sports Read: false

## 1. Executive conclusion

CF1-C2 在 75.33 秒内完成完整 19,412-user 五折 OOF，五折全部收敛、score 全部 finite、PCRF
anchor identity 与 train-fold isolation 均通过，但没有通过冻结 development gate。相对 frozen PCRF：

- Hit@10 delta `-0.000567`，51 gain / 62 loss，净损失 11；
- paired bootstrap 95% CI `[-0.001648,+0.000515]`，区间跨 0；
- Hit@50 delta `+0.008191`，净增加 159，但低于 `+0.020` 门槛；
- tail Hit@10 delta `+0.005814`，安全门通过；
- Hit@1 delta `+0.000103`，安全门通过；
- Hit@10 仅 2/5 折为正，低于 4/5 门槛。

C2 的 PCRF anchor 和 popularity-balanced objective 确实修复了 C1 的 tail degradation，却把损失转移到
head 与 `both` target，并且没有把任何 CF-only target 推入 top-10。按照预注册 stop logic，CF1 在
Toys validation 上到此停止：保留 frozen PCRF，不重跑、不扩大模型、不降低 gate、不读取 Toys test，
也不进入 CF1-D Beauty。

## 2. Formal execution result

- Command: `bash experiment/phase10/run_phase10_cf1_c2_pcrf_anchored.sh start`
- Working Directory: repository root
- Type: CPU five-fold optimization / analysis
- Status: `completed`
- Exit Code: 0
- Evaluator Wall Time: 75.33 seconds
- Hard Timeout: 3,600 seconds, not reached
- Fold iterations: `[32,34,28,30,30]`
- Fold evaluation users: `[3883,3883,3882,3882,3882]`
- Anomalies: initial sandbox无法连接 tmux；经用户批准访问已有 tmux 服务后正常启动。实验进程本身无
  crash、stall、resource anomaly 或 retry。

## 3. Frozen primary comparison

| metric | frozen PCRF | C2 OOF | delta | gate | status |
|---|---:|---:|---:|---:|---|
| Hit@1 | 0.041366 | 0.041469 | +0.000103 | >= -0.001 | PASS |
| Hit@10 | 0.125335 | 0.124768 | -0.000567 | >= +0.003 | FAIL |
| NDCG@10 | 0.078716 | 0.078698 | -0.000017 | diagnostic | neutral/negative |
| Hit@50 | 0.211931 | 0.220122 | +0.008191 | >= +0.020 | FAIL |
| NDCG@50 | 0.098071 | 0.099638 | +0.001568 | diagnostic | positive |

Hit@10 bootstrap 2,000 replicates、seed 2023 的 95% CI 为
`[-0.001648,+0.000515]`。该区间不支持正向 Hit@10 增益；这里不把“接近 0”改写为趋势。

## 4. Gate audit

| frozen check | observed | status |
|---|---:|---|
| Hit@10 delta >= +0.003 | -0.000567 | FAIL |
| Hit@50 delta >= +0.020 | +0.008191 | FAIL |
| tail Hit@10 delta >= 0 | +0.005814 | PASS |
| Hit@1 delta >= -0.001 | +0.000103 | PASS |
| Hit@10 bootstrap lower > 0 | -0.001648 | FAIL |
| positive Hit@10 folds >= 4 | 2/5 | FAIL |
| all folds converged | 5/5 | PASS |
| all OOF scores finite | 100% | PASS |
| train-only parameters | true | PASS |
| protected splits unread | true | PASS |

五折 Hit@10 delta 为 `[-0.002060,-0.000258,-0.001546,+0.000773,+0.000258]`；Hit@50 delta
为 `[+0.006696,+0.009786,+0.008501,+0.007470,+0.008501]`。Hit@50 五折方向一致，但幅度不足，
不能替代联合 gate。

## 5. Popularity and history subgroups

| subgroup | users | Hit@10 gain/loss/net | Hit@10 delta | Hit@50 net | Hit@50 delta |
|---|---:|---:|---:|---:|---:|
| target tail | 5,160 | 30 / 0 / +30 | +0.005814 | +82 | +0.015891 |
| target middle | 9,235 | 17 / 18 / -1 | -0.000108 | +152 | +0.016459 |
| target head | 5,017 | 4 / 44 / -40 | -0.007973 | -75 | -0.014949 |
| history 1--5 | 12,673 | 34 / 37 / -3 | -0.000237 | +70 | +0.005524 |
| history 6--10 | 4,319 | 10 / 15 / -5 | -0.001158 | +31 | +0.007178 |
| history 11--20 | 2,420 | 7 / 10 / -3 | -0.001240 | +58 | +0.023967 |

Popularity balancing 没有带来总体 top-10 增益，而是形成 tail/head 反向迁移。History 11--20 的
Hit@50 超过 `+0.020`，但它是预注册诊断 subgroup，不能用来替代 overall gate 或挑选用户上线。

## 6. Source-level mechanism

| target source | users | Hit@10 gain/loss/net | Hit@10 delta | Hit@50 net | Hit@50 delta |
|---|---:|---:|---:|---:|---:|
| GRAM-only | 1,789 | 36 / 28 / +8 | +0.004472 | -192 | -0.107323 |
| both | 2,325 | 15 / 34 / -19 | -0.008172 | -15 | -0.006452 |
| CF-only | 1,025 | 0 / 0 / 0 | 0 | +366 | +0.357073 |
| union miss | 14,273 | 0 / 0 / 0 | 0 | 0 | 0 |

C2 将 366 个 CF-only target 推入 top-50，但没有任何 CF-only target 进入 top-10；同时原 G50 中
GRAM-only/both 在 top-50 合计净损失 207，最终只剩 159 个 overall net gain。这说明 bounded residual
实现了“保守插入”，却没有解决新候选的 top-10 calibration；进一步增加 cap 或模型容量将构成未预注册
调参，且不受当前证据授权。

## 7. C1 versus C2

| delta vs PCRF | C1 | C2 | C2 - C1 |
|---|---:|---:|---:|
| Hit@10 | -0.000309 | -0.000567 | -0.000258 |
| Hit@50 | +0.014630 | +0.008191 | -0.006439 |
| tail Hit@10 | -0.004845 | +0.005814 | +0.010659 |
| head Hit@10 | +0.002193 | -0.007973 | -0.010165 |

C2 修复 tail 的代价超过收益，没有形成 Pareto improvement。C1 与 C2 都未通过同一冻结 gate，因此
不能在二者之间事后挑一个进入 Beauty。

## 8. Numerical and reproducibility audit

- implementation gate 12/12 通过；
- max absolute residual `0.997563 <= 1.0`；
- `|residual|>=0.95` fraction `0.008142`，`>=0.99` fraction `0.0000865`，无大面积 saturation；
- formal preflight tests 4/4 通过；
- evaluator、runner、tests、plan 和全部输入由 formal config SHA256 锁定；
- `per_user_oof.tsv`、fold models、fold metrics、transitions 的 SHA256 与 summary 记录 exact；
- 未进行第二次 full rerun，因此 Verification Status 为 `ANALYZED`，不是 `VERIFIED`。

## 9. Statistical validation and fallacy scan

- Overall Confidence: `SOLID` for the failed-gate conclusion；
- 11/11 statistical fallacy types checked；
- Simpson's paradox：未出现所有 subgroup 与 overall 完全反向的严格模式；存在 aggregation masking
  caution，overall 负向由 tail 正向与 head 负向合成；
- ecological、Berkson、collider、base-rate neglect、regression-to-mean、survivorship：设计中不适用或
  未发现；19,412 users 全部进入 OOF，没有 attrition；
- look-elsewhere / garden-of-forking-paths：primary、gate、fold、seed、loss 和 stop logic 均预注册；
  subgroup/source 只解释机制，不替代 gate；
- correlation/causation 与 reverse causality：本实验没有提出因果主张；
- multiple comparisons：联合 primary gate 不因诊断指标数量而放宽，也不对 subgroup 挑胜者。

## 10. Final decision

1. CF1-C2：`FAILED_DEVELOPMENT_GATE`；
2. CF1-D Beauty：不授权；
3. Toys test：继续关闭；
4. frozen PCRF `(1.0,0.5,1.0)`：保留为第十阶段最终安全方案；
5. CF1 calibration：在 Toys validation 上停止，不 retry、不换 seed、不扩大 cap/MLP、不降低 gate；
6. 若未来研究新的 retrieval/ranking 方向，应作为新的独立阶段重新提出假设和预注册，不能作为
   CF1-C2 的事后延伸。

## 11. Reproducibility pointers

- formal preregistration：`artifacts/phase10/configs/cf1_c2_toys_pcrf_anchored_formal_preregistered.json`
- summary：`artifacts/phase10/cf1_c2_toys_pcrf_anchored/summary.json`
- fold models：`artifacts/phase10/cf1_c2_toys_pcrf_anchored/fold_models.json`
- fold metrics：`artifacts/phase10/cf1_c2_toys_pcrf_anchored/fold_metrics.json`
- per-user OOF：`artifacts/phase10/cf1_c2_toys_pcrf_anchored/per_user_oof.tsv`
- transitions：`artifacts/phase10/cf1_c2_toys_pcrf_anchored/hit10_transitions.tsv`
- log/status：`artifacts/phase10/cf1_c2_toys_pcrf_anchored/run.log`, `status.json`
- evaluator：`experiment/phase10/eval_cf1_c2_pcrf_anchored.py`
- runner：`experiment/phase10/run_phase10_cf1_c2_pcrf_anchored.sh`

