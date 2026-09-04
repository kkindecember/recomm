# Stage17 FP1/FP2 External D0 正式结果报告

- 生成时间：`2026-09-02T13:17:08.385322+00:00`
- 正式分析：`artifacts/phase17/fullport/external_d0/attempt_001/analysis.json`（SHA256 `30933210b311253f525b6eb1610c8e9289145b3870a8bed3f4e708645948f620`）
- 外部用户数：`12833`
- 数据纪律：D0 仅物化一次；本次恢复只复用 sealed bundle；D1/D2/test/Sports 均未读取。

## 结论

- FP1：`FP1_NOT_STRONG_PASS`。
- FP2：`FP2_NOT_STRONG_PASS`。
- 冻结决策：`stop standalone LATTE migration; do not open D1 and do not tune on D0`。
- 只有 Gate 明确通过的分支才可进入计划规定的后续注册；任何情况下当前仍不得读取 D1。

## 主结果

| Arm | Primary variant | Hit@10 | NDCG@10 | MRR@10 | Hit@50 | NDCG@50 |
|---|---|---:|---:|---:|---:|---:|
| N0_NATIVE_PSID | beam500_identity | 0.059378 | 0.030134 | 0.021341 | 0.152264 | 0.050256 |
| N1_NATIVE_LATTE | beam500_agg_max | 0.058287 | 0.031528 | 0.023426 | 0.151874 | 0.051742 |
| G0_GRAM_B0_FRESH | beam500_identity | 0.097561 | 0.061700 | 0.050685 | 0.180472 | 0.079821 |
| G1_GRAM_PSID_FULL | beam500_identity | 0.052677 | 0.028915 | 0.021720 | 0.126237 | 0.044800 |
| G2_GRAM_LATTE_FULL | beam500_agg_max | 0.044728 | 0.025044 | 0.019026 | 0.116652 | 0.040433 |

## 配对效应

| Comparison | ΔNDCG@10 | 95% CI | ΔHit@10 | Gain/Loss/Tie | Changed target rank |
|---|---:|---|---:|---|---:|
| FP1_N1_MINUS_N0 | 0.001394 | [-0.000874, 0.003753] | -0.001091 | 527/512/11794 | 0.196369 |
| FP2_G1_MINUS_G0 | -0.032785 | [-0.036271, -0.029303] | -0.044884 | 415/1061/11357 | 0.222629 |
| FP2_G2_MINUS_G0 | -0.036657 | [-0.040028, -0.033205] | -0.052833 | 364/1090/11379 | 0.221616 |
| FP2_G2_MINUS_G1 | -0.003872 | [-0.005890, -0.001882] | -0.007948 | 347/435/12051 | 0.147822 |

## Gate 审计

| Gate | Verdict | Passed checks | Failed checks |
|---|---|---|---|
| FP1 | `FP1_NOT_STRONG_PASS` | aggregate_item_valid, integrity_valid, latent_not_collapsed, multi_path_item_rate_positive, ndcg_delta_positive | hit_delta_nonnegative, ndcg_ci95_low_positive |
| FP2 | `FP2_NOT_STRONG_PASS` | aggregate_item_valid, integrity_valid, item_aggregation_gain_positive, latent_not_collapsed, multi_path_item_rate_positive, target_path_survival_positive, tree_coupling_reduced_vs_g1 | g2_vs_g0_delta_ge_0.0015, g2_vs_g0_hit_delta_nonnegative, g2_vs_g1_ci95_low_positive, g2_vs_g1_delta_ge_0.0015, no_catastrophic_large_subgroup |

## G2 vs G0 子组

| Dimension | Group | Users | ΔNDCG@10 | ΔHit@10 |
|---|---|---:|---:|---:|
| history_length | long_ge10 | 1788 | -0.027185 | -0.046980 |
| history_length | medium_4_9 | 3435 | -0.037744 | -0.051528 |
| history_length | short_le3 | 7610 | -0.038391 | -0.054796 |
| memory | generalization | 12833 | -0.036657 | -0.052833 |
| memory | memorization | 0 | 0.000000 | 0.000000 |
| target_frequency | head | 4166 | -0.012046 | -0.015122 |
| target_frequency | mid | 3921 | -0.041810 | -0.056618 |
| target_frequency | tail | 4746 | -0.054002 | -0.082807 |

## 完整性与机制证据

- 用户严格对齐：`True`；五臂 primary ranking 均非空：`True`。
- G2 constrained path 全合法：`True`。
- PSID collision aliases after：`0`；reassigned items：`1337`。
- N1 机制：`{"available": true, "latent_collapsed": false, "latent_counts": {"1": 801169, "2": 798236, "3": 949110, "4": 798406, "5": 690222, "6": 697726, "7": 873882, "8": 751961}, "latent_entropy": 2.0743305219243444, "latent_normalized_entropy": 0.997542119048289, "latent_user_collapse_rate": 0.0, "mean_aggregation_gain_ndcg@10": 0.01586046241454608, "mean_duplicate_item_paths": 421.6929790384166, "mean_duplicate_path_rate": 0.8508522572206443, "mean_generated_paths": 500.0, "mean_post_aggregation_ndcg@10": 0.03152785213993814, "mean_post_aggregation_target_rank": 2.8511649653237745, "mean_pre_aggregation_ndcg@10": 0.01566738972539206, "mean_pre_aggregation_target_rank": 29.87399672718772, "mean_target_root_count": 1.3308657367723837, "mean_tree_distance_score_correlation": 0.020761949928359717, "mean_unique_items": 73.95979116340685, "mean_valid_paths": 495.65277020182344, "multi_path_item_rate": 0.9226477390223051, "target_path_survival_rate": 0.17969297903841658, "users": 12833, "valid_path_rate": 0.9913055404036468}`。
- G2 机制：`{"available": true, "latent_collapsed": false, "latent_counts": {"32868": 735319, "32869": 871180, "32870": 924108, "32871": 929024, "32872": 824097, "32873": 736329, "32874": 739655, "32875": 656788}, "latent_entropy": 2.0726227043994307, "latent_normalized_entropy": 0.9967208324236436, "latent_user_collapse_rate": 0.0, "mean_aggregation_gain_ndcg@10": 0.0124868918365721, "mean_duplicate_item_paths": 423.9429595573911, "mean_duplicate_path_rate": 0.8478859191147822, "mean_generated_paths": 500.0, "mean_post_aggregation_ndcg@10": 0.025043553119614693, "mean_post_aggregation_target_rank": 2.2148367490064675, "mean_pre_aggregation_ndcg@10": 0.012556661283042591, "mean_pre_aggregation_target_rank": 22.392269929089068, "mean_target_root_count": 1.0075586378866983, "mean_tree_distance_score_correlation": 0.003184831846806029, "mean_unique_items": 76.0570404426089, "mean_valid_paths": 500.0, "multi_path_item_rate": 0.9436270259563359, "target_path_survival_rate": 0.13714641938751657, "users": 12833, "valid_path_rate": 1.0}`。

## 受控恢复与运行审计

attempt_001 中 N0/N1 原进程保留；G0/G2 在生成任何预测前因 PyTorch 1.11 不接受 `weights_only` 参数而失败；G1 未越过 GPU admission。研究者明确回复“同意受控恢复”后，attempt_002 恢复 G0/G2；随后研究者把 G1 更正为立即并行运行并要求完成后保持资源占用，因此 G1 使用独立 attempt_003。所有恢复均复用同一 sealed bundle。

| Arm | Attempt | GPU | Wall time (s) | Prediction SHA256 |
|---|---|---:|---:|---|
| N0_NATIVE_PSID | attempt_001 | 7 | 1759.756857 | 513eb0d9dcfafa2d4298ff639ea01b2cae8e815d8365ba911b9bfa9d3fe9fbe7 |
| N1_NATIVE_LATTE | attempt_001 | 2 | 2057.848048 | 61df030582003cf273f44f4e54d54a47f9410fe34e40cb5ba7f835bb3d6b8c55 |
| G0_GRAM_B0_FRESH | attempt_002 | 5 | 30906.505418 | f7b2d327c3b1a1d33d29cd6c6497058b6d28b771c88ecb671b95d1e11f2301ea |
| G1_GRAM_PSID_FULL | attempt_003 | 4 | 23666.511461 | 95c3ced8963a8c9e82caed439c44ba5c55ec6606f937da2d0aac215ba91017d7 |
| G2_GRAM_LATTE_FULL | attempt_002 | 6 | 29276.871883 | 2f7dcd00d723dc4f17919008d6feaf9bb92dd540dc8de82d6490171abc2aa47d |

- 恢复授权：`artifacts/phase17/authorizations/s17_fp12_external_d0_recovery_g1_parallel_attempt_003.json`（SHA256 `33d16a0625d88c2c1fc442ccbf98b0dcc7c1409d5b09c7fbd720a4187a0ff113`）。
- 恢复证据：`artifacts/phase17/fullport/external_d0/recovery/attempt_003/provenance.json`（SHA256 `4e49a8979c84167257c247ef819b92c01e653926d247cddc40ae571cc07f6f8f`）。
- Bundle SHA256：`e677d5c5905e5298f4a51541f112765796d7e3599963ed8432d62191e5bcf2d6`；single materialization count：`1`。
- G1 完成后的 GPU4 资源维护使用独立 v2 守护与 `run-NNNN` 目录；`result_selection_eligible=false`、`repeat_metrics_ignored=true`、`affects_scientific_result=false`，不重开 external D0。
- 恢复与守护回归：相关 `25 passed`；Phase17 全量 `256 passed, 1 skipped, 1 warning`。
- `automatic_retry=false`、`raw_external_projection_reopened=false`；attempt_001 失败证据未覆盖。

## 下一步

严格执行分析冻结动作：`stop standalone LATTE migration; do not open D1 and do not tune on D0`。在新 preregistration、预算与显式授权完成前，不启动 D1、FP4 或任何 D0 调参。
