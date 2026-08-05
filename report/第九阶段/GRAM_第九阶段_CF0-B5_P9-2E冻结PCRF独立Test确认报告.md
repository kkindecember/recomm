# GRAM 第九阶段：CF0-B5 P9-2E 冻结 PCRF 独立 Test 确认报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-08-04
- Verification Status: `CONFIRMED`
- Experiment ID: `GRAM_PHASE9_CF0_B5_TOYS_PCRF_TEST_P2E_V1`
- Evidence Class: one-shot independent Toys test confirmation
- Frozen Mechanism: lambda=1.0、beta=0.5、gamma=1.0、q1=5、q3=26
- Excluded: test tuning、rerun、Beauty、Sports、checkpoint mutation

## 1. Executive conclusion

PCRF 通过全部 7 项预注册 confirmation gates。在从未参与 PCRF 机制选择的 19,412-user Toys
test 上，Hit@10 从 `0.095302` 提高到 `0.102875`，绝对增量 `+0.007573`、相对约
`+7.95%`；2,000 次 paired bootstrap 95% CI 为 `[+0.005461,+0.009633]`。NDCG@10
提高 `+0.004161`，Hit@1 提高 `+0.001391`，Hit@50 按 reranking 结构严格不变。

P9-2C 最主要的 tail 风险也在独立 test 上得到修复并转为正向：tail Hit@10 从 `0.069131`
提高到 `0.075372`，delta `+0.006241`，95% CI `[+0.003361,+0.009121]`。因此本轮支持：

> 冻结 item-head 的协同信号适合在 GRAM legal beams 内进行 popularity-calibrated、
> reliability-aware late fusion；P9-2B hidden injection 的失败不能归因于协同机制本身。

## 2. One-shot integrity

- 正式机制参数在读取 PCRF test 结果前由 P9-2D 冻结；
- 没有 test 网格、fold fitting、early stopping 或 target-derived gate feature；
- 19,412 users、11,924 items、50 unique legal candidates/user 全部通过 mapping；
- 重算 baseline Hit/NDCG@5/10/20/50 与历史 cache footer 在 `1e-12` 内一致；
- item-head history 使用 `items[:-1]` 最近 20 项，target 为 `items[-1]`；
- popularity 只统计 `items[:-2]` train-prefix interactions；
- frozen input/code hashes 完整通过；runner 在 summary 已存在时拒绝再次 start；
- test 正式执行一次，CPU wall `9.18 s`，未改写 checkpoint 或占用 GPU。

## 3. Primary test results

| metric | GRAM baseline | frozen PCRF | delta |
|---|---:|---:|---:|
| Hit@1 | 0.031063 | 0.032454 | +0.001391 |
| Hit@5 | 0.071193 | 0.076448 | +0.005254 |
| Hit@10 | 0.095302 | 0.102875 | +0.007573 |
| Hit@20 | 0.124768 | 0.132907 | +0.008139 |
| Hit@50 | 0.172883 | 0.172883 | 0 |
| NDCG@10 | 0.059193 | 0.063354 | +0.004161 |
| NDCG@20 | 0.066682 | 0.070928 | +0.004246 |
| NDCG@50 | 0.076188 | 0.078925 | +0.002737 |
| MRR@50 | 0.051715 | 0.054618 | +0.002903 |

固定 P9-2C lambda=0.75 在同一 test 上的 Hit@10 delta 为 `+0.005924`；PCRF 为
`+0.007573`。这项诊断说明 popularity calibration/reliability shrinkage 不只是提供 tail safety，
也带来额外 overall 增益；它未参与 confirmation 参数选择。

## 4. Subgroup confirmation

### 4.1 Target popularity

| group | n | baseline Hit@10 | PCRF Hit@10 | delta |
|---|---:|---:|---:|---:|
| tail | 6,249 | 0.069131 | 0.075372 | +0.006241 |
| middle | 8,931 | 0.099765 | 0.110850 | +0.011085 |
| head | 4,232 | 0.124527 | 0.126654 | +0.002127 |

三个组均为正，最大增益出现在 middle。tail bootstrap CI 下界为正，因此本轮可以描述为 tail
改善，而不只是未观测到伤害。head Hit@10 仍改善，但 head Hit@1 从 `0.039461` 小幅降至
`0.038516`；这是辅助分层风险，不改变预注册 overall Hit@1 已通过的结论。

### 4.2 History length

| history | n | baseline Hit@10 | PCRF Hit@10 | delta |
|---|---:|---:|---:|---:|
| 1–5 | 10,401 | 0.101433 | 0.108067 | +0.006634 |
| 6–10 | 6,169 | 0.087697 | 0.094667 | +0.006970 |
| 11–20 | 2,842 | 0.089374 | 0.101689 | +0.012315 |

三个 history groups 均改善，长历史组增益最大，与 P9-2D development 方向一致。

## 5. Confirmation gates

| check | observed | result |
|---|---:|---|
| Hit@10 delta ≥ +0.002 | +0.007573 | PASS |
| Hit@10 bootstrap lower > 0 | +0.005461 | PASS |
| NDCG@10 delta ≥ 0 | +0.004161 | PASS |
| tail Hit@10 delta ≥ 0 | +0.006241 | PASS |
| tail CI lower ≥ -0.002 | +0.003361 | PASS |
| Hit@1 delta ≥ -0.001 | +0.001391 | PASS |
| Hit@50 identity | 0 | PASS |

终态为 `confirmed`，不是“development passed”。

## 6. Remaining limits

1. confirmation 依赖历史 cached beams；还未验证重新解码后逐候选排序的工程复现性；
2. 当前证据只有 Toys、一个 GRAM checkpoint 和一个 item-head seed；
3. Hit@50 上限仅 `0.172883`，PCRF 无法挽救不在 beam50 的 target；
4. Beauty 外部验证需要独立训练/冻结 Beauty item-head，不能直接迁移 Toys embedding；
5. 本 test 已消费，不得用于 PCRF 后续公式或超参数调整。

合理的下一阶段是先做同 checkpoint 的 fresh-beam reproducibility，随后再做 Beauty 外部验证；
二者都应维持当前 PCRF 参数冻结。

## 7. Statistical integrity

- primary effect 同时报 effect size、paired CI 和绝对/相对变化；
- 11/11 fallacy classes checked；
- Simpson's paradox：overall 与 tail/middle/head Hit@10 同向，未见反转；head Hit@1 的辅助反向
  已单独披露；
- look-elsewhere / garden of forking paths：test 前冻结单一参数与 7 项 gates，无 test tuning；
- survivorship：全部 19,412 users 进入分析，无 attrition；
- regression-to-mean：未按 validation 极端用户筛选 test；
- collider/Berkson/ecological/base-rate：未发现改变主结论的适用结构；
- causality：受控 paired reranking 支持当前 PCRF 相对 baseline 的 test 效应，不外推到其他域、
  checkpoint 或在线因果效果；
- reverse causality：不适用；排序干预先于离线指标计算。

Overall Confidence：`CONFIRMED_ON_TOYS_TEST`，尚非跨 seed/跨域 `VERIFIED`。

## 8. Artifacts

- plan：`plan/第九阶段/GRAM_第九阶段_CF0-B5_P9-2E冻结PCRF独立Test确认计划.md`；
- config：`artifacts/phase9/configs/cf0_b5_toys_pcrf_test_p2e_preregistered.json`；
- summary：`artifacts/phase9/cf0_b5_toys_pcrf_test_p2e/summary.json`；
- per-user：`artifacts/phase9/cf0_b5_toys_pcrf_test_p2e/per_user_test.tsv`；
- runner/log/status：同一 artifact 目录及
  `experiment/phase9/run_phase9_cf0_b5_pcrf_test_p2e.sh`。

per-user SHA256：`ad67c81420525bbb3f2950a2c439b1565efb5c27b313f82c7439484a2a8b0f68`。
