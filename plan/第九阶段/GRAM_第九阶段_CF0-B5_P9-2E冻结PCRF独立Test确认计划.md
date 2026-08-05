# GRAM 第九阶段：CF0-B5 P9-2E 冻结 PCRF 独立 Test 确认计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → run
- Origin Date: 2026-08-04
- Verification Status: `EXECUTED_CONFIRMED`
- Experiment ID: `GRAM_PHASE9_CF0_B5_TOYS_PCRF_TEST_P2E_V1`
- Parent: P9-2D development gate passed
- Scope: one-shot Toys test confirmation、CPU-only frozen reranking
- Excluded: parameter tuning、validation reuse、Beauty、Sports、checkpoint mutation

## 1. Objective 与不可逆边界

验证 P9-2D 冻结机制 `(lambda=1.0, beta=0.5, gamma=1.0)` 能否在从未参与 PCRF 机制选择的
Toys test 上复现 overall top-10 提升，并保持 tail 与 top-1 安全。

本实验首次把 PCRF 应用于 test。一旦正式 evaluator 读取 test cache，无论结果通过、失败或异常，
都不得回到 validation 改公式后再次宣称同一 test 为独立确认；任何后续修改只能转向新数据域或
明确标记为 post-test development。

## 2. Frozen inputs

- GRAM baseline test cache：`GRAM/preds/20260722_094800_Toys_sequential_pred_test.tsv`；
- cache 来源与原 epoch-30 checkpoint 日志一致，19,412 users × 50 legal beams；
- frozen P9-2A item-head：`best_item_head.pt`；
- history：`items[:-1]` 最近 20 项，包含 validation interaction；
- target：`items[-1]`；
- popularity frequency：仍只由 `items[:-2]` train-prefix interactions 计算；
- tail/head boundaries 固定沿用 P9-2D：`q1=5, q3=26`，不根据 test target 重算。

## 3. Frozen mechanism

```text
pop_z = zscore(log(1 + train_frequency(candidate))) within 50 beams
cf_pc = zscore(cf_z - 0.5 * pop_z)
tail_mass = fraction of original GRAM top-10 candidates with train_frequency <= 5
reliability = 1 - tail_mass
joint = seq_z + 1.0 * reliability * cf_pc
```

不运行网格、不拟合 fold、不读取 target-derived feature。P9-2C fixed lambda=0.75 只作诊断对照，
不参与结论或参数选择。

## 4. Integrity Gate 0

- test cache 恰好 19,412 data rows + 12 footer metrics；
- cache/data user id 集合严格相同且无重复；
- 每用户 50 unique candidates 与 finite scores；
- cached gold 与 `items[-1]` lexical ID 严格一致；
- 所有 candidates 唯一映射至 11,924-item catalog；
- 重算 baseline Hit/NDCG@5/10/20/50 与 footer 在 `1e-12` 内一致；
- input/code SHA256 与 frozen config 一致。

失败即 `failed_integrity_gate`，不修补、不重新生成、不自动重试。

## 5. Confirmation gates

Primary：test Hit@10。2,000 次 paired user bootstrap，seed 2023。

必须全部满足：

1. `ΔHit@10 >= +0.002`；
2. Hit@10 paired bootstrap 95% CI lower > 0；
3. `ΔNDCG@10 >= 0`；
4. test tail `ΔHit@10 >= 0`；
5. test tail Hit@10 bootstrap 95% CI lower >= `-0.002`（预注册非劣 margin）；
6. `ΔHit@1 >= -0.001`；
7. Hit@50 identity tolerance `1e-12`。

只有全通过才是 `confirmed`。辅助指标为 Hit/NDCG@1/5/20/50、MRR@50、middle/head、history
length、P9-2C fixed comparator；不根据辅助指标改主结论。

## 6. Setup、monitoring 与产物

- Working directory：`/mnt/18T/jiangtangyunzhi/projects/recomm`；
- Entry：`bash experiment/phase9/run_phase9_cf0_b5_pcrf_test_p2e.sh start`；
- Status：`bash experiment/phase9/run_phase9_cf0_b5_pcrf_test_p2e.sh status`；
- Evaluator：`experiment/phase9/eval_cf0_b5_pcrf_test.py`；
- Tests：`experiment/phase9/test_cf0_b5_pcrf_test.py`；
- Config：`artifacts/phase9/configs/cf0_b5_toys_pcrf_test_p2e_preregistered.json`；
- Output：`artifacts/phase9/cf0_b5_toys_pcrf_test_p2e/`；
- 必需产物：`summary.json`、`per_user_test.tsv`、`status.json`、`run.log`；
- CPU-only，hard timeout 1,800 s；process/status/log 持续可查；不影响 GPU6 CodeLlama；
- 非零退出、门失败或统计不确定均不自动重试。

## 7. Interpretation

- `confirmed`：PCRF 获得一次独立 test confirmation，可进入重新解码复现/Beauty 外部验证；
- overall 通过而 tail/top1 失败：有总体收益但安全性未确认，不能称整体确认；
- overall 失败：P9-2D validation development 未泛化；
- tail CI 通过 non-inferiority 但跨 0：只称 tail 非劣，不称 tail 改善；
- Hit@50 不变是 reranking 结构性质，不代表召回瓶颈已解决。

## 8. 实际终态（2026-08-04）

- engineering：completed；3/3 synthetic tests；CPU wall `9.18 s`；未重跑；
- baseline test Hit@10：`0.095302`；PCRF：`0.102875`；delta `+0.007573`；
- Hit@10 paired bootstrap 95% CI：`[+0.005461,+0.009633]`；
- NDCG@10 delta：`+0.004161`；
- tail Hit@10 delta：`+0.006241`，95% CI `[+0.003361,+0.009121]`；
- Hit@1 delta：`+0.001391`；Hit@50 delta：`0`；
- confirmation gate：`confirmed`，7/7 checks passed。

本 test 已消费为一次性确认集，不得用于后续 PCRF 调参。详细报告见
`report/第九阶段/GRAM_第九阶段_CF0-B5_P9-2E冻结PCRF独立Test确认报告.md`。
