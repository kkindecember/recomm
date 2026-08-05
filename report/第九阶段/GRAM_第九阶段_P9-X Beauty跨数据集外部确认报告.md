# GRAM 第九阶段 P9-X：Beauty 跨数据集外部确认报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-08-04
- Verification Status: `CONFIRMED_CROSS_DATASET`
- Version Label: p9x_beauty_external_confirmation_v1
- Experiment IDs: `GRAM_PHASE9_P9X_BEAUTY_ITEM_HEAD_V1`、
  `GRAM_PHASE9_P9X_BEAUTY_VALIDATION_FIXED_PCRF_V1`、
  `GRAM_PHASE9_P9X_BEAUTY_TEST_FIXED_PCRF_V1`

## 1. Executive conclusion

冻结 PCRF 在 Beauty validation 和一次性独立 test 上均通过全部 7 项预注册门控。Beauty test
Hit@10 从 `0.088897` 提高到 `0.094173`，绝对增量 `+0.005277`、相对约 `+5.94%`；
2,000 次 paired bootstrap 95% CI 为 `[+0.003488,+0.007155]`。NDCG@10 提高
`+0.003037`，Hit@1 提高 `+0.001386`，Hit@50 严格不变。

结合 Toys independent test 的 Hit@10 `+0.007573` 与 P9-R fresh-beam 精确复现，当前证据支持：

> popularity-calibrated、reliability-aware 的协同 late fusion 不是 Toys 缓存偶然或单数据集特例；
> 它已在两个数据集、各自独立训练的 item-head 和各自独立 test 上保持正向效果。

## 2. Beauty item-head prerequisite

- 仅使用 Beauty train-prefix；seed 2023；架构与优化参数完全沿用 Toys；
- 10 epoch 全部 finite；best epoch = 7；
- Recall@10 `0.095738`，门槛 `0.018835`；
- Recall@50 `0.186737`，门槛 `0.053284`；
- non-head Recall@50 `0.119489`，门槛 `0.005`；
- item-head gate 3/3 PASS。

Beauty item embedding 独立训练，没有迁移 Toys embedding；PCRF 公式仍冻结为
`lambda=1.0, beta=0.5, gamma=1.0`，Beauty train-prefix 冻结 `q1=6`。

## 3. Validation admission

| metric | GRAM | PCRF | delta |
|---|---:|---:|---:|
| Hit@1 | 0.031749 | 0.032688 | +0.000939 |
| Hit@5 | 0.077628 | 0.081608 | +0.003980 |
| Hit@10 | 0.108751 | 0.113849 | +0.005098 |
| Hit@20 | 0.146939 | 0.153915 | +0.006976 |
| Hit@50 | 0.208112 | 0.208112 | 0 |
| NDCG@10 | 0.064974 | 0.067908 | +0.002934 |

Hit@10 paired 95% CI 为 `[+0.003085,+0.007110]`。tail Hit@10 从 `0.041825`
提高到 `0.043313`，tail CI 为 `[0,+0.002976]`。7/7 admission gates PASS 后才读取 test。

## 4. One-shot Beauty test confirmation

| metric | GRAM | frozen PCRF | delta |
|---|---:|---:|---:|
| Hit@1 | 0.024684 | 0.026070 | +0.001386 |
| Hit@5 | 0.063274 | 0.066673 | +0.003398 |
| Hit@10 | 0.088897 | 0.094173 | +0.005277 |
| Hit@20 | 0.121585 | 0.127040 | +0.005455 |
| Hit@50 | 0.174977 | 0.174977 | 0 |
| NDCG@10 | 0.052461 | 0.055497 | +0.003037 |
| NDCG@50 | 0.071281 | 0.073325 | +0.002044 |

Test tail 7,329 用户的 Hit@10 从 `0.036840` 提高到 `0.040115`，delta
`+0.003275`，95% CI `[+0.001637,+0.005185]`。因此不仅 overall 正向，主要风险组也在
独立 test 上得到正向确认。

## 5. Statistical integrity

- paired unit 为同一用户；所有 22,363 用户进入分析，无 attrition；
- 参数、q1、checkpoint 与 7 项 gates 均在读取 test 前冻结；
- test 只执行一次，没有 test 网格、checkpoint selection、阈值修改或自动重跑；
- baseline footer identity 在 `1e-12` 内通过；50-beam reranking 保证 Hit@50 恒等；
- primary effect 同时报绝对增量、相对增量和 paired CI。

### Fallacy scan（11/11 checked）

| Fallacy | Assessment |
|---|---|
| Simpson's paradox | overall 与 tail 均正向；未见方向反转 |
| Ecological fallacy | 用户级 paired 分析，不以群体统计推断个体机制 |
| Berkson's paradox | 结论限定于 GRAM beam50 内 reranking，不外推未入 beam 用户 |
| Collider bias | 未按结果变量筛选或控制共同结果变量 |
| Base-rate neglect | 同时报全体、tail 样本量与 baseline |
| Regression to mean | 未按极端 validation 用户选择 test |
| Survivorship bias | 22,363/22,363 用户全部保留 |
| Look-elsewhere effect | 单一冻结公式、单一 primary metric、预注册门控 |
| Garden of forking paths | Beauty 上未搜索 lambda/beta/gamma/q1 或阈值 |
| Correlation ≠ causation | 支持离线 paired reranking 效应，不声称在线因果收益 |
| Reverse causality | 排序干预先于指标计算；不适用 |

Overall Confidence：`SOLID / CONFIRMED_CROSS_DATASET`。限制仍包括单 seed、各数据集单一 GRAM
checkpoint、历史 Beauty beams，以及无法改善 beam50 coverage。

## 6. Artifacts

- plan：`plan/第九阶段/GRAM_第九阶段_P9-X Beauty跨数据集外部确认计划.md`
- Beauty item-head：`artifacts/phase9/p9x_beauty_item_head/`
- validation summary：`artifacts/phase9/p9x_beauty_validation/summary.json`
- test summary：`artifacts/phase9/p9x_beauty_test/summary.json`
- evaluator：`experiment/phase9/eval_p9x_fixed_pcrf.py`
- runner：`experiment/phase9/run_phase9_p9x_beauty_validation.sh`
