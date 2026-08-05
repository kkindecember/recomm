# GRAM 第九阶段 P9-R：原 checkpoint 新鲜 Beam 复现报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-08-04
- Verification Status: `REPRODUCED`
- Experiment ID: `GRAM_PHASE9_P9R_TOYS_FRESH_BEAM_512_V1`
- Evidence Class: deterministic 512-user validation engineering reproduction
- Frozen Mechanism: `lambda=1.0, beta=0.5, gamma=1.0`
- Excluded: training、test、Beauty、Sports、parameter selection、automatic retry

## 1. Executive conclusion

P9-R 通过全部 6 项预注册复现门。原 Toys epoch-30 checkpoint 在固定 512-user validation
subset 上重新 constrained decode 后，25,600 个候选与历史缓存逐项一致；sequence score 排序与冻结
PCRF 排序也完全一致。

因此，第九阶段的 PCRF 机制没有因缓存偶然性或当前解码环境而失效。第十阶段 CF1 的负结果应解释为
“候选扩展没有超过已经有效的 PCRF anchor”，而不是“第九阶段机制被推翻”或“模型完全无法提升”。

## 2. Reproducibility results

| check | observed | threshold | result |
|---|---:|---:|---|
| legal users | 512/512 | 512/512 | PASS |
| candidate-set overlap | 1.000000 | diagnostic | exact |
| sequence top-10 overlap | 1.000000 | diagnostic | exact |
| score Pearson | 1.000000 | ≥ 0.995 | PASS |
| score Spearman | 1.000000 | ≥ 0.995 | PASS |
| PCRF top-10 overlap | 1.000000 | ≥ 0.98 | PASS |
| baseline Hit@10 abs delta | 0 | ≤ 0.001 | PASS |
| PCRF Hit@10 abs delta | 0 | ≤ 0.001 | PASS |

历史和 fresh 的 subset 指标完全相同：

| metric | baseline | frozen PCRF | PCRF - baseline |
|---|---:|---:|---:|
| Hit@10 | 0.121094 | 0.123047 | +0.001953 |
| NDCG@10 | 0.082087 | 0.084175 | +0.002088 |
| Hit@20 | 0.150391 | 0.156250 | +0.005859 |
| Hit@50 | 0.208984 | 0.208984 | 0 |

该 512-user subset 的作用是工程复现，不用于重新估计或宣称 PCRF 显著性；独立 Toys test 的正式
confirmation 仍以 P9-2E 的 19,412-user paired bootstrap 结果为准。

## 3. Integrity and resources

- 用户按 `sha256("2023:" + user_id)` 确定性选择；sample hash 已写入 summary；
- checkpoint、item-head 与输出文件均记录 SHA256；
- constrained decoding 固定 50 beams / 50 returns、全商品 trie、length penalty 1.0；
- 正式运行一次，无自动重试；`test_read=false`、`beauty_read=false`、`sports_read=false`；
- 物理 GPU3；峰值 allocated memory `6877.50 MiB`；wall time `289.22 s`；
- 3 个单元测试和完整 preflight 均通过。

## 4. Scientific interpretation

P9-R 直接消除了第九阶段剩余限制中的第一项：历史 cached beams 可以由原 checkpoint 在当前环境中
精确重建。因此当前证据链为：

1. Toys validation 上完成机制选择与 cross-fitted development；
2. Toys independent test 上获得 Hit@10 `+0.007573`，95% CI `[+0.005461,+0.009633]`；
3. 原 checkpoint 的 fresh-beam 解码在 512 用户上逐项复现。

仍未解决的是跨数据集外部效度。下一步不是继续在 Toys/CF1 上搜索，而是独立训练并冻结 Beauty
item-head，先过 Beauty validation 的固定 PCRF 门，再读取一次 Beauty test。

## 5. Artifacts

- plan：`plan/第九阶段/GRAM_第九阶段_P9-R原Checkpoint新鲜Beam复现计划.md`
- evaluator：`experiment/phase9/eval_p9r_fresh_beam.py`
- tests：`experiment/phase9/test_p9r_fresh_beam.py`
- runner：`experiment/phase9/run_phase9_p9r_fresh_beam.sh`
- summary：`artifacts/phase9/p9r_toys_fresh_beam_512/summary.json`
- fresh beams：`artifacts/phase9/p9r_toys_fresh_beam_512/fresh_beams.tsv`
- per-user evidence：`artifacts/phase9/p9r_toys_fresh_beam_512/per_user.tsv`
- run log/status：同一 artifact 目录。
