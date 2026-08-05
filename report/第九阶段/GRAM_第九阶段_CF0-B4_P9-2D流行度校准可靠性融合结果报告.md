# GRAM 第九阶段：CF0-B4 P9-2D 流行度校准可靠性融合结果报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-08-04
- Verification Status: `ANALYZED_DEVELOPMENT`
- Experiment ID: `GRAM_PHASE9_CF0_B4_TOYS_RELIABILITY_FUSION_P2D_V1`
- Evidence Class: cross-fitted development, not independent confirmation
- Excluded: test、Beauty、Sports、checkpoint mutation

## 1. 结论

PCRF development gate 通过。五个 cross-fitting folds 独立选择出相同参数
`lambda=1.0, beta=0.5, gamma=1.0`。合并 OOF Hit@10 从 `0.119411` 提高到 `0.125335`，
绝对增量 `+0.005924`，2,000 次 paired bootstrap 95% CI
`[+0.003864,+0.007985]`；NDCG@10 增量 `+0.002441`，Hit@50 严格不变。

P9-2C fixed fusion 的 tail Hit@10 明显下降；PCRF 将全量 tail Hit@10 从 `0.091860` 提高到
`0.093023`，点估计 `+0.001163`。因此训练频次校准加 tail-heavy reliability shrinkage 在开发数据上
同时保留 overall 增益并消除了 tail 点估计伤害。

这仍不是独立确认：P9-2C 已暴露同一 validation 的 aggregate tail 结果。本轮 cross-fitting 保证
每个用户的参数未由其自身 label 选择，但不能恢复一个从未观察过的数据集。

## 2. 主结果

| metric | baseline | PCRF OOF | delta |
|---|---:|---:|---:|
| Hit@1 | 0.041675 | 0.041366 | -0.000309 |
| Hit@5 | 0.090923 | 0.093653 | +0.002730 |
| Hit@10 | 0.119411 | 0.125335 | +0.005924 |
| Hit@20 | 0.154441 | 0.164280 | +0.009839 |
| Hit@50 | 0.211931 | 0.211931 | 0 |
| NDCG@10 | 0.076275 | 0.078716 | +0.002441 |
| NDCG@20 | 0.085104 | 0.088530 | +0.003425 |

P9-2C fixed lambda=0.75 的全量 Hit@10 增量是 `+0.004276`；PCRF OOF 为 `+0.005924`。
在同一开发数据比较下，可靠性机制没有通过过度收缩来换取 tail safety，反而提高了 top-10
overall 增益。

## 3. Fold 稳定性

| fold | selected `(lambda,beta,gamma)` | eval ΔHit@10 | eval tail ΔHit@10 |
|---:|---|---:|---:|
| 0 | (1.0, 0.5, 1.0) | +0.006696 | +0.001907 |
| 1 | (1.0, 0.5, 1.0) | +0.006953 | 0 |
| 2 | (1.0, 0.5, 1.0) | +0.004894 | +0.000966 |
| 3 | (1.0, 0.5, 1.0) | +0.005925 | +0.000990 |
| 4 | (1.0, 0.5, 1.0) | +0.005152 | +0.001934 |

五折 overall 全正，tail 无一为负，且参数完全一致，降低了“某折偶然调参”的风险。

## 4. 未解决风险

1. tail Hit@10 bootstrap CI 为 `[-0.001744,+0.004264]`，不能声明 tail 显著改善；目前只支持
   预注册的 point non-degradation gate。
2. Hit@1 小幅下降 `-0.000309`，说明 popularity calibration 改善 top-5/10/20，但最顶部排序
   仍可能需要单独的 rank-aware mixing。
3. 所有开发证据来自 Toys validation 与同一组 cached beams；还没有 test、新 seed、新 cache 或
   跨域复现。
4. Hit@50 固定为 `0.211931`，late reranking 无法解决约 78.8% target 不在 beam 的召回上限。

下一步应冻结 `(1.0,0.5,1.0)`，只做一个独立确认实验；不要再用本 validation 扩网格或修改公式。

## 5. 完整性与产物

- 19,412 users、11,924 items、50 unique legal beams/user；
- baseline historical metrics identity tolerance `1e-12`；
- 4/4 tests，engineering completed，CPU wall `12.38 s`；
- test/Sports read：false/false；checkpoint 未改写；
- summary：`artifacts/phase9/cf0_b4_toys_reliability_p2d/summary.json`；
- per-user OOF：`artifacts/phase9/cf0_b4_toys_reliability_p2d/per_user_oof.tsv`；
- folds：`artifacts/phase9/cf0_b4_toys_reliability_p2d/fold_assignments.tsv`；
- frozen config：`artifacts/phase9/configs/cf0_b4_toys_reliability_p2d_preregistered.json`。

统计谬误扫描：已显式处理同数据重复开发、look-elsewhere、forking paths、Simpson/分层方向和
survivorship；没有把 cross-fitted association 外推为跨数据因果或独立泛化。Overall confidence：
`CAUTION`，development pass only。
