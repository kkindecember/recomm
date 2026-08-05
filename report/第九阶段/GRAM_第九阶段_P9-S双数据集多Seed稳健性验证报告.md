# GRAM 第九阶段 P9-S：双数据集多 Seed 稳健性验证报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-08-04
- Verification Status: `CONFIRMED_MULTI_SEED`
- Version Label: p9s_multiseed_validation_v1
- Experiment ID: `GRAM_PHASE9_P9S_MULTISEED_VALIDATION_V1`

## 1. Executive conclusion

冻结 PCRF `lambda=1.0, beta=0.5, gamma=1.0` 在 Toys、Beauty 各三个独立
item-head seed 上通过全部预注册稳健性门控。六个 dataset-seed 单元的 Hit@10、NDCG@10
和 tail Hit@10 delta 全部为正，Hit@50 全部严格不变；四个新训练 item-head 也全部通过
既有 scientific gate。

Toys 三个 seed 的 Hit@10 delta 均值为 `+0.006285`、中位数为 `+0.005924`；Beauty
均值为 `+0.004978`、中位数为 `+0.005098`。因此，第九阶段机制收益不依赖 seed 2023，
可以进入第十一阶段 beam-width validation pilot，诊断剩余瓶颈是否来自候选覆盖。

## 2. Frozen protocol

- 数据矩阵：Toys / Beauty × seed 2023 / 2024 / 2025；
- seed 2023 复用此前冻结 checkpoint；seed 2024、2025 按同一协议独立训练；
- 新 item-head 均为 10 epochs、batch 512、AdamW `3e-4`、weight decay `0.01`、
  warmup `0.05`、max history 20、d=512、2 layers、4 heads、dropout `0.1`；
- PCRF 参数和 q1 计算规则不变，只读 train-prefix 与 validation；
- `test_read=false`、`sports_read=false`，未再次读取已封存 test。

## 3. Seed-level results

| Dataset | Seed | item-head gate | Hit@10 delta | NDCG@10 delta | tail Hit@10 delta | Hit@50 delta |
|---|---:|---|---:|---:|---:|---:|
| Toys | 2023 | frozen existing | +0.005924 | +0.002441 | +0.001163 | 0 |
| Toys | 2024 | PASS | +0.007006 | +0.003263 | +0.002132 | 0 |
| Toys | 2025 | PASS | +0.005924 | +0.001913 | +0.001938 | 0 |
| Beauty | 2023 | frozen existing | +0.005098 | +0.002934 | +0.001488 | 0 |
| Beauty | 2024 | PASS | +0.004651 | +0.003161 | +0.000827 | 0 |
| Beauty | 2025 | PASS | +0.005187 | +0.003758 | +0.001488 | 0 |

跨 seed 离散度较小：Toys Hit@10 delta 的 sample std 为 `0.000625`，Beauty 为
`0.000287`。最弱单元仍为正：Toys 最小值 `+0.005924`，Beauty 最小值 `+0.004651`。

## 4. Preregistered robustness gate

| Check | Result |
|---|---|
| four new item-heads passed | PASS |
| all six Hit@10 deltas positive | PASS |
| both dataset median Hit@10 delta ≥ 0.002 | PASS |
| all six NDCG@10 non-degradation | PASS |
| all six tail Hit@10 non-degradation | PASS |
| all six Hit@50 identity | PASS |

Overall gate：`PASSED`。

## 5. Execution audit

首次启动在进入任何训练或科学评价前因 runner 的 bash 局部变量同语句展开触发
`seed: unbound variable`，exit code 1。该轮只留下 `run.log` 与失败 `status.json`，没有 checkpoint、
validation summary 或 test 读取。按照 no-auto-retry 规则，在研究者明确回复“继续”后才进行恢复。

恢复只把局部变量声明拆成顺序赋值并增加 EXIT 状态落盘；没有改变数据、seed、模型、PCRF 参数、
门槛或用户集合。恢复前 `bash -n` 通过，相关测试 `2 passed`；正式重试中完整预检 `7 passed`，
最终于 `2026-08-04T23:06:40+08:00` 正常完成。

## 6. Interpretation and boundary

P9-S 排除了“单 seed 偶然”这一主要替代解释，并与两个数据集的一次性 test 证据一致。当前最强结论是：
PCRF 在固定 beam50 候选集合内能稳定改善前十排序，且不损失候选集合覆盖。

它仍不能改善目标未进入 beam50 的用户，因此下一步不是继续调 PCRF，而是冻结现有机制，在相同 validation
cohort 上独立生成 beam width 50/100/200，分离“候选覆盖上限”和“候选内重排上限”。

## 7. Artifacts

- preregistration：`plan/第九阶段/GRAM_第九阶段_P9-S双数据集多Seed稳健性验证计划.md`
- aggregate：`artifacts/phase9/p9s_multiseed/summary.json`
- seed table：`artifacts/phase9/p9s_multiseed/seed_results.tsv`
- run audit：`artifacts/phase9/p9s_multiseed/status.json`、`run.log`
- runner：`experiment/phase9/run_phase9_p9s_multiseed.sh`
- aggregator：`experiment/phase9/summarize_p9s_multiseed.py`
