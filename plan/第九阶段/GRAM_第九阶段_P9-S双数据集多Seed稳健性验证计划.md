# GRAM 第九阶段 P9-S：双数据集多 Seed 稳健性验证计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan + run + validate
- Origin Date: 2026-08-04
- Verification Status: PREREGISTERED
- Experiment ID: `GRAM_PHASE9_P9S_MULTISEED_VALIDATION_V1`

## 目标

确认 PCRF 在 Toys 与 Beauty 上的正向 validation 效果不依赖唯一的 seed 2023 item-head。
新增 seed 2024、2025，item-head 的架构、优化、epoch 选择规则和 PCRF
`(lambda=1.0,beta=0.5,gamma=1.0)` 全部冻结。只读 train-prefix 与 validation，禁止读取 test。

## 实验矩阵

| Dataset | Seed | Item head |
|---|---:|---|
| Toys | 2023 | 复用冻结 P9-2A checkpoint |
| Toys | 2024/2025 | 按 P9-2A 同协议独立训练 |
| Beauty | 2023 | 复用冻结 P9-X checkpoint |
| Beauty | 2024/2025 | 按 P9-X 同协议独立训练 |

每个新 seed：10 epochs、batch 512、AdamW 3e-4、weight decay 0.01、warmup 0.05、
max history 20、d=512、2 layers、4 heads、dropout 0.1、temperature 0.07；按 validation
Recall@10、NDCG@10 字典序选择 best checkpoint。

## 门控

1. 四个新 item-head 均通过既有 P9-2A scientific gate；
2. 六个 dataset-seed 单元的 PCRF Hit@10 delta 全部大于 0；
3. Toys 与 Beauty 各自三个 seed 的 median Hit@10 delta 均至少 `+0.002`；
4. 六个单元 NDCG@10 delta 均不小于 0；
5. 六个单元 tail Hit@10 delta 均不小于 0；
6. 六个单元 Hit@50 与 baseline 严格一致。

PASS 后进入第十一阶段 beam-width validation pilot；FAIL 时只分析 seed 方差，不补 seed、不调 PCRF、
不读取 test。
