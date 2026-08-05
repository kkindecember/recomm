# GRAM 第九阶段 P9-X：Beauty 跨数据集外部确认计划

## 目的与证据边界

检验 Toys 上冻结的 PCRF 机制能否跨数据集成立。Beauty item embedding 不可从 Toys 迁移，故先按
P9-2A 完全相同的架构、seed、优化器和 10-epoch 规则，仅用 Beauty train-prefix 训练并按 Beauty
validation Recall@10/NDCG@10 选 item-head checkpoint。PCRF 公式与参数不得在 Beauty 上调整。

## 冻结配置

- Beauty GRAM：epoch-25 checkpoint；validation cache
  `GRAM/preds/20260722_125916_Beauty_sequential_pred_validation.tsv`；
- item-head：seed 2023、10 epochs、batch 512、AdamW 3e-4、weight decay 0.01、warmup 0.05、
  max history 20、d=512、2 layers、4 heads、dropout 0.1、temperature 0.07；
- PCRF：`lambda=1.0, beta=0.5, gamma=1.0`；
- popularity：只统计 `items[:-2]` train-prefix；q1 由 Beauty validation target frequency 冻结；
- 2,000 次 paired bootstrap；test 不参与训练、checkpoint 选择、参数或阈值选择。

## Validation admission

1. item-head 通过与 P9-2A 相同的相对 popularity gate；
2. PCRF Hit@10 delta ≥ 0.002；
3. paired bootstrap 95% CI lower > 0；
4. NDCG@10 delta ≥ 0；
5. tail Hit@10 不下降且 tail CI lower ≥ -0.002；
6. Hit@1 delta ≥ -0.001；
7. Hit@50 完全不变。

任一失败即停止，不读取 Beauty test，不修改公式或阈值。全部通过后，使用同一 Beauty item-head、
同一 PCRF 和 validation 冻结 q1，读取一次 Beauty test，并沿用相同 7 项 confirmation gates。

## 产物

- `artifacts/phase9/p9x_beauty_item_head/`
- `artifacts/phase9/p9x_beauty_validation/`
- 通过门控后才允许创建 `artifacts/phase9/p9x_beauty_test/`
