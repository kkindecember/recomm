# GRAM 第三阶段 LRC-UCRF F0 可学习性报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-07-22
- Verification Status: ANALYZED
- Version Label: lrc_ucrf_f0_v1
- Design Status: NEW PREREGISTERED CYCLE

## 1. 结论

LRC-F0 整体决定：**STOP**。本实验只检验 coverage reliability 是否可学习，不构成推荐效果结论。

| 数据集 | 模型 | Prevalence | AUROC | AUPRC lift | Brier 改善 | ECE | Active rate | Precision lift | Positive recall | Pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Toys | C2_hist_gradient_boosting | 10.427% | 0.6133 | 1.725× | +1.776% | 0.0466 | 26.180% | 1.572× | 41.156% | False |
| Beauty | C2_hist_gradient_boosting | 8.836% | 0.7064 | 3.045× | +12.992% | 0.1235 | 16.845% | 2.656× | 44.737% | False |

## 2. 数据与泄漏边界

训练/校准标签来自倒数第三次交互，validation 标签来自倒数第二次交互；最后一次 test 商品未使用。特征函数不接收 target，只读取历史与 SASRec top-20 邻居。用户哈希确定性划分 80%/20%。

## 3. 晋级规则

至少一个数据集未通过必要条件；不得实现或启动 LRC-S1，按计划转向方向 B。

## 4. 产物

- `artifacts/phase3/lrc_ucrf_f0/summary.json`
- `artifacts/phase3/lrc_ucrf_f0/model_metrics.csv`
- `artifacts/phase3/lrc_ucrf_f0/feature_schema.json`
- `artifacts/phase3/lrc_ucrf_f0/{Toys,Beauty}/dataset_summary.json`
