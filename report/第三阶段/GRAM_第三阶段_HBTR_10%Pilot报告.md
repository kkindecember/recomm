# GRAM 第三阶段 HBTR 10% Pilot 报告

- 决策：**STOP**
- 原因：8 preregistered gates failed
- 证据级别：探索性机制筛选；不允许效果声明；全程未读取测试目标。

## 主结果

| 数据集 | C0 NDCG@10 | C4 NDCG@10 | 相对变化 | C0 Recall@10 | C4 Recall@10 |
|---|---:|---:|---:|---:|---:|
| Toys | 0.079372 | 0.079882 | +0.64% | 0.123047 | 0.124512 |
| Beauty | 0.067215 | 0.067174 | -0.06% | 0.114746 | 0.114258 |

## 预注册门槛

- [PASS] `protocol_integrity`：observed=[]; threshold=[]
- [PASS] `Toys_C4_vs_C0_ndcg10_positive`：observed=0.0005104749914479761; threshold=> 0
- [PASS] `Toys_C4_vs_C0_recall10_no_decline`：observed=0.00146484375; threshold=>= 0
- [PASS] `Toys_tail_Recall@10_relative_decline`：observed=0.0074074074074074155; threshold=>= -0.01
- [PASS] `Toys_tail_NDCG@10_relative_decline`：observed=0.0021670911819155884; threshold=>= -0.01
- [FAIL] `Beauty_C4_vs_C0_ndcg10_positive`：observed=-4.09704644397102e-05; threshold=> 0
- [FAIL] `Beauty_C4_vs_C0_recall10_no_decline`：observed=-0.00048828125; threshold=>= 0
- [FAIL] `Beauty_tail_Recall@10_relative_decline`：observed=-0.016666666666666635; threshold=>= -0.01
- [FAIL] `Beauty_tail_NDCG@10_relative_decline`：observed=-0.011985687492931638; threshold=>= -0.01
- [FAIL] `at_least_one_dataset_C4_ndcg10_relative_gain_2pct`：observed={'Toys': 0.006431456916108329, 'Beauty': -0.0006095408323185979}; threshold=>= 0.02 for at least one dataset
- [PASS] `C4_macro_ndcg10_exceeds_C1`：observed=5.584436956723726e-06; threshold=> 0
- [FAIL] `C4_macro_ndcg10_exceeds_C2`：observed=-3.871729515492195e-06; threshold=> 0
- [PASS] `C4_macro_ndcg10_exceeds_C3`：observed=2.250910539583484e-05; threshold=> 0
- [PASS] `Toys_C4_within_0.5pct_best_component`：observed=0.0; threshold=>= -0.005
- [PASS] `Beauty_C4_within_0.5pct_best_component`：observed=-0.0002814618694210787; threshold=>= -0.005
- [FAIL] `Toys_peak_reserved_increase`：observed=0.8560072677719737; threshold=<= 0.25
- [PASS] `Toys_training_wall_time_increase`：observed=0.09894665742922461; threshold=<= 1.0
- [PASS] `Toys_validation_latency_increase`：observed=0.0028138334126683877; threshold=<= 0.05
- [FAIL] `Beauty_peak_reserved_increase`：observed=0.6130859808931348; threshold=<= 0.25
- [PASS] `Beauty_training_wall_time_increase`：observed=0.11319744416958534; threshold=<= 1.0
- [PASS] `Beauty_validation_latency_increase`：observed=0.009876082447065837; threshold=<= 0.05

## 解释边界

该 pilot 是从锁定全量基线继续训练的 10% 用户机制筛选，不是独立重复实验，也未使用测试集。Bootstrap 区间仅描述锁定验证用户上的配对不确定性，不构成确认性显著性结论。
