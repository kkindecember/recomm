# GRAM 第三阶段 S0b 可靠性拒绝探针报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-07-22
- Verification Status: ANALYZED
- Version Label: s0b_posthoc_v1
- Design Status: POST_HOC EXPLORATORY AMENDMENT

## 1. 设计边界

本探针在 S0 结果后提出，只使用锁定的 Beauty/Toys validation 预测和推理时可得的关系置信特征。没有读取 test、没有训练模型、没有使用目标商品构造 abstention。网格在运行前固定为 16 个共同配置。

## 2. 共同配置结果

整体决定：**STOP**。通过全部跨数据集门槛的配置数：0 / 16。

诊断最优配置：`b0_l0.2_t0.75_s2`（beta=0.0、lambda=0.2、tau=0.75、min_support=2）。

| 数据集 | Active rate | NDCG@10 相对变化 | Recall@10 绝对变化 | Tail NDCG@10 | Uncovered Recall@10 | Uncovered NDCG@10 | Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| Toys | 83.598% | +0.942% | +0.001494 | +2.569% | -1.208% | -2.881% | False |
| Beauty | 90.498% | +0.067% | -0.000402 | +2.632% | -6.329% | -9.502% | False |

## 3. 晋级解释

没有共同配置通过全部门槛。按修订计划停止 UCRF-v1 offline path，不得扩大网格或直接启动 S1。下一步需重新预注册 learned-gate 周期，或转向优先级 2。

## 4. 产物

- `artifacts/phase3/s0b/grid_metrics.csv`
- `artifacts/phase3/s0b/joint_configs.csv`
- `artifacts/phase3/s0b/summary.json`
- `artifacts/phase3/experiment_registry.csv`
- `artifacts/phase3/promotion_decisions.md`
