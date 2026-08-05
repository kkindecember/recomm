# GRAM 第十阶段：CF1-A2 预算化自适应候选融合实验计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → run
- Origin Date: 2026-08-04
- Experiment ID: `GRAM_PHASE10_CF1_A2_TOYS_BUDGETED_UNION_V1`
- Preregistration Status: `FROZEN_BEFORE_FULL_RUN`
- Dataset Boundary: Toys validation only；Toys test/Beauty/Sports closed

## 1. Motivation

CF1-A 证明 C50 可在 G50 外补回 `0.054760` coverage，但 naive U50 只有 `36.79%` 用户满足
候选数 `<=90`。A2 不修改 item-head/PCRF，也不训练新模型；只检验 target-blind 的候选预算化
能否在硬上限 90 内保留大部分互补覆盖。

## 2. Frozen primary policy

主策略固定为 `fill_cf_only_40`：

1. 完整保留有序 G50；
2. 按 item-head rank 扫描 C50；
3. 跳过已在 G50 中的 item；
4. 最多追加 40 个 unique CF-only items；
5. 最终每用户候选数严格 `<=90`。

该过程不读取 target、target frequency 或 hit label。结果后不从诊断策略中改选 primary。

## 3. Diagnostic policies

- `fixed_top_{10,20,30,40}`：G50 与 C top-k 普通并集，不回填 overlap；
- `fill_cf_only_{10,20,30}`：不同 CF-only slot budgets；
- `adaptive_history`：history 1–5/6–10/11–20 分配 25/30/40 slots。

adaptive_history 只回答能否用更低平均成本保留收益；本轮不把 target-popularity 用作分配特征。

## 4. Frozen scientific gates

以同一 CPU evaluator 重算的 CF1-A G50/U50 为 identity reference，primary 必须同时满足：

1. G50 coverage exact identity：`0.21193076447558212`；
2. U50 coverage exact identity：`0.2666907067793118`；
3. 100% 用户 union size `<=90`；
4. 保留至少 80% 的 `(U50−G50)` coverage gain；
5. 保留至少 80% 的 U50 tail C50-not-G50 complementary coverage。

任一失败则 A2 记 `failed_budgeted_union_gate`，不自动 retry、改 slot 或切换 primary。

## 5. Execution and outputs

- CPU-only，预计全量低于 1 分钟，hard timeout 30 分钟；
- 先执行 unit tests、compile、bash syntax 和 SHA256 locks；
- 正式输出：`summary.json`、`per_user_budget.tsv`、`run.log`、`status.json`；
- smoke 只验证工程，不计算科学 gate；
- A2 通过后才恢复 CF1-B constrained GRAM score identity/resource pilot。

