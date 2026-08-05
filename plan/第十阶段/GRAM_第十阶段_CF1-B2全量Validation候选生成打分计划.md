# GRAM 第十阶段：CF1-B2 全量 Validation 候选生成打分计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → run
- Origin Date: 2026-08-04
- Experiment ID: `GRAM_PHASE10_CF1_B2_TOYS_FULL_SCORE_V1`
- Preregistration Status: `FROZEN_BEFORE_RUN`
- Scope: all 19,412 Toys validation users

## 1. Objective

为冻结的 `fill_cf_only_40` 候选集合生成完整 GRAM path scores，形成 CF1-C cross-fitted
calibration 的唯一 candidate-level 输入。本轮不训练、选择或评估融合器。

## 2. Frozen identities and gates

- users exact `19,412`；
- total candidates exact `1,698,905`；
- CF-only candidates exact `728,305`；
- union size 在 `[50,90]`，100% legal、100% finite；
- G50 Pearson/Spearman `>=0.995`、top-10 overlap `>=0.98`；
- G50 cached/recomputed Hit@10 absolute delta `<=0.001`；
- peak allocated `<=12,000 MiB`；
- wall time `<=4 h`。

前三项来自 CF1-A2 的完整候选构造 identity；评分 identity 来自 CF1-B0/B1。任一失败不得进入
CF1-C，不自动 retry、换 batch、换 slot 或事后删用户。

## 3. Resource protocol

- 使用本项目 CodeLlama 当前占用的 physical GPU6；
- 开跑前确认 CodeLlama 在 GPU6，随后正常 stop；
- 等待 GPU6 free memory `>=30,720 MiB`；
- 申请 30,720 MiB sidecar lease，工作负载预期 peak 8,192 MiB；
- 5 秒记录一次 GPU telemetry；
- 无论成功、失败或 timeout，runner 都尝试恢复 CodeLlama 到 GPU6；
- process hard timeout 4.5 h，科学 gate 4 h。

## 4. Outputs

- `candidate_scores.tsv`：约 170 万行，包含 user、union rank、candidate、source、GRAM score；
- `summary.json`：B2 identity、合法性、资源和 artifact hash；
- `gpu_telemetry.csv`、`gpu_lease.json`、`run.log`、`status.json`；
- Toys test、Beauty、Sports 均不读取。

