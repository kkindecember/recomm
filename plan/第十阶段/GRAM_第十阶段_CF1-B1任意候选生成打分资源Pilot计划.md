# GRAM 第十阶段：CF1-B1 任意候选生成打分资源 Pilot 计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → run
- Origin Date: 2026-08-04
- Experiment ID: `GRAM_PHASE10_CF1_B1_TOYS_ARBITRARY_SCORE_PILOT_V1`
- Preregistration Status: `FROZEN_BEFORE_RUN`
- Scope: 512 deterministic Toys validation users

## 1. Research question

在 CF1-A2 已冻结的 `fill_cf_only_40`、每用户最多 90 candidates 下，B0 identity scorer 能否为
beam 外 CF-only lexical paths 产生合法、finite、资源可承受且与历史 G50 同口径的 GRAM 分数？

## 2. Frozen design

- 按 `sha256("2023:" + user_id)` 排序取前 512 validation users；
- 完整保留 cached G50，再按 item-head rank 回填最多 40 个 unique CF-only candidates；
- 使用 B0 已确认的全词表 teacher-forced path score；
- GPU5、float32、candidate micro-batch=10；
- 同一运行对 G50 做 cached score identity sentinel；
- 不训练模型、不拟合融合器、不读取 Toys test/Beauty/Sports。

## 3. Frozen gates

1. 512 用户均存在且每用户候选数在 `[50,90]`；
2. 所有 CF-only item 均能唯一映射到合法 lexical path；
3. 所有重算分数 finite；
4. G50 pooled Pearson/Spearman 均 `>=0.995`；
5. G50 mean top-10 set overlap `>=0.98`；
6. cached/recomputed G50 pilot Hit@10 absolute delta `<=0.001`；
7. peak allocated GPU memory `<=12000 MiB`；
8. wall time `<=600 s`；
9. 按 candidates/sec 线性外推 19,412-user full validation `<=4 h`。

任一失败均停止在 B1，不自动 retry、调 micro-batch、切换 adaptive slots 或启动全量。

## 4. Outputs and monitoring

- `summary.json`：identity、合法性、吞吐、显存和 full-run ETA；
- `candidate_scores.tsv`：user、candidate、source membership、GRAM score；
- `status.json`/`run.log`：process-alive、stage 与 hard timeout；
- 30 秒监控；hard timeout 30 分钟，预注册科学 wall gate 仍为 10 分钟。

B1 通过只授权 full validation scoring，不表示 CF1-C 排序效果已通过。

