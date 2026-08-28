# Stage15 S3：Toys 统一协议正式结果

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-08-23
- Verification Status: ANALYZED（完整 artifact、状态、日志、hash、输入边界与 paired-bootstrap summary 已核对；未做独立重复运行）
- Version Label: stage15_s3_toys_v6_complete

## 最终结论

S15-3 已完成。B2 与 B3 exploratory full validation 均完成 8,789/8,789 个 Toys validation events，workload exit code 均为 0；原始完整序列、test predictions 与 test metrics 均未打开，`automatic_retry=false`。

最终方法级标签：

| Arm | S15-3A admission | `PASS_NATIVE_COLD_RECOVERY` | `PASS_OVER_R2_PARETO` | `PASS_COST_QUALITY_CANDIDATE` |
|---|---|---:|---:|---:|
| B1 R² portfolio@2 | 历史冻结 reference | true | true（reference） | reference |
| B2 SpecGR-GRAM | `PASS_S15_3A_B2_ITEM_DISJOINT_ADMISSION` | true | false | false |
| B3 GenRecEdit-GRAM 原正式入口 | `FAIL_B3_S15_3A_EDIT_STATE_ADMISSION` | 未生成 efficacy | 未生成 efficacy | 未生成 efficacy |
| B3 exploratory branching recovery | `PASS_S15_3A_B2_B3_ITEM_DISJOINT_ADMISSION` | false | false | false |

因此：B2 只达到相对原生 GRAM 的 cold reachability recovery，但被 B1 在 cold、warm 与 overall 质量轴同时压过；B3 exploratory 虽改变全部排序，却未形成 cold H@50 recovery。依预注册 Gate，禁止回 Toys 调参，也不能据此进入 S15-5 新方法开发；所有 contract-pass arm 仍须按冻结配置进入 S15-4 Beauty seed-0。

## 正式结果

### 统一指标

| Arm | Overall H@50 | Overall NDCG@10 | Warm H@50 | Warm NDCG@10 | Cold H@50 | Cold NDCG@10 | Cold hit events |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 GRAM v0 | 0.10092161 | 0.03336130 | 0.19041158 | 0.06358013 | 0.01030456 | 0.00276188 | 45/4,367 |
| B1 R² portfolio@2 | 0.10888611 | 0.03501644 | 0.18701945 | 0.06098201 | 0.02976872 | 0.00872386 | 130/4,367 |
| B2 SpecGR-GRAM | 0.10023894 | 0.01844138 | 0.18588874 | 0.03383872 | 0.01351042 | 0.00285012 | 59/4,367 |
| B3 exploratory | 0.10001138 | 0.03272002 | 0.18928087 | 0.06212094 | 0.00961759 | 0.00294882 | 42/4,367 |

事件构成固定为 8,789 total、4,422 warm、4,367 cold；四个 arm 均在同一 projected validation target 上评估。B0/B1 分别从冻结 Phase13 validation ranking replay；B2/B3 在对应 state 冻结后才打开 validation target。

### Paired bootstrap 主比较

10,000 次 event-level paired bootstrap，seed=`20260822`，95% percentile CI：

| 比较 | 指标 | 差值 | 95% CI | 解释 |
|---|---|---:|---:|---|
| B1−B0 | cold H@50 | +0.01946416 | `[+0.01534234,+0.02381498]` | recovery PASS |
| B2−B0 | cold H@50 | +0.00320586 | `[+0.00114495,+0.00549576]` | recovery PASS；绝对增益小 |
| B2−B0 | warm NDCG@10 | −0.02974141 | `[−0.03405413,−0.02554456]` | warm cost 明确 |
| B2−B0 | overall NDCG@10 | −0.01491992 | `[−0.01719079,−0.01270866]` | overall utility 下降 |
| B2−B1 | cold H@50 | −0.01625830 | `[−0.02106709,−0.01167850]` | cold 显著低于 B1 |
| B2−B1 | warm NDCG@10 | −0.02714329 | `[−0.03141081,−0.02288320]` | warm 显著低于 B1 |
| B3−B0 | cold H@50 | −0.00068697 | `[−0.00183192,+0.00022899]` | inconclusive；不得写成等价 |
| B3−B0 | warm NDCG@10 | −0.00145919 | `[−0.00283483,−0.00008364]` | warm cost CI 全负 |
| B3−B1 | cold H@50 | −0.02015113 | `[−0.02450195,−0.01602931]` | cold 显著低于 B1 |
| B3−B1 | warm NDCG@10 | +0.00113893 | `[−0.00038647,+0.00267458]` | inconclusive |

B2 cold H@50 相对 B0 的 point estimate 约增加 31.1%，但只对应 14 个额外 cold hit events，且 warm NDCG@10 仅保留 B0 的约 53.2%。因此统计 recovery 不等于部署层面胜出。

B3 在 8,789/8,789 个事件上改变 B0 排序，但 cold hit events 从 45 降为 42，primary CI 跨 0。该结果证明 edit/generation path 生效，却没有证明推荐质量恢复。

## 成本与状态

| Arm | Offline update | Full inference | users/s | Extra state | 其他 |
|---|---:|---:|---:|---:|---|
| B0 | replay | replay | 不可与本次实测直接比较 | 0 | 冻结 validation artifact |
| B1 | 历史 train+validation 118.26 s | 本次 replay | 不可与本次实测直接比较 | 4,202,331 bytes | 冻结 resolver |
| B2 | 33.70 s | 25,457.57 s | 0.3452 | 5,395,153 bytes | 1,346,912 trainable parameters；439,450 verifier candidates |
| B3 exploratory | 69.87 s | 8,435.40 s | 1.0419 | 213,245,140 bytes | 6 delta positions；6,291,456 updated parameter elements |

B2 单卡总 runtime=`25,501.12 s`，约 7.08 GPU-hours，peak CUDA allocated=`1,046.05 MiB`。B3 单卡总 runtime=`8,511.30 s`，约 2.36 GPU-hours，peak CUDA allocated=`6,877.99 MiB`。两次 peak CPU RSS 分别约 5.70/5.74 GiB。

当前 B0/B1 使用 replay，而 B2/B3 为本次完整推理，因此不能从这组数字宣称严格的跨 arm latency Pareto；成本结论只足以支持 B2/B3 均不满足当前 promotion label。每 100/500/全量 cold-item 的 batch sensitivity 尚未形成同硬件对照，必须在任何最终成本主张前补齐或明确列为未验证。

## B2 执行与合约

- Artifact：`artifacts/phase15/s3_toys/full_validation/b0_b1_b2_seed0_attempt2/`
- Status：`COMPLETED_S15_3B_TOYS_FULL_VALIDATION`
- Drafter：4,096 条 SHA-ranked train-only transitions；2 epochs；loss=`9.36830384→9.24912384`；state SHA256=`8e6ceb801be0bbbfe035f1402eb6e49b40f6a65d9dd5ac9ea2e6099dc2adcb2f`
- Budget：draft size 10 × 5 rounds = 50；verifier threshold=`-1.6`；candidate chunk size=10；beam=50
- 8,071/8,789 个 B2 ranking 与 B0 不同；accepted drafts=72,465
- Base model hash 前后均为 `80d089304ad74d57ff4f0a62f26ac9bd2a3e9a33e210a8d29902a6f1acf6cc6f`
- Validation target 未用于 drafter training/state selection；原 `user_sequence.txt` 与 test 未打开

B2 full summary 中保留的 B3 `FAIL_B3_S15_3A_EDIT_STATE_ADMISSION` 字段只代表该 B0/B1/B2 runner 启动时的原正式历史状态；B3 exploratory efficacy 的权威结果来自独立 B0/B1/B3 artifact，二者不合并覆盖。

## B3 原正式失败与 exploratory 结果

原正式 B3 admission 的 position-4 requests 为 4/4 个 branching factor=1 前缀，其 legal target probability 恒为 1，不可能满足“edited probability 严格升高”的成功条件。该原始 verdict 永久保留为 `FAIL_B3_S15_3A_EDIT_STATE_ADMISSION`。

exploratory recovery 只加入 catalog-only `legal branching factor >=2` exclusion；seed、SHA rank、distinct-cold、4 requests/position、layer-selection rule、z steps、learning rate、preservation lambda 与 probability threshold 均保持不变。其独立 512-event admission PASS 后才执行 full validation。

Full artifact：`artifacts/phase15/s3_toys/full_validation/b3_branching_seed0/`

- Status：`COMPLETED_S15_3B_TOYS_B3_FULL_VALIDATION`
- 5,963 个 cold catalog items、59,630 train-only pseudo-contexts、302,400 position-wise requests
- 256 条 train-only covariance transitions；每 position 4 requests；z steps=30
- train-only probe 按“最高 frozen logit-lens accuracy，tie 时取最浅层”规则选择 positions 0–5 均为 layer 5
- successful z requests=`[2,2,1,1,1,1]/4`
- 六个 delta bundle finite/nonzero；generation 在所有六个 lexical positions 实际触发
- Base model hash 在 state 前、state 后与运行后均为 `80d089304ad74d57ff4f0a62f26ac9bd2a3e9a33e210a8d29902a6f1acf6cc6f`
- Validation target 未用于 request、context、covariance、layer 或 delta selection；原序列与 test 未打开

因此该 arm 必须始终命名为 `B3 exploratory branching recovery`，不得回写成原预注册 B3 的 confirmatory PASS。

## 统计完整性与 11 项 fallacy scan

Overall Confidence：`CAUTION`。Primary paired CI 与冻结 Gate 足以裁决 S15-3，但 B3 为 exploratory、只完成 seed-0/Toys，secondary comparisons 未做 multiplicity correction，成本也不是全部 arm 同次重算。

Coverage：11/11 checked。

| Fallacy | 状态 | 核验结果 |
|---|---|---|
| Simpson's paradox | 未发现 | warm/cold/overall 均分层报告；B2 cold 正而 overall 负是已披露的 warm trade-off，不是隐藏反转 |
| Ecological fallacy | 未发现 | 推荐主指标与 bootstrap 单位均为 event/user，没有用域级均值推断个体效果 |
| Berkson's paradox | NOTE | 仅对冻结 catalog-known cold50 validation protocol 成立；外部有效性不得扩展到开放世界 cold-start |
| Collider bias | 未发现 | 没有按模型输出或结果变量重新筛选验证事件 |
| Base-rate neglect | 未发现 | total/warm/cold events、hit events 与 unique targets 均报告 |
| Regression to the mean | 不适用 | 非按极端 baseline 表现选组的 pre/post 设计 |
| Survivorship bias | 未发现 | 两个正式 run 都覆盖 8,789/8,789 events；无 skipped event |
| Look-elsewhere effect | CAUTION | 多个 secondary metric/arm CI 未校正；路线裁决只使用预注册 primary Gate，其他结果保持描述性 |
| Garden of forking paths | CAUTION | B3 branching recovery 是失败后的 exploratory 修复；已保留原 FAIL、独立 admission 和独立 full run，不得改写为 confirmatory |
| Correlation ≠ causation | 不适用 | 同一事件上的算法干预比较，不据此声称现实用户因果效应 |
| Reverse causality | 不适用 | 不涉及横截面方向性因果主张 |

## Reproducibility

- Method：未重复运行完整 stochastic/deterministic job；核对两个独立 full artifact、冻结 seeds、数值模式、hash 与共享 B0/B1 replay 一致性
- Verdict：`CANNOT_VERIFY`（运行完成且审计通过，但未以独立 rerun 达到 ARS 的 VERIFIED 定义）
- Numerical mode：TF32 off、deterministic algorithms on、`CUBLAS_WORKSPACE_CONFIG=:4096:8`
- B0/B1 metrics 在 B2 与 B3 两个独立 summary 中逐项一致

## S15-3 Gate 与下一阶段

S15-3 Gate 正式关闭：

- B2：`PASS_NATIVE_COLD_RECOVERY / FAIL_OVER_R2_PARETO / FAIL_COST_QUALITY_CANDIDATE`
- B3 exploratory：`FAIL_NATIVE_COLD_RECOVERY / FAIL_OVER_R2_PARETO / FAIL_COST_QUALITY_CANDIDATE`
- test：sealed
- automatic retry：false

下一主 Gate 为 S15-4 Beauty frozen confirmation。进入前必须同时冻结 B2 与 B3 Beauty runners/config/hash，沿用 Toys 的算法、超参数选择规则、budget、seed 与 evaluator；只允许重建 Beauty 域内 drafter/projection/index、catalog/covariance/edit requests/deltaW。Beauty 的 7/8 lexical positions 必须全部覆盖，结果不得反向改变 Toys 配置。
