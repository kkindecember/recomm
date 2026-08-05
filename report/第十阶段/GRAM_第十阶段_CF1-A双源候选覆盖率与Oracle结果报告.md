# GRAM 第十阶段：CF1-A 双源候选覆盖率与 Oracle 结果报告

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-08-04
- Verification Status: `COMPLETED_WITH_FAILED_FROZEN_GATE`
- Experiment ID: `GRAM_PHASE10_CF1_A_TOYS_CANDIDATE_UNION_V1`
- Evidence Class: Toys validation candidate-coverage diagnostic
- Test Read: false
- GPU Used: false

## 1. Executive conclusion

CF1-A 工程执行成功，但预注册的四项联合科学 gate 未全部通过，因此正式状态为
`failed_candidate_union_gate`，不得写成 CF1-A 全面通过。

结果同时给出一个很明确的机制判断：**item-head 确实补回了大量 GRAM beam 外 target，候选扩展
方向成立；直接做 50+50 全并集则超出预设候选预算，下一步需要升级候选融合/预算分配，而不是
继续微调 beam 内 PCRF。**

GRAM G50 coverage 为 `0.211931`，加入 C50 后 U50 coverage 达到 `0.266691`，绝对提升
`+0.054760`，超过冻结的 `+0.030` 门槛。tail 中 C50-not-G50 complementary coverage 为
`0.023450`，也超过 `0.020` 门槛。然而原始 U50 平均包含 `90.75` 个候选，只有 `36.79%`
用户满足 union size `<=90`，显著低于 `80%` 资源门槛。

## 2. Integrity and execution

- 仅读取 Toys validation：19,412 users、11,924 items；未读取 Toys test、Beauty 或 Sports；
- validation cache、item-head checkpoint、user sequence、item mapping 和执行代码均通过预注册
  SHA256 校验；
- 每用户 GRAM beam 均为 50 个 unique legal items；
- runner 单次正式执行完成，4 个单元测试通过；CPU wall time `7.73 s`；
- 未改写 checkpoint，未占用 GPU，也未在结果后自动 retry 或修改冻结阈值。

## 3. Primary candidate coverage

| candidate set | coverage | vs G50 |
|---|---:|---:|
| G50 | 0.211931 | — |
| C10 | 0.090150 | — |
| C20 | 0.121574 | — |
| C50 | 0.174531 | — |
| G50 ∪ C10 | 0.225479 | +0.013548 |
| G50 ∪ C20 | 0.237121 | +0.025191 |
| G50 ∪ C50 | 0.266691 | +0.054760 |

U50 命中 5,177/19,412 个 validation targets，G50 命中 4,114 个；因此 1,063 个 target 是
item-head 单独补回的。该数值是候选 oracle coverage，不是实际 reranked Hit@10。

## 4. Complementarity and budget

| diagnostic | result |
|---|---:|
| G50/C50 intersection mean | 9.25 |
| CF-only candidates/user mean | 40.75 |
| Jaccard mean | 0.1094 |
| raw union size mean / median | 90.75 / 93 |
| raw union size p80 / p90 | 97 / 98 |
| raw union size `<=90` fraction | 0.3679 |
| history-filtered union mean | 87.66 |
| history-filtered union `<=90` fraction | 0.5457 |
| CF-only candidates requiring GRAM scoring | 791,057 |

两路候选重叠很低，这正是 coverage 增益大的原因，也是 naive 50+50 成本过高的原因。即使移除
历史已交互 item，`<=90` 比例也只有 `54.57%`，仍不能满足冻结预算 gate。

## 5. Stratified results

| target group | n | G50 | C50 | U50 | C50-not-G50 |
|---|---:|---:|---:|---:|---:|
| tail | 5,160 | 0.138953 | 0.076163 | 0.162403 | 0.023450 |
| middle | 9,235 | 0.204981 | 0.175528 | 0.255225 | 0.050244 |
| head | 5,017 | 0.299781 | 0.273869 | 0.395057 | 0.095276 |

tail complementary gate 以很小但明确的余量通过；绝对互补更集中在 middle/head。这说明后续候选
裁剪不能简单按 item-head raw score 或流行度保留，否则可能再次牺牲 tail。

| history length | n | G50 | C50 | U50 | C50-not-G50 |
|---|---:|---:|---:|---:|---:|
| 1–5 | 12,673 | 0.212183 | 0.167285 | 0.257713 | 0.045530 |
| 6–10 | 4,319 | 0.202130 | 0.167400 | 0.258393 | 0.056263 |
| 11–20 | 2,420 | 0.228099 | 0.225207 | 0.328512 | 0.100413 |

item-head 的互补价值随历史长度上升，长历史用户最值得分配额外 CF candidate budget。

## 6. Frozen gate accounting

| frozen check | threshold | observed | status |
|---|---:|---:|---|
| U50 − G50 coverage | >=0.030 | +0.054760 | PASS |
| tail C50-not-G50 | >=0.020 | 0.023450 | PASS |
| fraction union size <=90 | >=0.80 | 0.367917 | FAIL |
| C50 identity | exact 0.174634247 | 0.174531218 | FAIL |

C50 identity 相差 `0.000103029`，即净 2/19,412 个 hit。逐用户与 P9-2A 冻结
`best_validation_ranks.tsv` 对齐后，旧评测 3,390 hits、本轮 3,388 hits，共 4 个 top-50 边界
样本发生方向翻转；用户集合、顺序、target 和 checkpoint hashes 均一致。P9-2A 原结果来自 CUDA
rank 评测，本轮候选枚举为 CPU `topk`，因此最合理解释是 top-50 边界的跨设备/检索口径数值漂移。
这不改变覆盖互补的数量级，但按预注册 exact identity 规则必须记 FAIL，不能事后放宽 tolerance。

## 7. Decision

1. 不按原方案直接进入 full U50 arbitrary-candidate GRAM scoring：候选预算 gate 未通过；
2. 不回到 PCRF 小网格继续调参：CF1-A 已证明主要新增空间在 beam 外；
3. 下一步应先预注册 **budgeted adaptive union**：保留 G50，并按可观测 reliability/history/
   source-agreement 为用户分配 CF-only slots，使总候选硬上限固定在 90；
4. 在该预算版本重新做 validation-only coverage/oracle，并把 tail complementary retention 和
   U50-oracle retention 作为核心 gate；
5. 只有预算化 union 通过后，才执行 CF1-B score identity 与 constrained GRAM rescoring pilot。

该决定属于对 CF1-A 失败原因的机制修正，不把本轮 oracle coverage 冒充最终推荐指标，也不消耗
任何新的 test 机会。

## 8. Reproducibility pointers

- preregistration：`artifacts/phase10/configs/cf1_a_toys_candidate_union_preregistered.json`
- summary：`artifacts/phase10/cf1_a_toys_candidate_union/summary.json`
- per-user evidence：`artifacts/phase10/cf1_a_toys_candidate_union/per_user_coverage.tsv`
- run log：`artifacts/phase10/cf1_a_toys_candidate_union/run.log`
- evaluator：`experiment/phase10/eval_cf1_a_candidate_union.py`
- runner：`experiment/phase10/run_phase10_cf1_a_candidate_union.sh`

