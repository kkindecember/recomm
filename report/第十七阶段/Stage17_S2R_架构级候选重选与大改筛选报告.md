# Stage 17 S17-2R：架构级候选重选与大改筛选报告

## Material Passport

- Origin Skill：`academic-research-suite / experiment-agent`
- Origin Mode：`run closeout + canonical evidence validation`
- Step：`S17-2R`
- 日期：2026-08-31
- 科学终态：`COMPLETED_NO_R3_CANDIDATE`
- R1 汇总：`artifacts/phase17/s2r_r1/r1_summary.json`
- R2 profile：`artifacts/phase17/s2r_r2/profile/run-0001/profile_summary.json`
- R2 正式筛选：`artifacts/phase17/s2r_r2/screen/run-0001/screen_summary.json`
- LATTE 唯一修订：`artifacts/phase17/s2r_r2/latte_revision/run-0001/revision_summary.json`
- Attempt ledger：`artifacts/phase17/attempts/S17-2R.attempts.jsonl`
- Verification Status：`VERIFIED_FROM_FROZEN_PREDICTIONS`
- 数据边界：Toys D0 shadow fold；`official_test_read=false`；`sports_read=false`；`d1_read=false`

## 1. 结论

S17-2R 已完成 R0 静态审计、四个 P0 family 的 R1 端到端契约、R2 容量 profile 和 3,000-user 正式筛选。DiffGRM、Gryphon-item、SETRec-full 在首轮 R2 被拒绝；LATTE-full 获得预注册的唯一一次诊断修订，但修订后仍未达到强准入门槛，最终为 `REJECT_AFTER_ONE_REVISION`。

因此：

- `r3_eligible=[]`，不运行 full Toys D0 R3；
- 没有架构解锁新的 S17-5 D1 准入；
- S17-5 保持 `HOLD_NO_S2R_PROMOTION`；
- official test、Sports、Toys D1 和 Beauty D1 继续封存；
- GPU1 上 S17-4 成功后的隔离非科学重复轮未被停止，其指标未进入本报告。

这是一项有信息量的负结果：当前缩放预算与统一 evaluator 下，四类架构变化都没有同时满足效果、机制和稳健性门槛。不能继续调 LATTE，也不能因 SETRec 的点估计接近阈值而补发一次未注册修订。

## 2. 数据与执行合同

- 从 Toys D0 shadow fold 以 `sha256(s17-2r:<seed>:<user_id>)` 固定选择 3,000 用户；seed=`2023`。
- 三个 evaluation cohort 互斥，每个 1,000 用户；用户列表和 SHA-256 已冻结。
- item catalog 共 11,924 items，训练、生成、item resolution 与 evaluator 使用同一 catalog。
- 每个 family 的 treatment 与 native control 使用相同训练用户、seed、容量级、优化机会和 item-level evaluator。
- 正式外部 cohort 只在 best checkpoint 冻结后评测一次；保存逐用户 prediction 和 paired user metrics。
- 所有正式生成结果的 `valid_item_rate=1.0`；R1 的 treatment-specific gradient、loss 下降、prediction 非空和禁止 future context 检查均通过。

R0 来源审计未执行第三方代码。LATTE 仓库许可为 MIT；DiffGRM、SETRec、DIGER 的许可/下载审计不足以允许复制时，执行路径使用本仓库的独立缩放实现。Gryphon-item 也为根据论文机制独立实现。因而以下结果是本地 matched architecture contrast，不是论文官方代码或全尺度复现。

## 3. R1 与 R2 profile

R1 最终四个 family 全部通过。初始 R1 第二波曾因 `latte_aggregation` 参数跨入 parallel decoder 接口而失败；`run-0003` 只修复接口分发，科学配置未变，也没有把失败运行计入效果结果。

R2 容量 profile 的最大 process-local CUDA allocated memory 为 `2401.31 MiB`；各 arm 参数量约 `5.36M–8.12M`。LATTE beam-200 修订 profile 首次因共享 GPU 可用显存收缩而缺失终态工件，登记为 engineering failure；恢复仅把 evaluation batch size 从 8 降到 4，checkpoint、beam=200、top-k=50 与排序均未改变。恢复 profile 最大 allocated memory 为 `2336.13 MiB`。

这些资源数字只描述当前独立缩放实现，不能外推到论文原生 full-scale 配置。

## 4. R2 正式 3k 筛选

主门槛为 mean `ΔNDCG@10 >= +0.0015`、至少两个 cohort 为正、mean `ΔHit@10 >= 0`、任一 cohort `ΔHit@10 >= -0.002`，且预注册机制指标通过。置信区间为 1,000 次 paired user bootstrap。

| Family | ΔNDCG@10 | paired 95% CI | ΔHit@10 | 正向 NDCG cohort | 机制门 | 首轮决定 |
|---|---:|---:|---:|---:|---|---|
| Gryphon-item | -0.001990 | [-0.003748, -0.000060] | -0.003333 | 0/3 | FAIL | `REJECT` |
| LATTE-full | +0.000777 | [-0.000709, +0.002295] | +0.001667 | 2/3 | PASS | `BORDERLINE_ONE_REVISION` |
| DiffGRM | -0.002883 | [-0.004285, -0.001500] | -0.006667 | 0/3 | PASS | `REJECT` |
| SETRec-full | +0.001422 | [-0.000319, +0.003267] | +0.003000 | 3/3 | FAIL | `REJECT` |

机制解释：

- Gryphon 的 treatment/control candidate sets 一致，但 106 个可比较 target user 的 mean target rank gain 为 `-1.1604`，联合 item scorer 没有改善排序。
- DiffGRM 的 masked generation 明显快于 AR control（`7.23s` 对 `344.89s`），但效果显著负向；速度机制成立不等于推荐质量成立。
- SETRec 的并行生成明显快于 AR control（`13.01s` 对 `375.62s`），但 `set_token_recovery=0`，预注册机制门失败；其 NDCG 点估计还低于 `+0.0015` 且 CI 跨 0，因此不满足修订条件。
- LATTE 的 multi-path 机制成立，但 beam=50 时 mean unique candidates 只有 `14.33/50`，形成了有诊断依据的唯一修订。

## 5. LATTE 唯一正式修订

修订不训练新模型，不更换 identifier/checkpoint/aggregation，只对 LATTE 与 PSID control 同时把 beam 从 50 增到 200，并保持 top-k=50。两个冻结 checkpoint 和四个正式 prediction/metric 文件在 closeout 前重新核对 SHA-256；每个 arm 都有 3,000 条 prediction。

| 指标 | 修订结果 |
|---|---:|
| mean ΔNDCG@10 | +0.001018 |
| paired 95% CI | [-0.000569, +0.002708] |
| mean ΔHit@10 | 0.000000 |
| 正向 NDCG cohort | 3/3 |
| cohort ΔHit@10 | c0 -0.001；c1 +0.002；c2 -0.001 |
| valid item rate | 1.0 |
| mean unique candidates | 48.001/50 |
| multi-path item rate | 0.7094 |

beam-200 成功把 unique candidate coverage 从 `14.33` 恢复到 `48.00`，证明原诊断成立；但 mean `ΔNDCG@10` 仍低于 `+0.0015`，paired CI 仍跨 0。比较器给出第二次 borderline 信号，而 family 的修订预算已经消耗，因此按冻结规则落为 `REJECT_AFTER_ONE_REVISION`，不得再修。

## 6. 完整性与去重

- R2 正式首轮只存在 `r2-screen-0001`；LATTE 正式修订只存在 `r2-latte-revision-0001`。
- 本次续跑没有重新启动任何 R1/R2 GPU arm；只对已经完成的两个 LATTE revision arm 执行 CPU closeout。
- closeout 前定向回归测试 `7/7 passed`；终态后完整 Phase17 test suite 为 `131 passed, 0 failed`。
- LATTE revision 的 checkpoint、prediction、user metrics 与 run snapshot 源码/配置 SHA-256 均匹配冻结记录。
- 初始 profile failure、R1 接口 failure 及其 recovery 均保留在 append-only attempt ledger；失败工件没有被覆盖或伪装成新科学结果。
- 当前 dirty worktree 中的其他 Phase 16/17 变更不作为复现依据；每次正式运行以 immutable snapshot 为准。

## 7. 终态与下一动作

`S17-2R COMPLETED_NO_R3_CANDIDATE`。

本计划内没有可继续自动执行的科学 arm：R3 的前提是至少一个 R2 strong promotion，而当前为 0。继续新架构、放宽阈值、扩大 LATTE beam、给 SETRec 补修订或打开 D1 都会超出已冻结计划并改变研究问题，必须由研究者另行授权并建立新计划/新 step；不能伪装成 S17-2R 的继续运行。
