# GRAM 第六阶段：F0 多源候选覆盖与 Oracle 审计实验计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Created: 2026-08-03
- Verification Status: `EXECUTED_STOP_CANDIDATE_DRAFTING_WITH_PROTOCOL_DEVIATIONS`
- Version Label: `phase6_f0t_trained_drafter_coverage_oracle_v1`
- Parent decision: GACR-v8 closed the fixed-candidate residual growth line
- Governance: `plan/GRAM_后续结构性方向分阶段实验治理规则.md`
- Stage: `P0-T` — trained independent drafter plus coverage/oracle audit

## 1. 授权变更与唯一结构假设

原 F0 假定已存在可复用的独立 drafter checkpoint 与候选 cache；实际审计确认这些 artifact 缺失，只有无法对齐当前 cohort 的历史汇总，不能构成 F0 证据。研究者于 2026-08-03 明确授权训练一个新的独立 drafter。因此本计划是对原 F0“无训练”约束的**具名一次性例外**，不改变 v8 结论，也不授权 verifier、融合模型或后续 F1/F2。

当前 GRAM-beam + catalog projection union 的主要瓶颈是候选覆盖而非候选内 residual 重新排序。一个冻结、独立的 sequence/full-catalog drafter 若能在不移除任何原候选的前提下提供可重复的 target 独占覆盖，则有资格进入 F1 多源 drafter 资格实验；否则停止 candidate drafting，不训练 verifier。

## 2. 冻结范围

- Domains：Toys、Beauty development split；Sports/test 禁读。
- Seed：`2023`；仅 P0-T，不创建 fresh validation cohort。
- Drafter：单一 SASRec（hidden=`64`、blocks=`2`、heads=`2`、max history=`50`、dropout=`0.2`），full-catalog softmax top-50。只训练一次，Adam `lr=0.001`、weight decay=`0`、batch=`256`、最多 10 epochs；不进行 epoch/容量/lr/dropout 搜索。
- 训练协议：用户 sequence 的 `[:-2]` 为可见训练 prefix，`[-2]` 为隔离 calibration target，`[-1]`（test）不得索引、不得训练、不得评估。训练中的 10% 内部 holdout 固定 salt=`phase6-f0t-drafter-epoch-v1`，仅用于在 epochs 1–10 中选择一个共享 epoch（macro NDCG@10 最大，平手取更早 epoch）；该内部 holdout 不与外部 calibration 混用。
- Sources：冻结 GRAM constrained beam top-50、冻结 GRAM catalog projection top-50、上述 SASRec top-50。
- 每个 source 先去 history item、未知 item 与重复 item；合并采用稳定 insertion order。F0 不训练、不拟合融合权重、不执行 GRAM teacher forcing、不改变 GRAM 或 drafter checkpoint。
- 比较预算固定为每 source 50；报告 union(beam,catalog)、union(beam,catalog,drafter) 以及每个单源。不得因观察结果扩充 top-k。

## 3. 必须回答的审计量

对相同、冻结的 development users 逐用户报告：

1. target 不在原 GRAM union 的比例、在 union 但不在 top-10 的比例；
2. beam、catalog、drafter 的 target coverage、两两交集、三者交集和 drafter 独占命中；
3. 原 union 与扩展 union 的 Recall@10/50 coverage，以及“target 在候选中时排第一”的 oracle NDCG@10/Recall@10 上限；
4. head/tail、短/长 history、低/高 GRAM-confidence strata 的上述覆盖缺口；
5. 去重率、history/unknown filter 数、每 source 和 union 的候选大小、CPU/GPU latency。

Oracle 仅表示候选供给上限；不得作为最终 recommender 指标或与 v3 的实际排序结果混写。

## 4. 预注册 P0 机制门

外部 calibration 不参与训练或 epoch 选择；drafter 只有同时满足以下条件才允许进入 F1：

- Toys、Beauty 的 drafter 独占 Recall@50 coverage 均 `>0`，且每域至少有 10 个独占 target users；
- 两域扩展 union 相对原 union 的 Recall@50 coverage 均 `>= +1.0pp`；
- 至少一域的扩展 union Recall@50 coverage `>= +2.0pp`；
- 两域 tail Recall@50 coverage 均 `>= +0.5pp`；
- 无 target leakage、无 history/unknown item 泄漏、source-to-item 映射一一对应、所有计数可复算。

这些是 P0 的最低机制信号，不是 F1 的最终资格门。F1 仍须使用预先定义的更严格 `+2pp` 双域、至少一域 `+3pp`、tail `+1pp` 要求。

## 5. 停止与后续规则

- P0 失败：`STOP_CANDIDATE_DRAFTING; DO_NOT_TRAIN_VERIFIER`。转入独立的 backbone/identifier alignment 研究问题；不得靠更多 candidate source、top-k 网格或后验融合权重救援。
- P0 通过：只允许写入 F1 详细计划并锁定 checkpoint、candidate budget、用户 cohort、资格门和资源上限；不得直接训练 F2 verifier。
- F0 不自动启动 F1/F2、不读取 Sports/test、不修改 GACR-v3 incumbent。

## 6. 训练、实现与完整性门

运行前必须创建并冻结：

- `experiment/phase6/f0_multisource_coverage_oracle.py`、对应测试与 runner；
- `artifacts/phase6/configs/f0_multisource_coverage_oracle_preregistered.json`；
- `artifacts/phase6/f0_multisource_coverage_oracle/summary.json`、per-user CSV、source-overlap CSV、status/log/telemetry；
- `report/第六阶段/GRAM_第六阶段_F0多源候选覆盖与Oracle审计报告.md`。

测试至少覆盖 prefix 截断（`[-1]` test 永不见）、internal holdout 与 external calibration 隔离、target-free source construction、history/unknown filtering、stable dedup、source attribution、oracle 定义、strata completeness、Sports/test source guard，以及不加载 verifier/GRAM optimizer。运行前锁定数据 split、代码、测试、runner 与 config SHA256；运行后锁定所选 epoch、drafter checkpoint、candidate cache 与 cohort SHA256。

## 7. 资源与授权

预期优先复用缓存；本次授权将 workload 与 CodeLlama 恢复位置固定为 GPU6，项目离线 HuggingFace cache 固定为 `.cache/huggingface`。租约维持 `30,720 MiB` 总额（workload `24,576 MiB`、sidecar `6,144 MiB`）和 48h hard timeout。所有退出路径恢复 CodeLlama；非零退出、完整性失败或 P0 门失败均不自动重试。

## 8. 强制决策记录

```text
阶段：P0-T
唯一结构假设：独立 full-catalog sequence drafter 可补足固定 GRAM union 的 coverage。
固定 seed/cohort：2023；Toys/Beauty external calibration，各 128 head + 128 tail。
直接机制指标：drafter 独占 Recall@50、扩展 union Recall@50、tail Recall@50、oracle 上限。
最低有效信号：第 4 节四项条件全部满足。
通过后唯一下一步：撰写并冻结 F1 多源 drafter 资格实验。
失败后停止项：不训练 verifier；不增加 source/top-k/融合权重。
禁止的邻近补丁：epoch/容量/lr/dropout 搜索、target-conditioned drafting、读 fresh/Sports/test。
Sports/test read：false / false
```

## 9. 执行后状态

P0-T 已于 2026-08-03 完成。Toys/Beauty 的 SASRec 独占 target users 分别为
6/7，均低于每域 10 人门槛，因此固定决定为
`STOP_CANDIDATE_DRAFTING; DO_NOT_TRAIN_VERIFIER; F1_NOT_UNLOCKED`。

实现对 extended union@50 采用“先追加 SASRec，再截前 50”，使新候选无法进入该
字段；且未产出计划要求的全部 overlap/strata/filter/latency 产物。这些偏离已在结果报告
中明示登记。停止决定仅依赖可直接复算的独占用户门，不依赖有偏的 union@50
零增益。完整报告见
`report/第六阶段/GRAM_第六阶段_F0T多源候选覆盖与Oracle审计报告.md`。
