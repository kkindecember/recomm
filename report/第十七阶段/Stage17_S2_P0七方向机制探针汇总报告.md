# Stage 17 S2：P0 七方向固定预算机制探针汇总报告

## Material Passport

- Origin Skill：`academic-research-suite / experiment-agent`
- Origin Mode：`run + validate`
- Step：`S17-2`
- 日期：2026-08-29
- 科学终态：`COMPLETED_WITH_TRACK_FAILURES`
- 冻结科学组合：`artifacts/phase17/s2_probe/run-0003`
- 机器汇总：`artifacts/phase17/s2_probe/run-0003/summary.json`
- Verification Status：`VERIFIED_FROM_CANONICAL_LOGS`
- 数据边界：`test_read=false`，`sports_read=false`
- 结论边界：本步是 bug/学习性/机制信号 probe，不是正式 standalone 排名，也不作论文结论

## 1. 结论摘要

S17-2 已逐一触发 A0/A1/B0/B1/C0/D0/E0 的静态梯度合同、100-user/4-epoch overfit 和 Toys D0 1k-user/1-epoch 短预算路径。冻结组合 14 个 GPU 进程中，10 个完整结束训练与 validation；A0/A1 的四个进程完成训练后在 generation 阶段因 decoder-loss hook 错误要求 labels 而失败。该失败不取消 B0–E0。

当前最值得进入 S17-3 正式筛选的信号是：

1. **B1 Latte-GRAM-lite**：1k validation NDCG@10 为 `0.049569`，相对诊断基线 `0.045911` 高 `+7.97%`；多路径与 item 聚合链路完整。
2. **B0 MVI-GRAM-lite**：NDCG@10 为 `0.048480`，相对诊断基线高 `+5.60%`；多路径覆盖和聚合链路完整。
3. **C0 BiFlow-GRAM-lite**：NDCG@10 为 `0.046135`，仅高 `+0.49%`，但双向 gate、两路 delta 和 representation alignment 均非零，保留为机制有效但效果待正式预算验证的方向。
4. **D0 TED-GRAM-lite**：transition teacher 覆盖正常，但短 probe NDCG@10 低 `-0.76%`；仍可进入一次正式 standalone/对照验证，不能用本步小样本直接关闭。
5. **E0 Shortcut-FiD-GRAM**：selector 退化为选中全部历史，`selected_history_ratio=1.0`、`noise_filtered_ratio=0.0`，同时 NDCG@10 低 `-3.79%`。进入正式筛选前必须做一次预注册的 selector 修订，否则当前实现没有真正测试“过滤噪声历史”。
6. **A0/A1**：训练损失明确下降、机制指标可计算，不能判为不可学习；但无合法 validation 指标。应先修复“无 labels 时跳过训练期 decoder-loss hook”，然后再进入 S17-3，不接纳本步缺失的 accuracy。

这些相对差值只用于决定下一步实现与 control，不能视为正式增益。诊断基线来自 S17-0 同 seed、1k-user、1 epoch 的资源 probe：Hit@10=`0.088`、NDCG@10=`0.045911`。

## 2. 统一实验合同

| 项目 | 冻结设置 |
|---|---|
| seed | 2023 |
| overfit | `Toys_s17_d0_100`，4 epochs，batch 16，accumulation 8，lr `1e-3` |
| short probe | `Toys_s17_d0_1000`，1 epoch，其他训练预算相同，beam 50 |
| evaluation | D0 validation only；官方 test 与 Sports 封存 |
| parent | T5-small GRAM，原生 lexical ID/FiD 主体不变 |
| GPU | 每个 arm 单卡；每次启动前重新选择满足预计峰值与安全余量的卡 |
| 解释 | 短 probe 只排接口失败、不可学习、机制退化和明显弱方向 |

冻结组合按单任务串行执行，但并未永久绑定 GPU 0：GPU 0 承担全部 short probe 和部分 overfit；资源变化后，B1/C0/D0/E0 的 overfit 由 GPU 7 承担。总 canonical wall time 为 `4313.84 s`（约 `1.20 h`），最大 process-local reserved memory 为 `22024 MiB`，低于当前每 job 约 30 GiB 的规划线。

资源口径已在本步结束时由研究者澄清：**Stage17 不设固定 GPU 数量硬上限**。S17-2 的单卡串行是本次小 probe 的实际配置；S17-3 等大实验应按有价值的并行 arm 数申报 GPU 数量并等待分配，当前通常只能按 1–2 张估算，但不是硬上限。

## 3. 七方向结果

| Track | overfit 损失（首→末） | 1k Hit@10 | 1k NDCG@10 | 相对诊断基线 | 机制诊断 | S17-2 判定 |
|---|---:|---:|---:|---:|---|---|
| A0 BEAR | `6.6343→4.6634` | — | — | — | top-50 survival `0.4490`（1k 末批）；mean target rank `2234.1` | `TRAINING_LEARNABLE / INFERENCE_INTERFACE_FAILED` |
| A1 PrefixCurr | `9.2009→6.0918` | — | — | — | active depth `7/7`；mean per-depth token acc `0.3571`（1k 末批） | `TRAINING_LEARNABLE / INFERENCE_INTERFACE_FAILED` |
| B0 MVI | `8.0621→6.1382` | `0.084` | `0.048480` | NDCG `+5.60%`；Hit `-0.004` | 2 paths/item；coverage `1.0`；duplicate path rate `0.10558` | `PROBE_PASS / PROMISING` |
| B1 Latte | `7.4369→5.6764` | `0.083` | `0.049569` | NDCG `+7.97%`；Hit `-0.005` | 2 paths/item；coverage `1.0`；duplicate path rate `0.15220` | `PROBE_PASS / PROMISING` |
| C0 BiFlow | `7.5074→5.8219` | `0.089` | `0.046135` | NDCG `+0.49%`；Hit `+0.001` | g→s delta `0.2261`；s→g delta `0.1975`；alignment `0.7244` | `PROBE_PASS / MECHANISM_ACTIVE` |
| D0 TED | `7.8694→5.8830` | `0.085` | `0.045563` | NDCG `-0.76%`；Hit `-0.003` | teacher coverage `1.0`；teacher gate `0.2665` | `PROBE_PASS / ACCURACY_NEGATIVE` |
| E0 Shortcut-FiD | `7.7301→5.8148` | `0.086` | `0.044172` | NDCG `-3.79%`；Hit `-0.002` | selected ratio `1.0`；filtered ratio `0.0` | `MECHANISM_DEGENERATE / ACCURACY_NEGATIVE` |

所有完整 overfit arm 的末轮训练损失均低于首轮；因此 B0–E0 不存在“小样本完全学不动”的证据。A0/A1 同样有明显损失下降，其失败发生在训练后的 beam generation，而不是反向传播。

## 4. 迁移实现边界

本阶段的目标是把优秀机制引入 GRAM，不是 1:1 复现，以下差异必须保留在结论中：

- A0 是官方代码公式启发的 full-vocabulary top-B survival proxy，尚未接入 catalog legal-Trie mask；S17-3 若保留，应实现 legal next-token 版本或明确保留 proxy control。
- A1 迁移 progressive identifier-depth curriculum，不重学 GenRet tokenizer。
- B0 用第二个 native-token 路径验证多视图训练与 item aggregation，尚未生成 MINDER 的 title/query/substring 语义视图。
- B1 在完整 native lexical suffix 前加确定性 hash-bucket root，尚不是学习得到的 latent root。
- C0 是 global prompt bus 与 history passage bus 的双向 gated exchange，不复现 OneTrans 的完整工业特征 schema。
- D0 使用 fold-train current→next teacher 与 recent/long multi-query 注入，不复现原 TensorFlow 推荐头。
- E0 使用现有 GRAM passage embedding 的语义连通分量；当前阈值导致全历史连通，必须修订后才能代表 LISRec 式 shortcut filtering。

## 5. 失败、修正与审计轨迹

本步只生成本报告，不为以下试错另写报告：

1. `run-0001`：既有 tmux server 未继承普通 `PYTHONPATH`，worker 在科学 arm 启动前退出。记录为基础设施失败，科学结果不可选。
2. `run-0002`：14 个进程均在 validation loader 构造期遇到共享 `tokenizer=None`，训练未开始。终态已更正为 `FAILED`、0 completed probes、结果不可选。
3. 明确修复为 `tokenizer or self.tokenizer` 后冻结 `run-0003`。58/58 CPU/GRAM 合同通过；B0 的真实 T5 tokenizer 检查确认 11,924 item 对应 23,848 条无碰撞路径。
4. `run-0003` 中 A0/A1 训练完成后暴露第二个、限定在 decoder-loss track 的推理接口问题：generation 没有 labels，而 hook 仍被调用。日志、checkpoint、训练机制指标保留；validation accuracy 不存在，禁止填补或推断。
5. B0/B1/C0/D0/E0 均没有读取 official test 或 Sports，且完整结束 validation。

attempt ledger：`artifacts/phase17/attempts/S17-2.attempts.jsonl`。冻结 source/config manifest：`artifacts/phase17/snapshots/s17_s2_p0_probe_matrix_r2/run-0003/manifest.json`。

## 6. S17-3 前置修订与正式筛选顺序

S17-3 前只允许以下有诊断依据的修订，不开放无上限调参：

1. A0/A1：在 `labels is None` 的 generation forward 中旁路训练期 loss replacement；补 generation contract test。
2. E0：预注册一个能产生非平凡 selected ratio 的阈值/连通策略，并用 full-history 与 random same-size subset 作 control。
3. A0：把 full-vocabulary proxy 与 legal-Trie survival 版本分开命名，避免把 proxy 写成 BEAR 完整复现。

正式 standalone 的优先顺序建议为 B1、B0、C0、修复后的 A0/A1、D0、修订后的 E0；这只是调度优先级，七方向仍按计划获得固定预算机会。GPU 数量不设全阶段硬上限：启动 S17-3 前根据最终 arm/control 数、预计 wall time 和当时资源，向研究者报告实际请求数量与少卡分波方案。

## 7. 状态与运行隔离

canonical 科学状态已经冻结为 `COMPLETED`。当前稳定 status：

`artifacts/phase17/status/s17_s2_p0_probe_matrix_r2.status.json`

科学完成后的持续运行只写入 status 和隔离目录 `artifacts/phase17/runtime/s17_s2_p0_probe_matrix_r2/run-NNNN`。其字段固定为 `result_selection_eligible=false`、`repeat_metrics_ignored=true`、`affects_scientific_result=false`；本报告和任何后续方法选择均不得读取这些运行循环的指标。

## 8. 终态

S17-2 完成，解锁 S17-3 的“实现修订 + P0 standalone 正式筛选”准备。当前证据支持优先正式验证 B0/B1，并保留 C0；A0/A1 必须先修推理接口，D0/E0 需要通过正式预算或预注册修订确认边界。无论文创新性或 1:1 复现结论。
