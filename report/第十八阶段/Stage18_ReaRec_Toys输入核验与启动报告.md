# 第十八阶段：ReaRec Toys 输入核验与启动报告

**停止更新（2026-09-09 16:28:45）：** 用户已要求停止 ReaRec Toys，PID `3020613` 已退出，状态 `USER_STOPPED`；最后完整 checkpoint 为第 34 轮 / 7,276 updates，第 35 轮中断。未完成最小训练预算或全量验证，不能报告为完整负结果。最佳 checkpoint、预测和中断记录均保留。详见 [ReaRec 完整结果与 Toys 主动停止报告](Stage18_ReaRec_Beauty完整结果与Toys主动停止报告.md)。下文为启动时的核验记录。

## Material Passport

- 状态：**正式训练已启动；输入和 GPU 核验通过，效果待最终验证**。
- 实验：`S18-ReaRec-PRL2-Toys-S2023`；用户要求补充 Toys 并利用空闲资源。
- 时间：2026-09-09 16:19:34（北京时间）；GPU 0；PID `3020613`；tmux `s18_rearec_toys`。
- [执行计划](../../plan/第十八阶段/GRAM_第十八阶段_ReaRec_Toys并行筛选计划v0.1.md)、[输入核验](../../artifacts/phase18/rearec/toys/input_review.json)、[GPU profile](../../artifacts/phase18/rearec/toys/profile_v2/profile.json)。
- 本轮未读取 test。CPU 审核、GPU profile 和正式训练分别记录，profile 权重不进入训练。

## 为什么补充 Toys

ETEGRec Beauty 完成后 GPU 0 约 24.5 GiB 空闲，可以补跑一个轻量候选。ReaRec Beauty 也已完成：训练 150 轮，选中第 45 轮，全量 validation NDCG@10 为 0.04762500932，相对 GRAM 0.06497370269 明显落后。这个结果降低了当前方案的预期，但不直接回答 Toys 上是否同样无效。

Toys 只增加 seed2023、PRL 两步推理一条训练。如果也正常训练至平台且明显落后，可以降低这套适配与设置的优先级；若有正信号再确认，不提前展开参数搜索或多 seed。

## 输入与实现

使用全部 109,361 训练前缀、19,412 validation 用户、固定 2,000 人趋势子集，以及 11,924 个合法商品。历史上限 20、时间顺序保留、左 padding；PAD=0 不进入 softmax。ID embedding 随机初始化，不读取 ETEGRec 的 CF 向量。

CPU 审核重新检查冻结 SHA、完整用户集合、catalog 排序和商品 ID 范围；复算历史 GRAM / PCRF 全量与子集六项指标，与保存的 summary 一致。进一步通过另一个已冻结的 prepared validation 映射回原始商品，逐用户确认目标和时间顺序历史一致。

Toys adapter 的全量载入通过；错误域、不同历史上限和未冻结参照路径被拒绝。其 ReaRec 类与 collate 直接复用 Beauty 实现，模型和训练参数相同。Beauty adapter、runner 和 profile 引用文件的哈希保持不变。Toys 在独立模块实例中加载冻结的训练协议，仅绑定新输入 adapter 与来源快照清单，不复制训练循环。

作者源码仍为 `7f46432cba8435ae15fc744c53715024c3b2d32c`，沿用已有逐文件 SHA 校验。本轮延续的是作者 ID 路径的本地适配，不能视为论文全部原始设置的复现。

## GPU 核验与预算

`profile_v1` 在设备检查时因沙箱 CUDA 不可见退出，未进入训练。保留失败状态，获得 GPU 访问权限后运行 `profile_v2`；累计预算包含首次检查耗时。

| 项目 | 实测 |
|---|---:|
| 单 update P90 / 每 epoch updates | 0.060905 秒 / 214 |
| 峰值 allocated | 1.963 GiB |
| 推理位置权重更新 L1 | 2.598196 |
| CE / 递进 / 对比的推理位置梯度 L1 | 4.140554 / 0.180432 / 10.094693 |
| 带噪声 Adam/RNG 恢复 | 逐参数完全一致 |
| 最大 300 轮日程估算，含 1.25 系数 | 5,006.77 秒，83.45 分钟 |

合法 catalog top50、零步推理、无噪声 warmup 均通过。每进程分配上限 12 GiB，总实验预算 4 小时。启动阶段预计 30–90 分钟；平台早停和共享 GPU 负载会改变实际时间。

## 训练设置与后续产物

同 Beauty：2 层、256 维、2 heads、两次额外推理；batch512、Adam LR0.001、grad clip1；最多 300 epoch，100 epoch 起启用平台判据，每 5 epoch 验证，patience10、min_delta0.0001，最早第 150 轮平台早停。按最高趋势 NDCG@10 保存 best，训练结束后做全量验证。

启动已确认完整前两轮、428 次更新和 `last.pt` 保存。汇总监视器已加载第五项 `rearec_toys`，现有 DIFF→GRAM Beauty 与 ETEGRec Toys 保持原进程。启动核验保存至 `artifacts/phase18/rearec/toys/startup_verification.json`。

16:25 复核已完成第 22 轮、4,708 次更新；checkpoint 为 46,810,879 bytes，所有参数有限，配置与 profile 的来源哈希全部一致。第 5 轮首次验证的 2,000 条输出已独立核对 cohort 顺序、目标、50 个合法去重商品、重算 rank、六项指标和文件 SHA，全部通过，见 [首次验证核验](../../artifacts/phase18/rearec/toys/first_validation_verification.json)。这只确认训练和评估可运行，不构成最终提点结论。

结束后自动生成全量两步结果、相同 checkpoint 的零步诊断、对 GRAM / PCRF 的配对 bootstrap（2,000 次）。全量 Toys GRAM NDCG@10 = 0.07627451426，PCRF = 0.07871552229；趋势子集分别为 0.06937056512 / 0.07091031447，不能跨范围比较。零步诊断不等于独立训练基线，配对区间不代表多 seed 稳健性。

- [实时状态](../../artifacts/phase18/rearec/toys/run_v1/status.json)
- [汇总状态](../../artifacts/phase18/parallel_screen_20260909/status.json)，字段 `jobs.rearec_toys`
- 事件：`artifacts/phase18/rearec/toys/run_v1/events.jsonl`
- 结果：`artifacts/phase18/rearec/toys/run_v1/result.json`
- checkpoint：`artifacts/phase18/rearec/toys/run_v1/last.pt`、`best.pt`
