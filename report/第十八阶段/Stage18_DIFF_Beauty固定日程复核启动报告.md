# 第十八阶段：DIFF→GRAM Beauty 固定日程复核启动报告

## Material Passport

- Origin Skill / Mode：academic-research-suite / experiment-agent / run，inline。
- Origin Date：2026-09-09（Asia/Shanghai）；Version Label：s18_diff_beauty_fixed_launch_v1。
- Verification Status：UNVERIFIED（推荐效果尚未产生）；工程检查和真实训练启动已核实。
- 用户授权：先执行 Beauty 固定日程复核。实验计划见 [固定日程复核计划](../../plan/第十八阶段/GRAM_第十八阶段_DIFF_Beauty固定日程复核计划v0.1.md)。

## 1. 已启动的任务

**2026-09-09 13:28 启动，13:29 确认发生真实 optimizer 更新。**

| 项目 | 记录 |
| --- | --- |
| 实验 | DIFF→GRAM Beauty，`confirm_v1 / fixed_r1` |
| tmux | `s18_diff_gram_beauty_fixed` |
| PID | `2195156` |
| GPU | 4，`GPU-4cc17bbe-a85f-bdab-44ef-bbe29d1afc09` |
| 初始化 | 原始 Beauty epoch 25 GRAM parent；DIFF 新分支重新初始化；全新 Adam |
| seed / 模型配置 | 2023；沿用原 Beauty DIFF→GRAM 超参数 |
| 数据 | 131,413 条训练转移；22,363 validation 用户 |
| 固定训练 | 1 frozen + 10 joint，合计 11 轮、11,297 次更新；不按中间分数提前停止 |
| 验证 | epoch 1、3、5、7、9、11 固定旧 2,000 人选模；最终只对选中模型做完整 validation |
| 预算 | 预计 24–30 小时，最长 36 小时；共享 GPU 下 ETA 以实测修正 |

首次训练进度核对时为 `TRAINING / warmup / epoch 1`：已处理 **1,224 / 131,413** 条样本，完成 **9 次 optimizer 更新**，loss 有限。该读数是启动快照，不代表当前最新进度，更不代表推荐效果。实时数据见状态文件。

## 2. 运行前检查

实际 GRAM 环境 `torch 1.11.0+cu113` 下共 **11 项 CPU 检查通过**，包括：

- 冻结轮只训练新分支，联合阶段能够更新 GRAM；完整训练数据中的尾 batch 不丢弃。
- 两阶段学习率按各自固定位置推进，联合阶段保留 Adam 动量；最终学习率为 0。
- 即使模拟的中间分数连续下降，仍完成批准的 11 轮；完整验证加载选中的 checkpoint，而不是默认用最后一轮。
- 验证前后 Python、NumPy 和 Torch 随机状态保持；原 DIFF 结构、生成接口及 checkpoint 行为检查通过。

GPU 4 上真实最长历史 20、microbatch 8 的检查通过：冻结 / 联合阶段分别完成 2 次测试更新，所需分支梯度非零且有限；冻结时 GRAM 梯度为 0。联合训练峰值 reserved **8,104 MiB**，生成峰值 reserved **7,096 MiB**；生成 50 个候选，checkpoint 保存 / 重载后结果完全一致。检查更新不用于正式训练，正式进程重新加载原 parent 和同 seed 新分支。

正式入口核验了 smoke、模型配置 SHA、执行配置、数据哈希、固定 cohort、历史 rank 缓存，以及 **38 个源码文件**的 SHA。正式初始化记录确认 Adam 更新数为 0、未使用 smoke 权重、数据规模与每轮 1,027 次更新一致。宿主进程和 GPU 进程表均确认 PID 2195156 存活并使用 GPU 4。

检查证据：[GPU smoke](../../artifacts/phase18/diff_gram/beauty/confirm_smoke_v1_gpu4_mb8/smoke_summary.json)、[正式初始化](../../artifacts/phase18/diff_gram/beauty/confirm_v1/initialization_check.json)、[固定日程](../../artifacts/phase18/diff_gram/beauty/confirm_v1/schedule_plan.json)、[启动核验快照](../../artifacts/phase18/diff_gram/beauty/confirm_v1/launch_verification.json)。

## 3. 状态与结果入口

- **新状态：[confirm_v1/status.json](../../artifacts/phase18/diff_gram/beauty/confirm_v1/status.json)**。
- [实时事件](../../artifacts/phase18/diff_gram/beauty/confirm_v1/events.jsonl)、[console 日志](../../artifacts/phase18/diff_gram/beauty/confirm_v1.console.log)。
- 完成后生成 `artifacts/phase18/diff_gram/beauty/confirm_v1/result.json`，含选中 epoch、完整验证指标及相对历史 GRAM / PCRF 的差值。
- checkpoint 位于同目录：`initial.pt`、`last.pt`、`best_trend.pt`。旧 `train_v1/status.json` 仍对应已完成的 full_r2。

状态每约 30 秒更新训练或验证进度，记录 PID、时间、样本数 / 更新数或验证人数。运行有最长 36 小时的超时限制和异常记录，错误不自动重新训练。终端会话使用 tmux 持久运行；本报告只确认启动与当前进度，不声称助手在离开当前交互后持续人工监控。

## 4. 结果解释边界

原 Beauty full_r2 的 NDCG@10 相对 +0.5862% 继续作为待复核的弱正信号；原轮本来就使用全量数据。本次检验固定日程下能否重现收益，不是补做此前缺失的全量数据训练。

本次保持 seed 2023，日程与训练随机流处理比原切换流程更明确，但没有多 seed、匹配的纯 GRAM 续训对照或独立 test。新结果产生后先做同用户比较；正向结果可以支持下一步冻结 PCRF 组合检查，不能直接写成稳定提点或单一 DIFF 机制的因果贡献。

本轮只新增这条 Beauty 训练任务。Toys 尚未启动复核，DiffGRM 没有续训，ETEGRec 的既有进程仍在运行。
