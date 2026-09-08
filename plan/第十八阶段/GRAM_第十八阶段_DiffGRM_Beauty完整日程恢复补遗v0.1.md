# 第十八阶段：DiffGRM Beauty 完整日程恢复补遗 v0.1

## Material Passport

- 日期：2026-09-08。
- 来源：academic-research-suite / experiment-agent；inline。
- 用户授权：在核对尚未收敛、原日程可恢复和预计成本后，用户明确“那继续跑吧”。
- 任务：只恢复 DiffGRM Beauty 的原训练日程；其他三个任务维持现有运行。
- 状态：`EXECUTING / full_r2`；恢复检查和真实新增参数更新均已确认，启动核验见文末。

**后续范围更新：** 用户随后授权其余三条也采用完整预算，已另行启动自动接续，见 [四实验完整预算与自动接续补遗](GRAM_第十八阶段_四实验完整预算与自动接续执行补遗v0.1.md)。本页 Beauty 恢复起点、原日程和运行进程继续有效；“其他三条保持不变”描述的是本页启动时的历史状态。

## 1. 本次决策及覆盖范围

`screen_r1` 在累计 epoch 53 完成，Beauty 全量 validation NDCG@10=0.0439040109，低于已有 GRAM 的 0.0649737027；但固定子集四个检查点持续上升，最佳模型位于预算末端。该分支把剩余 warmup 缩到 500 updates，并在新增 20 epoch 结束时将 LR 降到 0。它不足以代表原训练日程的最终效果。

Beauty 实测训练约 100 秒 / epoch，给原日程一次完整机会的成本可控。用户授权本次恢复后，先前“Beauty 暂停”建议及通用“至多再追加 20 epoch / 2 GPU 小时”规则由本补遗覆盖。当前重点仍是候选效果，GRAM / PCRF 只复用历史预测，不训练、不重新推理。

## 2. 恢复起点与训练预算

| 项目 | 执行值 |
| --- | --- |
| 原 checkpoint | `artifacts/phase18/diffgrm_native/beauty/run_v1/last.pt` |
| SHA256 | `3876ad9785dd04ae18bc3b94f4103ba748577393abd8f9920052c83a938c4115` |
| 完整已训练 epoch / updates | 33 / 4,257 |
| 下一轮 | epoch 34 |
| 最大总 epoch / updates | 200 / 25,800；最多追加 167 epoch |
| 原 warmup / peak LR | 10,000 updates / 0.01 |
| 恢复时第一步使用的 LR | 0.004257 |
| 下一步完成后的 LR | 0.004258 |
| 有效 batch / microbatch | 1,024 / 128，沿用原配置 |
| 原早停约束 | epoch ≥100 且连续 5 次子集检查无 NDCG@10 改善 |
| GPU | GPU 1，`GPU-2afcc1df-7df8-1f3f-8970-a252a06c6160` |
| 耗时估计 | 当前吞吐约 5–6 小时，包含稀疏验证；共享负载会改变 ETA |
| 运行超时上限 | 10 小时，超时退出并保留最后完整 checkpoint，不自动重试 |

从原 epoch 33 恢复模型、Adam、原 scheduler、CPU / CUDA RNG；原 `screen_r1` 的 epoch 34–53 是另一条已保存的训练分支，其更新数不累加到本次恢复路径。原 checkpoint、原配置、短日程预测和结果全部保留。

必须先构造原 LambdaLR，再加载其状态并重新加载 Adam 状态，防止 scheduler 构造时把 optimizer 的恢复 LR 重置。核对 4,257 更新位置、起始 LR、下一步 LR、warmup 终点和总步数。恢复检查不推进 optimizer。

## 3. 低成本评估与停止

1. 对恢复的 epoch 33 做一次同一 2,000 用户子集评估，建立可选的起始模型分数。它也可被选为最终模型，避免训练下降后仍被迫选择较差的续训模型。
2. 此后在绝对 epoch 40、50、……、200 检查同一固定 cohort；按 validation NDCG@10 选择最佳 checkpoint。epoch 100 前不因 patience 停止，满足原早停约束后结束。
3. 每次结果直接与第九阶段已有 GRAM / PCRF 的相同用户 rank 比较。Beauty 子集 GRAM NDCG@10=0.0763515009、GRAM+PCRF=0.0766019942，不使用全量均值替代子集均值。
4. 选定最佳 checkpoint 后只做一次全量 validation，22,363 用户。封存 test，不运行 PCRF 组合、不增开 seed 或消融。
5. 评估前后保存恢复训练 RNG，防止改变监控频率额外消耗训练随机流。恢复的是原学习率与训练设置；验证频率、选模集合和运行条件不同，因此不声称完整执行轨迹与未中断原程序逐位相同。
6. 到 200 epoch 仍在上涨时，记录“预算用尽、未确认收敛”；不自动追加训练，也不凭 `COMPLETED` 判定方向有效或整个架构无效。

## 4. 入口与产物

- 配置：`experiment/phase18/config/s18_full_r2_diffgrm_beauty.json`。
- 启动：`bash experiment/phase18/run_stage18_diffgrm_full_resume.sh`。
- Runner：`experiment/phase18/protocol/s18_diffgrm_full_resume.py`。
- 新产物：`artifacts/phase18/diffgrm_native/beauty/run_v1/full_r2/`。
- Console：`artifacts/phase18/diffgrm_native/beauty/run_v1/full_r2.console.log`。
- 状态仍为：`artifacts/phase18/diffgrm_native/beauty/run_v1/status.json`，`revision=full_r2`。
- `status.json` 中 `result_path` 指向本次 `full_r2/result.json`；本次未完成前，根目录旧 `result.json` 仍属于已完成的 `screen_r1`，不能把它当本次结果。
- 每个 epoch 保存完整 checkpoint 和原 scheduler；每次验证保存逐用户预测、同用户差值与选择状态。源码和配置快照保存于新目录。

## 5. 检查与启动记录

新增原日程恢复、绝对 epoch 早停、评估 RNG 隔离及同用户差值、尾 batch / 起始模型保留四项 CPU 测试，与现有五项恢复测试合计 9/9 通过。恢复后的连续更新跨过 warmup 边界，与未中断的 Adam / scheduler 更新逐参数一致。脚本语法、JSON 配置及 diff 格式检查通过。

实际 checkpoint 的 CPU scheduler 核查记录于 `artifacts/phase18/diffgrm_native/beauty/full_r2_cpu_restore_check.json`；运行前后 GPU 恢复检查、原日程检查和真实参数更新由 runner 保存。

2026-09-08 15:34 左右已在 GPU 1 启动，tmux=`s18_full_r2_diffgrm_beauty`，训练 PID=`3647654`。GPU 恢复检查 PASSED，恢复检查过程 optimizer updates=0；原学习率起点 0.004257、下一步 0.004258、warmup=10,000、总步数 25,800 全部核对一致。

恢复的 epoch 33 初始子集验证耗时约 49 秒，NDCG@10=0.0356376190。15:35:57 已进入 epoch 34，处理 40,576 / 131,413 样本，总 optimizer steps=4,296，新增 39 步，LR=0.004296，loss=3.54039，真实训练更新正常。其他三条任务仍为原 PID / screen_r1；原 epoch 33 checkpoint 哈希未改变。

启动核验快照：`artifacts/phase18/diffgrm_native/beauty/run_v1/full_r2/startup_verification.json`。下一个常规候选检查点为绝对 epoch 40；后续自动按已记录日程执行，不需要手动启动每一轮。
