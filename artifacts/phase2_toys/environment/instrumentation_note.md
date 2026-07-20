# Toys 资源仪表化说明

## 目标

在共享 GPU 上将 GRAM 本进程显存与整卡总占用分开，并给出同步后的训练/test 墙钟时间。

## 主指标

Toys 新脚本显式传入 `--resource_metrics 1`。在训练、训练结束自动 test 和独立最佳 checkpoint test 的边界，代码执行：

1. `torch.cuda.synchronize(device)`；
2. `torch.cuda.reset_peak_memory_stats(device)`；
3. `time.perf_counter()` 记录起点；
4. 原始 runner 方法；
5. 再次 CUDA synchronize；
6. 记录 wall time、peak/end allocated 和 reserved。

`torch.cuda.max_memory_allocated` 和 `max_memory_reserved` 只统计当前 GRAM Python 进程的 PyTorch allocator，不会加入其他用户进程。它们是最终报告的主显存指标。

## 辅助指标

后台守护每 5 秒记录：

- 整卡 memory.used/free 和 utilization；
- 目标 GPU 上每个 compute PID 的 used GPU memory；
- PID 是否与 `workload_pid` 一致。

整卡指标只作为共享环境背景。PID 采样可包含 CUDA context 开销，但可能错过 5 秒内的短暂峰值，因此不替代 PyTorch 进程内峰值。

## 时间口径

- `phase=training`：`runner.train_generator()`，包含 30 epoch、周期 validation 和 checkpoint 保存，不包含最后自动 test。
- `phase=automatic_last_checkpoint_test`：训练任务最后自动使用的 checkpoint test，不作为最终指标，除非它恰好是 validation 最佳。
- `phase=selected_checkpoint_test`：独立最佳 checkpoint full-ranking test。
- 内部 GRAM 文件日志自带时间戳，后续用于统计每 epoch 和每次 validation 时间。

## 数值影响

仪表化默认值为 0，官方脚本和 Beauty 脚本不启用它。Toys 脚本启用时，只在阶段边界做 CUDA 同步、重置/读取统计和日志写入，不改变数据、forward、backward、optimizer、scheduler、生成或评测公式。预期不影响指标；边界同步的开销会被如实包含在墙钟时间中。
