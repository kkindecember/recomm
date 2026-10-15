# Beauty GRAM/C1 断点恢复

2026-09-22 10:43（北京时间）。用户授权恢复 Beauty；使用 academic-research-suite 的实验执行流程。

## 恢复状态

10:41:57 启动后台恢复；10:43:27 已从第 29,700 步推进到第 29,726 步。
原 GPU 6、7、双卡划分、训练输入、模型代码和超参数保持原协议。
GRAM 共 29,850 次更新，完成后由原运行器自动执行 C1 五轮。
本次未读取开发或测试标签，尚无新的推荐效果指标。

## 中断与保存

旧 rank0 PID 2053916 已退出，日志在 08:28 停止，但旧 status 仍为 RUNNING。
原日志有 29,742 条完整记录和一条截断记录，错误文件为空；具体中断原因未确认。
已核实身份并清理原监督进程、残留训练子进程和资源跟踪进程。

08:25:17 的第 29,700 步检查点已单独备份，SHA256 为
`f30af362f4219d09a39ab170e790f72d72a0b8e4251e6b883d2a8b9ed7f9b9a9`。
116 项冻结输入/代码哈希全部匹配；模型和优化器张量均有限，调度步数一致，双卡随机状态完整。
原日志完整归档；当前训练日志回到保存边界后接续，以免产生重复步号或拼接截断记录。

## 续跑核验

恢复后的前 26 步与原中断前对应记录的 loss、梯度范数、学习率、批次形状、epoch/window 均完全一致。
双卡训练进程存活，当前日志步号连续，loss 和梯度范数有限。

独立后台管理进程每 15 秒记录存活与进度；超过五分钟无推进标记疑似停滞，训练主进程消失时记录失败，避免只依赖旧 RUNNING 字段。没有自动追加实验或自动重试。

- [训练状态](../../artifacts/phase20/lime_parallel_beauty_20260921_v1/status.json)
- [进程与进度监控](../../artifacts/phase20/lime_parallel_beauty_20260921_v1/recovery_20260922_v1/health.json)
- [恢复核验](../../artifacts/phase20/lime_parallel_beauty_20260921_v1/recovery_20260922_v1/resume_verification.json)
- [重放一致性核验](../../artifacts/phase20/lime_parallel_beauty_20260921_v1/recovery_20260922_v1/replay_check.json)
- [恢复日志](../../artifacts/phase20/lime_parallel_beauty_20260921_v1/recovery_20260922_v1/training.log)

管理 PID：3012773；训练监督 PID：3012774；两张卡训练 PID：3012886、3012887。以上为启动时记录，实时状态以链接产物为准。
