# 22.1 多卡分片执行补充

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / execution amendment
- Origin Date: 2026-10-17
- Verification Status: PLANNED before parallel run
- Version Label: repeat_split_minimal_20261017_parallel_v2

科学协议沿用[22.1最小全量验证](22.1_探索重复拆分最小全量验证_20261017.md)：相同41,775人、三专家和冻结β/α、六个方法、训练侧一个全局ρ、4项主比较、官方不过滤评测，均不改变。

用户补充要求“服务器上的卡只要有位置都能用，可以多卡并行提高速度”。原22.1两域各一任务的调度于18:18主动停止，尚无新评估预测，保留旧目录和中断证据，记为SUPERSEDED；此为调度迁移，不是效果失败或根据结果重跑。唯一新入口：

- 启动：`bash experiment/phase22/run_repeat_split_parallel.sh prepare` 后 `bash experiment/phase22/run_repeat_split_parallel.sh start`。
- 主状态：`artifacts/phase22/repeat_split_minimal_20261017_parallel_v2/status.json`。
- 报告：`report/第二十二阶段/探索重复拆分最小全量验证_多卡_20261017.md`。

每域先运行一个fit任务，只拟合一次ρ并完成真实GRAM启动核验。fit任务释放租约后，该域4个生成分片才准入。每片固定取全量索引 `j % 4 == shard`；Toys分片为4,853人，Beauty为5,591/5,591/5,591/5,590人。所有分片对每名用户完整运行4个新生成方法，共8个生成任务、167,100次新生成；不按效果、目标或历史分组调度。每个生成任务共享该域同一ρ、同一冻结模型、同一输入哈希。

最多8个任务、一张GPU同时只放本轮一个任务，动态选择全部可见且满足准入门槛的GPU。按用户本次“有位置都能用”的要求与第21阶段近期执行配置，采用**16GiB总租约、12GiB预期工作峰值、4GiB侧车、18GiB空闲准入、11GiB模型分配上限**；这明确替代22.1引用的旧30GiB租约配置。不停止、抢占或修改其他实验，不启动显存占位器。可用卡不足时排队，单任务计算硬超时48小时、资源等待上限48小时。

每片独立status、日志、预测、启动核验、模型digest和租约释放记录。共享fit文件校验哈希后才能读取。resume只恢复未完整写入的末行，已完成用户不重算；任务锁防止活跃worker重复启动。科学失败不自动重试，资源准入不足仅退让排队。

最终按原始全量索引交织合并4个分片，逐条检查用户、索引、六臂商品与分数齐备，检查每个分片租约已释放，再调用同一独立统计代码。主status在统计完成后才标COMPLETED。用户可直接查看根status中的各域、分片、GPU绑定和剩余时间；程序自行更新，会话不持续监看，不自动启动下一轮。
