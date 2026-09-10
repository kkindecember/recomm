# 第十八阶段：DIFF→GRAM Beauty 验证吞吐诊断与换卡安排

## Material Passport

- 日期：2026-09-09；academic-research-suite / experiment-agent / run + validate，inline。
- 用户要求：处理当前 Beauty 验证缓慢问题。
- 状态：固定案例测速已完成、恢复逻辑测试通过；安排完整 epoch 边界迁移。迁移前后状态需查看实时文件，不能把安排迁移写成已经提速。
- 科学范围：性能工程检查；没有新增候选训练轮数、基线推理或 test 数据读取。

**执行记录：** 迁移协调已在tmux `s18_diff_gram_beauty_move`启动，PID **3489466**；17:53观察状态为 `WAITING_FOR_EPOCH_CHECKPOINT`，最后完整checkpoint仍为第1轮，原Beauty PID2195156继续第2轮训练。协调启动时GPU5空闲20,550MiB，实际切换前重新检查；当时第2轮剩余估计约1.9小时，负载变化会影响边界到达时间。统一监控已展示 `jobs.diff_gram_beauty.pending_migration`。[协调启动核验](../../artifacts/phase18/diff_gram/beauty/confirm_v1/gpu5_move_v1/launch_verification.json)。

## 1. 查到的问题

Beauty 第一遍2,000人趋势验证耗时8,113.17秒，即4.06秒/人。当前GPU4同时有多个计算任务，短采样中本训练进程的SM利用率在约11%–42%间波动；其他任务同时使用GPU计算。显存有余量不能等同于可独占计算资源。

在同一份Beauty第1轮checkpoint、同一批13个验证用户上进行了分时短测。12人取自固定cohort前缀，额外覆盖最长历史20。保留FP32、TF32关闭、beam50、原生成与缓存路径。

| 设备 | 13人总墙钟 | 平均墙钟/人 | 平均生成/人 | 平均数据等待/人 | 候选、分数与历史输出 |
|---|---:|---:|---:|---:|---|
| GPU4 | 129.95秒 | 10.00秒 | 9.94秒 | 0.0041秒 | 13/13逐位一致 |
| GPU5 | 21.27秒 | 1.64秒 | 1.58秒 | 0.0054秒 | 13/13逐位一致 |

两次使用同一checkpoint SHA `9a5321ea8dba67f2d6e0f569eab18246d9ac387180e28bacbcc7bd7ccd0b944d`。评分最大差均为0，没有optimizer更新。数据等待远小于生成时间，当前证据优先支持改善设备计算环境。

**比较限制：** 两次不是独占GPU、同时随机化的性能对照；GPU4测速与原训练并存，也临时增加了争用。负载在采样期间变化，13个案例还包含刻意加入的最长历史。因此约6.1倍的短测差值不能当作长期加速承诺，也不能全部归因于某一个其他进程。

观察到缓存reserved显存随不同历史长度增长，两个短测的最高reserved分别约22,028和14,632MiB。尚未单独干预分配器，不能认定缓存就是主因；本次先采用已有直接证据的设备迁移，不改变推理实现。此前Beauty缓存重排优化未通过完整精确一致性检查，不能未经新核验直接打开；TF32和beam同样保持原设置。

证据：[GPU4结果](../../artifacts/phase18/diff_gram/beauty/speed_probe_20260909/gpu4_v1/result.json)、[GPU5结果](../../artifacts/phase18/diff_gram/beauty/speed_probe_20260909/gpu5_v1/result.json)、[测速脚本](../../experiment/phase18/analysis/s18_diff_gram_speed_probe.py)。

## 2. 实施方式

1. 原Beauty任务继续完成当前第2轮，等待新提交的完整epoch checkpoint。
2. 后台协调程序在保存边界重新验证原进程身份、冻结源码、checkpoint的模型配置/Adam步数/学习率位置/随机状态，以及GPU5至少17,000MiB空闲。
3. 条件满足后，只中断已验证的Beauty旧进程；归档中断状态，复制完整checkpoint和历史最佳选模，再在GPU5继续下一轮。保存之后、停止之前的少量边界batch可能重放；不丢弃当前整轮已完成的训练。
4. 如果GPU5在边界时容量不足，保留原训练继续，等待后续新保存边界；不停止其他人的任务。
5. 若原进程已被用户停止或身份/源码变化，协调程序退出，不自动恢复它。

仍是原始Beauty父模型出发的 **1 frozen + 10 joint** 固定训练，不重新初始化DIFF/Adam，不重启学习率曲线，不增加选模检查点。第1、3、5、7、9、11轮的固定cohort选模和最终全量validation保留。

**原36小时预算保留**：按原进程启动时间保守锚定，北京时间 **2026-09-11 01:28:24** 截止；换卡不重新获得36小时预算。若最终仍达不到完成速度，原硬时限仍会触发，不能承诺换卡必然保证按时完成。

## 3. 恢复核验

在带随机训练、尾部不满batch、冻结/联合阶段转换和选模的测试模型上，对比连续训练与第2轮保存后恢复：第11轮权重、Adam所有状态、scheduler、CPU随机状态完全一致；剩余验证点和最终选中模型一致。错误scheduler位置会拒绝恢复。此为真实执行的等价性测试，不代表所有真实GPU未来梯度逐位一致的普遍证明。

真实Beauty第1轮的13个GPU案例，两张卡均精确复现历史候选/分数。实际交接时还会校验完整checkpoint和剩余学习率位置，不把工程检查权重用于训练。

## 4. 状态与后续时间估计

- 原主状态仍为 `artifacts/phase18/diff_gram/beauty/confirm_v1/status.json`。
- 迁移协调状态为同目录 `migration_pending.json`：`WAITING_FOR_EPOCH_CHECKPOINT`表示仍在等待，`RESUMED`才代表已经换卡恢复。
- 迁移产物放入同目录 `gpu5_move_v1/`，旧checkpoint、预测、源码和中断记录保留。
- 执行配置：[迁移合同](../../experiment/phase18/config/s18_diff_gram_beauty_gpu5_move.json)；入口：[协调与恢复代码](../../experiment/phase18/protocol/s18_diff_gram_fixed_move.py)。

若GPU5正式验证能保持约1.6秒/人，仅最终全量验证大约10小时，较第一次趋势验证外推的25小时明显缩短；这是条件外推。应在迁移后的实际训练和下一次验证进度产生后更新总ETA，而不是直接把短测加速倍数乘到全部训练上。
