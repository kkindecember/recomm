# 第十八阶段：第四候选 ReaRec 筛选准备 v0.1

## Material Passport

- 日期：2026-09-09；模式：候选规划与源码静态核对。
- 当前状态：**2026-09-09 15:17:59 已在 GPU 6 启动正式训练；适配、CPU/GPU 核验及预算准入均通过**。下文准备阶段记录保留，最新执行设置见文末。
- 用户需求：ETEGRec Toys 启动后，有资源则考虑另一个方向。
- 当前运行：DIFF→GRAM Beauty、ETEGRec Beauty、ETEGRec Toys、ReaRec Beauty，见 [汇总状态](../../artifacts/phase18/parallel_screen_20260909/status.json)。

## 选择依据

建议第四候选优先考察 ReaRec 的潜在推理机制：把序列末端隐状态与推理位置编码再次输入序列模型，逐步细化用户表示；PRL 对不同推理步施加训练信号，并加入带噪声的推理对比。这与 DIFF 的属性交互和 ETEGRec 的编码/推荐联合优化是不同的模型改动。[论文](https://arxiv.org/html/2503.22675v3)、[作者 PRL 实现](https://github.com/TangJiakai/ReaRec/blob/7f46432cba8435ae15fc744c53715024c3b2d32c/src/models/PRL.py)。

本地 Toys 验证用户中 12,673 / 19,412（65.3%）历史长度不超过 5，Beauty 对应比例约 62.9%。因此我优先选择可以直接检验短历史表示是否改善的候选，而不是仅凭大规模、长历史实验安排 HSTU。这个排序是研究判断，短历史占比本身不能证明 ReaRec 有效。

论文实验包括 Yelp 及 Amazon 的 Video Games、Software、CDs/Vinyl、Baby Products，未提供本项目 Beauty / Toys 的直接证据。其“30%–50%”来自事后最优推理步上界，不能用作固定策略的预期提升。仓库描述标注 TKDE '26，目前仅确认作者自述及 arXiv，不将其冒称 SIGIR 主会已录用工作。[作者仓库](https://github.com/TangJiakai/ReaRec)。

另一候选 LARES 还包含预训练、样本过滤和强化后训练，首轮适配链条更长，暂不同时展开。[LARES 作者入口](https://github.com/BishopLiu/LARES)。本次只比较了这几项相关候选，不声称穷尽最新工作或找到唯一最优方向。

## 固定源码与适配前置工作

作者 commit：`7f46432cba8435ae15fc744c53715024c3b2d32c`，GitHub API 返回提交日期 2025-12-28。归档、22 个文件 SHA 与来源路径保存在 [源码清单](../../artifacts/phase18/parallel_screen_20260909/rearec_source_manifest.json)。仅阅读源码，未运行原始数据入口或训练脚本。

已核对的具体事项：

1. `utils/layers.py` 的 `ReaRecAutoRegressiveWrapper` 实际包含隐状态回馈、独立 reasoning position embedding 和 KV cache，具有可复用的核心机制。
2. 原 `BaseReader` 同时载入 train / valid / test，必须替换为本地 prepared train / validation 入口。原左 padding 和最后一行 PAD 约定也需要与本地 PAD=0 对齐；候选输出要明确排除 PAD。
3. 原 PRL 使用最后一个推理状态生成分数，并对更早状态做温度递进监督。直接将 `reason_step=0` 套进原 loss 会得到空的中间监督维度，不能拿它原样训练零步基线。零步推理可以单独调用 wrapper，不进入该 loss。
4. 核验应覆盖不同历史长度的 padding、缓存与非缓存评分一致、推理位置参数的真实梯度、随机种子及 checkpoint 恢复。未完成这些检查前，不把源码已下载写成实现已就绪。

## 拟定的最小筛选

只增加 **Beauty、seed 2023、PRL 两步推理** 一个训练候选；沿用同一份完整训练前缀、固定 2,000 人趋势子集及最终 22,363 人 validation。选择 Beauty 是为了首先评价一条新机制，避免又扩成两域参数矩阵；其 GRAM / PCRF 参照已经可复用。

结构以固定作者 PRL 的 ID 路径为起点：2 层、256 维、2 heads、2 次额外潜在推理，商品 embedding 可训练，最长历史按本地 20。暂定 batch 512、Adam LR 0.001、最多 300 epoch、至少 100 epoch，每 5 epoch 趋势验证；最小预算后连续 10 次未改善 0.0001 NDCG@10 则结束。PRL 的温度、递进和噪声先取固定源码默认值，不展开网格搜索。这是待 profile 的实验规格，尚未生成可执行训练配置。

模型采用全 catalog 打分取 top 50，与自回归 beam50 的成本不同，报告时同时记录延迟和显存。首先比较独立候选与保存的 GRAM / PCRF 同用户输出；有正信号后再考虑组合与跨域。现阶段不重训 GRAM / PCRF，也不为了归因先启动一个额外零步训练任务。

可以对同一个已训练两步 checkpoint 额外做零步推理，观察继续推理是否有帮助。这只是廉价诊断，不能代替匹配预算的独立零步训练，更不能据此声称推理机制贡献已被严格识别。完整归因和多 seed 留到候选值得投入后。

## 资源与预算准入

三任务启动后的连续快照显示 GPU 6 仍有约 18–25 GiB 空闲，GPU 1 约 20 GiB；有容纳轻量候选的显存空间，但两张卡已有其他进程，空间未被预约。优先在启动前重新核对 GPU 6；不在 GPU 0 / 4 / 5 追加第四条训练。

本轮建议新增投入上限为 **4 GPU 小时，含 profile、训练、验证与保存**，这是拟定预算，不是已测耗时。先测真实 batch 的 P90 吞吐和峰值显存，按最大 300 epoch、全部计划验证、最终全量验证加 1.25 系数计算；仅在预算容纳完整计划时启动。若不满足，应重新制定并记录预算，而不是缩短训练后宣称方向失败。

最终报告 NDCG / Hit @5、10、50，配对用户差值与 bootstrap 区间；弱正信号、训练未充分、工程失败分别表述。超过 GRAM 但仍未超过 PCRF 的结果，也要明确实际价值边界。

这份准备计划回答“还有没有位置、下一条考虑什么”。目前正式训练仍为三条，第四方向尚未启动。


## 2026-09-09 执行更新

用户已授权“可以开始这个”。现已完成本地适配及 5 项 CPU 测试，实际 Beauty 输入/历史参照核验通过。GPU 6 上的真实 batch profile 已通过：batch 512，每 epoch 257 updates，最大 300 epoch；P90 单 update 0.1250 秒，峰值 allocated 1.97 GiB，最大完整日程估计 12,169.39 秒（3.38 小时），低于含 profile 的 14,400 秒硬预算。GPU 共享负载会改变实际耗时。

实际实现保持固定作者的 Transformer 与 ReaRec wrapper，只在隔离导入中将 max history 从 50 设为本地 20；本地适配负责 PAD=0、去除 PAD softmax 类别、输入校验、训练和评估。PRL 损失在有噪声时与作者函数数值一致；warmup 无噪声时对比项初始化为 0，修复原函数未定义变量的路径。训练采用固定 Adam LR 0.001、gradient clip 1；参数、源码、输入 review 均按 profile 哈希锁定。

GPU 实测确认 CE、递进监督、噪声对比对推理位置参数的梯度均非零，真实推理位置参数已更新，带噪声 Adam/RNG checkpoint 恢复逐参数完全一致。profile 权重全部丢弃，正式实验从 seed 2023 重新初始化。

执行配置：`experiment/phase18/config/s18_rearec_beauty.json`；入口：

```bash
bash experiment/phase18/run_stage18_rearec_beauty.sh --mode run
```

- [速度/显存与梯度实测](../../artifacts/phase18/rearec/beauty/profile_v1/profile.json)
- [输入核验](../../artifacts/phase18/rearec/beauty/input_review.json)
- 实时状态：`artifacts/phase18/rearec/beauty/run_v1/status.json`
- 事件日志：`artifacts/phase18/rearec/beauty/run_v1/events.jsonl`
- 最终结果：`artifacts/phase18/rearec/beauty/run_v1/result.json`
- tmux：`s18_rearec_beauty`

本节覆盖上文“未完成本地适配/实测”的准备阶段状态；效果结论仍待正式训练与最终验证。

正式 PID `2752109`，15:17:59（北京时间）启动；真实完整 epoch、参数更新与 checkpoint 证据见 [ReaRec 启动报告](../../report/第十八阶段/Stage18_ReaRec_Beauty实现核验与启动报告.md)。
