# 第十八阶段：ETEGRec Toys 启动与并行资源报告

## Material Passport

- 状态：**已启动、已确认真实参数更新，效果待最终 validation**。
- 用户授权：补充 ETEGRec Toys，然后检查能否并行其他方向。
- 实验：`S18-ETEG-CF512-Toys-S2023`，seed 2023。
- 启动时间：2026-09-09 14:42:57（北京时间）。
- 核验证据：[启动快照](../../artifacts/phase18/parallel_screen_20260909/toys_startup_verification.json)、[执行计划](../../plan/第十八阶段/GRAM_第十八阶段_ETEGRec_Toys并行筛选计划v0.1.md)。

## 已完成的工作

Toys 复用完整 109,361 条训练前缀、19,412 位验证用户、11,924 商品。商品编号逐项匹配原 CF checkpoint 的排序索引，冻结协同向量为 `[11925,512]`（含 PAD），训练商品 11,876 个。重新计算完整验证和固定 2,000 人子集上的 GRAM / PCRF 六项指标，均与保存参照一致。只加载已分离的 train / validation。

模型核心沿用已核验 ETEGRec 作者实现；5 项测试全部通过。Toys 独立 runner 与 Beauty 的差异仅为 adapter 导入及源码快照/哈希路径；原 Beauty 核心与 runner SHA 均未改变。已有 Beauty 任务保持运行。

GPU 5 实测验证了 KL / 对比学习的非零梯度、ID / REC / finetune 三阶段真实参数更新、checkpoint 与 RNG 恢复一致，以及完整 catalog 的合法 beam50。profile 峰值 allocated 为 4.42 GiB，最大碰撞组 48，小于 suffix 容量 256。profile 权重全部丢弃，正式训练重新初始化。

## 预算与真实启动

含完整 10,000 商品 pass 的 RQ 上限、400 联合 epoch、100 微调 epoch、全部验证与保存，并加 1.25 安全系数，预计 **6.05 小时**；硬预算 **12 小时，包含 profile**。按开始时吞吐，完整日程约在今晚 20:45 附近结束；共享 GPU 波动或合法平台早停会改变实际时间。

正式 PID `2572693`，tmux `s18_etegrec_toys`。启动快照已记录 RQ 第 1050 个商品 pass、12600 次更新；checkpoint 为 52,553,083 bytes。RQ pass 是商品编码预训练，不是推荐训练 epoch，不能将该数值直接与后续联合训练轮数相比。

## 当前三条任务与状态入口

| 任务 | GPU | 状态文件 |
| --- | --- | --- |
| DIFF→GRAM Beauty 固定日程复核 | 4 | `artifacts/phase18/diff_gram/beauty/confirm_v1/status.json` |
| ETEGRec Beauty | 0 | `artifacts/phase18/etegrec/beauty/run_v1/status.json` |
| ETEGRec Toys | 5 | `artifacts/phase18/etegrec/toys/run_v1/status.json` |

推荐查看 [三任务汇总 status](../../artifacts/phase18/parallel_screen_20260909/status.json)，每 15 秒更新。该监视器只读取任务文件与 PID，单独写汇总，不控制训练。每个任务含 `process_alive`、`seconds_since_activity`、`raw_status` 和 `latest_event`；RQ 阶段有时事件日志比原始 status 更早更新，用 `latest_event.epoch/steps` 查看最新进展。tmux 为 `s18_parallel_status`，三进程结束后自动退出。

Toys 原始 [status](../../artifacts/phase18/etegrec/toys/run_v1/status.json) 与 [events](../../artifacts/phase18/etegrec/toys/run_v1/events.jsonl) 保留，最终结果写入 `run_v1/result.json`。

## 第四方向的资源与选型

14:44–14:46 三次 GPU 快照显示，GPU 6 空闲显存从 18,392 MiB 增至 25,128 MiB，利用率分别为 14%、21%、0%。GPU 1 约有 20,843 MiB 空闲。这说明显存上还可安排一个轻量候选，但不能把瞬时利用率当成独占算力。GPU 0 / 4 / 5 优先保障已运行任务；GPU 2 仅余约 5.5 GiB，不列为优先位置。原始快照保存于 `artifacts/phase18/parallel_screen_20260909/gpu_sample_01.csv` 至 `gpu_sample_03.csv`。

第四候选收敛到 **ReaRec 潜在多步推理**。已下载固定作者源码并静态核对，当前仍是筛选准备，**没有启动第四条训练，也没有第四方向效果结果**。采用单域、单 seed、单个两步 PRL 候选；零步比较优先使用同 checkpoint 的廉价推理诊断，不额外重训 GRAM / PCRF 或安排完整消融矩阵。具体依据、适配问题和预算准入见 [第四候选准备计划](../../plan/第十八阶段/GRAM_第十八阶段_第四候选ReaRec筛选准备v0.1.md)。
