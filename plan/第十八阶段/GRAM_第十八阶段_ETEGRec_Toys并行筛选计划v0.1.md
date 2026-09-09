# 第十八阶段：ETEGRec Toys 并行筛选计划 v0.1

## Material Passport

- 模式：run；用户已授权补充 ETEGRec Toys，并检查额外并行空间。
- 实验：`S18-ETEG-CF512-Toys-S2023`；单域、单 seed，独立候选筛选。
- 2026-09-09：数据与 CPU / GPU 核验通过，最大日程估计 6.05 小时；含 profile 的硬预算 12 小时。
- 输入核验：[input_review_toys_20260909.json](../../artifacts/phase18/etegrec_planning/input_review_toys_20260909.json)。不读取 test。

## 目的与固定设置

检验 ETEGRec 的联合编码学习是否能在 Toys 上形成可用提点，避免仅凭 Beauty 决定跨域前景。模型、损失、初始化、学习率和训练日程沿用 [Beauty 计划](GRAM_第十八阶段_ETEGRec联合编码学习与Beauty单域筛选计划v0.1.md)。复用固定作者 commit `58d9736afc28e03e190a107a5fa22f5241be6088`，使用本地冻结 CF512；这属于本地核心适配，不是作者原始 embedding 的逐项复现。

训练使用 109,361 条完整训练前缀，最长历史 20；趋势验证固定 2,000 用户，最终在 19,412 用户全量 validation 评价，catalog 为 11,924 商品。训练历史保持时间正序。仅 11,876 个训练可见商品参与 RQ 重建优化；完整 catalog 均保留可生成编码。

- RQ：batch 1024，最多 10,000 商品 pass，至少 200；每 50 pass 检查训练重建误差，连续 10 次改善不足 0.1% 可停止。
- 联合训练：batch 512，每 pass 214 updates，最多 400 外层 epoch（ID / REC 各 200）；各 optimizer 最多 42,800 updates，warmup 分别 4,000 / 8,000，日程分别走完。
- 至少联合训练 160 epoch，每 10 epoch 验证；达到最小预算后以 0.0001 NDCG@10 改善和 patience 6 判断平台。
- 固定编码微调：最多 100 epoch、21,400 updates，至少 40；每 5 epoch 验证，patience 5；保留跨联合/微调阶段最佳模型。
- constrained beam 50，输出合法且去重的 50 个商品，固定用户顺序与历史 GRAM / PCRF 配对。

## 核验与计算预算

复用的模型核心通过 5 项测试：作者损失公式一致、双向对齐梯度及冻结边界、完整小词表穷举与 beam 分数一致、连续训练与 checkpoint 恢复一致、最小预算与 patience 分离。Toys 的商品编号与 CF 原训练的排序索引完全匹配；全量/子集历史参照的六项指标重新计算一致。

GPU 5（`GPU-07374397-b213-09c1-2988-718acb6ed231`）实测 batch 512：峰值 allocated 4.42 GiB；ID / REC / finetune P90 单 update 0.0512 / 0.1457 / 0.1220 秒。完整日程包含全部 RQ 上限、联合、微调、验证、checkpoint 和 1.25 安全系数，估计 21,764.56 秒（6.05 小时），预算 43,200 秒。GPU 共享负载会影响耗时，估计不是保证。profile 的训练权重全部丢弃，正式实验重新 seed 和初始化。

Toys 使用独立 adapter 和 runner，复用未修改的 Beauty 模型核心；runner 仅改输入导入及快照/哈希路径。保留 Beauty 原 source hash，两个已运行 Beauty 实验持续原任务。

## 结果判据

参照全量 GRAM NDCG@10 = 0.07627451426，GRAM + PCRF = 0.07871552229。趋势子集只用于检查走势和选 checkpoint，不能与全量参照混比。

最终报告 NDCG / Hit @5、10、50，并以同用户配对 bootstrap 2,000 次给 NDCG@10 / Hit@10 区间。点估计超过 GRAM 但区间跨零记为弱正信号；需进一步同预算、多 seed 验证才能声称稳定提升。超过 GRAM 但未超过 PCRF，应明确实用增益边界。工程失败、硬预算截断或仅早期低分，均不能直接宣称机制失败。

## 执行与状态

配置：`experiment/phase18/config/s18_etegrec_toys.json`。

```bash
bash experiment/phase18/run_stage18_etegrec_toys.sh --mode profile
bash experiment/phase18/run_stage18_etegrec_toys.sh --mode run
```

- [实时状态](../../artifacts/phase18/etegrec/toys/run_v1/status.json)
- [事件日志](../../artifacts/phase18/etegrec/toys/run_v1/events.jsonl)
- [显存与速度实测](../../artifacts/phase18/etegrec/toys/profile_v1/profile.json)
- 最终结果：`artifacts/phase18/etegrec/toys/run_v1/result.json`
- tmux：`s18_etegrec_toys`

本计划新增 Toys 筛选。第四方向在三任务实际占用稳定后检查资源并形成单独选型说明。

2026-09-09 14:42 正式启动：PID `2572693`，GPU 5，启动后真实 RQ updates 与 checkpoint 已核验。新增 [三任务汇总 status](../../artifacts/phase18/parallel_screen_20260909/status.json)（15 秒刷新），详见 [启动报告](../../report/第十八阶段/Stage18_ETEGRec_Toys启动与并行资源报告.md)。
