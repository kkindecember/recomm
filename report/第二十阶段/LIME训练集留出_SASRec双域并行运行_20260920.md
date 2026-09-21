# LIME训练集留出：SASRec双域并行运行

2026-09-20完成更新：两域均已完成200轮并通过独立核验。Toys结束于17:02:11，Beauty结束于17:56:19；用时分别54.67/70.45分钟。GRAM/C1新整轮训练尚未启动。用户提醒后也已确认阶段一、二的旧GRAM模型完整可用，其复用边界见[旧模型核查](历史GRAM模型复用核查_阶段一二三_20260920.md)。下方运行中表述保留历史时点。

2026-09-20 17:54更新：Toys已于17:02:11完成200轮，墙钟54.67分钟；[独立完成核验](../../artifacts/phase20/lime_clean_sasrec_verification_20260920_v1/Toys.json)通过，包含全量相似商品复算、冻结输入和训练记录核对及模型加载评分。Beauty仍在第195/200轮，约剩5分钟，速度受共享CPU负载影响；以[实时状态](../../artifacts/phase20/lime_clean_sasrec_beauty_20260920_v1/status.json)为准。GRAM／C1尚未启动；当前GPU空余显存未达到此前目标batch16检查的30 GiB门槛。本次没有新的推荐效果结果。下文运行中表述保留启动时点。

Origin Skill: academic-research-suite。2026-09-20。当前执行安排：Toys与Beauty的正式SASRec同时运行，分别使用CPU四线程、独立输出目录与检查点。Beauty于16:45启动。GRAM／C1整轮训练均尚未启动；本记录不包含新的推荐效果。

此前Toys先启动是为了先核对整条流程，不是SASRec必须按域串行。用户要求双域同步后，补齐Beauty的数据核验并立即并行启动；Toys原进程和所有已冻结输入保持不变。后续两域相互独立的准备与轻量训练按资源并行，重GPU任务分别核对目标批量和空闲显存，不在实验配置上强制等待Toys结束。

## 两个任务

| 项目 | Toys | Beauty |
|---|---:|---:|
| 截断后的训练用户 | 19412 | 22363 |
| 截断后的训练前缀 | 105867 | 127328 |
| 已留出末次事件 | 3494 | 4085 |
| 商品目录 | 11924 | 12101 |
| 固定训练轮数 | 200 | 200 |
| CPU线程 | 4 | 4 |
| SASRec种子 | 0 | 0 |
| 更新批量 | 128 | 128 |
| 每轮更新 | 152 | 175 |

两域使用相同SASRec结构、Adam、学习率0.001、普通CE＋0.1标签平滑、梯度裁剪1.0；都不使用留出标签、开发标签或test挑选轮数。第1轮及每10轮保存模型、优化器和随机数状态，各自硬上限2小时。完成后生成相似商品提示并退出，不自动启动GRAM。

Beauty使用独立训练入口，避免修改Toys进程正在校验的源码。已比较两个训练循环的AST：除预期用户数之外一致；配置的训练超参一致。导出提示沿用各域GRAM原设置：Toys Top5、Beauty Top10，这只改变提示条数。

## Beauty启动前核验

- 22363名用户逐一核对：仅删除指定4085名用户的末次训练事件，没有该事件之后的训练前缀。
- 截断后的交互总数149691；新旧频次之差恰为留出目标的计数。775名门控校准用户与3310名排序器拟合用户无交集。
- 全量检查SASRec实际数据适配器：历史、目标和50步截断对应正确，负例不在该用户的训练序列中；每轮监督位置123976个。
- ItemCF与频次重新计算；独立复算12个预定商品的共现，674个保存邻居及分数一致。
- Toys的11项冻结输入哈希保持一致，没有重启或覆盖其训练。

CPU是共享资源，两域同时运行后预计时间会随主机负载变化。以各自`training_curve.json`中最近完整轮的耗时估算，不把Toys旧预计直接复制给Beauty。

证据：[Toys实时状态](../../artifacts/phase20/lime_clean_sasrec_toys_20260920_v1/status.json)、[Beauty实时状态](../../artifacts/phase20/lime_clean_sasrec_beauty_20260920_v1/status.json)、[Beauty边界与ItemCF核验](../../artifacts/phase20/lime_clean_beauty_preflight_20260920_v1/verification.json)、[训练循环与设置匹配](../../artifacts/phase20/lime_clean_beauty_preflight_20260920_v1/runner_comparison.json)、[Beauty固定配置](../../experiment/phase20/lime_clean_sasrec_beauty_v1.json)、[此前上游运行检查与成本](LIME训练集留出_上游接入检查与正式SASRec启动_20260920.md)。
