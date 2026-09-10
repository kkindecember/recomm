# 第十八阶段：ETEGRec Beauty 实现核验与启动报告

**结果更新（2026-09-09）：** Beauty 已于 16:08:45 完成全部计划阶段并平台早停，最终选择 `joint_290`，全量 NDCG@10 = 0.04912738797，相对 GRAM -24.39%。48 次趋势和一次完整验证的保存预测已 CPU 复核通过；完整指标、训练量、分组与投入决策见 [Beauty 完整结果报告](Stage18_ETEGRec_Beauty完整结果与投入结论报告.md)。当前配置归档、不追加同设置 Beauty 训练。下文为启动时记录。

## Material Passport

- 日期：2026-09-09。
- 来源：academic-research-suite / experiment-agent；inline implementation + run。
- 用户接续：“那继续吧”。依据已制定的 Beauty 单域、单 seed、12 GPU 小时计划实施。
- 状态：`RUNNING`。工程核验通过、真实训练已启动；尚无候选完整推荐效果结论。
- [执行计划](../../plan/第十八阶段/GRAM_第十八阶段_ETEGRec联合编码学习与Beauty单域筛选计划v0.1.md)。

## 1. 实际实现

作者版本固定为 `58d9736afc28e03e190a107a5fa22f5241be6088`，来自 [ETEGRec 作者仓库](https://github.com/BishopLiu/ETEGRec/tree/58d9736afc28e03e190a107a5fa22f5241be6088)。通过 GitHub API 确认 commit，下载该 commit 的归档；[来源清单](../../artifacts/phase18/etegrec/sources/source_manifest.json)保存归档和 24 个文件的哈希。早先 Git clone 受本机失效代理和连接超时影响，没有用浮动源码或另一模型代替。

直接载入作者 `model.py`、`vq.py`、`layers.py`，不修改作者文件。适配代码负责本地 train / validation、已有 CF512 向量、独立 optimizer 日程、checkpoint、合法商品 beam 和历史预测配对。

主要差异均保留在结果解释中：

- 商品输入使用已有 Beauty item-head 的归一化 512 维向量，冻结；不新训协同模型，不冒称作者 SASRec 向量。
- 按 1-based item ID 分配含 PAD 的 12,102 行 embedding，保留全部 12,101 件商品。
- 使用作者根目录的 RQ-VAE 类执行重建与量化预训练；该 commit 的预训练入口引用的 `models` 目录未随仓库提供，未另外换一个量化器。
- 原样保留作者两项对齐的计算，包括距离 logits 的约定；CPU 测试与从固定作者源码提取的函数逐项比较。
- REC / ID 日程按各自实际更新数计算；冻结商品向量、两阶段交替及固定编码训练均落实。码本的 `initted` 状态随 checkpoint 保存。
- 生成时先编码一次历史，再执行目录约束的四 token beam 50；保存真实 item ID 与平均 token 对数概率，不补入旧 GRAM 的候选。
- 为 transformers 4.56 的 wrapper 兼容性补足旧 cache 属性；训练和推理都使用现有本地环境，未安装新依赖。

实现入口：[adapter](../../experiment/phase18/core/etegrec_adapter.py)、[runner](../../experiment/phase18/protocol/s18_etegrec.py)、[配置](../../experiment/phase18/config/s18_etegrec_beauty.json)、[启动脚本](../../experiment/phase18/run_stage18_etegrec_beauty.sh)。

## 2. 核验结果

运行 `python -m unittest experiment.phase18.tests.test_etegrec -v`，5 项通过：

1. 对齐公式与固定作者实现一致。
2. 两项对齐分别向 tokenizer encoder 传递有限非零梯度；ID / REC 阶段冻结边界有效。
3. 小目录完整枚举的四 token 概率排序与 beam 的 item 排序、分数一致。
4. Adam / scheduler / RNG / 码本初始化状态恢复后，后续参数更新与连续训练一致。
5. 最低训练预算前的验证次数不携带到之后的 patience。

首次 GPU smoke 在 25 次 RQ 更新后检查目录，发生 `Code collision overflow: 257`。当时只做了测速预热，尚未完成预训练，不是科学负结果。第二次 smoke 先完成额外 50 个 RQ 商品 pass，编码检查通过；完整码本大小仍为 256，目录未截断。两次记录均保留，正式训练未继承 smoke 模型。

[GPU 核验记录](../../artifacts/phase18/etegrec/beauty/profile_v2/profile.json)：

| 项目 | 实测 |
| --- | ---: |
| 实际训练 batch | 512，无梯度累积 |
| 峰值 allocated 显存 | 4,740,637,184 bytes，约 4.42 GiB |
| KL 对 tokenizer encoder 的梯度 L1 | 0.64239 |
| contrastive 对 tokenizer encoder 的梯度 L1 | 228.22414 |
| 真实 ID / REC / finetune 参数更新 | 三者均非零 |
| GPU checkpoint replay | PASSED |
| 预热后不同三位前缀数 | 5,507 |
| 最大同前缀商品数 | 24；第四位消歧后完整编码唯一 |

以上为工程通过证据，不能解释为 NDCG 已提升。

## 3. 预算与启动

实测 P90：RQ update 0.00742 秒、ID update 0.0510 秒、REC update 0.1379 秒、finetune update 0.1227 秒；验证生成约 0.00320 秒 / 用户。估算还计入 checkpoint 写盘、码表刷新和 RQ 监控。

最大计划为 RQ 10,000 个商品 pass、交替阶段 400 个 epoch、固定编码 100 个 epoch、稀疏趋势验证和一次完整 validation。加入 25% 余量与两次 smoke 后预计 **24,947 秒，约 6.93 小时**；实际受共享 GPU 与存储负载影响。计划预算仍为 12 GPU 小时，超时保留最后完整 checkpoint，无自动追加训练。

- 开始：2026-09-09 12:26:32（Asia/Shanghai）。
- GPU：0，`GPU-4e97077a-5bab-99a7-2fdc-598df6be74cc`。
- tmux：`s18_etegrec_beauty`；训练 PID：`1854985`。
- 命令：`bash experiment/phase18/run_stage18_etegrec_beauty.sh --mode run`。
- 日志：[run_v1.console.log](../../artifacts/phase18/etegrec/beauty/run_v1.console.log)。
- 状态：[status.json](../../artifacts/phase18/etegrec/beauty/run_v1/status.json)；详细进展：[events.jsonl](../../artifacts/phase18/etegrec/beauty/run_v1/events.jsonl)。
- 启动证据：[startup_verification.json](../../artifacts/phase18/etegrec/beauty/run_v1/startup_verification.json)。

12:31 CPU 核验真实 checkpoint：RQ epoch 786，9,432 个 optimizer updates；所有 RQ Adam state 的 step 均为 9,432，码本初始化状态已保存，当前最低重建 MSE 约 0.001573。源码快照与当前实现哈希一致。该轮次是商品向量预训练，不是推荐器 epoch，也不是推荐效果指标。

## 4. 当前结论与后续

单域正式筛选已开始。后续自动执行既定的预训练、交替训练、固定编码训练与一次完整验证；主指标与已有相同用户的 GRAM / PCRF 输出比较。当前没有重训或重新推理基线，没有读取 test，没有启动 DiffGRM finish 续训，也没有新增其他候选。

完整推荐效果、平台 / 预算停止原因和是否保留候选，要以训练后 `result.json` 及逐用户结果为准。本报告只确认实现、预算及启动，不预先判定方向有效。
