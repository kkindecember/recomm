# 第十八阶段：DIFF Beauty 并行筛选补充计划 v0.1

## Material Passport

- 日期：2026-09-08；academic-research-suite / experiment-agent，inline。
- 用户指示：Beauty 与 Toys 同时跑；继续复用第一、二阶段 GRAM 和既有 PCRF 数据，目标是筛出能单独提点的内部结构。
- 本文覆盖主计划中“等 Toys 后再跑 Beauty”的顺序安排。两域独立训练和选模，不等待另一域结果；不合并数据，不跑新基线。

## 1. Beauty 的已有参照与数据

- 原 config：`GRAM/log/Beauty/4_20260718_2153/config.json`。
- parent：第一阶段 validation 最佳 **epoch 25**，`GRAM/log/Beauty/4_20260718_2153/id_0_rec_30/model_rec_phase_1_epoch_25.pt`。
- parent SHA：`0df16263344afec8a40e29a7a33e43e7bbe254e0ab97799c9103ad757e60fa89`。
- validation NDCG@10：**0.06497370269327131**；Hit@10：**0.10875106202209006**。
- 参照从 `artifacts/phase1_beauty/metrics_seed2023.json` 的选模字段及 `logs/train_seed2023.log` 的 epoch 25 validation 块交叉核对；不能使用同一 metrics 文件中的 test NDCG@10 代替 validation。
- 原 lexical ID：`hierarchy_v1_c128_l7_len32768_split`；SASRec 相似商品 top 10。复用 Beauty 自己的完整文本与原生 Collator，并验证样本 / 分词一致性。
- 目录 12,101 件商品。原始字段核查：仅 2 件无类别；2,114 件无品牌，其余品牌字段覆盖 2,043 个不同值，因此本域同时使用 categories / brand。最终词表仅在 train 出现商品上拟合，PAD / UNK 分开。
- 同原始 leave-one-out；仅构造 train 和 official validation，不构造或评估 test。

## 2. 模型、训练与判断

与 Toys 共用 DIFF 双路频率过滤、独立注意力、共享 FFN、对齐与 item CE，逐商品表示追加为 GRAM decoder 的 memory。保留本域原词法输出，机制迁移不等于原生 DIFF 端到端复现。

按已固定作者 commit `ef6283a3bf9ed242a9b867fcf63368f0fb36eb33` 的 Beauty 配置与实际 run_diff.sh：hidden=256、layers=2、heads=2、FFN=256、hidden dropout=0.5、**attention dropout=0.3、c=5、concat、alpha=0.3、lambda=100**。历史长度 50→20 及总辅助权重 0.1 沿用迁移方案；不把 Toys 的 c=9 / alpha=0.5 照搬到 Beauty。

训练预算、学习率、FP32、effective batch=128 与 Toys 相同：1 epoch 冻结 GRAM 学习新分支，再最多 10 joint epoch；第 1、3、5、7、9、11 epoch 后完整 validation，至少 4 joint epoch 后按连续 3 次无改善停止，以 NDCG@10 选模。Smoke 更新不用于正式训练。新结构先与 Beauty 已有 GRAM 比较，出现有希望的独立收益后再检查 PCRF 叠加。

## 3. 并行资源与验证

Toys 继续使用已有独立进程、配置及启动源码快照。Beauty 使用另一个 GPU 和独立目录；不停止其他用户任务、不做占卡循环。

曾因 beam 50 缓存重排峰值考虑可选 cross-attention KV 缓存复用，但 GPU 精确一致性检查未通过，**正式 Beauty 配置已关闭该优化，沿用原生成路径**。GPU 7 后续空余已足够，不继续该优化研究。下面检查要求与执行记录保留为工程尝试溯源。

所需检查：缓存张量逐元素等于原完整重排；多用户生成候选 / 分数精确一致；Beauty 真实模型短历史 beam 50 与原流程候选 / 分数精确一致；最长历史 20 的真实更新、生成、checkpoint 重载和显存测量。此为运行所需的一次工程检查，不开展新一轮数值机制研究。

## 4. 文件与状态

- 配置：`experiment/phase18/config/s18_diff_gram_beauty.json`。
- prepare：`artifacts/phase18/diff_gram/beauty/prepared_v1`。
- 正式输出：`artifacts/phase18/diff_gram/beauty/train_v1`。
- 日志：`artifacts/phase18/diff_gram/beauty/train_v1.console.log`。
- tmux：`s18_diff_gram_beauty`。

```bash
bash experiment/phase18/run_stage18_diff_gram.sh status --config experiment/phase18/config/s18_diff_gram_beauty.json --output artifacts/phase18/diff_gram/beauty/train_v1
```

`TRAINING` / `VALIDATING` 表示仍在运行；`COMPLETED` + `result.json` 表示该轮结束；`FAILED` 需区分工程异常与效果差。两域分别报告与各自历史 baseline 的 delta，不能比较 Beauty 与 Toys 的绝对分数来判断结构优劣。

## 5. 执行记录

2026-09-08：已核对 Beauty parent、原配置、属性可用性与作者实际配置；准备执行 CPU / GPU 检查并启动并行训练。尚无 Beauty DIFF 效果结果。

- Prepare 完成：train 131,413、validation 22,363、目录 12,101；train 出现商品 12,068；train-fit 类别词表 1,069、品牌词表 2,041（均含 PAD / UNK）。224 个训练样本、7 个验证样本与原生 GRAM 构造逐字段一致，分词张量检查通过，parent SHA 和 validation 参照匹配。
- 9 项 CPU 测试通过，包括新增缓存重排等值 / 跨用户保护及多用户生成候选、分数精确一致性检查。
- GPU 7 空余显存已升至约 24 GB，使用 UUID `GPU-7876ca7a-7bb3-eb10-f22b-2ec3b17b9253`；真实 smoke 选择 microbatch=8、最长历史 20，正式仍 effective batch=128（accumulation=16）。Toys 保持 GPU 3 / microbatch=4 / accumulation=32，均不改变有效 batch。
- 共享代码仅做有默认值的域配置扩展和 Beauty 显式启用的缓存优化；Toys 正在运行的进程使用已加载模块与启动快照。Beauty 单独保存本次源码、数据和参数哈希。

- `smoke_v1_gpu7_mb8`：冻结 / 联合训练、长历史生成与 checkpoint 重载完成；但可选缓存复用未通过短历史 beam 50 候选 / 分数整体精确比较。配置已改为 `reuse_cross_attention_cache=false`，正式采用原生成路径；不使用该次 smoke 权重，不开展追加数值诊断。此条覆盖上文“显式启用”的先前准备状态。

- `smoke_v2_gpu7_mb8_native`：**SMOKE_PASSED**。真实冻结 / 联合更新均有有限 loss 和有效梯度；最长历史 20、beam 50 合法唯一商品生成、checkpoint 保存重载一致性通过。联合更新 peak allocated 7,732.11 MiB / reserved 8,104 MiB；原生 beam 50 peak reserved 7,096 MiB，不含额外 CUDA 运行时占用。仅证明工程能运行，不能作为提点结论。
- 2026-09-08 12:49 左右，已向 tmux 会话 `s18_diff_gram_beauty` 提交 GPU 7 的正式任务，microbatch=8 / accumulation=16；重新加载原 epoch 25 parent 和同 seed 新分支。Toys 的 GPU 3 任务持续运行，两域均自动按各自训练日程执行验证与选模。

- 正式训练已确认实际推进：2026-09-08 12:50:40（Asia/Shanghai），Beauty PID `2068629`，状态 `TRAINING / warmup / epoch 1`，840 / 131,413 样本、6 次 optimizer update，loss 有限；总参数 66,928,048，其中新增 6,410,672。Toys PID `1911709` 同时存活并持续训练。当前两域均尚无完整 validation 结果，不声明提点。
