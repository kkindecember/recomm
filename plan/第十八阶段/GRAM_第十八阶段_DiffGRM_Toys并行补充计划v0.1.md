# 第十八阶段：DiffGRM Toys 并行补充计划 v0.1

日期：2026-09-08。用户明确要求增加 DiffGRM Toys，形成 DIFF→GRAM / DiffGRM × Toys / Beauty 四个实验；只通过四个 status 文件观察，不添加总览工具。

## 配置和预算

沿用 [DiffGRM 原生计划](GRAM_第十八阶段_耗时估计与DiffGRM原生并行计划v0.1.md) 的固定作者源码、现有 CUDA 12.6 环境、本地 Sentence-T5 权重、train-fit PCA / OPQ、guided least 四视图训练和 CPD 解码。此前“先只跑 Beauty”的资源决策由本次用户指令更新。

根据固定 commit `238911435b6ca4f576d0b4d943359b98c865c172` 的 README Toys 命令：hidden=1024、heads=8、encoder 1 层、decoder 4 层、FFN=1024、lr=0.003、label smoothing=0.15；其余与原生 Beauty 流水线一致。使用完整作者模型，不把 Beauty 小模型改名为 Toys。

- 复用原 Toys 序列和商品文本：训练 109,361 条完整可见转移；原 validation；最短历史 1 与 GRAM 训练信号一致，最长历史 50 沿用来源模型。最后一项 test 不构造评估样本。
- 直接读取第二阶段 `artifacts/phase2_toys/checkpoint_selection.csv` 中 epoch 30 的 validation：NDCG@10=0.07627451426000033，Hit@10=0.11941067381001443。不重跑基线或 PCRF。
- 作者有效 batch 1024；microbatch=32、梯度累积 32 次；FP32。生成 batch=2、CPD beam=256，以完整模型 GPU profile 验证显存可行性。
- 最多 200 epoch；epoch 10 起每 10 epoch 做完整 validation，至少 100 epoch 后允许连续 5 次无改善早停。每轮约 107 updates，最低预算覆盖 10,000-step warmup。完整运行耗时需要实测，不能套用 Beauty 的速度。
- 以真实 item NDCG@10 选模；SID 碰撞按 train 频次和固定 ID 展开，去重、不填充重复项。history / beam / 容量与 GRAM 有差异，当前只作探索比较，最终方法确定后统一重跑匹配预算与 PCRF 组合。

## 执行

配置：`experiment/phase18/config/s18_diffgrm_native_toys.json`。

计划 GPU 4 / `GPU-4cc17bbe-a85f-bdab-44ef-bbe29d1afc09`；独立 tmux `s18_diffgrm_native_toys`。原 GPU 3 DIFF Toys、GPU 7 DIFF Beauty、GPU 1 DiffGRM Beauty 继续运行。

自动顺序：完整模型 GPU profile → 本地商品编码和 OPQ → 最长真实历史 smoke → 从头全量训练 / validation。使用已通过 4 项 CPU 检查的同一 runner；Toys 再做本域完整模型 GPU 和真实数据检查。

输出：`artifacts/phase18/diffgrm_native/toys/run_v1`。启动时禁止覆盖已有目录；异常写入 status 的 FAILED / error / traceback。

## 四个观察入口

1. DIFF→GRAM Toys：`artifacts/phase18/diff_gram/train_v1/status.json`
2. DIFF→GRAM Beauty：`artifacts/phase18/diff_gram/beauty/train_v1/status.json`
3. DiffGRM Toys：`artifacts/phase18/diffgrm_native/toys/run_v1/status.json`
4. DiffGRM Beauty：`artifacts/phase18/diffgrm_native/beauty/run_v1/status.json`

`INITIALIZING` / `PREPARING` 是检查或准备；`TRAINING` 是训练；`VALIDATING` 是验证；`COMPLETED` 才是本轮结束；`FAILED` 看 error / traceback。训练时看 epoch、seen、optimizer_steps 和 updated_at 是否持续变化；验证时看 evaluated。epoch_eta_seconds 只估计当前训练轮，不是整个实验的剩余时间。轮次刚切换的短时间内，旧 runner 的部分统计可能沿用上一条记录，以随后约 30 秒更新后的同轮进度为准。OPQ 是 CPU 拟合，期间 status 可能数分钟没有新计数。

2026-09-08 13:25 启动记录：tmux `s18_diffgrm_native_toys` 已建立，PID `2451416`，GPU 4。完整作者 Toys 模型 54,636,544 参数，microbatch 32 的 GPU profile 通过，训练峰值 reserved 1,504 MiB、生成 3,940 MiB，生成形状 `[2, 256, 4]`，checkpoint 恢复结果一致。当前 `PREPARING / sentence_t5`，目录 11,924 商品；随后自动 OPQ、真实数据 smoke 和全量训练。四个 status 文件均已存在，另外三个仍在 TRAINING。准备状态尚不算正式训练；以上 profile 不作提点证据。
