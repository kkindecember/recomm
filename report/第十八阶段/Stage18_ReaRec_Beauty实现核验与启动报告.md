# 第十八阶段：ReaRec Beauty 实现核验与启动报告

## Material Passport

- 状态：**正式训练已启动，真实更新与 checkpoint 已确认；效果待最终验证**。
- 用户授权：“可以开始这个”，指第四候选 ReaRec。
- 实验：`S18-ReaRec-PRL2-Beauty-S2023`。
- 启动：2026-09-09 15:17:59（北京时间），GPU 6，PID `2752109`，tmux `s18_rearec_beauty`。
- 证据：[启动核验快照](../../artifacts/phase18/rearec/beauty/startup_verification.json)、[执行计划](../../plan/第十八阶段/GRAM_第十八阶段_第四候选ReaRec筛选准备v0.1.md)。

## 实现与原作者的关系

固定作者 commit `7f46432cba8435ae15fc744c53715024c3b2d32c`，源码及 22 个文件的 SHA 已核对。直接加载该版本的 Transformer、KV cache 和 `ReaRecAutoRegressiveWrapper`；保留隐状态回馈、独立推理位置编码和两次额外推理。作者源码保持原样，隔离导入中仅把最大历史常量设为本地 20。[固定作者实现](https://github.com/TangJiakai/ReaRec/tree/7f46432cba8435ae15fc744c53715024c3b2d32c)、[本地源码清单](../../artifacts/phase18/parallel_screen_20260909/rearec_source_manifest.json)。

本地 adapter 负责完整训练前缀、PAD=0、左 padding、合法商品分数与训练/评估。直接训练随机初始化的商品 ID embedding，不加载协同向量或 LLM 特征，不调用原始会读 test 的数据入口。它是作者 ID 路径的本地适配，不能写成论文原数据、原设置的完整复现。

PRL 使用最后状态的 CE、前两状态的递进温度监督，以及带噪声推理视图的对比项。对比相似度保留作者未归一化的点积公式。修复一处运行错误：原 PRL 在前两轮关闭噪声时没有初始化 `cl_loss`，本地在该分支设为 0；开启噪声后的总损失已与固定作者函数数值比较一致。

本地合法商品为 1–12,101；PAD 同时从最终和中间监督的 softmax 类别中移除。全 catalog 打分取 top 50，保存合法商品上的 log-softmax 分数，分数并列时按商品 ID 稳定排序。这与 GRAM 的 beam50 生成成本不同，后续报告需保留该边界。

## 数据与训练预算

完整 train 为 131,413 条前缀，完整 validation 为 22,363 用户；固定趋势子集 2,000 用户。输入文件哈希、用户集合、历史长度以及全量/子集 GRAM 和 PCRF 的六项指标均重新核对一致，见 [输入核验](../../artifacts/phase18/rearec/beauty/input_review.json)。未读取 test，也没有重训或推理历史基线。

设置为 2 层、256 维、2 heads、FFN 300、两步额外推理；层 dropout 0.5，作者 wrapper dropout 0.2。batch 512，每 epoch 257 updates，Adam 固定 LR 0.001、weight decay 0、gradient clip 1。最多 300 epoch；至少 100 epoch 后才启用平台停止，每 5 epoch 趋势验证，连续 10 次未获得 0.0001 NDCG@10 的有效改善可停止。最佳 checkpoint 按实际最高趋势 NDCG@10 选择，独立于 patience 的改善阈值。

温度 0.07、递进系数 5、噪声强度 0.01，递进/对比权重各 1；前两轮关闭噪声。正式训练重新设 seed 2023 和初始化，profile 权重全部丢弃。

## 核验结果

5 项 CPU 测试通过：

1. 带缓存的三个状态与完整非缓存重算一致；零步输出等于同一模型的初始状态。
2. 有噪声的 PRL 损失与固定作者函数一致。
3. 无噪声 warmup 可运行，推理位置梯度非零，PAD embedding 无梯度。
4. 修改被遮蔽的 padding 向量不改变合法商品分数，目标映射正确。
5. 保存和恢复 Adam/RNG 后，带噪声的连续更新轨迹逐参数一致。

GPU 6 真实 batch 512 的核验也通过：推理位置权重发生真实更新；CE / 递进监督 / 对比项的推理位置梯度 L1 分别为 5.116078 / 0.245378 / 6.208284；GPU Adam/RNG 恢复后逐参数一致；全 catalog top50 和零步推理均可执行。

实测峰值 allocated 为 **1.97 GiB**，单 update P90 为 **0.1250 秒**。按最大 300 epoch、全部趋势验证、两次最终全量推理及 checkpoint 计算，加 1.25 安全系数后估计 **3.38 小时**，低于 **4 小时硬预算（含 profile）**。共享 GPU 的后续负载会改变实际耗时，平台早停也可能使其提前结束。详见 [profile](../../artifacts/phase18/rearec/beauty/profile_v1/profile.json)。

启动快照已经记录完整第 8 轮、2,056 次更新，`last.pt` 大小 47,354,623 bytes；第 3 轮开启噪声后出现有限且非零的对比损失。此处只确认训练正确启动，不据初期 loss 判断推荐提点。

## 后续自动产物与判断

训练结束后自动加载趋势验证选中的两步模型，生成全量 validation 的逐用户 top50 与六项指标，和保存的 GRAM / PCRF 做配对 bootstrap（2,000 次）。全量 Beauty GRAM NDCG@10 = 0.06497370269，GRAM + PCRF = 0.06790768155；必须比较相同用户和相同验证范围。

随后用同一个已选 checkpoint 做零步推理，保存独立诊断结果；不按用户事后挑选最优步，也不拿它代替独立训练的零步基线。新增训练始终只有这一条候选。弱正信号、预算截断和工程异常分别报告；配对区间不代表多 seed 稳健性。

- [ReaRec 实时 status](../../artifacts/phase18/rearec/beauty/run_v1/status.json)
- [ReaRec events](../../artifacts/phase18/rearec/beauty/run_v1/events.jsonl)
- 最终结果：`artifacts/phase18/rearec/beauty/run_v1/result.json`
- 最佳/最新完整 checkpoint：`run_v1/best.pt`、`run_v1/last.pt`
- [四任务汇总 status](../../artifacts/phase18/parallel_screen_20260909/status.json)：15 秒刷新，已加入 `rearec_beauty`。本次仅重启只读监视器以加载第四项，其他三条训练保持原进程。

15:21 启动后复核：任务已完成第 10 轮、2,570 次 updates。第 5 轮首次真实趋势验证的 2,000 条输出已经独立核对：预测文件 SHA、固定 cohort 顺序、目标商品、50 个合法去重商品，以及六项指标重算均通过，见 [首次验证核验](../../artifacts/phase18/rearec/beauty/first_validation_verification.json)。两次趋势验证正常完成，`best.pt` 已生成；该进展不构成最终提点结论。
