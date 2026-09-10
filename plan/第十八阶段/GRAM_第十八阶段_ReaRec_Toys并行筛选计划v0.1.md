# 第十八阶段：ReaRec Toys 并行筛选计划 v0.1

## Material Passport

- 日期：2026-09-09；模式：run，Toys 单 seed 补充筛选。
- 用户需求：“有必要跑下 rearec toys 吗，闲着也是闲着”。沿用此前并行探索提点方向的授权。
- 实验：`S18-ReaRec-PRL2-Toys-S2023`。
- 当前阶段：**16:28:45 已按用户指令停止，状态 USER_STOPPED**。最后完整第 34 轮，第 35 轮中断；未完成最小预算和全量验证。输入审核、profile 及 16:19:34 启动记录保留为历史。
- 输入核验：`artifacts/phase18/rearec/toys/input_review.json`；未读取 test。

## 要回答的问题

Beauty 的趋势结果尚未超过 GRAM，不能据此推出 Toys 必然无效。本轮检验同一套 ReaRec PRL 两步设置在 Toys 是否能超过保存的 GRAM / PCRF 验证基线。它是新增一个域的探索性筛选，不是多 seed 确认，也不是参数搜索。

ETEGRec Beauty 已于 16:08 完成，GPU 0 约 24.5 GiB 空闲、利用率约 1%。使用这张卡做 Toys profile 和训练。资源快照不代表独占；正式启动时重新检查显存，最多分配 12 GiB，保留至少 3 GiB 空余。

## 固定数据、实现与训练设置

- 作者源码仍为 `7f46432cba8435ae15fc744c53715024c3b2d32c`，复用已有 source manifest 及逐文件 SHA 验证。
- Beauty 的模型、PRL loss、训练、验证和预算代码保持原文件哈希；Toys 入口在独立模块实例中绑定 Toys 输入 adapter，避免复制训练逻辑。
- 完整 train：109,361 前缀；validation：19,412 用户；固定趋势子集：2,000 用户；合法 catalog：11,924 商品，PAD=0，历史上限 20。
- 同 Beauty：seed 2023；2 层、256 维、2 heads、两次额外推理；batch 512；Adam LR 0.001；grad clip 1；PRL 温度/递进/对比/噪声参数全部相同。
- 最多 300 epoch；100 epoch 起启用平台判据，每 5 epoch 验证，连续 10 次改善未超过 0.0001 NDCG@10 时早停。若一直无改善，最早在第 150 轮停止。
- 按实际最高趋势 NDCG@10 选 checkpoint，训练后对全部 validation 评估，另做同一 checkpoint 的零步推理诊断。

CPU 输入核验重新检查文件哈希、完整用户集合、原始商品映射、目标、时间顺序历史、同用户历史长度以及全量/子集的六项历史指标。商品 ID embedding 从随机初始化训练，不加载 ETEGRec 的 CF 向量。历史 GRAM / PCRF 使用保存的逐用户结果。

## 预算与评价

总预算上限 **4 小时**，含本次 profile、训练、验证和保存。先对真实 Toys batch 测量 P90 update/评估成本、checkpoint 耗时与显存峰值；最大 300 epoch 的全部工作乘以 1.25 系数仍在预算内才启动。profile 同时验证真实推理位置更新、分项梯度、带噪声 Adam/RNG 恢复和合法 top50；profile 权重不进入正式训练。

主指标为全量 validation NDCG@10，GRAM = **0.07627451426**，GRAM + PCRF = **0.07871552229**。同时报告 Hit / NDCG @5、10、50、配对用户 bootstrap（2,000 次）及训练轨迹。趋势子集的 GRAM NDCG@10 为 0.06937056512，不能与全量数字直接比较。

有正信号再安排确认；两域均充分训练后明显落后，则降低当前适配/设置的优先级。预算截断、工程失败和充分训练后的负结果分别记录。单 seed、验证子集选模及相同 checkpoint 的零步诊断都不足以单独证明推理机制有效或无效。

## 执行与状态

```bash
bash experiment/phase18/run_stage18_rearec_toys.sh --mode profile
bash experiment/phase18/run_stage18_rearec_toys.sh --mode run
```

- 配置：`experiment/phase18/config/s18_rearec_toys.json`
- profile：`artifacts/phase18/rearec/toys/profile_v2/profile.json`
- 实时状态：`artifacts/phase18/rearec/toys/run_v1/status.json`
- 事件：`artifacts/phase18/rearec/toys/run_v1/events.jsonl`
- 结果：`artifacts/phase18/rearec/toys/run_v1/result.json`
- tmux：`s18_rearec_toys`
- 汇总已加入：`artifacts/phase18/parallel_screen_20260909/status.json` 的 `rearec_toys`。

## 实测与启动记录

`profile_v1` 在沙箱内因 CUDA 不可见退出，未进行模型更新；保留原失败状态。获得 GPU 访问权限后使用新的 `profile_v2`，预算累计包含前一次检查耗时。正式配置与已通过 profile 按 SHA 锁定。

GPU 0 实测 batch512 的 P90 update 为 0.060905 秒，峰值 allocated 1.963 GiB；每轮 214 updates，300 轮全部训练/验证/保存加 1.25 系数后约 **5,006.77 秒（83.45 分钟）**，通过 4 小时预算准入。推理位置实际更新、三项损失梯度、带噪声 GPU Adam/RNG 精确恢复以及合法 top50 均通过。

正式 PID `3020613`，tmux `s18_rearec_toys`，2026-09-09 16:19:34（北京时间）启动。已确认完整第 2 轮、428 updates 和 `last.pt`。现阶段预计约 30–90 分钟；早停和共享 GPU 负载会影响完成时间。结果仍待训练与最终验证。

启动期间 Beauty 已完成 150 轮，最佳 checkpoint 为第 45 轮，全量 NDCG@10 = 0.04762500932，GRAM = 0.06497370269。本轮 Toys 用于补充域间证据，不能把 Beauty 的负结果预先当成 Toys 的结论。

详见 [Toys 启动报告](../../report/第十八阶段/Stage18_ReaRec_Toys输入核验与启动报告.md)。

## 用户停止与归档

用户随后要求“那你停止toys吧”，已终止 ReaRec Toys PID `3020613`，未停止 ETEGRec Toys 或 DIFF→GRAM Beauty。保留第 34 轮 / 7,276 updates 的 `last.pt` 和第 20 轮 `best.pt`；累计耗时约 9.26 分钟（含 profile）。最佳趋势 NDCG@10 = 0.04995475943，同子集 GRAM = 0.06937056512，仅作为截断阶段记录。

停止原因是依据 Beauty 负结果重新分配实验投入，不是当前训练异常、达到平台或预算到期。原始 KeyboardInterrupt 状态保留在 `run_v1/status_on_interrupt.json`，规范 status/result 为 `USER_STOPPED`；没有完整 Toys validation 结论。详见 [ReaRec 完整结果与主动停止报告](../../report/第十八阶段/Stage18_ReaRec_Beauty完整结果与Toys主动停止报告.md)。本节覆盖上文继续训练和预计完成时间，不自动恢复。
