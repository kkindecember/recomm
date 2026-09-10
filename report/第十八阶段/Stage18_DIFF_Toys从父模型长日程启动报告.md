# 第十八阶段：DIFF→GRAM Toys 从父模型长日程启动报告

## Material Passport

- 日期：2026-09-09；academic-research-suite / experiment-agent / run，inline。
- 状态：工程检查 PASSED，正式训练已启动；推荐效果 UNVERIFIED。
- 用户要求：Toys 从原 GRAM 父模型重新开始，预先固定更长日程，排除之前中途设置变化的影响。
- 执行计划：[Toys 从父模型固定长日程](../../plan/第十八阶段/GRAM_第十八阶段_DIFF_Toys从父模型固定长日程计划v0.1.md)。

## 1. 实际启动内容

2026-09-09 **17:18:41（北京时间）**，GPU0、tmux `s18_diff_gram_toys_long` 启动，训练 PID **3299278**。从原 Toys epoch30 GRAM checkpoint 严格加载骨干，重新初始化 DIFF 和 Adam；没有使用旧 DIFF epoch11 或 GPU smoke 的训练权重。

本轮固定 **1 frozen + 20 joint = 21轮**，全量109,361训练样本，每轮855次更新，总17,955次。有效batch128、microbatch4、FP32、TF32关闭；模型配置相对原 Toys 唯一差异为 `joint_epochs: 10 → 20`，入口做严格校验。

每阶段5% warmup和linear decay，联合阶段从第一步按20轮安排。第1、3、5、…、21轮在同一2,000用户cohort选模，然后对本轮最佳模型评估19,412名完整validation用户。关闭按分数提前停止；60小时硬上限，无自动重启或追加。

**17:24:55 核验快照：** `TRAINING / warmup / epoch1`，已处理 **16,152 / 109,361** 样本、完成 **126次 optimizer update**，loss有限且状态持续推进。该数字是启动核验，不是当前实时状态或推荐效果。

## 2. 检查证据

- 13项测试通过：原DIFF行为、固定日程、21轮全数据和尾批累积、冻结边界、第11轮后学习率仍有效、末轮LR归零、加载选中的权重进行最终验证。
- GPU真实最长历史20、microbatch4：冻结/联合各2次更新，要求的分支梯度非零且有限；冻结阶段GRAM梯度为0。
- 联合训练峰值reserved **4,610 MiB**；最长历史beam50生成峰值reserved **7,068 MiB**；保存/重载后候选与分数完全一致。
- 正式初始化记录确认 `optimizer_steps=0`、`fresh_new_branch=true`、`old_diff_checkpoint_used=false`、`smoke_weights_used=false`。
- 正式源码快照与当前源文件SHA一致；原数据、父模型、cohort和历史排名缓存均核验身份。

证据：[GPU检查](../../artifacts/phase18/diff_gram/toys/long_fixed_smoke_v1_gpu0_mb4/smoke_summary.json)、[速度抽样](../../artifacts/phase18/diff_gram/toys/long_fixed_smoke_v1_gpu0_mb4/runtime_profile.json)、[正式初始化](../../artifacts/phase18/diff_gram/toys/long_fixed_v1/initialization_check.json)、[日程](../../artifacts/phase18/diff_gram/toys/long_fixed_v1/schedule_plan.json)、[启动核验](../../artifacts/phase18/diff_gram/toys/long_fixed_v1/launch_verification.json)。

## 3. 成本与结论边界

GPU0短检查中，16个固定抽样用户平均生成0.199秒/人；按这一时刻外推，全量验证约1.07小时，全部11次子集检查加最终验证约2.29小时。该测量使用一次性smoke权重，且共享GPU负载会变化，不作为正式完成承诺。正式首轮前6分钟冻结训练约44.8样本/秒，整轮约40.7分钟；联合训练尚未产生实测，应继续按计划约36–48小时规划、60小时硬上限监控。

这轮同时采用全程固定日程并增加预算，与旧full_r2的差异不能全部归因于多10轮。是否提点由最终完整validation判断，是否属于DIFF贡献仍缺匹配纯GRAM续训对照；没有重跑GRAM/PCRF基线或使用test。

旧epoch11接续方案只准备过代码，未启动训练，已被当前从父模型方案取代。ETEGRec/ReaRec停止任务没有恢复。

## 4. Beauty配置复核

逐字段比较旧 `beauty/train_v1/manifest.json` 和当前 `beauty/confirm_v1/manifest.json`：**模型配置完全一致**；当前配置文件SHA也与冻结执行配置匹配。Beauty沿用本地categories+brand、hidden256、2层2heads、cutoff5、alpha0.3、seed2023、新分支LR1e-3、GRAM LR1e-5、有效batch128。

Beauty同样从原父模型重新初始化DIFF和Adam，但父模型是Beauty epoch25，日程仍为 **1+10轮**。本轮主要固定原日程并关闭按分数提前停止，不是增加到20联合轮，也不等于复现作者独立DIFF的原论文训练配置。

17:25观察，Beauty PID2195156处于第2轮联合训练。第1轮仅冻结适配后的子集NDCG@10=0.075939086，对同子集GRAM的差值-0.000412415；这只是早期选模观测，没有本轮最终全量结果，不据此变更已冻结日程。

## 5. 查看状态

- [统一状态](../../artifacts/phase18/parallel_screen_20260909/status.json)：`jobs.diff_gram_toys_long` 和 `jobs.diff_gram_beauty`，每15秒更新并检查进程存活。
- [Toys状态](../../artifacts/phase18/diff_gram/toys/long_fixed_v1/status.json)：`TRAINING`／`VALIDATING`为进行中，`COMPLETED`为本轮完成。
- [Beauty状态](../../artifacts/phase18/diff_gram/beauty/confirm_v1/status.json)。
- Toys最终结果为同目录 `result.json`；当前尚未产生。
