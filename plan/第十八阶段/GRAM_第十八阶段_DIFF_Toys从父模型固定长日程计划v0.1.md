# 第十八阶段：DIFF→GRAM Toys 从父模型固定长日程计划 v0.1

## Material Passport

- 日期：2026-09-09；academic-research-suite / experiment-agent / run，inline。
- 用户要求：考虑 Toys 末段仍略涨，检验更长训练；随后明确希望像 Beauty 一样从原父模型重新开始，排除前面设置变化的影响。
- 状态：17:18 已在GPU0启动，PID3299278；13项测试和真实GPU检查通过，效果仍 UNVERIFIED。见[启动报告](../../report/第十八阶段/Stage18_DIFF_Toys从父模型长日程启动报告.md)。

## 1. 本轮回答的问题

在原 GRAM 父模型上重新初始化 DIFF 分支和 Adam，预先固定 1 frozen + 20 joint 的完整日程，能否改善 Toys 的最终全量 validation？

原 `full_r2` 已用全量训练数据，累计 1+10 轮；第 7、9、11 轮子集 NDCG@10 为 0.069260604、0.069649344、0.069738526，末段仍略涨，但最后增量已缩小。最终全量 NDCG@10=0.076092257，相对原 GRAM 0.076274514 为 -0.2389%。这些结果支持一次有上限的更长日程检查，尚不能证明旧任务没有收敛或承诺延长会提点。

本轮不读取旧 DIFF checkpoint 作为训练起点。“重头跑”指重新训练 DIFF 适配，GRAM 加载原 Toys epoch30 checkpoint；不是把 GRAM 从随机初始化重新训练。刚准备的旧 epoch11 续训方案未启动，已被本方案取代，没有新增训练开销。

## 2. 冻结合同

| 项目 | 固定内容 |
|---|---|
| 父模型 | `GRAM/log/Toys/1_20260720_1830/id_0_rec_30/model_rec_phase_1_epoch_30.pt` |
| 父模型 SHA256 | `b0d76ea4da9a40b1be43c55c1c7e20cdca4e1eff0194acbbe1b3cc15f471fd82` |
| 初始化 | 严格加载原 GRAM；DIFF / memory 新初始化；全新 Adam；seed2023 |
| 数据 | 沿用冻结 prepared：109,361 train、19,412 validation、11,924 商品；不读 test |
| 结构 | 原 Toys categories、hidden256、2 层、2 heads、history20、cutoff9、alpha0.5 |
| 训练 | 1 frozen + 20 joint，共21轮；不按分数提前停止，不自动续档 |
| 学习率 | 原峰值：DIFF 1e-3、GRAM 1e-5；每阶段5% warmup + linear decay；联合阶段从第一步按20轮安排 |
| 更新数 | 每轮855次；冻结855次＋联合17,100次＝17,955次；冻结 warmup42、联合 warmup855 steps |
| batch / 精度 | 有效128，microbatch4，累积32；FP32，TF32关闭，原 beam50 生成路径 |
| 选模 | 同一固定2,000用户 cohort；epoch1、3、5、…、21，共11次；按 NDCG@10 选本轮最佳模型 |
| 完整验证 | 对本轮选中 checkpoint 评估19,412用户，复用历史 GRAM / PCRF 参照；旧 DIFF 单独作为比较对象 |
| 资源 | GPU0，UUID `GPU-4e97077a-5bab-99a7-2fdc-598df6be74cc`；启动前核对余量并实测 |
| 硬预算 | 最多60小时，包含训练和验证；达到上限记录 TIMED_OUT，不自动重启 |

与原模型配置相比，只改 `training.joint_epochs: 10 → 20`，由入口做严格差异校验。早停字段沿用原配置但固定日程 runner 不启用它。每阶段学习率变化和所有选模分数均落盘，不根据中途结果临时改变日程。

训练复用现有固定日程函数，Beauty 在跑的源码、配置和任务不修改。旧 DiffGRM、ETEGRec Toys、ReaRec Toys 不启动。

## 3. 核验与成本

先核验21轮全量覆盖、尾批累积、阶段冻结、旧第11轮位置学习率仍有效、最后衰减至零，以及加载实际最佳模型做最终验证。GPU smoke 检查最长历史下真实冻结/联合更新、各分支梯度、beam50候选和 checkpoint 保存重载一致；额外16个固定用户只测生成时间，不作效果评价。正式任务重新加载父模型，smoke 权重不复用。

旧 Toys 接续训练和全量验证耗时受共享 GPU 影响。本轮轮数翻倍，先按约36–48小时规划，GPU smoke 和正式首轮实测速率再更新，不能把此估计当完成承诺。60小时是硬上限，不代表预期必须用满。

## 4. 结果判读

1. 比较本轮完整 validation 与原 GRAM、旧 DIFF→GRAM；同时报告选模子集和子集外，避免将子集上涨当作全量提升。
2. 若本轮最佳出现在11轮之后且全量改善，记录“长日程方案产生正信号”；是否稳定提点仍需后续复现，是否属于 DIFF 的贡献仍需匹配的纯 GRAM 续训对照。
3. 若结果持平或下降，降低这档方案的投入优先级；不无限增加轮数追逐正值，也不宣称所有更长训练均无效。
4. 原实验有短日程接续/学习率桥接，本轮同时改成固定日程并增加训练预算。因此与旧实验的差值不能单独归因为多10轮；本轮第11轮 checkpoint 的学习率历史也不同于独立的1+10短日程实验。
5. 仍是同seed、同validation选模的探索实验，不是独立 test 或多seed确认；本轮没有 PCRF 新组合结果。

## 5. 入口与状态

- 配置：[执行配置](../../experiment/phase18/config/s18_diff_gram_toys_long_fixed.json)、[模型配置](../../experiment/phase18/config/s18_diff_gram_toys_long.json)。
- 入口：[s18_diff_gram_toys_long.py](../../experiment/phase18/protocol/s18_diff_gram_toys_long.py)。
- 状态：`artifacts/phase18/diff_gram/toys/long_fixed_v1/status.json`。
- 日志：`artifacts/phase18/diff_gram/toys/long_fixed_v1.console.log`。
- 最终：同目录 `result.json`；只有 `COMPLETED` 才代表完整训练和最终验证结束。
- 旧 `artifacts/phase18/diff_gram/train_v1/` 保留原已完成结果，不能与本轮状态混读。

```bash
bash experiment/phase18/run_stage18_diff_gram_toys_long.sh --mode smoke
bash experiment/phase18/run_stage18_diff_gram_toys_long.sh --mode run
```
