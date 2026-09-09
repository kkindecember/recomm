# 第十八阶段：DIFF→GRAM Beauty 固定日程复核计划 v0.1

## Material Passport

- Origin Skill / Mode：academic-research-suite / experiment-agent / run，inline。
- Origin Date：2026-09-09（Asia/Shanghai）；Version Label：s18_diff_beauty_fixed_v1。
- Verification Status：UNVERIFIED（新候选效果尚未产生）；执行检查结果另行记录。
- 用户授权：“那你先做beauty的吧”。承接前文确认：只做 Beauty，从原始 GRAM 父模型初始化，固定完整日程，复核现有小幅正信号。

## 1. 问题与已有证据

原 `full_r2` 已用全量训练数据，完成 1 轮冻结适配 + 10 轮联合训练；中间按 2,000 用户选模，最终评估全部 22,363 个 validation 用户。Beauty NDCG@10 = 0.065354591，相对 GRAM 的 0.064973703 提升 0.5862%，Hit@10 净增 23 人。NDCG@10 增量的配对 95% 区间 [-0.000133315, +0.000910499] 包含 0。详见 [双域最终报告](../../report/第十八阶段/Stage18_DIFF_GRAM_双域full_r2结果与方向结论报告.md)。

该轮经过原日程 → screen_r1 短档 → full_r2 学习率桥接，不能称为全程固定日程。本次检验：保持相同父模型、模型配置、seed 和数据，用事先固定的训练日程，能否再次取得相对 GRAM 的正增量。

这次仍为 seed 2023，不是多 seed 稳健性确认，也不增加匹配预算的纯 GRAM 续训对照。新数据、独立 test 和单算子因果归因均不在本轮结论范围内。

## 2. 固定实验合同

| 项目 | 配置 |
| --- | --- |
| 数据集 | Beauty，全量 131,413 条训练转移，22,363 validation 用户，12,101 商品 |
| 父模型 | 第一阶段 Beauty epoch 25 GRAM checkpoint |
| 父模型 SHA | `0df16263344afec8a40e29a7a33e43e7bbe254e0ab97799c9103ad757e60fa89` |
| 初始化 | 原 GRAM 参数严格加载；DIFF 分支重新初始化；Adam 全新；不加载 screen_r1 / full_r2 / smoke 权重 |
| seed | 2023，与此前相同 |
| 结构 / 属性 | 沿用原 DIFF→GRAM Beauty：categories + brand、hidden 256、2 层、2 heads、cutoff 5、alpha 0.3 |
| 训练 | 1 frozen epoch + 10 joint epochs，共 11；本次关闭按分数提前停止，固定完成全部轮数 |
| 学习率 | 新分支峰值 1e-3，GRAM 1e-5；每阶段固定 5% warmup + linear decay，两个阶段之间保留 Adam 动量 |
| 更新次数 | 每轮 ceil(131413 / 128) = 1,027；总共 11,297 次；frozen warmup 51 steps，joint warmup 513 steps |
| Batch / 精度 | 有效 128，microbatch 8，累积 16；FP32；保持原 TF32 关闭设置 |
| 生成 | 原 GRAM 词法生成路径，beam 50；cross-attention cache 复用关闭 |
| 选模 | 固定旧 2,000 人 cohort；epoch 1、3、5、7、9、11 检查，按 NDCG@10 选 checkpoint |
| 最终评估 | 只对本次选中 checkpoint 评估全量 validation；与历史同用户 GRAM / PCRF 比较 |
| 资源 | GPU 4，UUID `GPU-4cc17bbe-a85f-bdab-44ef-bbe29d1afc09`；一条任务 |

冻结阶段独立日程从 0 线性升降，联合阶段按预设重新开始其 10 轮日程；这是计划内阶段转换。中途不因效果修改预算、重设曲线或接入旧分支。验证恢复 Python / NumPy / CPU / CUDA RNG，防止中间评估改变训练随机流。

配置复用 [原模型配置](../../experiment/phase18/config/s18_diff_gram_beauty.json)，新 [执行配置](../../experiment/phase18/config/s18_diff_gram_beauty_fixed.json) 固定其 SHA、cohort SHA 和历史参照 SHA。准备数据继续沿用原 manifest 的文件哈希校验，不重建训练集，也不重新训练 / 推理 GRAM 与 PCRF 基线。

## 3. 预算与停止边界

原 Beauty 接续阶段 joint 每轮约 1.49–1.92 小时，中位数约 1.66 小时；frozen 约 1.09 小时，最终完整验证约 4.14 小时。按 10 joint、6 次子集检查与一次完整验证估计，整轮先按 **24–30 小时**。GPU 4 与原 GPU 7 的共享负载不同，后续用实际吞吐修正 ETA，不承诺固定完成时间。

最长运行 **36 小时**，包含初始化、训练与验证；达到上限记录 TIMED_OUT，不自动重启或追加。非有限 loss / gradient、数据身份不一致、文件覆盖或 GPU smoke 不通过时停止并保留记录。每轮保存模型、Adam、scheduler 和训练 RNG；运行中的科学负信号不会临时缩短已固定的 11 轮日程。

Toys 本次不启动；DiffGRM 不续训；ETEGRec 使用其已有任务和配置。

## 4. 如何判读新结果

1. 同时报告新候选相对原 GRAM 和原 DIFF→GRAM full_r2 的完整验证差分。复用原同用户 rank 作配对分析，报告绝对 / 相对增量和命中用户得失。
2. NDCG@10 与 Hit@10 再次正向且无明显整体退化时，记为“固定日程下复现正信号”，再检查冻结 PCRF 组合的价值；不自动升级为已证明稳定或结构因果贡献。
3. 若结果仍接近零或反向，保留该事实和区间，不通过反复换 seed、扩大轮数或挑部分用户追逐正值。是否继续投入另作判断，不自动接下一档。
4. 若仅子集为正而全量为负，按全量探索结果记录；不把全量 validation 或子集外人群改称独立 test。

## 5. 运行入口与产物

```bash
bash experiment/phase18/run_stage18_diff_gram_fixed.sh --mode smoke
bash experiment/phase18/run_stage18_diff_gram_fixed.sh --mode run
```

两步顺序执行。正式入口校验真实 GPU smoke、执行配置及运行源码 SHA，重新加载父模型，拒绝覆盖已有 run 目录。

- 正式状态：`artifacts/phase18/diff_gram/beauty/confirm_v1/status.json`
- 正式日志：`artifacts/phase18/diff_gram/beauty/confirm_v1.console.log`
- 逐事件日志：`artifacts/phase18/diff_gram/beauty/confirm_v1/events.jsonl`
- 最终结果：`artifacts/phase18/diff_gram/beauty/confirm_v1/result.json`（完成后生成）
- checkpoint：同目录 `initial.pt`、`last.pt`、`best_trend.pt`
- GPU 检查：`artifacts/phase18/diff_gram/beauty/confirm_smoke_v1_gpu4_mb8/`
- 训练入口：[s18_diff_gram_fixed.py](../../experiment/phase18/protocol/s18_diff_gram_fixed.py)
- 检查：[test_diff_gram_fixed.py](../../experiment/phase18/tests/test_diff_gram_fixed.py)，与原 DIFF 行为检查共 11 项通过。

旧 `train_v1/status.json` 继续对应已完成的 `full_r2`；本轮看 `confirm_v1/status.json`，不要混读。

**2026-09-09 13:28 启动记录：** GPU 4，PID 2195156，tmux `s18_diff_gram_beauty_fixed`。11 项 CPU 检查与真实 GPU smoke 均通过；13:30 已完成第 1 轮中的 32 次 optimizer 更新，处理 4,176 条训练样本。新任务已进入真实训练，详见 [启动报告](../../report/第十八阶段/Stage18_DIFF_Beauty固定日程复核启动报告.md)。
