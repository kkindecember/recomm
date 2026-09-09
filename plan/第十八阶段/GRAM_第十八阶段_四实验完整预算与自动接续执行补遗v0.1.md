# 第十八阶段：四实验完整预算与自动接续执行补遗 v0.1

**2026-09-09 完成更新：** 四条 `full_r2` 均已完成。DIFF→GRAM 两域均训练到绝对 epoch 11，分别选择 Toys epoch 11、Beauty epoch 9；DiffGRM 两域因 patience 分别停在 Toys epoch 110、Beauty epoch 160，选择 epoch 60 / 110，不能称为均跑满 200。

最终结论见 [DiffGRM 双域报告](../../report/第十八阶段/Stage18_DiffGRM_双域full_r2结果与方向结论报告.md)和 [DIFF→GRAM 双域报告](../../report/第十八阶段/Stage18_DIFF_GRAM_双域full_r2结果与方向结论报告.md)。DiffGRM 明显落后；DIFF→GRAM Toys 略负、Beauty NDCG@10 相对 +0.59%，但配对区间包含 0，保留弱正信号。本轮不追加旧方向预算，Beauty `full_r3` finish 未启动。以下日期、PID、自动接管及“当前”描述保留为 09-08 执行历史。

日期：2026-09-08。用户授权：“那还是按照你现在的这个来改下这四个实验吧 我不想白跑”。

本补遗覆盖此前“四条按 screen_r1 短档跑完，再手动决定是否追加”以及“仅 Beauty 恢复完整日程”的执行规则。当前四条候选统一使用完整训练上限、固定子集稀疏验证和早停。优先寻找可提点方向；GRAM / PCRF 只复用已有逐用户结果，不训练、不重新推理，不增加 seed 或消融，不读取 test。最终方法确定后统一重跑正式实验。

## 1. 四条执行预算

| 实验 | 总训练上限 | 早停条件 | 固定 2,000 用户检查间隔 | 执行方式 |
| --- | --- | --- | --- | --- |
| DIFF→GRAM Toys | 已有冻结适配 1 轮 + 最多联合训练 10 轮，即绝对 epoch 11 | 联合训练至少 4 轮，连续 3 次检查 NDCG@10 无改善 | 每 2 个联合 epoch | 当前绝对 epoch 2 保存后自动接续 |
| DIFF→GRAM Beauty | 已有冻结适配 1 轮 + 最多联合训练 10 轮，即绝对 epoch 11 | 联合训练至少 4 轮，连续 3 次检查无改善 | 每 2 个联合 epoch | 当前绝对 epoch 3 保存后自动接续 |
| DiffGRM Toys | 最多总 epoch 200 | 至少 epoch 100，连续 5 次检查无改善 | 每 10 个绝对 epoch | 当前 epoch 19 保存后自动接续 |
| DiffGRM Beauty | 最多总 epoch 200 | 至少 epoch 100，连续 5 次检查无改善 | 每 10 个绝对 epoch | 保留已运行的 full_r2，PID 3647654 |

以上是训练上限，不要求即使长期退化也机械跑满。每条在接续起点检查一次同一子集，起始 checkpoint 也可入选；按子集 NDCG@10 保存最佳模型，结束后只对最佳模型做一次全量 validation。达到轮数上限仍上涨时，标记预算用尽、未确认收敛，不能据此宣称方向无效；不自动无限追加。

DIFF→GRAM 的 10 轮联合训练是本项目结构迁移方案的完整预算，不声称等于作者 DIFF 原系统的完整复现。

## 2. 保留训练成果与学习率接续

DiffGRM Beauty 已从原 epoch 33 恢复原 Adam / cosine scheduler，继续原 10,000-step warmup 与 200-epoch 总日程。本次保留其正在运行的进程，不再次回滚。具体起点、校验与资源上限见 [Beauty 恢复补遗](GRAM_第十八阶段_DiffGRM_Beauty完整日程恢复补遗v0.1.md)。

另外三条在本次接管时仍处于短日程中，采用以下已执行接续方案：

1. 等当前整轮 checkpoint 原子保存完成，校验进程 PID、命令行、用户及进程启动时间，然后只中断该条旧训练。旧轮的权重、Adam 动量 / 更新计数、CPU / CUDA RNG 全部保留。保存点后旧进程来得及处理的少量 batch 可重放，不丢弃当前整轮训练。
2. 单独保存不可变的 `full_r2/source_checkpoint.pt` 和 SHA256，旧 `screen_r1` checkpoint、日志和预测均保留。绝对 epoch 不清零，已完成训练计入总预算。
3. 新日程第一步使用 checkpoint 中实际 LR；用接下来一个 epoch 的更新平滑接入原长日程在当前绝对位置的 LR。此后 DIFF→GRAM 按原 10 联合 epoch、5% warmup 的线性曲线；DiffGRM Toys 按原 10,000-step warmup、200 epoch 的 cosine 曲线。不会在旧 4 / 20 轮短档终点把 LR 降到零。
4. 接入的是原长日程的剩余部分，过去的短日程更新已经发生，无法事后改写。因此三条明确标为 `full_budget_preserved_short_prefix`，不冒称从头未中断的原始学习率轨迹，也不当作匹配训练预算的结构归因实验。它们用于当前方向筛选；正式方法仍须统一重跑。

这三条自动接管最多等待 6 小时；等待期间旧训练正常进行，协调器不建立 GPU 模型或另占一份训练显存。接续后每条最多 48 小时，足以覆盖目前估计的完整预算并给共享负载留余量。异常保留最后完整 checkpoint，记录原因，不自动重试。Beauty 已恢复进程仍沿用原 10 小时上限。

## 3. 评判口径

固定 cohort 继续沿用 screen_r1 按 seed、dataset、user ID 的哈希选择，不根据分数选用户。验证前后恢复训练 RNG。所有子集指标均与历史 GRAM / PCRF 的相同用户比较，不能与全量均值直接比较：

| 数据集 | 同一子集 GRAM NDCG@10 | 同一子集 GRAM+PCRF NDCG@10 |
| --- | --- | --- |
| Toys | 0.0693705651 | 0.0709103145 |
| Beauty | 0.0763515009 | 0.0766019942 |

只有最终全量 validation 才与历史全量指标比较。筛选负结果限定为“当前配置与预算未发现足够收益”，不直接否定整篇论文或整个架构。

## 4. 入口与运行状态

三条新接续配置：

- `experiment/phase18/config/s18_full_r2_diff_gram_toys.json`
- `experiment/phase18/config/s18_full_r2_diff_gram_beauty.json`
- `experiment/phase18/config/s18_full_r2_diffgrm_toys.json`

Runner：`experiment/phase18/protocol/s18_full_budget_resume.py`；启动器：`experiment/phase18/run_stage18_full_budget_resume.sh`。完整 LR 过渡由 `experiment/phase18/core/full_budget.py` 实现。配置中的 `additional_epochs` 是接管前上界，接管后 `resolved_config.json` 写入由真实 source epoch 得到的剩余轮数。

继续使用原四个 `status.json` 路径；每个实验的新产物位于其 `full_r2/`。等待期间 `status.json` 仍如实显示当前 screen_r1 训练，`pending_revision.json` 显示 `WAITING_FOR_EPOCH_CHECKPOINT`、目标 epoch 和活跃协调器 PID；接管后主状态变成 full_r2，pending 标为 APPLIED。不要把仍保留的旧 result.json 当作本次结果，以主状态的 result_path 为准。

2026-09-08 16:15:56–58 已启动三个协调器：

| tmux session | PID | GPU |
| --- | --- | --- |
| s18_full_r2_diff_gram_toys | 4047887 | 3，GPU-7952ad6e-2db7-fb2e-581b-34be56287640 |
| s18_full_r2_diff_gram_beauty | 4047906 | 7，GPU-7876ca7a-7bb3-eb10-f22b-2ec3b17b9253 |
| s18_full_r2_diffgrm_toys | 4047908 | 4，GPU-4cc17bbe-a85f-bdab-44ef-bbe29d1afc09 |

DiffGRM Beauty 原 session `s18_full_r2_diffgrm_beauty`、PID 3647654、GPU 1 不变。

## 5. 校验

原恢复与筛选测试加新长预算测试共 12 项，在 native 环境通过；3 项新测试也在实际 GRAM 环境通过。覆盖恢复时不改 Adam / 首步 LR、桥接后接入原曲线、旧短档终点 LR 仍为正、DIFF 冻结阶段偏移、尾 batch 的有效更新计数、按正确联合 epoch 早停、起始最佳模型保留和仅一次最终全量验证。另已检查 Python 编译、shell 语法和 diff 格式。

运行时另存 `handoff.json`、`restore_check.json`、`schedule_restore_check.json`；只有 checkpoint 还原后一致生成、有限梯度、零额外探测更新等校验通过才开始训练。完整启动核验记录于 `artifacts/phase18/full_r2_four_run_verification.json`，其中将明确区分已接管训练与等待本轮完成的自动接续。

16:21 启动核验：两条 Toys 已完成接管，旧训练 PID 3105889 / 3106135 已退出，新的 PID 4047887 / 4047908 位于原 GPU 3 / 4。DIFF→GRAM Toys 保留 epoch 2 / 1,710 updates，DiffGRM Toys 保留 epoch 19 / 2,033 updates；两条真实 source checkpoint 与 restored_start 的全部模型张量、Adam 状态、CPU / CUDA RNG 逐项完全相同，source_restore_identity.json 均 PASSED。DiffGRM Toys 已完成初始固定子集验证并进入 epoch 20，确认真实新增 23 次 optimizer 更新。DIFF→GRAM Toys 正在起点固定子集验证。

DIFF→GRAM Beauty 仍由 PID 3105925 训练绝对 epoch 3，PID 4047906 等待本轮 checkpoint 后自动接管，目前未创建第二份 GPU 模型；不需要手动续跑。DiffGRM Beauty 仍为 PID 3647654，原完整日程持续训练。宿主 GPU / 进程快照：artifacts/phase18/full_r2_host_verification.json。
