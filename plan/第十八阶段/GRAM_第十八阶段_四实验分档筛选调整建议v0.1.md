# 第十八阶段：四实验分档筛选调整建议 v0.1

日期：2026-09-08。状态：EXECUTING。用户随后明确要求“重新调整了再跑”，现已实现并启动 `screen_r1` 检查点切换。本修订覆盖四条任务的原训练上限与验证频率；模型结构和既有数据保留。

**后续已授权覆盖（16:15 起）：** 用户要求四条全部改用完整训练上限、稀疏验证和早停，已启动自动 checkpoint 接续；详见 [四实验完整预算与自动接续补遗](GRAM_第十八阶段_四实验完整预算与自动接续执行补遗v0.1.md)。本页 4 / 20 epoch 短档与“下一档 20 epoch / 2 小时”仅保留为历史讨论，不再约束本次四条已授权完整预算。

## 依据

当前完整训练方案作为第一轮架构筛选偏重。DiffGRM 的 warmup=10,000 updates，Beauty 每轮约 129 updates，Toys 约 107 updates，分别约 78 / 94 轮才结束 warmup。最低 100 轮的停止约束会锁定大量投入。

13:36 DIFF Toys 已完成冻结适配轮，开始完整 validation；前 22 用户用时 31.3 秒，初始速率约 1.4 秒 / 用户。若此速度持续，19,412 用户约需 7.7 小时。该外推样本少，后续需用稳定吞吐修正，但已足以说明先前“2–4 小时得到首份完整验证”的估计偏乐观。其余 DIFF 完整验证尚未实测，不能把训练 epoch ETA 当整体 ETA。

## 建议调整

1. 保留四个已启动模型的结构、已准备数据与编码。DIFF 仍为迁移结构；DiffGRM Beauty hidden=256、Toys hidden=1024。本轮不新增第五条模型容量实验。
2. DIFF 两域：完成已有冻结适配轮后，以 2 个 joint epoch 为一个投入档，第一档上限 4 个 joint epoch。用固定验证子集跟踪变化；在阶段末选定候选后做一次完整 validation。warmup-only checkpoint 的完整验证不作为联合训练的必经长任务。
3. DiffGRM 两域：从最近完整 epoch checkpoint 延续模型和 optimizer，第一档追加 20 个 epoch；用约 500 个新增 optimizer updates 将当前学习率平滑升至已有域配置的 peak LR，随后按该档剩余更新数 cosine decay。撤销该探索续跑的最低 100 epoch 约束。该 500 步日程是本地筛选适配，不冒称作者原始日程或已经证明最佳。
4. 两类模型的趋势检查均使用预先固定 seed / user-ID 规则选出的 2,000 个原 validation 用户。DIFF 每 2 个 joint epoch、DiffGRM 每 5 个追加 epoch 检查一次。子集仅跟踪同一模型在同一批用户上的变化与选模；不把其 NDCG 与已有全量 validation 均值直接相减。不重跑基线。
5. 第一档结束后，按趋势与完整 validation 的绝对差距决定是否追加训练。子集噪声、warmup 中的低分或一次波动不能独立判定失败。差距明显且走势停滞时降低当前配置优先级；正信号或仍明显改善时允许追加预算。该阶段结果不用于否定整篇论文或整个架构方向。

## 实施要求

- 这不是修改原配置文件后立即对运行中进程生效的改动。当前 runner 将配置加载到内存，且没有完整 resume 命令；执行前需补充从 prepared artifacts 和 checkpoint 恢复的路径。
- 保留来源代码快照、旧配置、已完成 checkpoint 与日志；保存 optimizer、实际 global step、学习率过渡起点和新日程。避免直接替换 scheduler 参数造成学习率突然跳变。
- 在完整 epoch 边界切换；没有可恢复 checkpoint 的任务先完成当前 epoch。中途生成的 validation partial 文件只保留为 partial，不改名为完整结果。
- 检查恢复前后的模型输出、optimizer 状态、下一步 LR 及梯度累积边界，再启动续跑。各档的 split / validation 范围必须明确记录。
- 继续使用用户指定的四个 status.json 路径观察，通过 status 标明当前档、配置修订、当前验证人数、是否全量、最近完整验证指标；不添加总览工具。

成本变化预期：主要减少 DIFF 重复全量验证，以及 DiffGRM 为 10,000-step warmup 等待的时间。不能承诺所有方向在几小时内均能可靠作出有效性结论。

## screen_r1 执行配置与观察

- DIFF Toys / Beauty：从冻结适配轮完整 checkpoint 继续，新增 4 个 joint epoch；第 2 / 4 轮各验证固定 2,000 用户；按该子集选中 checkpoint 后执行一次完整 validation。原 warmup-only validation 的未完成文件保留为 partial。
- DiffGRM Toys / Beauty：在切换请求时所在 epoch 保存后继续，新增 20 个 epoch；新增 500 updates 内从 checkpoint 中的 LR 平滑升至原域 peak，随后 cosine decay 至该档结束；新增第 5 / 10 / 15 / 20 轮做同一固定子集，最后对选中模型做一次完整 validation。
- 选择子集按 `sha256(seed|dataset|user_id)` 排序，不读目标或模型分数；同一域两个模型选择相同用户。子集结果标记 `trend_subset`、`delta=null`；只有 `full_validation` 才与已有全量基线计算差值。
- 恢复 runner：`experiment/phase18/protocol/s18_screen_resume.py`；启动脚本：`experiment/phase18/run_stage18_screen_resume.sh`；四个配置：`experiment/phase18/config/s18_screen_r1_*.json`。
- GRAM 与 native 两个实际环境分别通过 5 项 CPU 检查，包括 Adam 保存恢复后下一步与未中断更新完全一致、学习率衔接、最后不足一个 batch 的更新、子集/全量指标边界。记录：`artifacts/phase18/screen_r1_cpu_checks.json`。
- 每条任务先保存 handoff 记录，在完整 checkpoint 存在后对已核对身份的旧训练 PID 发 SIGINT，等待其退出，再启动恢复。此时旧 runner 写出的中断异常归档为用户授权的预算切换，不解释成科学失败。
- 原 checkpoint、配置和源码快照保留；新产物位于原 run 根目录的 `screen_r1/`，新 console 为根目录 `screen_r1.console.log`。每条任务的 GPU 恢复检查单独保存为 `screen_r1/restore_check.json`。
- 原四个 `status.json` 路径保持不变。切换等待期间有 `pending_revision`；生效后有 `revision=screen_r1`、`inherited_epoch`、`inherited_optimizer_steps`、`stage_epoch`、`stage_epochs`、`stage_optimizer_steps`。`epoch` / `optimizer_steps` 仍包含继承部分；`stage_*` 仅表示本次追加训练。
- 验证状态有 `validation_scope=trend_subset/full_validation`，`validation_total` 分别为 2,000 / 原全部验证用户。`COMPLETED` 表示本次筛选档训练及最后一次完整 validation 完成；`direction_rejected=false`，不会自动宣布方向失败，也不会自动追加另一档。
- 两条旧 GRAM checkpoint 未保存 RNG，本次恢复保留权重与 optimizer，随机流按固定 seed 初始化新阶段；两条 native checkpoint 的 CPU / CUDA RNG 会恢复。不能把新日程称作旧训练的逐位无中断重现。

## 切换完成记录

2026-09-08 14:40 左右，四条均已进入 `revision=screen_r1` 并发生真实新增更新：DIFF Toys 继承 epoch 1 / 855 steps，DIFF Beauty 继承 epoch 1 / 1,027 steps；DiffGRM Toys 继承 epoch 6 / 642 steps，DiffGRM Beauty 继承 epoch 33 / 4,257 steps。检查时新增步数分别已达 66、102、13；DiffGRM Beauty 已完成新增第 5 轮并进入 2,000 用户趋势验证。

四条 `restore_check.json` 全部 PASSED：checkpoint 生成结果保存恢复一致、Adam 状态及起始 LR 保留、恢复后训练 loss / 梯度有限，检查过程未推进 optimizer。Toys 两模型和 Beauty 两模型的 2,000 用户列表分别完全一致。验证记录：`artifacts/phase18/screen_r1_switch_verification.json`。

具体学习率起点：DIFF 两域的原冻结轮 checkpoint 已走到 LR=0，新档以总新增更新数的 5% 为上升段，然后使用 cosine decay；峰值仍为新增分支 0.001 / GRAM 0.00001。DiffGRM Toys 从 0.0001926 平滑升至 0.003，Beauty 从 0.004257 升至 0.01，两者过渡均为新增 500 steps。所有实际日程参数保存在各自 `screen_r1/schedule.json`。

四条状态文件保持原路径。原配置、旧 checkpoint、已完成验证和未完成 partial 文件均保留；本次完整训练 / 验证尚未结束，切换成功不等于模型已提点。

## 最新补充：方向发现优先与已有预测复用（2026-09-08）

用户再次明确：当前是寻找能提点的新方法组件，允许架构大改；最终方法形成后全部正式实验会重跑。当前目标是缩短方向筛选的周转时间。

### 已实际完成的改进

发现第九阶段 seed2023 的 validation `per_user.tsv` 已同时保存 GRAM 和 GRAM+PCRF 的逐用户 rank。两域文件哈希与旧 summary 一致，CPU 聚合重现第一、第二阶段的全量 GRAM 指标；当前 2,000 用户 cohort 可完整对齐。新增 [CPU 比较脚本](../../experiment/phase18/analysis/screen_saved_reference.py)，只读取已有完整候选预测和历史 rank，核对用户、当前目标、历史长度、记录的划分来源与指标。耗时约 2.5 秒，GPU 训练和模型推理均为 0。

| 同一固定 2,000 用户 | Toys NDCG@10 | Beauty NDCG@10 |
| --- | ---: | ---: |
| 已有 GRAM | 0.0693705651 | 0.0763515009 |
| 已有 GRAM + PCRF | 0.0709103145 | 0.0766019942 |

因此，旧规则“子集不能与全量基线相减”保持；新增的是**子集与历史同用户子集可以直接比较**。不需要等完整 validation 才知道候选离强基线有多远。旧 runner 的 `delta=null` 仍按其原含义保留；比较输出单独落盘，不改写运行中的结果。

首次结果：[review_20260908.json](../../artifacts/phase18/screen_reference_reuse/review_20260908.json)。以后出现新的完整检查点预测时，运行下列命令并使用新的输出文件名；脚本拒绝覆盖已有快照，忽略 partial 和尚未提交 summary 的预测。

```bash
python3 experiment/phase18/analysis/screen_saved_reference.py --output artifacts/phase18/screen_reference_reuse/review_next.json
```

### 当前四任务如何处理

1. 维持已经成功恢复的 `screen_r1`，从现有 checkpoint 获取这一档结果；本轮没有中断、重启或新增 GPU 任务。两条 DIFF 尚在第一个 joint epoch，不能从冻结适配结果判定结构无效。
2. 每个完整 2,000 用户检查点一旦产生，就用已有 rank 比较 NDCG@10 / Hit@10，立即形成投入判断，不等四条任务全部结束才看结果。
3. 当前 runner 在档末**仍会无条件执行一次全量 validation**，并且不会自动追加下一档。CPU 比较功能没有改变这个行为；后续“按信号决定是否全量验证”的规则尚未接入运行进程，不能宣称已经生效。
4. DiffGRM Beauty 子集从新增第 5 轮 0.03205 涨到第 10 轮 0.03740，但同用户 GRAM 为 0.07635；目前是学习中的明显落后，不是提点信号。先完成已启动的这一档，再依据后续曲线决定是否给一次有限延长，不能因 loss 下降无限加预算。

### 后续新候选与下一档的默认投入规则

- **按机制筛选，不按论文数量铺任务。** 最多保留两个不同机制的活跃方向。每个新方向先单域、单 seed，选来源配置和本地接入最完整的域；有信号再扩第二域。当前四任务是已授权的存量，不据此启动更多双域矩阵。
- **首轮默认配置 + 一次有理由的适配。** 先保留来源核心计算；允许合理改 encoder / decoder / identifier / 训练范式。只检查数据边界、可训练、可生成与实际指标正确；小样本 smoke 不承担方向否定。工程检查以 15–30 分钟为目标，超时先定位具体接口问题，不自动扩成多日诊断计划。
- **同时写 GPU 小时和训练量。** 下一个新方向目标是在 1–2 GPU 小时内得到第一份同用户指标；预计超过时，先根据实测吞吐选择合适的首档训练量与单域配置。这个时间目标是资源优先级，不是所有模型两小时必须收敛的科学要求。不得削掉核心结构后仍称完整方法失败。
- **全量验证按用途安排。** 后续 runner 应在子集指标可直接对比的前提下，只给正向、接近基线或需要解决子集不确定性的候选做一次全量验证；明显落后且走势已停滞的配置可直接保存结果并让出资源。停止的是当前配置的投入，不是否定整个论文方向。这项控制需要在下一次 runner 修订中实现。
- **追加预算有上限。** 当前配置只有正向、接近基线仍上升，或差距持续缩小且后续训练成本低，才考虑一次追加。DiffGRM 最多再 20 epoch / 2 GPU 小时，DIFF 最多再 2 joint epoch / 4 GPU 小时，二者均取先到者；必须在启动前记录具体数值和下一次读数点。预算末仍没有可用信号，归档为当前预算下无收益或未收敛并换方向，不串行申请无数“再一档”。这些是后续默认资源上限，不是已加载到当前进程的超时参数。
- **正信号先保留，再提高要求。** 同用户 NDCG@10 正向且 Hit@10 无明显代价，即可进入候选保留；相对提升约 3% 及以上优先，但 3% 仅是排序偏好，不是显著性门槛，也不因一次小屏波动淘汰较小正向。边际结果可扩大候选验证人数，旧基线继续从缓存取相同用户。
- **PCRF 和正式对照后置。** 先回答新候选自身能否超过 GRAM；有希望再用候选自己的输出检查冻结 PCRF 组合。当前不重训 GRAM、PCRF、匹配续训对照，不铺多 seed / 全消融，不打开 test。最终方法确定后再统一重跑并归因。

DIFF 与 DiffGRM 继续是当前两个来源不同的方向，作者仓库分别确认 [SIGIR 2025 DIFF](https://github.com/HyeYoung1218/DIFF) 和 [WWW 2026 DiffGRM](https://github.com/liuzhao09/DiffGRM)。若二者本轮均没有可用信号，下一次选型应考察与其不同的机制，例如任务与 identifier 联合学习。不能直接把 DIGER 放成必跑首选：作者当前明确提示历史 main 截断了推荐梯度，新实验推荐 gradient-fix；修正路径的收益需另行验证。[作者说明](https://github.com/junchen-fu/DIGER)。PCPS、已完成负向迁移的 LATTE / SETRec 暂不重启；旧 cold 场景的正负结果不能冒充当前普通推荐场景的收益。
