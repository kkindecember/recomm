# GRAM 第20阶段实验进展报告

建立日期：2026-09-10。SPRec正式后台队列已于北京时间21:57启动，S-DPO并行分支于22:43启动。

**2026-09-11执行更新：S-DPO已按用户要求于14:08安全停止；round1_sft全量验证已在GPU0启动。GACR-v6/PCRF首次GPU5尝试因吞吐和共享显存变化保留收尾记录，当前批量特征版通过16人一致性检查后改在GPU6正式运行；SPRec继续GPU7。后续实时进度与终态结论统一见[第20阶段实验总报告](GRAM_第二十阶段_实验总报告.md)，并自动同步各专项报告。下方13:49审计是停止前快照。**

**2026-09-11 13:49更新：两条Toys任务均进入第3轮，已有前两轮3000人选择validation结果，尚无新方法全量validation终态结果。SPRec两个DPO轮的raw N10相对同精度父模型为-1.66%/-6.12%；S-DPO两个epoch为-46.29%/-86.38%。本次建议停止当前S-DPO配置、SPRec完成Toys有限预算收尾；尚未执行停止或追加GPU训练。详细逐用户复算、预计剩余时间和补充实验建议见[9月11日中期审计](第20阶段_20260911中期审计与继续停止建议.md)。下文保留9月10日的启动记录和当时估计。**

## 已完成工作

- 汇总第1—19阶段核心证据，整理230份计划/报告的路径与哈希；见[全项目复盘](全项目实验复盘与第20阶段选型报告.md)。
- 核对SPRec、S-DPO、UGR、MiniOneRec及HSTU的主要来源；选择SPRec流程作为首条实际运行主线。
- 实现全参数GRAM的完整响应DPO、自生成训练负例、逐轮SFT/DPO、checkpoint选择、全量validation及冻结PCRF评估。
- 最终完成5项数学与梯度单元测试。初次GPU smoke因共享位置嵌入的参数命名检查不正确而退出，未启动正式训练；修复为直接检查共享参数对象，失败attempt保留。
- 第二次GPU检查通过完整梯度和原模型raw rank检查，但发现新推理和历史候选排序/分数不完全相同，造成16人中一人的PCRF深截断rank差1。正式训练尚未启动；据此增加历史候选上的PCRF适配器检查、TF32诊断，并固定每域先补一次同精度全量原模型推理参考。

## 实验定义

这是SPRec思路向原GRAM的独立迁移，不是作者Llama模型复现。三轮全量训练，Toys先行；符合总计划的正点估计条件时，同配置自动执行Beauty。完整协议见[总计划](../../plan/第二十阶段/GRAM_第二十阶段_提点实验总计划.md)，配置见[Toys](../../experiment/phase20/sprec_toys_v1.json)与[Beauty](../../experiment/phase20/sprec_beauty_v1.json)。

原GRAM为主比较，原GRAM+PCRF为辅助强参照。现阶段不能声称DPO提点；GPU smoke、loss下降或检查通过均不作为推荐指标增益。

## 运行与证据入口

- 两条实验的统一总状态：[overview/status.json](../../artifacts/phase20/overview/status.json)，每30秒刷新；`all_finished=true`表示两条都已结束，`all_succeeded=true`表示都正常执行完成，均不能据此声称提点。
- SPRec队列状态：`artifacts/phase20/status.json`。
- S-DPO分支状态：`artifacts/phase20/sdpo_gram_toys_v1/status.json`。
- Toys正式目录：`artifacts/phase20/sprec_gram_toys_v1/`。
- Beauty正式目录：`artifacts/phase20/sprec_gram_beauty_v1/`，仅在Toys完成并触发固定规则后创建。
- 子阶段训练和负例生成过程写入`events.jsonl`；checkpoint选择写入`selection.json`；终态指标与决策写入`summary.json`。
- 各次GPU工程检查保留在独立`smoke`目录；最终通过的是`sprec_gram_toys_smoke_v4`。

## 正式启动及检查结果

后台会话：`s20_sprec_gram_v1`；GPU7，UUID `GPU-7876ca7a-7bb3-eb10-f22b-2ec3b17b9253`。启动时监督PID为2175668，Toys子进程PID为2175678；PID只表示本次启动记录，后续以有时间戳的状态文件和实际进程为准。

首个正式步骤是使用未修改父权重生成同精度全量validation参考，随后按总计划自动完成三轮负例生成、SFT与DPO。当前没有DPO正式增益可报告。Toys触发固定正信号条件后，Beauty先做自身工程检查再执行同样日程。

最终GPU检查在GPU7完成，耗时约48秒：

- 60,517,376个参数全部可训练；encoder、decoder、共享词嵌入和位置嵌入梯度有限且非零，reference没有梯度。
- 策略与reference相同时，DPO margin最大绝对值为0，loss为0.693147182，符合log2。
- 最长历史、microbatch4的DPO训练检查峰值allocated约3,224.89MiB、reserved约3,670MiB；这是该检查的峰值，不代表完整运行的最大显存。
- 历史候选上的PCRF逐用户rank检查64/64精确一致；TF32打开后，16/16组候选顺序恢复到历史结果。FP32/TF32关闭用于正式新模型及重新推理的父参考，避免把数值差异计入方法增益。
- 8条训练样本完成真实负例生成、SFT和DPO执行路径，独立smoke产物不进入正式选择。
- 已修复批量采样中部分序列提前EOS后，空trie分支导致概率非有限的问题，并增加对应单元测试。

正式准入采用至少12GiB可用显存，allocator上限为整卡40%；不使用holder或旧30GiB占位规则。核对记录见[工程验证汇总](../../artifacts/phase20/review_20260910/engineering_validation.json)。

## S-DPO并行启动与等待时间更新

用户希望等待期间同时探索其他方法，因此将候选S-DPO提前启动。该分支在GPU0执行，SPRec继续在GPU7执行。S-DPO从同一原GRAM父模型开始，使用四个高排名错误商品和冻结父模型reference，全参数训练三个epoch；复用同精度父模型评估参考。它与SPRec在负例、训练日程和reference设置上都有差异，比较不能单独归因于负例数量。完整设置见[并行计划](../../plan/第二十阶段/S-DPO并行实验计划.md)。

四项单元测试和GPU工程检查通过，8条训练数据完成真实负例缓存及两个optimizer更新；正式任务已进入109,361条训练样本的负例生成。检查结果及源文件哈希见[工程记录](../../artifacts/phase20/review_20260910/sdpo_engineering_validation.json)。后台会话`s20_sdpo_gram_toys_v1`，启动PID2408042；首条正式负例生成事件时间22:44:18。

22:47快照：SPRec父模型评估5,538/19,412，S-DPO负例生成1,476/109,361。近期速度外推各自当前步骤约剩7.65小时、3.96小时，均不包括后续训练及验证。此前按启动速度给出的18–30小时和10–20小时估计偏乐观；按目前吞吐波动，SPRec Toys预留1–3天，条件Beauty触发后整个队列预留2–6天，S-DPO Toys预留1–2天。尚无完整epoch计时，硬预算可能先于全部实验完成。查看说明见[artifacts入口](../../artifacts/phase20/README.md)。

## 结果解释边界

两条方法的论文正证据、反例及本地迁移差异已补充于[提点依据复核](SPRec与S-DPO提点依据复核.md)。特别是SPRec兼顾去偏目标，S-DPO原论文采用候选选择评测且主实验beta与当前适配值不同；不能将论文结果直接当作本项目提点证据。

正常validation已参与过许多历史实验。本轮是寻找有效方法的探索，选中模型的validation配对区间是描述性分析，不是独立确认。一个seed、一个数据集的正点估计只构成继续跨域检查的信号。
