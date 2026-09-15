# Stage20 Toys完整列表内部蒸馏进展与结果

**2026-09-15分卡调度：** 当前状态FAILED；三臂优先GPU4/1/6，仅在空闲显存≥30,000 MiB时分配不同A6000，实际分配见status。每臂训练后在原卡验证，全部完成后汇总。原缓存继续计算，训练协议、步数、输入和模型代码保持冻结。

三卡可用且吞吐接近profile时，估计剩余7.04小时；不含等卡时间，共享负载可能使估算变化。

## Material Passport

- Origin Skill：academic-research-suite / experiment-agent；Origin Mode：run / validate。
- 更新时间：2026-09-15T00:57:41+0800；状态：**FAILED**；test未读取。

观察文件：`artifacts/phase20/gacr_list_distill_toys_gpu1_v2/status.json`。30秒心跳，`child_status`含当前阶段、进度与阶段ETA，`projected_remaining_seconds`为profile外推剩余时间；共享GPU负载可能使估算变化。

## 实验问题与固定协议

检验已有效的B1+PCRF商品排序能否通过完整50候选列表蒸馏，进入C1解码器并改善重新生成的候选和排序。此次确实更新生成器；编码器、共享词嵌入、LM head、B1和CF head保持冻结。

Toys先做三个匹配臂：CE_anchor（CE+父C1列表KL保持）、true_teacher（再加真实B1+PCRF分布KL）、shuffled_teacher（加每个前缀固定打乱的教师分布）。三臂相同C1起点、数据、顺序、优化器和预算，不按中途验证结果筛选。

21,839个fit前缀/3,882个用户，12,399个目标在候选50内、9,440个在外；源缓存另256个calibration前缀不训练。每臂1个epoch、有效batch128、171步，AdamW lr=1e-06，5% warmup，线性衰减，clip1；仅固定末步checkpoint。

CE覆盖全部fit前缀，父C1完整列表KL覆盖全部fit前缀；教师KL只用于候选内目标的前缀，按21839/12399归一。权重CE=1、父KL=1、教师KL=0.1，温度1。教师分布来自B1 seed2023的最终PCRF分数标准化后softmax，打乱保持概率集合和熵。候选不强行加入真值。

完整商品分数为含EOS、不含BOS的token对数概率之和除以标签长度。全50候选统一标准化及KL；先计算50个分数的梯度，再每4条候选重放反传，CPU测试验证与完整图梯度一致。FP32骨干、FP64列表概率/损失，TF32和dropout关闭。父保持使用同接口实际teacher-forced分数缓存，记录其与native beam分数的数值差异。

真实GPU profile检查原C1新生成候选一致性、分数、完整列表重放梯度、冻结参数与一次丢弃的工程更新，并测量显存和各阶段耗时；不产生正式训练步。共用父保持缓存完成后，三臂分卡各自训练、完成后各完整生成19,412名验证用户beam50并加固定PCRF、汇总匹配对照。

## 进度与资源

| 任务 | 状态 | 阶段 |
|---|---|---|
| profile | COMPLETED | finished |
| cache | COMPLETED | finished |
| CE_anchor_train | STOPPED | decoder_training |
| true_teacher_train | FAILED | initialization |
| shuffled_teacher_train | PENDING |  |
| CE_anchor_evaluate | PENDING |  |
| true_teacher_evaluate | PENDING |  |
| shuffled_teacher_evaluate | PENDING |  |
| aggregate | PENDING |  |

GPU：`CE_anchor=GPU1, true_teacher=GPU4`；正式解码器优化步：1/513。无自动重试；技术错误、超时或资源等待超时会停止队列，保存已有日志。

实测profile状态：PASSED；GPU峰值已分配 6877 MiB，保留 19096 MiB。

外推：父列表缓存 1.74小时；三臂训练合计 13.93小时；每臂完整验证 2.47小时；原单卡串行工作量约 23.08小时。这是8个历史长度分位样本外推，非完成承诺。

## 判断与边界

结果尚未完成时不判断有效。需要同时看真教师与CE/打乱对照、当前冻结B1+PCRF强基线的比较；raw结果判断内部迁移程度，重新生成后的H50及低频H50判断候选覆盖。只改变候选、只优于弱父模型、或只靠验证选择最优超参数，都不足以支持达到当前提点目标。

所有八指标、五分组、N10/H10的3000次用户配对bootstrap与H5/10/20/50 gained/lost保存在各臂summary和最终summary中。单次训练、重复使用验证集、未校正多重比较；用户区间不包含训练随机性和历次开发选择的不确定性。B1教师沿用已观察seed2023，不挑最佳初始化。

此次训练仍限于既有fit用户/前缀，教师只覆盖12,399个候选内样本，未彻底消除旧内部迁移实验的分布差异。与旧PCPS的区别是完整实际候选分布、跨父前缀竞争、父保持和参数范围；旧PCPS已有部分完整序列损失，不能把“完整序列训练”本身声称为首次创新。打乱检验完整教师的商品对应关系，不能单独归因CF。

[预登记方案](../../plan/第二十阶段/Toys完整候选列表内部蒸馏实验计划.md)；[B1多初始化结果](Stage20_B1缓存复用多初始化验证进展与结果.md)。
