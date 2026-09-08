# 真实 Beam 诊断后的数值评分复核与下一步决策

## Material Passport

- 日期：2026-09-08。
- 状态：`DESIGN_ONLY_NOT_EXECUTED`。本轮完成 CPU 配对统计与工程案例冻结，尚未实现或执行新的 GPU 复核。
- 依据：研究者“那接下来该怎么做”；[已完成的诊断](../../artifacts/phase18/beam_boundary_diagnostic/run-0001/summary.json)。
- 定位：已有测量的工程复核，不是新 treatment、S18-2 重试或晋级实验。

## 当前结果支持什么

Toys evaluation 的 100 个输入中有 85 次首次非 EOS 丢失；这 85 次目标局部 rank 均不超过 50，
且保留边界均来自其他父分支。旧 final-sibling proxy 仅覆盖 126/4250 个保留竞争前缀（2.96%）。
这是重新考虑跨父累计分数竞争的依据；该集合覆盖率不能解释为 loss 对全部竞争者的因果作用比例。

探索性平均方向：CE=0.112307，generic=0.380507，真实 CF=0.379940，shuffled=0.370590。
真实 CF 减 generic 的逐输入平均差为 -0.000567，85 对中 39 对为正、46 对为负。
当前没有可据以扩大 CF 训练的明确增量证据，也不能据此证明二者等效。
CE 的平均方向为正，不能把此次结果写成“CE/PCPS 局部方向共同有害”；
它也不能解释或推翻此前实际 continuation 的性能下降。

三个输入组的 native/teacher-forcing margin 符号不一致数均为 0；累计分数仍存在非零残差，
最大分别为 Toys training 0.004271、Toys evaluation 0.003366、Beauty training 0.008982。
Toys evaluation 没有按原规则标记的不确定 margin，另外两组各 1 个。
这些结果支持原生剪枝描述；分数符号一致并不足以验证梯度的数值稳定性。
原状态 `MEASUREMENT_UNRESOLVED` 保持不变。

## 立即下一步：最多 9 个案例的评分工程复核

固定案例见 [case_manifest](../../artifacts/phase18/beam_boundary_followup_design/case_manifest.json)。
从三个已消费输入组各取：首个可比事件、残差最大事件、绝对 native margin 最小事件，按用户去重。
这种针对误差的选择只适用于工程排错，不能用来估计新方法收益或补充科学样本量。

本机固定环境为 PyTorch 1.11.0+cu113，新进程查询的 matmul/cudnn allow_tf32 均为 true。
原 runtime 的 set_seed 未修改这两个开关。历史 worker 未显式记录开关，因此这里是可复现的默认状态检查，
并非对历史进程精度状态的直接测量。
代码中 native generation 使用单输入 encoder、50 路 decoder 和增量缓存；
可微 prefix scorer 使用两份输入 encoder、2 路 decoder 和完整前缀计算。
TF32、批次形状和缓存是待验证的误差来源，当前不宣称已找到原因。

实施时固定权重、输入、已记录的 target/boundary 前缀和原生 frontier。按以下顺序逐项比较：

1. 重放原设置，核对旧 trace 的逐 token 分数、累计相加和长度/EOS 处理。
2. teacher forcing 复用同一份 encoder hidden state，分离 encoder 重复计算的影响。
3. 在同一 encoder 输出下对齐 decoder 批次/前缀，比较缓存增量计算与完整前缀计算。
4. 对上述配对计算关闭 TF32，再比较误差来源和数量级；不得改写旧实验配置或结果。
5. 分别记录 logits、log-softmax、累计相加的差异；必要时只对选中的小案例使用更高精度参考，
   不能假定模型内部 float32 softmax/投影在 model.double 后自动成为完整 float64 参考。

此次只定位数值差异，无 optimizer、无新 checkpoint、不重算 1000 用户训练，不增加 alpha 候选。
先做 1 例 smoke，再做剩余固定案例。单卡排除 GPU1/GPU4 与预约/占位，最长 30 分钟；
超过 10 分钟采用具名 tmux、heartbeat 和 hard timeout，不自动重试。
实现完成后、运行前冻结具体比较条件、源码和输入 SHA；本文件不提供尚未实现的启动命令。

## 明确的结束条件与后续分支

- 若发现 token、mask、cache 索引或累计公式不一致：修复新的测量代码并通过 identity 测试，
  保留原始结果。受影响的梯度结果只能以独立 correction 记录复核，不能直接改成通过。
- 若可控比较支持数值计算路径解释：记录解释与误差范围；小样本分数复核本身不能让全部梯度晋级。
  若后续论证需要梯度结论，应另行冻结对原 100 用户梯度/85 个 margin 的数值稳定性复核，
  检查方向、配对差和不确定事件，而非仅凭“非零残差很小”判断可靠。
- 若预算内无法解释：保留 MEASUREMENT_UNRESOLVED，停止扩展梯度论证，保留可信的原生剪枝描述。
- 当前暂缓扩大 CF 方向。若后续可信结果仍不显示 CF 对 generic 的额外信息，
  可结束该 CF 设计，单独评估 generic 跨父前缀目标是否值得研究。
- 新训练前须另立 continuation 协议，保留 frozen parent 与 matched CE 基线，
  每次只改变一个因素；比较 batch 时必须明确固定样本遍历量还是更新次数，
  两者不可同时暗示已匹配。训练样本分布、batch、优化器重置均为候选原因，尚无因果结论。

本轮工程复核是明确的下一步；不立即进入新 loss 训练，也不把恢复 parent 表现算作新方法成功。
