# 全局KL对照OOM后的Toys单域恢复

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent，内联执行
- Origin Mode: run / resource recovery；继续完成用户已授权的全局KL匹配对照
- Origin Date: 2026-10-17
- Verification Status: PLANNED；原失败证据保留
- Version Label: global_kl_control_20261017_v1_recovery1

原任务Toys在2026-10-17 18:50:36的模型加载阶段报CUDA out of memory，当时使用GPU6（GPU-e2e37406-aaaa-da68-cf74-1a487b1c93e2），尚未写预测。Beauty已在GPU7完成512用户，运行核验通过；两域CPU拟合和历史GPU重放也均通过。原根状态INCOMPLETE意味着Toys未完成，不代表两域训练失败。

确定事实：OOM栈位于model.to(cuda)，原侧车仅预留4GiB，模型工作预算没有提前保留；原日志没有加载瞬间的空闲显存数据，不能确定具体竞争进程或断言唯一根因。

## 恢复范围

保留global_kl_control_20261017_v1全部原文件。恢复到独立目录，逐字节复用两域已训练α和所有科学输入。Beauty已有512预测、smoke和runtime核验直接复用并校验；只重新生成Toys同一512，不重训、不扩样、不改λ/β/beam/历史处理，不读test。

## 资源修正

1. 只安排Toys一个GPU任务；暂避原OOM的GPU6，优先空闲显存最多的其他GPU。3次间隔5秒的快照均达到18GiB空闲才准入；租约前和侧车启动后再次记录显存。
2. 沿用16GiB总预算、4GiB侧车、11GiB模型分配上限。生成进程在加载模型之前分配并释放一个10GiB CUDA张量，保留在同进程PyTorch缓存分配器中供模型和beam复用；不调用empty_cache。这样工作内存已归该进程持有，缩小检查与实际使用之间的窗口。缓存10GiB加侧车4GiB，仍为上下文和工作开销留余量。
3. 保存保留前后设备free/total及allocated/reserved。若工作缓存预留失败，尚未加载模型且没有科学输出，按资源准入退让处理并保留该次证据；不重试模型加载/生成中的科学失败。后者仍记录并停止。
4. 预留缓存完成后重置峰值统计，工作负载峰值与预留峰值分开记录。原生、零α、固定.3、用户KL和新全局α的smoke全部重做；模型摘要和逐路径重评分不变检查保留。

这降低共享负载变化导致OOM的风险，不声称能够独占GPU或阻止所有外部资源变化。不停止任何其他任务。

## 完成与统计

Toys512补齐后，与已完成Beauty512一起调用原配对汇总；保留全部参照，不按结果改变规模。恢复目录记录Beauty预测哈希和复用来源；最终报告注明资源恢复。原失败状态不覆写，只另存恢复入口。最多等待资源24小时、单次生成30分钟；不自动扩大4096。

- 恢复状态：artifacts/phase21/global_kl_control_20261017_v1_recovery1/status.json
- 入口：experiment/phase21/global_kl_control_recovery.py
- tmux：s21_global_kl_control_20261017_v1_recovery1
