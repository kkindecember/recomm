# 固定案例评分数值复核：执行补遗 v0.1

2026-09-08，研究者“继续”授权实施 [9 案例设计](GRAM_真实Beam诊断后_数值评分复核与下一步决策v0.1.md)。
本补遗在 GPU 输出前冻结比较条件。输入沿用已冻结 case_manifest，不增加案例。

## 固定计算矩阵

每案例先精确重放旧设置的 native beam50：临时只读 forward hooks 保存 encoder 输出与目标/边界祖先所在行的
logits/log-softmax。移除 hooks 后退出。新 trace 必须与旧 trace（至丢失深度及最终序列）完全一致。
从捕获的 full-vocabulary token log-probability 顺序相加必须精确恢复旧 native 累计分数。
这一次重放用于核对原始记录；后续矩阵不执行 topk 或重选 frontier。

分别在 TF32 开启/关闭下，固定以下六个条件（共 12 条件加 1 个 native 捕获参照）：

| 名称 | encoder | decoder |
|---|---|---|
| frontier_cached | 同一单输入 hidden | 原 trace 的 50 路父索引，逐步缓存并按记录重排 |
| frontier_full | 同一单输入 hidden | 同一 50 路前缀，每步完整解码，无缓存 |
| pair_full_shared | 同一单输入 hidden 复制 | target/boundary 两路完整前缀，无缓存 |
| pair_full_duplicate_encoder | 两份相同输入重新编码 | target/boundary 两路完整前缀，无缓存 |
| pair_cached | 同一单输入 hidden 复制 | target/boundary 两路增量缓存 |
| single_full | 同一单输入 hidden | 两条前缀依次以单路完整解码 |

关闭 TF32 时重新计算相同输入的 encoder hidden；不同精度下路径、frontier、权重不变。
两种 TF32 状态下，分离 encoder 的 pair_full_duplicate_encoder 必须与旧 prefix_scores 函数重新计算值完全一致。
旧历史 teacher-forcing 值是否精确复现另报，不假定历史 worker 的精度状态已被直接记录。
开启 TF32 的 frontier_cached 与 native 捕获的全词表 logits/log-probability 比较，概率要求完全一致。

记录全词表 logits/log-softmax 最大差、选中 token 的逐步分数、顺序 float32 累加、
float32 reduction 与 float64 累加的差；不把 float64 累加称为模型完整双精度推理。
不对 Trie 的合法集合重新归一化，不添加 EOS，不做长度平均。
结果报告残差及减少比例，不设置事后 epsilon，不自动把旧梯度状态改成通过。

## 工程合同与运行边界

CPU 小模型测试缓存重排、完整/增量/单路/双路计算、初始无效 beam、全词表归一化与 TF32 开关异常恢复。
测试中的小模型浮点容差仅用于单元测试，不作为真实模型的科学 Gate。
每案例检查合法前缀和前后完整 state_dict SHA；model.eval + no_grad，无 optimizer/backward/new checkpoint。

固定第一个案例为 smoke；identity 和所有有限性检查通过后，记录 smoke 峰值显存并检查剩余至少 1 GiB，
继续其余 8 个案例。首次入场要求 12,000 MiB 空闲，单卡，排除 GPU1/GPU4 与已存在的预约/占位进程。
同一进程内的 smoke 分配仍驻留，因此不把已分配内存再次计入外部 free-memory 要求。

一次运行独立 tmux，10 秒 heartbeat，整体 30 分钟 hard timeout，无自动重试、结束后不占位。
全部原实验和诊断 sources/inputs SHA 验证；新源码、配置、补遗与 case_manifest 快照。
原始结果和科学 Gate 均不修改。小案例结果不能替代 100 用户平均梯度的稳定性复核。

```bash
bash experiment/phase18/run_stage18_score_numerics_audit.sh launch --run-name run-0001 --gpu 7
```

运行状态以 `artifacts/phase18/score_numerics_audit/run-0001/` 内实际产物为准。
