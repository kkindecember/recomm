# 真实 Beam 边界诊断：执行补遗 v0.1

2026-09-08，研究者“那继续吧”授权实施既定诊断。原设计见
[诊断计划](GRAM_第十八阶段_后续研究_真实Beam边界诊断计划v0.1.md)。
该计划的 NOT_EXECUTED 是设计时状态；执行情况由新目录的 status/summary 与本轮报告记录。
S18-2 不重跑、不改判；本轮没有优化器、参数更新、新 checkpoint 或自动训练晋级。

## 执行前固定的实现细节

- 原 cohort 顺序前 100 用户/域，训练输入；仅 Toys 增加同 100 用户已消费 I0 下一 offset。
- 模型/数据/缓存/S18-2 原运行源码 SHA 逐一核验。新源码、配置、固定 Transformers
  generation 实现与执行补遗在每次 smoke/正式运行前快照。正式运行必须匹配通过 smoke 的源码。
- beam=50，原始 Trie/EOS=1、length penalty=1、cache 设置与原 helper 一致。
  追踪器追加在原生 Trie logits processor 后，原样返回张量；临时 hook scorer process/finalize，退出时恢复。
- 真实边界定义为原生 scorer 当步最后保留的有效 non-EOS 前缀；不足 50 个有效保留项时不定义边界。
  EOS offered 与当前 bounded hypothesis heap 分开记录，避免把入队尝试当作最终保留。
  初始仅第一条 beam 有效，逻辑无效性沿父索引传播，不靠任意负分数阈值推断。
- sibling proxy 使用原缓存 beam50 完成 item 中共享目标父前缀的错误 child，prefix 去重后与真实保留前缀相交。
  原 top8 complete-path 前缀与三种 prefix loss 的 mined child 另报覆盖。
- 首次 drop 以 scorer 实际 non-EOS retention 判定；same-score 个数含目标自身，严格排名不随意打破并列。
  报告全部 100 用户分母及候选不足、EOS、并列事件。有效事件不足 30 不补样本。
- 原生 train dataset 的最后一个可见 transition 与 S18-2 单样本 collator 所有字段逐项完全比较。
  两者都使用原 I0 materialized 数据；不同的是训练分布、batch 和优化器状态。
- 四目标各自平均 100 个 training transition 梯度，alpha 固定 0.1；参数保持 eval 状态。
  梯度按 model.parameters 顺序、CPU float32 保存于内存，记录向量 SHA/范数；点积使用 CPU float64 分块归约。
  不落模型或梯度 checkpoint。辅助增量为 total gradient - CE gradient，已包含 alpha。
- 可微 margin 为同深度 target 与 frozen native boundary 的 full-vocabulary 累计 log probability 差，不做长度平均。
  两条前缀以相同输入成对 teacher forcing；不附加 EOS。与 native 原始累计分数分别报告残差。
- 本次不设任意残差 epsilon。非零残差未经独立解释，梯度总状态为 MEASUREMENT_UNRESOLVED。
  仍可保存有限的探索性方向统计，但不据此作训练决策。原生 margin 绝对值小于等于两侧绝对残差之和、
  两者符号不同、或 native 目标存在并列时，单列不确定。NaN/Inf 立即停止，记录失败与测量未决。
- Toys eval 报告 -dot(grad_margin, mean_training_grad) 及除以 training gradient norm 的值。
  这是 SGD 一阶局部方向；不声称代表 AdamW 或有限步更新的准确率效果。

## 验证与资源

CPU synthetic 验证局部第 1/单 child 被跨父挤出、并列、EOS、变长路径、初始无效 beam、异常后 hook 恢复。
真实 smoke 使用每域固定训练 cohort 中 encoded batch 最大的样本（同尺寸按 cohort 顺序），
比较有/无 tracer 与原生 helper 的全部 beam50 序列顺序和分数，并核验 alpha-zero 参数梯度及前后权重 SHA。
再运行四目标 backward 和成对 margin backward，测量峰值；全部输入的 native batch identity 一并检查。

单 GPU，排除 1/4 和占位/预约进程。smoke 10 分钟 timeout；正式运行独立 tmux，30 秒 heartbeat，
2 小时 hard timeout，结束退出，不占位、不自动重试。实际资源检查用 smoke reserved peak + 1024 MiB。

```bash
bash experiment/phase18/run_stage18_beam_boundary_diagnostic.sh smoke --run-name smoke-0001 --gpu 7
bash experiment/phase18/run_stage18_beam_boundary_diagnostic.sh launch --run-name run-0001 --smoke-run smoke-0001 --gpu 7
```

以上是实现的接口；是否已启动/成功以 `artifacts/phase18/beam_boundary_diagnostic/` 内的记录为准。
