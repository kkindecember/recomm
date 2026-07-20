# GRAM 第二阶段 Toys 阶段 A 静态审计报告

日期：2026-07-20

## 1. 执行结论

- 阶段 A 状态：成功。
- Toys 数据规模与论文完全一致，四类输入非空且无缺失映射。
- `gram-repro` 依赖一致性通过，T5-small 与代码中的辅助 T5 权重可从项目缓存离线加载。
- `/home` 可用空间约 428 GiB，超过正常启动门槛；由于是共享盘，每个 GPU 任务启动前仍必须重新检查。
- Toys 单卡、smoke、最佳 checkpoint 脚本以及后台守护/遥测脚本已新建并通过语法检查。
- 共享 GPU 显存采用 PyTorch 进程内峰值作为主指标，PID 采样作为辅助，整卡数值只作为背景。
- 本阶段未选择 GPU，未停止 CodeLlama，未启动 CUDA 训练或评测。

## 2. 官方配置复核

官方 `command/train_gram_toys.sh` 配置为 seed 2023、T5-small、learning rate 1e-3、30 epoch、ID length 5、32 clusters、5 个 SASRec 相似商品、history 20、prompt length 128、每 5 epoch 保存并 validation。默认 beam size 为 50，warmup ratio 为 0.05。

新单卡脚本只将：

```text
2 GPU distributed, batch 32, accumulation 2
```

改为：

```text
1 GPU non-distributed, batch 16, accumulation 8
```

两者有效 batch 均为 128。此外新脚本显式启用只记录资源的 `resource_metrics=1` 和官方默认 beam 50，其他核心参数不变。

## 3. 协议和 checkpoint 检查

- Leave-one-out：训练使用最后两个交互之前的序列，validation 目标为倒数第二个，test 目标为最后一个。
- Full ranking：全部 11,924 个商品文本 ID 进入 Trie，`generate()` 使用 `prefix_allowed_tokens_fn` 和 50 beams。
- 官方单卡 runner 虽然定义 `best_score` 和 `best_epoch`，但未更新最佳值；训练结束自动 test 使用当前最后 checkpoint。
- 阶段 D 必须从 epoch 5/10/15/20/25/30 的 validation 日志中选择 NDCG@10 最高者，再用独立 test 脚本加载。

## 4. 资源仪表化

新增 `--resource_metrics`，默认为 0。Toys 脚本传入 1 时，在训练与 test 阶段边界记录：

- 同步后 wall time；
- PyTorch peak/end allocated；
- PyTorch peak/end reserved。

后台守护每 5 秒记录整卡和每个 compute PID 显存，每 5 分钟记录 `/home` 剩余空间。正式报告以 PyTorch 本进程值为主，不把其他用户的占用算入 GRAM。

## 5. 新增/修改文件

| 文件 | 原因 | 可能影响数值 |
|---|---|---|
| `GRAM/src/arguments.py` | 增加默认关闭的资源统计开关 | 否；默认值 0 |
| `GRAM/src/main_generative_gram.py` | 在单卡阶段边界记录同步 wall time 和进程内 CUDA 峰值 | 预期否；不改变模型/数据/优化/生成，仅边界同步会产生可记录的微小时间开销 |
| `GRAM/command/train_gram_toys_single.sh` | 保持有效 batch 128 的 Toys 单卡正式入口 | 单/多卡浮点顺序可有微小差异；核心协议不变 |
| `GRAM/command/smoke_test_gram_toys_single.sh` | 1 epoch、100 条样本的链路测试 | 否；不用于正式结果 |
| `GRAM/command/test_gram_toys_best_single.sh` | 显式加载 validation 最佳 checkpoint | 否；使用原生 test 入口 |
| `experiment/phase2_toys/run_phase2_toys_job.sh` 及三个 wrapper | 后台运行、状态、门槛、PID/整卡/磁盘遥测和资源恢复 | 否；不参与模型计算 |

## 6. 静态验证

- 新增 7 个 shell 脚本均通过 `bash -n`。
- `arguments.py` 和 `main_generative_gram.py` 通过 `py_compile`。
- Parser 能解析 `--resource_metrics 1`。
- 仪表化关闭时的静态函数测试返回原操作结果。
- `git diff --check` 通过。
- 输出目录可写。
- 离线模型加载通过。
- 发现 Conda 环境的 `bin/python` 链接缺失；第二阶段 runner 已改用实测有效的 `bin/python3.9`，并重新完成校验。

一次未设置项目 `HF_HOME` 的静态 import 触发了 Transformers 缓存迁移警告，但没有迁移文件、安装依赖或影响项目缓存。GPU 脚本会显式将 `HF_HOME` 与 `TRANSFORMERS_CACHE` 指向项目可写缓存。

## 7. 阶段 B 前置条件

阶段 A 已满足进入 Toys smoke test 的静态条件。启动阶段 B 前仍需：

1. 请用户指定物理 GPU；
2. 用 `tools/run_codellama.sh stop` 释放用户保留资源；
3. 检查目标 GPU 至少有 40,960 MiB 可用显存；
4. 重新检查 `/home` 磁盘空间；
5. 在 tmux 中启动 `experiment/phase2_toys/run_phase2_toys_smoke.sh <GPU>`。
