# GRAM 第一阶段 Beauty Smoke Test 报告

日期：2026-07-18

## 1. 执行结论

- 状态：成功。
- 物理 GPU：GPU 3，NVIDIA RTX A6000（48 GiB）；进程内映射为逻辑 GPU 0。
- 成功运行区间：2026-07-18 21:39:21 至 21:47:03（约 7 分 42 秒，包含模型加载、训练、验证和测试）。
- 成功运行 PID：守护进程 `3461397`，实际 GRAM Python workload `3461409`。
- 训练、checkpoint 保存、checkpoint 重载、validation、test、指标输出和预测文件保存链路均已通过。
- 退出码：0。
- 实验退出后已自动执行 CodeLlama `start 3`，GPU 3 的资源占用已恢复。

本次指标只来自 100 条调试样本和 1 epoch，不能作为论文复现结果，也不用于与论文 Table 2 比较。

## 2. 实际执行配置

成功运行通过以下后台守护脚本启动：

```bash
tmux new-session -d -s gram_phase1_smoke \
  'bash /home/jiangtangyunzhi/projects/recomm/experiment/run_phase1_beauty_smoke.sh 3'
```

守护脚本最终调用：

```bash
cd /home/jiangtangyunzhi/projects/recomm/GRAM/command
PHYSICAL_GPU=3 \
PYTHON_BIN=/home/jiangtangyunzhi/miniconda3/envs/gram-repro/bin/python \
bash smoke_test_gram_beauty_single.sh
```

关键 smoke 配置：

| 配置 | 值 |
|---|---:|
| dataset | Beauty |
| seed | 2023 |
| epochs | 1 |
| 训练样本 | 100 |
| validation 样本 | 100 |
| test 样本 | 100 |
| rec batch size | 4 |
| gradient accumulation | 1 |
| learning rate | 1e-3 |
| beam size | 50 |
| ID length | 7 |
| clusters | 128 |
| similar items | 10 |
| max history | 20 |
| item prompt max length | 128 |

正式 30-epoch 实验不会传入 `debug_train_100` 或 `debug_test_100`，因此仍使用完整训练数据和 22,363 用户的 full-ranking 测试。

## 3. 训练与评测结果

### 3.1 训练

- 训练步数：25（100 样本 / batch 4）。
- epoch 1 平均 recommender loss：`8.22916353225708`。
- 模型 checkpoint 已保存并由测试入口重新加载。

### 3.2 Validation（100 条，仅诊断）

| 指标 | 数值 |
|---|---:|
| Hit/Recall@5 | 0.0000 |
| NDCG@5 | 0.0000 |
| Hit/Recall@10 | 0.0000 |
| NDCG@10 | 0.0000 |
| Hit/Recall@50 | 0.0200 |
| NDCG@50 | 0.0038415938 |

推理总时间 20.16 秒，平均每条 0.2016 秒。

### 3.3 Test（100 条，仅诊断）

| 指标 | 数值 |
|---|---:|
| Hit/Recall@5 | 0.0000 |
| NDCG@5 | 0.0000 |
| Hit/Recall@10 | 0.0100 |
| NDCG@10 | 0.0031546488 |
| Hit/Recall@20 | 0.0400 |
| NDCG@20 | 0.0108577476 |
| Hit/Recall@50 | 0.0700 |
| NDCG@50 | 0.0168494308 |

推理总时间 20.63 秒，平均每条 0.2063 秒。

1 epoch、100 条样本出现较低指标属于预期现象；本阶段只据此判断评测链路和指标计算能够执行。

## 4. GPU 与资源

- 成功运行监测到的峰值总显存：24,932 MiB（约 24.35 GiB）。
- 峰值样本时间：2026-07-18 21:46:41+08:00。
- 峰值时剩余显存：23,638 MiB。
- 峰值主要出现在 constrained beam search 推理阶段；加载阶段曾只有约 2.2 GiB，占用会随模型和 beam 状态建立逐步上升。
- 正式训练使用 batch 16，而 smoke 使用 batch 4，因此正式训练不能直接按 24.35 GiB 峰值估算；应在启动后重新测量，必要时按计划改为 batch 8 / accumulation 16 或 batch 4 / accumulation 32。

## 5. 异常、原因与修正

### 5.1 Hugging Face 下载缺少 SOCKS 支持

首次可观察失败为：

```text
requests.exceptions.InvalidSchema: Missing dependencies for SOCKS support.
```

主机设置了 SOCKS 代理，但 `gram-repro` 缺少对应传输依赖。已安装并锁定 `PySocks==1.7.1`，随后 `pip check` 通过。该依赖只影响 HTTP 代理连接，不参与模型计算，不影响数值。

### 5.2 重复启动两个 GRAM 进程

首次 `setsid` 启动的宿主机子进程没有退出，但受限环境内的 PID 检查错误地将其判断为已结束；依赖修复后又启动了一份，导致 GPU 3 同时出现：

- PID `2178077`，约 11,992 MiB；
- PID `2231732`，约 26,240 MiB。

两者共同占用约 38,250 MiB。2026-07-18 21:38:02 已按各自精确 SID 发送 TERM，退出码均为 143；随后确认旧 GPU 计算进程已清除。最终成功运行只启动一个 workload。

守护状态原来的 `background_pid` 实际记录外层 Bash PID，容易被误认为 GPU Python PID。现改为分别记录：

- `runner_pid`：资源恢复和状态管理进程；
- `workload_pid`：实际 GRAM Python 进程。

### 5.3 Smoke 意外进入完整测试

官方单卡入口在训练结束后执行：

```python
runner.args.debug_test_100 = 0
```

这会覆盖 smoke 脚本显式传入的 `--debug_test_100 1`，使最终 test 对全部 22,363 用户运行。两个误启动进程在约 43% 处被终止。

现删除该强制重置语句。只有显式传入 `debug_test_100=1` 的 smoke 运行会限制最终 test 为 100 条；正式脚本默认值为 0，仍执行完整 full-ranking，正式实验协议不变。

## 6. 文件修改及数值影响

| 文件 | 修改 | 是否影响正式数值 |
|---|---|---|
| `GRAM/command/smoke_test_gram_beauty_single.sh` | 支持 `PHYSICAL_GPU`、显式环境 Python，并保留 100 条训练/评测限制 | 否，仅用于 smoke |
| `GRAM/src/main_generative_gram.py` | 不再强制覆盖显式传入的 `debug_test_100` | 否；正式脚本未启用该调试参数 |
| `experiment/run_phase1_beauty_smoke.sh` | 记录 runner/workload PID、GPU 遥测，退出后恢复 CodeLlama | 否，不参与模型计算 |
| `gram-repro` 环境 | 增加 `PySocks==1.7.1` | 否，仅用于代理传输 |

## 7. 产物路径

- 汇总日志：`artifacts/phase1_beauty/logs/smoke_test.log`
- 状态：`experiment/phase1_beauty_status.json`
- GPU 遥测：`experiment/phase1_beauty_smoke_gpu.csv`
- 成功运行目录：`GRAM/log/Beauty/3_20260718_2139/`
- 模型 checkpoint：`GRAM/log/Beauty/3_20260718_2139/id_0_rec_1/model_rec_phase_1_epoch_1.pt`
- optimizer：`GRAM/log/Beauty/3_20260718_2139/id_0_rec_1/optimizer_rec_phase_1_epoch_1.pt`
- validation 预测：`GRAM/preds/20260718_214610_Beauty_sequential_pred_validation.tsv`
- test 预测：`GRAM/preds/20260718_214639_Beauty_sequential_pred_test.tsv`

## 8. 下一步

Smoke test 已满足进入正式训练的前置条件。下一步在用户重新指定物理 GPU 后，使用正式单卡脚本运行 seed 2023、30 epochs；每 5 epoch 保存并验证，训练完成后按 validation NDCG@10 选择最佳 checkpoint，再进行完整 full-ranking 测试。
