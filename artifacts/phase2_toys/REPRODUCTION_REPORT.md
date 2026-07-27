# GRAM 第二阶段 Toys 单卡复现最终报告

完成日期：2026-07-22  
官方 GRAM commit：`7ac4d9272a57beed9df35c27ea34221f6e4a8fb1`  
数据集：Toys  
随机种子：2023

## 1. 最终结论

- 执行状态：**成功**。静态审计、smoke、30 epoch 正式训练、6 个周期 validation/checkpoint、最佳 checkpoint 独立 full-ranking test 均完成。
- 对齐等级：**A：基本对齐**。四项核心指标相对论文 Table 2 的误差均小于 5%。
- 最佳 checkpoint：epoch 30，validation NDCG@10 为 `0.07627451426000033`。
- 正式 test 覆盖 19,412 个用户，Trie 候选为全部 11,924 个商品，beam size 为 50。
- 数据、模型、优化器、scheduler、学习率、生成约束和评测公式均未为接近论文值而调整。
- 训练发生了明确的共享 GPU 外部干扰，因此墙钟时间是本次共享环境实测，不代表独占 RTX A6000 性能。

## 2. 核心指标

Leave-one-out 单目标评测下，代码输出的 `Hit@K` 等价于 Recall@K。绝对误差为“本地 - 论文”。

| 指标 | 论文 | 本地 | 绝对误差 | 相对误差 |
|---|---:|---:|---:|---:|
| Recall@5 | 0.0718 | 0.0711930764 | -0.0006069236 | 0.8453% |
| NDCG@5 | 0.0516 | 0.0514008682 | -0.0001991318 | 0.3859% |
| Recall@10 | 0.0987 | 0.0953018751 | -0.0033981249 | 3.4429% |
| NDCG@10 | 0.0603 | 0.0591926919 | -0.0011073081 | 1.8363% |

论文值为多次运行汇总，本阶段依计划只运行 seed 2023。最大相对误差为 3.4429%，满足 A 级标准。

## 3. 硬件与软件

| 项目 | 实际值 |
|---|---|
| GPU | 物理 GPU 3，NVIDIA RTX A6000，49,140 MiB |
| 程序内 GPU | `CUDA_VISIBLE_DEVICES=3` 后的逻辑 `cuda:0` |
| NVIDIA driver | 560.35.03 |
| Python | 3.9.25 |
| PyTorch | 1.11.0+cu113 |
| Transformers | 4.26.0 |
| NumPy | 1.23.1 |
| Conda 环境 | `gram-repro` |
| 模型缓存 | 项目 `.cache/huggingface/` |

## 4. 数据审计与评测协议

Toys 官方预处理数据只读审计通过：19,412 用户、11,924 商品、167,597 交互、密度 0.0724059%，与论文 Table 10 一致。每个用户最后一个交互为 test、倒数第二个为 validation，其余为 train。文本、层次 ID、用户序列和 SASRec 相似商品映射完整，无缺失、空值或重复层次 ID。

评测将全部 11,924 个商品层次 ID 放入 Trie，通过 `prefix_allowed_tokens_fn` 做 constrained beam search；没有负采样候选。

## 5. 正式配置与修改

| 配置 | 值 |
|---|---:|
| backbone | T5-small |
| seed / learning rate | 2023 / 1e-3 |
| epochs | 30 |
| per-device batch / accumulation | 16 / 8 |
| 有效 batch | 128 |
| prompt length / history | 128 / 20 |
| ID length / clusters | 5 / 32 |
| SASRec 相似商品 | 5 |
| checkpoint / validation interval | 5 epochs |
| checkpoint selection | validation NDCG@10 最高 |
| beam size | 50 |

单卡脚本将官方两卡近似有效 batch `32 × 2 × 2 = 128` 改为 `16 × 1 × 8 = 128`。源码只增加默认关闭的 `resource_metrics` 参数，以及训练/test 阶段边界的 CUDA 同步、计时和 allocator 峰值读取。守护脚本负责门槛、状态、PID/整卡/磁盘遥测和 CodeLlama 恢复。这些改动不改变模型计算；单卡与多卡浮点归约顺序可能产生正常的微小数值差异。

## 6. Smoke test

阶段 B 使用 1 epoch、100 条训练和 100 条 validation/test 样本，验证了数据加载、forward/backward、optimizer step、checkpoint 保存与重载、Trie constrained decoding、预测保存和三层资源指标。退出码为 0，`allocated <= reserved`，任务结束后 CodeLlama 恢复成功。Smoke 指标未用于论文对比。

## 7. 训练与 checkpoint 选择

正式任务从 2026-07-20 18:30:53 运行至 2026-07-22 03:51:56，退出码 0。训练 loss 从 `5.6837753371` 平稳下降至 `1.1582025722`，无 OOM、NaN、traceback 或数据错误。

| Epoch | Validation Recall@5 | NDCG@5 | Recall@10 | NDCG@10 | Validation 推理时间 |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.074593 | 0.054986 | 0.099062 | 0.062872 | 3,143.47 s |
| 10 | 0.083917 | 0.060886 | 0.112147 | 0.070040 | 7,872.26 s |
| 15 | 0.087369 | 0.063777 | 0.116989 | 0.073332 | 6,660.99 s |
| 20 | 0.089481 | 0.066326 | 0.118174 | 0.075564 | 3,037.13 s |
| 25 | 0.089120 | 0.065835 | 0.118689 | 0.075337 | 3,066.21 s |
| **30** | **0.090923** | **0.067135** | **0.119411** | **0.076275** | **3,035.40 s** |

epoch 30 的 validation NDCG@10 最高，故选为最佳 checkpoint。训练自动 test 恰好也使用 epoch 30，但阶段 D 仍按计划以独立进程显式重载该 checkpoint。

```text
GRAM/log/Toys/1_20260720_1830/id_0_rec_30/model_rec_phase_1_epoch_30.pt
SHA-256: b0d76ea4da9a40b1be43c55c1c7e20cdca4e1eff0194acbbe1b3cc15f471fd82
```

## 8. 时间

| 指标 | 结果 |
|---|---:|
| 训练阶段（30 epoch + 周期 validation/checkpoint，不含自动 test） | 116,657.35 s / 32.405 h |
| 正式训练任务端到端（含加载和自动 test） | 120,663 s / 33.518 h |
| epoch 纯训练平均 / 中位 | 2,934.83 s / 2,865.50 s |
| epoch 纯训练最小 / 最大 | 2,740 s / 3,487 s |
| 6 次 validation 推理合计 | 26,815.46 s / 7.449 h |
| 最佳 checkpoint 独立 test 阶段 | 3,481.12 s / 58.02 min |
| 阶段 D 守护任务端到端 | 3,501 s / 58.35 min |
| test 纯推理 | 3,198.77 s |
| 平均每用户推理 | 0.1648 s |

epoch 10/15 validation 与其他点相比异常偏慢，对应共享 GPU 干扰时段；训练时间不可冒充独占 GPU 基准。

## 9. 显存、遥测与磁盘

主指标为 GRAM 进程内 PyTorch allocator；PID NVML 包含 CUDA context；整卡仅用于解释共享环境。

| 阶段/口径 | MiB | GiB |
|---|---:|---:|
| 训练 peak allocated | 15,750.984 | 15.382 |
| 训练 peak reserved | 25,174 | 24.584 |
| 训练 PID NVML 峰值 | 27,184 | 26.547 |
| 最佳 test peak allocated | 7,155.063 | 6.987 |
| 最佳 test peak reserved | 24,368 | 23.797 |
| 最佳 test PID NVML 峰值 | 26,330 | 25.713 |
| 训练整卡背景峰值 | 47,106 | 46.002 |
| 最佳 test 整卡背景峰值 | 26,342 | 25.725 |

训练期间观察到 5 个外部 PID。最大外部分配为 19,944 MiB（2026-07-21 18:09–18:32），整卡峰值出现在该时段。独立最佳 checkpoint test 的 PID 遥测未观察到外部进程。训练和 test 期间 `/home` 最低可用空间分别为 392.81 GiB 和 390.20 GiB，始终高于启动门槛。

## 10. 阶段 D 与资源恢复

阶段 D 从 2026-07-22 09:47:41 运行至 10:46:02，workload PID 为 2328643，退出码 0。测试日志确认 19,412 个样本、beam 50 和选定模型路径。独立 test 与训练自动 test 的四项核心指标完全一致。

训练和阶段 D 退出后，守护脚本均立即执行 `run_codellama.sh start 3`；命令返回成功，状态文件记录 `resource_reservation=restored`，后续 CodeLlama cycle 输出证明任务继续运行。

## 11. 异常与限制

1. 正式训练共享 GPU 上有显著外部进程干扰，影响时间和整卡显存，未污染进程内 allocator 主指标。
2. 阶段 D 首次用普通 `nohup` 启动时，子进程被受控命令会话回收，尚未加载数据或占用 GPU；随后使用独立 tmux 会话成功运行，正式数据只来自成功任务。
3. `nvidia-smi` 在普通沙箱中无法访问驱动，GPU 门槛查询与启动改在获批的沙箱外执行；模型运行本身正常。
4. 本阶段只运行 seed 2023，不能替代论文的多种子均值。
5. 未清理任何 checkpoint、optimizer 或 scheduler；后续如需释放空间须由用户另行授权。

## 12. 完成条件

- [x] 环境与 Toys 数据已验证且未修改。
- [x] Smoke、30 epoch 正式训练和 6 个 validation/checkpoint 已完成。
- [x] validation NDCG@10 最佳 checkpoint 已选择、哈希并独立重载。
- [x] 11,924 商品 full-ranking test 和四项指标已完成。
- [x] 论文误差与 A/B/C 等级已给出。
- [x] 训练、epoch、validation、独立 test 时间已归档。
- [x] PyTorch allocator、PID NVML、整卡背景和干扰已区分。
- [x] 日志、状态、磁盘和 GPU 遥测已归档。
- [x] 实验退出后 CodeLlama reservation 已恢复。

## 13. 可复现命令

启动同配置正式训练前应重新由用户指定物理 GPU，并检查磁盘和至少 30,720 MiB（30 GiB）空闲显存。该门槛依据本次训练 27,184 MiB（26.55 GiB）和独立 test 26,330 MiB（25.71 GiB）的 PID NVML 实测峰值设置，并留有约 3 GiB 安全余量。以 GPU 3 为例：

```bash
cd /home/jiangtangyunzhi/projects/UnitTest
tools/run_codellama.sh stop
tmux new-session -d -s gram_phase2_toys_train \
  'bash /home/jiangtangyunzhi/projects/recomm/experiment/phase2_toys/run_phase2_toys_train.sh 3'
```

重跑本次最佳 checkpoint test：

```bash
cd /home/jiangtangyunzhi/projects/UnitTest
tools/run_codellama.sh stop
tmux new-session -d -s gram_phase2_toys_best_test \
  'bash /home/jiangtangyunzhi/projects/recomm/experiment/phase2_toys/run_phase2_toys_best_test.sh 3 /home/jiangtangyunzhi/projects/recomm/GRAM/log/Toys/1_20260720_1830/id_0_rec_30/model_rec_phase_1_epoch_30.pt'
```

两个 runner 均会在退出时自动恢复同一物理 GPU 上的 CodeLlama reservation。

## 14. 产物索引

- `data_audit.md`：数据审计。
- `checkpoint_selection.csv`：6 个 checkpoint 的选择表与最佳模型哈希。
- `metrics_seed2023.json`：论文对比和 A 级结论。
- `resource_summary.json`：时间、显存、遥测和干扰汇总。
- `epoch_timing.csv`：30 epoch loss、训练耗时和 validation。
- `environment/`：环境、磁盘、GPU、仪表化和代码修改。
- `logs/`：smoke、训练和最佳 checkpoint test 完整日志。
- `experiment/phase2_toys/`：状态和原始 GPU/进程/磁盘遥测。
