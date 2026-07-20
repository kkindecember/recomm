# GRAM 第二阶段 Toys 阶段 B Smoke 报告

日期：2026-07-20

## 1. 执行结论

- 状态：成功。
- 物理 GPU：3；程序内逻辑 GPU：0。
- 配置：seed 2023、1 epoch、100 条训练样本、100 条 validation/test 样本、batch 4、gradient accumulation 1、beam 50。
- forward、backward、optimizer step、checkpoint 保存/重载、validation、test、Trie constrained decoding 和预测保存均通过。
- 任务退出码为 0，退出后 CodeLlama 已立即恢复并占住物理 GPU 3。
- Smoke 指标仅用于链路诊断，不与论文正式结果比较。

## 2. 结果

| 数据 | Hit@5 | NDCG@5 | Hit@10 | NDCG@10 | 推理时间 |
|---|---:|---:|---:|---:|---:|
| validation（100 条） | 0.0200 | 0.012619 | 0.0400 | 0.019191 | 44.39 s |
| test（100 条） | 0.0200 | 0.009307 | 0.0300 | 0.012461 | 44.19 s |

训练 loss 为 7.651441。训练阶段（包含该 epoch 后的 validation）墙钟时间为 58.992 s；自动末 checkpoint test 墙钟时间为 47.656 s。runner 从 18:20:03 到 18:22:10，总计约 127 s。

## 3. 显存与共享 GPU

| 口径 | 峰值 |
|---|---:|
| PyTorch peak allocated | 7,850.108 MiB |
| PyTorch peak reserved（训练） | 27,252 MiB |
| PyTorch peak reserved（test） | 27,294 MiB |
| NVML 中 GRAM workload PID 3322819 | 29,264 MiB |
| 整卡 memory.used | 32,864 MiB |

PID 遥测正确匹配 GRAM Python workload。任务期间另有 PID 2428185 持续占用约 3,582 MiB，因此整卡数值包含外部进程，不能作为 GRAM 独占显存；正式报告继续以 PyTorch 进程内数值为主。`allocated <= reserved` 条件满足。

## 4. 产物

- checkpoint：`GRAM/log/Toys/0_20260720_1820/id_0_rec_1/model_rec_phase_1_epoch_1.pt`，约 231 MiB；对应 optimizer 约 462 MiB。
- validation 预测：`GRAM/preds/20260720_182028_Toys_sequential_pred_validation.tsv`。
- test 预测：`GRAM/preds/20260720_182122_Toys_sequential_pred_test.tsv`。
- 主日志：`artifacts/phase2_toys/logs/smoke_test.log`。
- 状态与遥测：`experiment/phase2_toys/`。
- 实验结束时 `/home` 约有 427 GiB 可用空间。

## 5. 运行异常说明

第一次尝试使用 `nohup` 启动时，子进程随受控执行环境退出而被回收，只写入了启动标识，未进入数据加载或生成实验产物。资源尚未恢复时立即改用独立 tmux 会话运行；本报告只统计 18:20:03 开始的成功运行。后续 GPU 任务统一使用 tmux。

`run_codellama.sh stop` 返回后 CUDA 显存约需数秒才完全释放。本次加入轮询后，GPU 3 空闲显存从 13,929 MiB 上升到 44,976 MiB，再通过 40,960 MiB 门槛启动任务。

## 6. 下一阶段

阶段 C 为 Toys seed 2023、30 epoch 正式单卡训练，batch 16 / accumulation 8。启动前必须由用户重新指定物理 GPU，并重复资源释放、显存/磁盘门槛检查、tmux 后台启动和退出恢复流程。
