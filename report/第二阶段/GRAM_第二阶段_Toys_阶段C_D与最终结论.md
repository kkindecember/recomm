# GRAM 第二阶段 Toys 阶段 C/D 与最终结论

完成日期：2026-07-22

## 结论

- 阶段 C：seed 2023、30 epoch 单卡训练成功，退出码 0，6 个周期 checkpoint/validation 完整。
- 阶段 D：按 validation NDCG@10 选择 epoch 30，并以独立进程完成 19,412 用户、11,924 商品 full-ranking test。
- 对齐等级：A。Recall@5、NDCG@5、Recall@10、NDCG@10 相对论文误差分别为 0.8453%、0.3859%、3.4429%、1.8363%。
- 资源恢复：两个正式任务退出后均成功恢复 GPU 3 上的 CodeLlama reservation。

## 核心结果

| 指标 | 本地 |
|---|---:|
| Recall@5 | 0.0711930764 |
| NDCG@5 | 0.0514008682 |
| Recall@10 | 0.0953018751 |
| NDCG@10 | 0.0591926919 |
| 训练阶段墙钟时间 | 116,657.35 s |
| 最佳 checkpoint test | 3,481.12 s |
| 训练 peak allocated / reserved | 15,750.984 / 25,174 MiB |
| Test peak allocated / reserved | 7,155.063 / 24,368 MiB |

训练期间存在显著外部 GPU 干扰，时间不可视为独占 GPU 基准；独立最佳 checkpoint test 未观察到外部进程。

完整证据和可复现命令见 `artifacts/phase2_toys/REPRODUCTION_REPORT.md`。
