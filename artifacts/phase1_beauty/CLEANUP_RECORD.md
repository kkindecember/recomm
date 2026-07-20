# Phase 1 Beauty 实验清理记录

日期：2026-07-20

用户授权在保留最终复现必需产物的前提下，删除不再需要的训练状态和中间结果。

## 保留

- 最佳 epoch 25 模型：`GRAM/log/Beauty/4_20260718_2153/id_0_rec_30/model_rec_phase_1_epoch_25.pt`
  - 大小：242,132,665 bytes
  - SHA-256：`0df16263344afec8a40e29a7a33e43e7bbe254e0ab97799c9103ad757e60fa89`
- 阶段 D 最终预测：`GRAM/preds/20260720_150243_Beauty_sequential_pred_test.tsv`
  - 大小：61,394,839 bytes
  - SHA-256：`ff8b22b9697bf8ad4cbd1a58734944414790adee34090c2153b1c96fe5f4bf07`
- `artifacts/phase1_beauty/` 中的报告、指标、环境快照和全部日志。
- `report/`、`experiment/`、Beauty 数据、代码和 Hugging Face 缓存。
- 各运行目录内的小型参数/元数据文件；只清理其中的 `.pt` checkpoint 状态。

## 删除

- 正式训练 epoch 5/10/15/20/25/30 的 6 个 optimizer 状态和 6 个 scheduler 状态。
- 非最佳 epoch 5/10/15/20/30 的 5 个模型 checkpoint。
- smoke/失败尝试运行目录 `0_20260718_1955`、`1_20260718_1957`、`2_20260718_1959`、`3_20260718_2139` 中的所有 `.pt` checkpoint（目录 1 实际无 `.pt`）。
- `GRAM/preds/` 中除阶段 D 最终预测和 `.gitkeep` 之外的 13 个旧预测文件。

删除前精确合计：6,777,359,422 bytes，约 6.312 GiB。

执行后验证：

- 项目总占用从约 7.7 GiB 降至约 1.4 GiB。
- `GRAM/log/Beauty/` 中只剩 epoch 25 模型 checkpoint。
- `GRAM/preds/` 中只剩阶段 D 最终预测和 `.gitkeep`。
- 保留的模型和预测文件 SHA-256 与删除前完全一致。
- 删除完成时整个共享文件系统显示约 254 GiB 可用；该数字同时受其他用户操作影响，不能全部归因于本次 6.312 GiB 清理。

## 影响

- 最终 epoch 25 推理、指标核验和阶段一报告不受影响。
- 不再能从本地 optimizer/scheduler 状态无损续训，也不能直接重新测试已删除的非最佳 epoch 或 smoke 模型。
- 被删除文件未移入回收站，不能从本工作区恢复；只能通过重新训练/评测再生成。
