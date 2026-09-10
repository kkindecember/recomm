# 第十八阶段：ETEGRec Toys 主动停止记录

## Material Passport

- 状态：**USER_STOPPED**；2026-09-09 16:46:28（北京时间）停止。
- 用户指令：“那把 ETEGRec Toys 也停了吧”。
- 实验：`S18-ETEG-CF512-Toys-S2023`；GPU 5；原 PID `2572693` 已退出。
- 未完成整个筛选流程，不作为第二个完整负结果；未读取 test。
- [规范停止结果](../../artifacts/phase18/etegrec/toys/run_v1/result.json)、[汇总状态](../../artifacts/phase18/parallel_screen_20260909/status.json)。

## 停止时的完整进度

RQ 预训练完成 2,950 商品 pass / 35,400 updates，并按重建平台停止。联合阶段最后完整保存至 **第 222 轮**，ID 和 REC 各 111 轮、各 23,754 updates；之后的部分更新没有计入可恢复 checkpoint。

联合阶段已超过原计划 160 轮最小预算，但尚未完成平台停止判据（stale4，patience6）。固定编码微调尚未开始，没有最终 19,412 用户全量验证或配对结果。含 profile 已用 7,423.54 秒，约 **2.06 小时**；本次不是 12 小时预算到期。

最佳模型为 `joint_180`，固定 2,000 人趋势子集 NDCG@10 = **0.04182672847**，同子集 GRAM = **0.06937056512**。该数值只描述截断阶段，不与全量基线混比，不代表完成训练后的最终效果。

## 停止原因与保留内容

在 Beauty 已完成并明显落后于 GRAM 的证据下，用户决定停止当前 Toys 适配、降低 ETEGRec 当前设置的优先级。此处记录资源决策，不宣称整个联合编码学习机制无效。Beauty 的完整证据见 [结果报告](Stage18_ETEGRec_Beauty完整结果与投入结论报告.md)。

已保留 `last.pt`、`joint_best.pt`、`rq_best.pt`、全部已完成趋势预测、日志和来源快照；checkpoint 均可读取，最新模型参数有限，文件大小和 SHA 写入规范停止结果。

原始 Ctrl-C 产生 `FAILED / KeyboardInterrupt`，原 status 保存在 `status_on_interrupt.json`，中断事件原样保留。依据用户指令将 status / result 分类为 **USER_STOPPED**，并补记 `user_stop_recorded`；不混记为数值失败、自然早停或完整训练结束。

DIFF→GRAM Beauty 继续保持原 PID `2195156` 运行。汇总中的 ETEGRec Toys 已为 `USER_STOPPED / process_alive=false`；本次没有启动其他训练。
