# 第20阶段状态查看

**2026-09-11新增任务后的统一入口：** [第20阶段实验总报告](../../report/第二十阶段/GRAM_第二十阶段_实验总报告.md)，每30秒更新，并自动纳入后续SPRec、round1_sft和GACR/PCRF结果。报告观察器状态：[report_watch/status.json](report_watch/status.json)。`all_monitored_jobs_finished=true`表示这些任务已全部进入终态，不表示全部成功或提点。

S-DPO已按用户要求停止。新增SFT状态：[round1_sft_full_v1/status.json](round1_sft_full_v1/status.json)；GACR当前执行目录由[机器配置](../../experiment/phase20/gacr_v6_pcrf_toys_v1.json)的`output`指定，调度状态：[followup_queue/status.json](followup_queue/status.json)。旧`overview/status.json`仅覆盖原SPRec/S-DPO，**不包含新增评估，不能据此判断本阶段全部结束**。下文保留旧入口的字段说明。

## 实时查看当前阶段

在项目根目录运行：

```bash
watch -n 30 -d cat report/第二十阶段/GRAM_第二十阶段_实验总报告.md
```

按Ctrl+C只退出查看，不停止实验。总报告链接到各专项报告，SFT和GACR专项报告还会依据近期实际吞吐估算当前步骤剩余时间。共享GPU负载可能改变速度。

这些Markdown报告在实验机器上自动更新；Git远端只显示最近一次提交的快照。运行状态、日志、模型和逐用户结果不会随Git分发，最终汇总结果与必要来源记录可以提交。

## 原SPRec/S-DPO总览（不含新增评估）

[`overview/status.json`](overview/status.json)每30秒刷新，仅用于原两条实验。

| 字段 | 含义 |
|---|---|
| `all_finished` | `true`表示SPRec队列与S-DPO分支都结束了，包括失败或主动停止 |
| `all_succeeded` | `true`表示两条都正常执行完成；不表示研究提点 |
| `state` | `RUNNING`仍有未结束任务；`COMPLETED`全部正常完成；`FINISHED_WITH_ERRORS`全部结束但有失败/停止 |
| `updated_at` | 总览更新时间；长时间不变时直接检查下面的原始状态 |
| `jobs.*.state` | 对应任务的执行状态 |
| `jobs.*.progress.stage` | 当前是原模型评估、负例生成、SFT、DPO/S-DPO或最终验证 |
| `jobs.*.progress.round_or_epoch` | 当前轮次；0通常表示训练前准备 |
| `jobs.*.progress.examples / total` | 当前步骤的进度，不是全部实验的总进度 |
| `estimated_current_step_remaining_hours` | 最近吞吐估算的当前步骤剩余时间，不是全流程剩余时间 |

单独观察：

- **SPRec整个队列：** [`status.json`](status.json)。先Toys，有正信号再自动Beauty。`state=COMPLETED`才表示该队列结束；Toys单独结束不一定表示队列结束。
- **SPRec Toys：** [`sprec_gram_toys_v1/status.json`](sprec_gram_toys_v1/status.json)。
- **S-DPO Toys：** [`sdpo_gram_toys_v1/status.json`](sdpo_gram_toys_v1/status.json)。
- SPRec Beauty若被触发，状态在`sprec_gram_beauty_v1/status.json`。

`RUNNING`在跑；`COMPLETED`正常完成；`FAILED`工程失败，读`error`和`run.log`；`STOPPED`被停止/到预算边界。总览中的`ORPHANED`表示状态写着运行但所记进程已经不存在，应检查原始日志。不要只看文件夹出现、round到3或loss下降就判断跑完。

完成后看对应正式目录的`summary.json`；过程中`selection.json`记录选模结果，`events.jsonl`和`run.log`记录详细进展。`smoke`文件夹仅为工程测试，不是正式实验结果。

运行计划：[SPRec](../../plan/第二十阶段/GRAM_第二十阶段_提点实验总计划.md)、[S-DPO](../../plan/第二十阶段/S-DPO并行实验计划.md)。
