# 第20阶段状态查看

**2026-10-15 Git归档：** 已补充LIME专家删减、前缀权重、过滤与路径融合、生成机制、干净监督、提前纠错及当前生成状态检查的完成证据，并保留两域上游训练汇总。代码和日期报告同时归档；后续研究见[第21阶段索引](../../report/第二十一阶段/README.md)。本页下方状态说明均有各自历史时点，实时运行文件仍仅保留在实验机器。

**2026-09-21 Git归档：** 近期实验的代码、配置、计划报告、已完成的聚合指标与独立核验记录已保留。当前生成引导仅归档固定配置、来源清单、完整分卡协议及[提交快照](../../report/第二十阶段/LIME信号引导原GRAM生成_提交快照_20260921.md)，不将进行中结果视为终态。实时状态、日志、逐用户数据、模型和重复源码副本仍留在本机；外部仓库按[固定版本说明](../../third_party/README.md)恢复。

## Git归档范围（2026-09-15）

Git保留实验实现与测试、固定配置、计划报告、汇总指标、必要来源与恢复凭据，以及可复算的诊断脚本。今天的[内部蒸馏提交快照](../../report/第二十阶段/Stage20_Toys完整列表内部蒸馏提交快照_20260915.md)和[教师诊断数据](review_20260915/training_teacher_diagnostic.json)已归档。

实时status、动态进度报告、日志、权重、逐用户预测、候选缓存、重复源码／启动配置、一次性报告写入脚本继续保留在本机，并由`.gitignore`排除。下文实时入口供实验机器使用；远端请查看带日期的快照和正式结果报告。

**2026-09-15恢复入口：** 凌晨`gacr_list_distill_toys_gpu1_v2`已FAILED，原状态保留。当前看[`gacr_list_distill_toys_recovery_v3/status.json`](gacr_list_distill_toys_recovery_v3/status.json)；复用缓存、三臂分卡重新训练，原科学配置不变。启动前21项检查通过；管理脚本`bash experiment/phase20/run_gacr_list_distill_recovery.sh status`。[恢复凭据](gacr_list_distill_toys_recovery_v3/recovery_receipt.json)和[计划第10节](../../plan/第二十阶段/Toys完整候选列表内部蒸馏实验计划.md)。下方旧分卡描述保留为历史记录，实时状态只看新目录。

**2026-09-15已切换三臂分卡调度：** 原GPU1缓存由新调度器接管，完成后CE/真教师/打乱教师优先GPU4/1/6，并按≥30,000 MiB准入选择未绑定的A6000。每臂训练后独立验证，全部完成后汇总。继续看[`status.json`](gacr_list_distill_toys_gpu1_v2/status.json)的`arm_gpu_bindings`、`active_children`和`task_status`；`child_status`只在共享缓存阶段使用。新管理脚本`bash experiment/phase20/run_gacr_list_distill_parallel.sh status`；[交接验证](gacr_list_distill_toys_gpu1_v2/parallel_v2/launch_verification.json)确认缓存PID和计算进度保留。首次工程失败保留于`parallel/`，成功版本记录于`parallel_v2/`。

**2026-09-14内部迁移实验已启动：** 看[`gacr_list_distill_toys_gpu1_v2/status.json`](gacr_list_distill_toys_gpu1_v2/status.json)，GPU1后台9项任务：工程profile、父列表缓存、三臂训练、三臂完整真实beam验证、配对汇总。状态每30秒更新，`child_status`为当前阶段，`projected_remaining_seconds`为实测外推ETA。见[动态报告](../../report/第二十阶段/Stage20_Toys完整列表内部蒸馏进展与结果.md)。`gacr_list_distill_toys_v1`是GPU3资源等待后主动停止的初版，未训练。以下保留历史批次时间点。

**2026-09-13 15:05 B1多初始化验证已完成：** [`gacr_b1_multiseed_v1/status.json`](gacr_b1_multiseed_v1/status.json)为COMPLETED，6/6任务成功、120步新残差更新、实际11分57秒；完成后心跳停止。两域均3/3个初始化N10/H10高于自身B0，B1平均N10对原PCRF+4.65%/+4.09%。见[最终报告](../../report/第二十阶段/Stage20_B1缓存复用多初始化验证进展与结果.md)、[独立复核](review_20260913/b1_multiseed_completion_verification.json)及[下一步裁决第11节](../../plan/第二十阶段/GACR_PCRF双域提点后的方法推进计划.md)。两轮新训练均已完成，内部迁移尚未启动。

```bash
cat artifacts/phase20/gacr_b1_multiseed_v1/status.json
```

**2026-09-13 05:04 B1/B2已完成：** [`gacr_fusion_train_v1/status.json`](gacr_fusion_train_v1/status.json)为COMPLETED，4/4任务成功、120步残差更新、0步骨干更新，耗时7小时58分57秒；终态不再刷新心跳。两域B1/B2均超过B0，B2相对B1的额外优势未获支持。见[完整结果分析](../../report/第二十阶段/Stage20_固定候选最终融合训练进展与结果.md)、[summary](gacr_fusion_train_v1/summary.json)及[下一步计划第9节](../../plan/第二十阶段/GACR_PCRF双域提点后的方法推进计划.md)。两域训练缓存可供后续残差训练复用；新seed和内部化未启动。下方保留旧批次各自时间点。

```bash
watch -n 30 cat artifacts/phase20/gacr_fusion_train_v1/status.json
```

**2026-09-12 16:39 A1/A2已完成：** [统一status](gacr_controls_multiseed_v1/status.json)为COMPLETED，4/4任务、6/6阶段全部成功，耗时34.9分钟；完成后不再刷新心跳。各域详细状态为该目录下`Beauty/status.json`、`Toys/status.json`；结果见[summary](gacr_controls_multiseed_v1/summary.json)和[最终报告](../../report/第二十阶段/Stage20_Beauty固定控制与双域残差多seed进展报告.md)。full平均N10相对原PCRF为Toys +3.13%、Beauty +1.78%，fixed为+3.26%/+1.78%；两域平均H50仍下降。已选择fixed作为下一轮匹配训练平台，B/C尚未启动，现有status保留队列结束时的协议待定字段。下一步见[计划第7节](../../plan/第二十阶段/GACR_PCRF双域提点后的方法推进计划.md)。

```bash
watch -n 30 cat artifacts/phase20/gacr_controls_multiseed_v1/status.json
```

`completed_scientific_phases`为已完成控制/seed阶段数（总数6），运行时`child_status`含当前步骤和用户进度，终态`domain_status`保存两域完成状态。`COMPLETED`表示本轮A1/A2完成，不能据此推断B/C也完成。失败/超时不自动重试。

**本轮开跑前的seed2023结果：** Toys固定控制和Beauty冻结六臂均已结束。GACR+PCRF对原PCRF的N10在Toys/Beauty分别+2.81%/+2.00%。见[双域结果分析](../../report/第二十阶段/Stage20_GACR_PCRF双域结果与提点路线_20260912.md)；Beauty[独立复核](review_20260912/beauty_completion_verification.json)通过。后续控制和残差seed现已完成，新训练尚未启动；下方运行命令为历史入口。

**2026-09-12终态核查：** 本批正式任务均进入终态，报告观察器已结束；后续建议尚未开跑。请先读[终态复盘](../../report/第二十阶段/第20阶段_终态复盘与下一步决策_20260912.md)和[完整索引](../../report/第二十阶段/README.md)。下方watch、启动入口和估计是历史运行说明。

EAGER/PPAC新增实验统一看[简洁增量报告](../../report/第二十阶段/EAGER_PPAC实验进展报告.md)。后台入口：`bash experiment/phase20/run_personal_popularity.sh start`；单次查看：`bash experiment/phase20/run_personal_popularity.sh status`。该队列不在旧三任务观察器中，报告由新程序按关键步骤自动追加，无需持续watch。

2026-09-11研究收敛补充：[PCRF组件对照与研究复盘](../../report/第二十阶段/第20阶段_研究收敛复盘与PCRF组件对照报告.md)。固定7臂CPU消融已完成，19,412名用户的原GRAM/PCRF rank与参考精确一致；后续候选实验和未验证边界见[计划](../../plan/第二十阶段/PCRF机制对照与研究收敛计划.md)。

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
