# GRAM 单卡复现实验

**2026-10-15 Git归档：** 补充第20阶段LIME生成诊断、专家删减及当前状态实验，并归档第21阶段专家辅助生成、候选权重、信息来源与双／三专家融合对照的代码、计划、报告和精简证据。最新入口见[第21阶段索引](report/第二十一阶段/README.md)，产物范围见[归档说明](artifacts/phase21/README.md)。下文日期较早的运行描述保留为历史记录；远端报告是本次提交时的快照。

**2026-09-21 Git归档：** 本次保留近期实验代码、固定配置、计划报告、已完成实验的汇总与核验记录，以及[生成引导运行安排快照](report/第二十阶段/LIME信号引导原GRAM生成_提交快照_20260921.md)。当前完整分卡协议和配置也纳入Git；实时状态与动态报告继续留在实验机器。LIME-Rec、SEAR完整下载目录通过固定提交恢复，见[第三方说明](third_party/README.md)。此次归档不改变后台实验。

**2026-09-21最新接续：** GRAM＋LIME训练集留出验证已改为每域双卡、两域同时推进。Toys从原第452步完整接续到GPU4、5，实跑约3.6秒/更新，原单卡平均8.8秒；Beauty双卡检查通过后已在GPU6、7启动。保持全局批量128、GRAM30轮→C1五轮，短测纯训练投影约27/44小时，随共享负载变化。双卡梯度、缓存张量与真实保存恢复检查通过；无新推荐效果。见[提速与接续记录](report/第二十阶段/LIME训练集留出_双卡提速与检查点接续_20260921.md)、[Toys状态](artifacts/phase20/lime_parallel_toys_20260921_v1/status.json)、[Beauty状态](artifacts/phase20/lime_parallel_beauty_20260921_v1/status.json)。旧native_v2的停止状态是主动迁移记录，下方条目保留历史时点。

**2026-09-15 Git归档：** 本次内部蒸馏进度见[提交快照](report/第二十阶段/Stage20_Toys完整列表内部蒸馏提交快照_20260915.md)，方法研究见[文献借鉴与下一版建议](report/第二十阶段/排序迁移文献借鉴与下一版方法建议_20260915.md)。下文的实时status和动态报告仅在实验机器上存在，已加入忽略规则；固定配置、测试、汇总结果、恢复凭据和诊断脚本纳入版本管理。当前实验的最终评估完成后再归档最终结果。

**2026-09-15恢复入口：** 凌晨分卡尝试因显存准入失败已终止。修订版复用已校验缓存，三臂从原C1重新开始；新增启动余量和进程内资源等待。当前看[恢复status](artifacts/phase20/gacr_list_distill_toys_recovery_v3/status.json)及[动态报告](report/第二十阶段/Stage20_Toys完整列表内部蒸馏进展与结果.md)，管理脚本为`bash experiment/phase20/run_gacr_list_distill_recovery.sh status`。[恢复协议第10节](plan/第二十阶段/Toys完整候选列表内部蒸馏实验计划.md)。下方旧启动记录为历史时间点。

2026-09-15已按用户要求改为三臂分卡调度：原GPU1缓存保留，完成后CE/真教师/打乱教师优先GPU4/1/6，各臂训练后独立完整验证。统一[status](artifacts/phase20/gacr_list_distill_toys_gpu1_v2/status.json)继续更新，实际GPU绑定见`arm_gpu_bindings`；管理入口改为`bash experiment/phase20/run_gacr_list_distill_parallel.sh status`。见[调度修订](plan/第二十阶段/Toys完整候选列表内部蒸馏实验计划.md)和[接管复核](artifacts/phase20/gacr_list_distill_toys_gpu1_v2/parallel_v2/launch_verification.json)。

2026-09-14已启动Toys完整列表向生成器内部迁移的三臂实验：匹配CE＋父保持、真实B1＋PCRF教师、打乱教师。只更新C1解码器，随后重新生成候选并评估全部八指标。GPU1后台状态见[status](artifacts/phase20/gacr_list_distill_toys_gpu1_v2/status.json)，[报告](report/第二十阶段/Stage20_Toys完整列表内部蒸馏进展与结果.md)随队列更新；初版GPU3仅等待资源后停止，未训练。

第20阶段B1多初始化验证于2026-09-13 15:05完成，耗时11分57秒。Toys/Beauty均3/3个残差初始化的N10/H10超过自身B0；B1平均N10相对原PCRF提升4.65%/4.09%，H10提升4.65%/3.68%。完整指标、分组例外和下一步见[最终报告](report/第二十阶段/Stage20_B1缓存复用多初始化验证进展与结果.md)与[完成status](artifacts/phase20/gacr_b1_multiseed_v1/status.json)。下一步准备完整候选列表向生成器迁移的最小验证，尚未启动。

本仓库保存 GRAM Beauty/Toys 单卡复现所需的代码修改、命令脚本、实验编排、计划、审计记录和结果报告。

第20阶段B1/B2匹配训练于2026-09-13 05:04完成，耗时约7小时59分。B1相对原PCRF的N10在Toys/Beauty提升4.68%/4.02%，H10提升4.69%/3.58%；B2与B1接近，融合纳入训练尚未显示额外优势。详见[最终分析报告](report/第二十阶段/Stage20_固定候选最终融合训练进展与结果.md)和[完成status](artifacts/phase20/gacr_fusion_train_v1/status.json)。下一步建议复用缓存验证B1的其他残差初始化；内部化C尚未启动。

## 仓库内容

- `GRAM/src/`：在官方 GRAM 基础上的最小代码修改；
- `GRAM/command/`：Beauty/Toys 单卡训练、smoke 与最佳 checkpoint 测试入口；
- `experiment/phase1/`、`experiment/phase2_toys/`：后台运行与资源恢复脚本；
- `experiment/phase18/`：DIFF→GRAM、DiffGRM 候选实现、诊断、测试和 checkpoint 接续；
- `experiment/phase19/`：RaSeRec→GRAM Beauty 补充实验、预检查及后台状态管理；Toys训练复用`experiment/phase18/`中的RaSeRec实现；
- `experiment/phase20/`：全参数GRAM自博弈偏好训练、真实beam评估与条件跨域队列；
- `third_party/RaSeRec/`：固定提交的作者参考代码、默认配置、来源记录和MIT许可证；
- `plan/`：阶段实验计划；
- `report/`：分阶段报告；
- `artifacts/`：小型数据审计、环境快照、指标和代码差异记录。

## 未纳入 Git 的本地产物

以下内容体积大、可重新生成或在运行中持续变化，因此通过 `.gitignore` 排除：

- GRAM 官方数据集和展示 assets；
- Hugging Face 模型缓存；
- checkpoint、optimizer、scheduler 和预测文件；
- 完整训练日志；
- PID、运行状态以及 GPU/磁盘高频遥测 CSV。

第十八阶段的 `artifacts/phase18/` 默认保留在实验机器，仅将 `.gitignore` 中明确列出的已完成实验汇总、固定案例清单和来源哈希纳入 Git。训练目录中的逐用户预测、生成数据、嵌入、源码副本和进行中的指标均忽略；忽略或取消跟踪不会删除本地文件。后续实验完成后再整理必要的终态汇总。

第十九阶段采用相同规则：提交固定CPU检索检查的协议、脚本与汇总，以及已完成的输入检查；训练中的status、日志、模型、逐用户结果和重复源码快照保留在本机。旧阶段的GPU租用状态、CPU遥测及当前数据集标记也只保留在本机。各阶段已有的固定参考、划分契约与科学结果仍纳入Git，其中包括后续实验依赖的PCRF验证参考。

当前第20阶段结果见[双域残差三seed最终报告](report/第二十阶段/Stage20_Beauty固定控制与双域残差多seed进展报告.md)和[实验报告索引](report/第二十阶段/README.md)。截至2026-09-12 16:39，Beauty固定控制和两域残差三seed验证也已完成；完整GACR+PCRF平均NDCG@10相对原PCRF在Toys/Beauty提高3.13%/1.78%，固定候选方案提高3.26%/1.78%，两域平均Hit@50仍下降。此处只改变残差seed，不是完整模型三次重训。下一轮匹配融合训练方案见[方法推进计划](plan/第二十阶段/GACR_PCRF双域提点后的方法推进计划.md)，B/C未启动。旧`artifacts/phase20/overview/status.json`只覆盖原SPRec/S-DPO，不能代表全部后续实验状态；各批次入口见[状态查看说明](artifacts/phase20/README.md)。

第20阶段提交实现、固定配置、计划报告、汇总审计和模型来源记录；模型权重、日志、逐用户预测、候选特征、重复源码快照和进程状态保留在本机。Git中的自动报告是提交时的快照，本机后续更新不会自动推送；正式评估的最终`summary.json`已在忽略规则中预留，生成后可随结论一并提交。

第18、19阶段已于2026-09-10按用户决定收尾，分别见[第18阶段报告](report/第十八阶段/GRAM_第十八阶段_实验收尾总报告.md)和[第19阶段报告](report/第十九阶段/GRAM_第十九阶段_实验进展与结论总报告.md)。旧状态文件不再代表仍有运行任务。RaSeRec参考源码的保留范围见[第三方代码说明](third_party/README.md)。

历史四条候选的预算与接续约定见[第十八阶段执行补遗](plan/第十八阶段/GRAM_第十八阶段_四实验完整预算与自动接续执行补遗v0.1.md)，已被最终收尾决定终止。原生 DiffGRM 的作者源码按 [source manifest](artifacts/phase18/diffgrm_native/sources/source_manifest.json) 中的 URL、commit 和文件哈希准备，源码下载目录不重复提交。已有 checkpoint、数据和模型缓存仍需在实验机器上准备；恢复配置中的 PID、GPU 和 checkpoint 路径对应记录时的本机运行。

数据文件沿用官方 [GRAM](https://github.com/skleee/GRAM) 仓库。当前本地 GRAM 基线提交与环境信息记录在各阶段的 `artifacts/*/environment/` 中。

部分旧阶段的 GPU 资源通过仓库内的 `tools/run_codellama.sh stop` 释放，实验退出后使用
`tools/run_codellama.sh start <GPU>` 恢复占用，使用 `tools/run_codellama.sh status` 查看当前
tmux、holder heartbeat、显存和日志。运行状态保存在当前磁盘的 `.runtime/codellama/`，不纳入 Git；
模型缓存仍复用 `/home/jiangtangyunzhi/hf_cache`，不复制大模型权重。具体实验命令和复现参数见对应
阶段计划与报告；第20阶段延续用户确认的按实际可用显存运行方式，不执行上述旧占位流程。
