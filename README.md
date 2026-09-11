# GRAM 单卡复现实验

本仓库保存 GRAM Beauty/Toys 单卡复现所需的代码修改、命令脚本、实验编排、计划、审计记录和结果报告。

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

当前第20阶段安排见[总计划](plan/第二十阶段/GRAM_第二十阶段_提点实验总计划.md)、[全项目复盘与选型](report/第二十阶段/全项目实验复盘与第20阶段选型报告.md)和[实验总报告](report/第二十阶段/GRAM_第二十阶段_实验总报告.md)。S-DPO已于2026-09-11按用户要求停止；SPRec继续运行，另并行评估已保存的round1_sft及GACR-v6+PCRF组合。实验机器上的总报告每30秒更新，查看命令和字段说明见[状态查看说明](artifacts/phase20/README.md)。旧`artifacts/phase20/overview/status.json`只覆盖原SPRec/S-DPO，不包含新增评估。

第20阶段提交实现、固定配置、计划报告、汇总审计和模型来源记录；模型权重、日志、逐用户预测、候选特征、重复源码快照和进程状态保留在本机。Git中的自动报告是提交时的快照，本机后续更新不会自动推送；正式评估的最终`summary.json`已在忽略规则中预留，生成后可随结论一并提交。

第18、19阶段已于2026-09-10按用户决定收尾，分别见[第18阶段报告](report/第十八阶段/GRAM_第十八阶段_实验收尾总报告.md)和[第19阶段报告](report/第十九阶段/GRAM_第十九阶段_实验进展与结论总报告.md)。旧状态文件不再代表仍有运行任务。RaSeRec参考源码的保留范围见[第三方代码说明](third_party/README.md)。

历史四条候选的预算与接续约定见[第十八阶段执行补遗](plan/第十八阶段/GRAM_第十八阶段_四实验完整预算与自动接续执行补遗v0.1.md)，已被最终收尾决定终止。原生 DiffGRM 的作者源码按 [source manifest](artifacts/phase18/diffgrm_native/sources/source_manifest.json) 中的 URL、commit 和文件哈希准备，源码下载目录不重复提交。已有 checkpoint、数据和模型缓存仍需在实验机器上准备；恢复配置中的 PID、GPU 和 checkpoint 路径对应记录时的本机运行。

数据文件沿用官方 [GRAM](https://github.com/skleee/GRAM) 仓库。当前本地 GRAM 基线提交与环境信息记录在各阶段的 `artifacts/*/environment/` 中。

部分旧阶段的 GPU 资源通过仓库内的 `tools/run_codellama.sh stop` 释放，实验退出后使用
`tools/run_codellama.sh start <GPU>` 恢复占用，使用 `tools/run_codellama.sh status` 查看当前
tmux、holder heartbeat、显存和日志。运行状态保存在当前磁盘的 `.runtime/codellama/`，不纳入 Git；
模型缓存仍复用 `/home/jiangtangyunzhi/hf_cache`，不复制大模型权重。具体实验命令和复现参数见对应
阶段计划与报告；第20阶段延续用户确认的按实际可用显存运行方式，不执行上述旧占位流程。
