# GRAM 单卡复现实验

本仓库保存 GRAM Beauty/Toys 单卡复现所需的代码修改、命令脚本、实验编排、计划、审计记录和结果报告。

## 仓库内容

- `GRAM/src/`：在官方 GRAM 基础上的最小代码修改；
- `GRAM/command/`：Beauty/Toys 单卡训练、smoke 与最佳 checkpoint 测试入口；
- `experiment/phase1/`、`experiment/phase2_toys/`：后台运行与资源恢复脚本；
- `experiment/phase18/`：DIFF→GRAM、DiffGRM 候选实现、诊断、测试和 checkpoint 接续；
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

当前四条候选的预算与接续约定见[第十八阶段执行补遗](plan/第十八阶段/GRAM_第十八阶段_四实验完整预算与自动接续执行补遗v0.1.md)。原生 DiffGRM 的作者源码按 [source manifest](artifacts/phase18/diffgrm_native/sources/source_manifest.json) 中的 URL、commit 和文件哈希准备，源码下载目录不重复提交。已有 checkpoint、数据和模型缓存仍需在实验机器上准备；恢复配置中的 PID、GPU 和 checkpoint 路径对应记录时的本机运行。

数据文件沿用官方 [GRAM](https://github.com/skleee/GRAM) 仓库。当前本地 GRAM 基线提交与环境信息记录在各阶段的 `artifacts/*/environment/` 中。

GPU 实验统一通过仓库内的 `tools/run_codellama.sh stop` 释放资源，实验退出后使用
`tools/run_codellama.sh start <GPU>` 恢复占用，使用 `tools/run_codellama.sh status` 查看当前
tmux、holder heartbeat、显存和日志。运行状态保存在当前磁盘的 `.runtime/codellama/`，不纳入 Git；
模型缓存仍复用 `/home/jiangtangyunzhi/hf_cache`，不复制大模型权重。具体实验命令和复现参数见对应
阶段计划与报告。
