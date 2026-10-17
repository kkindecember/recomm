# 第22阶段状态入口

2026-10-17 Git归档：本目录默认忽略生成产物，仅保留当前协议、固定配置、数据审计、环境版本、实现检查、来源哈希，以及调度迁移和故障接续的固定凭据。下文status、预测、训练曲线和实时报告仅在实验机器可用；最终结果需完成后检查再加入[显式保留清单](.gitignore)。本地参考代码副本`plan/第二十二阶段/claude/recomm-main/`和网页下载资源不纳入Git；本轮实现位于`experiment/phase22/`。

本轮为探索／重复拆分的最小全量validation验证，计划先于启动登记于
`plan/第二十二阶段/22.1_探索重复拆分最小全量验证_20261017.md`，
用户要求多卡并行后的调度补充为`22.2_多卡分片执行补充_20261017.md`。

- **当前主状态：`repeat_split_minimal_20261017_parallel_v2/status.json`**。旧`repeat_split_minimal_20261017_v1`在尚无新评估预测时主动迁移，标记SUPERSEDED。
- 分域状态：同目录 `Toys/status.json`、`Beauty/status.json`。
- `PREPARING` / `PREPARED`：准备；`WAITING_FOR_RESOURCES`：等待足够显存；`RUNNING`：后台运行；`SUMMARIZING`：生成完成、正在独立统计；`COMPLETED`：两域全量、校验、释放租约和汇总均完成；`INCOMPLETE` / `FAILED`：查看域状态和failure日志，科学失败不会自动重试。
- `rho.json`：训练侧一个全局ρ及完整拟合曲线；`training_components.npz`：拟合输入；`smoke.json`：GPU启动核验；`predictions.jsonl`：六臂逐用户推荐；`ranks.npz`、`summary.json`：完整排名和结果。
- `jobs/Toys_fit`、`jobs/Beauty_fit`分别拟合一次ρ，其后每域4个独立生成分片动态使用有空间的卡，最多8卡并行。各job下有独立`status.json`和GPU绑定。
- 2026-10-17 20:08检查发现`Beauty_s2`启动自检时因共享GPU显存耗尽退出。资源接续说明见`plan/第二十二阶段/22.3_Beauty显存故障接续记录_20261017.md`，新资源调度器接管原活跃worker，未中断它们。主status路径不变，`resource_recovery`字段与`resource_recovery_20261017/status.json`记录故障接续；原失败证据保存在该目录`before/`。
- 运行命令：`bash experiment/phase22/run_repeat_split_parallel.sh status`。中断后用户明确接续时可执行`resume`；运行中的任务由锁和tmux防止重复启动。

报告生成于 `report/第二十二阶段/探索重复拆分最小全量验证_多卡_20261017.md`。不会实时监看、读取test、自动启动下一轮或替换旧阶段产物。
