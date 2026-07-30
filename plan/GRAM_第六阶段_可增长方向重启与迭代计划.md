# GRAM 第六阶段：可增长方向重启与迭代计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Created: 2026-07-30
- Verification Status: PLANNED
- Version Label: `phase6_growth_search_v1`
- Development Domains: Toys、Beauty
- Confirmation Domain: Sports（继续封存）
- Execution Device: 物理 GPU6，`CUDA_VISIBLE_DEVICES=6`

## 1. 阶段目标

第六阶段的目标不是一次找到可以立即进行全量实验和论文定稿的完整方法，而是：

> **找到一个可重复、可解释、能够继续放大的推荐指标增长方向，并围绕它持续迭代。**

早期 pilot 的作用是提供方向信息。只要实现有效，并出现 overall、tail、特定数据域或
特定用户群的可信正向信号，就可以据此修改方法；没有达到最终论文收益要求，不等于
方向失败。

本阶段区分两种结论：

- **当前配置停止**：该参数、损失或融合方式不再原样重复；
- **方向关闭**：经过至少两次有依据的修改仍无正向推荐信号，或持续产生明显伤害。

除完整性错误和灾难性退化外，不再用一次 pilot 的固定收益阈值直接关闭整个方向。

## 2. 候选方向优先级

| 优先级 | 来源 | 方向 | 既有证据 | 第六阶段主要尝试 |
|---|---|---|---|---|
| P1 | 第四阶段 | **GACR 残差排序** | Toys/Beauty NDCG@10 分别 `+0.79%/+2.30%`，tail 同向，Recall 安全 | 优先修正 Toys 增长不足；扩大有效用户覆盖，尝试更平滑的残差权重或分群强度 |
| P2 | 第五阶段 | **CET-v1 证据一致性训练** | 双域 NDCG@10 `+0.82%/+1.00%`，宏平均 `+0.91%`，安全门通过 | 保留 v1，不重复单纯增大 beta 或原 Rank-R1；优先尝试与 GACR 组合，或提高真正会改变排序的样本覆盖 |
| P3 | 第三阶段 | **S0/S0b 邻域重排** | S0 Toys/Beauty `+1.72%/+0.47%`，Recall 均正 | 修复 uncovered 用户损失；比较共享参数、域内参数和保守触发式重排 |
| P4 | 第四阶段 | **CCRR 条件候选重排** | 校准集 Toys/Beauty `+6.31%/+11.15%`，但 Beauty tail `-3.79%` | 加入 tail-safe 约束或分群模型，然后在未参与拟合的新 cohort 上审计 |
| P5 | 第四阶段 | **FPUG 证据门控** | Toys overall/tail `+2.37%/+5.69%`，Beauty overall `-5.87%` | 不再使用统一 hard gate；尝试域自适应、soft gate，或只把门控分数作为 GACR/CCRR 特征 |
| P6 | 第四阶段 | **RPCD/FCRD 候选扩展** | 两域候选 Recall@50 约提升 3pp，overall NDCG 小幅正向，但 tail 受损 | 不单独做强融合；作为 GACR/CCRR 的候选供给，并增加 tail 保护 |
| P7 | 第三阶段 | **HBTR** | Toys NDCG@10 `+0.64%`，Beauty 轻微负向，显存开销高 | 仅在前述方向停滞后尝试轻量化、域条件权重 |
| P8 | 第四/三阶段 | **TCDR / MARC-CF** | 有机制或可预测性信号，尚无直接推荐收益 | 作为储备特征或辅助损失，不优先独立训练 |

第三、第四阶段的其余方向也纳入候选池，但不直接重启：GCDH 双头读出、CPGV、
PRPD、PENS、CPBD、SMBR、LEI、NLPL、CGI、FFNF、IALC、LNDR、SCDL、CPIA。
它们可以提供诊断或特征，但已有版本没有足够正向推荐证据，只有在高优先级方向需要
相应组件时才重新调用。

## 3. 尝试顺序

### A. GACR-v2：先放大最干净的双域增长

- 复现 GACR-P0 作为 matched baseline；
- 只围绕“有效用户覆盖”和“残差强度”做一个小型有界修改；
- 使用配对 development cohort 报告 overall/tail NDCG@10、Recall@10、changed-user
  coverage 和 broad harm；
- 若双域仍同向或至少宏平均进一步提高，继续在 GACR 内迭代。

### B. CET × GACR：组合训练收益与排序收益

- 以 CET-v1 checkpoint 代替原 GRAM checkpoint，加载同一套 GACR-v2；
- 同时比较 `GRAM`、`CET-v1`、`GACR-v2`、`CET-v1+GACR-v2`；
- 只有组合确实超过两个单独方法，才保留组合；否则回到表现更好的单方法。

### C. S0/CCRR：第二条重排路线

- 先做 tail/uncovered 安全修复；
- S0 与 CCRR 先分别小试，不立即叠加；
- 若其中一个在双域或稳定子群上明显优于 GACR，再替换主线或与候选扩展结合。

### D. FPUG 与候选扩展

- FPUG 先改为 soft/domain-aware 信号，不再直接删除证据；
- RPCD/FCRD 只负责扩大候选集，由当前最优 reranker 决定最终顺序；
- 这一步只在 A–C 暴露出候选覆盖瓶颈时执行。

### E. 储备方向

只有前四级方向连续没有可放大的增长时，才依次考虑轻量 HBTR、TCDR 辅助损失和
MARC collaborative-only。每次仍只改变一个核心因素。

## 4. 每轮如何决定下一步

每轮 pilot 完成后只回答四个问题：

1. 推荐指标是否出现正向点估计，增长发生在哪个域和哪些用户？
2. changed-user coverage 是否足够，还是效果被少数用户主导？
3. 主要损失来自 tail、uncovered、跨域迁移还是候选覆盖？
4. 下一次修改能否直接针对这个损失来源？

满足以下任一情况即可继续方向：

- 双域 overall 同向增长；
- 宏平均增长且另一域没有明显灾难性下降；
- 某一域或预先定义子群增长较强，并有明确、可测试的跨域修复方案；
- 推荐效果暂弱，但 changed-user coverage、候选覆盖或排序机制出现明确改善，可直接
  导出下一次修改。

置信区间、effect size 和多 seed 稳定性必须报告，但在探索期用于衡量证据强弱与设计
下一轮样本量，不作为“一票否决”。Sports 和 test 继续封存，直到某一方向的开发收益
达到研究者满意并冻结方法后再使用。

## 5. 执行原则与当前下一步

- 一次只推进一个主实验；实验结束后不自动分析、不自动写下一轮方案，等待研究者根据
  `status` 主动回来并明确要求分析；收到要求后再在 `report/` 写对应报告，并根据结果
  补充本计划；
- 所有实验使用物理 GPU6，并保留日志、逐用户结果、配置和 checkpoint lineage；
- 不重复已经得到明确负结果的原配置；
- 组合实验必须包含各单组件对照，避免把增长错误归因给组合；
- 当前第一项实验为 **GACR-v2 小型增长迭代**，完成后再决定进入
  `CET-v1 + GACR-v2`，还是继续单独改进 GACR。

### 5.1 CodeLlama 资源协议

- 每个 GPU 实验启动前必须执行
  `/home/jiangtangyunzhi/projects/UnitTest/tools/run_codellama.sh stop`，释放
  CodeLlama 在 GPU6 上的资源；
- 等待 CUDA context 释放并确认 GPU6 满足该实验的空闲显存门后，才能启动实验；
- 实验无论成功、失败、超时、收到 `INT/TERM/HUP`，还是后处理失败，都必须在退出路径
  立即执行 `/home/jiangtangyunzhi/projects/UnitTest/tools/run_codellama.sh start 6`
  恢复 CodeLlama 资源；
- runner 必须使用可靠的 `trap` 执行恢复，并将
  `releasing_resource / released_for_experiment / restoring_resource /
  restored / restore_failed` 写入状态记录；
- 不得因为恢复 CodeLlama 成功而覆盖实验本身的失败退出码；恢复失败也必须单独写入
  `status`，不得静默忽略。

### 5.2 后台运行与 status 协议

- 预计运行时间 **超过 20 分钟** 的实验必须使用具名持久 `tmux` 会话后台运行，
  不在对话调用中前台等待，也不依赖普通 `nohup`；
- 每个实验在 `experiment/phase6/` 下提供稳定的 runner 和状态文件，例如：
  `run_phase6_<experiment>.sh {start|status}` 与
  `experiment/phase6/<experiment>/status.json`；
- `status.json` 至少记录 experiment id、`starting/running/succeeded/failed/blocked/
  restoring_resource/failed_to_restore_resource`、当前 stage、reason、started/updated
  time、runner/workload PID、tmux session、物理 GPU6、日志路径、结果路径和
  CodeLlama reservation 状态；
- 启动后只向研究者返回准确的 `status` 查询命令，不在本轮对话中持续轮询长实验；
- 研究者会根据 `status` 的变化再次联系。除非研究者明确要求，不主动读取最终指标、
  不作科学结论、不启动后续实验；
- 研究者明确要求分析后，才核对运行完整性、读取结果、写 `report/`，并依据数据设计
  下一步；任何失败均不得自动重试。
