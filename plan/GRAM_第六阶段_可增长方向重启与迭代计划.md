# GRAM 第六阶段：可增长方向重启与迭代计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Created: 2026-07-30
- Verification Status: PLANNED
- Version Label: `phase6_growth_search_v1`
- Development Domains: Toys、Beauty
- Confirmation Domain: Sports（继续封存）
- Execution Device: 当前为物理 GPU0；早期 v2/v3 使用物理 GPU6

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
- 当前及后续实验使用物理 GPU0，并保留日志、逐用户结果、配置和 checkpoint lineage；
- 不重复已经得到明确负结果的原配置；
- 组合实验必须包含各单组件对照，避免把增长错误归因给组合；
- 当前第一项实验为 **GACR-v2 小型增长迭代**，完成后再决定进入
  `CET-v1 + GACR-v2`，还是继续单独改进 GACR。

### 5.1 CodeLlama 资源协议

- 每个 GPU 实验启动前必须执行
  `/home/jiangtangyunzhi/projects/UnitTest/tools/run_codellama.sh stop`，释放
  CodeLlama 在 GPU0 上的资源；
- 等待 CUDA context 释放并确认 GPU0 满足该实验的空闲显存门后，才能启动实验；
- 实验无论成功、失败、超时、收到 `INT/TERM/HUP`，还是后处理失败，都必须在退出路径
  使用正确的 CodeLlama Hugging Face cache 在 GPU0 恢复 CodeLlama 资源；
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
  time、runner/workload PID、tmux session、物理 GPU0、日志路径、结果路径和
  CodeLlama reservation 状态；
- 启动后只向研究者返回准确的 `status` 查询命令，不在本轮对话中持续轮询长实验；
- 研究者会根据 `status` 的变化再次联系。除非研究者明确要求，不主动读取最终指标、
  不作科学结论、不启动后续实验；
- 研究者明确要求分析后，才核对运行完整性、读取结果、写 `report/`，并依据数据设计
  下一步；任何失败均不得自动重试。

## 6. GACR-v2 完成后的阶段更新（2026-07-31）

### 6.1 工程结论

- 正式实验 exit code = 0；Toys/Beauty 各 3 seeds、逐用户结果、残差 checkpoint、
  `summary.json` 和 GPU telemetry 均已完整生成；
- 终态 `failed_to_restore_resource` 只来自实验结束后的 CodeLlama 自动恢复检查失败，
  不代表科学计算失败；CodeLlama 后于 2026-07-31 19:52 在物理 GPU6 人工恢复；
- 2026-07-30 首次启动因缺失两个 ignored C1 checkpoint 而在候选生成前退出；经研究者
  授权后按锁定配置精确重建，SHA256 与历史值一致，未改变科学设计。

### 6.2 科学结论

- GACR-v2 选择的共享 scale 为 `1.0`，与 matched GACR-P0 完全相同，因此本次强度
  搜索没有产生 P0 以上的增量；该配置不再原样重复；
- Toys 三 seed mean overall NDCG@10 为 `-0.436%`，Beauty 为 `+2.893%`，六个
  域-seed cell 宏平均为 `+1.229%`；Beauty 3/3 正向，Toys 仅 1/3 正向；
- Toys changed-user 集合跨 seed 高度稳定，但三 seed 共同改变用户的平均 NDCG delta
  为负，说明主要问题是 Toys 上系统性排序伤害，而非随机 seed 不一致；
- GACR 方向保留，但在 Toys 安全问题解决前不进入 CET-v1 + GACR 组合。

### 6.3 当前下一步

当前唯一主实验调整为 **GACR-v3 target-free 残差安全衰减**：

- 固定 GACR-v2 的 parent checkpoint、候选构造、6 维特征、训练损失、训练步数、
  fit/calibration split、三个 seed 和 fresh development validation；
- 只增加一个部署因素：依据 target-free 排序置信度对残差作平滑衰减，不使用 target、
  validation label 或 Sports/test；
- calibration 必须逐域-seed满足 Recall@10 nondecrease、tail NDCG@10 nondecrease、
  broad harm <= 1%，并优先减少 Toys 的稳定 harm；
- 必须同时报告 matched P0、GACR-v2 identity control 与 GACR-v3，只有 GACR-v3
  改善 Toys 且保留 Beauty 信号，才进入 CET-v1 + GACR-v3。

## 7. GACR-v3 完成后的阶段更新（2026-07-31）

### 7.1 工程与完整性结论

- 科学计算于 2026-07-31 23:28 完成，Toys/Beauty 各 3 seeds、18 份逐用户结果、
  `summary.json` 和 GPU telemetry 均已生成；
- fit/calibration/fresh validation 用户隔离、parent checkpoint SHA 不变、残差状态与 v2 一致、
  target-free safety gate 均通过；Sports/test 未读取；
- 启动时的 session 名泄漏失败只属于无效首次尝试；有效实验产物完整。实验后 CodeLlama
  已重新加载 GPU6，资源恢复探针终态与科学结果分开记录。

### 7.2 科学结论

- Toys/Beauty 均选中最大 safety budget `0.4`，validation 的 attenuation rate 为 0、
  multiplier 为 1；GACR-v3 因此与 GACR-v2 identity control 完全一致；
- v3 相对 v2 的 overall NDCG@10、Recall@10 和 tail NDCG@10 增量在两域均为 0；
  **GACR-v3 作为当前最佳安全版本保留并冻结，只停止继续重复或调整无增量的衰减因素**；
- 在新 fresh cohort 上，冻结 GACR 的 Toys 三 seed mean overall NDCG@10 为 `+1.238%`，
  Beauty 为 `+4.470%`，六个域-seed cell 全部正向，宏平均 `+2.854%`；
- Beauty 三个 CI 下界为正；Toys 三个 CI 仍跨 0，但旧 cohort 的负向点估计未重现。
  该证据支持保留并冻结 GACR 主方向，不支持 v3 衰减机制的额外贡献。

### 7.3 当前下一步

当前唯一主实验进入 **CET-v1 × 冻结 GACR-v3 组合验证**：

- 冻结已验证的 GACR 配置，不再引入新的安全衰减参数；
- 以 CET-v1 checkpoint 替换原 GRAM checkpoint，其余候选、特征、训练和验证协议对齐；
- 必须同时比较 `GRAM`、`CET-v1`、`GACR-v3`、`CET-v1+GACR-v3`；
- 只有组合超过两个单组件，才保留组合；否则回到更强的单方法；
- Sports/test 继续封存。

详细预注册设计见 `plan/GRAM_第六阶段_CET-v1_GACR-v3组合实验计划.md`。

## 8. CET-v1 × GACR-v3 运行后阶段更新（2026-08-01）

### 8.1 工程与完整性结论

- 运行在预注册的 6 小时硬超时处以 exit `124` 终止；Toys 三个 seeds 的四组逐用户
  结果完整，Beauty 仅完成 GRAM arm `192/1024`，没有总 `summary.json`；
- 原状态 `failed_to_restore_resource` 同时记录科学工作负载超时和当时的 CodeLlama
  恢复探针失败；当前 CodeLlama tmux 已运行；
- Sports/test 未读取；不自动重试；
- 当前 runner 的串行双 backbone、双域评估无法稳定放进 6 小时门，后续应按域原子写出、
  支持只读 merge，并让 timeout 覆盖实测运行时间。

### 8.2 科学结论

- Toys 三 seed mean NDCG@10：GRAM=`0.070871`、CET-v1=`0.067475`、
  GACR-v3=`0.071839`、组合=`0.069886`；
- 组合相对 CET-v1 为 `+3.573%`，但相对 GACR-v3 为 `-2.718%`，相对 GRAM 为
  `-1.390%`；组合 3/3 seeds 均低于 GACR-v3；
- 组合相对 GRAM 的 Recall@10 与 broad-harm 门在 Toys 3/3 cells 通过，因此失败原因
  是 NDCG 互补性不足，不是安全门失败；
- 虽然 Beauty 不完整、不能计算双域宏平均，但预注册要求每个域都严格超过两个单组件，
  所以 Toys 已使组合保留门不可达。当前决定为
  **`REJECT_COMBINATION_RETURN_TO_GACR_V3`**。

### 8.3 当前下一步（研究者修正后）

- 保留冻结 GACR-v3 为当前 incumbent；不对 CET+GACR 做结果后调参；
- 组合失败只否定 CET backbone 与冻结 GACR residual 的互补性，不关闭已经跨 cohort
  出现增长的 GACR 方向；
- 下一主实验进入 **GACR-v4 target-free learned residual-application gate**：冻结 v3
  residual，只增加用户级收益门，针对 Toys tail/Recall@50 的局部负 cell；同时比较
  GRAM、冻结 GACR-v3 与 GACR-v4；
- v4 使用物理 GPU0；启动前停止 CodeLlama，退出后恢复并继续占用物理 GPU0；
- 若 v4 当前门控无增量，返回 v3 并继续依据诊断决定下一项 GACR 内部修改，不自动转向
  S0/CCRR；
- Sports/test 继续封存。

详细结果见
`report/第六阶段/GRAM_第六阶段_CET_v1_GACR_v3组合实验结果与验证报告.md`。

GACR-v4 预注册计划见
`plan/GRAM_第六阶段_GACR-v4目标无关收益门控实验计划.md`。

## 9. GACR-v4 完成后的阶段更新（2026-08-01）

### 9.1 工程与完整性结论

- 有效实验 exit=`0`，两域三 seed 的逐用户结果、gate checkpoints、summary 与 telemetry
  均完整；fit/calibration/fresh-validation 隔离、checkpoint lineage、target-free 和
  Sports/test 封存门全部通过；
- 首次 GPU0 cache 失败发生在科学计算前，并经研究者明确授权后重启；
- 实验后 CodeLlama 自动恢复失败源于 T5 与 CodeLlama `HF_HOME` 混用，已修复恢复脚本
  并在 GPU0 恢复为 `running`；科学结果不受影响。

### 9.2 科学结论

- Toys/Beauty 都选中 hard-gate threshold 0，v4 与 v3 在六个 cell 上精确一致；v4
  相对 v3 没有新增收益，停止当前 hard gate；
- 冻结 GACR 在又一个 fresh cohort 上保持 6/6 overall NDCG 正向点估计：Toys mean
  `+1.169%`、Beauty mean `+3.925%`、六 cell 宏平均 `+2.547%`；
- Beauty 三个 CI 下界为正；Toys 三个 CI 跨 0，并出现 mean overall/tail Recall@50
  `-0.130pp/-0.231pp`；GACR 方向继续，但 Toys safety 仍是主要瓶颈；
- 结果后 gate AUC 显示概率仍有信息，hard gating 的失败主要来自丢失改善用户，而不是
  完全无法区分改善与伤害。

### 9.3 当前下一步

当前唯一主实验计划为 **GACR-v5 target-free soft benefit weighting**：

- 冻结 v3 residual 和 v4 gate，不重新训练任何模型；
- 只将 hard apply/skip 改为 `alpha + (1-alpha)*p` 的连续 residual multiplier；
- `alpha=1` 是 v3 精确 identity；若校准或 fresh validation 没有严格增量，则停止整个
  gate/soft-weighting 因素并返回 v3；
- 主要保留门要求严格提高 Toys 与双域宏平均 NDCG，同时修复 Toys Recall@50；
- Sports/test 继续封存；设备与实验后 CodeLlama 均为物理 GPU0。

详细结果见
`report/第六阶段/GRAM_第六阶段_GACR_v4结果与验证报告.md`；详细计划见
`plan/GRAM_第六阶段_GACR-v5目标无关软收益加权实验计划.md`。

## 10. GACR-v5 完成后的阶段更新（2026-08-01）

### 10.1 工程与完整性结论

- 科学 workload 正常完成，exit=`0`；Toys/Beauty 各 3 seeds、每 cell 1024 用户，逐用户
  CSV、summary、log 和 telemetry 完整；
- 用户隔离、parent SHA、冻结 v3 residual/v4 gate、target-free soft weight 与 Sports/test
  封存检查全部通过；
- 实验后 CodeLlama 已按约定恢复并继续占用物理 GPU0。

### 10.2 科学结论

- Toys、Beauty 均选择 `alpha=1`，v5 与冻结 v3 在六个 cell 精确一致；v5 相对 v3
  的严格增量为 0，未通过保留门；
- v5 fresh cohort 上，冻结 GACR 相对 GRAM 的 Toys mean NDCG 为 `+3.092%`，Beauty
  为 `+2.122%`，六 cell 宏平均 `+2.607%`，6/6 点估计为正；
- 结合 v3/v4/v5 三批互斥 fresh cohort，冻结 GACR 累计 18/18 overall NDCG 正向点估计；
- 正式停止 residual 之后的 gate/attenuation/soft-weighting 修改族，但继续核心 GACR。

### 10.3 当前下一步

研究者确认当前 residual 仅使用小样本训练后，当前唯一主实验改为
**GACR-v6 全量 residual fit**：

- 保持 GRAM、候选、6 维特征、residual 架构、hinge loss、优化器、步数和部署强度不变；
- 只把每域 1024 个抽样 fit records 扩大为既有 fit split 的全部 records；
- 固定比较 GRAM、冻结 v3 与 v6，以严格 v6-v3 增量和双域安全门决定是否保留；
- 当前仅完成计划，尚未实现或启动；Sports/test 继续封存，设备与实验后 CodeLlama 均为
  物理 GPU0。

详细结果见 `report/第六阶段/GRAM_第六阶段_GACR_v5结果与验证报告.md`；详细计划见
`plan/GRAM_第六阶段_GACR-v6全量残差训练实验计划.md`。原指标对齐 loss 计划在实现前
被后移，不与训练规模因素混合。

## 11. GACR-v6 全量 fit 完成后的阶段更新（2026-08-02）

### 11.1 工程与完整性结论

- 原 full-fit 科学 workload 完成 6 个 residual checkpoint；validation-only recovery 随后完整写出
  两域 3 seeds 的 summary 与 12 份逐用户 CSV；runner 的后处理 Bash exit 在结果写完后发生，
  不影响科学产物；
- fit/calibration/fresh cohort 隔离、parent SHA、backbone zero-step 与 Sports/test 封存均通过；
- v6 runner 的显存治理经验已固化为 GPU0 30 GiB 总租约规则。

### 11.2 科学结论

- v6 相对 GRAM：Toys/Beauty mean overall NDCG@10 分别为 `+2.469%`/`+2.972%`，six-cell macro
  为 `+2.720%`，6/6 为正；相对 v3 macro 增量 `+0.559%`，5/6 为正；
- 但 Beauty mean tail NDCG@10 为 `-0.0181pp`、overall Recall@50 为 `-0.0977pp`、tail
  Recall@50 为 `-0.0679pp`（相对 v3），触发 v6 的预注册 safety 门；
- 决定为 **`KEEP_GACR_V3_FULL_FIT_SCALE_NOT_BENEFICIAL`**：停止只扩大 fit records 且保持
  hinge loss 的配置，保留 GACR 主线与 v3 incumbent。

### 11.3 当前下一步

唯一主实验为 **GACR-v7 全量指标对齐残差损失**：保持 v6 的全量 fit，仅替换为 NDCG@10/
Recall@50 截断敏感 pairwise loss，并把 Beauty tail/Recall@50 恢复设为硬保留门。详细方案见
`plan/GRAM_第六阶段_GACR-v7全量指标对齐残差训练实验计划.md`；详细结果见
`report/第六阶段/GRAM_第六阶段_GACR_v6全量残差训练结果与验证报告.md`。
