# GRAM 第五阶段：研究重置与证据一致性训练计划

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate + plan
- Created: 2026-07-29
- Verification Status: ANALYZED（M0）；PLANNED（CET 后续）
- Version Label: `phase5_reset_cet_v1`
- Upstream Audit: `artifacts/phase5/M0_search_process_audit.md`
- Development Domains: Toys、Beauty
- Confirmation Domain: Sports（继续封存）

## 0. 这次重置什么

第三、第四阶段最大的优点是证据链干净，最大的缺点是把研究流程做成了“高维串行
否决器”。第五阶段不降低完整性要求，但改变科学筛选方式：

1. 不再用一个 proxy deficit 是否超过任意阈值决定方法有没有资格训练；
2. 不再要求每个早期 pilot 同时证明 overall、tail、机制、校准、成本和跨域确认；
3. 不再一天换一个方向；本阶段只开发一个方法族；
4. 先用足够分辨率的真实 recommendation effect 决定去留，再用机制分析解释；
5. 论文目标从“必须找到绝对首创组件”改为“清楚的问题、可归因的方法、稳定效果和
   新的实证结论”。

## 1. 对过去失败的正式解释

过去不是二十个完整方法都训练失败，而是二十多个候选假设中，大多数在以下位置结束：

- 新颖性或 premise threshold；
- frozen checkpoint 的 teacher-forced proxy；
- 小样本 learnability/calibration；
- 双域多条件合取；
- 512-user 左右的低功效 effect gate。

真正改变可学习参数并读取 recommendation effect 的方法数量远少于方向数量。
因此第五阶段不继续堆 N0/N1 deficit audit，而进行一次有预算的 constructive method
development。

## 2. 唯一主方向：CET

方法暂名：

> **CET：Counterfactual Evidence-consistency Training**
> （反事实证据一致性训练）

### 2.1 研究问题

GRAM 的 coarse history passage 与多个 fine item passages 在 decoder 中共同融合。
既有 FPUG-N1 只说明“单独移除某个 passage 时 gold-path utility 有异质性”，但
FPUG-P0 进一步说明：在推理期预测并执行 per-user gate 会发生跨域迁移失败。

CET 不预测“该删哪一条”，也不改变推理输入。它把 passage variation 当作训练期
结构化扰动，要求模型在部分 detailed evidence 缺失时仍保持与 clean full-evidence
view 一致的合法 Trie-child 分布：

```text
clean view x:
  coarse passage + all current fine passages

perturbed view x~:
  coarse passage always kept
  newest fine passage always kept
  remaining fine passages independently masked with fixed probability q

p_t = softmax(z_clean[C_t] / tau)
r_t = softmax(z_perturbed[C_t] / tau)

L = CE_clean
  + alpha * CE_perturbed
  + beta * KL(stopgrad(p_t) || r_t)
```

`C_t` 是当前 gold prefix 下的合法 Trie children。正式推理只运行原始 clean view，
因此不增加推理参数、token、beam、reranker 或第二次解码。

### 2.2 为什么这不是 FPUG rescue

- FPUG 学的是 target-free deterministic gate，并在推理期改变 evidence；
- CET 不学习 gate，不使用 FPUG 的 Toys/Beauty validation 差异选动作；
- CET 的独立依据来自 consistency regularization、structured dropout 与 RAG
  context-perturbation training；
- CET 的目标是让同一个 generator 对合法 catalog continuations 在证据子采样下稳定，
  而不是预测某 passage 的正负 utility。

相关工作边界：

- [R-Drop](https://arxiv.org/abs/2106.14448) 已提出 dropout 子模型之间的输出一致性；
- [Passage-Mask](https://aclanthology.org/2022.emnlp-main.260/) 已研究 passage
  masking 作为 reader regularization；
- [CORD](https://aclanthology.org/2025.naacl-short.66/) 已研究 RAG context
  perturbation 下的一致性与 rank preservation；
- CET 不声称上述组件首创。候选新增点仅限于：
  **GRAM 类 multi-granular generative recommendation 中，clean-anchored structured
  passage subsampling 与 Trie-local legal-child consistency 的组合及其跨域机制证据。**

### 2.3 为什么它比继续做 selector 更有成功先验

1. 不需要 target-free features 准确预测每个 passage 的 utility sign；
2. 不在推理期做可能伤害 Beauty 的 hard routing；
3. clean CE 始终存在，正式输入与 baseline 完全相同；
4. consistency 是训练期正则，允许从所有样本学习，而不是只覆盖小 active subset；
5. 最小方法只有一个扰动机制和一个 Trie-local KL，容易做 matched ablation。

## 3. 新的证据门分层

### 3.1 Integrity gate

只判断执行是否有效：

- coarse passage 永不被 mask；
- newest fine passage永不被 mask；
- target 不进入扰动策略；
- clean view 与原 GRAM input 完全一致；
- `beta=alpha=0` 精确复现 matched CE；
- legal-child membership、finite、gradient、checkpoint reload 全通过；
- validation/test/Sports 不用于训练或配置选择。

Integrity 失败只能修实现，不能作科学 STOP。

### 3.2 Optimization gate

只判断 CET 是否真的被优化：

- 在 fit-disjoint training-prefix calibration 上，clean/perturbed legal-child
  symmetric KL 相对 matched C0 下降；
- clean lexical CE 不发生预注册的灾难性恶化；
- 不要求先达到某个“baseline defect prevalence”。

Optimization 失败才关闭当前实现。

### 3.3 Development effect gate

Toys/Beauty 是 development evidence，不再伪装成两个独立确认实验。主判据为：

1. 两域 NDCG@10 的等权平均相对变化 `>= +1%`；
2. 两域中至少一域 NDCG@10 `>= +1%`，另一域不得低于 `-0.5%`；
3. 两域 Recall@10 的绝对变化均不得低于 `-0.2pp`；
4. 合并的 tail NDCG@10 不得下降超过 `0.5% relative`；
5. broad-harm rate 相对 matched C0 增加不得超过 `0.5pp`。

不再要求两个数据集、overall、tail 的每个置信区间下界全部同时大于 0。置信区间继续
完整报告，用于决定证据强度和下一阶段样本量，不作为低功效 pilot 的连环否决器。

### 3.4 Confirmation gate

只有 development effect 通过才解封 Sports。Sports 使用冻结方法、超参数和 seed
规则；此时才要求：

- primary NDCG@10 提升的 95% CI 下界不低于 0；
- 至少 3 seeds 或由 pilot 方差得到的更大 seed 数；
- matched compute baseline 与关键消融完整；
- test 只在 checkpoint selection rule 冻结后读取一次。

## 4. M0：测量系统审计（已完成）

结果见 `artifacts/phase5/M0_search_process_audit.md`。最关键的定量结论是：

- FPUG Toys overall 在 512 users 上点估计 `+2.37% relative`，但只有 29 个用户
  NDCG@10 发生变化；
- 按观测配对差方差，约需 1,196 users 才能在相同点效应下使近似 95% 区间下界高于 0；
- 因此第五阶段 effect cohort 不再固定为 512。

M0 固定决定：
**`RESET_SEARCH_PROCESS_AND_START_ONE_METHOD_CYCLE`**。

## 5. CET 渐进实验

### C0：文献与代码边界

只做一次有界审查，不设“零重叠才通过”：

- 确认没有工作已经同时覆盖 GRAM multi-granular passages、structured evidence
  subsampling、gold-prefix legal-child consistency 和 clean-only inference；
- 若组件已有，收窄主张而不是自动停止；
- 只有整体问题、方法和实验协议均被实质覆盖才停止。

### C1：correctness smoke

- 每域 32 个 training-prefix users；
- 从 baseline checkpoint 出发，只更新 decoder 最后一层；
- 固定 `q=0.25, tau=1, alpha=1, beta=0.1`，不扫描；
- 5 optimizer steps；
- 检查 clean identity、mask localization、finite gradient、KL 下降和 reload。

C1 不读取 validation，不作效果结论。

### C2：高信息量 effect pilot

训练：

- C0：matched lexical-CE continuation；
- C1：CE clean + CE perturbed（控制 augmentation）；
- C2：完整 CET；
- 每域同一 baseline checkpoint、同一 25% training users、同一 batch/step；
- 5 epochs，seed 2023；
- 超参数只用 fit-disjoint training-prefix calibration 锁定；
- 不增加第四个候选，不做网格搜索。

评测：

- validation cohort 每域至少 2,048 users；若推理资源允许，直接使用完整 validation；
- cohort salt 在读取 candidate target 前冻结；
- baseline/candidate paired full-catalog Trie beam-50；
- 报告 overall、head/tail、history bins、broad harm、latency、显存与逐用户差；
- primary bootstrap 单位为 user，训练稳定性由后续 seeds 单独回答。

决定：

- 完整 CET 通过 3.3 节：`CET_FREEZE_FOR_CONFIRMATION`；
- CET 不优于 augmentation control：`STOP_CET_NO_CONSISTENCY_VALUE`；
- 两者都无效：`STOP_CET_NO_REGULARIZATION_EFFECT`；
- 工程无效：`INVALID_RUN_FIX_AND_EXACT_RERUN`。

### C3：机制消融

只在 C2 通过后执行：

1. 去掉 Trie-local restriction，改全词表 KL；
2. 去掉 clean anchor，只做双扰动一致性；
3. uniform token dropout control；
4. fixed oldest-passage dropout control。

论文机制必须由完整 CET 超过这些对照支持；否则把贡献降级为实证 regularization
结果，不夸大机制。

### C4：Sports 与多 seed 确认

- 方法、`q/tau/alpha/beta`、训练预算、checkpoint rule 全冻结；
- Sports 为主要确认域；
- seed 数由 C2 的 run-level 方差估计，最低 3；
- 通过后才运行 Toys/Beauty test，并按冻结配置回报所有正负结果。

## 6. 统计与报告原则

1. development 阶段看效应大小、方向一致性和伤害上界，不用 `p<.05` 代替判断；
2. 不把用户数当作训练重复；user bootstrap 与 seed variance 分开；
3. 不再用任意 subgroup 的轻微负值否掉 overall 有意义且安全的方法；
4. tail 是重要 secondary endpoint，但只有明确以 tail 为主张时才作为 primary；
5. 所有候选和失败保留，但第五阶段在 CET 关闭前不另起方法 M/N/O。

## 7. 资源与执行协议

- 默认物理 GPU3，`CUDA_VISIBLE_DEVICES=3`；
- GPU 任务继续使用 tmux、status JSON、5 秒 GPU telemetry；
- 启动前停止 CodeLlama reservation，退出路径恢复；
- 不静默 retry，不删除 invalid run；
- C1 预计低于 20 分钟；C2 每域/配置预算上限 8 GPU 小时；
- C2 总预算在启动前锁定，禁止因中间 loss 好看追加 epoch。

## 8. 当前状态

- M0：**完成**
- CET-C0：**`CET_C0_PASS_WITH_TRANSFER_NARROWING`**，机制说明见
  `artifacts/phase5/cet_c0/mechanism_brief.md`
- CET-C1：机器预注册已冻结为
  `artifacts/phase5/configs/cet_c1_preregistered.json`
- CET-C1 实现与预检：5/5 单元测试、真实双域 Collator/mask 集成检查、Toys
  真实模型 CPU forward/backward micro-smoke 均通过；记录见
  `artifacts/phase5/cet_c1/preflight_summary.json`
- CET-C1 GPU correctness smoke：**`CET_C1_CORRECTNESS_PASS`**。Toys/Beauty
  分别使用 32 个 training-prefix users 和 5 optimizer steps；Trie-local KL
  分别下降 5.79%/13.56%，clean CE 分别下降 8.62%/8.56%。clean replay、coarse/
  newest passage identity、target-free mask、zero-weight identity、legal-child、
  finite/nonzero gradient、source SHA、reload 与 validation/test/Sports exclusion
  全部通过。结果见 `artifacts/phase5/cet_c1/summary.json`。
- CET-C2 已冻结：机器预注册见
  `artifacts/phase5/configs/cet_c2_preregistered.json`；双域各 2,048 个配对
  validation users，split 文件与 user-set SHA 均已冻结；三臂固定为 C0 clean CE、
  C1 clean+perturbed CE、C2 完整 CET，5 epochs、seed 2023、无 checkpoint
  selection、无超参数扫描。
- CET-C2 初始工程运行：**`INVALID_RUN_FIX_AND_EXACT_RERUN`**。原因是 smoke 抽到
  不含可遮蔽旧 fine passage 的短历史，`masked_passages=0` 却错误标为 PASS。
  在读取 validation/test/Sports 前已终止，证据原样保存在
  `artifacts/phase5/cet_c2/invalid_run.md`。
- CET-C2 run2：修正版 smoke 强制覆盖遮蔽路径，Toys/Beauty 均实际 mask 5 个
  passages 且 finite/Trie competitive-step 检查通过；正式训练的冻结 `q=0.25`
  未改变。独立输出根为 `artifacts/phase5/cet_c2_run2`。
- 新计划：**已建立**
- 当前运行状态：**`CET_C2_RUN2_TRAINING`**
- 当前科学决定：**`CET_C1_CORRECTNESS_PASS`**
- 下一动作：完成 C2 六个训练臂后统一读取双域 validation，执行冻结 effect gate；
  C1 correctness 不构成 Recall/NDCG 改善证据。

本计划明确只押 CET 一个方法族。CET 未完成一次真实、高信息量 effect pilot 之前，
不再新增其他创新方向。
