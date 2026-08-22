# GRAM 第十五阶段：统一 Cold Adaptation 基线复现与条件式方法开发计划 v0.1

> **建立日期**：2026-08-21
> **修订日期**：2026-08-22（v0.1-r6）
> **当前状态**：`PLAN_REVISED / S15_0_COMPLETE / NATIVE_SANITY_NON_BLOCKING / S15_2_CONTRACT_PASS / B2_PASS_S15_3A_ITEM_DISJOINT_ADMISSION / B3_FAIL_S15_3A_EDIT_STATE_ADMISSION_WITH_EXPLORATORY_BRANCHING_RECOVERY / S15_3B_B0_B1_B2_RUNNING / TEST_NOT_OPENED / PRIOR_ATTEMPTS_PRESERVED`
> **阶段定位**：先复现并对齐 SpecGR / GenRecEdit，再依据同协议证据决定是否开发新方法
> **上一阶段**：Stage14 R2PD 在 M2 14-1 得到 `FAIL_STOP_PATH_TRANSFER_STAGE14_1`，14-2/M3/M4 已取消

---

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-08-21
- Verification Status: PARTIALLY_VERIFIED（S15-3A B2 512-event admission 已完成并 PASS；B3 edit-state admission FAIL；S15-3B full validation 与官方原生 GPU 闭环尚未执行）
- Version Label: phase15_cold_adaptation_plan_v0.1-r6

---

## 0. 执行摘要

第十五阶段不再从一个未经比较的新机制起步，而是先回答一个更基础、可证伪的问题：

> **在冻结相同 GRAM backbone、hierarchical lexical ID、cold50 split、candidate/beam budget 和 strict item evaluator 后，推理期 retrieval、参数 editing 与外部 R² portfolio 哪一种干预位置能提供可重复的 cold gain，并以什么 warm/cost 代价实现？**

本阶段依次执行：

1. 冻结官方源码、依赖、artifact 与许可状态；
2. 将 Video Games 限定为官方代码/机制的最小 native sanity，不把其权重或数字带入 GRAM 主表；
3. 以已冻结的 Toys/Beauty GRAM v0 为各域共同起点，只训练或求解方法特有状态；
4. SpecGR-GRAM 每域只训练 drafter/projection/index，GenRecEdit-GRAM 每域只生成 covariance/edit requests/deltaW；
5. 先完成 Toys contract、admission 与 validation，再以 Toys 冻结的 adapter/超参数在 Beauty 做双域确认；
6. 即使 Toys efficacy 为负，只要 contract 成立也必须运行 Beauty seed-0；只有机制/安全 contract 失败才禁止迁移；
7. 只有双域同协议结果明确暴露“可改进且未被已有方法覆盖”的瓶颈，才另写方法子计划。

**本计划不预注册一个必做的新方法。** `R²-guided contextual editing` 只是 GenRecEdit 通过后的一条条件候选，不是本阶段默认任务。R2PD/A3 已归档，不换名字复活。

---

## 1. 为什么是这个分支

### 1.1 已有本地证据

- Stage13 R² 证明 warm-only content resolver 能恢复部分 cold candidate reachability，但外部 portfolio 存在 warm trade-off，teacher Recall@50 约 11%，候选池仍是显著天花板。
- Stage14 14-0B 证明 GRAM hierarchical lexical cold path 存在 learned failure，且 R² prefix prior 相对固定 prior 有信息。
- Stage14 14-0C 证明冻结 GRAM likelihood verifier 会保留 R² cold H@50，却显著压低 cold top-10 placement；简单 verifier 不能自动解决问题。
- Stage14 M2 证明 soft subtree distillation 未显著优于 hard-path CE：A2−A1 item-level paired-bootstrap MRR CI=`[−0.000779,+0.000869]`。继续调 R2PD 不再符合预注册纪律。

### 1.2 最近邻方法边界

| 方法 | 官方机制 | 本地状态 | Stage15 角色 |
|---|---|---|---|
| SpecGR | inductive drafter + GR verifier + guided re-drafting；另有 self-drafting 变体 | 源码完整；官方预训练 quick-start artifact 缺失；允许从头训练 | 主表使用冻结 GRAM verifier + 域内 drafter/projection/index；整套 TIGER 重训只作附表 |
| GenRecEdit | context-to-next-token、position-wise SID edit + warm preservation + One-One trigger | 源码完整；官方 LFS 尚未物化；processed cache 未构建 | 主表使用冻结 GRAM base + 域内 covariance/edit requests/deltaW |
| R² portfolio@2 | domain-local resolver + warm-anchored external insertion | 已有双域 validation 结果 | 当前本地强基线/外部干预基准 |
| GRAM v0 | 标准 native beam | 已冻结 | reference |
| R2PD | soft subtree distillation + retention | M2 主 Gate FAIL | 历史负结果，不进入 Stage15 arm |

官方来源：

- SpecGR：[AAAI 2026 论文](https://ojs.aaai.org/index.php/AAAI/article/view/38486)；[官方代码](https://github.com/Jamesding000/SpecGR)
- GenRecEdit：[论文](https://arxiv.org/abs/2603.14259)；[官方代码](https://github.com/Starrylay/GenRecEdit)
- ColdGenRec：[SIGIR 2026 论文](https://arxiv.org/abs/2603.29845)；论文给出的代码入口为 `https://anonymous.4open.science/r/ColdGenrec-0DEC`，S15-0 当前环境访问重定向 API 后返回 HTTP 401；原计划记录的 GitHub 地址不存在

ColdGenRec 已覆盖“统一 cold protocol 下隔离 backbone/identifier/training strategy”的一般复现问题。因此 Stage15 不能只写成“又一个 cold benchmark”；可检验增量必须限定为：

> **同一冻结 GRAM + 同一 hierarchical lexical ID 下，隔离 intervention location（external portfolio / retrieval-verification / parameter editing），并同时比较 strict cold quality、warm cost 与本地 update/inference cost。**

---

## 2. 研究问题与假设

### RQ1：官方方法能否形成可审计闭环？

SpecGR 与 GenRecEdit 的公开源码、依赖、checkpoint、数据和运行入口，能否在不修改算法的前提下完成最小官方原生 sanity？该问题与 Toys/Beauty 主结果解耦，不再作为 GRAM port 的阻塞 Gate。

- H1a：SpecGR 在官方预训练 artifact 缺失时，允许按官方源码从头生成 Video Games 数据、semantic ID 和 checkpoint；必须标为 `TRAIN_FROM_SCRATCH_NATIVE_SANITY`，不得称官方 checkpoint reproduction。
- H1b：GenRecEdit 可使用官方 LFS 或从头训练 checkpoint 形成 request preparation→edit→evaluation 闭环；LFS 未物化必须明确记录。

### RQ2：能否在不改变机制语义的前提下适配 GRAM？

- H2a：SpecGR 的 drafter、guided re-drafting 与 verifier acceptance 可映射到 GRAM variable-length hierarchical lexical path。
- H2b：GenRecEdit 的 SID position、edit layer、token-level update 与 trigger 可映射到 GRAM variable-length path，并保持 strict item-level 可反解。

### RQ3：哪种干预位置在同协议下有效？

对 GRAM v0、R² portfolio@2、SpecGR-GRAM 与 GenRecEdit-GRAM 做配对比较，确定 cold gain、warm cost、overall utility、update/inference cost 和额外状态之间的 Pareto 关系。

### RQ4：是否存在值得开发的新方法缺口？

只有 baseline 结果明确显示某个可定位瓶颈，且改进不等价于 SpecGR/GenRecEdit/R² 的已有机制时，才进入条件式方法开发。

---

## 3. 范围与非目标

### 本阶段范围

- 官方原生最小 sanity（Video Games，仅验证实现链路，非主表）；
- TIGER/RQ-VAE SID → GRAM hierarchical lexical ID 的显式接口适配；
- frequency cold50 的 Toys/Beauty validation；
- 域内 method-specific artifact：SpecGR drafter/projection/index 与 GenRecEdit covariance/edit requests/deltaW；
- strict item evaluator、collision hard-fail；
- cold/warm/overall quality和部署成本并列报告；
- 先 seed-0，过 Gate 后才讨论 seed expansion。

### 明确不做

- 不恢复 R2PD、A3、旧 allocator、resolver 继续训练或事后调参；
- 不把 R² drafter 版本冒充官方 SpecGR；
- 不把自行简化的 edit 版本命名为 GenRecEdit；
- 不引用官方论文数字冒充 GRAM 本地 baseline；
- 不在官方复现阶段同时修改算法；
- 不打开 Toys/Beauty test；
- 不在 Toys contract/admission 与 adapter/超参数冻结前启动 Beauty；Toys efficacy 负结果本身不取消 Beauty seed-0；
- 不在 baseline 定位前预付 full training、多 seed 或第三域预算；
- 不因为“代码能跑”就宣称复现论文结论。

---

## 4. 统一协议冻结

### 4.1 数据与评测

| 项 | 冻结口径 |
|---|---|
| 主开发域 | Toys_cold50 |
| 冻结确认域 | Beauty_cold50；Toys contract/admission PASS 且配置冻结后执行，不以 Toys efficacy PASS 为前提 |
| split | 项目现有 frequency cold50；不替换为官方 temporal split |
| 数据语义 | catalog-known、metadata-available、zero-interaction item cold-start simulation |
| development | train + validation only |
| test | 全程封存；任何 `test_read=true` 直接 kill |
| item evaluator | collision-hard-fail multimap；ambiguous/unknown/duplicate output 必须显式计数 |
| top-K | H@10/H@50、NDCG@10；candidate method 最大候选评分预算 50，native method beam=50 |
| 配对单位 | user/event 为推荐指标主单位；涉及重复 target item 的机制诊断另报 item-level bootstrap |
| bootstrap | paired 10,000 resamples，95% CI；seed 在 artifact 固定 |

### 4.2 冻结 backbone 与 ID

- GRAM v0 checkpoint、item path、item metadata、split 和 evaluator 均使用 Phase14 已审计版本与 SHA256。
- Stage15 不重建 lexical ID，不把 identifier 改动和 adaptation 方法混在一起。
- SpecGR/GenRecEdit 的官方 TIGER/RQ-VAE 结果只用于原生复现；进入 GRAM 比较后，所有 arm 使用同一个 GRAM backbone 与 path catalog。
- GenRecEdit variable-length path 必须明确 `position_map`、EOS/padding 处理、edit layer 选择和 trigger 语义；不得静默截断到固定长度。

各域冻结 base：

| 域 | GRAM v0 checkpoint | SHA256 |
|---|---|---|
| Toys_cold50 | `artifacts/phase13/explore/v0_toys/gram_logs/Toys_cold50/0_20260808_2256/id_0_rec_30/model_rec_phase_1_epoch_30.pt` | `d71fcf5a09057a6cda22d1f682b036b9174742d3c78a92a52ec0921dd3048550` |
| Beauty_cold50 | `artifacts/phase13/explore/v0_beauty/gram_logs/Beauty_cold50/0_20260810_1004/id_0_rec_30/model_rec_phase_1_epoch_30.pt` | `6e5d43f2a3d51f02b351314bad31bff00249cee536731f105215883c022c27b8` |

### 4.2.1 Cold adaptation 输入与防泄漏合约

- 现有 `user_sequence.txt` 每行同时包含 train、validation target（倒数第二项）和 test target（末项）；现有 GRAM indexing 会先读取整行再在 dataset 层切片，因此仅设置 `cf0_phase9=1` 或“不构建 test loader”**不足以**满足 `test_read=false`。
- 在任何 Stage15 adapter/training/evaluation job 前，必须由独立审计步骤逐行生成 `user_sequence_train_validation.txt`：保留 user id 与原序列的 `[:-1]`，明确丢弃原末项；记录原文件 SHA、输出 SHA、行数、最短序列检查和 `test_target_retained=false`。该审计步骤只允许执行固定的末项剥离，不得统计、输出或用于决策原末项；完成后所有 Stage15 model/dataset/indexing 只允许打开投影文件，禁止再打开原 `user_sequence.txt`。
- 投影后的训练切片为 `projected_items[:-1]`，validation target 为 `projected_items[-1]`；必须用 synthetic fixture 证明与历史 `train=original[:-2]`、`validation=original[-2]` 完全一致，并证明原 `original[-1]` 不存在于投影字节流的对应行尾。
- cold item universe 只来自阶段开始前已冻结的 `cold_split_meta/cold_items.txt`；允许知道 catalog membership、item metadata 和冻结 GRAM path，不允许读取 validation/test 中该 item 是否出现、出现次数或对应用户历史。
- SpecGR drafter 可使用 train interaction、全部 catalog metadata 和 train-derived content/index；不得使用 validation/test `history → target` 监督。GRAM verifier checkpoint 全程冻结。
- GenRecEdit covariance 只由 train histories 估计；edit request 对**全部冻结 cold catalog items**生成，context 只能来自 train warm histories 与 metadata/content 相似性，不得先扫描 validation/test target item 集合再决定编辑哪些 item。
- `similar_item_sasrec.txt` 在证明其构建只使用 train interaction 前不得作为 SpecGR drafter 或 GenRecEdit request 的输入；主线默认使用冻结 item metadata/BGE content embedding 构建域内相似性。历史 cold split manifest 只作为已冻结 membership 使用，不重新读取其 dropped-user/test 诊断来做方法选择。
- validation 只用于 Toys 的预注册超参数选择和 efficacy 评估；不得反向改写 edit requests、cold universe、exclusion rule 或 path catalog。
- Beauty 复用 Toys 冻结的算法、超参数选择规则、position/layer 规则和 budget；只允许重建 Beauty 域内 catalog/index/covariance/edit requests/deltaW，不在 Beauty 上重新调参。
- 任何读取项目 test、从 validation/test 事件构造训练/编辑监督、或按 validation target occurrence 筛选 cold item 的行为均令该 run `KILLED_TARGET_LEAKAGE`，不得进入正式表。

### 4.3 arm 语义

| ID | 方法 | 训练/更新 | 推理状态 |
|---|---|---|---|
| B0 | GRAM v0 | 无 | 标准 beam=50 |
| B1 | R² portfolio@2 | 复用冻结 resolver | 外部候选插入；明确额外 resolver 状态 |
| B2 | SpecGR-GRAM official-mechanism port | 每域训练 drafter/projection/index；对应域 GRAM v0 frozen | drafter + frozen GRAM verifier + guided re-drafting |
| B2d | R²-as-drafter diagnostic | 无新训练 | 只作机制诊断，**不得称真 SpecGR** |
| B3 | GenRecEdit-GRAM official-mechanism port | 从对应域 GRAM v0 求解 position-wise edit；不另训 base | `GRAM_v0 + covariance + edit_requests + deltaW + trigger`，额外状态必须披露 |

整套 `TIGER + SpecGR` 或 `TIGER + GenRecEdit` 在 Toys/Beauty 的从头训练若执行，只能作为 `native_reimplementation` 附表；它改变 backbone/identifier，不能与 B0/B1/B2/B3 的 same-backbone 主表混算。

same-backbone verifier 已在 Stage14 14-0C 完成，不重复作为新 arm；需要时只复用已有结果作为 B2 的简化下界。

### 4.4 成本字段

每个 arm 必须记录：

- offline preparation/update wall time；
- GPU-hours、peak GPU memory、CPU memory；
- inference wall time、users/s、candidate/model forward 次数；
- 额外训练参数量、更新参数量、checkpoint/delta 大小；
- drafter、covariance、deltaW、trigger/index 等额外运行状态；
- 每 100/500/全部 cold items 的 batch sensitivity（只在对应方法进入正式 cost stage 后执行）。

不同硬件/协议的论文报告值单列为 `reported_external`，不得与 `local_measured` 混算。

---

## 5. 分阶段执行计划

### S15-0：Source、artifact 与协议冻结（CPU，只读优先）

**目标**：在任何 GPU job 前确定官方源码到底能否复现，以及移植边界是什么。

任务：

1. 获取 SpecGR、GenRecEdit、必要的 ColdGenRec 官方仓库；记录 remote、commit SHA、branch、license、submodule/LFS 状态。
2. 生成文件清单、缺失 artifact 清单、checkpoint/data 下载需求、依赖冲突和预计磁盘占用。
3. 分别建立隔离环境方案；不得为了兼容而污染现有 `gram-repro` 环境。
4. 审计官方入口是否硬编码 test、数据路径、固定 SID 长度、GPU 数量或 evaluator。
5. 写出 `official_native` 与 `gram_port` 两套配置边界，禁止交叉使用结果。

产物：

```text
experiment/phase15/protocol/source_compatibility_audit.py
experiment/phase15/configs/stage15_s0_sources.json
artifacts/phase15/s0_source_freeze/{source_manifest,artifact_inventory,dependency_matrix,compatibility_matrix}.json
report/第十五阶段/Stage15_S0_源码与协议冻结报告.md
```

**主 Gate**：每个方法都必须得到一个可审计状态：

- `READY_OFFICIAL_NATIVE_RUN`
- `SOURCE_READY_ARTIFACT_BLOCKED`
- `SOURCE_OR_LICENSE_BLOCKED`
- `FAIL_PORT_SEMANTICS_UNRESOLVED`

单个方法 blocked 不自动阻塞另一个；两者都 blocked 才停止 Stage15。禁止用自行补写算法把 blocked 伪装成官方复现。

**S15-0 完成记录（2026-08-21）**：

| 对象 | commit | 状态 | 证据摘要 |
|---|---|---|---|
| SpecGR | `f0ded8884b1df97b5f0599d4ec300bb20b5d1eff` | `SOURCE_READY_ARTIFACT_BLOCKED` | HEAD/历史均无 README 声称的 Video Games dataset、semantic ID、model/embedding checkpoint；无 LFS pointer |
| GenRecEdit | `e6878d9c7c6e57479e840ccb8c045b11a2bd69b5` | `READY_OFFICIAL_NATIVE_RUN` | 官方 LFS 共 517.68 MiB；Video Games 最小已知三对象为 107.11 MiB，processed cache 仍需单独构建 |
| ColdGenRec | — | `SOURCE_OR_LICENSE_BLOCKED` | 论文 4open 入口当前 401；只作协议参考，不阻塞目标方法 |

S15-0 的状态是对“现成官方 artifact 能否直接 quick-start”的历史裁决，不等于源码不可用。经 2026-08-21 用户确认，SpecGR 允许从头训练，因此其执行状态修订为 `PRETRAINED_QUICKSTART_BLOCKED / TRAIN_FROM_SCRATCH_AVAILABLE`；GenRecEdit 的 `READY_OFFICIAL_NATIVE_RUN` 解释为 `READY_FOR_LFS_MATERIALIZATION_AND_PREFLIGHT`，不是当前已经可运行。未使用 GPU、未下载模型权重、未读取项目 test。详见 `report/第十五阶段/Stage15_S0_源码与协议冻结报告.md`。

### S15-1：官方原生最小 sanity（非阻塞支线）

**目的**：验证官方代码和机制链路，不验证 GRAM 质量，不作为 Toys/Beauty same-backbone port 的前置阻塞 Gate。

#### S15-1A SpecGR

- 官方预训练 artifact 当前缺失；允许按官方源码从头生成 Video Games 数据、semantic ID 和 checkpoint；
- 先做 CPU import/entry/data-schema preflight，再决定是否值得支付完整 Video Games 训练成本；
- 若执行，完成最小 train-from-scratch→inference→evaluation；
- 核对 drafter output、guided re-drafting、acceptance/verifier、最终 item metric；
- 结果必须标为 `TRAIN_FROM_SCRATCH_NATIVE_SANITY`；若没有冻结 expected metric，只能记 `RUNNABLE_WITHOUT_NUMERIC_REPRO_TARGET`，不能称论文数字 reproduced。

#### S15-1B GenRecEdit

- Video Games 只用于最小官方实现 sanity；先确认 LFS checkpoint/cache/covariance 是否完整；
- 运行最小 `train/checkpoint（或官方 checkpoint）→ prepare edit request → edit → evaluate`；
- 核对 position-wise update、warm preservation、One-One trigger 和 delta artifact；
- 不完整 artifact 必须保留缺失清单，不自动下载来源不明的镜像。

产物：

```text
artifacts/phase15/s1_official_native/specgr/
artifacts/phase15/s1_official_native/genrecedit/
report/第十五阶段/Stage15_S1_官方原生复现报告.md
```

**支线 Gate**：完成输入→官方算法→metric 闭环时记 `PASS_NATIVE_RUNNABLE`。运行时修改仅限路径/设备/已确认的入口 bug；任何算法修改都使该结果降级为 `PORT_PROTOTYPE`。支线 blocked 必须如实记录，但不阻塞 S15-2。

**执行顺序**：native sanity 不与主线 GPU job 并行占卡。Video Games 不做完整双方法、多 seed 或论文数字追逐；优先保证 Toys/Beauty same-backbone 主线。

### S15-2：Toys/Beauty GRAM adapter、域内状态与 contract smoke

**目的**：只验证语义等价和接口正确，不做效果选择。

#### S15-2P：CPU-only 双域输入与权重 preflight

- 核验两个冻结 GRAM checkpoint、historical config、path catalog、item metadata、cold/warm manifest 和既有 B0/B1 产物的存在性与 SHA256；
- 静态审计 monolithic `user_sequence.txt` 的 GRAM loader 读取路径；在 adapter job 前生成并验证去掉每行原末项的 `user_sequence_train_validation.txt`，生成后禁止 Stage15 job 打开原序列；
- 静态证明除上述固定末项剥离步骤外，preflight 不打开任何 test 内容，生成 allowlist/open-file contract；
- 冻结 Toys/Beauty 各自的 SpecGR drafter artifact 路径与 GenRecEdit covariance/edit-request/deltaW 路径；
- 统计 train/validation/cold catalog 规模、variable path length 和 lexical vocabulary，只读 train/validation/cold manifest；
- 在任何 GPU job 前冻结首个 smoke 的 exact command、预计时长、显存与后台 `status.json`。

共同 adapter contract：

```text
history + catalog + metadata + frozen GRAM
    -> method-specific adaptation/drafting
    -> ranked unique catalog item ids
    -> strict item evaluator
```

SpecGR port 最少覆盖：

- 每域 auxiliary content drafter/projection/index 的训练输入与 GRAM path catalog 映射；
- 对应域 GRAM v0 verifier 参数逐位冻结，禁止被 optimizer 注册；
- guided re-drafting 的 lexical-prefix 定义；
- verifier score/acceptance 与 variable-length path；
- candidate budget 和模型 forward 次数；
- official mechanism port 与 R²-as-drafter diagnostic 分名输出。

GenRecEdit port 最少覆盖：

- 从对应域同一 GRAM v0 出发，只保存/加载 covariance、edit requests 与 deltaW；
- variable-length `position_map`；
- context→next-token edit request；
- edit layer probe/selection；
- iterative token edit、covariance、deltaW；
- trigger 是否改变标准 beam，额外状态如何加载；
- 未编辑 prompt parity 与 warm preservation contract。
- edit request 必须覆盖冻结 cold catalog universe，不能由 validation/test occurrence list 反向筛选。

测试必须包含：

- split/test guard；
- target leakage guard；
- deterministic tie break；
- candidate/item exact 去重；
- collision/unknown hard-fail；
- variable path length/EOS/padding；
- candidate budget exactness；
- B0 parity；
- 未编辑样本 parity；
- official/diagnostic 命名隔离。
- monolithic-sequence redaction parity：投影训练/validation 与历史切片逐事件一致，且 Stage15 job 的 open-file manifest 不含原 `user_sequence.txt`。

**主 Gate**：train+validation 投影与 synthetic parity 先 PASS；固定 16–64 users smoke 中，B0 与冻结 v0 逐位 parity；B2/B3 全部输出合法唯一 item，`test_read=false`，无 target leakage，官方机制字段完整；冻结 GRAM 参数在 SpecGR 训练前后 hash 不变，GenRecEdit 未应用 delta 前与 B0 逐位 parity。smoke 指标不用于 efficacy。

产物与单一报告：

```text
experiment/phase15/protocol/{common_adapter,specgr_gram_adapter,genrecedit_gram_adapter}.py
experiment/phase15/tests/
artifacts/phase15/s2_contract_smoke/
report/第十五阶段/Stage15_S2_GRAM适配与ContractSmoke报告.md
```

### S15-3：Toys validation seed-0 正式同协议比较

#### S15-3A 固定规模 admission

先使用 Stage14 item-disjoint pseudo-cold 数据做 512-event admission，仅检查：

- 完整 beam/candidate 路径；
- finite score/loss/update；
- edit/drafter 确实改变预期对象；
- 显存、runtime、artifact schema；
- 未读取 held target 进行训练/选参。

admission 不判显著性；通过后才做 full validation。

#### S15-3B Toys full validation

> **执行更新（2026-08-22）**：B2 已通过 512-event admission，B3 已在 edit-state admission 失败。当前 S15-3B 只允许 B0、B1、B2；B3 记 `FAIL_B3_S15_3A_EDIT_STATE_ADMISSION`，不生成伪造的 efficacy 数字。B2d 只在机制附表，不参与“真 SpecGR”主表。

> **B3 exploratory recovery（2026-08-22）**：失败根因定位为 deterministic request sampler 在 position 4 选中 4/4 个 legal branching factor=1 的结构确定前缀；其 legal probability 恒为 1，不可能满足预注册的“edited probability 严格高于 baseline”条件。recovery 只允许在冻结 catalog trie 上排除 branching factor=1 的不可编辑请求，再沿用原 SHA rank、distinct-cold、4 requests/position、seed、layer probe、z optimizer 与 probability threshold=0.3。该修复不得改变正在运行的 B0/B1/B2 S15-3B；只有新的独立 512-event admission 完整 PASS 后，B3 才可另行进入 full validation。

主要报告：

```text
cold_H@50, cold_NDCG@10,
warm_NDCG@10, overall_NDCG@10,
hit_events, unique_target_items,
paired bootstrap CI,
update/inference cost fields
```

定义：

```text
G_c(M) = cold_H@50(M) - cold_H@50(B0)
C_w(M) = warm_NDCG@10(B0) - warm_NDCG@10(M)
```

成功标签：

1. `PASS_NATIVE_COLD_RECOVERY`：`cold_H@50(M) - cold_H@50(B0)` paired CI 下界 > 0；
2. `PASS_OVER_R2_PARETO`：满足 1，且满足以下之一：
   - cold H@50 对 B1 的 paired CI 下界 >0，warm NDCG@10 point estimate 不低于 B1；
   - warm NDCG@10 对 B1 的 paired CI 下界 >0，cold H@50 point estimate 不低于 B1；
3. `PASS_COST_QUALITY_CANDIDATE`：满足 1，且在质量、update/inference cost、额外状态三轴上不被 B1 严格支配；成本只作 promotion 分类，不替代质量主 Gate。

**主 Gate**：为每个 contract-pass arm 分配 Toys efficacy 标签并冻结其 adapter、超参数、budget、数据构造和 metric hash。Toys efficacy 不决定是否进入 Beauty；只要机制/安全 contract 成立，B2/B3 均进入 Beauty seed-0。只有 `PASS_OVER_R2_PARETO` 才足以把相应干预位置作为优先方法方向。

**stop**：两者均未达到 native recovery 时，停止在 Toys 上调参或发明第三个 adaptation 机制；仍按冻结配置完成 Beauty seed-0，以获得完整双域负/混合证据。

单一报告：`report/第十五阶段/Stage15_S3_Toys统一协议正式结果.md`。

### S15-4：Beauty 冻结双域确认（contract-pass arm 必做）

运行 Toys 通过机制/安全 contract 的 B2/B3，另带 B0/B1；不要求 Toys efficacy PASS。全部 adapter、超参数选择规则、budget、evaluator 和 cost 字段从 Toys 冻结；只允许域内 SpecGR drafter/projection/index 与 GenRecEdit catalog/covariance/edit requests/deltaW 重建。

**主 Gate**：在 Beauty 达到 `PASS_NATIVE_COLD_RECOVERY`；若 Toys 为 `PASS_OVER_R2_PARETO`，Beauty 至少不得被 B1 在 cold/warm 两轴同时严格支配。

失败时写 mixed/negative evidence，不回 Toys 调参。Beauty 不重新选择超参数。单一报告：`report/第十五阶段/Stage15_S4_Beauty冻结确认报告.md`。

### S15-5：条件式方法选择（本计划不自动执行）

| 同协议结果 | 裁决 |
|---|---|
| GenRecEdit-GRAM 达 `PASS_OVER_R2_PARETO` | 可新建 `R²-guided contextual editing` 子计划：只研究 R² 选择 context/failure depth 是否改善 sparse edit；必须与 GenRecEdit 主 arm直接比较 |
| SpecGR-GRAM 达 `PASS_OVER_R2_PARETO`，GenRecEdit 不达 | 先做 failure decomposition；只有 hierarchical lexical guided re-drafting 存在明确未覆盖缺口时才立方法，不把换 backbone 当创新 |
| 两者都达 | 比较 retrieval 与 editing 的 Pareto/成本；优先选择可定位且未被已有工作覆盖的瓶颈 |
| 只有 native recovery、均未超越 R² | 可做统一干预位置的复现/分析论文定位；不宣称新 SOTA，不自动开发方法 |
| 双域完成后两者都不达 native recovery | `STOP_GRAM_COLD_ADAPTATION_METHOD_BRANCH`；收束 Stage13–15 负结果与边界，不再在相同验证集造机制 |

任何 S15-5 新方法必须另建 plan、重新做创新性审计、重新冻结 Gate 和资源；本 v0.1 不构成执行授权。

---

## 6. 统计与证据纪律

- cold H@50 是 primary reachability；warm NDCG@10 是 primary cost；overall NDCG@10 是 secondary utility。
- 不用 H@10 的少数事件单独决定路线。
- 不以 point estimate 代替 paired CI；不把“不显著”写成“等价”。
- B2/B3 同时比较产生的多个结论必须完整报告；不只展示获胜 arm。
- 失败的官方原生复现、port、smoke 和正式尝试均保留 artifact；不覆盖、不静默 retry。
- 工程 retry 不单独建 report；每个 S15 阶段只写一份 report，阶段结束时合并试错摘要。
- test 在所有模型、超参数、seed、exclusion rules 和 metric code hash 冻结前不得打开。
- Beauty 是冻结确认域但仍属于已使用 source domain；不能称独立终验。
- 官方数字、官方原生复现和本地 GRAM port 三者必须在表格中明确区分。
- `native_reimplementation`、`SpecGR-GRAM`、`GenRecEdit-GRAM` 三类名称严格隔离；不同 backbone/identifier 的结果不得进入 same-backbone 主表。

---

## 7. 资源、后台与状态规则

### 7.1 用户执行规则（硬约束）

- 预计超过 10 分钟的实验必须后台运行；不实时监看。
- 每个后台实验在 `artifacts/phase15/<stage>/<run>/status.json` 记录状态、PID、stage、started/updated time、GPU、exit code、自动 retry=false、test_read、log/summary 路径。
- 小 GPU 实验先查询当前空闲卡；资源不足时只报告所需显存，由用户指定 GPU。
- 不停止、不挪动其他用户进程。
- GPU5 当前项目 holder 约 20 GiB；若经用户允许临时释放用于实验，任务无论成功、失败、超时或正常终止都必须用同一控制器恢复原约 20 GiB，占位恢复写入 artifact status。
- 不触碰 GPU0 或其他既有 holder，除非用户对该次任务明确授权。

### 7.2 初步资源分层

| Stage | 主要资源 | 当前授权 |
|---|---|---|
| S15-0 source freeze | CPU/磁盘/网络，原则上无 GPU | 已完成；0 GPU；第三方源码仅本地保存，权重未下载 |
| S15-1 native sanity | CPU/网络/磁盘；GPU 仅在主线不占卡时 | 非阻塞；不优先启动完整 Video Games 训练 |
| S15-2P dual-domain preflight | CPU，只读为主；投影生成是唯一受控写入 | PASS；双域末项剥离、synthetic parity、输出 manifest 与 test guard 已完成 |
| S15-2 adapter smoke | CPU + 小 GPU，预计单卡 | PASS；B0 parity、B2/B3 input contract、attempt-4 deterministic hook/probe、B2 drafter state 与 B3 edit state 均已完成；35/35 tests PASS，历史失败/blocked attempts 原样保留 |
| S15-3A Toys admission | 单卡；实测后再定 full | S15-2 contract PASS 后授权 |
| S15-3B Toys full | 由 S15-2 telemetry 定版 | S15-3A PASS 后授权 |
| S15-4 Beauty | Toys contract/admission PASS 且配置冻结后 | contract-pass B2/B3 必做；首次启动前重新检查资源 |
| S15-5 新方法 | 另立计划和预算 | 禁止自动执行 |

用户于 2026-08-21 已明确授权“修改 plan 并开始实验”；该授权允许按上述 Gate 连续推进，但不取消每阶段 preflight、空闲卡检查、后台状态、test 封存和不干扰既有进程规则。在 S15-2P 前不承诺 GPU-hours；adapter、依赖和运行模式未核实前给出训练预算属于伪精确。

---

## 8. 目录与 artifact 合约

```text
experiment/phase15/
  configs/
  protocol/
  tests/
  run_stage15_*.sh

artifacts/phase15/
  s0_source_freeze/
  s1_official_native/{specgr,genrecedit}/
  s2_dual_domain_preflight/
  s2_contract_smoke/{toys,beauty}/
  s3_toys/{admission,formal}/
  s4_beauty/formal/

report/第十五阶段/
  Stage15_S0_源码与协议冻结报告.md
  Stage15_S1_官方原生复现报告.md
  Stage15_S2_GRAM适配与ContractSmoke报告.md
  Stage15_S3_Toys统一协议正式结果.md
  Stage15_S4_Beauty冻结确认报告.md
```

每个正式 run 至少包含：

```text
status.json
config.json
summary.json
data_provenance.json
input_file_sha256.json
open_file_manifest.json
resource_summary.json
```

本地第三方源码、官方 checkpoint、LFS blob、dataset cache、模型权重、predictions、per-user dump、日志和 telemetry 不提交 Git；只提交 adapter 源码、测试、冻结配置、聚合 summary/provenance、阶段 report 和 plan 更新。

---

## 9. 风险与止损

| 风险 | 处理 |
|---|---|
| 官方 artifact/LFS 不完整 | 明确标 `ARTIFACT_BLOCKED`；不从非官方镜像拼凑“复现” |
| 官方依赖与 GRAM 冲突 | 隔离环境；不原地升级 `gram-repro` |
| SpecGR port 退化为 R² + verifier | 保留 B2/B2d 命名隔离；guided re-drafting/acceptance 缺失则不得称 SpecGR |
| GenRecEdit fixed SID position 无法映射 variable path | S15-2 `FAIL_PORT_SEMANTICS_UNRESOLVED`，不自行改成新算法后仍沿用名称 |
| GenRecEdit edit request 使用 validation/test target occurrence | 立即 `KILLED_TARGET_LEAKAGE`；改用预冻结全量 cold catalog + train-only context，失败 run 保留且不静默 retry |
| SpecGR 训练误更新冻结 GRAM | optimizer 参数 allowlist + 训练前后 GRAM state hash；不一致即 contract FAIL |
| GRAM loader 即使 validation-only 仍把 test 末项读进内存 | 先生成审计 train+validation 投影；Stage15 运行时原序列 denylist；投影/历史切片 synthetic parity 未过则禁止启动 GPU |
| 小样本事件稀疏 | smoke 不判 efficacy；正式 Gate 使用 full validation paired CI |
| 结果被 R² 轻量基线支配 | 如实停止；不以“机制更复杂”作为继续理由 |
| benchmark 与 ColdGenRec 重叠 | 贡献只限同一冻结 GRAM 下的 intervention-location + cost 隔离；证据不足则不包装成新贡献 |
| 再次出现 positive arm 但非预注册 primary 胜出 | 作为诊断记录；新主张必须另立计划，不能事后换 Gate |
| 资源消耗失控 | 每一阶段独立 admission；未通过前不预付下阶段 GPU |

---

## 10. 时间顺序与完成定义

| 顺序 | Stage | 最早完成定义 |
|---:|---|---|
| 1 | S15-0 | 两个官方方法均有 source/artifact/compatibility 状态和一份报告 |
| 2 | S15-2P | 双域 base/data/leakage/artifact 路径冻结；train+validation 投影 Gate PASS；首个 exact smoke command 与资源合约完成 |
| 3 | S15-2 | Toys adapters contract smoke PASS；Beauty 仅做 frozen-domain schema/contract 确认；报告完成 |
| 4 | S15-3A | B2 512-event Toys admission PASS；B3 edit-state admission FAIL，均不作 efficacy 结论 |
| 5 | S15-3B | Toys full validation 标签 + 配置冻结 + cost 初测 + 单一 S15-3 报告 |
| 6 | S15-4 | 所有 Toys contract-pass B2/B3 的 Beauty seed-0 冻结确认，不以 Toys efficacy 为进入条件 |
| 7 | S15-1 | native sanity 可在不与主线争用资源时完成或保持 blocked；不改变主表 Gate |
| 8 | S15-5 | 用户根据双域分支表选择；必须另写 plan |

阶段结束后更新本计划顶部状态、对应 Stage 小节、资源实测和决策记录。过程中的多次工程尝试只写进该阶段最终 report 的“试错摘要”，不创建重复报告。

---

## 11. 当前唯一下一步

**B3 冻结为 `FAIL_B3_S15_3A_EDIT_STATE_ADMISSION`，不再以放宽 threshold、增加 request、修改 tie-break 或换 seed 救援。等待用户确认独立 B0+B2 admission；exact command 为 `bash experiment/phase15/run_stage15_s3a_toys_b2_only_admission.sh start 7`。两个 B3 失败 attempt 原样保留，不自动重试。**

S15-0 已完成；native Video Games sanity 改为非阻塞支线。双域静态 preflight 和 train+validation 投影 Gate 已 PASS；Toys B0 projection-parity 在 GPU4 attempt-3 上 16/16 用户 beam-50 逐位一致。B2/B3 真实 contract-input CPU Gate 已 PASS；GPU hook/probe 合约与完整 Stage15 CPU 测试已 27/27 PASS。当前主线：

- Toys/Beauty `user_sequence_train_validation.txt` 已生成并通过审计；后续 model/adapter job denylist 原 `user_sequence.txt`；
- 沿用已在 B0 smoke 验证的 train/validation、catalog metadata、cold manifest 输入 allowlist；禁止 adapter/model job 读取任何项目 test 内容；
- 全量 cold catalog 的 SpecGR index / GenRecEdit pseudo-context 已由冻结 BGE content embedding 与 train-only warm occurrence 构造完成：5,963/5,963 cold items、59,630 pseudo-context、302,400 position-wise requests；未使用 validation/test occurrence/history/target；
- GRAM token-likelihood hook 与 train-only decoder layer probe 已实现；26/26 CPU 合约测试 PASS，tokenizer 静态核对 11,924/11,924 catalog paths 的 lexical segment 数与 GRAM token 数一致；
- attempt-1 已获确认并启动，但 probe transition lookup 错把截断后的 `history_item_ids` 长度作为 chronological key，在模型加载前因长历史 `KeyError` 失败；GPU forward=0，未训练、未读 test、未自动 retry，artifact 原样保留；
- 修复后用 attempt-1 真实 view 核验 855/855 unique chronological keys、64/64 selected transitions target match，覆盖最长历史 224；attempt-2 完成 16/16 verifier users 与 16/16 probe batches，512/512 score finite、6 positions × 6 layers 完整、GRAM hash 不变，但 TF32 下 chunk-8 与 direct batch-1 最大差 `8.63075e-4`，超过冻结 `2e-5`，故科学 Gate FAIL；
- attempt-3 保持同一数据/样本/threshold/容差并启用 deterministic algorithms，但在首个完整 encoder forward 前由 PyTorch guard 拒绝：CUDA ≥10.2 还要求进程启动前设置 `CUBLAS_WORKSPACE_CONFIG`；未完成 user forward、未训练、未读 test；
- attempt-4 按已确认 exact command 完成：16/16 verifier users、512/512 score finite、8/8 direct acceptance 一致，hook/direct 最大误差 `4.29153e-6 < 2e-5`；64 个 train-only transitions 覆盖 6 positions × 6 layers，selected layers=`[5,5,5,5,5,4]`，GRAM hash 不变，未读 test；运行 40.73 s，peak CUDA allocated 1,061.59 MiB；
- verifier/probe Gate PASS 后不再重复该 smoke；B2 drafter state 已 PASS，B3 covariance/edit requests/deltaW/trigger state 已完成实现与 CPU/真实输入 dry preflight，当前只等待执行冻结的 B3 GPU exact command。
- B2 auxiliary content drafter 已实现：冻结 BGE catalog vector → trainable projection → 2-layer causal Transformer history encoder → normalized full-catalog retrieval；cold item 只通过 content vector 进入 retrieval，绝不进入 train label；新增 optimizer allowlist、warm-label guard、finite/tie/unique 合约后 Stage15 tests 累计 30/30 PASS；
- 首条 B2 state smoke 仅是机制/状态合约，不是 efficacy 或正式 Toys drafter：4,096 条 SHA 固定 train-only transitions、2 epochs、batch 128；16 个 target-independent validation histories 只验证 top-50 输出；admission=`8,192 MiB`、预计增量上界=`4,096 MiB`、hard timeout=`1,800 s`；
- GPU2 attempt-1 在启动确认后因资源快照漂移而 admission blocked：preflight 30/30 tests PASS，但 worker 读取 used/free=`44,682/3,889 MiB`，rc=9；训练 workload 未启动，未改动 PID 3143585 或其他进程，未读 test、未自动重试，独立 artifact 原样保留；
- 用户指定 GPU1 后以独立 attempt-2 执行并得到 `PASS_B2_TRAIN_ONLY_DRAFTER_STATE_SMOKE`：启动前 free=`19,232 MiB`，2 epoch loss=`[9.36830384,9.24912384]` 且 finite，drafter state hash 改变、probe 最大绝对 score change=`4.78616`，16×top-50 输出均为唯一已知 item；trainable parameters=`1,346,912`，运行 `11.88 s`，peak CUDA allocated=`229.47 MiB`，state=`5,395,153 bytes`、SHA256=`8e6ceb801be0bbbfe035f1402eb6e49b40f6a65d9dd5ac9ea2e6099dc2adcb2f`；
- attempt-2 未把 validation target 用于训练、采样或选模，未打开原 `user_sequence.txt` 或 test；GRAM checkpoint 未作为模型加载、未注册 optimizer，SHA256 前后均为 `d71fcf5a09057a6cda22d1f682b036b9174742d3c78a92a52ec0921dd3048550`；B2 状态 Gate 已关闭，不再重复运行。
- B3 clean-room state smoke 已实现：全量物化 5,963/5,963 cold catalog 的 302,400 条 position-wise requests；固定 256 条 SHA-ranked train-only transitions 求 position-specific second moment，其中预留 32 条最长路径以覆盖 position 5；每位置固定 4 个互异 cold item，共 24 条 request 执行 z-residual 优化，再按 `ΔW = RKᵀ(λC + KKᵀ)⁻¹` 求 6 个 full-shape position bundle；
- B3 冻结 position→layer=`[5,5,5,5,5,4]`、`covariance_ridge=0.01`、`preservation_lambda=10000`、z steps=`30`、z lr=`0.5`、legal probability threshold=`0.3`；One-One trigger 只激活当前 lexical position，EOS/padding 必须 inactive，临时参数 materialization 后 base logits 与 model hash 必须 exact restore；
- B3 新增 full-universe、确定性互异 request sample、lexical legal-child、最长路径 covariance admission 和 reverse-history prompt 合约，Stage15 tests 累计 35/35 PASS；真实只读 dry preflight 得到 requests by position=`[59630,59630,59630,59630,59630,4250]`、256/256 covariance targets 为 warm train-only、32 条覆盖最长路径；
- 首条 B3 state smoke 冻结到 GPU1：快照 used/free=`29,622/18,948 MiB`，保留既有 PID 1170034、3470445，不做资源修改；admission=`8,192 MiB`、预计增量上界=`6,144 MiB`、hard timeout=`3,600 s`、automatic retry=false；
- B3 attempt-1 已按确认的 exact command `bash experiment/phase15/run_stage15_s2_toys_b3_edit_state_smoke.sh start 1` 执行；35/35 worker preflight PASS，但 GPU1 admission 时 free 已降至约 `7,820 MiB < 8,192 MiB`，worker rc=9；科学 workload 未启动，未物化 request/covariance/deltaW，未加载 GRAM、未读 test、未改既有进程、未自动重试；
- attempt-1 独立输出 `artifacts/phase15/s2_contract_smoke/toys/b3_edit_state_smoke` 原样保留；GPU7 attempt-2 已按冻结 exact command `bash experiment/phase15/run_stage15_s2_toys_b3_edit_state_smoke_attempt2.sh start 7` 完成并得到 `PASS_B3_TRAIN_ONLY_EDIT_STATE_SMOKE`：35/35 worker preflight PASS，256 条 train-only covariance transitions 覆盖各 lexical position（position 5 为 32 条最长路径），24 条固定 request 形成 6 个 full-shape delta bundle；delta finite/nonzero、6/6 position trigger 均改变输出、EOS/padding inactive、未编辑 prompt exact parity；冻结 GRAM model hash 与 checkpoint SHA 前后不变；运行 `55.72 s`，peak CUDA allocated=`2,649.24 MiB`，artifact=`196,733,573 bytes`；未读 validation supervision、原 `user_sequence.txt`、SASRec 相似项或 test，automatic retry=false。
- S15-3A 入口与 One-One cached beam generation hook 已实现，Stage15 tests 38/38 PASS；真实 CPU preflight 核对 4,096 train-only transitions、11,924 catalog、1,176 pseudo-cold、5,963 real-cold、4,785 retained warm，Stage14 clean base 136/136 state keys strict load，historical v0 checkpoint 不用于 admission；
- S15-3A attempt-1 在 GPU7 通过 16,384 MiB admission 后启动：B2 loss=`[9.36309439,9.24616081]`，1,176/1,176 pseudo-cold 的 train-only BGE contexts 与完整 requests 已构建，256 条 covariance 完成；positions 0–3 z-success=`[2,1,2,1]/4`，position 4 为 `0/4`，worker rc=1；held events 尚未打开、beam evaluation 未启动、test 未读、未自动 retry；
- attempt-2 已按确认命令在 GPU7 执行：clean-base train-only probe 完成，selected layers=`[5,5,3,5,0,0]`；positions 4/5 在全部 6 层 token accuracy 均为 0，position 4 的固定 4 requests 再次 `0/4`，worker rc=1；held events 仍未打开、beam evaluation 未启动、test 未读、未自动 retry；先前“仅为 layer-map 迁移错误”的假设被证伪；
- B3 admission 结论冻结为 `FAIL_B3_S15_3A_EDIT_STATE_ADMISSION`：不放宽 legal probability threshold=`0.3`，不增加 requests、不修改浅层 tie-break、不换 seed。B2 独立 admission 已完成 512/512 events，完整 B0 beam 与 B2 draft→verify→redraft 路径、finite/unique top-50、base hash 不变、防泄漏与 test 封存 checks 均 PASS；workload rc=0，runtime=`3,154.16 s`，peak CUDA allocated=`6,935.35 MiB`，B2 rankings 与 B0 在 164/512 events 不同。
- B2 workload 原始 verdict 因 reducer 对 `held target used=false`、`test opened=false` 直接执行 `all(values)` 而被错误写为 FAIL；该确定性布尔方向 bug 已修为正向安全断言并补回归测试，基于完整 artifact 重算为 `PASS_S15_3A_B2_ITEM_DISJOINT_ADMISSION`。原始 verdict 与修正理由保留在 summary；不重跑模型，不把 admission 的 B0/B2 指标解释为 efficacy。

---

## 12. 决策记录

| 日期 | 决策 | 理由 |
|---|---|---|
| 2026-08-21 | Stage14 R2PD 主线停止 | M2 A2−A1 primary CI 跨 0；不调参/换 seed rescue |
| 2026-08-21 | Stage15 采用“先复现定位、再条件开发方法” | 避免第三次先造机制再寻找问题；SpecGR/GenRecEdit 分别覆盖 retrieval 与 editing |
| 2026-08-21 | SpecGR 先于 GenRecEdit（已由后续双域主线决策取代） | 原判断基于官方 artifact 预期；保留为历史记录 |
| 2026-08-21 | R² portfolio@2 保留为本地强基线 | 它是当前已验证的轻量 cold reachability endpoint，复杂方法必须证明不被其支配 |
| 2026-08-21 | R2PD/A3 不进入 Stage15 | 预注册主 Gate 已失败；A3 点估计不能事后替代成功标准 |
| 2026-08-21 | 方法创新推迟到 S15-5 | 只有同协议 baseline 暴露明确缺口后才能定义新方法与 Gate |
| 2026-08-21 | S15-0 完成；SpecGR official native 暂停（后续改为非阻塞 sanity） | 当前官方 commit 不含 README 声称的 Video Games artifacts，且无 LFS/release path；不从非官方镜像拼接复现 |
| 2026-08-21 | S15-1 下一步曾转 GenRecEdit selective preflight（已由 S15-2P 取代） | GenRecEdit 官方 LFS 指针可审计；保留为修订前历史记录 |
| 2026-08-21 | 两个官方仓库均按无 license 处理 | 本地研究审计可保留 commit-pinned clone；不复制/再分发实现代码，GRAM adapter 依据论文与本地 contract 独立实现 |
| 2026-08-21 | 用户允许 SpecGR 从头训练 | SpecGR 状态改为 `PRETRAINED_QUICKSTART_BLOCKED / TRAIN_FROM_SCRATCH_AVAILABLE`；缺官方权重不再等价于缺源码或主线 blocked |
| 2026-08-21 | Video Games 降为非阻塞 native sanity | 主结果必须来自 Toys/Beauty；官方 Video Games 权重与 metric 不进入 same-backbone 主表 |
| 2026-08-21 | 主表统一使用域内冻结 GRAM v0 | SpecGR 只训练 drafter/projection/index；GenRecEdit 只生成 covariance/edit requests/deltaW，避免重训不同 backbone 造成混杂 |
| 2026-08-21 | Beauty 改为 contract-pass arm 必做 | Toys efficacy 负结果不取消 Beauty seed-0，避免单域筛选；只有机制/安全 contract 失败才停止该 arm 迁移 |
| 2026-08-21 | 冻结 GenRecEdit 防泄漏构造 | edit 全量预冻结 cold catalog；covariance/context 只来自 train，禁止按 validation/test occurrence 选 target |
| 2026-08-21 | 用户授权按修订计划开始实验 | 立即启动 S15-2P CPU preflight；后续 GPU 仍须逐阶段满足资源、后台状态、test guard 与 Gate |
| 2026-08-21 | S15-2P 静态双域 preflight 完成；GPU 暂不启动 | checkpoint、catalog、metadata、cold/warm、B0/B1 validation 产物齐全且 checkpoint 接口一致；但现有 GRAM indexing 会读取含 test 末项的完整序列，必须先通过 train+validation 投影 Gate |
| 2026-08-21 | train+validation 投影工具合约测试 5/5 PASS | 工具只机械删除每行末项且不记录/聚合该值；短序列、重复用户、覆盖已有输出和路径逃逸均 hard-fail；真实双域投影需在 exact command 确认后执行 |
| 2026-08-21 | 双域 train+validation 投影 Gate PASS | 用户确认 exact command 后执行 exit 0；Toys 8,789 行、Beauty 10,655 行；审计记录 test target 未 materialize/log/aggregate/use，后续模型禁止打开原序列 |
| 2026-08-21 | 首个 Toys B0 projection-parity smoke 代码与合约测试完成 | 16 个用户按 `sha256(1502:user_id)` 选择，不使用 target；累计 Stage15 合约测试 9/9 PASS；启动命令冻结为 GPU4 后台运行，启动时重新做 12 GiB admission，资源查询失败或不足即 blocked |
| 2026-08-21 | 首个 smoke 的指定设备由 GPU4 改为 GPU3 | 用户指出 GPU4 剩余空间不足并明确指定 GPU3；exact command 更新为 `bash experiment/phase15/run_stage15_s2_toys_b0_projection_parity.sh start 3`，仍保持 12 GiB admission 与不自动换卡规则 |
| 2026-08-21 | GPU3 smoke 启动在 admission 阶段 blocked | tmux 沙箱权限经用户批准后只在沙箱外补齐同一 worker 启动；GPU3 空闲显存低于 12,288 MiB，worker rc=9，模型未加载、0 user forward、未自动换卡或重试 |
| 2026-08-21 | 用户确认 GPU3 attempt-2，但再次在 admission blocked | attempt-1 原样保留，attempt-2 使用独立目录；从确认到沙箱外启动获批期间 GPU3 占用再次上升，16:48 worker rc=9；随后快照为 used 44,665 / free 3,905 MiB、util 100%；未授权 attempt-3 |
| 2026-08-21 | 用户指定 GPU4 执行 attempt-3 | 启动前 GPU4 used 34,284 / free 14,287 MiB、util 82%，通过 12,288 MiB 静态门槛；attempt-3 使用独立目录并在 worker 内重新 admission，不覆盖前两次记录 |
| 2026-08-21 | GPU4 attempt-3 完成，Toys B0 projection-parity PASS | 16/16 个 target-independent sampled validation users 的 beam-50 结果逐位一致，workload rc=0；运行 40.31 s，进程峰值 CUDA allocated 5,055.17 MiB；未训练，未打开原完整序列或 test 产物，无自动 retry，GPU3 两次 blocked artifact 保留 |
| 2026-08-21 | B2/B3 clean-room adapter contract 与真实输入构造器完成 | 完整 Stage15 CPU 测试 22/22 PASS；覆盖 train-only 监督、variable-length verifier/prefix redraft/budget、全 cold position-wise request、train-only probe、deltaW shape 与 One-One trigger；真实产物命令已冻结但未执行，等待用户确认 |
| 2026-08-21 | Toys B2/B3 contract-input CPU Gate PASS | 冻结命令 exit 0，运行 80.84 s；5,963/5,963 cold catalog coverage、59,630 train-only pseudo-context、302,400 position-wise requests；原完整序列、SASRec 相似项和 test 产物均未打开；未训练、未使用 GPU |
| 2026-08-21 | GRAM token-likelihood hook 与 train-only layer probe 实现完成 | Stage15 合约测试累计 25/25 PASS；11,924 个 catalog path 的 lexical segment/token 长度完全一致；首条合并 GPU smoke 固定为 16 users × 32 candidates、8 个 direct parity pair、64 train-only transitions × 6 positions × 6 decoder layers，GRAM 参数前后 hash 必须一致 |
| 2026-08-21 | 首条 B2 GPU smoke 资源合约冻结，等待确认 | 当前 GPU5 空闲 12,030 MiB；命令只使用剩余显存，不触碰约 20 GiB holder；worker admission=8,192 MiB、预计增量上界=7,168 MiB、hard timeout=3,600 s、automatic retry=false |
| 2026-08-21 | B2 GPU smoke attempt-1 在模型加载前失败 | tmux 沙箱权限只在沙箱外补齐同一 worker；25 项 preflight 与 GPU5 admission PASS，但 probe lookup 使用截断后的 history 长度，长历史产生 `KeyError`；workload rc=1、GPU forward=0、未训练、未读 test、未自动 retry，attempt-1 artifact 保留 |
| 2026-08-21 | 长历史 probe lookup 修复并冻结 attempt-2 | 改用每用户 chronological sample ordinal，对应 `len(TrainTransition.history)`，不再依赖被 `max_his=20` 截断的数组长度；新增回归后 26/26 tests PASS，真实 attempt-1 view 上 64/64 selected transitions 匹配，attempt-2 使用独立目录且等待用户确认 |
| 2026-08-21 | B2 GPU smoke attempt-2 完成但 numerical parity Gate FAIL | rc=0，16/16 verifier users、512 candidates、64 train-only transitions、6×6 probe 均完成；score finite、参数 hash 不变、无泄漏，selected layers=`[5,5,5,5,5,4]`；但 hook/direct 最大绝对差 `8.63075e-4 > 2e-5`，不能记 PASS；运行 23.69 s，peak CUDA allocated 1,061.59 MiB |
| 2026-08-21 | 冻结 deterministic attempt-3，等待确认 | PyTorch 1.11 环境 `cuda.matmul.allow_tf32=true`、`cudnn.allow_tf32=true` 且 deterministic algorithms 关闭；attempt-3 不放宽原 `2e-5` Gate，只关闭 TF32、启用 deterministic algorithms，并保存全部 direct pair 分数/差值/acceptance equality；先前 artifact 均不覆盖 |
| 2026-08-21 | deterministic attempt-3 在 CuBLAS 前置条件失败 | 26 项 preflight 与 GPU5 admission PASS，模型已加载，但首个 encoder linear 在完整 user forward 前由 PyTorch deterministic guard 拒绝；缺少进程启动前的 `CUBLAS_WORKSPACE_CONFIG`，rc=1、未训练、未读 test、未自动 retry |
| 2026-08-21 | 补齐 CuBLAS deterministic 环境并冻结 attempt-4 | worker 显式注入 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，运行时代码 hard-fail 校验该值；新增回归后 Stage15 27/27 tests PASS；数据、样本、threshold、`2e-5` tolerance 与资源 admission 均不改变，等待用户确认 |
| 2026-08-22 | deterministic attempt-4 完成，GPU hook/probe Gate PASS | rc=0，16/16 verifier users、512 candidates、64 train-only transitions 和 6×6 probe 完整；8 个 direct pair 最大绝对差 `4.29153e-6 < 2e-5`且 acceptance 全一致；参数 hash 不变、无泄漏、无自动 retry；转入 drafter/edit 状态构建 |
| 2026-08-22 | B2 auxiliary content drafter 与首条 state smoke 合约冻结 | clean-room 实现冻结 BGE content projection + causal Transformer history encoder + full-catalog normalized retrieval；30/30 tests PASS；smoke 只使用固定 train-only 监督，GRAM 不加载/不进 optimizer，GPU2 只用剩余显存，exact command 待用户确认 |
| 2026-08-22 | B2 drafter state smoke GPU2 attempt-1 admission blocked | 用户确认后执行 frozen command；preflight 30/30 PASS，但 GPU2 仅余 3,889 MiB，低于 8,192 MiB admission，worker rc=9；workload 未启动、未改进程、未读 test、未自动 retry，artifact 保留 |
| 2026-08-22 | B2 drafter state smoke GPU1 attempt-2 PASS | 用户指定 GPU1；独立目录执行 rc=0，2 epoch finite、drafter state 改变、16×top-50 合法唯一；GRAM 未加载/未进 optimizer 且 SHA 不变，未用 validation target、未读 test；运行 11.88 s，peak CUDA allocated 229.47 MiB，转入 B3 状态构建 |
| 2026-08-22 | B3 edit state smoke 实现与资源合约冻结 | clean-room 实现全量 302,400 requests、train-only position covariance、z-residual、full-shape deltaW 与 One-One trigger；35/35 tests 及真实 dry preflight PASS。GPU1 当前 free 18,948 MiB，worker admission 8,192 MiB、预计增量上界 6,144 MiB、hard timeout 3,600 s；exact command 等待确认 |
| 2026-08-22 | B3 edit state smoke GPU1 attempt-1 admission blocked | 用户确认后执行 frozen command；35/35 preflight PASS，但启动等待期间 GPU1 新增约 16 GiB workload，worker admission 时 free 约 7,820 MiB，低于 8,192 MiB，rc=9；科学 workload 未启动、无 summary/telemetry、未改进程、未读 test、未自动 retry，artifact 保留 |
| 2026-08-22 | B3 edit state smoke GPU7 attempt-2 冻结 | 用户指定 GPU7；冻结时 free 32,859 MiB，admission 8,192 MiB，预计增量上界 6,144 MiB，hard timeout 3,600 s；保留 GPU7 既有 3 个 PID，使用独立 wrapper/session/artifact，automatic retry=false |
| 2026-08-22 | B3 edit state smoke GPU7 attempt-2 PASS，S15-2 contract Gate 关闭 | attempt-2 于 15:09:13 启动并于 15:10:14 完成，rc=0；35/35 tests PASS，delta finite/nonzero、One-One trigger 6/6 positions 生效、EOS/padding inactive、未编辑 prompt exact parity，GRAM model/checkpoint hash 前后不变；未读 test 或 validation supervision。S15-2 证据链完整，转入 S15-3A 512-event admission 准备 |
| 2026-08-22 | S15-3A attempt-1 在 B3 position 4 state admission 失败 | GPU7 worker preflight 38/38 PASS；B2 state 与 B3 context/covariance 正常，但复用 v0 layer map 后 position 4 的固定 4 requests 无一通过 z-success contract，rc=1。held events 未打开、beam evaluation 未启动、test 未读、无自动 retry，artifact 原样保留 |
| 2026-08-22 | S15-3A attempt-2 修复冻结，等待确认 | 失败源是 layer selection 与 admission base 参数状态不匹配；修复为在 Stage14 clean base 上先用 64 条 train-only transitions 重做 6×6 probe，再按既有规则选择 layer。数据、B2/B3 预算、512 held events、threshold、GPU admission 与 Gate 不变；独立 wrapper/output/session 已冻结，不自动启动 |
| 2026-08-22 | S15-3A attempt-2 仍在 B3 position 4 admission 失败 | clean-base 6×6 probe 完成，selected layers=`[5,5,3,5,0,0]`；positions 4/5 全层 accuracy=0，position 4 固定 requests 再次 0/4。held events 未打开、test 未读；证明失败不是单纯 v0 layer-map 迁移错误 |
| 2026-08-22 | B3 admission 冻结 FAIL，拆分 B2-only admission | 不通过放宽 0.3 threshold、增加 requests、改 tie-break 或换 seed 救援 B3；B3 不进入 full validation。B2 两次 state build 均正常，方法级 Gate 应独立裁决，故冻结 B0+B2 512-event admission 独立入口，等待用户确认 |
| 2026-08-22 | B0+B2 独立 S15-3A admission 已确认并启动 | GPU7 启动时 free=21,441 MiB，满足 16,384 MiB admission；exact command=`bash experiment/phase15/run_stage15_s3a_toys_b2_only_admission.sh start 7`。38/38 tests PASS，B2 train-only loss=`[9.36309439,9.24616081]`，已进入 512 held-event evaluation；独立 artifact/session，不含 B3、不读 test、不自动 retry |
| 2026-08-22 | B2 S15-3A 512-event admission 完成并重裁决 PASS | workload rc=0，512/512 events、25,600 verifier candidates、所有 unique known top-50、base hash 不变、test 未读；runtime 3,154.16 s，peak CUDA allocated 6,935.35 MiB。原 reducer 将两个正确的负向安全事实误判为失败；改为正向断言并用完整 artifact 确定性重算 `PASS_S15_3A_B2_ITEM_DISJOINT_ADMISSION`，原始 verdict 留档。S15-3B 冻结为 B0/B1/B2，等待资源与 exact command 授权 |
| 2026-08-22 | B3 branching recovery 合约冻结 | 两次失败均在 held 打开前停于 position 4；真实 request preflight 证明旧 sampler 的 position-4 branching factors=`[1,1,1,1]`，修复后=`[22,2,3,2]`，positions 0–5 均保留 4 个 distinct cold requests。未改 seed、layer、threshold、step 或 request budget；恢复结果只认新的独立 512-event admission。 |
